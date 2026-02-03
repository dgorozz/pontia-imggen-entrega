from diffusers import DDPMScheduler
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datasets import load_dataset
from torchvision import transforms
import os
from torch import nn
import pickle
import diffusers
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F


diffusers.logging.set_verbosity_error()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Se usará '{device}' para entrenar")


DATASET_ID = "gigant/oldbookillustrations"
MODEL_ID = "CompVis/stable-diffusion-v1-4"
NUM_EPOCHS = 2

FORCE_COMPUTE_LATENTS = False


class CustomSquaredCenterCropSmallestSize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
        else:
            w, h = img.size
        smallest = min(h, w)
        return transforms.functional.center_crop(img, (smallest, smallest))
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class Text2ImageDataset(Dataset):
    def __init__(self, dataset, transforms, tokenizer):
        self.dataset = dataset
        self.transforms = transforms
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = self.transforms(example["1600px"].convert("RGB"))
        token = self.tokenizer(example["info_alt"], padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt")
        return {
            "pixel_values": image,
            "input_ids": token.input_ids.squeeze(0),
            "attention_mask": token.attention_mask.squeeze(0),
        }


class LatentsDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]


def precompute_latents(vae: nn.Module, text_encoder: nn.Module, dataloader: DataLoader):

    precomputed = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating latents"):

            latents = vae.encode(batch["pixel_values"].to("cuda")).latent_dist.sample()
            latents = latents * 0.18215
            
            encoder_hidden_states = text_encoder(batch["input_ids"].to("cuda"))[0]
            
            # .cpu() to save memory and .clone() to avoid passing a view
            for i in range(latents.shape[0]):
                precomputed.append({
                    "latents": latents.cpu()[i].clone(),
                    "encoder_hidden_states": encoder_hidden_states.cpu()[i].clone()
                })
        
    return precomputed


def train(noise_scheduler, unet, dataloader: DataLoader):

    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)
    accelerator = Accelerator(mixed_precision="fp16")
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    print("Entrenando ...")
    for _ in range(NUM_EPOCHS):
        unet.train()
        for batch in tqdm(dataloader):
            latents = batch["latents"].to(device, dtype=torch.float16)
            encoder_hidden_states = batch["encoder_hidden_states"].to(device, dtype=torch.float16).squeeze(1)
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

    return unet


def main():

    if not os.path.exists("latents.pkl") or FORCE_COMPUTE_LATENTS: # only execute if not precomputed or if it is forced to compute again
        
        print("Latents not found. Latents will be computed and saved as latents.pkl")

        # load dataset
        dataset = load_dataset(DATASET_ID, split="train")
        dataset = dataset.select(range(200)) # select only the first 200 rows
        print("✅ Dataset loaded! Selected only the first 200 rows")

        # create transforms
        tr = transforms.Compose([
            transforms.Grayscale(num_output_channels=3), # model expects 3 channels
            CustomSquaredCenterCropSmallestSize(), # custom transform defined above
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        # precompute latents with VAE
        # this approach will avoi loading in memory VAE, Optimizer and Unet, which may raise CUDAOutOfMemory error
        # (in my case it happens, even with batch_size=1)
        tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder='tokenizer')
        text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to(device)
        vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(device)

        # fix VAE and TE: it is not required to train
        vae.eval(); vae.requires_grad_(False)
        text_encoder.eval(); text_encoder.requires_grad_(False)

        # create dataset and dataloader for training
        train_dataset = Text2ImageDataset(dataset=dataset, transforms=tr, tokenizer=tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True)

        # generate latents and save them as pickle
        precomputed_data = precompute_latents(vae=vae, text_encoder=text_encoder, dataloader=train_dataloader)
        with open("latents.pkl", "wb") as f:
            pickle.dump(precomputed_data, f)

        # delete unused content and free memory in VRAM
        del vae, tokenizer, text_encoder
        torch.cuda.empty_cache()


    # load precomputed latents
    with open("latents.pkl", "rb") as f:
        precomputed_data = pickle.load(f)

    # instance unet and noise_scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")

    # create dataset for latents
    train_dataset = LatentsDataset(precomputed_data)
    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True)

    # train
    unet = train(noise_scheduler=noise_scheduler, unet=unet, dataloader=train_dataloader)
    unet.save_pretrained("./normal-trained")
    

if __name__ == "__main__":
    main()
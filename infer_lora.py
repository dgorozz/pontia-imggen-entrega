from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import PeftModel
import torch

MODEL_ID = "CompVis/stable-diffusion-v1-4"
LORA_FINETUNED = "danigr7/unet-lora-finetuned"
PROMPT = "an astronaut riding a horse"
NUM_STEPS = 10

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # load unet (separated)
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").to(device)

    # load unet with LoRa
    unet = PeftModel.from_pretrained(unet, LORA_FINETUNED)
    # unet = PeftModel.from_pretrained(unet, LORA_FINETUNED)

    # load pipeline with predefined unet
    # with few inference steps I get a NFSW advice
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, 
        unet=unet,
        safety_checker = None, # because of the NSFW advice
        requires_safety_checker = False # because of the NSFW advice
    ).to(device)

    # make inference
    image = pipe(PROMPT, num_inference_steps=NUM_STEPS).images[0]

    # save output
    image.save(f"outputs/lora_output_{NUM_STEPS}_steps.png")


if __name__ == "__main__":
    main()
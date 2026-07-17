from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
)

pipe = pipe.to(device)

image = Image.open("input_images/input.jpg").convert("RGB")
image = image.resize((512,512))

prompt = input("Enter transformation prompt: ")

print("Generating...")

result = pipe(
    prompt=prompt,
    image=image,
    strength=0.7,
    guidance_scale=7.5
).images[0]

os.makedirs("output_images",exist_ok=True)

output_path="output_images/output.png"

result.save(output_path)

print("Done!")
print("Saved:",output_path)
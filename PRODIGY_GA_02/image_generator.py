from diffusers import StableDiffusionPipeline
import torch
import os

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)

pipe = pipe.to("cpu")

# Create output folder
os.makedirs("generated_images", exist_ok=True)

print("=" * 50)
print(" AI Image Generator")
print("=" * 50)

while True:
    prompt = input("\nEnter your prompt (or type 'exit'): ")

    if prompt.lower() == "exit":
        break

    print("\nGenerating image... Please wait.\n")

    image = pipe(prompt).images[0]

    filename = prompt.replace(" ", "_")[:40] + ".png"
    path = os.path.join("generated_images", filename)

    image.save(path)

    print(f"Image saved as: {path}")
import os
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*60)
print(" Neural Style Transfer ")
print("="*60)

print("\nLoading Stable Diffusion Model...\n")

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipe = pipe.to(device)

image_path = "input_images/input.jpg"

if not os.path.exists(image_path):
    print("Input image not found!")
    exit()

image = Image.open(image_path).convert("RGB")
image = image.resize((512,512))

print("\nChoose a Style\n")

print("1. Van Gogh")
print("2. Picasso")
print("3. Watercolor")
print("4. Oil Painting")
print("5. Anime")
print("6. Cyberpunk")
print("7. Pencil Sketch")
print("8. Fantasy Art")
print("9. Custom Prompt")

choice = input("\nEnter choice : ")

styles = {
    "1":"A painting in the style of Vincent Van Gogh",
    "2":"A cubist painting in the style of Pablo Picasso",
    "3":"Watercolor painting",
    "4":"Oil painting masterpiece",
    "5":"Anime artwork",
    "6":"Cyberpunk art",
    "7":"Detailed pencil sketch",
    "8":"Fantasy digital artwork"
}

if choice=="9":
    prompt=input("Enter custom style prompt : ")
else:
    prompt=styles.get(choice,"Oil painting masterpiece")

print("\nGenerating Stylized Image...\n")

result = pipe(
    prompt=prompt,
    image=image,
    strength=0.65,
    guidance_scale=8.0
).images[0]

os.makedirs("output_images",exist_ok=True)

output_path="output_images/stylized_output.png"

result.save(output_path)

print("="*60)
print("Style Transfer Completed Successfully!")
print("Saved to :",output_path)
print("="*60)
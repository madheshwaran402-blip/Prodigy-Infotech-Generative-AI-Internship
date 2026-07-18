# PRODIGY_GA_05
# Neural Style Transfer using Stable Diffusion

## 📌 Overview

This project was developed as part of the **Prodigy InfoTech Generative AI Internship**.

The objective of this task is to perform **Neural Style Transfer**, where the artistic style of an image is applied to another image using a pre-trained generative AI model.

The application takes an input image and transforms it into different artistic styles based on user-selected or custom text prompts.

---

## 🚀 Features

- Load an input image
- Apply different artistic styles
- Support predefined style options
- Accept custom style prompts
- Generate high-quality stylized images
- Automatically save output images

---

## 🛠 Technologies Used

- Python
- PyTorch
- Hugging Face Diffusers
- Transformers
- Stable Diffusion Image-to-Image
- Pillow

---

## 📂 Project Structure

```
PRODIGY_GA_05/
│
├── input_images/
│   └── input.jpg
│
├── output_images/
│   └── stylized_output.png
│
├── neural_style_transfer.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/PRODIGY_GA_05.git
```

Navigate to the project

```bash
cd PRODIGY_GA_05
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python neural_style_transfer.py
```

Choose one of the available styles:

```
1. Van Gogh
2. Picasso
3. Watercolor
4. Oil Painting
5. Anime
6. Cyberpunk
7. Pencil Sketch
8. Fantasy Art
9. Custom Prompt
```

If you choose **Custom Prompt**, enter your own artistic style description.

The generated image will be saved inside:

```
output_images/
```

---

## Example Prompts

- A painting in the style of Vincent Van Gogh
- Cubist painting by Pablo Picasso
- Watercolor painting
- Oil painting masterpiece
- Anime artwork
- Cyberpunk art
- Pencil sketch
- Fantasy digital artwork
- Disney cartoon style
- Ancient Indian painting style

---

## Learning Outcomes

- Understanding Neural Style Transfer
- Image-to-Image Translation
- Working with Stable Diffusion
- Prompt Engineering
- Image Processing using Python
- Building AI-powered Image Applications

---

## Future Improvements

- Desktop GUI using Tkinter
- Drag-and-drop image upload
- Multiple style blending
- Batch image processing
- Support for additional diffusion models
- Real-time style preview

---

## Author

**Madheshwaran M.**

Generative AI Internship – Prodigy InfoTech

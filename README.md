# Product Placement Tool

## Overview
This tool automates the process of placing e-commerce product images into realistic lifestyle backgrounds using Generative AI. It leverages open-source models and libraries to ensure seamless integration of products into diverse scenes while preserving product details and maintaining natural lighting, perspective, and scale.

### Key Features
- **Batch Processing**: Handles multiple product images simultaneously.
- **Realistic Placement**: Ensures natural integration of products into lifestyle backgrounds.
- **Customization**: Allows adjustment of product placement, size, and rotation.
- **Open-Source**: Uses freely available, unlimited-usage tools and models.

---

## Libraries and Tools Used
The tool relies on the following open-source libraries:
- **PyTorch**: For deep learning model inference.
- **Diffusers**: For Stable Diffusion and ControlNet pipelines.
- **Transformers**: For semantic segmentation and scene understanding.
- **OpenCV**: For image processing and edge detection.
- **Pillow (PIL)**: For image manipulation.
- **Rembg**: For background removal from product images.
- **NumPy**: For numerical operations.
- **Matplotlib**: For visualizing results.

---

## Hardware and Software Requirements
### Hardware
- **Minimum**: CPU with 8 GB RAM.
- **Recommended**: GPU with at least 4 GB VRAM (e.g., NVIDIA GTX 1050 or higher).
- **Storage**: Sufficient space for input/output images and model weights.

### Software
- **Python**: 3.8 or higher.
- **Dependencies**: Listed in `requirements.txt`.

---

## How to Run the Code

### Step 1: Clone the Repository
```bash
git clone https://github.com/riffhi/Product-Placement-Tool-using-Gen-AI.git
cd pp

### Step 2: Install Dependencies
pip install -r requirements.txt


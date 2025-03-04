# Product Placement Tool using Generative AI

## Overview
This tool automates the process of placing e-commerce product images into realistic lifestyle backgrounds using Generative AI. It leverages open-source models and libraries to ensure seamless integration of products into diverse scenes while preserving product details and maintaining natural lighting, perspective, and scale.

## Features
- **Batch Processing**: Handles multiple product images simultaneously.
- **Realistic Placement**: Ensures natural integration of products into lifestyle backgrounds.
- **Customization**: Allows adjustment of product placement, size, and rotation.
- **Open-Source**: Uses freely available, unlimited-usage tools and models.

## Libraries and Tools Used
This tool relies on the following open-source libraries:
- **PyTorch**: For deep learning model inference.
- **Diffusers**: For Stable Diffusion and ControlNet pipelines.
- **Transformers**: For semantic segmentation and scene understanding.
- **OpenCV**: For image processing and edge detection.
- **Pillow (PIL)**: For image manipulation.
- **Rembg**: For background removal from product images.
- **NumPy**: For numerical operations.
- **Matplotlib**: For visualizing results.

## Hardware and Software Requirements
### Hardware
- **Minimum**: CPU with 8 GB RAM.
- **Recommended**: GPU with at least 4 GB VRAM (e.g., NVIDIA GTX 1050 or higher).
- **Storage**: Sufficient space for input/output images and model weights.

### Software
- **Python**: 3.8 or higher.
- **Dependencies**: Listed in `requirements.txt`.

## Installation and Usage
### Step 1: Clone the Repository
```bash
git clone https://github.com/riffhi/Product-Placement-Tool-using-Gen-AI.git
cd Product-Placement-Tool-using-Gen-AI
```

### Step 2: Install Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Input Data
- Place product images in the `products/` folder.
- Place background images in the `backgrounds/` folder.

### Step 4: Run the Tool
Run the tool with the following command:
```bash
python product_placement.py --products products/ --backgrounds backgrounds/ --output results/
```

### Command-Line Arguments
| Argument         | Description                                     | Default |
|-----------------|-------------------------------------------------|---------|
| `--products`    | Path to the directory containing product images | Required |
| `--backgrounds` | Path to the directory containing background images | Required |
| `--output`      | Path to the directory where results will be saved | `results/` |
| `--device`      | Device to run on (`cuda` for GPU or `cpu` for CPU) | `cuda` if available |
| `--batch_size`  | Number of products to process                    | All |
| `--steps`       | Number of inference steps                        | 30 |
| `--guidance_scale` | Guidance scale for Stable Diffusion         | 9.0 |

### Step 5: View Results
- Generated images will be saved in the `results/` folder.
- A summary of results will be saved as `results_summary.json` in the output directory.

## License
This project is open-source and available under the MIT License.

## Contributing
Feel free to fork this repository, submit issues, or create pull requests to improve the tool.


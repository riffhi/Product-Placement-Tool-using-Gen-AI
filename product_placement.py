# import os
# import argparse
# import torch
# import numpy as np
# from PIL import Image
# import cv2
# from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
# from rembg import remove
# from tqdm import tqdm
# import random
# import json
# import logging
# from typing import List, Dict, Tuple, Any, Optional
# import matplotlib.pyplot as plt

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class ProductPlacementTool:
#     def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
#         """
#         Initialize the Product Placement Tool with the necessary models.
        
#         Args:
#             device (str): The device to run the models on (cuda or cpu)
#         """
#         self.device = device
#         logger.info(f"Using device: {device}")
        
#         # Import diffusers modules here to handle imports properly
#         try:
#             # Try to import from diffusers with specific version compatibility
#             from diffusers import (
#                 StableDiffusionControlNetPipeline, 
#                 ControlNetModel, 
#                 UniPCMultistepScheduler,
#                 DPMSolverMultistepScheduler
#             )
            
#             # Load models
#             logger.info("Loading ControlNet model...")
#             self.controlnet = ControlNetModel.from_pretrained(
#                 "lllyasviel/sd-controlnet-canny", 
#                 torch_dtype=torch.float16 if device == "cuda" else torch.float32
#             ).to(device)
            
#             logger.info("Loading Stable Diffusion pipeline...")
#             self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
#                 "runwayml/stable-diffusion-v1-5",
#                 controlnet=self.controlnet,
#                 torch_dtype=torch.float16 if device == "cuda" else torch.float32,
#                 safety_checker=None  # Disable safety checker to avoid additional dependencies
#             ).to(device)
            
#             # Explicitly disable xformers
#             try:
#                 self.pipe.disable_xformers_memory_efficient_attention()
#             except:
#                 pass  # If not available, ignore
            
#             # Make sure xformers is not enabled
#             self.pipe.enable_xformers_memory_efficient_attention = False
            
#             # Try UniPC scheduler first, fall back to DPMSolver if not available
#             try:
#                 self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
#             except:
#                 logger.info("UniPCMultistepScheduler not available, using DPMSolverMultistepScheduler instead")
#                 self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            
#             # Enable attention slicing to save memory
#             self.pipe.enable_attention_slicing()
            
#         except Exception as e:
#             # If importing fails with the newer diffusers, try the older version without Triton dependency
#             logger.error(f"Error initializing diffusers pipeline: {str(e)}")
#             logger.info("Trying alternative implementation...")
            
#             # This is more likely to work without triton
#             import subprocess
#             import sys
            
#             # We should recommend installing an older version of diffusers
#             logger.info("You may need to run: pip install diffusers==0.14.0 transformers accelerate")
#             raise RuntimeError("Could not initialize diffusers models. Please install diffusers==0.14.0 which doesn't require triton.")
        
#         logger.info("Loading segmentation model for scene understanding...")
#         self.image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
#         self.segmentation_model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small").to(device)
        
#         # Set default parameters
#         self.default_params = {
#             "num_inference_steps": 30,
#             "guidance_scale": 9.0,
#             "controlnet_conditioning_scale": 0.8
#         }
        
#         logger.info("Model initialization complete")
    
#     def extract_product_mask(self, product_image_path: str) -> Tuple[Image.Image, Image.Image]:
#         """
#         Extract the product from its background using rembg.
        
#         Args:
#             product_image_path (str): Path to the product image
            
#         Returns:
#             Tuple of (masked product image, alpha mask)
#         """
#         logger.info(f"Extracting product mask for {product_image_path}")
#         # Load the product image
#         product_img = Image.open(product_image_path).convert("RGBA")
        
#         # Remove background
#         output = remove(product_img)
        
#         # Create alpha mask
#         alpha_mask = output.split()[-1]
        
#         return output, alpha_mask
    
#     def analyze_scene(self, background_image_path: str) -> Dict[str, Any]:
#         """
#         Analyze the background scene to identify suitable placement areas.
        
#         Args:
#             background_image_path (str): Path to the background image
            
#         Returns:
#             Dictionary with scene analysis results
#         """
#         logger.info(f"Analyzing scene in {background_image_path}")
#         # Load the background image
#         background_img = Image.open(background_image_path).convert("RGB")
        
#         # Perform semantic segmentation to understand the scene
#         inputs = self.image_processor(images=background_img, return_tensors="pt").to(self.device)
#         outputs = self.segmentation_model(**inputs)
        
#         # Get segmentation map
#         seg_map = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
        
#         # Identify potential placement surfaces (tables, shelves, etc.)
#         # This is a simplified approach; more sophisticated detection would be implemented in a full solution
#         potential_surfaces = []
        
#         # Convert to dimensions suitable for CV2 processing
#         bg_np = np.array(background_img)
#         bg_np = cv2.cvtColor(bg_np, cv2.COLOR_RGB2BGR)
        
#         # Detect edges in the image
#         gray = cv2.cvtColor(bg_np, cv2.COLOR_BGR2GRAY)
#         edges = cv2.Canny(gray, 50, 150)
        
#         # Use line detection to find potential horizontal surfaces
#         lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
#         if lines is not None:
#             for line in lines:
#                 x1, y1, x2, y2 = line[0]
#                 # Horizontal lines (potential surfaces)
#                 if abs(y2 - y1) < 20:  # Almost horizontal
#                     # Only consider lines in the middle to lower part of the image (more likely to be surfaces)
#                     if y1 > background_img.height * 0.4:
#                         potential_surfaces.append({
#                             'y': min(y1, y2),
#                             'x_start': min(x1, x2),
#                             'x_end': max(x1, x2),
#                             'width': abs(x2 - x1)
#                         })
        
#         # Sort surfaces by width (prefer wider surfaces)
#         potential_surfaces.sort(key=lambda x: x['width'], reverse=True)
        
#         # Get dominant colors for matching
#         # Simplified color analysis
#         img_np = np.array(background_img.resize((100, 100)))
#         pixels = img_np.reshape(-1, 3)
#         from sklearn.cluster import KMeans
#         kmeans = KMeans(n_clusters=5)
#         kmeans.fit(pixels)
#         dominant_colors = kmeans.cluster_centers_.astype(int)
        
#         return {
#             'dimensions': (background_img.width, background_img.height),
#             'potential_surfaces': potential_surfaces[:5],  # Top 5 potential surfaces
#             'dominant_colors': dominant_colors.tolist(),
#             'seg_map': seg_map
#         }
    
#     def determine_placement_location(self, scene_analysis: Dict[str, Any], product_size: Tuple[int, int]) -> Dict[str, Any]:
#         """
#         Determine the best location to place the product based on scene analysis.
        
#         Args:
#             scene_analysis (dict): Scene analysis results
#             product_size (tuple): Size of the product (width, height)
            
#         Returns:
#             Dictionary with placement information
#         """
#         logger.info("Determining optimal placement location")
#         bg_width, bg_height = scene_analysis['dimensions']
#         product_width, product_height = product_size
        
#         # Scale product to reasonable size relative to background
#         scale_factor = min(bg_width * 0.4 / product_width, bg_height * 0.4 / product_height)
#         scaled_width = int(product_width * scale_factor)
#         scaled_height = int(product_height * scale_factor)
        
#         # If we have detected surfaces, use them
#         placement = {}
#         if scene_analysis['potential_surfaces']:
#             surface = random.choice(scene_analysis['potential_surfaces'][:3])  # Choose from top 3 surfaces
            
#             # Position on the surface - ensure y_pos is a regular int, not np.int32
#             x_pos = random.randint(surface['x_start'], max(surface['x_start'], surface['x_end'] - scaled_width))
#             y_pos = int(surface['y'] - scaled_height)  # Convert to regular int to avoid numpy type issues
            
#             placement = {
#                 'x': x_pos,
#                 'y': y_pos,
#                 'scale': scale_factor,
#                 'rotation': random.randint(-5, 5)  # Slight random rotation for realism
#             }
#         else:
#             # Fallback: place in a reasonable location in the image
#             x_pos = random.randint(int(bg_width * 0.1), int(bg_width * 0.7))
#             y_pos = random.randint(int(bg_height * 0.5), int(bg_height * 0.8))
            
#             placement = {
#                 'x': x_pos,
#                 'y': y_pos,
#                 'scale': scale_factor,
#                 'rotation': random.randint(-5, 5)
#             }
        
#         # Ensure all placement values are Python ints, not numpy types
#         placement = {k: int(v) if isinstance(v, (np.integer, np.int32, np.int64)) else v 
#                     for k, v in placement.items()}
        
#         logger.info(f"Placement determined: {placement}")
#         return placement
    
#     def create_control_image(self, background_path: str, product_image: Image.Image, 
#                         placement: Dict[str, Any]) -> np.ndarray:
#         """
#         Create a control image for ControlNet guidance.
        
#         Args:
#             background_path (str): Path to background image
#             product_image (Image.Image): Product image with transparent background
#             placement (dict): Placement information
            
#         Returns:
#             Control image as numpy array
#         """
#         try:
#             logger.info("Creating control image for ControlNet")
            
#             # Load background
#             background = Image.open(background_path).convert("RGB")
#             bg_width, bg_height = background.size
            
#             # Ensure product_image is in RGBA mode
#             if product_image.mode != 'RGBA':
#                 product_image = product_image.convert('RGBA')
            
#             # Resize and rotate product according to placement
#             product_width, product_height = product_image.size
#             new_width = max(1, int(product_width * placement.get('scale', 1)))
#             new_height = max(1, int(product_height * placement.get('scale', 1)))
            
#             # Resize product image
#             try:
#                 resized_product = product_image.resize((new_width, new_height), Image.LANCZOS)
                
#                 # Rotate if rotation is specified
#                 if placement.get('rotation', 0) != 0:
#                     resized_product = resized_product.rotate(
#                         placement['rotation'], 
#                         expand=True, 
#                         resample=Image.BICUBIC
#                     )
#             except Exception as resize_error:
#                 logger.error(f"Error resizing product image: {resize_error}")
#                 # Fallback to original product image if resizing fails
#                 resized_product = product_image
            
#             # Convert to numpy array
#             product_np = np.array(resized_product)
            
#             # Ensure we have an alpha channel
#             if product_np.shape[2] == 4:
#                 alpha_channel = product_np[:, :, 3]
#             else:
#                 # If no alpha channel, create a full opacity mask
#                 alpha_channel = np.ones(product_np.shape[:2], dtype=np.uint8) * 255
            
#             # Create control image using Canny edge detection
#             # Convert background to numpy array
#             bg_np = np.array(background)
            
#             # Perform Canny edge detection on the background
#             control_image = cv2.Canny(bg_np, 50, 150)
            
#             # Ensure control image is always at least the background size
#             if control_image.shape[:2] != bg_np.shape[:2]:
#                 logger.warning("Resizing control image to match background")
#                 control_image = cv2.resize(control_image, (bg_width, bg_height))
            
#             # Overlay product edges if possible
#             try:
#                 # Compute edges from the alpha channel
#                 product_edges = cv2.Canny(alpha_channel, 100, 200)
                
#                 # Placement coordinates
#                 x = placement.get('x', 0)
#                 y = placement.get('y', 0)
                
#                 # Ensure coordinates are within image bounds
#                 x = max(0, min(x, bg_width - product_edges.shape[1]))
#                 y = max(0, min(y, bg_height - product_edges.shape[0]))
                
#                 # Overlay product edges onto control image
#                 for i in range(product_edges.shape[0]):
#                     for j in range(product_edges.shape[1]):
#                         if product_edges[i, j] > 0:
#                             yi, xi = y + i, x + j
#                             if 0 <= yi < control_image.shape[0] and 0 <= xi < control_image.shape[1]:
#                                 control_image[yi, xi] = 255
#             except Exception as overlay_error:
#                 logger.warning(f"Could not overlay product edges: {overlay_error}")
            
#             logger.info(f"Control image created with shape {control_image.shape}")
#             return control_image
        
#         except Exception as e:
#             logger.error(f"Catastrophic error in create_control_image: {e}")
#             # Fallback: create a simple edge map of the background
#             bg_np = np.array(Image.open(background_path).convert("RGB"))
#             return cv2.Canny(bg_np, 50, 150)
        
#     def create_prompt(self, product_path: str, background_path: str) -> str:
#         """
#         Create an appropriate prompt for stable diffusion based on the product and background.
        
#         Args:
#             product_path (str): Path to the product image
#             background_path (str): Path to the background image
            
#         Returns:
#             Prompt string for Stable Diffusion
#         """
#         # Extract product type from filename
#         product_name = os.path.basename(product_path).split('.')[0]
#         product_words = product_name.replace('_', ' ').replace('-', ' ')
        
#         # Extract background type from filename
#         background_name = os.path.basename(background_path).split('.')[0]
#         background_words = background_name.replace('_', ' ').replace('-', ' ')
        
#         # Create a descriptive prompt
#         prompt = f"A realistic photograph of a {product_words} placed in a {background_words}, professional product photography, perfect lighting, high quality, photorealistic"
        
#         logger.info(f"Generated prompt: {prompt}")
#         return prompt
    
#     def place_product(self, product_path: str, background_path: str, output_path: str, 
#                  custom_params: Dict[str, Any] = None) -> Optional[str]:
#         """
#         Place a product into a background image with extensive error handling and diagnostics.
#         """
#         try:
#             logger.info(f"Starting place_product for product: {product_path}")
#             logger.info(f"Background: {background_path}")
#             logger.info(f"Output path: {output_path}")
            
#             # Validate input files
#             if not os.path.exists(product_path):
#                 logger.error(f"Product image not found: {product_path}")
#                 return None
            
#             if not os.path.exists(background_path):
#                 logger.error(f"Background image not found: {background_path}")
#                 return None
            
#             # Extract product mask with detailed logging
#             try:
#                 product_img, alpha_mask = self.extract_product_mask(product_path)
#                 logger.info(f"Product image extracted. Size: {product_img.size}")
#             except Exception as mask_error:
#                 logger.error(f"Failed to extract product mask: {mask_error}")
#                 logger.error(f"Product path: {product_path}")
#                 import traceback
#                 logger.error(traceback.format_exc())
#                 return None
            
#             # Validate product image
#             if product_img is None:
#                 logger.error("Product image is None after mask extraction")
#                 return None
            
#             # Analyze scene with detailed logging
#             try:
#                 scene_analysis = self.analyze_scene(background_path)
#                 logger.info(f"Scene analysis complete. Potential surfaces: {len(scene_analysis.get('potential_surfaces', []))}")
#             except Exception as scene_error:
#                 logger.error(f"Scene analysis failed: {scene_error}")
#                 import traceback
#                 logger.error(traceback.format_exc())
#                 return None
            
#             # Determine placement with detailed logging
#             try:
#                 placement = self.determine_placement_location(scene_analysis, product_img.size)
#                 logger.info(f"Placement determined: {placement}")
                
#                 # Add explicit type checking for placement values
#                 for key, value in placement.items():
#                     if value is None:
#                         logger.error(f"Placement value {key} is None")
#                         return None
#             except Exception as placement_error:
#                 logger.error(f"Placement determination failed: {placement_error}")
#                 import traceback
#                 logger.error(traceback.format_exc())
#                 return None
            
#             # Create control image with detailed logging
#             try:
#                 control_image = self.create_control_image(background_path, product_img, placement)
                
#                 # Validate control image
#                 if control_image is None:
#                     logger.error("Control image is None")
#                     return None
                
#                 logger.info(f"Control image shape: {control_image.shape}")
#             except Exception as control_error:
#                 logger.error(f"Control image creation failed: {control_error}")
#                 import traceback
#                 logger.error(traceback.format_exc())
#                 return None
            
#             # Create prompt
#             prompt = self.create_prompt(product_path, background_path)
#             logger.info(f"Generated prompt: {prompt}")
            
#             # Negative prompt
#             negative_prompt = "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"
            
#             # Merge parameters
#             params = self.default_params.copy()
#             if custom_params:
#                 params.update(custom_params)
            
#             # Detailed logging of parameters
#             logger.info(f"Generation parameters: {params}")
            
#             # Ensure control image meets ControlNet requirements
#             try:
#                 h, w = control_image.shape[:2]
#                 h = (h // 8) * 8
#                 w = (w // 8) * 8
#                 control_image = cv2.resize(control_image, (w, h))
#                 logger.info(f"Resized control image to {w}x{h}")
#             except Exception as resize_error:
#                 logger.error(f"Control image resize failed: {resize_error}")
#                 return None

#                 return None
            
#             # Generate image with extensive error handling
#             try:
#                 if self.device == "cuda":
#                     torch.cuda.empty_cache()
                
#                 # Add explicit type checking before generation
#                 assert isinstance(control_image, np.ndarray), f"Control image is {type(control_image)}"
#                 assert control_image is not None, "Control image cannot be None"
#                 assert prompt is not None, "Prompt cannot be None"
                
#                 # Detailed logging of generation inputs
#                 logger.info("Attempting image generation with:")
#                 logger.info(f"Prompt length: {len(prompt)}")
#                 logger.info(f"Control image shape: {control_image.shape}")
#                 logger.info(f"Inference steps: {params['num_inference_steps']}")
                
#                 # Add explicit type conversion if needed
#                 params_for_generation = {
#                     'prompt': str(prompt),
#                     'negative_prompt': str(negative_prompt),
#                     'image': control_image,
#                     'num_inference_steps': int(params['num_inference_steps']),
#                     'guidance_scale': float(params['guidance_scale']),
#                     'controlnet_conditioning_scale': float(params.get('controlnet_conditioning_scale', 0.8))
#                 }
                
#                 image = self.pipe(**params_for_generation).images[0]
                
#                 logger.info("Image generation successful")
#             except Exception as gen_error:
#                 logger.error(f"Image generation failed: {gen_error}")
#                 import traceback
#                 logger.error(traceback.format_exc())
#                 return None
            
#             # Save result
#             try:
#                 os.makedirs(os.path.dirname(output_path), exist_ok=True)
#                 image.save(output_path)
#                 logger.info(f"Image saved to {output_path}")
#                 return output_path
#             except Exception as save_error:
#                 logger.error(f"Failed to save image: {save_error}")
#                 import traceback
#                 logger.error(traceback.format_exc())
#                 return None
        
#         except Exception as e:
#             logger.error(f"Unexpected error in place_product: {e}")
#             import traceback
#             logger.error(traceback.format_exc())
#             return None

#     def batch_process(self, product_paths: List[str], background_paths: List[str], 
#                      output_dir: str, custom_params: Dict[str, Any] = None) -> List[str]:
#         """
#         Process multiple products and backgrounds in batch.
        
#         Args:
#             product_paths (list): List of paths to product images
#             background_paths (list): List of paths to background images
#             output_dir (str): Directory to save results
#             custom_params (dict, optional): Custom parameters for generation
            
#         Returns:
#             List of paths to output images
#         """
#         logger.info(f"Starting batch processing of {len(product_paths)} products")
#         os.makedirs(output_dir, exist_ok=True)
        
#         results = []
        
#         # For each product, choose a random background and place the product
#         for i, product_path in enumerate(tqdm(product_paths, desc="Processing products")):
#             # Choose a random background
#             background_path = random.choice(background_paths)
            
#             # Create output filename
#             product_name = os.path.basename(product_path).split('.')[0]
#             background_name = os.path.basename(background_path).split('.')[0]
#             output_filename = f"{product_name}_in_{background_name}.png"
#             output_path = os.path.join(output_dir, output_filename)
            
#             # Place the product
#             try:
#                 result_path = self.place_product(product_path, background_path, output_path, custom_params)
#                 if result_path:  # Only add to results if successful
#                     results.append(result_path)
                
#                 # Actively manage memory after each image generation
#                 if self.device == "cuda":
#                     torch.cuda.empty_cache()
                    
#             except Exception as e:
#                 logger.error(f"Error processing {product_path}: {str(e)}")
        
#         logger.info(f"Batch processing complete. Generated {len(results)} images.")
#         return results
    
#     def show_results(self, result_paths: List[str], cols: int = 3) -> None:
#         """
#         Display the results in a grid.
        
#         Args:
#             result_paths (list): List of paths to result images
#             cols (int): Number of columns in the grid
#         """
#         n_images = len(result_paths)
#         if n_images == 0:
#             logger.warning("No images to display")
#             return
            
#         rows = (n_images + cols - 1) // cols
        
#         plt.figure(figsize=(15, 5 * rows))
#         for i, img_path in enumerate(result_paths):
#             plt.subplot(rows, cols, i + 1)
#             img = Image.open(img_path)
#             plt.imshow(np.array(img))
#             plt.title(os.path.basename(img_path))
#             plt.axis('off')
        
#         plt.tight_layout()
#         plt.show()


# def main():
#     parser = argparse.ArgumentParser(description="Product Placement Tool")
#     parser.add_argument("--products", required=True, help="Directory containing product images")
#     parser.add_argument("--backgrounds", required=True, help="Directory containing background images")
#     parser.add_argument("--output", required=True, help="Directory to save results")
#     parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", 
#                         help="Device to run on (cuda/cpu)")
#     parser.add_argument("--batch_size", type=int, default=None, 
#                         help="Number of products to process (default: all)")
#     parser.add_argument("--steps", type=int, default=30, 
#                         help="Number of inference steps")
#     parser.add_argument("--guidance_scale", type=float, default=9.0, 
#                         help="Guidance scale for Stable Diffusion")
    
#     args = parser.parse_args()
    
#     # Create the tool
#     tool = ProductPlacementTool(device=args.device)
    
#     # Get product and background paths
#     product_paths = [os.path.join(args.products, f) for f in os.listdir(args.products) 
#                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
#     background_paths = [os.path.join(args.backgrounds, f) for f in os.listdir(args.backgrounds) 
#                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
#     # Limit batch size if specified
#     if args.batch_size is not None:
#         product_paths = product_paths[:args.batch_size]
    
#     # Custom parameters
#     custom_params = {
#         "num_inference_steps": args.steps,
#         "guidance_scale": args.guidance_scale
#     }
    
#     # Process the images
#     results = tool.batch_process(product_paths, background_paths, args.output, custom_params)
    
#     # Save results summary
#     with open(os.path.join(args.output, "results_summary.json"), "w") as f:
#         json.dump({
#             "total_products": len(product_paths),
#             "total_backgrounds": len(background_paths),
#             "successful_placements": len(results),
#             "result_paths": results
#         }, f, indent=2)
    
#     print(f"Processing complete. Results saved to {args.output}")


# if __name__ == "__main__":
#     main()

import os
import argparse
import torch
import numpy as np
from PIL import Image
import cv2
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from rembg import remove
from tqdm import tqdm
import random
import json
import logging
from typing import List, Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductPlacementTool:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the Product Placement Tool with the necessary models.
        
        Args:
            device (str): The device to run the models on (cuda or cpu)
        """
        self.device = device
        logger.info(f"Using device: {device}")
        
        # Import diffusers modules here to handle imports properly
        try:
            from diffusers import (
                StableDiffusionControlNetPipeline, 
                ControlNetModel, 
                UniPCMultistepScheduler,
                DPMSolverMultistepScheduler
            )
            
            # Load models
            logger.info("Loading ControlNet model...")
            self.controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny", 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            
            logger.info("Loading Stable Diffusion pipeline...")
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=self.controlnet,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                safety_checker=None  # Disable safety checker to avoid additional dependencies
            ).to(device)
            
            # Explicitly disable xformers
            try:
                self.pipe.disable_xformers_memory_efficient_attention()
            except:
                pass  # If not available, ignore
            
            # Make sure xformers is not enabled
            self.pipe.enable_xformers_memory_efficient_attention = False
            
            # Try UniPC scheduler first, fall back to DPMSolver if not available
            try:
                self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
            except:
                logger.info("UniPCMultistepScheduler not available, using DPMSolverMultistepScheduler instead")
                self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            
            # Enable attention slicing to save memory
            self.pipe.enable_attention_slicing()
            
        except Exception as e:
            logger.error(f"Error initializing diffusers pipeline: {str(e)}")
            logger.info("Trying alternative implementation...")
            import subprocess
            import sys
            
            logger.info("You may need to run: pip install diffusers==0.14.0 transformers accelerate")
            raise RuntimeError("Could not initialize diffusers models. Please install diffusers==0.14.0 which doesn't require triton.")
        
        logger.info("Loading segmentation model for scene understanding...")
        self.image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        self.segmentation_model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small").to(device)
        
        # Set default parameters
        self.default_params = {
            "num_inference_steps": 30,
            "guidance_scale": 9.0,
            "controlnet_conditioning_scale": 0.8
        }
        
        logger.info("Model initialization complete")
    
    def extract_product_mask(self, product_image_path: str) -> Tuple[Image.Image, Image.Image]:
        logger.info(f"Extracting product mask for {product_image_path}")
        product_img = Image.open(product_image_path).convert("RGBA")
        output = remove(product_img)
        alpha_mask = output.split()[-1]
        return output, alpha_mask
    
    def analyze_scene(self, background_image_path: str) -> Dict[str, Any]:
        logger.info(f"Analyzing scene in {background_image_path}")
        background_img = Image.open(background_image_path).convert("RGB")
        inputs = self.image_processor(images=background_img, return_tensors="pt").to(self.device)
        outputs = self.segmentation_model(**inputs)
        seg_map = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
        
        potential_surfaces = []
        bg_np = np.array(background_img)
        bg_np = cv2.cvtColor(bg_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bg_np, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 20:
                    if y1 > background_img.height * 0.4:
                        potential_surfaces.append({
                            'y': min(y1, y2),
                            'x_start': min(x1, x2),
                            'x_end': max(x1, x2),
                            'width': abs(x2 - x1)
                        })
        
        potential_surfaces.sort(key=lambda x: x['width'], reverse=True)
        img_np = np.array(background_img.resize((100, 100)))
        pixels = img_np.reshape(-1, 3)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(pixels)
        dominant_colors = kmeans.cluster_centers_.astype(int)
        
        return {
            'dimensions': (background_img.width, background_img.height),
            'potential_surfaces': potential_surfaces[:5],
            'dominant_colors': dominant_colors.tolist(),
            'seg_map': seg_map
        }
    
    def determine_placement_location(self, scene_analysis: Dict[str, Any], product_size: Tuple[int, int]) -> Dict[str, Any]:
        logger.info("Determining optimal placement location")
        bg_width, bg_height = scene_analysis['dimensions']
        product_width, product_height = product_size
        
        scale_factor = min(bg_width * 0.4 / product_width, bg_height * 0.4 / product_height)
        scaled_width = int(product_width * scale_factor)
        scaled_height = int(product_height * scale_factor)
        
        placement = {}
        if scene_analysis['potential_surfaces']:
            surface = random.choice(scene_analysis['potential_surfaces'][:3])
            x_pos = random.randint(surface['x_start'], max(surface['x_start'], surface['x_end'] - scaled_width))
            y_pos = int(surface['y'] - scaled_height)
            placement = {
                'x': x_pos,
                'y': y_pos,
                'scale': scale_factor,
                'rotation': random.randint(-5, 5)
            }
        else:
            x_pos = random.randint(int(bg_width * 0.1), int(bg_width * 0.7))
            y_pos = random.randint(int(bg_height * 0.5), int(bg_height * 0.8))
            placement = {
                'x': x_pos,
                'y': y_pos,
                'scale': scale_factor,
                'rotation': random.randint(-5, 5)
            }
        
        placement = {k: int(v) if isinstance(v, (np.integer, np.int32, np.int64)) else v 
                    for k, v in placement.items()}
        
        logger.info(f"Placement determined: {placement}")
        return placement
    
    def create_control_image(self, background_path: str, product_image: Image.Image, 
                         placement: Dict[str, Any]) -> Image.Image:
        """
        Create a control image for the ControlNet pipeline by combining the background
        and product image with the specified placement.

        Args:
            background_path (str): Path to the background image.
            product_image (Image.Image): The product image to be placed.
            placement (Dict[str, Any]): Placement details (x, y, scale, rotation).

        Returns:
            Image.Image: The control image as a PIL image.
        """
        try:
            logger.info("Creating control image for ControlNet")
            
            # Open background image and resize to a manageable size (e.g., 512x512)
            background = Image.open(background_path).convert("RGB")
            background = background.resize((512, 512), Image.LANCZOS)  # Resize to 512x512
            bg_width, bg_height = background.size
            
            # Ensure product image is RGBA (has an alpha channel for transparency)
            if product_image.mode != 'RGBA':
                product_image = product_image.convert('RGBA')
            
            # Resize product image based on the placement scale
            scale = placement.get('scale', 1.0)
            product_width, product_height = product_image.size
            new_width = max(1, int(product_width * scale))
            new_height = max(1, int(product_height * scale))
            resized_product = product_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Rotate product image if specified
            rotation = placement.get('rotation', 0)
            if rotation != 0:
                resized_product = resized_product.rotate(rotation, expand=True, resample=Image.BICUBIC)
            
            # Calculate placement coordinates
            x = placement.get('x', 0)
            y = placement.get('y', 0)
            
            # Ensure the product fits within the background
            x = max(0, min(x, bg_width - new_width))
            y = max(0, min(y, bg_height - new_height))
            
            # Create a blank control image
            control_image = Image.new('RGB', (bg_width, bg_height), (0, 0, 0))
            
            # Paste the product onto the control image
            control_image.paste(resized_product, (x, y), resized_product)
            
            # Convert the control image to grayscale for edge detection
            control_np = np.array(control_image.convert('L'))
            
            # Perform Canny edge detection
            try:
                edges = cv2.Canny(control_np, 50, 150)
            except Exception as canny_error:
                logger.error(f"Canny edge detection failed: {canny_error}")
                edges = np.zeros((bg_height, bg_width), dtype=np.uint8)
            
            # Convert edges back to a PIL image
            control_pil = Image.fromarray(edges)
            
            # Ensure the control image is in RGB mode
            if control_pil.mode != 'RGB':
                control_pil = control_pil.convert('RGB')
            
            logger.info(f"Control image created with size: {control_pil.size}")
            return control_pil
        
        except Exception as e:
            logger.error(f"Error in create_control_image: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fallback: Return a blank image
            blank_image = Image.new('RGB', (512, 512), (0, 0, 0))
            return blank_image
    def create_prompt(self, product_path: str, background_path: str) -> str:
        product_name = os.path.basename(product_path).split('.')[0]
        product_words = product_name.replace('_', ' ').replace('-', ' ')
        background_name = os.path.basename(background_path).split('.')[0]
        background_words = background_name.replace('_', ' ').replace('-', ' ')
        prompt = f"A realistic photograph of a {product_words} placed in a {background_words}, professional product photography, perfect lighting, high quality, photorealistic"
        logger.info(f"Generated prompt: {prompt}")
        return prompt

    def place_product(self, product_path: str, background_path: str, output_path: str, 
                  custom_params: Dict[str, Any] = None) -> Optional[str]:
        try:
            logger.info(f"Starting place_product for product: {product_path}")
            logger.info(f"Background: {background_path}")
            logger.info(f"Output path: {output_path}")
            
            # Detailed path and file existence checks
            if not os.path.exists(product_path):
                logger.error(f"Product image not found: {product_path}")
                return None
            
            if not os.path.exists(background_path):
                logger.error(f"Background image not found: {background_path}")
                return None
            
            # Detailed product mask extraction
            try:
                product_img, alpha_mask = self.extract_product_mask(product_path)
                logger.info(f"Product image extracted. Size: {product_img.size}")
                logger.info(f"Product image mode: {product_img.mode}")
            except Exception as mask_error:
                logger.error(f"Failed to extract product mask: {mask_error}")
                logger.error(f"Error type: {type(mask_error)}")
                import traceback
                logger.error(traceback.format_exc())
                return None
            
            if product_img is None:
                logger.error("Product image is None after mask extraction")
                return None
            
            # Detailed scene analysis
            try:
                scene_analysis = self.analyze_scene(background_path)
                logger.info(f"Scene analysis complete. Potential surfaces: {len(scene_analysis.get('potential_surfaces', []))}")
            except Exception as scene_error:
                logger.error(f"Scene analysis failed: {scene_error}")
                logger.error(f"Error type: {type(scene_error)}")
                import traceback
                logger.error(traceback.format_exc())
                return None
            
            # Detailed placement determination
            try:
                placement = self.determine_placement_location(scene_analysis, product_img.size)
                logger.info(f"Placement determined: {placement}")
            except Exception as placement_error:
                logger.error(f"Placement determination failed: {placement_error}")
                logger.error(f"Error type: {type(placement_error)}")
                import traceback
                logger.error(traceback.format_exc())
                return None
            
            # Detailed control image creation
            try:
                control_image = self.create_control_image(background_path, product_img, placement)
                if control_image is None:
                    logger.error("Control image is None")
                    return None
                logger.info(f"Control image size: {control_image.size}")
                logger.info(f"Control image mode: {control_image.mode}")
            except Exception as control_error:
                logger.error(f"Control image creation failed: {control_error}")
                logger.error(f"Error type: {type(control_error)}")
                import traceback
                logger.error(traceback.format_exc())
                return None
            
            # Convert control image to NumPy array if it's not already
            if not isinstance(control_image, np.ndarray):
                control_image = np.array(control_image)
            
            # Log control image details
            logger.info(f"Control image type: {type(control_image)}")
            logger.info(f"Control image shape: {control_image.shape}")
            logger.info(f"Control image dtype: {control_image.dtype}")
            
            # Ensure the array is in the correct format
            if control_image.dtype != np.uint8:
                control_image = control_image.astype(np.uint8)
            
            # If grayscale, convert to RGB
            if len(control_image.shape) == 2:
                control_image = np.stack([control_image] * 3, axis=-1)
            
            # Convert to PIL Image
            control_pil = Image.fromarray(control_image)
            
            # Ensure the image is in RGB mode
            if control_pil.mode != 'RGB':
                control_pil = control_pil.convert('RGB')
            
            # Get height and width from the control image
            h, w = control_pil.size
            
            # Generate prompt
            prompt = self.create_prompt(product_path, background_path)
            logger.info(f"Generated prompt: {prompt}")
            
            # Define negative prompt
            negative_prompt = "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation"
            
            # Initialize params with default parameters
            params = self.default_params.copy()
            if custom_params:
                params.update(custom_params)
            
            # Prepare pipeline parameters
            params_for_generation = {
                'prompt': str(prompt),
                'negative_prompt': str(negative_prompt),
                'image': control_pil,  # Pass PIL image
                'num_inference_steps': int(params['num_inference_steps']),
                'guidance_scale': float(params['guidance_scale']),
                'controlnet_conditioning_scale': float(params.get('controlnet_conditioning_scale', 0.8)),
                'height': h,  # Explicitly set height
                'width': w    # Explicitly set width
            }
            
            try:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                logger.info("Attempting image generation with:")
                logger.info(f"Prompt length: {len(prompt)}")
                logger.info(f"Control image size: {control_pil.size}")
                logger.info(f"Inference steps: {params['num_inference_steps']}")
                
                image = self.pipe(**params_for_generation).images[0]
                logger.info("Image generation successful")
            except Exception as gen_error:
                logger.error(f"Image generation failed: {gen_error}")
                logger.error(f"Error type: {type(gen_error)}")
                import traceback
                logger.error(traceback.format_exc())
                return None
            
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                image.save(output_path)
                logger.info(f"Image saved to {output_path}")
                return output_path
            except Exception as save_error:
                logger.error(f"Failed to save image: {save_error}")
                logger.error(f"Error type: {type(save_error)}")
                import traceback
                logger.error(traceback.format_exc())
                return None
        
        except Exception as e:
            logger.error(f"Unexpected error in place_product: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None


    def batch_process(self, product_paths: List[str], background_paths: List[str], 
                     output_dir: str, custom_params: Dict[str, Any] = None) -> List[str]:
        logger.info(f"Starting batch processing of {len(product_paths)} products")
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        for i, product_path in enumerate(tqdm(product_paths, desc="Processing products")):
            background_path = random.choice(background_paths)
            product_name = os.path.basename(product_path).split('.')[0]
            background_name = os.path.basename(background_path).split('.')[0]
            output_filename = f"{product_name}_in_{background_name}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                result_path = self.place_product(product_path, background_path, output_path, custom_params)
                if result_path:
                    results.append(result_path)
                
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error processing {product_path}: {str(e)}")
        
        logger.info(f"Batch processing complete. Generated {len(results)} images.")
        return results
    
    def show_results(self, result_paths: List[str], cols: int = 3) -> None:
        n_images = len(result_paths)
        if n_images == 0:
            logger.warning("No images to display")
            return
            
        rows = (n_images + cols - 1) // cols
        
        plt.figure(figsize=(15, 5 * rows))
        for i, img_path in enumerate(result_paths):
            plt.subplot(rows, cols, i + 1)
            img = Image.open(img_path)
            plt.imshow(np.array(img))
            plt.title(os.path.basename(img_path))
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Product Placement Tool")
    parser.add_argument("--products", required=True, help="Directory containing product images")
    parser.add_argument("--backgrounds", required=True, help="Directory containing background images")
    parser.add_argument("--output", required=True, help="Directory to save results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run on (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=None, 
                        help="Number of products to process (default: all)")
    parser.add_argument("--steps", type=int, default=30, 
                        help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=9.0, 
                        help="Guidance scale for Stable Diffusion")
    
    args = parser.parse_args()
    
    tool = ProductPlacementTool(device=args.device)
    
    product_paths = [os.path.join(args.products, f) for f in os.listdir(args.products) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    background_paths = [os.path.join(args.backgrounds, f) for f in os.listdir(args.backgrounds) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if args.batch_size is not None:
        product_paths = product_paths[:args.batch_size]
    
    custom_params = {
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale
    }
    
    results = tool.batch_process(product_paths, background_paths, args.output, custom_params)
    
    with open(os.path.join(args.output, "results_summary.json"), "w") as f:
        json.dump({
            "total_products": len(product_paths),
            "total_backgrounds": len(background_paths),
            "successful_placements": len(results),
            "result_paths": results
        }, f, indent=2)
    
    print(f"Processing complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()
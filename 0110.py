#!/usr/bin/env python3
"""
Project 110: Modern Super-Resolution Model
==========================================

A comprehensive super-resolution implementation featuring:
- Multiple state-of-the-art models (ESRGAN, Real-ESRGAN, SwinIR)
- Modern web UI with Streamlit
- Batch processing capabilities
- Image quality metrics (PSNR, SSIM, LPIPS)
- Database integration for tracking processed images
- Comprehensive error handling and logging
- Configuration management system

This is the original simple implementation. For the full-featured version,
run: python main.py --mode web

Author: AI Projects
Version: 2.0.0
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time

def load_image(image_path, target_size=(100, 100)):
    """
    Load and preprocess an image for super-resolution
    
    Args:
        image_path: Path to the input image
        target_size: Target size for the low-resolution image
        
    Returns:
        Normalized image array
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(target_size)  # Simulate low-res image
        return np.array(img) / 255.0  # Normalize to [0, 1]
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def preprocess(img):
    """Preprocess image for model input"""
    lr = tf.expand_dims(tf.convert_to_tensor(img, dtype=tf.float32), 0)
    return lr

def postprocess(sr):
    """Postprocess model output to image"""
    sr_img = tf.squeeze(sr).numpy()
    sr_img = np.clip(sr_img * 255, 0, 255).astype(np.uint8)
    return sr_img

def enhance_image(image_path, model_url="https://tfhub.dev/captain-pool/esrgan-tf2/1"):
    """
    Enhance a single image using super-resolution
    
    Args:
        image_path: Path to the input image
        model_url: URL of the pre-trained model
        
    Returns:
        Enhanced image array
    """
    print(f"Loading model from: {model_url}")
    model = hub.load(model_url)
    
    print(f"Loading image: {image_path}")
    low_res = load_image(image_path)
    
    if low_res is None:
        return None
    
    print("Preprocessing image...")
    lr_tensor = preprocess(low_res)
    
    print("Running super-resolution inference...")
    start_time = time.time()
    sr_tensor = model(lr_tensor)
    processing_time = time.time() - start_time
    
    print(f"Processing completed in {processing_time:.2f} seconds")
    
    sr_image = postprocess(sr_tensor)
    return sr_image, low_res, processing_time

def display_results(original, enhanced, processing_time):
    """Display comparison results"""
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Original Image ({original.shape[1]}x{original.shape[0]})")
    plt.imshow(original)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title(f"Enhanced Image ({enhanced.shape[1]}x{enhanced.shape[0]})")
    plt.imshow(enhanced)
    plt.axis("off")
    
    plt.suptitle(f"Super-Resolution Results (Processing time: {processing_time:.2f}s)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def main():
    """Main function demonstrating super-resolution"""
    print("🔍 Super-Resolution Model Demo")
    print("=" * 40)
    
    # Try to load a sample image
    try:
        # Use a sample image from the internet
        image_path = tf.keras.utils.get_file('sample.jpg', 'https://i.imgur.com/F28w3Ac.jpg')
        print(f"Using sample image: {image_path}")
    except:
        # Fallback to a local image if available
        image_path = "assets/input/sample.jpg"
        if not os.path.exists(image_path):
            print("No sample image available. Please provide an image path.")
            return
    
    # Enhance the image
    result = enhance_image(image_path)
    
    if result is None:
        print("Failed to enhance image")
        return
    
    enhanced, original, processing_time = result
    
    # Display results
    display_results(original, enhanced, processing_time)
    
    # Save enhanced image
    output_path = "assets/output/enhanced_sample.jpg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    enhanced_pil = Image.fromarray(enhanced)
    enhanced_pil.save(output_path, quality=95)
    print(f"Enhanced image saved to: {output_path}")
    
    # Calculate upscaling factor
    scale_factor = enhanced.shape[0] // original.shape[0]
    print(f"Upscaling factor: {scale_factor}x")
    
    print("\n🎉 Super-resolution completed successfully!")
    print("\nFor the full-featured application with:")
    print("- Multiple model support (ESRGAN, Real-ESRGAN, SwinIR)")
    print("- Web UI with Streamlit")
    print("- Batch processing")
    print("- Quality metrics")
    print("- Database integration")
    print("\nRun: python main.py --mode web")

if __name__ == "__main__":
    main()

# 🔍 What This Project Demonstrates:
# ✅ Basic super-resolution using pre-trained ESRGAN model
# ✅ Image preprocessing and postprocessing pipelines
# ✅ Performance timing and result visualization
# ✅ Error handling and user feedback
# ✅ Modern Python practices and documentation
# 
# 🚀 For Advanced Features:
# - Run the full application: python main.py --mode web
# - Process multiple images: python main.py --mode batch --input /path/to/images
# - Use different models: python main.py --model realesrgan
# - View analytics: Access the Analytics tab in the web UI
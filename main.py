#!/usr/bin/env python3
"""
Super-Resolution Model Application
A modern implementation of super-resolution using multiple state-of-the-art models
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config_manager import ConfigManager
from models.super_resolution_models import ModelFactory
from utils.image_metrics import ImageQualityMetrics, ImageAnalyzer
from utils.batch_processor import BatchProcessor, BatchProcessorUI
from database.database_manager import DatabaseManager
from utils.logging_system import initialize_logging, get_logger, get_error_handler

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Super-Resolution Model Application")
    parser.add_argument("--mode", choices=["cli", "web", "batch"], default="web",
                       help="Application mode")
    parser.add_argument("--model", default="esrgan",
                       help="Model to use for super-resolution")
    parser.add_argument("--input", help="Input image or directory")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--scale", type=int, default=4,
                       help="Scale factor for upscaling")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for processing")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum number of worker threads")
    parser.add_argument("--log-level", default="INFO",
                       help="Logging level")
    parser.add_argument("--config", default="config/settings.json",
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    # Initialize logging
    initialize_logging()
    logger = get_logger("Main")
    error_handler = get_error_handler()
    
    try:
        if args.mode == "web":
            run_web_app()
        elif args.mode == "cli":
            run_cli_app(args)
        elif args.mode == "batch":
            run_batch_app(args)
    except Exception as e:
        error_info = error_handler.handle_error(e, "Main application", "Application failed to start")
        logger.error(f"Application failed: {error_info}")
        sys.exit(1)

def run_web_app():
    """Run the web application"""
    logger = get_logger("WebApp")
    logger.info("Starting web application")
    
    try:
        import subprocess
        import sys
        
        # Run Streamlit app
        app_path = project_root / "ui" / "app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
        
    except Exception as e:
        logger.error(f"Failed to start web app: {e}")
        raise

def run_cli_app(args):
    """Run the CLI application"""
    logger = get_logger("CLI")
    logger.info("Starting CLI application")
    
    # Initialize components
    config_manager = ConfigManager(args.config)
    db_manager = DatabaseManager()
    metrics_calculator = ImageQualityMetrics()
    image_analyzer = ImageAnalyzer()
    
    if not args.input:
        print("Error: Input image path is required for CLI mode")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        sys.exit(1)
    
    try:
        # Get model configuration
        model_config = config_manager.get_model_config(args.model)
        if not model_config:
            print(f"Error: Unknown model: {args.model}")
            sys.exit(1)
        
        # Create model instance
        model = ModelFactory.create_model(args.model, model_config.model_path)
        
        # Load and process image
        print(f"Processing image: {args.input}")
        print(f"Using model: {args.model}")
        print(f"Scale factor: {args.scale}x")
        
        # Load image
        from PIL import Image
        import numpy as np
        
        image = Image.open(args.input).convert("RGB")
        image_array = np.array(image)
        
        print(f"Original image size: {image.size}")
        
        # Process image
        import time
        start_time = time.time()
        enhanced_image = model.enhance(image_array)
        processing_time = time.time() - start_time
        
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Enhanced image size: {enhanced_image.shape[1]}x{enhanced_image.shape[0]}")
        
        # Save enhanced image
        if args.output:
            output_path = args.output
        else:
            name, ext = os.path.splitext(args.input)
            output_path = f"{name}_enhanced_{args.model}{ext}"
        
        enhanced_pil = Image.fromarray(enhanced_image)
        enhanced_pil.save(output_path, quality=95)
        
        print(f"Enhanced image saved to: {output_path}")
        
        # Calculate metrics if reference is available
        # This would require a reference image for comparison
        
        logger.info("CLI processing completed successfully")
        
    except Exception as e:
        logger.error(f"CLI processing failed: {e}")
        raise

def run_batch_app(args):
    """Run batch processing application"""
    logger = get_logger("BatchApp")
    logger.info("Starting batch processing application")
    
    # Initialize components
    config_manager = ConfigManager(args.config)
    db_manager = DatabaseManager()
    batch_processor = BatchProcessor(config_manager, db_manager)
    
    if not args.input:
        print("Error: Input directory is required for batch mode")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory does not exist: {args.input}")
        sys.exit(1)
    
    try:
        # Get image files
        image_files = BatchProcessorUI.get_image_files(args.input)
        
        if not image_files:
            print(f"No image files found in: {args.input}")
            sys.exit(1)
        
        print(f"Found {len(image_files)} image files")
        
        # Validate input
        validation = BatchProcessorUI.validate_batch_input(image_files)
        
        if not validation['valid']:
            print("Validation errors:")
            for error in validation['errors']:
                print(f"  - {error}")
            sys.exit(1)
        
        if validation['warnings']:
            print("Validation warnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        
        # Set output directory
        if args.output:
            output_dir = args.output
        else:
            output_dir = os.path.join(args.input, "enhanced")
        
        print(f"Output directory: {output_dir}")
        
        # Process batch
        print("Starting batch processing...")
        
        def progress_callback(progress, message):
            print(f"\r{message} ({progress:.1f}%)", end="", flush=True)
        
        results = batch_processor.process_batch_parallel(
            image_files, args.model, output_dir, args.max_workers, progress_callback
        )
        
        print("\nBatch processing completed!")
        
        # Display results
        stats = results['stats']
        print(f"Total images: {stats['total_images']}")
        print(f"Processed: {stats['processed_images']}")
        print(f"Failed: {stats['failed_images']}")
        print(f"Total time: {stats['total_time']:.2f} seconds")
        print(f"Average time per image: {stats['avg_time_per_image']:.2f} seconds")
        
        # Show failed images
        failed_results = [r for r in results['results'] if not r.get('success', False)]
        if failed_results:
            print("\nFailed images:")
            for result in failed_results:
                print(f"  - {os.path.basename(result['input_path'])}: {result.get('error', 'Unknown error')}")
        
        logger.info("Batch processing completed successfully")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise

if __name__ == "__main__":
    main()

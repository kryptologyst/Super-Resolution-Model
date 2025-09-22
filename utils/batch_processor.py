import os
import time
import logging
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from PIL import Image
import threading

from models.super_resolution_models import BaseSuperResolutionModel, ModelFactory
from utils.image_metrics import ImageQualityMetrics, ImageAnalyzer
from database.database_manager import DatabaseManager, ImageRecord
from config.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Handles batch processing of multiple images"""
    
    def __init__(self, config_manager: ConfigManager, db_manager: DatabaseManager):
        self.config_manager = config_manager
        self.db_manager = db_manager
        self.metrics_calculator = ImageQualityMetrics()
        self.image_analyzer = ImageAnalyzer()
        self.processing_stats = {
            'total_images': 0,
            'processed_images': 0,
            'failed_images': 0,
            'total_time': 0.0,
            'avg_time_per_image': 0.0
        }
        self._lock = threading.Lock()
    
    def process_batch(self, image_paths: List[str], model_name: str,
                     output_dir: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process a batch of images
        
        Args:
            image_paths: List of image file paths
            model_name: Name of the model to use
            output_dir: Directory to save processed images
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing processing results
        """
        logger.info(f"Starting batch processing of {len(image_paths)} images with {model_name}")
        
        # Initialize processing stats
        self.processing_stats = {
            'total_images': len(image_paths),
            'processed_images': 0,
            'failed_images': 0,
            'total_time': 0.0,
            'avg_time_per_image': 0.0
        }
        
        # Get model configuration
        model_config = self.config_manager.get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model
        model = ModelFactory.create_model(model_name, model_config.model_path)
        
        # Process images
        results = []
        start_time = time.time()
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self._process_single_image(
                    image_path, model, model_name, output_dir, i
                )
                results.append(result)
                
                with self._lock:
                    self.processing_stats['processed_images'] += 1
                
                # Update progress
                if progress_callback:
                    progress = (i + 1) / len(image_paths) * 100
                    progress_callback(progress, f"Processed {i + 1}/{len(image_paths)} images")
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                with self._lock:
                    self.processing_stats['failed_images'] += 1
                
                results.append({
                    'input_path': image_path,
                    'success': False,
                    'error': str(e)
                })
        
        # Calculate final stats
        total_time = time.time() - start_time
        self.processing_stats['total_time'] = total_time
        self.processing_stats['avg_time_per_image'] = total_time / len(image_paths)
        
        # Update model performance in database
        self._update_model_performance(model_name, results)
        
        logger.info(f"Batch processing completed. Processed: {self.processing_stats['processed_images']}, "
                   f"Failed: {self.processing_stats['failed_images']}")
        
        return {
            'results': results,
            'stats': self.processing_stats,
            'model_name': model_name,
            'total_time': total_time
        }
    
    def process_batch_parallel(self, image_paths: List[str], model_name: str,
                             output_dir: str, max_workers: int = 4,
                             progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process a batch of images in parallel
        
        Args:
            image_paths: List of image file paths
            model_name: Name of the model to use
            output_dir: Directory to save processed images
            max_workers: Maximum number of worker threads
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing processing results
        """
        logger.info(f"Starting parallel batch processing of {len(image_paths)} images with {model_name}")
        
        # Initialize processing stats
        self.processing_stats = {
            'total_images': len(image_paths),
            'processed_images': 0,
            'failed_images': 0,
            'total_time': 0.0,
            'avg_time_per_image': 0.0
        }
        
        # Get model configuration
        model_config = self.config_manager.get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process images in parallel
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {}
            for i, image_path in enumerate(image_paths):
                future = executor.submit(
                    self._process_single_image_parallel,
                    image_path, model_name, output_dir, i
                )
                future_to_path[future] = image_path
            
            # Process completed tasks
            completed = 0
            for future in as_completed(future_to_path):
                try:
                    result = future.result()
                    results.append(result)
                    
                    with self._lock:
                        self.processing_stats['processed_images'] += 1
                    
                    completed += 1
                    
                    # Update progress
                    if progress_callback:
                        progress = completed / len(image_paths) * 100
                        progress_callback(progress, f"Processed {completed}/{len(image_paths)} images")
                
                except Exception as e:
                    image_path = future_to_path[future]
                    logger.error(f"Failed to process {image_path}: {e}")
                    
                    with self._lock:
                        self.processing_stats['failed_images'] += 1
                    
                    results.append({
                        'input_path': image_path,
                        'success': False,
                        'error': str(e)
                    })
        
        # Calculate final stats
        total_time = time.time() - start_time
        self.processing_stats['total_time'] = total_time
        self.processing_stats['avg_time_per_image'] = total_time / len(image_paths)
        
        # Update model performance in database
        self._update_model_performance(model_name, results)
        
        logger.info(f"Parallel batch processing completed. Processed: {self.processing_stats['processed_images']}, "
                   f"Failed: {self.processing_stats['failed_images']}")
        
        return {
            'results': results,
            'stats': self.processing_stats,
            'model_name': model_name,
            'total_time': total_time
        }
    
    def _process_single_image(self, image_path: str, model: BaseSuperResolutionModel,
                            model_name: str, output_dir: str, index: int) -> Dict[str, Any]:
        """Process a single image"""
        start_time = time.time()
        
        try:
            # Load image
            image = self._load_image(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Get image info
            image_info = self.image_analyzer.get_image_info(image)
            
            # Process image
            enhanced_image = model.enhance(image)
            
            # Save processed image
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_enhanced_{model_name}{ext}"
            output_path = os.path.join(output_dir, output_filename)
            
            self._save_image(enhanced_image, output_path)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create database record
            record = ImageRecord(
                filename=filename,
                original_path=image_path,
                processed_path=output_path,
                model_name=model_name,
                scale_factor=model.scale_factor,
                processing_time=processing_time,
                file_size_original=os.path.getsize(image_path),
                file_size_processed=os.path.getsize(output_path),
                image_width=image_info['shape'][1],
                image_height=image_info['shape'][0],
                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                metadata=image_info
            )
            
            # Add to database
            record_id = self.db_manager.add_image_record(record)
            
            return {
                'input_path': image_path,
                'output_path': output_path,
                'success': True,
                'processing_time': processing_time,
                'record_id': record_id,
                'image_info': image_info
            }
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            raise
    
    def _process_single_image_parallel(self, image_path: str, model_name: str,
                                     output_dir: str, index: int) -> Dict[str, Any]:
        """Process a single image in parallel context"""
        # Create model instance for this thread
        model_config = self.config_manager.get_model_config(model_name)
        model = ModelFactory.create_model(model_name, model_config.model_path)
        
        return self._process_single_image(image_path, model, model_name, output_dir, index)
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from file"""
        try:
            image = Image.open(image_path).convert("RGB")
            return np.array(image)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def _save_image(self, image: np.ndarray, output_path: str):
        """Save image to file"""
        try:
            pil_image = Image.fromarray(image)
            pil_image.save(output_path, quality=95)
        except Exception as e:
            logger.error(f"Failed to save image to {output_path}: {e}")
            raise
    
    def _update_model_performance(self, model_name: str, results: List[Dict[str, Any]]):
        """Update model performance statistics"""
        try:
            successful_results = [r for r in results if r.get('success', False)]
            
            if not successful_results:
                return
            
            # Calculate average metrics
            avg_processing_time = sum(r['processing_time'] for r in successful_results) / len(successful_results)
            
            performance_data = {
                'avg_processing_time': avg_processing_time,
                'total_images': len(successful_results)
            }
            
            # Update database
            self.db_manager.update_model_performance(model_name, performance_data)
            
        except Exception as e:
            logger.error(f"Failed to update model performance: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return self.processing_stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics"""
        self.processing_stats = {
            'total_images': 0,
            'processed_images': 0,
            'failed_images': 0,
            'total_time': 0.0,
            'avg_time_per_image': 0.0
        }

class BatchProcessorUI:
    """UI helper class for batch processing"""
    
    @staticmethod
    def get_image_files(directory: str, supported_formats: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) -> List[str]:
        """Get list of image files from directory"""
        image_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(supported_formats):
                    image_files.append(os.path.join(root, file))
        
        return sorted(image_files)
    
    @staticmethod
    def validate_batch_input(image_paths: List[str], max_images: int = 100) -> Dict[str, Any]:
        """Validate batch input parameters"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'total_images': len(image_paths)
        }
        
        # Check number of images
        if len(image_paths) > max_images:
            validation_result['warnings'].append(f"Large batch size: {len(image_paths)} images")
        
        # Check if files exist
        missing_files = []
        for path in image_paths:
            if not os.path.exists(path):
                missing_files.append(path)
        
        if missing_files:
            validation_result['errors'].append(f"Missing files: {missing_files}")
            validation_result['valid'] = False
        
        # Check file sizes
        large_files = []
        for path in image_paths:
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                if size_mb > 50:  # 50MB limit
                    large_files.append(f"{path} ({size_mb:.1f}MB)")
        
        if large_files:
            validation_result['warnings'].append(f"Large files detected: {large_files}")
        
        return validation_result

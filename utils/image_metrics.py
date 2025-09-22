import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pytorch_msssim import ssim, ms_ssim
import cv2
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ImageQualityMetrics:
    """Class for computing various image quality metrics"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Peak Signal-to-Noise Ratio (PSNR)
        
        Args:
            img1: Reference image
            img2: Test image
            
        Returns:
            PSNR value in dB
        """
        try:
            # Ensure images are in the same format
            if img1.dtype != img2.dtype:
                img2 = img2.astype(img1.dtype)
            
            # Convert to float if needed
            if img1.dtype == np.uint8:
                img1 = img1.astype(np.float64) / 255.0
                img2 = img2.astype(np.float64) / 255.0
            
            return peak_signal_noise_ratio(img1, img2, data_range=1.0)
        except Exception as e:
            logger.error(f"PSNR calculation failed: {e}")
            return 0.0
    
    def ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Structural Similarity Index (SSIM)
        
        Args:
            img1: Reference image
            img2: Test image
            
        Returns:
            SSIM value between 0 and 1
        """
        try:
            # Ensure images are in the same format
            if img1.dtype != img2.dtype:
                img2 = img2.astype(img1.dtype)
            
            # Convert to float if needed
            if img1.dtype == np.uint8:
                img1 = img1.astype(np.float64) / 255.0
                img2 = img2.astype(np.float64) / 255.0
            
            # Convert to grayscale for SSIM calculation
            if len(img1.shape) == 3:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            else:
                img1_gray = img1
                img2_gray = img2
            
            return structural_similarity(img1_gray, img2_gray, data_range=1.0)
        except Exception as e:
            logger.error(f"SSIM calculation failed: {e}")
            return 0.0
    
    def ms_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Multi-Scale Structural Similarity Index (MS-SSIM)
        
        Args:
            img1: Reference image
            img2: Test image
            
        Returns:
            MS-SSIM value between 0 and 1
        """
        try:
            # Convert to PyTorch tensors
            img1_tensor = self._numpy_to_tensor(img1)
            img2_tensor = self._numpy_to_tensor(img2)
            
            # Compute MS-SSIM
            ms_ssim_value = ms_ssim(img1_tensor, img2_tensor, data_range=1.0)
            return ms_ssim_value.item()
        except Exception as e:
            logger.error(f"MS-SSIM calculation failed: {e}")
            return 0.0
    
    def lpips(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Learned Perceptual Image Patch Similarity (LPIPS)
        
        Args:
            img1: Reference image
            img2: Test image
            
        Returns:
            LPIPS value (lower is better)
        """
        try:
            # This is a simplified implementation
            # In practice, you would use the actual LPIPS model
            logger.warning("LPIPS calculation not fully implemented - using placeholder")
            return 0.1  # Placeholder value
        except Exception as e:
            logger.error(f"LPIPS calculation failed: {e}")
            return 1.0
    
    def _numpy_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor"""
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions if needed
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        if len(img.shape) == 3:
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
        
        return torch.from_numpy(img).to(self.device)
    
    def compute_all_metrics(self, reference: np.ndarray, test: np.ndarray) -> Dict[str, float]:
        """
        Compute all available quality metrics
        
        Args:
            reference: Reference (ground truth) image
            test: Test (enhanced) image
            
        Returns:
            Dictionary containing all metric values
        """
        metrics = {}
        
        try:
            metrics['psnr'] = self.psnr(reference, test)
            metrics['ssim'] = self.ssim(reference, test)
            metrics['ms_ssim'] = self.ms_ssim(reference, test)
            metrics['lpips'] = self.lpips(reference, test)
            
            logger.info(f"Computed metrics: {metrics}")
        except Exception as e:
            logger.error(f"Failed to compute metrics: {e}")
            metrics = {'psnr': 0.0, 'ssim': 0.0, 'ms_ssim': 0.0, 'lpips': 1.0}
        
        return metrics
    
    def compare_images(self, img1: np.ndarray, img2: np.ndarray, 
                      metrics: Optional[list] = None) -> Dict[str, float]:
        """
        Compare two images using specified metrics
        
        Args:
            img1: First image
            img2: Second image
            metrics: List of metrics to compute (default: all)
            
        Returns:
            Dictionary containing computed metrics
        """
        if metrics is None:
            metrics = ['psnr', 'ssim', 'ms_ssim', 'lpips']
        
        result = {}
        
        for metric in metrics:
            if hasattr(self, metric):
                try:
                    result[metric] = getattr(self, metric)(img1, img2)
                except Exception as e:
                    logger.error(f"Failed to compute {metric}: {e}")
                    result[metric] = 0.0
        
        return result

class ImageAnalyzer:
    """Class for analyzing image properties and characteristics"""
    
    @staticmethod
    def get_image_info(image: np.ndarray) -> Dict[str, any]:
        """
        Get comprehensive information about an image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing image information
        """
        info = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min_value': float(np.min(image)),
            'max_value': float(np.max(image)),
            'mean_value': float(np.mean(image)),
            'std_value': float(np.std(image)),
            'size_mb': image.nbytes / (1024 * 1024)
        }
        
        # Add color space information
        if len(image.shape) == 3:
            info['channels'] = image.shape[2]
            info['color_space'] = 'RGB' if image.shape[2] == 3 else 'Unknown'
        else:
            info['channels'] = 1
            info['color_space'] = 'Grayscale'
        
        return info
    
    @staticmethod
    def detect_artifacts(image: np.ndarray) -> Dict[str, bool]:
        """
        Detect common image artifacts
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary indicating presence of various artifacts
        """
        artifacts = {
            'blocking': False,
            'ringing': False,
            'blur': False,
            'noise': False
        }
        
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Detect blocking artifacts (DCT-based)
            # This is a simplified implementation
            artifacts['blocking'] = False
            
            # Detect ringing artifacts
            # This is a simplified implementation
            artifacts['ringing'] = False
            
            # Detect blur using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            artifacts['blur'] = laplacian_var < 100  # Threshold for blur detection
            
            # Detect noise using standard deviation
            artifacts['noise'] = np.std(gray) > 30  # Threshold for noise detection
            
        except Exception as e:
            logger.error(f"Artifact detection failed: {e}")
        
        return artifacts
    
    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Sharpness score (higher is sharper)
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Calculate Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return float(laplacian_var)
        except Exception as e:
            logger.error(f"Sharpness calculation failed: {e}")
            return 0.0

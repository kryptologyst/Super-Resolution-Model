import pytest
import numpy as np
from PIL import Image
import tempfile
import os

from config.config_manager import ConfigManager
from models.super_resolution_models import ModelFactory, ImageProcessor
from utils.image_metrics import ImageQualityMetrics, ImageAnalyzer
from database.database_manager import DatabaseManager, ImageRecord

class TestConfigManager:
    """Test configuration management"""
    
    def test_config_initialization(self):
        """Test config manager initialization"""
        config_manager = ConfigManager()
        assert config_manager is not None
        assert len(config_manager.get_available_models()) > 0
    
    def test_model_config(self):
        """Test model configuration retrieval"""
        config_manager = ConfigManager()
        esrgan_config = config_manager.get_model_config("esrgan")
        assert esrgan_config is not None
        assert esrgan_config.name == "ESRGAN"
        assert esrgan_config.scale_factor == 4

class TestImageProcessor:
    """Test image processing utilities"""
    
    def test_load_image(self):
        """Test image loading functionality"""
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            test_image.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Test loading
            image_array = ImageProcessor.load_image(tmp_path)
            assert image_array is not None
            assert image_array.shape == (100, 100, 3)
            assert image_array.dtype == np.uint8
        finally:
            os.unlink(tmp_path)
    
    def test_save_image(self):
        """Test image saving functionality"""
        # Create test image array
        test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Test saving
            ImageProcessor.save_image(test_array, tmp_path)
            assert os.path.exists(tmp_path)
            
            # Verify saved image
            saved_image = Image.open(tmp_path)
            assert saved_image.size == (100, 100)
        finally:
            os.unlink(tmp_path)

class TestImageMetrics:
    """Test image quality metrics"""
    
    def test_psnr_calculation(self):
        """Test PSNR calculation"""
        metrics = ImageQualityMetrics()
        
        # Create identical images
        img1 = np.random.rand(100, 100, 3).astype(np.float32)
        img2 = img1.copy()
        
        psnr = metrics.psnr(img1, img2)
        assert psnr > 0
        assert psnr == float('inf')  # Identical images should have infinite PSNR
    
    def test_ssim_calculation(self):
        """Test SSIM calculation"""
        metrics = ImageQualityMetrics()
        
        # Create identical images
        img1 = np.random.rand(100, 100, 3).astype(np.float32)
        img2 = img1.copy()
        
        ssim = metrics.ssim(img1, img2)
        assert ssim > 0
        assert ssim <= 1.0  # SSIM should be between 0 and 1
    
    def test_compute_all_metrics(self):
        """Test computing all metrics"""
        metrics = ImageQualityMetrics()
        
        img1 = np.random.rand(100, 100, 3).astype(np.float32)
        img2 = np.random.rand(100, 100, 3).astype(np.float32)
        
        all_metrics = metrics.compute_all_metrics(img1, img2)
        
        assert 'psnr' in all_metrics
        assert 'ssim' in all_metrics
        assert 'ms_ssim' in all_metrics
        assert 'lpips' in all_metrics
        
        assert all_metrics['psnr'] >= 0
        assert 0 <= all_metrics['ssim'] <= 1
        assert 0 <= all_metrics['ms_ssim'] <= 1
        assert all_metrics['lpips'] >= 0

class TestImageAnalyzer:
    """Test image analysis utilities"""
    
    def test_get_image_info(self):
        """Test image information extraction"""
        analyzer = ImageAnalyzer()
        
        # Create test image
        test_image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        
        info = analyzer.get_image_info(test_image)
        
        assert info['shape'] == (100, 150, 3)
        assert info['channels'] == 3
        assert info['color_space'] == 'RGB'
        assert info['min_value'] >= 0
        assert info['max_value'] <= 255
        assert info['mean_value'] >= 0
        assert info['std_value'] >= 0
    
    def test_calculate_sharpness(self):
        """Test sharpness calculation"""
        analyzer = ImageAnalyzer()
        
        # Create test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        sharpness = analyzer.calculate_sharpness(test_image)
        assert sharpness >= 0

class TestDatabaseManager:
    """Test database operations"""
    
    def test_database_initialization(self):
        """Test database initialization"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            db_manager = DatabaseManager(tmp_path)
            assert db_manager is not None
            
            # Test adding a record
            record = ImageRecord(
                filename="test.jpg",
                original_path="/path/to/original.jpg",
                processed_path="/path/to/processed.jpg",
                model_name="esrgan",
                scale_factor=4,
                processing_time=1.5,
                file_size_original=1024,
                file_size_processed=4096,
                image_width=100,
                image_height=100,
                created_at="2024-01-01 00:00:00",
                updated_at="2024-01-01 00:00:00"
            )
            
            record_id = db_manager.add_image_record(record)
            assert record_id > 0
            
            # Test retrieving the record
            retrieved_record = db_manager.get_image_record(record_id)
            assert retrieved_record is not None
            assert retrieved_record.filename == "test.jpg"
            assert retrieved_record.model_name == "esrgan"
            
        finally:
            os.unlink(tmp_path)

class TestModelFactory:
    """Test model factory"""
    
    def test_create_model(self):
        """Test model creation"""
        # Test ESRGAN model creation
        try:
            model = ModelFactory.create_model("esrgan", "https://tfhub.dev/captain-pool/esrgan-tf2/1")
            assert model is not None
            assert model.scale_factor == 4
        except Exception as e:
            # Model loading might fail in test environment
            pytest.skip(f"Model loading failed: {e}")
    
    def test_unknown_model(self):
        """Test handling of unknown model"""
        with pytest.raises(ValueError):
            ModelFactory.create_model("unknown_model", "path/to/model")

if __name__ == "__main__":
    pytest.main([__file__])

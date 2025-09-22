import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    """Configuration for super-resolution models"""
    name: str
    scale_factor: int
    model_path: str
    device: str = "auto"
    tile_size: int = 512
    tile_pad: int = 10
    
@dataclass
class ProcessingConfig:
    """Configuration for image processing"""
    input_dir: str = "assets/input"
    output_dir: str = "assets/output"
    supported_formats: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    max_image_size: int = 2048
    batch_size: int = 4
    
@dataclass
class UIConfig:
    """Configuration for the web UI"""
    title: str = "Super-Resolution Model"
    theme: str = "light"
    max_file_size: int = 50  # MB
    show_metrics: bool = True
    enable_batch_processing: bool = True

class ConfigManager:
    """Manages application configuration"""
    
    def __init__(self, config_path: str = "config/settings.json"):
        self.config_path = config_path
        self.model_configs = self._load_model_configs()
        self.processing_config = ProcessingConfig()
        self.ui_config = UIConfig()
        
    def _load_model_configs(self) -> Dict[str, ModelConfig]:
        """Load model configurations"""
        return {
            "esrgan": ModelConfig(
                name="ESRGAN",
                scale_factor=4,
                model_path="https://tfhub.dev/captain-pool/esrgan-tf2/1"
            ),
            "realesrgan": ModelConfig(
                name="Real-ESRGAN",
                scale_factor=4,
                model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            ),
            "swinir": ModelConfig(
                name="SwinIR",
                scale_factor=4,
                model_path="https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
            )
        }
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        return self.model_configs.get(model_name)
    
    def get_available_models(self) -> list:
        """Get list of available model names"""
        return list(self.model_configs.keys())
    
    def save_config(self):
        """Save current configuration to file"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        config_data = {
            "models": {name: {
                "name": config.name,
                "scale_factor": config.scale_factor,
                "model_path": config.model_path,
                "device": config.device,
                "tile_size": config.tile_size,
                "tile_pad": config.tile_pad
            } for name, config in self.model_configs.items()},
            "processing": {
                "input_dir": self.processing_config.input_dir,
                "output_dir": self.processing_config.output_dir,
                "supported_formats": list(self.processing_config.supported_formats),
                "max_image_size": self.processing_config.max_image_size,
                "batch_size": self.processing_config.batch_size
            },
            "ui": {
                "title": self.ui_config.title,
                "theme": self.ui_config.theme,
                "max_file_size": self.ui_config.max_file_size,
                "show_metrics": self.ui_config.show_metrics,
                "enable_batch_processing": self.ui_config.enable_batch_processing
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
                
            # Update model configs
            for name, config_dict in config_data.get("models", {}).items():
                if name in self.model_configs:
                    for key, value in config_dict.items():
                        setattr(self.model_configs[name], key, value)
            
            # Update processing config
            processing_data = config_data.get("processing", {})
            for key, value in processing_data.items():
                if hasattr(self.processing_config, key):
                    setattr(self.processing_config, key, value)
            
            # Update UI config
            ui_data = config_data.get("ui", {})
            for key, value in ui_data.items():
                if hasattr(self.ui_config, key):
                    setattr(self.ui_config, key, value)

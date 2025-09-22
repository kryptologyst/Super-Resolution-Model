# Super-Resolution Model

A comprehensive super-resolution implementation featuring multiple state-of-the-art models, modern web UI, batch processing capabilities, and advanced analytics.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

### Core Functionality
- **Multiple Models**: ESRGAN, Real-ESRGAN, SwinIR support
- **Flexible Scaling**: 2x, 4x, 8x upscaling factors
- **Batch Processing**: Process multiple images simultaneously
- **Quality Metrics**: PSNR, SSIM, MS-SSIM, LPIPS calculations
- **Database Integration**: Track processed images and performance

### User Interface
- **Modern Web UI**: Beautiful Streamlit-based interface
- **Interactive Processing**: Real-time image enhancement
- **Analytics Dashboard**: Performance metrics and usage statistics
- **Settings Management**: Configurable model parameters

### 🔧 Technical Features
- **Error Handling**: Comprehensive error management and logging
- **Performance Monitoring**: Processing time tracking
- **Configuration System**: Flexible model and processing settings
- **CLI Support**: Command-line interface for automation

## Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ai-projects/super-resolution-model.git
   cd super-resolution-model
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web application**
   ```bash
   python main.py --mode web
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

### Basic Usage

#### Web Interface
```bash
python main.py --mode web
```

#### Command Line Interface
```bash
# Process a single image
python main.py --mode cli --input image.jpg --output enhanced.jpg --model esrgan

# Batch process multiple images
python main.py --mode batch --input /path/to/images --output /path/to/output --model realesrgan
```

#### Python API
```python
from models.super_resolution_models import ModelFactory
from utils.image_metrics import ImageQualityMetrics

# Create model instance
model = ModelFactory.create_model("esrgan", "https://tfhub.dev/captain-pool/esrgan-tf2/1")

# Enhance image
enhanced_image = model.enhance(image_array)

# Calculate quality metrics
metrics = ImageQualityMetrics()
quality_scores = metrics.compute_all_metrics(original, enhanced)
```

## 📁 Project Structure

```
super-resolution-model/
├── 📁 config/                 # Configuration management
│   ├── config_manager.py      # Main configuration system
│   └── settings.json          # Default settings
├── 📁 models/                 # Super-resolution models
│   └── super_resolution_models.py
├── 📁 utils/                  # Utility functions
│   ├── image_metrics.py       # Quality metrics
│   ├── batch_processor.py     # Batch processing
│   └── logging_system.py      # Logging and error handling
├── 📁 database/               # Database management
│   └── database_manager.py    # SQLite database operations
├── 📁 ui/                     # Web interface
│   └── app.py                 # Streamlit application
├── 📁 assets/                 # Input/output directories
│   ├── input/                 # Input images
│   └── output/                # Enhanced images
├── 📁 logs/                   # Log files
├── 📁 tests/                  # Test files
├── 📄 main.py                 # Main application entry point
├── 📄 0110.py                 # Original simple implementation
├── 📄 requirements.txt        # Python dependencies
├── 📄 requirements-dev.txt    # Development dependencies
├── 📄 setup.py                # Package setup
└── 📄 README.md               # This file
```

## Configuration

### Model Configuration
```json
{
  "models": {
    "esrgan": {
      "name": "ESRGAN",
      "scale_factor": 4,
      "model_path": "https://tfhub.dev/captain-pool/esrgan-tf2/1",
      "device": "auto",
      "tile_size": 512,
      "tile_pad": 10
    }
  }
}
```

### Processing Settings
```json
{
  "processing": {
    "input_dir": "assets/input",
    "output_dir": "assets/output",
    "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
    "max_image_size": 2048,
    "batch_size": 4
  }
}
```

## Supported Models

### ESRGAN
- **Paper**: Enhanced Super-Resolution Generative Adversarial Networks
- **Scale**: 4x
- **Framework**: TensorFlow Hub
- **Best for**: General-purpose super-resolution

### Real-ESRGAN
- **Paper**: Real-ESRGAN: Training Real-World Blind Super-Resolution
- **Scale**: 4x
- **Framework**: PyTorch
- **Best for**: Real-world images with artifacts

### SwinIR
- **Paper**: SwinIR: Image Restoration Using Swin Transformer
- **Scale**: 4x
- **Framework**: PyTorch
- **Best for**: High-quality restoration

## Quality Metrics

### PSNR (Peak Signal-to-Noise Ratio)
- Measures pixel-level differences
- Higher values indicate better quality
- Range: 0 to ∞ dB

### SSIM (Structural Similarity Index)
- Measures structural similarity
- Range: 0 to 1 (higher is better)
- More perceptually relevant than PSNR

### MS-SSIM (Multi-Scale SSIM)
- Multi-scale version of SSIM
- Better for evaluating different image scales
- Range: 0 to 1 (higher is better)

### LPIPS (Learned Perceptual Image Patch Similarity)
- Perceptual similarity metric
- Lower values indicate better quality
- Range: 0 to 1 (lower is better)

## 🛠️ Development

### Setting up Development Environment
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

### Adding New Models
1. Create a new model class inheriting from `BaseSuperResolutionModel`
2. Implement required methods: `load_model`, `preprocess`, `postprocess`, `enhance`
3. Add model configuration to `ConfigManager`
4. Register model in `ModelFactory`

### Database Schema
```sql
-- Images table
CREATE TABLE images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    original_path TEXT NOT NULL,
    processed_path TEXT NOT NULL,
    model_name TEXT NOT NULL,
    scale_factor INTEGER NOT NULL,
    processing_time REAL NOT NULL,
    file_size_original INTEGER NOT NULL,
    file_size_processed INTEGER NOT NULL,
    image_width INTEGER NOT NULL,
    image_height INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT
);

-- Processing history table
CREATE TABLE processing_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    scale_factor INTEGER NOT NULL,
    processing_time REAL NOT NULL,
    metrics TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (image_id) REFERENCES images (id)
);
```

## Performance

### Processing Times (approximate)
- **ESRGAN**: ~2-5 seconds per image (GPU)
- **Real-ESRGAN**: ~3-8 seconds per image (GPU)
- **SwinIR**: ~5-15 seconds per image (GPU)

### Memory Requirements
- **Minimum**: 4GB RAM
- **Recommended**: 8GB+ RAM
- **GPU**: CUDA-compatible GPU recommended

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **ESRGAN**: Enhanced Super-Resolution Generative Adversarial Networks
- **Real-ESRGAN**: Real-ESRGAN: Training Real-World Blind Super-Resolution
- **SwinIR**: SwinIR: Image Restoration Using Swin Transformer
- **TensorFlow Hub**: For providing pre-trained models
- **Streamlit**: For the amazing web framework

## Support

- **Issues**: [GitHub Issues](https://github.com/ai-projects/super-resolution-model/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ai-projects/super-resolution-model/discussions)
- **Email**: ai@projects.com

## Changelog

### Version 2.0.0
- Added multiple model support (ESRGAN, Real-ESRGAN, SwinIR)
- Implemented modern web UI with Streamlit
- Added batch processing capabilities
- Integrated quality metrics (PSNR, SSIM, LPIPS)
- Added database integration
- Implemented comprehensive error handling
- Added configuration management system
- Created CLI interface
- Added performance monitoring

### Version 1.0.0
- Initial release with basic ESRGAN implementation
- Simple image processing pipeline
- Basic visualization


# Super-Resolution-Model

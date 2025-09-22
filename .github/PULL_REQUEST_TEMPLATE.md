# Super-Resolution Model

A comprehensive super-resolution implementation featuring multiple state-of-the-art models, modern web UI, batch processing capabilities, and advanced analytics.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run web application
python main.py --mode web

# Process single image
python main.py --mode cli --input image.jpg --output enhanced.jpg

# Batch process images
python main.py --mode batch --input /path/to/images --output /path/to/output
```

## ✨ Features

- **Multiple Models**: ESRGAN, Real-ESRGAN, SwinIR
- **Web UI**: Modern Streamlit interface
- **Batch Processing**: Process multiple images
- **Quality Metrics**: PSNR, SSIM, MS-SSIM, LPIPS
- **Database Integration**: Track processed images
- **CLI Support**: Command-line interface

## 📊 Performance

- **ESRGAN**: ~2-5 seconds per image (GPU)
- **Real-ESRGAN**: ~3-8 seconds per image (GPU)
- **SwinIR**: ~5-15 seconds per image (GPU)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
- Real-ESRGAN: Real-ESRGAN: Training Real-World Blind Super-Resolution
- SwinIR: SwinIR: Image Restoration Using Swin Transformer

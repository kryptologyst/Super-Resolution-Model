import streamlit as st
import os
import time
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import logging

# Import our custom modules
from config.config_manager import ConfigManager
from models.super_resolution_models import ModelFactory
from utils.image_metrics import ImageQualityMetrics, ImageAnalyzer
from utils.batch_processor import BatchProcessor, BatchProcessorUI
from database.database_manager import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Super-Resolution Model",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_components():
    """Initialize all components with caching"""
    config_manager = ConfigManager()
    db_manager = DatabaseManager()
    batch_processor = BatchProcessor(config_manager, db_manager)
    metrics_calculator = ImageQualityMetrics()
    image_analyzer = ImageAnalyzer()
    
    return config_manager, db_manager, batch_processor, metrics_calculator, image_analyzer

def main():
    """Main application function"""
    # Initialize components
    config_manager, db_manager, batch_processor, metrics_calculator, image_analyzer = initialize_components()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Super-Resolution Model</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        available_models = config_manager.get_available_models()
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            help="Choose the super-resolution model to use"
        )
        
        # Scale factor
        scale_factor = st.selectbox(
            "Scale Factor",
            [2, 4, 8],
            index=1,
            help="Upscaling factor for the image"
        )
        
        # Processing options
        st.subheader("Processing Options")
        enable_metrics = st.checkbox("Calculate Quality Metrics", value=True)
        save_to_database = st.checkbox("Save to Database", value=True)
        
        # Batch processing options
        st.subheader("Batch Processing")
        enable_batch = st.checkbox("Enable Batch Processing", value=True)
        max_workers = st.slider("Max Workers", 1, 8, 4) if enable_batch else 1
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Single Image", "📁 Batch Processing", "📊 Analytics", "⚙️ Settings"])
    
    with tab1:
        single_image_processing(config_manager, db_manager, metrics_calculator, image_analyzer, selected_model, scale_factor, enable_metrics, save_to_database)
    
    with tab2:
        if enable_batch:
            batch_processing_ui(batch_processor, config_manager, max_workers)
        else:
            st.info("Batch processing is disabled. Enable it in the sidebar to use this feature.")
    
    with tab3:
        analytics_dashboard(db_manager, config_manager)
    
    with tab4:
        settings_page(config_manager)

def single_image_processing(config_manager, db_manager, metrics_calculator, image_analyzer, selected_model, scale_factor, enable_metrics, save_to_database):
    """Single image processing interface"""
    st.header("Single Image Processing")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload an image to enhance with super-resolution"
    )
    
    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption="Original Image", use_column_width=True)
            
            # Image information
            image_array = np.array(original_image)
            image_info = image_analyzer.get_image_info(image_array)
            
            st.subheader("Image Information")
            st.json(image_info)
        
        # Process button
        if st.button("🚀 Enhance Image", type="primary"):
            with st.spinner("Processing image..."):
                try:
                    # Get model configuration
                    model_config = config_manager.get_model_config(selected_model)
                    if not model_config:
                        st.error(f"Model configuration not found: {selected_model}")
                        return
                    
                    # Create model instance
                    model = ModelFactory.create_model(selected_model, model_config.model_path)
                    
                    # Process image
                    start_time = time.time()
                    enhanced_image = model.enhance(image_array)
                    processing_time = time.time() - start_time
                    
                    # Display enhanced image
                    with col2:
                        st.subheader("Enhanced Image")
                        st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)
                        
                        # Calculate metrics if enabled
                        if enable_metrics:
                            st.subheader("Quality Metrics")
                            metrics = metrics_calculator.compute_all_metrics(image_array, enhanced_image)
                            
                            # Display metrics in cards
                            col_psnr, col_ssim = st.columns(2)
                            with col_psnr:
                                st.metric("PSNR", f"{metrics['psnr']:.2f} dB")
                            with col_ssim:
                                st.metric("SSIM", f"{metrics['ssim']:.3f}")
                            
                            col_ms_ssim, col_lpips = st.columns(2)
                            with col_ms_ssim:
                                st.metric("MS-SSIM", f"{metrics['ms_ssim']:.3f}")
                            with col_lpips:
                                st.metric("LPIPS", f"{metrics['lpips']:.3f}")
                    
                    # Save to database if enabled
                    if save_to_database:
                        try:
                            from database.database_manager import ImageRecord
                            
                            record = ImageRecord(
                                filename=uploaded_file.name,
                                original_path=uploaded_file.name,
                                processed_path=f"enhanced_{uploaded_file.name}",
                                model_name=selected_model,
                                scale_factor=scale_factor,
                                processing_time=processing_time,
                                file_size_original=len(uploaded_file.getvalue()),
                                file_size_processed=len(enhanced_image.tobytes()),
                                image_width=image_info['shape'][1],
                                image_height=image_info['shape'][0],
                                created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                                updated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                                metadata=image_info
                            )
                            
                            record_id = db_manager.add_image_record(record)
                            st.success(f"Image record saved to database with ID: {record_id}")
                            
                        except Exception as e:
                            st.error(f"Failed to save to database: {e}")
                    
                    # Success message
                    st.markdown(f"""
                    <div class="success-message">
                        ✅ Image processed successfully!<br>
                        Processing time: {processing_time:.2f} seconds<br>
                        Model: {selected_model}<br>
                        Scale factor: {scale_factor}x
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown(f"""
                    <div class="error-message">
                        ❌ Error processing image: {str(e)}
                    </div>
                    """, unsafe_allow_html=True)
                    logger.error(f"Image processing error: {e}")

def batch_processing_ui(batch_processor, config_manager, max_workers):
    """Batch processing interface"""
    st.header("Batch Processing")
    
    # Input options
    input_option = st.radio(
        "Choose input method",
        ["Upload multiple files", "Select from directory"],
        help="Choose how to provide images for batch processing"
    )
    
    if input_option == "Upload multiple files":
        uploaded_files = st.file_uploader(
            "Choose multiple image files",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Upload multiple images to process"
        )
        
        if uploaded_files:
            st.info(f"Selected {len(uploaded_files)} files for processing")
            
            # Model selection
            available_models = config_manager.get_available_models()
            selected_model = st.selectbox("Select Model", available_models)
            
            # Output directory
            output_dir = st.text_input("Output Directory", "assets/output/batch")
            
            # Process button
            if st.button("🚀 Process Batch", type="primary"):
                if not uploaded_files:
                    st.error("Please select files to process")
                    return
                
                # Validate input
                validation = BatchProcessorUI.validate_batch_input([f.name for f in uploaded_files])
                
                if not validation['valid']:
                    for error in validation['errors']:
                        st.error(error)
                    return
                
                if validation['warnings']:
                    for warning in validation['warnings']:
                        st.warning(warning)
                
                # Process batch
                with st.spinner("Processing batch..."):
                    try:
                        # Create temporary files
                        temp_dir = "temp_batch"
                        os.makedirs(temp_dir, exist_ok=True)
                        
                        temp_paths = []
                        for uploaded_file in uploaded_files:
                            temp_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getvalue())
                            temp_paths.append(temp_path)
                        
                        # Process batch
                        results = batch_processor.process_batch_parallel(
                            temp_paths, selected_model, output_dir, max_workers
                        )
                        
                        # Display results
                        st.success(f"Batch processing completed!")
                        
                        # Statistics
                        stats = results['stats']
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Images", stats['total_images'])
                        with col2:
                            st.metric("Processed", stats['processed_images'])
                        with col3:
                            st.metric("Failed", stats['failed_images'])
                        with col4:
                            st.metric("Avg Time", f"{stats['avg_time_per_image']:.2f}s")
                        
                        # Results table
                        st.subheader("Processing Results")
                        results_data = []
                        for result in results['results']:
                            results_data.append({
                                'File': os.path.basename(result['input_path']),
                                'Status': 'Success' if result['success'] else 'Failed',
                                'Processing Time': f"{result.get('processing_time', 0):.2f}s",
                                'Error': result.get('error', '')
                            })
                        
                        st.dataframe(results_data, use_container_width=True)
                        
                        # Cleanup temp files
                        for temp_path in temp_paths:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                        os.rmdir(temp_dir)
                        
                    except Exception as e:
                        st.error(f"Batch processing failed: {e}")
                        logger.error(f"Batch processing error: {e}")
    
    else:  # Select from directory
        st.info("Directory selection feature coming soon!")

def analytics_dashboard(db_manager, config_manager):
    """Analytics dashboard"""
    st.header("Analytics Dashboard")
    
    # Get database statistics
    stats = db_manager.get_database_stats()
    
    if not stats:
        st.info("No data available in the database yet.")
        return
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", stats['total_images'])
    with col2:
        st.metric("Total Processing Time", f"{stats['total_processing_time']:.1f}s")
    with col3:
        st.metric("Recent Images (7 days)", stats['recent_images'])
    with col4:
        st.metric("Database Size", f"{stats['database_size_mb']:.1f} MB")
    
    # Model distribution chart
    if stats['model_distribution']:
        st.subheader("Model Usage Distribution")
        
        fig = px.pie(
            values=list(stats['model_distribution'].values()),
            names=list(stats['model_distribution'].keys()),
            title="Images Processed by Model"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.subheader("Recent Activity")
    recent_records = db_manager.get_all_image_records(limit=10)
    
    if recent_records:
        activity_data = []
        for record in recent_records:
            activity_data.append({
                'Filename': record.filename,
                'Model': record.model_name,
                'Scale Factor': record.scale_factor,
                'Processing Time': f"{record.processing_time:.2f}s",
                'Created': record.created_at
            })
        
        st.dataframe(activity_data, use_container_width=True)
    else:
        st.info("No recent activity found.")

def settings_page(config_manager):
    """Settings page"""
    st.header("Settings")
    
    # Model configurations
    st.subheader("Model Configurations")
    
    for model_name in config_manager.get_available_models():
        with st.expander(f"Configure {model_name}"):
            model_config = config_manager.get_model_config(model_name)
            
            if model_config:
                st.text_input(f"{model_name} Model Path", value=model_config.model_path, key=f"path_{model_name}")
                st.number_input(f"{model_name} Scale Factor", value=model_config.scale_factor, key=f"scale_{model_name}")
                st.number_input(f"{model_name} Tile Size", value=model_config.tile_size, key=f"tile_{model_name}")
                st.number_input(f"{model_name} Tile Pad", value=model_config.tile_pad, key=f"pad_{model_name}")
    
    # Processing settings
    st.subheader("Processing Settings")
    
    processing_config = config_manager.processing_config
    st.text_input("Input Directory", value=processing_config.input_dir, key="input_dir")
    st.text_input("Output Directory", value=processing_config.output_dir, key="output_dir")
    st.number_input("Max Image Size", value=processing_config.max_image_size, key="max_size")
    st.number_input("Batch Size", value=processing_config.batch_size, key="batch_size")
    
    # UI settings
    st.subheader("UI Settings")
    
    ui_config = config_manager.ui_config
    st.text_input("Application Title", value=ui_config.title, key="title")
    st.selectbox("Theme", ["light", "dark"], index=0 if ui_config.theme == "light" else 1, key="theme")
    st.number_input("Max File Size (MB)", value=ui_config.max_file_size, key="max_file_size")
    st.checkbox("Show Metrics", value=ui_config.show_metrics, key="show_metrics")
    st.checkbox("Enable Batch Processing", value=ui_config.enable_batch_processing, key="enable_batch")
    
    # Save settings
    if st.button("💾 Save Settings", type="primary"):
        try:
            config_manager.save_config()
            st.success("Settings saved successfully!")
        except Exception as e:
            st.error(f"Failed to save settings: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Model preloading script for marker-pdf GPU optimization.
This script preloads all marker models at Docker startup to avoid cold start times.
"""

import os
import sys
import time
import torch
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Check and log GPU availability and status."""
    logger.info("=== GPU Status Check ===")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            logger.info(f"GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        
        # Set GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("GPU optimizations enabled")
        return True
    else:
        logger.warning("CUDA not available - falling back to CPU")
        return False

def preload_datalab_models():
    """Download all datalab/marker models to cache."""
    logger.info("=== Downloading Datalab Models ===")
    start_time = time.perf_counter()

    try:
        # Import marker modules to trigger model downloads
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        logger.info("Initializing PdfConverter to download models...")

        # Create a dummy converter instance to trigger model downloads
        converter = PdfConverter()

        download_time = time.perf_counter() - start_time
        logger.info(f"Datalab models downloaded in {download_time:.2f}s")

        return True

    except Exception as e:
        logger.error(f"Datalab model download failed: {e}")
        return False

def preload_marker_models():
    """Preload all marker models with GPU optimizations."""
    logger.info("=== Starting Model Preloading ===")
    start_time = time.perf_counter()

    try:
        # First download datalab models
        if not preload_datalab_models():
            logger.warning("Datalab model download failed, continuing with torch models only")

        # Import marker modules
        from marker.models import create_model_dict
        from marker.util import download_font

        logger.info("Creating model dictionary...")
        model_start = time.perf_counter()

        # Create models with GPU optimizations
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                models = create_model_dict()
        else:
            models = create_model_dict()

        model_load_time = time.perf_counter() - model_start
        logger.info(f"Models loaded in {model_load_time:.2f}s")
        
        # Apply GPU optimizations to models
        if torch.cuda.is_available():
            gpu_opt_start = time.perf_counter()
            
            for name, model in models.items():
                if hasattr(model, 'model') and model.model is not None:
                    try:
                        # Set to eval mode
                        model.model.eval()
                        
                        # Enable mixed precision
                        if hasattr(model.model, 'half'):
                            model.model.half()
                        
                        # Apply channels_last for conv models
                        model_type = str(type(model.model)).lower()
                        if 'conv' in model_type or hasattr(model.model, 'conv1'):
                            model.model = model.model.to(memory_format=torch.channels_last)
                            logger.info(f"  {name}: Applied channels_last format")
                        
                        # Compile model if available (PyTorch 2.0+)
                        if hasattr(torch, 'compile') and torch.__version__ >= '2.0':
                            try:
                                model.model = torch.compile(model.model, mode='reduce-overhead')
                                logger.info(f"  {name}: Compiled with torch.compile")
                            except Exception as e:
                                logger.warning(f"  {name}: Compilation failed: {e}")
                        
                        logger.info(f"  {name}: GPU optimizations applied")
                        
                    except Exception as e:
                        logger.warning(f"  {name}: Optimization failed: {e}")
            
            gpu_opt_time = time.perf_counter() - gpu_opt_start
            logger.info(f"GPU optimizations applied in {gpu_opt_time:.2f}s")
            
            # Warm up GPU with dummy inference
            logger.info("Warming up GPU...")
            torch.cuda.synchronize()
            dummy_tensor = torch.randn(1, 3, 224, 224, device='cuda', dtype=torch.float16)
            with torch.cuda.amp.autocast():
                _ = torch.nn.functional.conv2d(dummy_tensor, torch.randn(64, 3, 7, 7, device='cuda', dtype=torch.float16))
            torch.cuda.synchronize()
            logger.info("GPU warmed up")
            
        # Verify fonts are downloaded
        font_start = time.perf_counter()
        download_font()
        font_time = time.perf_counter() - font_start
        logger.info(f"Font verification completed in {font_time:.2f}s")
        
        total_time = time.perf_counter() - start_time
        logger.info(f"=== Model Preloading Complete in {total_time:.2f}s ===")
        
        # Store preloaded models globally for reuse
        global _preloaded_models
        _preloaded_models = models
        
        # Also store in module-level for get_preloaded_models function
        get_preloaded_models._preloaded_models = models
        
        return models
        
    except Exception as e:
        logger.error(f"Model preloading failed: {e}")
        raise

def get_preloaded_models():
    """Get preloaded models if available."""
    global _preloaded_models
    # Try module attribute first, then global variable
    return getattr(get_preloaded_models, '_preloaded_models', _preloaded_models)

def validate_preload():
    """Validate that models are properly preloaded."""
    logger.info("=== Validating Preloaded Models ===")
    
    models = get_preloaded_models()
    if models is None:
        logger.error("No preloaded models found!")
        return False
    
    logger.info(f"Found {len(models)} preloaded models:")
    for name, model in models.items():
        model_info = f"  {name}: "
        if hasattr(model, 'model') and model.model is not None:
            device = next(model.model.parameters()).device if hasattr(model.model, 'parameters') else 'unknown'
            dtype = next(model.model.parameters()).dtype if hasattr(model.model, 'parameters') else 'unknown'
            model_info += f"device={device}, dtype={dtype}"
        else:
            model_info += "no model attribute"
        logger.info(model_info)
    
    logger.info("=== Validation Complete ===")
    return True

# Global variable to store preloaded models
_preloaded_models = None

if __name__ == "__main__":
    """Main execution when run as script."""
    # Check if GPU is enabled via environment variable
    gpu_enabled = os.getenv("GPU_ENABLED", "false").lower() == "true"

    if not gpu_enabled:
        logger.info("GPU_ENABLED=false, skipping model preloading")
        logger.info("✅ Skipped model preloading (GPU disabled)")
        sys.exit(0)

    logger.info("Starting marker model preloading...")

    # Check GPU
    gpu_available = check_gpu_availability()
    
    # Preload models
    try:
        models = preload_marker_models()
        
        # Validate preload
        if validate_preload():
            logger.info("✅ Model preloading successful!")
            sys.exit(0)
        else:
            logger.error("❌ Model preloading validation failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Model preloading failed: {e}")
        sys.exit(1)
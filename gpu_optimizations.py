"""GPU optimizations for marker-pdf processing"""

import os
import torch
import time
from functools import wraps

# Set CPU threading to avoid contention
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# Enable CUDA optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Optimize conv operations
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
    torch.backends.cudnn.allow_tf32 = True
    
    # Set default dtype for better performance
    torch.set_float32_matmul_precision('medium')

def timing_decorator(name):
    """Decorator to time function execution"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            result = func(*args, **kwargs)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            print(f"[TIMING] {name}: {end-start:.3f}s")
            return result
        return wrapper
    return decorator

class GPUOptimizedMarker:
    """Wrapper for GPU-optimized marker processing"""
    
    _models = None
    _load_time = None
    
    @classmethod
    def load_models_once(cls):
        """Load models once at startup with optimizations"""
        if cls._models is not None:
            return cls._models
            
        print("Loading models with GPU optimizations...")
        start = time.perf_counter()
        
        from marker.models import create_model_dict
        
        # Load models with mixed precision
        with torch.cuda.amp.autocast(dtype=torch.float16):
            models = create_model_dict()
        
        # Apply optimizations to each model
        for name, model in models.items():
            if hasattr(model, 'model'):
                # Set to eval mode
                model.model.eval()
                
                # Move to GPU with channels_last if it's a conv model
                if hasattr(model.model, 'conv1') or 'conv' in str(type(model.model)).lower():
                    model.model = model.model.to(memory_format=torch.channels_last)
                    print(f"  {name}: Applied channels_last format")
                
                # Compile model if available (PyTorch 2.0+)
                if hasattr(torch, 'compile'):
                    try:
                        model.model = torch.compile(model.model, mode='reduce-overhead')
                        print(f"  {name}: Compiled with torch.compile")
                    except:
                        pass
        
        cls._models = models
        cls._load_time = time.perf_counter() - start
        print(f"Models loaded in {cls._load_time:.3f}s")
        return models
    
    @classmethod
    @timing_decorator("process_with_gpu")
    def process_with_timing(cls, file_path, output_dir):
        """Process file with timing and GPU optimizations"""
        models = cls.load_models_once()
        
        # Run with mixed precision
        with torch.cuda.amp.autocast(dtype=torch.float16):
            with torch.inference_mode():
                # This is where marker processing would happen
                # The actual processing is done by marker_single command
                pass
        
        return True

# Initialize at module import
if __name__ != "__main__":
    # Pre-load models when module is imported
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
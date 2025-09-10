#!/usr/bin/env python3
"""Test script to verify GPU usage in marker"""

import torch
import os
import sys

print("="*50)
print("GPU Configuration Test")
print("="*50)

# Check environment variables
print("\nEnvironment Variables:")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
print(f"TORCH_DEVICE: {os.environ.get('TORCH_DEVICE', 'Not set')}")
print(f"MARKER_DEVICE: {os.environ.get('MARKER_DEVICE', 'Not set')}")
print(f"SURYA_DEVICE: {os.environ.get('SURYA_DEVICE', 'Not set')}")

# Check PyTorch CUDA
print("\nPyTorch CUDA Status:")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")

# Try to import marker and check its device settings
try:
    from marker.models import create_model_dict
    print("\nTrying to load Marker models...")
    models = create_model_dict()
    print("Models loaded successfully!")
    
    # Check device for each model
    for name, model in models.items():
        if hasattr(model, 'device'):
            print(f"  {name}: device = {model.device}")
        if hasattr(model, 'model'):
            if hasattr(model.model, 'device'):
                print(f"  {name}.model: device = {model.model.device}")
            # Check if model is on CUDA
            for param_name, param in model.model.named_parameters():
                print(f"  {name} first param on: {param.device}")
                break
except ImportError:
    # Try alternative import
    try:
        from marker.converters.pdf import PdfConverter
        print("\nChecking PdfConverter GPU settings...")
        # This is just to check imports work
        print("PdfConverter import successful")
    except Exception as e2:
        print(f"Alternative import also failed: {e2}")
except Exception as e:
    print(f"\nError loading models: {e}")

print("\n" + "="*50)
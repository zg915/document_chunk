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
    from marker.models import load_all_models
    print("\nTrying to load Marker models...")
    models = load_all_models()
    print("Models loaded successfully!")
    
    # Check device for each model
    for name, model in models.items():
        if hasattr(model, 'device'):
            print(f"  {name}: {model.device}")
        elif hasattr(model, 'model') and hasattr(model.model, 'device'):
            print(f"  {name}: {model.model.device}")
except Exception as e:
    print(f"\nError loading models: {e}")

print("\n" + "="*50)
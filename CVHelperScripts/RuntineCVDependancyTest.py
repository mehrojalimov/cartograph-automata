#!/usr/bin/env python3
"""
Dependency Test Script for TensorFlow Model
This script tests if all required dependencies are properly installed
without actually loading or running the model.

Author: Rokawoo
"""

import sys
import importlib

def test_dependencies():
    """Test if all required dependencies are installed"""
    required_packages = [
        'tensorflow',
        'numpy',
        'opencv-python',
        'pillow'
    ]
    
    package_to_module = {
        'tensorflow': 'tensorflow',
        'numpy': 'numpy',
        'opencv-python': 'cv2',
        'pillow': 'PIL'
    }
    
    missing_packages = []
    version_info = {}
    
    print("=" * 50)
    print("DEPENDENCY TEST FOR TENSORFLOW MODEL")
    print("=" * 50)
    print("\nChecking required packages:")
    
    for package in required_packages:
        module_name = package_to_module[package]
        
        try:
            module = importlib.import_module(module_name)
            
            # Try to get version information
            try:
                if hasattr(module, '__version__'):
                    version = module.__version__
                elif hasattr(module, 'version'):
                    version = module.version
                elif module_name == 'PIL':
                    version = module.PILLOW_VERSION if hasattr(module, 'PILLOW_VERSION') else 'Unknown'
                else:
                    version = 'Installed (version unknown)'
                
                version_info[package] = version
                print(f"✓ {package:<15} - {version}")
                
            except:
                print(f"✓ {package:<15} - Installed (version unknown)")
                
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package:<15} - NOT INSTALLED")
    
    # Test TensorFlow GPU support
    if 'tensorflow' not in missing_packages:
        try:
            import tensorflow as tf
            print("\nTesting TensorFlow GPU support:")
            
            # Check for GPU devices
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"✓ TensorFlow GPU support - {len(gpus)} GPU(s) detected")
                for i, gpu in enumerate(gpus):
                    print(f"  - GPU {i+1}: {gpu.name}")
            else:
                print("ℹ TensorFlow is using CPU only (no GPUs detected)")
                
            # Additional TensorFlow details
            print(f"\nTensorFlow Details:")
            print(f"- Version: {tf.__version__}")
            try:
                print(f"- Keras Version: {tf.keras.__version__}")
                print(f"- Built with CUDA: {'Yes' if tf.test.is_built_with_cuda() else 'No'}")
            except:
                pass
            
        except Exception as e:
            print(f"ℹ Could not test TensorFlow GPU support: {e}")
    
    # Summary and recommendations
    print("\n" + "=" * 50)
    if missing_packages:
        print("❌ SOME DEPENDENCIES ARE MISSING")
        print("=" * 50)
        print("\nPlease install the missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\nOr use conda:")
        print(f"conda install {' '.join(missing_packages)}")
        return False
    else:
        print("✅ ALL DEPENDENCIES INSTALLED SUCCESSFULLY")
        print("=" * 50)
        print("\nYour system has all the required dependencies to run the TensorFlow model.")
        
        # Optional: Test import of specific TensorFlow components used by the model
        try:
            from tensorflow.keras.models import load_model
            import numpy as np
            import cv2
            import tensorflow as tf
            
            print("\nSuccessfully imported all critical components:")
            print("- tensorflow.keras.models (load_model)")
            print("- numpy")
            print("- cv2 (OpenCV)")
            print("- tensorflow.image (resize)")
        except Exception as e:
            print(f"\nWarning: Could not import specific components: {e}")
        
        return True

if __name__ == "__main__":
    test_dependencies()
#!/usr/bin/env python3
"""
Test script to verify project setup and dependencies
"""

import sys
import os

def test_imports():
    """Test if basic dependencies can be imported"""
    print("=" * 60)
    print("Testing Project Dependencies")
    print("=" * 60)
    
    results = {}
    
    # Test basic packages
    try:
        import numpy as np
        results["numpy"] = f"✅ {np.__version__}"
    except ImportError as e:
        results["numpy"] = f"❌ {str(e)}"
    
    try:
        import cv2
        results["opencv"] = f"✅ {cv2.__version__}"
    except ImportError as e:
        results["opencv"] = f"❌ {str(e)}"
    
    try:
        import yaml
        results["yaml"] = "✅ Installed"
    except ImportError as e:
        results["yaml"] = f"❌ {str(e)}"
    
    try:
        import sklearn
        results["scikit-learn"] = f"✅ {sklearn.__version__}"
    except ImportError as e:
        results["scikit-learn"] = f"❌ {str(e)}"
    
    try:
        import matplotlib
        results["matplotlib"] = f"✅ {matplotlib.__version__}"
    except ImportError as e:
        results["matplotlib"] = f"❌ {str(e)}"
    
    # Test PyTorch
    try:
        import torch
        results["pytorch"] = f"✅ {torch.__version__}"
        results["cuda_available"] = f"✅ {torch.cuda.is_available()}" if torch.cuda.is_available() else "❌ CPU only"
    except ImportError as e:
        results["pytorch"] = f"❌ {str(e)}"
        results["cuda_available"] = "N/A"
    
    # Test TensorFlow
    try:
        import tensorflow as tf
        version = getattr(tf, '__version__', 'Installed (version unknown)')
        results["tensorflow"] = f"✅ {version}"
    except ImportError as e:
        results["tensorflow"] = f"❌ {str(e)}"
    
    # Test MTCNN
    try:
        from mtcnn import MTCNN
        results["mtcnn"] = "✅ Installed"
    except (ImportError, AttributeError) as e:
        results["mtcnn"] = f"❌ {str(e)}"
    
    # Test ONNX
    try:
        import onnx
        version = getattr(onnx, '__version__', 'unknown')
        results["onnx"] = f"✅ {version}"
    except ImportError as e:
        results["onnx"] = f"❌ {str(e)}"
    
    # Print results
    print("\nDependency Status:")
    print("-" * 60)
    for package, status in results.items():
        print(f"{package:20s}: {status}")
    
    return results

def test_project_structure():
    """Test if project structure is correct"""
    print("\n" + "=" * 60)
    print("Testing Project Structure")
    print("=" * 60)
    
    required_dirs = [
        "pytorch",
        "pytorch/models",
        "tensorflow",
        "tensorflow/model",
        "evaluation",
        "data/processed/train/real",
        "data/processed/train/fake",
        "data/processed/val/real",
        "data/processed/val/fake",
        "onnx"
    ]
    
    required_files = [
        "pytorch/train.py",
        "pytorch/dataset.py",
        "pytorch/config.yaml",
        "pytorch/models/xception.py",
        "tensorflow/webcam_detector.py",
        "tensorflow/preprocess.py",
        "tensorflow/temporal_logic.py",
        "evaluation/evaluate_model.py",
        "requirements.txt"
    ]
    
    print("\nChecking directories:")
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - MISSING")
    
    print("\nChecking files:")
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")

def test_data_availability():
    """Check if training data is available"""
    print("\n" + "=" * 60)
    print("Checking Data Availability")
    print("=" * 60)
    
    data_dirs = {
        "Train Real": "data/processed/train/real",
        "Train Fake": "data/processed/train/fake",
        "Val Real": "data/processed/val/real",
        "Val Fake": "data/processed/val/fake"
    }
    
    for name, path in data_dirs.items():
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if files:
                print(f"✅ {name}: {len(files)} images found")
            else:
                print(f"⚠️  {name}: Directory exists but no images found")
        else:
            print(f"❌ {name}: Directory not found")

def main():
    print(f"\nPython Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}\n")
    
    # Run tests
    results = test_imports()
    test_project_structure()
    test_data_availability()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    critical = ["pytorch", "tensorflow"]
    critical_status = [results.get(pkg, "❌ Not tested") for pkg in critical]
    
    if all("✅" in status for status in critical_status):
        print("✅ All critical dependencies are installed!")
        print("\nYou can now:")
        print("  1. Add training data to data/processed/train/ and data/processed/val/")
        print("  2. Run: cd pytorch && python train.py")
    else:
        print("⚠️  Some critical dependencies are missing:")
        for pkg in critical:
            print(f"  - {pkg}: {results.get(pkg, 'Not tested')}")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        print("\nNote: PyTorch and TensorFlow require Python 3.11 or 3.12")
        print("Current Python version may not be compatible.")

if __name__ == "__main__":
    main()

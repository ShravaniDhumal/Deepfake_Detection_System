#!/usr/bin/env python3
"""
Project Runner - Tests what can be run and shows what's needed
"""
import os
import sys

def check_dependencies():
    """Check which dependencies are available"""
    print("=" * 80)
    print("DEPENDENCY CHECK")
    print("=" * 80)
    
    deps = {
        'numpy': 'numpy',
        'opencv': 'cv2',
        'yaml': 'yaml',
        'sklearn': 'sklearn',
        'matplotlib': 'matplotlib',
        'pytorch': 'torch',
        'tensorflow': 'tensorflow',
        'pillow': 'PIL',
        'onnx': 'onnx'
    }
    
    available = []
    missing = []
    
    for name, module in deps.items():
        try:
            __import__(module)
            available.append(name)
            print(f"✅ {name:15s} - Available")
        except (ImportError, AttributeError) as e:
            missing.append(name)
            print(f"❌ {name:15s} - Missing ({type(e).__name__})")
    
    print(f"\nAvailable: {len(available)}/{len(deps)}")
    return available, missing

def check_project_structure():
    """Check if project structure is correct"""
    print("\n" + "=" * 80)
    print("PROJECT STRUCTURE CHECK")
    print("=" * 80)
    
    required_files = [
        'pytorch/train_improved.py',
        'pytorch/dataset_improved.py',
        'pytorch/models/xception.py',
        'pytorch/config.yaml',
        'tensorflow/webcam_detector_improved.py',
        'tensorflow/preprocess.py',
        'tensorflow/temporal_logic.py',
        'evaluation/evaluate_model.py',
        'requirements.txt',
        'README.md'
    ]
    
    for filepath in required_files:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            status = "✅" if size > 0 else "⚠️  (empty)"
            print(f"{status} {filepath} ({size} bytes)")
        else:
            print(f"❌ {filepath} - Missing")

def check_data():
    """Check for training data"""
    print("\n" + "=" * 80)
    print("DATA AVAILABILITY CHECK")
    print("=" * 80)
    
    data_dirs = {
        'Train Real': 'data/processed/train/real',
        'Train Fake': 'data/processed/train/fake',
        'Val Real': 'data/processed/val/real',
        'Val Fake': 'data/processed/val/fake'
    }
    
    total_images = 0
    for name, path in data_dirs.items():
        if os.path.exists(path):
            files = [f for f in os.listdir(path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            count = len(files)
            total_images += count
            if count > 0:
                print(f"✅ {name:15s} - {count:4d} images")
            else:
                print(f"⚠️  {name:15s} - No images (directory exists)")
        else:
            print(f"❌ {name:15s} - Directory not found")
    
    return total_images

def test_imports():
    """Test if improved modules can be imported"""
    print("\n" + "=" * 80)
    print("MODULE IMPORT TEST")
    print("=" * 80)
    
    tests = [
        ('Dataset (improved)', 'pytorch/dataset_improved.py'),
        ('Training (improved)', 'pytorch/train_improved.py'),
        ('Webcam Detector (improved)', 'tensorflow/webcam_detector_improved.py'),
        ('Temporal Logic', 'tensorflow/temporal_logic.py'),
        ('Preprocessing', 'tensorflow/preprocess.py'),
    ]
    
    for name, filepath in tests:
        if os.path.exists(filepath):
            try:
                # Try to parse the file
                with open(filepath, 'r') as f:
                    code = f.read()
                    compile(code, filepath, 'exec')
                print(f"✅ {name:30s} - Syntax OK")
            except SyntaxError as e:
                print(f"❌ {name:30s} - Syntax error: {e}")
            except Exception as e:
                print(f"⚠️  {name:30s} - {e}")
        else:
            print(f"❌ {name:30s} - File not found")

def show_what_can_run():
    """Show what can be run with current setup"""
    print("\n" + "=" * 80)
    print("WHAT CAN BE RUN")
    print("=" * 80)
    
    available, missing = check_dependencies()
    
    print("\n📊 Analysis & Testing:")
    print("  ✅ analyze_project.py - Can run (no special deps)")
    print("  ✅ test_project.py - Can run (basic deps)")
    
    if 'numpy' in available and 'opencv' in available:
        print("\n🎥 Demo:")
        print("  ✅ demo.py - Can run (has numpy, opencv)")
    else:
        print("\n🎥 Demo:")
        print("  ❌ demo.py - Needs: numpy, opencv")
    
    if 'pytorch' in available:
        print("\n🏋️  Training:")
        print("  ✅ train_improved.py - Can run (has PyTorch)")
        print("  ⚠️  Needs: Training data in data/processed/")
    else:
        print("\n🏋️  Training:")
        print("  ❌ train_improved.py - Needs: PyTorch (Python 3.11/3.12)")
    
    if 'tensorflow' in available and 'opencv' in available:
        print("\n📹 Webcam Detection:")
        print("  ✅ webcam_detector_improved.py - Can run (has TensorFlow, OpenCV)")
        print("  ⚠️  Needs: Trained model in tensorflow/model/saved_model/")
    else:
        print("\n📹 Webcam Detection:")
        print("  ❌ webcam_detector_improved.py - Needs: TensorFlow, OpenCV")
    
    if 'sklearn' in available:
        print("\n📈 Evaluation:")
        print("  ✅ evaluate_model.py - Can run (has sklearn)")
        print("  ⚠️  Needs: y_true.npy and y_pred.npy files")
    else:
        print("\n📈 Evaluation:")
        print("  ❌ evaluate_model.py - Needs: scikit-learn")

def main():
    print("\n" + "=" * 80)
    print("DEEPFAKE DETECTION SYSTEM - PROJECT RUNNER")
    print("=" * 80)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Working Directory: {os.getcwd()}\n")
    
    # Run checks
    available, missing = check_dependencies()
    check_project_structure()
    total_images = check_data()
    test_imports()
    show_what_can_run()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\n✅ Dependencies Available: {len(available)}")
    print(f"❌ Dependencies Missing: {len(missing)}")
    
    if missing:
        print(f"\nMissing: {', '.join(missing)}")
        if 'pytorch' in missing or 'tensorflow' in missing:
            print("\n⚠️  Note: PyTorch and TensorFlow require Python 3.11 or 3.12")
            print("   Current Python version may not be compatible.")
    
    print(f"\n📊 Training Data: {total_images} images found")
    if total_images == 0:
        print("   ⚠️  No training data found. Add images to data/processed/ directories.")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    
    if 'pytorch' not in available:
        print("\n1. Install Python 3.11 or 3.12")
        print("2. Recreate virtual environment:")
        print("   python3.12 -m venv venv")
        print("   source venv/bin/activate")
        print("   pip install -r requirements.txt")
    
    if total_images == 0:
        print("\n3. Add training data:")
        print("   - Place real images in: data/processed/train/real/")
        print("   - Place fake images in: data/processed/train/fake/")
        print("   - Add validation data to: data/processed/val/")
    
    if 'pytorch' in available and total_images > 0:
        print("\n4. Run training:")
        print("   cd pytorch")
        print("   python train_improved.py")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

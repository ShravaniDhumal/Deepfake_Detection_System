# Setup Instructions

## Python Version Compatibility Issue

**Important**: This project requires PyTorch and TensorFlow, which currently do not support Python 3.13. 

### Solution: Use Python 3.11 or 3.12

1. **Install Python 3.12** (recommended):
   ```bash
   brew install python@3.12
   ```

2. **Create a new virtual environment with Python 3.12**:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Alternative: Use Python 3.11**:
   ```bash
   brew install python@3.11
   python3.11 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Current Status

- ✅ Most dependencies installed (numpy, opencv, scikit-learn, matplotlib, etc.)
- ❌ PyTorch - Not available for Python 3.13 (requires Python ≤ 3.12)
- ❌ TensorFlow - Not available for Python 3.13 (requires Python ≤ 3.12)
- ❌ MTCNN - Requires TensorFlow (will work once TensorFlow is installed)

## Running the Project

Once you have Python 3.11 or 3.12 installed:

1. **For PyTorch training**:
   ```bash
   cd pytorch
   python train.py
   ```
   Note: Requires data in `../data/processed/train` and `../data/processed/val`

2. **For TensorFlow webcam detection**:
   ```bash
   cd tensorflow
   python webcam_detector.py
   ```
   Note: Requires a trained model in `model/saved_model/`

3. **For evaluation**:
   ```bash
   cd evaluation
   python evaluate_model.py
   ```
   Note: Requires `y_true.npy` and `y_pred.npy` files

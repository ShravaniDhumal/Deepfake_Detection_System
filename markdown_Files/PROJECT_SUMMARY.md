# Deepfake Detection System - Project Summary

## 📋 Overview

A **comprehensive deepfake detection system** that uses deep learning to identify manipulated media content. The system supports both **training custom models** and **real-time detection via webcam**.

**Purpose**: Detect deepfake images/videos using AI/ML techniques to identify manipulated media content.

---

## 🎯 Core Features

1. **Model Training** (PyTorch)
   - Train custom deepfake detection models using transfer learning
   - MobileNetV2-based architecture (pretrained on ImageNet)
   - Binary classification: Real vs Fake

2. **Real-time Detection** (TensorFlow)
   - Live webcam-based deepfake detection
   - Automatic face detection using MTCNN
   - Temporal smoothing for stable predictions

3. **Model Export**
   - Export trained models to ONNX format for deployment
   - Cross-platform compatibility

4. **Evaluation Tools**
   - Performance metrics calculation
   - Confusion matrix generation
   - Classification reports

---

## 📁 Project Structure

```
Deepfake_Detection_System/
│
├── 📄 Core Files
│   ├── README.md              # Main documentation
│   ├── requirements.txt       # Python dependencies
│   ├── run_project.py         # Project runner/checker
│   └── demo.py                # Demo script
│
├── 📊 Data Directory
│   ├── data/
│   │   ├── raw/               # Raw dataset (real/fake images)
│   │   │   ├── real/
│   │   │   └── fake/
│   │   └── processed/         # Processed dataset for training
│   │       ├── train/
│   │       │   ├── real/
│   │       │   └── fake/
│   │       └── val/
│   │           ├── real/
│   │           └── fake/
│
├── 🧠 PyTorch Training Pipeline
│   ├── pytorch/
│   │   ├── models/
│   │   │   └── xception.py    # Model architecture (MobileNetV2)
│   │   ├── train.py           # Original training script
│   │   ├── train_improved.py  # Improved training (with validation, early stopping)
│   │   ├── dataset.py         # Original dataset loader
│   │   ├── dataset_improved.py # Improved dataset (error handling, augmentation)
│   │   ├── config.yaml        # Training configuration
│   │   ├── export_to_onnx.py  # ONNX export script
│   │   ├── utils.py           # Utility functions
│   │   └── visualize_training.py # Training visualization
│
├── 🎥 TensorFlow Inference Pipeline
│   ├── tensorflow/
│   │   ├── model/
│   │   │   └── saved_model/   # Saved TensorFlow model
│   │   ├── webcam_detector.py # Original webcam detector
│   │   ├── webcam_detector_improved.py # Improved detector
│   │   ├── inference.py      # Inference utilities
│   │   ├── preprocess.py     # Image preprocessing
│   │   └── temporal_logic.py # Temporal smoothing for predictions
│
├── 📈 Evaluation
│   ├── evaluation/
│   │   ├── evaluate_model.py  # Model evaluation script
│   │   ├── generate_predictions.py # Generate predictions
│   │   └── metrics.py         # Metrics calculation
│
├── 📦 Model Files
│   ├── onnx/
│   │   └── xception_deepfake.onnx # Exported ONNX model
│
└── 📚 Documentation
    ├── docs/
    │   ├── proposal.pdf
    │   ├── architecture_diagram.png
    │   ├── presentation.pptx
    │   └── results/           # Training results
    │       ├── confusion_matrix.png
    │       ├── accuracy_plot.png
    │       └── performance_metrics.txt
    ├── SETUP.md               # Setup instructions
    ├── ANALYSIS_SUMMARY.md    # Project analysis
    ├── IMPROVEMENTS.md        # Improvement recommendations
    └── FIX_SUMMARY.md         # Bug fixes summary
```

---

## 🔧 Technical Stack

### Deep Learning Frameworks
- **PyTorch** - Model training
- **TensorFlow** - Real-time inference
- **ONNX** - Model export/deployment

### Key Libraries
- **torch & torchvision** - PyTorch for training
- **tensorflow** - TensorFlow for inference
- **opencv-python** - Image processing and webcam access
- **mtcnn** - Face detection
- **numpy, pillow** - Image handling
- **scikit-learn** - Evaluation metrics
- **matplotlib** - Visualization
- **pyyaml** - Configuration files

### System Requirements
- **Python**: 3.11 or 3.12 (PyTorch/TensorFlow requirement)
- **GPU**: Optional (CUDA-capable for faster training)
- **Webcam**: Required for real-time detection

---

## 🧬 Model Architecture

### Base Model
- **Architecture**: MobileNetV2 (pretrained on ImageNet)
- **Input Size**: 224x224 RGB images
- **Output**: Binary classification (Real/Fake)
- **Training Method**: Transfer learning with frozen backbone

### Model Details
```python
# Model structure
MobileNetV2 (pretrained)
├── Features (frozen) - Feature extraction layers
└── Classifier (trainable) - Custom binary classifier
    └── Linear layer: in_features → 2 classes
```

### Training Configuration
- **Epochs**: 5 (configurable)
- **Batch Size**: 16
- **Learning Rate**: 0.0001
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Features**: Early stopping, learning rate scheduling, gradient clipping

---

## 🚀 Key Components

### 1. Training Pipeline (`pytorch/train_improved.py`)

**Features:**
- ✅ Validation loop with metrics tracking
- ✅ Early stopping to prevent overfitting
- ✅ Learning rate scheduling (ReduceLROnPlateau)
- ✅ Gradient clipping for stability
- ✅ Best model saving
- ✅ Comprehensive logging
- ✅ Error handling for empty datasets
- ✅ Division by zero protection

**Workflow:**
1. Load configuration from `config.yaml`
2. Load and preprocess training/validation data
3. Initialize MobileNetV2 model
4. Train for specified epochs
5. Validate after each epoch
6. Save best model based on validation loss
7. Export final model

### 2. Dataset Handling (`pytorch/dataset_improved.py`)

**Features:**
- ✅ Error handling for corrupted images
- ✅ File filtering (removes .gitkeep, .DS_Store)
- ✅ Data augmentation support
- ✅ Logging for debugging
- ✅ Fallback handling

**Data Augmentation:**
- Random horizontal flip
- Random crop/resize
- Color jitter
- Normalization

### 3. Real-time Detection (`tensorflow/webcam_detector_improved.py`)

**Features:**
- ✅ Automatic face detection (MTCNN)
- ✅ Real-time classification
- ✅ Temporal smoothing (30-frame window)
- ✅ Visual feedback with bounding boxes
- ✅ FPS limiting and display
- ✅ Confidence score display
- ✅ Proper exception handling

**How It Works:**
1. Capture frame from webcam
2. Detect faces using MTCNN
3. Preprocess detected faces (resize, normalize)
4. Classify using trained model
5. Apply temporal smoothing (average over 30 frames)
6. Display results with labels and confidence

### 4. Temporal Logic (`tensorflow/temporal_logic.py`)

**Purpose**: Smooth predictions across video frames for stability

**Parameters:**
- **Window Size**: 30 frames (default)
- **Threshold**: 0.6 (60% fake predictions = DEEPFAKE)
- **Status States**: "Analyzing" → "REAL" or "DEEPFAKE"

### 5. Evaluation Tools (`evaluation/`)

**Metrics Provided:**
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report

---

## 📊 Data Flow

### Training Flow
```
Raw Images → Data Processing → Train/Val Split
    ↓
DataLoader → Augmentation → Model Training
    ↓
Validation → Metrics → Best Model Save
```

### Inference Flow
```
Webcam → Frame Capture → Face Detection (MTCNN)
    ↓
Face Preprocessing → Model Inference → Prediction
    ↓
Temporal Smoothing → Display Result
```

---

## 🛠️ Usage Workflow

### 1. Setup
```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data
```
data/processed/
├── train/
│   ├── real/    # Real images
│   └── fake/    # Fake images
└── val/
    ├── real/    # Real images
    └── fake/    # Fake images
```

### 3. Configure Training
Edit `pytorch/config.yaml`:
```yaml
data:
  train_dir: ../data/processed/train
  val_dir: ../data/processed/val
  batch_size: 16

training:
  epochs: 5
  learning_rate: 0.0001
```

### 4. Train Model
```bash
cd pytorch
python train_improved.py
```

### 5. Real-time Detection
```bash
cd tensorflow
python webcam_detector_improved.py
```

### 6. Export Model
```bash
cd pytorch
python export_to_onnx.py
```

---

## 🐛 Recent Fixes

### Bug Fixes Applied:
1. ✅ **Division by Zero in Validation** - Fixed empty validation dataset handling
2. ✅ **Division by Zero in Training** - Added safety checks for empty loaders
3. ✅ **Model Loading Error** - Fixed missing model file handling
4. ✅ **Path Handling** - Fixed directory creation for model saving

### Improvements Made:
- Enhanced error handling throughout
- Better logging and debugging
- Validation metrics tracking
- Early stopping implementation
- Learning rate scheduling
- Gradient clipping

---

## 📈 Project Status

### ✅ Completed
- [x] Training pipeline with validation
- [x] Real-time webcam detection
- [x] Model export to ONNX
- [x] Evaluation tools
- [x] Improved versions of core scripts
- [x] Error handling improvements
- [x] Documentation

### ⚠️ Known Issues
- Model file named "xception.py" but uses MobileNetV2 (naming mismatch)
- Some utility files are empty (inference.py, metrics.py)
- Original training script lacks validation loop

### 🔄 Recommended Next Steps
1. Replace original files with improved versions
2. Fix model naming (rename xception.py → mobilenet.py)
3. Implement empty utility files
4. Add unit tests
5. Add type hints for better code quality

---

## 📝 Key Files Explained

| File | Purpose |
|------|---------|
| `train_improved.py` | Main training script with validation, early stopping |
| `dataset_improved.py` | Dataset loader with error handling and augmentation |
| `webcam_detector_improved.py` | Real-time detection with temporal smoothing |
| `config.yaml` | Training configuration (paths, hyperparameters) |
| `xception.py` | Model architecture (MobileNetV2-based) |
| `temporal_logic.py` | Smooths predictions across video frames |
| `run_project.py` | Checks dependencies and project structure |

---

## 🎓 Learning Resources

### Concepts Used:
- **Transfer Learning**: Using pretrained MobileNetV2
- **Binary Classification**: Real vs Fake detection
- **Temporal Smoothing**: Averaging predictions over time
- **Face Detection**: MTCNN for face localization
- **Data Augmentation**: Improving model generalization
- **Early Stopping**: Preventing overfitting
- **Model Export**: ONNX for deployment

### Datasets:
- FaceForensics++
- DFDC (Deepfake Detection Challenge)
- Celeb-DF

---

## 🔒 Important Notes

⚠️ **Data Privacy**: Ensure you have permission to use images
⚠️ **Python Version**: Requires Python 3.11 or 3.12 (not 3.13+)
⚠️ **GPU**: Optional but recommended for faster training
⚠️ **Data**: Need training images in proper directory structure

---

## 📞 Support

For issues or questions:
- Check `README.md` for general information
- Check `SETUP.md` for installation help
- Check `ANALYSIS_SUMMARY.md` for project analysis
- Review error logs in `training.log`

---

**Last Updated**: Based on current project state
**Status**: Functional with recent bug fixes applied
**Version**: Improved version with enhanced error handling

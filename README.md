# Deepfake Detection System

A comprehensive deepfake detection system that uses deep learning to identify manipulated media content. The system supports both training custom models and real-time detection via webcam.

## 🎯 Features

- **PyTorch-based Training**: Train custom deepfake detection models using transfer learning
- **Real-time Detection**: Live webcam-based deepfake detection with face detection
- **Temporal Smoothing**: Temporal logic for stable predictions across video frames
- **Model Export**: Export trained models to ONNX format for deployment
- **Comprehensive Evaluation**: Metrics and evaluation tools for model performance

## 📁 Project Structure

```
Deepfake-Detection-System/
│
├── README.md                 # This file
├── SETUP.md                  # Detailed setup instructions
├── requirements.txt          # Python dependencies
│
├── data/                     # Data directory
│   ├── raw/                  # Raw dataset (real/fake images)
│   │   ├── real/
│   │   └── fake/
│   ├── processed/            # Processed dataset for training
│   │   ├── train/
│   │   │   ├── real/
│   │   │   └── fake/
│   │   └── val/
│   │       ├── real/
│   │       └── fake/
│   └── README.md
│
├── pytorch/                  # PyTorch training pipeline
│   ├── models/
│   │   └── xception.py       # Model architecture (MobileNetV2-based)
│   ├── train.py              # Training script
│   ├── dataset.py            # Dataset loader
│   ├── config.yaml           # Training configuration
│   ├── export_to_onnx.py     # ONNX export script
│   └── utils.py              # Utility functions
│
├── tensorflow/               # TensorFlow inference pipeline
│   ├── model/
│   │   └── saved_model/      # Saved TensorFlow model
│   ├── webcam_detector.py    # Real-time webcam detection
│   ├── inference.py          # Inference utilities
│   ├── preprocess.py         # Image preprocessing
│   └── temporal_logic.py     # Temporal smoothing for predictions
│
├── onnx/                     # ONNX models
│   └── xception_deepfake.onnx
│
├── evaluation/               # Evaluation scripts
│   ├── evaluate_model.py     # Model evaluation
│   └── metrics.py            # Metrics calculation
│
└── docs/                     # Documentation
    ├── proposal.pdf
    ├── architecture_diagram.png
    ├── presentation.pptx
    └── results/              # Training results
        ├── confusion_matrix.png
        ├── accuracy_plot.png
        └── performance_metrics.txt
```

## 🔧 Requirements

### System Requirements
- Python 3.11 or 3.12 (PyTorch and TensorFlow do not support Python 3.13 yet)
- CUDA-capable GPU (optional, for faster training)
- Webcam (for real-time detection)

### Python Dependencies
All dependencies are listed in `requirements.txt`:
- `torch` & `torchvision` - PyTorch for training
- `tensorflow` - TensorFlow for inference
- `opencv-python` - Image processing and webcam access
- `mtcnn` - Face detection
- `numpy`, `pillow` - Image handling
- `scikit-learn` - Evaluation metrics
- `onnx`, `onnxruntime`, `tf2onnx` - Model export
- `matplotlib` - Visualization
- `pyyaml` - Configuration files

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/ShravaniDhumal/Deepfake_Detection_System.git
cd Deepfake_Detection_System
```

### 2. Install Python 3.12 (if needed)
```bash
# macOS
brew install python@3.12

# Verify installation
python3.12 --version
```

### 3. Create Virtual Environment
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For detailed setup instructions, see [SETUP.md](SETUP.md).

## 📊 Data Preparation

### Dataset Structure
Organize your dataset as follows:
```
data/
├── raw/
│   ├── real/          # Real images
│   └── fake/          # Deepfake images
└── processed/
    ├── train/
    │   ├── real/
    │   └── fake/
    └── val/
        ├── real/
        └── fake/
```

### Data Processing
1. Place raw images in `data/raw/real/` and `data/raw/fake/`
2. Split data into train/validation sets
3. Copy processed images to `data/processed/train/` and `data/processed/val/`

**Note**: The dataset loader expects images in `real/` and `fake/` subdirectories.

## 🎓 Training

### Configuration
Edit `pytorch/config.yaml` to customize training parameters:

```yaml
data:
  train_dir: ../data/processed/train
  val_dir: ../data/processed/val
  image_size: 224
  batch_size: 16

training:
  epochs: 5
  learning_rate: 0.0001
  num_classes: 2

model:
  pretrained: true
  save_path: ../pytorch/xception_deepfake.pth
```

### Start Training
```bash
cd pytorch
python train.py
```

The training script will:
- Load data from configured directories
- Train a MobileNetV2-based model (transfer learning)
- Save the trained model to the specified path
- Display training loss for each epoch

### Model Architecture
The model uses a pretrained MobileNetV2 backbone with:
- Frozen feature extractor (transfer learning)
- Custom classifier head for binary classification (real/fake)
- Input size: 224x224 RGB images

## 🔍 Real-time Detection

### Webcam Detection
Run the real-time webcam detector:

```bash
cd tensorflow
python webcam_detector.py
```

**Features**:
- Automatic face detection using MTCNN
- Real-time deepfake classification
- Temporal smoothing for stable predictions
- Visual feedback with bounding boxes and labels

**Controls**:
- Press `ESC` to exit

### How It Works
1. **Face Detection**: MTCNN detects faces in each frame
2. **Preprocessing**: Detected faces are resized and normalized
3. **Classification**: Model predicts real/fake
4. **Temporal Smoothing**: Predictions are averaged over a window (30 frames) for stability
5. **Display**: Results are overlaid on the video feed

### Temporal Logic
The `TemporalDetector` class smooths predictions:
- **Window size**: 30 frames (default)
- **Threshold**: 0.6 (60% fake predictions = DEEPFAKE)
- **Status**: "Analyzing" → "REAL" or "DEEPFAKE"

## 📤 Model Export

### Export to ONNX
Convert PyTorch model to ONNX format:

```bash
cd pytorch
python export_to_onnx.py
```

This creates `onnx/xception_deepfake.onnx` for deployment.

## 📈 Evaluation

### Evaluate Model Performance
```bash
cd evaluation
python evaluate_model.py
```

**Requirements**:
- `y_true.npy`: Ground truth labels
- `y_pred.npy`: Model predictions

The script outputs:
- Confusion matrix
- Classification report (precision, recall, F1-score)

### Generate Predictions
To create evaluation files, run inference on your test set and save predictions.

## 🛠️ Usage Examples

### Training a New Model
```bash
# 1. Prepare your data
# 2. Update config.yaml
# 3. Train
cd pytorch
python train.py
```

### Real-time Detection
```bash
# 1. Ensure model is in tensorflow/model/saved_model/
# 2. Run webcam detector
cd tensorflow
python webcam_detector.py
```

### Export Model
```bash
cd pytorch
python export_to_onnx.py
```

## 🔬 Technical Details

### Model
- **Architecture**: MobileNetV2 (pretrained on ImageNet)
- **Input**: 224x224 RGB images
- **Output**: Binary classification (real/fake)
- **Training**: Transfer learning with frozen backbone

### Preprocessing
- Resize to 224x224
- Normalize: `(pixel - 0.5) / 0.5`
- Convert to tensor format

### Face Detection
- **Library**: MTCNN
- **Output**: Bounding boxes for detected faces

## 🐛 Troubleshooting

### Python Version Issues
If you encounter errors with PyTorch/TensorFlow installation:
- Ensure you're using Python 3.11 or 3.12
- See [SETUP.md](SETUP.md) for detailed instructions

### Data Loading Errors
- Verify data directory structure matches expected format
- Ensure images are in `real/` and `fake/` subdirectories
- Check image file formats (JPEG, PNG supported)

### Webcam Issues
- Ensure webcam is connected and accessible
- Check camera permissions
- Try changing camera index in `webcam_detector.py` (default: 0)

### Model Loading Errors
- Verify model path is correct
- Ensure model format matches (PyTorch `.pth` or TensorFlow SavedModel)
- Check model architecture compatibility

## 📝 License

See [LICENSE](LICENSE) file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 🙏 Acknowledgments

- MobileNetV2 architecture from torchvision
- MTCNN for face detection
- Deepfake datasets used for training

---

**Note**: This project is for research and educational purposes. Always verify important decisions with multiple sources and methods.

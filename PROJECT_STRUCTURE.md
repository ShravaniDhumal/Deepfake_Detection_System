# Project Structure - Deepfake Detection System

## 📁 Clean Directory Structure

```
Deepfake_Detection_System/
│
├── backend/                    # Backend API and Training
│   ├── app.py                 # Flask API server (main entry point)
│   ├── pytorch/               # PyTorch training pipeline
│   │   ├── train_improved.py  # Main training script
│   │   ├── dataset_improved.py
│   │   ├── config.yaml        # Training configuration
│   │   ├── utils.py           # Training utilities
│   │   └── models/            # Model architecture
│   │       ├── __init__.py
│   │       └── xception.py
│   ├── tensorflow/            # TensorFlow inference (optional)
│   │   ├── inference.py
│   │   ├── preprocess.py
│   │   ├── temporal_logic.py
│   │   └── webcam_detector_improved.py
│   ├── utils/                 # Utility scripts
│   │   ├── extract_frames.py
│   │   ├── extract_frames_final.py
│   │   └── split_dataset.py
│   ├── uploads/               # Temporary file uploads
│   ├── train.sh               # Training helper script
│   └── README.md              # Backend documentation
│
├── frontend/                  # Frontend User Interface
│   ├── index.html             # Complete web UI (HTML/CSS/JS)
│   └── README.md              # Frontend documentation
│
├── models/                    # Trained Model Weights
│   ├── xception_deepfake.pth  # Trained model (after training)
│   └── README.md              # Models documentation
│
├── data/                      # Dataset (if using local data)
│   ├── raw/                   # Raw dataset
│   └── processed/            # Processed dataset
│       ├── train/
│       │   ├── real/
│       │   └── fake/
│       └── val/
│           ├── real/
│           └── fake/
│
├── evaluation/                # Model Evaluation
│   ├── evaluate_model.py
│   ├── generate_predictions.py
│   └── metrics.py
│
├── docs/                      # Documentation
│   ├── archive/               # Archived documentation
│   ├── architecture_diagram.png
│   ├── presentation.pptx
│   └── proposal.pdf
│
├── Test/                      # Test Videos
│   ├── Real/                  # Real video samples
│   └── deepfake/              # Deepfake video samples
│
├── README.md                   # Main project documentation
├── QUICK_START.md             # Quick start guide
├── PROJECT_STRUCTURE.md       # This file
├── RESTRUCTURE_SUMMARY.md     # Restructure details
└── requirements.txt           # Python dependencies
```

## 🎯 Directory Purposes

### Backend (`backend/`)
- **Purpose**: API server, model training, and inference
- **Main Entry**: `backend/app.py` - Flask server
- **Training**: `backend/pytorch/train_improved.py`
- **Key Files**:
  - `app.py` - Flask API with all endpoints
  - `pytorch/train_improved.py` - Model training
  - `pytorch/config.yaml` - Training configuration

### Frontend (`frontend/`)
- **Purpose**: User interface
- **Main File**: `frontend/index.html` - Complete web UI
- **Features**: Image upload, webcam, results display

### Models (`models/`)
- **Purpose**: Store trained model weights
- **Benefit**: Share models across devices without retraining
- **File**: `xception_deepfake.pth` (after training)

### Data (`data/`)
- **Purpose**: Training datasets
- **Structure**: `processed/train/` and `processed/val/` with `real/` and `fake/` subdirectories

### Evaluation (`evaluation/`)
- **Purpose**: Model evaluation and metrics
- **Scripts**: Evaluation, prediction generation, metrics calculation

### Utils (`backend/utils/`)
- **Purpose**: Helper scripts for data processing
- **Scripts**: Frame extraction, dataset splitting

## 🚀 How to Use

### Training (One Time)
```bash
cd backend/pytorch
python train_improved.py
```
→ Model saved to `models/xception_deepfake.pth`

### Running the Application
```bash
cd backend
python app.py
```
→ Server starts on `http://localhost:3000`

### Sharing Models
```bash
# Copy model to other device
cp models/xception_deepfake.pth /path/to/other/device/models/
```

## ✅ Clean Structure Benefits

- **No Duplicates**: Single source of truth for each component
- **Clear Separation**: Frontend, backend, and data are separate
- **Easy Navigation**: Logical organization makes finding files easy
- **Scalable**: Easy to add new features or components
- **Maintainable**: Clear structure reduces confusion

## 📝 File Organization Rules

1. **Backend code** → `backend/`
2. **Frontend code** → `frontend/`
3. **Trained models** → `models/`
4. **Training data** → `data/`
5. **Documentation** → `docs/` or root `.md` files
6. **Utility scripts** → `backend/utils/`
7. **Test files** → `Test/`

## 🔄 Migration Notes

If you have old files:
- Old `app.py` → Use `backend/app.py`
- Old `pytorch/` → Use `backend/pytorch/`
- Old `templates/` → Use `frontend/`
- Old `tensorflow/` → Use `backend/tensorflow/`

All old files have been moved or removed for a clean structure.

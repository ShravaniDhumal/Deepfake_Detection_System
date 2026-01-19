# Folder Organization Guide

This document explains the organization of all folders in the project.

## 📁 Root Level Folders

### `backend/` - Backend Code
**Purpose**: All backend API, training, and inference code

```
backend/
├── app.py              # Flask API server (main entry point)
├── pytorch/               # PyTorch training pipeline
├── tensorflow/            # TensorFlow inference (optional)
├── evaluation/            # Model evaluation scripts
├── utils/                 # Utility scripts (data processing)
└── uploads/               # Temporary file uploads
```

**Why here?** All backend code is centralized for easy maintenance.

---

### `frontend/` - Frontend Code
**Purpose**: User interface code

```
frontend/
└── index.html             # Complete web UI (HTML/CSS/JS)
```

**Why here?** Separated from backend for clear frontend/backend distinction.

---

### `models/` - Trained Models
**Purpose**: Store trained model weights and exports

```
models/
├── xception_deepfake.pth  # PyTorch model (after training)
└── onnx/                  # ONNX exported models
    └── xception_deepfake.onnx
```

**Why here?** 
- Models are shared across devices
- Easy to find and copy
- Separate from code

---

### `data/` - Training Data
**Purpose**: Datasets for training and validation

```
data/
├── raw/                   # Raw dataset (original files)
│   ├── real/
│   └── fake/
└── processed/             # Processed dataset (ready for training)
    ├── train/
    │   ├── real/
    │   └── fake/
    └── val/
        ├── real/
        └── fake/
```

**Why here?** 
- Data is separate from code
- Easy to manage datasets
- Can be large, so kept separate

---

### `docs/` - Documentation
**Purpose**: Project documentation and presentations

```
docs/
├── architecture_diagram.png
├── presentation.pptx
├── proposal.pdf
└── archive/               # Old documentation files
```

**Why here?** 
- All documentation in one place
- Easy to find project docs
- Archive for old files

---

### `test/` - Test Videos
**Purpose**: Test video samples for evaluation

```
test/
├── Real/                  # Real video samples
└── deepfake/              # Deepfake video samples
```

**Why here?** 
- Test data separate from training data
- Easy to access for testing
- Clear purpose

---

## 📋 Folder Summary

| Folder | Location | Purpose | Keep at Root? |
|--------|----------|---------|---------------|
| `backend/` | Root | Backend code | ✅ Yes |
| `frontend/` | Root | Frontend code | ✅ Yes |
| `models/` | Root | Trained models | ✅ Yes |
| `data/` | Root | Training data | ✅ Yes |
| `docs/` | Root | Documentation | ✅ Yes |
| `test/` | Root | Test videos | ✅ Yes |
| `evaluation/` | `backend/` | Evaluation scripts | ❌ Moved to backend |
| `onnx/` | `models/` | ONNX exports | ❌ Moved to models |
| `uploads/` | `backend/` | File uploads | ❌ In backend |

## 🎯 Organization Principles

1. **Backend code** → `backend/`
   - API server
   - Training scripts
   - Inference code
   - Evaluation scripts
   - Utilities

2. **Frontend code** → `frontend/`
   - HTML/CSS/JavaScript
   - UI components

3. **Data & Models** → Root level
   - `data/` - Training datasets
   - `models/` - Trained models
   - `test/` - Test samples

4. **Documentation** → `docs/`
   - Project docs
   - Presentations
   - Archived files

## ✅ Benefits

- **Clear Structure**: Easy to find what you need
- **Logical Grouping**: Related files together
- **No Confusion**: Each folder has a clear purpose
- **Easy Navigation**: Intuitive organization
- **Scalable**: Easy to add new features

## 📝 Quick Reference

```bash
# Backend
cd backend                    # API server
cd backend/pytorch            # Training
cd backend/evaluation         # Evaluation
cd backend/utils              # Utilities

# Frontend
cd frontend                   # Web UI

# Data & Models
cd data                       # Training data
cd models                     # Trained models
cd test                       # Test videos

# Documentation
cd docs                       # Project docs
```

---

**Result**: Clean, well-organized structure that's easy to understand and navigate! 🎉

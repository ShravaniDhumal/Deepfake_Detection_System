# Deepfake Detection System

A comprehensive deepfake detection system with separated frontend and backend architecture. Train models once and use them across all devices without retraining.

## 🎯 Features

- **PyTorch-based Training**: Train custom deepfake detection models
- **Web Interface**: Modern, responsive web UI for image upload and analysis
- **Real-time Detection**: Live webcam-based deepfake detection
- **Model Sharing**: Train once, use everywhere - no retraining needed
- **RESTful API**: Clean backend API for integration

## 📁 Project Structure

```
Deepfake_Detection_System/
├── backend/              # Backend API and training
│   ├── app.py           # Flask API server
│   ├── pytorch/         # Training scripts and models
│   │   ├── train_improved.py
│   │   ├── config.yaml
│   │   └── models/
│   ├── tensorflow/      # TensorFlow inference (optional)
│   ├── utils/           # Utility scripts
│   └── uploads/         # Temporary uploads
│
├── frontend/            # Frontend UI
│   └── index.html       # Complete web interface
│
├── models/              # Trained model weights (shared)
│   └── xception_deepfake.pth
│
├── data/                # Dataset (if using local data)
├── evaluation/          # Model evaluation scripts
├── docs/                # Documentation
└── Test/                # Test videos
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a Model (One Time Only)

**Option A: Using Hugging Face Dataset (Recommended)**
```bash
cd backend/pytorch
python train_improved.py
```

**Option B: Using Local Dataset**
1. Prepare your dataset in `data/processed/train/` and `data/processed/val/`
2. Update `backend/pytorch/config.yaml` to set `use_huggingface_dataset: false`
3. Run training:
```bash
cd backend/pytorch
python train_improved.py
```

The model will be saved to `models/xception_deepfake.pth`

### 3. Share Model to Other Devices (Optional)

Copy the trained model to other devices:
```bash
cp models/xception_deepfake.pth /path/to/other/device/models/
```

### 4. Run the Application

```bash
cd backend
python app.py
```

Open `http://localhost:3000` in your browser.

## 📖 Documentation

- **Quick Start**: See `QUICK_START.md`
- **Project Structure**: See `PROJECT_STRUCTURE.md`
- **Backend Details**: See `backend/README.md`
- **Frontend Details**: See `frontend/README.md`
- **Models**: See `models/README.md`

## 🎯 Key Benefits

✅ **Train Once, Use Everywhere**: Train on one device, copy model to others  
✅ **No Retraining**: Use the same model on all devices  
✅ **Clear Structure**: Frontend and backend separated  
✅ **Easy Sharing**: Models in dedicated directory  
✅ **Clean Codebase**: Well-organized, no duplicate files  

## 🔧 Requirements

- Python 3.11 or 3.12
- PyTorch 2.1.2+
- Flask 3.0.0+
- OpenCV 4.8+
- See `requirements.txt` for complete list

## 🐛 Troubleshooting

### Model Not Found
- Train a model first: `cd backend/pytorch && python train_improved.py`
- Check that `models/xception_deepfake.pth` exists

### Training Issues
- Check dataset paths in `backend/pytorch/config.yaml`
- Verify dataset structure (real/fake folders)
- Check training logs

### Port Already in Use
- Change port in `backend/app.py`: `app.run(port=3001)`

## 📝 License

See LICENSE file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Note**: This project is for research and educational purposes. Always verify important decisions with multiple sources and methods.

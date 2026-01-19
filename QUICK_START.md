# Quick Start Guide - Deepfake Detection System

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train a Model (One Time Only)

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

### Step 3: Share Model to Other Devices (Optional)

Copy the trained model to other devices:
```bash
# Copy model file
cp models/xception_deepfake.pth /path/to/other/device/models/
```

### Step 4: Run the Application

```bash
cd backend
python app.py
```

Open `http://localhost:3000` in your browser.

## 📁 Project Structure

```
Deepfake_Detection_System/
├── backend/          # API server and training
├── frontend/         # Web UI
├── models/           # Trained models (shared)
└── data/             # Dataset (if using local)
```

## 🎯 Key Benefits

✅ **Train Once, Use Everywhere**: Train on one device, copy model to others  
✅ **No Retraining**: Use the same model on all devices  
✅ **Clear Structure**: Frontend and backend separated  
✅ **Easy Sharing**: Models in dedicated directory  

## 📖 More Information

- **Project Structure**: See `PROJECT_STRUCTURE.md`
- **Backend Details**: See `backend/README.md`
- **Frontend Details**: See `frontend/README.md`
- **Models**: See `models/README.md`

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

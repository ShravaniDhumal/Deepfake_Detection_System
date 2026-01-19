# Backend - Deepfake Detection API

This directory contains the backend server and training scripts for the Deepfake Detection System.

## Structure

```
backend/
├── app.py                 # Flask API server
├── pytorch/               # Training scripts and model definitions
│   ├── train_improved.py # Main training script
│   ├── dataset_improved.py
│   ├── config.yaml       # Training configuration
│   ├── models/           # Model architecture definitions
│   └── utils.py          # Utility functions
├── tensorflow/           # TensorFlow inference (if needed)
└── uploads/              # Temporary upload directory
```

## Running the Backend Server

```bash
# From project root
cd backend
python app.py
```

The server will start on `http://localhost:3000`

## Training a Model

### Option 1: Using Hugging Face Dataset (Recommended)

1. Edit `pytorch/config.yaml`:
   ```yaml
   data:
     use_huggingface_dataset: true
     huggingface_dataset: "prithivMLmods/OpenDeepfake-Preview"
   ```

2. Run training:
   ```bash
   cd backend/pytorch
   python train_improved.py
   ```

### Option 2: Using Local Dataset

1. Prepare your dataset:
   ```
   data/processed/
   ├── train/
   │   ├── real/
   │   └── fake/
   └── val/
       ├── real/
       └── fake/
   ```

2. Edit `pytorch/config.yaml`:
   ```yaml
   data:
     use_huggingface_dataset: false
     train_dir: ../../data/processed/train
     val_dir: ../../data/processed/val
   ```

3. Run training:
   ```bash
   cd backend/pytorch
   python train_improved.py
   ```

## Model Storage

Trained models are saved to the `models/` directory at the project root. This allows you to:

- **Share models across devices**: Copy the model file to other machines
- **Avoid retraining**: Use the same trained model on different devices
- **Version control**: Keep different model versions

The model will be saved as `models/xception_deepfake.pth` after training completes.

## API Endpoints

- `GET /` - Serve frontend
- `POST /api/upload` - Upload and analyze an image
- `POST /api/webcam/start` - Start webcam capture
- `GET /api/webcam/frame` - Get latest webcam frame with detections
- `POST /api/webcam/stop` - Stop webcam capture
- `GET /api/history` - Get analysis history
- `GET /api/stats` - Get statistics

## Troubleshooting

### Model Not Found Error

If you see "Model not available" errors:
1. Train a model first using `train_improved.py`
2. Ensure the model is saved to `models/xception_deepfake.pth`
3. Check that the model file exists and is readable

### Training Issues

If training shows wrong information:
1. Check your dataset - ensure images are properly labeled (real/fake)
2. Verify dataset paths in `config.yaml`
3. Increase training epochs if accuracy is low
4. Check training logs for errors

### Path Issues

All paths in the backend are relative to the backend directory. The training script automatically resolves paths from the config file location.

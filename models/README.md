# Models Directory

This directory stores trained model weights that can be shared across different devices.

## Purpose

- **Avoid Retraining**: Once trained, copy the model file to other devices
- **Consistency**: Use the same trained model across all devices
- **Version Control**: Keep different model versions for comparison

## Model File

After training, the model will be saved as:
```
models/xception_deepfake.pth
```

## Sharing Models Across Devices

### Method 1: Direct Copy
```bash
# On training device
cp models/xception_deepfake.pth /path/to/shared/location

# On other device
cp /path/to/shared/location/xception_deepfake.pth models/
```

### Method 2: Git (for small models)
```bash
# Add model to git (if repository is set up)
git add models/xception_deepfake.pth
git commit -m "Add trained model"
git push

# On other device
git pull
```

### Method 3: Cloud Storage
Upload to Google Drive, Dropbox, etc., and download on other devices.

## Model Information

- **Architecture**: MobileNetV2 (Xception-based)
- **Input Size**: 224x224 RGB images
- **Output**: Binary classification (Real/Fake)
- **Format**: PyTorch state dict (.pth)

## Usage

The backend automatically looks for models in this directory. No configuration needed - just place the model file here and restart the backend server.

## Important Notes

- **File Size**: Model files can be large (50-200MB typically)
- **Compatibility**: Models trained on one device work on all devices (CPU/GPU)
- **Version**: Ensure PyTorch versions are compatible across devices

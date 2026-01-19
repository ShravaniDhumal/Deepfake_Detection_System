# Data Directory

This directory contains training and test datasets for the deepfake detection system.

## Structure

```
data/
├── raw/                    # Raw dataset (original images/videos)
│   ├── real/              # Real face images/videos
│   └── fake/              # Deepfake images/videos
│
└── processed/              # Processed dataset for training
    ├── train/             # Training data
    │   ├── real/          # Real training images
    │   └── fake/          # Fake training images
    │
    └── val/               # Validation data
        ├── real/          # Real validation images
        └── fake/          # Fake validation images
```

## Usage

### For Local Training

1. Place your raw data in `data/raw/real/` and `data/raw/fake/`
2. Process and split data using utility scripts:
   ```bash
   cd backend/utils
   python extract_frames.py  # Extract frames from videos
   python split_dataset.py   # Split into train/val
   ```
3. Processed data will be in `data/processed/train/` and `data/processed/val/`
4. Update `backend/pytorch/config.yaml` to use local data:
   ```yaml
   data:
     use_huggingface_dataset: false
     train_dir: ../../data/processed/train
     val_dir: ../../data/processed/val
   ```

### For Hugging Face Dataset

The system can also use Hugging Face datasets. See `backend/pytorch/config.yaml` for configuration.

## Notes

- Images should be in formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Videos should be in `.mp4` format
- Ensure proper labeling: real images in `real/`, fake images in `fake/`

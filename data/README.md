# Data Directory

This directory contains the dataset for deepfake detection training and evaluation.

## Directory Structure

```
data/
├── raw/              # Raw, unprocessed images
│   ├── real/         # Real (authentic) images
│   └── fake/         # Deepfake images
│
└── processed/        # Processed images ready for training
    ├── train/        # Training set
    │   ├── real/     # Real images for training
    │   └── fake/     # Fake images for training
    └── val/          # Validation set
        ├── real/     # Real images for validation
        └── fake/     # Fake images for validation
```

## Data Requirements

### Image Format
- **Supported formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`
- **Recommended size**: 224x224 pixels (will be resized automatically)
- **Color mode**: RGB (3 channels)

### Dataset Split
- **Training set**: 70-80% of total data
- **Validation set**: 20-30% of total data
- **Balance**: Try to maintain similar number of real and fake images

### Minimum Requirements
- At least 100 images per class for basic training
- 1000+ images per class recommended for better performance
- Balanced dataset (equal real/fake) for best results

## Data Preparation

### Step 1: Organize Raw Data
Place your raw images in:
```
data/raw/real/    # Authentic images
data/raw/fake/    # Deepfake images
```

### Step 2: Process Data
Use the preprocessing script to prepare data:
```bash
python scripts/preprocess_data.py
```

Or manually organize:
1. Split data into train/val sets (e.g., 80/20)
2. Copy images to appropriate directories:
   - `data/processed/train/real/`
   - `data/processed/train/fake/`
   - `data/processed/val/real/`
   - `data/processed/val/fake/`

### Step 3: Verify Data
Check data structure:
```bash
python scripts/verify_data.py
```

## Sample Data

To create sample test data:
```bash
cd pytorch
python create_sample_structure.py
```

**Note**: Sample data is for testing only. Replace with real deepfake dataset for actual training.

## Data Sources

### Public Datasets
- **FaceForensics++**: https://github.com/ondyari/FaceForensics
- **DFDC (Deepfake Detection Challenge)**: https://www.kaggle.com/c/deepfake-detection-challenge
- **Celeb-DF**: https://github.com/yuezunli/celeb-deepfakeforensics

### Data Collection Tips
1. **Diversity**: Include various face angles, lighting, backgrounds
2. **Quality**: Use high-resolution images when possible
3. **Balance**: Maintain similar distribution of real and fake
4. **Validation**: Manually verify labels before training

## Data Augmentation

The training pipeline automatically applies augmentation:
- Random horizontal flip
- Random crop/resize
- Color jitter
- Normalization

See `pytorch/dataset_improved.py` for details.

## File Naming

Recommended naming convention:
- Real images: `real_001.jpg`, `real_002.jpg`, ...
- Fake images: `fake_001.jpg`, `fake_002.jpg`, ...

Or use descriptive names:
- `person1_real.jpg`
- `person1_fake.jpg`

## Data Privacy

⚠️ **Important**: 
- Ensure you have permission to use images
- Respect privacy and consent
- Follow data protection regulations
- Do not share sensitive personal data

## Troubleshooting

### No images found
- Check file paths in `pytorch/config.yaml`
- Verify images are in correct directories
- Ensure file extensions are supported

### Corrupted images
- The dataset loader will skip corrupted images
- Check logs for warnings about failed images
- Re-download or re-process problematic files

### Memory issues
- Reduce batch size in `pytorch/config.yaml`
- Use smaller image resolution
- Process data in smaller batches

## Statistics

To check dataset statistics:
```bash
python scripts/data_stats.py
```

This will show:
- Total number of images
- Class distribution
- Image dimensions
- File sizes

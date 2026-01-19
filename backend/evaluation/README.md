# Model Evaluation

This directory contains scripts for evaluating trained deepfake detection models.

## Scripts

- **evaluate_model.py** - Main evaluation script
  - Loads ground truth labels (`y_true.npy`)
  - Loads model predictions (`y_pred.npy`)
  - Generates confusion matrix and classification report

- **generate_predictions.py** - Generate predictions from a trained model
  - Loads a trained PyTorch model
  - Runs inference on test data
  - Saves predictions to `.npy` files

- **metrics.py** - Additional metrics calculation utilities

## Usage

### Generate Predictions

```bash
cd backend/evaluation
python generate_predictions.py \
    --model_path ../../models/xception_deepfake.pth \
    --data_dir ../../data/processed/val \
    --output_dir .
```

This will create `y_true.npy` and `y_pred.npy` files.

### Evaluate Model

```bash
cd backend/evaluation
python evaluate_model.py
```

This will load `y_true.npy` and `y_pred.npy` and display:
- Confusion matrix
- Classification report (precision, recall, F1-score)

## Requirements

- Trained model in `models/xception_deepfake.pth`
- Test data with ground truth labels
- NumPy and scikit-learn installed

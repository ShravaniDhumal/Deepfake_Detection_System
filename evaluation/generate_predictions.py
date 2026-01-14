"""
Generate predictions from trained model for evaluation
"""
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pytorch'))

try:
    from dataset_improved import DeepfakeDataset
except ImportError:
    from dataset import DeepfakeDataset
from models.xception import get_xception
from torch.utils.data import DataLoader

def generate_predictions(model_path: str, data_dir: str, 
                        output_dir: str = ".", batch_size: int = 16):
    """
    Generate predictions for evaluation
    
    Args:
        model_path: Path to trained PyTorch model
        data_dir: Directory containing test data (with real/ and fake/ subdirs)
        output_dir: Directory to save predictions
        batch_size: Batch size for inference
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}")
    model = get_xception(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    # Load dataset
    print(f"Loading data from {data_dir}")
    dataset = DeepfakeDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Generate predictions
    all_predictions = []
    all_labels = []
    all_probs = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    y_pred = np.array(all_predictions)
    y_true = np.array(all_labels)
    y_proba = np.array(all_probs)
    
    # Save predictions
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path / "y_true.npy", y_true)
    np.save(output_path / "y_pred.npy", y_pred)
    np.save(output_path / "y_proba.npy", y_proba)
    
    print(f"\nPredictions saved to {output_path}")
    print(f"  - y_true.npy: {len(y_true)} true labels")
    print(f"  - y_pred.npy: {len(y_pred)} predictions")
    print(f"  - y_proba.npy: {len(y_proba)} probability scores")
    
    # Print summary
    accuracy = np.mean(y_pred == y_true)
    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Real predictions: {np.sum(y_pred == 0)}")
    print(f"Fake predictions: {np.sum(y_pred == 1)}")
    
    return y_true, y_pred, y_proba

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate predictions for evaluation')
    parser.add_argument('--model', type=str, default='../pytorch/xception_deepfake.pth',
                       help='Path to trained model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--output', type=str, default='.',
                       help='Output directory for predictions')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for inference')
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        exit(1)
    
    if not Path(args.data).exists():
        print(f"Error: Data directory not found: {args.data}")
        exit(1)
    
    generate_predictions(args.model, args.data, args.output, args.batch_size)

"""
Visualize training results and metrics
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from utils import plot_training_history, plot_confusion_matrix

def load_training_log(log_path: str = "training.log"):
    """Parse training log file"""
    epochs = []
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    if not Path(log_path).exists():
        print(f"Log file not found: {log_path}")
        return None
    
    with open(log_path, 'r') as f:
        for line in f:
            if "Epoch" in line and "Loss:" in line and "Acc:" in line:
                # Parse line like: "Epoch 1/5 - Train Loss: 0.1234, Train Acc: 85.50% - Val Loss: 0.2345, Val Acc: 80.00%"
                parts = line.split(" - ")
                if len(parts) >= 2:
                    epoch_part = parts[0]
                    train_part = parts[1]
                    val_part = parts[2] if len(parts) > 2 else ""
                    
                    # Extract epoch number
                    epoch = int(epoch_part.split()[1].split('/')[0])
                    epochs.append(epoch)
                    
                    # Extract train loss and acc
                    if "Train Loss:" in train_part:
                        train_loss = float(train_part.split("Train Loss:")[1].split(",")[0].strip())
                        train_losses.append(train_loss)
                        train_acc = float(train_part.split("Train Acc:")[1].split("%")[0].strip())
                        train_accs.append(train_acc)
                    
                    # Extract val loss and acc
                    if "Val Loss:" in val_part:
                        val_loss = float(val_part.split("Val Loss:")[1].split(",")[0].strip())
                        val_losses.append(val_loss)
                        val_acc = float(val_part.split("Val Acc:")[1].split("%")[0].strip())
                        val_accs.append(val_acc)
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }

def visualize_training(log_path: str = "training.log", save_dir: str = "../docs/results"):
    """Create visualizations from training log"""
    data = load_training_log(log_path)
    
    if data is None or len(data['epochs']) == 0:
        print("No training data found in log file")
        return
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Plot training history
    plot_training_history(
        data['train_losses'],
        data['val_losses'],
        data['train_accs'],
        data['val_accs'],
        str(save_path / "training_history.png")
    )
    
    print(f"Training visualizations saved to {save_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total Epochs: {len(data['epochs'])}")
    if data['train_losses']:
        print(f"Final Train Loss: {data['train_losses'][-1]:.4f}")
        print(f"Final Train Acc: {data['train_accs'][-1]:.2f}%")
    if data['val_losses']:
        print(f"Final Val Loss: {data['val_losses'][-1]:.4f}")
        print(f"Final Val Acc: {data['val_accs'][-1]:.2f}%")

def compare_models(model_paths: list, test_data_path: str):
    """Compare multiple trained models"""
    # This would load models and compare on test set
    pass

if __name__ == "__main__":
    import sys
    
    log_path = "training.log"
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    
    visualize_training(log_path)

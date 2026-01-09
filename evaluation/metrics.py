"""
Metrics calculation and evaluation utilities
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

class MetricsCalculator:
    """Calculate and visualize classification metrics"""
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_proba: np.ndarray = None, class_names: list = None):
        """
        Initialize metrics calculator
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional, for ROC/PR curves)
            class_names: Names of classes (default: ['Real', 'Fake'])
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.class_names = class_names or ['Real', 'Fake']
        
        # Calculate basic metrics
        self.accuracy = accuracy_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        self.recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        self.f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        self.cm = confusion_matrix(y_true, y_pred)
        
    def get_all_metrics(self) -> dict:
        """Get all calculated metrics as dictionary"""
        return {
            'accuracy': float(self.accuracy),
            'precision': float(self.precision),
            'recall': float(self.recall),
            'f1_score': float(self.f1),
            'confusion_matrix': self.cm.tolist()
        }
    
    def print_metrics(self):
        """Print all metrics to console"""
        print("=" * 60)
        print("CLASSIFICATION METRICS")
        print("=" * 60)
        print(f"Accuracy:  {self.accuracy:.4f}")
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall:    {self.recall:.4f}")
        print(f"F1-Score:  {self.f1:.4f}")
        print("\nConfusion Matrix:")
        print(self.cm)
        print("\nClassification Report:")
        print(classification_report(self.y_true, self.y_pred, 
                                   target_names=self.class_names))
    
    def plot_confusion_matrix(self, save_path: str = None, show: bool = True):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_roc_curve(self, save_path: str = None, show: bool = True):
        """Plot ROC curve (requires y_proba)"""
        if self.y_proba is None:
            print("Warning: y_proba not provided, cannot plot ROC curve")
            return
        
        # Get probabilities for positive class (fake)
        if self.y_proba.ndim > 1:
            y_scores = self.y_proba[:, 1]
        else:
            y_scores = self.y_proba
        
        fpr, tpr, thresholds = roc_curve(self.y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return roc_auc
    
    def plot_precision_recall_curve(self, save_path: str = None, show: bool = True):
        """Plot Precision-Recall curve (requires y_proba)"""
        if self.y_proba is None:
            print("Warning: y_proba not provided, cannot plot PR curve")
            return
        
        # Get probabilities for positive class (fake)
        if self.y_proba.ndim > 1:
            y_scores = self.y_proba[:, 1]
        else:
            y_scores = self.y_proba
        
        precision, recall, thresholds = precision_recall_curve(self.y_true, y_scores)
        avg_precision = average_precision_score(self.y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR curve saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return avg_precision
    
    def save_metrics(self, filepath: str):
        """Save metrics to JSON file"""
        metrics = self.get_all_metrics()
        metrics['classification_report'] = classification_report(
            self.y_true, self.y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to {filepath}")

def calculate_metrics_from_files(y_true_path: str, y_pred_path: str, 
                                 y_proba_path: str = None) -> MetricsCalculator:
    """
    Load predictions from files and calculate metrics
    
    Args:
        y_true_path: Path to y_true.npy
        y_pred_path: Path to y_pred.npy
        y_proba_path: Path to y_proba.npy (optional)
        
    Returns:
        MetricsCalculator instance
    """
    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)
    y_proba = np.load(y_proba_path) if y_proba_path and Path(y_proba_path).exists() else None
    
    return MetricsCalculator(y_true, y_pred, y_proba)

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python metrics.py <y_true.npy> <y_pred.npy> [y_proba.npy]")
        sys.exit(1)
    
    y_true_path = sys.argv[1]
    y_pred_path = sys.argv[2]
    y_proba_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    metrics = calculate_metrics_from_files(y_true_path, y_pred_path, y_proba_path)
    metrics.print_metrics()
    metrics.plot_confusion_matrix(save_path="confusion_matrix.png", show=False)
    
    if y_proba_path:
        metrics.plot_roc_curve(save_path="roc_curve.png", show=False)
        metrics.plot_precision_recall_curve(save_path="pr_curve.png", show=False)
    
    metrics.save_metrics("metrics.json")

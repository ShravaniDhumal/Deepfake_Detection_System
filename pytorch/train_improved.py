"""
Improved training script with validation, early stopping, and better logging
"""
import torch
import yaml
import os
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from torch import nn, optim
try:
    from dataset_improved import DeepfakeDataset
except ImportError:
    from dataset import DeepfakeDataset
    # Original dataset doesn't support augment parameter
    import warnings
    warnings.warn("Using original dataset.py - augment parameter will be ignored")
from models.xception import get_xception

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def main():
    try:
        # Load configuration
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        logger.info("Configuration loaded successfully")
        
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load datasets
        train_dir = config["data"]["train_dir"]
        val_dir = config["data"]["val_dir"]
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not os.path.exists(val_dir):
            raise FileNotFoundError(f"Validation directory not found: {val_dir}")
        
        # Try to use improved dataset with augmentation, fallback to original
        try:
            train_dataset = DeepfakeDataset(train_dir, augment=True)
            val_dataset = DeepfakeDataset(val_dir, augment=False)
        except TypeError:
            # Original dataset doesn't support augment parameter
            train_dataset = DeepfakeDataset(train_dir)
            val_dataset = DeepfakeDataset(val_dir)
        
        if len(train_dataset) == 0:
            logger.error(f"No training images found in {train_dir}")
            logger.error("\n" + "="*60)
            logger.error("TRAINING DATA REQUIRED")
            logger.error("="*60)
            logger.error(f"\nPlease add images to:")
            logger.error(f"  - {os.path.join(train_dir, 'real')}/")
            logger.error(f"  - {os.path.join(train_dir, 'fake')}/")
            logger.error(f"\nSupported formats: .jpg, .jpeg, .png, .bmp")
            logger.error(f"\nExample structure:")
            logger.error(f"  {train_dir}/")
            logger.error(f"    ├── real/")
            logger.error(f"    │   ├── image1.jpg")
            logger.error(f"    │   └── image2.jpg")
            logger.error(f"    └── fake/")
            logger.error(f"        ├── image1.jpg")
            logger.error(f"        └── image2.jpg")
            raise ValueError(f"No training images found in {train_dir}")
        
        if len(val_dataset) == 0:
            logger.warning(f"No validation images found in {val_dir}")
            logger.warning("Training will continue but validation metrics won't be available")
            logger.warning("Consider adding validation data for better model evaluation")
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=True,
            num_workers=2,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["data"]["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        # Initialize model
        model = get_xception(config["training"]["num_classes"]).to(device)
        logger.info("Model initialized")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.classifier.parameters(),
            lr=config["training"]["learning_rate"]
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        best_model_path = config["model"]["save_path"].replace('.pth', '_best.pth')
        
        # Training loop
        logger.info("Starting training...")
        for epoch in range(config["training"]["epochs"]):
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            
            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            logger.info(
                f"Epoch {epoch+1}/{config['training']['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model and save final version
        model.load_state_dict(torch.load(best_model_path))
        final_path = config["model"]["save_path"]
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        torch.save(model.state_dict(), final_path)
        logger.info(f"Final model saved to {final_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

"""
Improved Dataset class with error handling and file filtering
"""
import os
import logging
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

class DeepfakeDataset(Dataset):
    """
    Dataset class for loading deepfake detection images.
    
    Args:
        root_dir: Root directory containing 'real' and 'fake' subdirectories
        transform: Optional custom transform (default: resize + normalize)
        augment: Whether to apply data augmentation (default: False)
    """
    def __init__(self, root_dir, transform=None, augment=False):
        self.samples = []
        self.root_dir = root_dir
        
        # Default transform if none provided
        if transform is None:
            if augment:
                # Training: with augmentation
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                       std=[0.5, 0.5, 0.5])
                ])
            else:
                # Validation: no augmentation
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                       std=[0.5, 0.5, 0.5])
                ])
        else:
            self.transform = transform

        # Load samples with error handling
        self._load_samples()
        
        if len(self.samples) == 0:
            logger.warning(f"No valid images found in {root_dir}")

    def _is_image_file(self, filename: str) -> bool:
        """Check if file is a valid image file"""
        ext = os.path.splitext(filename.lower())[1]
        return ext in IMAGE_EXTENSIONS

    def _load_samples(self):
        """Load image samples with error handling"""
        failed_images = []
        
        for label, cls in enumerate(["real", "fake"]):
            class_path = os.path.join(self.root_dir, cls)
            
            if not os.path.exists(class_path):
                logger.warning(f"Directory not found: {class_path}")
                continue
            
            for img_file in os.listdir(class_path):
                # Filter out non-image files
                if not self._is_image_file(img_file):
                    continue
                
                img_path = os.path.join(class_path, img_file)
                
                # Try to verify image is valid
                try:
                    # Quick validation - try to open image
                    with Image.open(img_path) as img:
                        img.verify()  # Verify it's a valid image
                    
                    self.samples.append((img_path, label))
                except Exception as e:
                    failed_images.append((img_path, str(e)))
                    logger.debug(f"Failed to load {img_path}: {e}")
        
        if failed_images:
            logger.warning(f"Skipped {len(failed_images)} corrupted/invalid images")
            if logger.isEnabledFor(logging.DEBUG):
                for path, error in failed_images[:5]:  # Show first 5
                    logger.debug(f"  {path}: {error}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Re-open image (verify() closes it)
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            # In production, you might want to skip or retry
            fallback_image = Image.new('RGB', (224, 224), color='black')
            return self.transform(fallback_image), label

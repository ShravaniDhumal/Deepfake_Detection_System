#!/usr/bin/env python3
"""
Helper script to create sample data structure and test the training pipeline
"""
import os
import numpy as np
from PIL import Image
from pathlib import Path

def create_sample_images(output_dir, num_images=5, label="real"):
    """Create sample test images"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating {num_images} sample {label} images in {output_dir}")
    
    for i in range(num_images):
        # Create a simple test image (random colored square)
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        img_path = os.path.join(output_dir, f"{label}_{i+1:03d}.jpg")
        img.save(img_path, "JPEG")
        print(f"  Created: {img_path}")
    
    return num_images

def main():
    print("="*60)
    print("Creating Sample Data Structure")
    print("="*60)
    
    base_dir = Path("../data/processed")
    
    # Create directories
    dirs = [
        base_dir / "train" / "real",
        base_dir / "train" / "fake",
        base_dir / "val" / "real",
        base_dir / "val" / "fake"
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    print("\n" + "="*60)
    print("Creating Sample Images")
    print("="*60)
    
    # Create sample images
    train_real = create_sample_images(base_dir / "train" / "real", 10, "real")
    train_fake = create_sample_images(base_dir / "train" / "fake", 10, "fake")
    val_real = create_sample_images(base_dir / "val" / "real", 3, "real")
    val_fake = create_sample_images(base_dir / "val" / "fake", 3, "fake")
    
    total = train_real + train_fake + val_real + val_fake
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"✅ Created {total} sample images")
    print(f"  - Train Real: {train_real}")
    print(f"  - Train Fake: {train_fake}")
    print(f"  - Val Real: {val_real}")
    print(f"  - Val Fake: {val_fake}")
    print("\n⚠️  NOTE: These are random test images, not actual deepfake data.")
    print("   For real training, replace with actual real/fake images.")
    print("\n✅ You can now run: python train_improved.py")

if __name__ == "__main__":
    main()

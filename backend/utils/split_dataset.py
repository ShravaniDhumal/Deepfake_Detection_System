import os
import shutil
import random

def split_dataset(train_dir, val_dir, val_ratio=0.2):
    """Split dataset into train and validation sets."""
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # Get all image files
    all_files = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.jpeg', '.png')) and f != '.gitkeep']

    # Shuffle the files
    random.shuffle(all_files)

    # Calculate split point
    val_count = int(len(all_files) * val_ratio)
    val_files = all_files[:val_count]
    train_files = all_files[val_count:]

    # Move validation files
    for file in val_files:
        src = os.path.join(train_dir, file)
        dst = os.path.join(val_dir, file)
        shutil.move(src, dst)

    print(f"Moved {len(val_files)} files to validation set")
    print(f"Training set: {len(train_files)} files")
    print(f"Validation set: {len(val_files)} files")

def main():
    # Split real images
    print("Splitting real images...")
    split_dataset('data/processed/train/real', 'data/processed/val/real')

    # Split fake images
    print("Splitting fake images...")
    split_dataset('data/processed/train/fake', 'data/processed/val/fake')

    print("Dataset split complete!")

if __name__ == "__main__":
    main()

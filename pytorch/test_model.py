"""
Test trained model on images
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from models.xception import get_xception
from pathlib import Path
import argparse

def load_model(model_path: str, num_classes: int = 2, device: str = 'cpu'):
    """Load trained PyTorch model"""
    model = get_xception(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def preprocess_image(image_path: str, device: str = 'cpu'):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def predict_image(model, image_path: str, device: str = 'cpu'):
    """Predict on a single image"""
    image_tensor = preprocess_image(image_path, device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    label = "FAKE" if predicted_class == 1 else "REAL"
    return label, confidence, predicted_class

def test_model(model_path: str, image_path: str, device: str = 'cpu'):
    """Test model on an image"""
    print(f"Loading model from {model_path}")
    model = load_model(model_path, device=device)
    
    print(f"Testing image: {image_path}")
    label, confidence, class_idx = predict_image(model, image_path, device)
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Class Index: {class_idx}")
    print("=" * 60)
    
    return label, confidence

def test_directory(model_path: str, image_dir: str, device: str = 'cpu'):
    """Test model on all images in a directory"""
    model = load_model(model_path, device=device)
    
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Testing {len(image_files)} images...")
    results = []
    
    for img_path in image_files:
        try:
            label, confidence, _ = predict_image(model, str(img_path), device)
            results.append({
                'image': str(img_path),
                'label': label,
                'confidence': confidence
            })
            print(f"{img_path.name}: {label} ({confidence:.2%})")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Summary
    real_count = sum(1 for r in results if r['label'] == 'REAL')
    fake_count = sum(1 for r in results if r['label'] == 'FAKE')
    avg_confidence = np.mean([r['confidence'] for r in results])
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Images: {len(results)}")
    print(f"Predicted REAL: {real_count}")
    print(f"Predicted FAKE: {fake_count}")
    print(f"Average Confidence: {avg_confidence:.2%}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test trained deepfake detection model')
    parser.add_argument('--model', type=str, default='xception_deepfake.pth',
                       help='Path to trained model')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image file')
    parser.add_argument('--dir', type=str, default=None,
                       help='Path to directory of images')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use (cpu or cuda)')
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        exit(1)
    
    if args.image:
        test_model(args.model, args.image, args.device)
    elif args.dir:
        test_directory(args.model, args.dir, args.device)
    else:
        print("Error: Please provide either --image or --dir")
        parser.print_help()

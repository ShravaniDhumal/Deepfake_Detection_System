"""
Export PyTorch model to ONNX format for deployment
"""
import torch
import yaml
import os
from models.xception import get_xception

def export_to_onnx(model_path, output_path, input_size=(224, 224), num_classes=2):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model_path: Path to PyTorch model (.pth file)
        output_path: Output path for ONNX model (.onnx file)
        input_size: Input image size (height, width)
        num_classes: Number of classes
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Initialize model
    print(f"Loading model from {model_path}...")
    model = get_xception(num_classes=num_classes)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    
    # Export to ONNX
    print(f"Exporting to ONNX format...")
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Model exported successfully to {output_path}")

def main():
    """Main function to export model using config.yaml"""
    try:
        # Load configuration
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Get model path from config
        model_path = config["model"]["save_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Please train the model first using train_improved.py"
            )
        
        # Set output path
        output_dir = "../onnx"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "xception_deepfake.onnx")
        
        # Get parameters from config
        num_classes = config["training"]["num_classes"]
        image_size = config["data"]["image_size"]
        input_size = (image_size, image_size)
        
        # Export model
        export_to_onnx(
            model_path=model_path,
            output_path=output_path,
            input_size=input_size,
            num_classes=num_classes
        )
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        raise

if __name__ == "__main__":
    main()

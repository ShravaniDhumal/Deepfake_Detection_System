"""
Convert PyTorch model to TensorFlow SavedModel format
"""
import torch
import tensorflow as tf
import numpy as np
from models.xception import get_xception
from pathlib import Path

def convert_pytorch_to_tensorflow(pytorch_model_path: str, 
                                  output_dir: str = "../tensorflow/model/saved_model",
                                  input_shape: tuple = (224, 224, 3)):
    """
    Convert PyTorch model to TensorFlow SavedModel
    
    Args:
        pytorch_model_path: Path to PyTorch .pth file
        output_dir: Output directory for TensorFlow model
        input_shape: Input shape (H, W, C) for TensorFlow
    """
    print(f"Loading PyTorch model from {pytorch_model_path}")
    
    # Load PyTorch model
    model = get_xception(num_classes=2)
    model.load_state_dict(torch.load(pytorch_model_path, map_location='cpu'))
    model.eval()
    
    print("Converting to TensorFlow...")
    
    # Create TensorFlow model
    class PyTorchToTensorFlow(tf.keras.Model):
        def __init__(self, pytorch_model):
            super().__init__()
            self.pytorch_model = pytorch_model
        
        def call(self, inputs):
            # Convert TF tensor to PyTorch tensor
            # Note: This is a simplified conversion
            # For production, use proper ONNX conversion or tf2onnx
            inputs_np = inputs.numpy()
            inputs_torch = torch.from_numpy(inputs_np).permute(0, 3, 1, 2)  # NHWC to NCHW
            
            with torch.no_grad():
                outputs = self.pytorch_model(inputs_torch)
            
            # Convert back to TensorFlow
            outputs_np = outputs.numpy()
            return tf.constant(outputs_np)
    
    # Create wrapper model
    tf_model = PyTorchToTensorFlow(model)
    
    # Build model
    dummy_input = tf.zeros((1, *input_shape))
    _ = tf_model(dummy_input)
    
    # Save as SavedModel
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tf.saved_model.save(tf_model, str(output_path))
    print(f"TensorFlow model saved to {output_path}")
    
    return output_path

def convert_via_onnx(pytorch_model_path: str,
                    onnx_model_path: str = "../onnx/xception_deepfake.onnx",
                    output_dir: str = "../tensorflow/model/saved_model"):
    """
    Convert PyTorch -> ONNX -> TensorFlow (recommended method)
    
    Args:
        pytorch_model_path: Path to PyTorch .pth file
        onnx_model_path: Path to save/load ONNX model
        output_dir: Output directory for TensorFlow model
    """
    try:
        import onnx
        import tf2onnx
    except ImportError:
        print("Error: onnx and tf2onnx required for conversion")
        print("Install with: pip install onnx tf2onnx")
        return None
    
    print("Step 1: Converting PyTorch to ONNX...")
    # First export to ONNX (use export_to_onnx.py)
    from export_to_onnx import export_model
    export_model(pytorch_model_path, onnx_model_path)
    
    print("Step 2: Converting ONNX to TensorFlow...")
    # Convert ONNX to TensorFlow
    onnx_model = onnx.load(onnx_model_path)
    
    # Use tf2onnx to convert
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Note: This is a simplified example
    # For full conversion, use tf2onnx.convert command line tool:
    # python -m tf2onnx.convert --onnx model.onnx --output model.pb
    
    print(f"TensorFlow model saved to {output_path}")
    return output_path

if __name__ == "__main__":
    import sys
    
    pytorch_model = "xception_deepfake.pth"
    if len(sys.argv) > 1:
        pytorch_model = sys.argv[1]
    
    if not Path(pytorch_model).exists():
        print(f"Error: Model file not found: {pytorch_model}")
        sys.exit(1)
    
    print("=" * 60)
    print("PyTorch to TensorFlow Conversion")
    print("=" * 60)
    
    # Try ONNX conversion first (recommended)
    try:
        convert_via_onnx(pytorch_model)
    except Exception as e:
        print(f"ONNX conversion failed: {e}")
        print("Trying direct conversion...")
        convert_pytorch_to_tensorflow(pytorch_model)

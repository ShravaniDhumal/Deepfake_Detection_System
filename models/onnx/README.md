# ONNX Models

This directory contains exported ONNX model files for cross-platform deployment.

## Purpose

ONNX (Open Neural Network Exchange) models allow you to:
- Deploy models on different platforms (CPU, GPU, mobile)
- Use different inference engines (ONNX Runtime, TensorRT, etc.)
- Optimize models for production

## Exporting Models

To export a PyTorch model to ONNX format:

```bash
cd backend/pytorch
python export_to_onnx.py
```

The exported model will be saved to `models/onnx/xception_deepfake.onnx`

## Usage

ONNX models can be used with ONNX Runtime:

```python
import onnxruntime as ort

# Load model
session = ort.InferenceSession("models/onnx/xception_deepfake.onnx")

# Run inference
outputs = session.run(None, {"input": preprocessed_image})
```

## Notes

- ONNX models are typically smaller and faster than PyTorch models
- Good for production deployment
- Cross-platform compatible

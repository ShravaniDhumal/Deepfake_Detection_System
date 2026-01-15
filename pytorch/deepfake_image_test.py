import torch
from transformers import AutoProcessor, AutoModelForImageClassification
from PIL import Image

# Load model and processor
MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()

# Load image
image = Image.open("test.jpg").convert("RGB")

# Preprocess
inputs = processor(images=image, return_tensors="pt")

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)

# Output
labels = model.config.id2label
prediction = torch.argmax(probs, dim=1).item()

print("Prediction:", labels[prediction])
print("Confidence:", probs[0][prediction].item())

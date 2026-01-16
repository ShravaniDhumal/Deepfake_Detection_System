import cv2
import torch
from transformers import AutoProcessor, AutoModelForImageClassification
from PIL import Image

MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()

cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run inference every 10 frames (reduces CPU load)
    if frame_count % 10 == 0:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)

        inputs = processor(images=pil_img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        label = model.config.id2label[pred]
        confidence = probs[0][pred].item()

    # Display
    cv2.putText(
        frame,
        f"{label} ({confidence:.2f})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255) if label == "FAKE" else (0, 255, 0),
        2
    )

    cv2.imshow("Deepfake Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

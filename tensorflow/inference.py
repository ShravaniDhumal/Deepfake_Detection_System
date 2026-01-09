"""
Inference utilities for TensorFlow model
"""
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
from typing import Union, List, Tuple
from preprocess import preprocess_face

class DeepfakeInference:
    """Class for running inference on images/videos"""
    
    def __init__(self, model_path: str = "model/saved_model"):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to saved TensorFlow model
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = tf.saved_model.load(model_path)
        self.model_signature = list(self.model.signatures.keys())[0]
        print(f"Model loaded from {model_path}")
    
    def predict_image(self, image: np.ndarray) -> Tuple[int, float]:
        """
        Predict on a single image
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Tuple of (predicted_label, confidence)
            - predicted_label: 0 for real, 1 for fake
            - confidence: Confidence score (0-1)
        """
        # Preprocess image
        processed = preprocess_face(image)
        
        # Run inference
        prediction = self.model(processed)
        
        # Get probabilities
        probs = tf.nn.softmax(prediction, axis=1).numpy()[0]
        
        # Get predicted label and confidence
        predicted_label = int(tf.argmax(prediction, axis=1).numpy()[0])
        confidence = float(probs[predicted_label])
        
        return predicted_label, confidence
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Tuple[int, float]]:
        """
        Predict on a batch of images
        
        Args:
            images: List of input images
            
        Returns:
            List of (predicted_label, confidence) tuples
        """
        results = []
        for image in images:
            label, conf = self.predict_image(image)
            results.append((label, conf))
        return results
    
    def predict_from_path(self, image_path: str) -> Tuple[int, float, str]:
        """
        Predict from image file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (predicted_label, confidence, label_name)
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Predict
        label, confidence = self.predict_image(image)
        label_name = "FAKE" if label == 1 else "REAL"
        
        return label, confidence, label_name
    
    def predict_video_frame(self, frame: np.ndarray, face_box: dict) -> Tuple[int, float]:
        """
        Predict on a face crop from video frame
        
        Args:
            frame: Video frame (BGR)
            face_box: Face bounding box from MTCNN
            
        Returns:
            Tuple of (predicted_label, confidence)
        """
        x, y, w, h = face_box["box"]
        crop = frame[y:y+h, x:x+w]
        return self.predict_image(crop)

def load_model(model_path: str = "model/saved_model"):
    """Convenience function to load model"""
    return DeepfakeInference(model_path)

def predict_single_image(image_path: str, model_path: str = "model/saved_model") -> dict:
    """
    Predict on a single image file
    
    Args:
        image_path: Path to image
        model_path: Path to model
        
    Returns:
        Dictionary with prediction results
    """
    inference = DeepfakeInference(model_path)
    label, confidence, label_name = inference.predict_from_path(image_path)
    
    return {
        "image_path": image_path,
        "prediction": label_name,
        "label": label,
        "confidence": confidence,
        "is_fake": label == 1
    }

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [model_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "model/saved_model"
    
    result = predict_single_image(image_path, model_path)
    print(f"\nPrediction for {result['image_path']}:")
    print(f"  Label: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Is Fake: {result['is_fake']}")

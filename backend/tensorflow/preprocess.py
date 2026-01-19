"""
Image preprocessing utilities for deepfake detection
"""
import cv2
import numpy as np

def preprocess_face(face: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess a face image for model input
    
    Args:
        face: Input face image (BGR format from OpenCV)
        target_size: Target size (width, height) - default (224, 224)
        
    Returns:
        Preprocessed image as numpy array ready for model input
        Shape: (1, height, width, 3) - batch dimension included
    """
    # Resize to target size
    face_resized = cv2.resize(face, target_size)
    
    # Convert to float32 and normalize to [0, 1]
    face_normalized = face_resized.astype("float32") / 255.0
    
    # Normalize to [-1, 1] range (same as PyTorch normalization)
    # Formula: (pixel - 0.5) / 0.5
    face_final = (face_normalized - 0.5) / 0.5
    
    # Add batch dimension: (height, width, channels) -> (1, height, width, channels)
    face_batch = np.expand_dims(face_final, axis=0)
    
    return face_batch

def preprocess_image(image_path: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Load and preprocess an image from file path
    
    Args:
        image_path: Path to image file
        target_size: Target size (width, height) - default (224, 224)
        
    Returns:
        Preprocessed image as numpy array ready for model input
    """
    # Load image (BGR format)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    return preprocess_face(image, target_size)

def preprocess_batch(faces: list, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess a batch of face images
    
    Args:
        faces: List of face images (BGR format)
        target_size: Target size (width, height) - default (224, 224)
        
    Returns:
        Preprocessed batch as numpy array
        Shape: (batch_size, height, width, 3)
    """
    preprocessed = []
    for face in faces:
        processed = preprocess_face(face, target_size)
        # Remove batch dimension and add to list
        preprocessed.append(processed[0])
    
    # Stack into batch
    return np.array(preprocessed)

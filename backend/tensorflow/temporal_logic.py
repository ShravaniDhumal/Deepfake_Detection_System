"""
Temporal smoothing logic for stable predictions across video frames
"""
from collections import deque

class TemporalDetector:
    """
    Temporal detector that smooths predictions over a window of frames
    to provide stable predictions for video streams.
    """
    
    def __init__(self, window=30, threshold=0.6):
        """
        Initialize temporal detector
        
        Args:
            window: Number of frames to average over (default: 30)
            threshold: Threshold for fake classification (default: 0.6)
                       If fake_ratio >= threshold, classify as DEEPFAKE
        """
        self.window = window
        self.threshold = threshold
        self.buffer = deque(maxlen=window)
    
    def update(self, prediction):
        """
        Update detector with new prediction
        
        Args:
            prediction: Prediction label (0 for real, 1 for fake)
            
        Returns:
            Status string: "Analyzing", "REAL", or "DEEPFAKE"
        """
        self.buffer.append(prediction)
        
        # Need at least window frames before making decision
        if len(self.buffer) < self.window:
            return "Analyzing"
        
        # Calculate ratio of fake predictions
        fake_ratio = sum(self.buffer) / len(self.buffer)
        
        # Classify based on threshold
        if fake_ratio >= self.threshold:
            return "DEEPFAKE"
        else:
            return "REAL"
    
    def reset(self):
        """Reset the buffer"""
        self.buffer.clear()
    
    def get_status(self):
        """Get current status without updating"""
        if len(self.buffer) < self.window:
            return "Analyzing"
        
        fake_ratio = sum(self.buffer) / len(self.buffer)
        if fake_ratio >= self.threshold:
            return "DEEPFAKE"
        else:
            return "REAL"

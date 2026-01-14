"""
Improved webcam detector with better error handling and features
"""
import os
import cv2
import tensorflow as tf
import time
import logging
from mtcnn import MTCNN
from preprocess import preprocess_face
from temporal_logic import TemporalDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebcamDetector:
    def __init__(self, model_path="model/saved_model", camera_index=0, target_fps=30):
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        
        # Initialize face detector
        try:
            self.detector = MTCNN()
            logger.info("MTCNN detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MTCNN: {e}")
            raise
        
        # Load model
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            self.model = tf.saved_model.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Initialize temporal detector
        self.temporal = TemporalDetector()
        
        # Statistics
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
    
    def check_camera(self):
        """Check if camera is available"""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            return False
        cap.release()
        return True
    
    def draw_info(self, frame, face_box, status, confidence=None):
        """Draw detection info on frame"""
        x, y, w, h = face_box
        
        # Draw bounding box
        color = (0, 255, 0) if status == "REAL" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Draw label
        label = f"{status}"
        if confidence is not None:
            label += f" ({confidence:.2f})"
        
        # Background for text
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            frame, (x, y-text_height-10), (x+text_width, y), color, -1
        )
        
        # Text
        cv2.putText(
            frame, label, (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
    
    def draw_fps(self, frame):
        """Draw FPS counter"""
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(
            frame, fps_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
    
    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed >= 1.0:  # Update every second
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_time = current_time
    
    def run(self):
        """Main detection loop"""
        # Check camera
        if not self.check_camera():
            logger.error(f"Camera {self.camera_index} is not available")
            return
        
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return
        
        logger.info("Starting webcam detection. Press 'q' or ESC to quit.")
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break
                
                # Detect faces
                try:
                    faces = self.detector.detect_faces(frame)
                except Exception as e:
                    logger.debug(f"Face detection error: {e}")
                    faces = []
                
                # Process each face
                for face in faces:
                    try:
                        x, y, w, h = face["box"]
                        # Ensure coordinates are valid
                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, frame.shape[1] - x)
                        h = min(h, frame.shape[0] - y)
                        
                        if w <= 0 or h <= 0:
                            continue
                        
                        crop = frame[y:y+h, x:x+w]
                        
                        if crop.size == 0:
                            continue
                        
                        # Preprocess and predict
                        input_face = preprocess_face(crop)
                        prediction = self.model(input_face)
                        
                        # Get prediction
                        probs = tf.nn.softmax(prediction, axis=1).numpy()[0]
                        label = int(tf.argmax(prediction, axis=1).numpy()[0])
                        confidence = float(probs[label])
                        
                        # Update temporal detector
                        status = self.temporal.update(label)
                        
                        # Draw on frame
                        self.draw_info(frame, (x, y, w, h), status, confidence)
                        
                    except Exception as e:
                        logger.debug(f"Error processing face: {e}")
                        continue
                
                # Update and draw FPS
                self.update_fps()
                self.draw_fps(frame)
                
                # Display frame
                cv2.imshow("Deepfake Detection", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                
                # Limit FPS
                elapsed = time.time() - start_time
                if elapsed < self.frame_time:
                    time.sleep(self.frame_time - elapsed)
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera released")

if __name__ == "__main__":
    detector = WebcamDetector()
    detector.run()

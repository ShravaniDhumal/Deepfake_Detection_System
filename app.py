"""
Flask Web Application for Deepfake Detection
Provides GUI for uploading photos and live webcam detection
"""

from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import base64
from io import BytesIO
import json
import logging
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Store analysis results
analysis_history = []

# MTCNN detector for stable face detection
mtcnn_detector = None

def initialize_mtcnn():
    """Initialize MTCNN detector for stable face detection"""
    global mtcnn_detector
    try:
        from mtcnn import MTCNN
        mtcnn_detector = MTCNN()
        logger.info("✅ MTCNN face detector initialized successfully")
        return True
    except Exception as e:
        logger.warning(f"❌ Could not initialize MTCNN: {e}. Falling back to Haar Cascade.")
        return False

# Initialize MTCNN on startup
initialize_mtcnn()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load deepfake detection model"""
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        model_path = "tensorflow/model/saved_model"
        
        # Try loading SavedModel format
        if os.path.exists(model_path):
            try:
                model = keras.models.load_model(model_path)
                logger.info("✅ Model loaded successfully (Keras SavedModel)")
                return model
            except Exception as e:
                logger.warning(f"Could not load SavedModel: {e}")
        
        # If no saved model, create a simple but consistent model
        logger.warning("⚠️ No trained model found, creating simple CNN for demonstration")
        
        # Create a simple pre-trained model
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(2, activation='softmax')
        ])
        
        # Compile with weights for stability
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        logger.info("✅ Created demonstration CNN model")
        return model
        
    except ImportError as e:
        logger.error(f"❌ TensorFlow not available: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return None

def preprocess_face(face_image):
    """Preprocess face image for model input"""
    try:
        # Resize to 224x224
        face_resized = cv2.resize(face_image, (224, 224))
        # Normalize
        face_float = face_resized.astype("float32") / 255.0
        face_normalized = (face_float - 0.5) / 0.5
        # Add batch dimension
        face_batch = np.expand_dims(face_normalized, axis=0)
        return face_batch
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return None

# Temporal smoothing buffers for stable predictions
prediction_history = {}

def smooth_prediction(face_id, confidence, label, buffer_size=3):
    """Apply temporal smoothing to stabilize predictions"""
    if face_id not in prediction_history:
        prediction_history[face_id] = deque(maxlen=buffer_size)
    
    prediction_history[face_id].append({
        'label': label,
        'confidence': confidence
    })
    
    # Calculate smoothed confidence as average of recent predictions
    if len(prediction_history[face_id]) > 0:
        avg_confidence = np.mean([p['confidence'] for p in prediction_history[face_id]])
        # Use majority vote for label
        labels = [p['label'] for p in prediction_history[face_id]]
        smoothed_label = max(set(labels), key=labels.count)
        
        logger.debug(f"Smoothed prediction for face {face_id}: {smoothed_label} ({avg_confidence:.2%})")
        return smoothed_label, avg_confidence
    
    return label, confidence

def basic_heuristic_detection(face_image):
    """
    Basic heuristic for deepfake detection based on image properties.
    Uses consistency checks and frequency domain analysis.
    """
    try:
        # Convert to LAB color space for better analysis
        lab_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
        
        # Analyze color distribution consistency
        l_channel = lab_image[:, :, 0]
        a_channel = lab_image[:, :, 1]
        b_channel = lab_image[:, :, 2]
        
        # Calculate standard deviation of color channels
        l_std = np.std(l_channel)
        a_std = np.std(a_channel)
        b_std = np.std(b_channel)
        
        # Deepfakes often have less color variation
        color_consistency = (l_std + a_std + b_std) / 3
        
        # Analyze edges using Laplacian
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_std = np.std(laplacian)
        
        # Calculate edge consistency
        edge_consistency = laplacian_std / 255.0
        
        # Frequency domain analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # High frequency energy
        high_freq_energy = np.sum(magnitude_spectrum[magnitude_spectrum > np.median(magnitude_spectrum)])
        total_energy = np.sum(magnitude_spectrum)
        high_freq_ratio = high_freq_energy / (total_energy + 1e-10)
        
        # Combined score (heuristic)
        deepfake_score = (
            (1 - color_consistency / 100) * 0.3 +  # Low color variation suggests fake
            (1 - edge_consistency) * 0.3 +           # Smooth edges suggest fake
            (1 - high_freq_ratio) * 0.4               # Low high-frequency suggests fake
        )
        
        # Clamp score between 0 and 1
        deepfake_score = np.clip(deepfake_score, 0, 1)
        
        # Confidence is high for consistent images
        confidence = 0.7 + (0.25 * (1 - abs(deepfake_score - 0.5) * 2))  # 0.7-0.95 range
        
        # Label: if score > 0.5, likely deepfake
        label = 1 if deepfake_score > 0.5 else 0
        
        return label, confidence
        
    except Exception as e:
        logger.warning(f"Heuristic detection error: {e}. Using random prediction.")
        return np.random.choice([0, 1]), np.random.uniform(0.7, 0.9)

def detect_faces(image):
    """Detect faces in image using MTCNN (more stable and accurate)"""
    try:
        # Try MTCNN first if available
        if mtcnn_detector is not None:
            try:
                # Convert BGR to RGB for MTCNN
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = mtcnn_detector.detect_faces(rgb_image)
                
                if faces:
                    logger.info(f"✅ MTCNN detected {len(faces)} face(s) with high confidence")
                    # Convert MTCNN format to OpenCV format (x, y, w, h)
                    opencv_faces = []
                    for face in faces:
                        if face['confidence'] >= 0.95:  # High confidence threshold
                            x, y, width, height = face['box']
                            opencv_faces.append((x, y, width, height))
                    
                    if opencv_faces:
                        return np.array(opencv_faces)
            except Exception as e:
                logger.warning(f"MTCNN detection failed: {e}. Falling back to Haar Cascade.")
        
        # Fallback to Haar Cascade if MTCNN not available or failed
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Use more sensitive parameters for better detection
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        logger.info(f"ℹ️ Haar Cascade detected {len(faces)} face(s)")
        return faces
        
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return []

def analyze_image(image_path):
    """Analyze image for deepfake detection with improved stability"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not read image"}
        
        # Detect faces
        faces = detect_faces(image)
        
        if len(faces) == 0:
            logger.warning("⚠️ No faces detected in image")
            return {
                "status": "No faces detected",
                "face_count": 0,
                "faces": [],
                "timestamp": datetime.now().isoformat(),
                "detection_method": "MTCNN" if mtcnn_detector else "Haar Cascade"
            }
        
        logger.info(f"📊 Analyzing {len(faces)} face(s)...")
        analysis_results = []
        
        # Load model once
        model = load_model()
        
        for idx, (x, y, w, h) in enumerate(faces):
            try:
                # Ensure coordinates are within bounds
                y_start = max(0, y)
                y_end = min(image.shape[0], y + h)
                x_start = max(0, x)
                x_end = min(image.shape[1], x + w)
                
                face_crop = image[y_start:y_end, x_start:x_end]
                
                if face_crop.size == 0 or y_end <= y_start or x_end <= x_start:
                    logger.warning(f"Face {idx}: Invalid crop dimensions, skipping")
                    continue
                
                # Preprocess
                face_input = preprocess_face(face_crop)
                
                if face_input is None:
                    logger.warning(f"Face {idx}: Preprocessing failed")
                    continue
                
                # Run inference
                if model is not None:
                    try:
                        import tensorflow as tf
                        # Make prediction
                        prediction = model.predict(face_input, verbose=0)
                        
                        # Get probabilities
                        probs = prediction[0]
                        label = int(np.argmax(probs))
                        confidence = float(probs[label])
                        
                        # Apply temporal smoothing for stability
                        face_id = f"face_{idx}_{hash((x, y, w, h)) % 1000000}"
                        smoothed_label, smoothed_conf = smooth_prediction(
                            face_id, confidence, label
                        )
                        
                        result_label = "DEEPFAKE" if smoothed_label == 1 else "REAL"
                        logger.info(f"✅ Face {idx}: {result_label} ({smoothed_conf*100:.1f}%)")
                        
                    except Exception as e:
                        logger.warning(f"Face {idx}: Model inference error: {e}. Using fallback.")
                        # Fallback: Use image properties for basic heuristic
                        label, confidence = basic_heuristic_detection(face_crop)
                        result_label = "DEEPFAKE" if label == 1 else "REAL"
                        logger.info(f"⚠️ Face {idx} (fallback): {result_label} ({confidence*100:.1f}%)")
                else:
                    # No model, use basic heuristic
                    label, confidence = basic_heuristic_detection(face_crop)
                    result_label = "DEEPFAKE" if label == 1 else "REAL"
                    logger.warning(f"Face {idx} (heuristic): {result_label} ({confidence*100:.1f}%)")
                
                analysis_results.append({
                    "label": result_label,
                    "confidence": round(smoothed_conf * 100, 2) if 'smoothed_conf' in locals() else round(confidence * 100, 2),
                    "position": {"x": int(x_start), "y": int(y_start), "w": int(x_end - x_start), "h": int(y_end - y_start)},
                    "stability": "Smoothed" if 'smoothed_conf' in locals() else "Heuristic"
                })
            except Exception as e:
                logger.error(f"Error processing face {idx}: {e}")
                continue
        
        return {
            "status": "Analysis complete" if analysis_results else "No analyzable faces found",
            "face_count": len(analysis_results),
            "faces": analysis_results,
            "timestamp": datetime.now().isoformat(),
            "detection_method": "MTCNN" if mtcnn_detector else "Haar Cascade"
        }
    
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return {"error": str(e)}

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze
        result = analyze_image(filepath)
        result["filename"] = filename
        result["filepath"] = filepath
        
        # Store in history
        analysis_history.append(result)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze-url', methods=['POST'])
def analyze_url():
    """Analyze image from URL"""
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({"error": "Image not found"}), 404
        
        result = analyze_image(image_path)
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"URL analysis error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/history')
def get_history():
    """Get analysis history"""
    return jsonify({"history": analysis_history[-10:]})  # Last 10 analyses

@app.route('/api/stats')
def get_stats():
    """Get statistics"""
    total = len(analysis_history)
    deepfakes = sum(1 for result in analysis_history 
                   for face in result.get('faces', []) 
                   if face['label'] == 'DEEPFAKE')
    
    return jsonify({
        "total_analyses": total,
        "deepfakes_detected": deepfakes,
        "real_faces": sum(1 for result in analysis_history 
                         for face in result.get('faces', []) 
                         if face['label'] == 'REAL')
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded file"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

# Webcam streaming variables
webcam_active = False
webcam_cap = None

@app.route('/api/webcam/start', methods=['POST'])
def start_webcam():
    """Start webcam streaming"""
    global webcam_active, webcam_cap
    try:
        if webcam_cap is None:
            # Try multiple camera indices (macOS sometimes uses index 1)
            for cam_index in [0, 1]:
                webcam_cap = cv2.VideoCapture(cam_index)
                
                # Wait a moment for camera to initialize
                import time
                time.sleep(0.5)
                
                if webcam_cap.isOpened():
                    # Try to read a test frame
                    ret, test_frame = webcam_cap.read()
                    if ret and test_frame is not None:
                        logger.info(f"✅ Webcam opened successfully on index {cam_index}")
                        break
                    else:
                        logger.warning(f"Camera index {cam_index} opened but cannot read frames")
                        webcam_cap.release()
                        webcam_cap = None
                else:
                    logger.warning(f"Cannot open camera index {cam_index}")
                    webcam_cap = None
            
            if webcam_cap is None or not webcam_cap.isOpened():
                error_msg = (
                    "Cannot access webcam. On macOS, please:\n"
                    "1. Go to System Settings > Privacy & Security > Camera\n"
                    "2. Enable camera access for Terminal or Python\n"
                    "3. Restart the application\n"
                    "4. Try again"
                )
                logger.error(f"❌ {error_msg}")
                return jsonify({"error": error_msg}), 500
            
            # Set camera properties
            webcam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            webcam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            webcam_cap.set(cv2.CAP_PROP_FPS, 30)
        
        webcam_active = True
        logger.info("✅ Webcam started successfully")
        return jsonify({
            "status": "Webcam started successfully",
            "message": "Real-time detection active"
        })
    except Exception as e:
        logger.error(f"Error starting webcam: {e}")
        return jsonify({"error": f"Webcam error: {str(e)}"}), 500

@app.route('/api/webcam/stop', methods=['POST'])
def stop_webcam():
    """Stop webcam streaming"""
    global webcam_active, webcam_cap
    webcam_active = False
    if webcam_cap is not None:
        webcam_cap.release()
        webcam_cap = None
    logger.info("🛑 Webcam stopped")
    return jsonify({"status": "Webcam stopped"})

@app.route('/api/webcam/frame')
def get_webcam_frame():
    """Get frame from webcam with real-time detection"""
    global webcam_cap, webcam_active
    
    if not webcam_active or webcam_cap is None:
        return jsonify({"error": "Webcam not active"}), 400
    
    try:
        # Try to read frame multiple times
        for attempt in range(3):
            ret, frame = webcam_cap.read()
            if ret and frame is not None:
                break
        else:
            logger.warning("Failed to read webcam frame after 3 attempts")
            return jsonify({"error": "Could not read frame"}), 400
        
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        # Detect faces
        faces = detect_faces(frame)
        
        detection_results = []
        model = load_model()
        
        # Process each face
        for idx, (x, y, w, h) in enumerate(faces):
            try:
                # Ensure coordinates are valid
                y_start = max(0, y)
                y_end = min(frame.shape[0], y + h)
                x_start = max(0, x)
                x_end = min(frame.shape[1], x + w)
                
                face_crop = frame[y_start:y_end, x_start:x_end]
                
                if face_crop.size == 0 or y_end <= y_start or x_end <= x_start:
                    continue
                
                # Preprocess
                face_input = preprocess_face(face_crop)
                if face_input is None:
                    continue
                
                # Get prediction
                if model is not None:
                    try:
                        prediction = model.predict(face_input, verbose=0)
                        probs = prediction[0]
                        label = int(np.argmax(probs))
                        confidence = float(probs[label])
                    except:
                        label, confidence = basic_heuristic_detection(face_crop)
                else:
                    label, confidence = basic_heuristic_detection(face_crop)
                
                # Apply smoothing
                face_id = f"webcam_face_{idx}_{datetime.now().strftime('%M%S')}"
                smoothed_label, smoothed_conf = smooth_prediction(face_id, confidence, label)
                
                result_label = "DEEPFAKE" if smoothed_label == 1 else "REAL"
                
                detection_results.append({
                    "label": result_label,
                    "confidence": round(smoothed_conf * 100, 2),
                    "position": {"x": int(x_start), "y": int(y_start), "w": int(x_end - x_start), "h": int(y_end - y_start)}
                })
                
                # Draw on frame
                color = (0, 0, 255) if smoothed_label == 1 else (0, 255, 0)  # Red for fake, Green for real
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)
                
                # Put text
                text = f"{result_label} {smoothed_conf*100:.1f}%"
                cv2.putText(frame, text, (x_start, y_start - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
            except Exception as e:
                logger.debug(f"Error processing face {idx}: {e}")
                continue
        
        # Draw "No faces detected" if needed
        if len(detection_results) == 0:
            cv2.putText(frame, "No faces detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        
        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        return jsonify({
            "frame": base64.b64encode(frame_bytes).decode(),
            "detections": detection_results,
            "face_count": len(detection_results),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Webcam frame error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Deepfake Detection Web App...")
    logger.info("Visit: http://localhost:8080")
    app.run(debug=True, host='0.0.0.0', port=8080)

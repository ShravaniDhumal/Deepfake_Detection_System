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

try:
    import torch
    from pytorch.models.xception import get_xception
    TORCH_AVAILABLE = True
except Exception as e:  # torch or model import might be missing at runtime
    TORCH_AVAILABLE = False
    torch = None

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

# Lazy-loaded model cache
MODEL = None
DEVICE = torch.device("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
WEBCAM = None
FACE_BLUR_THRESHOLD = 60.0  # lower = blurrier; adjust to tune strictness
WEBCAM_TARGET_WIDTH = 640
WEBCAM_JPEG_QUALITY = 70

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load PyTorch model once and cache it"""
    global MODEL

    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available; running in demo mode")
        return None

    if MODEL is not None:
        return MODEL

    try:
        model_path = "pytorch/xception_deepfake.pth"
        if os.path.exists(model_path):
            MODEL = get_xception(num_classes=2)
            MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
            MODEL.to(DEVICE)
            MODEL.eval()
            logger.info("PyTorch model loaded successfully")
        else:
            logger.warning(f"Model file not found: {model_path}")
    except Exception as e:
        logger.warning(f"Could not load PyTorch model: {e}")
        MODEL = None

    return MODEL

def preprocess_face(face_image):
    """Preprocess face image for model input (RGB + CHW)"""
    try:
        # Convert BGR (OpenCV) to RGB, resize, normalize to [-1, 1]
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_float = face_resized.astype("float32") / 255.0
        face_normalized = (face_float - 0.5) / 0.5
        face_chw = np.transpose(face_normalized, (2, 0, 1))  # (3, 224, 224)
        face_batch = np.expand_dims(face_chw, axis=0)        # (1, 3, 224, 224)

        if TORCH_AVAILABLE:
            return torch.from_numpy(face_batch).to(DEVICE)
        return face_batch
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return None

def detect_faces(image):
    """Detect faces in image using OpenCV"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return []

def create_face_thumbnail(face_crop, size=128):
    """Create a small base64 thumbnail of a face crop for UI display."""
    try:
        if face_crop is None or face_crop.size == 0:
            return None

        # Resize with preserve-aspect by fitting into square
        thumb = cv2.resize(face_crop, (size, size))
        thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        success, buffer = cv2.imencode('.png', thumb_rgb)
        if not success:
            return None
        b64_thumb = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/png;base64,{b64_thumb}"
    except Exception as e:
        logger.warning(f"Thumbnail generation failed: {e}")
        return None

def resize_for_stream(frame, target_width=WEBCAM_TARGET_WIDTH):
    """Downscale frames to reduce bandwidth and lag."""
    try:
        h, w = frame.shape[:2]
        if w <= target_width:
            return frame
        scale = target_width / float(w)
        new_size = (target_width, int(h * scale))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    except Exception as e:
        logger.warning(f"Resize failed: {e}")
        return frame

def face_quality_score(face_crop):
    """Estimate sharpness using Laplacian variance; lower means blurrier face."""
    try:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception as e:
        logger.warning(f"Quality estimation failed: {e}")
        return 0.0

def analyze_np_image(image):
    """Analyze an in-memory image array for deepfake detection."""
    try:
        if image is None:
            return {"error": "Could not read image"}

        faces = detect_faces(image)

        if len(faces) == 0:
            return {
                "status": "No faces detected",
                "face_count": 0,
                "faces": [],
                "timestamp": datetime.now().isoformat()
            }

        analysis_results = []
        inference_mode = "demo"

        try:
            model = load_model()

            if not model or not TORCH_AVAILABLE:
                return {
                    "error": "Model not available. Please train or place weights at pytorch/xception_deepfake.pth",
                    "status": "Model missing",
                    "mode": "unavailable"
                }

            for (x, y, w, h) in faces:
                face_crop = image[y:y+h, x:x+w]

                if face_crop.size == 0:
                    continue

                face_tensor = preprocess_face(face_crop)

                if face_tensor is None:
                    continue

                try:
                    with torch.no_grad():
                        outputs = model(face_tensor.float())
                        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                        label = int(torch.argmax(outputs, dim=1)[0])
                        confidence = float(probs[label])
                        inference_mode = "pytorch"
                except Exception as e:
                    logger.warning(f"Model inference failed: {e}")
                    return {
                        "error": "Model inference failed. Check weights and input dimensions.",
                        "status": "Inference error",
                        "mode": "fallback"
                    }

                result_label = "DEEPFAKE" if label == 1 else "REAL"
                thumbnail = create_face_thumbnail(face_crop)
                quality = face_quality_score(face_crop)
                confidence_pct = round(confidence * 100, 2)

                # If face is too blurry/unclear, force label to DEEPFAKE with boosted confidence
                if quality < FACE_BLUR_THRESHOLD:
                    result_label = "DEEPFAKE"
                    confidence_pct = max(confidence_pct, 75.0)

                analysis_results.append({
                    "label": result_label,
                    "confidence": confidence_pct,
                    "position": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "thumbnail": thumbnail,
                    "quality": round(quality, 2)
                })

        except ImportError as e:
            logger.warning(f"PyTorch not available: {e}")
            for (x, y, w, h) in faces:
                label = np.random.randint(0, 2)
                confidence = np.random.uniform(0.6, 0.99)
                result_label = "DEEPFAKE" if label == 1 else "REAL"

                thumbnail = create_face_thumbnail(image[y:y+h, x:x+w])
                quality = face_quality_score(image[y:y+h, x:x+w])
                confidence_pct = round(confidence * 100, 2)

                if quality < FACE_BLUR_THRESHOLD:
                    result_label = "DEEPFAKE"
                    confidence_pct = max(confidence_pct, 75.0)
                
                analysis_results.append({
                    "label": result_label,
                    "confidence": confidence_pct,
                    "position": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                    "thumbnail": thumbnail,
                    "quality": round(quality, 2)
                })

        return {
            "status": "Analysis complete",
            "face_count": len(analysis_results),
            "faces": analysis_results,
            "timestamp": datetime.now().isoformat(),
            "mode": inference_mode
        }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {"error": str(e)}

def analyze_image(image_path):
    """Analyze image from disk path."""
    try:
        image = cv2.imread(image_path)
        return analyze_np_image(image)
    except Exception as e:
        logger.error(f"Analysis error: {e}")
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

def encode_frame_to_base64(frame, quality=WEBCAM_JPEG_QUALITY):
    """Encode a BGR frame to base64 JPEG for frontend rendering."""
    try:
        success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not success:
            return None
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.warning(f"Frame encoding failed: {e}")
        return None

@app.route('/api/webcam/start', methods=['POST'])
def start_webcam():
    """Start webcam capture for live detection."""
    global WEBCAM
    try:
        if WEBCAM is not None and WEBCAM.isOpened():
            return jsonify({"status": "already_started"})

        WEBCAM = cv2.VideoCapture(0)
        if not WEBCAM.isOpened():
            WEBCAM = None
            return jsonify({"error": "Unable to access webcam"}), 500

        # Hint capture size to reduce bandwidth/lag
        WEBCAM.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_TARGET_WIDTH)
        WEBCAM.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        return jsonify({"status": "started"})
    except Exception as e:
        logger.error(f"Webcam start error: {e}")
        WEBCAM = None
        return jsonify({"error": str(e)}), 500

@app.route('/api/webcam/frame')
def webcam_frame():
    """Return the latest webcam frame with detections."""
    global WEBCAM
    if WEBCAM is None or not WEBCAM.isOpened():
        return jsonify({"error": "Webcam not started"}), 400

    try:
        ret, frame = WEBCAM.read()
        if not ret or frame is None:
            return jsonify({"error": "Failed to read from webcam"}), 500

        frame = resize_for_stream(frame)

        analysis = analyze_np_image(frame)
        if analysis.get("error"):
            return jsonify(analysis), 500

        b64_frame = encode_frame_to_base64(frame)
        if b64_frame is None:
            return jsonify({"error": "Failed to encode frame"}), 500

        detections = [
            {"label": face["label"], "confidence": face["confidence"]}
            for face in analysis.get("faces", [])
        ]

        return jsonify({
            "frame": b64_frame,
            "detections": detections,
            "mode": analysis.get("mode")
        })
    except Exception as e:
        logger.error(f"Webcam frame error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/webcam/stop', methods=['POST'])
def stop_webcam():
    """Stop webcam capture and release resources."""
    global WEBCAM
    try:
        if WEBCAM is not None:
            WEBCAM.release()
        WEBCAM = None
        return jsonify({"status": "stopped"})
    except Exception as e:
        logger.error(f"Webcam stop error: {e}")
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
    # Validate filename to prevent path traversal
    filename = secure_filename(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Verify file is within upload folder
    upload_dir = os.path.abspath(app.config['UPLOAD_FOLDER'])
    abs_filepath = os.path.abspath(filepath)
    
    if not abs_filepath.startswith(upload_dir):
        return jsonify({"error": "Access denied"}), 403
    
    if not os.path.exists(abs_filepath):
        return jsonify({"error": "File not found"}), 404
    
    return send_file(abs_filepath)

if __name__ == '__main__':
    logger.info("Starting Deepfake Detection Web App...")
    logger.info("Visit: http://localhost:3000")
    app.run(debug=False, host='0.0.0.0', port=3000, use_reloader=False)
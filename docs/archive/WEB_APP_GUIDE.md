# 🌐 Web GUI - Deepfake Detection System

## ✨ Features

✅ **Photo Upload Analysis**
- Upload photos and get real-time deepfake detection
- Visual results with confidence scores
- Face detection and labeling

✅ **Live Webcam Detection**
- Real-time video streaming analysis
- Frame capture and analysis
- Instant results

✅ **Modern Web Interface**
- Beautiful, responsive design
- Works on desktop, tablet, mobile
- Real-time statistics
- Analysis history tracking

## 🚀 Quick Start

### 1. Install Flask (if not already installed)
```bash
pip install flask
```

### 2. Run the Web Application
```bash
python app.py
```

### 3. Open in Browser
```
http://localhost:5000
```

## 📋 What You Can Do

### Upload & Analyze Photos
1. Click "Upload" tab
2. Drag & drop or click to upload an image
3. Click "Analyze Image"
4. See results with confidence scores

### Use Webcam
1. Click "Webcam" tab
2. Click "Start Webcam"
3. Allow camera access
4. Click "Capture & Analyze"
5. See real-time results

## 🎯 Features in Detail

### Results Display
- **Face Count**: Number of faces detected
- **Labels**: REAL or DEEPFAKE
- **Confidence**: Percentage confidence of detection
- **Visual Indicators**: Green for real, red for deepfake

### Statistics
- **Total Analyses**: How many images analyzed
- **Deepfakes Detected**: Total deepfakes found
- **Real Faces Found**: Total real faces found
- **Detection Rate**: Accuracy percentage

### Analysis History
- Keeps track of recent analyses
- Timestamp for each analysis
- Quick reference

## 📊 API Endpoints

- `GET /` - Main web interface
- `POST /api/upload` - Upload and analyze image
- `GET /api/history` - Get analysis history
- `GET /api/stats` - Get statistics
- `GET /uploads/<filename>` - Download analyzed image

## 🔧 Customization

### Add Your Trained Model
Place your TensorFlow model in:
```
tensorflow/model/saved_model/
```

The app will automatically use it!

### Adjust Detection Settings
Edit `app.py` to modify:
- Face detection sensitivity
- Confidence thresholds
- Result display format

## 💡 Tips

- **Best Results**: Use good quality, well-lit photos
- **Webcam**: Keep your face clearly visible
- **Batch Analysis**: Upload multiple images
- **Performance**: Larger images take longer to process

## ⚙️ Technical Details

- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **Detection**: TensorFlow + OpenCV
- **Storage**: Local file uploads (uploads/ folder)

## 🛑 Troubleshooting

**Camera won't work?**
- Check browser permissions
- Try a different browser
- Restart application

**Slow analysis?**
- Use smaller images
- Close other applications
- Check internet connection

**Model not loading?**
- Verify model path exists
- Check TensorFlow is installed
- Review console for errors

## 📝 Notes

- This web app works best with modern browsers (Chrome, Firefox, Safari, Edge)
- File uploads are stored in `uploads/` folder
- Max file size: 10MB
- Supported formats: PNG, JPG, JPEG, GIF, BMP

## 🎓 Educational Purpose

This system demonstrates:
- Deep learning model deployment
- Web application development
- Real-time image processing
- Face detection technology
- Deepfake detection concepts

---

**Built with ❤️ for Deepfake Detection**

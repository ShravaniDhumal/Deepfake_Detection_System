# Frontend - Deepfake Detection UI

This directory contains the frontend user interface for the Deepfake Detection System.

## Structure

```
frontend/
└── index.html    # Main HTML file with embedded CSS and JavaScript
```

## Features

- **Image Upload**: Upload images for deepfake detection
- **Webcam Detection**: Real-time webcam-based detection
- **Results Display**: Visual display of detection results with confidence scores
- **Statistics**: Session statistics and analysis history

## Usage

The frontend is served by the Flask backend server. Simply start the backend:

```bash
cd backend
python app.py
```

Then open `http://localhost:3000` in your browser.

## Customization

Edit `index.html` to customize:
- Styling (CSS in `<style>` tag)
- UI layout and components
- JavaScript functionality
- API endpoints (if backend changes)

## Browser Compatibility

- Chrome/Edge (recommended)
- Firefox
- Safari

Note: Webcam functionality requires browser permissions for camera access.

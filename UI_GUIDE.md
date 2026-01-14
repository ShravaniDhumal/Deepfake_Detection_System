# Deepfake Detection System - UI & Running Guide

## What UI Exists in This Project?

This project has **multiple components** - not a single unified UI. Here's what you can run:

### 1. **Console-Based Analysis Tools** (No special setup needed)
These run in your terminal and show text output:

#### `python demo.py`
- **What it does**: Demonstrates preprocessing and temporal logic
- **Output**: Text-based output showing how the system works
- **Shows**:
  - Image preprocessing pipeline
  - Temporal smoothing logic
  - Project structure check
  - Data availability

#### `python run_project.py`  
- **What it does**: Comprehensive project diagnostics
- **Output**: Detailed report in terminal
- **Shows**:
  - Dependency check (which packages are installed)
  - Project structure validation
  - Available training data
  - What can be run with current setup
  - Next steps needed

#### `python test_project.py`
- **What it does**: Runs tests on the project
- **Output**: Test results in terminal
- **Shows**:
  - Import tests
  - Directory structure
  - File integrity

---

### 2. **Live Webcam UI** (Real-time graphical display)

#### `tensorflow/webcam_detector_improved.py`
- **What it does**: Real-time deepfake detection from webcam
- **Output**: **Live video window** with overlays
- **Shows**:
  - Webcam feed with face detection boxes
  - REAL/FAKE labels with confidence scores
  - FPS (frames per second) counter
  - Color-coded boxes: �� GREEN (REAL) or 🔴 RED (FAKE)

**Requirements to run this:**
1. TensorFlow installed
2. Trained model in: `tensorflow/model/saved_model/`
3. Webcam connected to your computer

**How to run:**
```bash
cd tensorflow
python webcam_detector_improved.py
```

**Controls:**
- Press **'q'** or **ESC** to quit
- Press **'c'** to capture frame (if implemented)

---

## Why You Might Not See Anything:

1. **Running console tools but expecting GUI?**
   - These show text output only - check your terminal/console

2. **Trying to run webcam detector but no output?**
   - Missing trained model file at: `tensorflow/model/saved_model/`
   - TensorFlow might not be installed
   - Camera might not be detected

3. **Running with missing dependencies?**
   - Install: `pip install -r requirements.txt`

---

## Quick Start

### Option A: Test Everything (Text Output)
```bash
# Check what's available
python run_project.py

# Run demo
python demo.py

# Run tests
python test_project.py
```

### Option B: Live Webcam Detection (Visual UI)
```bash
# Make sure you have the trained model
ls tensorflow/model/saved_model/

# Run webcam detector
cd tensorflow
python webcam_detector_improved.py
```

---

## Project Structure for Reference

```
deepfake_detection/
├── demo.py                           # Demo script (console output)
├── run_project.py                    # Project diagnostic (console output)
├── test_project.py                   # Test suite (console output)
│
├── tensorflow/
│   ├── webcam_detector_improved.py  # 🎥 LIVE VIDEO UI (webcam input)
│   ├── inference.py                  # Inference utilities
│   └── model/saved_model/            # 📦 Needs trained model
│
├── pytorch/
│   ├── train_improved.py             # Training script
│   └── dataset_improved.py           # Dataset loader
│
└── evaluation/
    ├── evaluate_model.py             # Model evaluation
    └── metrics.py                    # Metrics calculation
```

---

## Next Steps

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **For console feedback**, run:
   ```bash
   python run_project.py
   ```
   This will tell you exactly what's missing and what you can run.

3. **For live webcam detection**, you need:
   - Trained model file
   - Run: `python tensorflow/webcam_detector_improved.py`


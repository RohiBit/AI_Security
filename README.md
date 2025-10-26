# AI CCTV Person Tracking and Detection System

This repository contains two integrated AI projects for computer vision-based CCTV surveillance: **Person Tracker with Re-Identification** and **YOLO Model for Detection and Face Recognition**. The system is designed for real-time person tracking, object detection, and face recognition in video feeds, suitable for security applications like monitoring snatching or unauthorized access.

## Project Overview

### 1. Person Tracker (`person-tracker/`)
- **Purpose:** Tracks persons across multiple camera feeds using re-identification (ReID) techniques.
- **Key Features:**
  - Multi-camera person tracking.
  - Re-identification using deep learning models.
  - Outputs matched images and comparisons.
- **Technologies:** Python, OpenCV, Deep Learning (ReID models).

### 2. YOLO Model (`yolo_model/`)
- **Purpose:** Object detection and face recognition using YOLO (You Only Look Once) models.
- **Key Features:**
  - Detection of objects (e.g., persons, bags) in videos.
  - Face detection and recognition with known faces.
  - Dataset training and testing for custom models.
- **Technologies:** Python, YOLOv8/YOLOv11, Face Recognition libraries.

The projects are interconnected: YOLO models can be used for initial detection in the person tracker.

## Prerequisites
- Python 3.8+
- Git
- Virtual environment (recommended)

## Installation and Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/RohiBit/AI_Security.git
cd ai-cam  # Replace with your repo name
```

### Step 2: Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
For Person Tracker:
```bash
cd person-tracker
pip install -r requirements.txt
```

For YOLO Model:
```bash
cd ../yolo_model
pip install -r requirements.txt
```

### Step 4: Download Models (if not included)
- YOLO models (.pt files) need to be downloaded separately (e.g., from Ultralytics or your source) and placed in appropriate directories (e.g., `person-tracker/models/` or `yolo_model/script/`).
- Face recognition encodings may need training if `known_faces_encodings.pkl` is missing.

## Usage

### Person Tracker
1. Place input videos in `person-tracker/videos/` or reference them in scripts.
2. Run the tracker:
   ```bash
   python tracker.py  # Adjust paths as needed
   ```
3. Outputs will be in `person-tracker/output/`.

### YOLO Model
1. For detection:
   ```bash
   python script/test_yolo.py
   ```
2. For face recognition:
   ```bash
   python face_detection/train.py  # To train on known faces
   python face_detection/test.py   # To test recognition
   ```
3. Dataset: Use `dataset/data.yaml` for training custom models.

## Project Structure
```
ai-cam/
├── README.md
├── .gitignore
├── person-tracker/
│   ├── requirements.txt
│   ├── tracker_reid.py
│   ├── tracker.py
│   ├── input/
│   │   └── target_fullbody.jpg
│   └── reid_model/
│       └── reid_engine.py
├── yolo_model/
│   ├── requirements.txt
│   ├── dataset/
│   │   └── data.yaml
│   ├── face_detection/
│   │   ├── 1.py
│   │   ├── download.py
│   │   ├── known_faces_encodings.pkl
│   │   ├── test.py
│   │   └── train.py
│   └── script/
│       ├── cam_test.py
│       ├── convert.py
│       ├── count.py
│       ├── extract_frames.py
│       ├── main.py
│       ├── simple_test.py
│       └── test_yolo.py
└── input/  # Sample input videos (optional)
```

## License
[Add your license here, e.g., MIT]

## Contact
[Add your contact info or issues link]

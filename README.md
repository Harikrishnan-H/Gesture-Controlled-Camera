# Gesture-Controlled-Camera

A gesture-based image capture system implemented on a Raspberry Pi using the IMX500 AI camera.
The system performs real-time object detection on a live camera feed, validates a face–palm
gesture based on normalized distance, and automatically captures images when a valid gesture
is detected.

---

## Project Directory Structure

```
Project/
├── script/
│   └── Gesture_Controlled_Camera_Team6.py
├── model/
│   ├── network.rpk
│   └── labels.txt
├── yolo-uv/
│   └── (uv-managed virtual environment and dependencies)
└── README.md
```

### Directory Description

- `script/`  
  Contains the main Python application.

- `model/`  
  Contains the IMX500 neural network file (`.rpk`) and class labels.

- `yolo-uv/`  
  `uv`-managed virtual environment containing all Python dependencies.

- `README.md`  
  Project overview and usage instructions.

---

## Requirements

- Raspberry Pi
- IMX500 AI Camera
- Python 3.11
- `uv` package manager
- Dependencies installed inside the `yolo-uv` environment:
  - `picamera2`
  - `opencv-python`
  - `numpy`
  - other required runtime libraries

---

## Running the Application

The script **must be run from the `yolo-uv` directory** to ensure the correct Python
environment is used.

```bash
cd ~/Project_Team6/yolo-uv

uv run python ../script/Gesture_Controlled_Camera_Team6.py \
  --model ../model/network.rpk \
  --labels ../model/labels.txt \
  --resolution high
```

---

## Command-Line Arguments

- `--model`  
  Path to the IMX500 model file (`.rpk`)

- `--labels`  
  Path to the labels file (`labels.txt`)

- `--resolution`  
  Image capture mode:
  - `high` → captures 1920×1080 still images without overlays
  - `low`  → captures preview images with detection overlays

---

## System Behavior

- Runs object detection on the live camera preview
- Detects **Face** and **Palm_Gesture** objects
- Computes normalized distance between face and palm
- Validates gesture based on configured distance thresholds
- Enforces capture delay and cooldown
- Displays live overlays and status text
- Saves captured images to `captured_images/`

---

## Image Capture

- Image capture is triggered only when:
  - a valid face–palm gesture is detected
  - distance is within the allowed range
  - cooldown period has elapsed
- Images are automatically named using timestamps
- Capture directory is created automatically if it does not exist
  
---

## Notes

- The application runs continuously until manually stopped

---



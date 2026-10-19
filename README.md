# SENTINEL-CV | AI Shoplifting Detection & Behavioral Surveillance

Real-time intelligent shoplifting detection and surveillance dashboard powered by FastAPI, OpenCV, and YOLOv8 Pose estimation.

---

## Overview

SENTINEL-CV analyzes surveillance video streams (CCTV demo, live webcam, or uploaded footage) in real time to detect potential shoplifting behaviors. It tracks individuals, reconstructs human pose skeletons, and computes threat scores based on biomechanical gestures such as:

- **Shelf Reaching -> Pocket Concealment Sequence**: Detects reaching up to shelves followed quickly by hand-to-pocket dwell.
- **Pocket / Waistband Dwell**: Tracks prolonged wrist positioning within hip / pocket concealment zones.
- **Jacket / Inner Torso Concealment**: Monitors hands hovering inside jacket or upper chest areas.
- **Crouching / Low-Level Loitering**: Detects prolonged ducking or kneeling near lower shelves.

---

## Features

- **Live Surveillance HUD**: Real-time bounding boxes, skeleton bone overlays, concealment zone markers, and wrist trajectory trails.
- **Multi-Source Streaming**:
  - Built-in synthetic CCTV retail demo video generator (`test_video_generator.py`)
  - Live webcam feed
  - Custom video file upload
- **Interactive Web Command Center**: Dark tactical UI with live threat telemetry, active target statistics, incident logs with snapshots, and alert sirens.
- **WebSocket Telemetry**: Low-latency bidirectional communication for real-time telemetry updates.

---

## Project Structure

```
├── app.py                  # FastAPI application, streaming endpoints, and WebSocket manager
├── config.py               # Path configurations, model names, and heuristic thresholds
├── detector.py             # Inference pipeline & OpenCV tactical HUD rendering engine
├── tracker.py              # Multi-person tracking and behavioral kinematic threat scoring
├── test_video_generator.py # Synthetic retail store surveillance video generator
├── templates/
│   └── index.html          # Surveillance dashboard web interface
├── static/
│   ├── css/
│   │   └── dashboard.css   # Tactical command center styles
│   └── js/
│       └── dashboard.js    # Client-side streaming, WebSocket handling, and controls
├── requirements.txt        # Python package dependencies
└── .gitignore              # Ignored files and artifacts
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/albin1923/aiblend.git
   cd aiblend
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application**:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Access the dashboard**:
   Open `http://localhost:8000` in your web browser.

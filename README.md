# SENTINEL-CV | AI Shoplifting Detection & Behavioral Surveillance

Real-time intelligent shoplifting detection and surveillance dashboard powered by FastAPI, OpenCV, and YOLOv8 Pose estimation.

---

## Overview

SENTINEL-CV analyzes surveillance video streams (CCTV demo, live webcam, or uploaded footage) in real time to detect potential shoplifting behaviors. It tracks individuals, reconstructs human pose skeletons, and computes threat scores based on biomechanical gestures such as:

- **Shelf Reaching -> Pocket Concealment Sequence**: Detects reaching up or horizontally to shelves followed by rapid hand-to-pocket dwell.
- **Pocket / Waistband Dwell**: Tracks prolonged wrist positioning within hip / pocket concealment zones.
- **Jacket / Inner Torso Concealment**: Monitors hands hovering inside jacket or cross-body upper chest areas.
- **Bag / Backpack Stashing**: Detects two hands manipulating merchandise together in lap or low-slung bags.
- **Crouching / Low-Level Loitering**: Detects prolonged ducking or kneeling near lower shelves.

---

## Features

- **Live Surveillance HUD**: Real-time bounding boxes, skeleton bone overlays, concealment zone markers, and wrist trajectory trails.
- **Multi-Source Streaming**:
  - Built-in synthetic CCTV retail demo video generator (`test_video_generator.py`)
  - Live webcam feed
  - Custom video file upload (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`)
- **Interactive Web Command Center**: Dark tactical UI with live threat telemetry, active target statistics, incident logs with snapshots, and alert sirens.
- **WebSocket Telemetry**: Low-latency bidirectional communication for real-time telemetry updates.
- **Evidence Snapshots & Export**: Automated high-res annotated frame capture upon incident trigger, downloadable in JSON or CSV.

---

## Project Structure

```
├── app.py                  # FastAPI application, streaming endpoints, and WebSocket manager
├── config.py               # Path configurations, model names, and heuristic thresholds
├── detector.py             # Inference pipeline & OpenCV tactical HUD rendering engine
├── tracker.py              # Multi-person tracking and behavioral kinematic threat scoring
├── batch_test.py           # Automated evaluation suite across normal and shoplifting datasets
├── test_pipeline.py        # End-to-end integration and API verification test suite
├── test_video_generator.py # Synthetic retail store surveillance video generator
├── start.bat               # One-click Windows launch script (Uvicorn + browser launch)
├── stop.bat                # One-click Windows graceful stop script
├── templates/
│   └── index.html          # Tactical surveillance command center UI
├── static/
│   ├── css/
│   │   └── dashboard.css   # Dark glassmorphism styles & responsive HUD grid
│   └── js/
│       └── dashboard.js    # Client-side streaming, WebSocket handling, and controls
├── requirements.txt        # Python package dependencies
└── .gitignore              # Ignored files, weights, and temporary artifacts
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
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Starting the Site**:
   - **Method A (One-Click Windows)**: Double-click [`start.bat`](file:///d:/movies/aiml/aiblend/start.bat). This boots Uvicorn on port 8000 and automatically opens `http://localhost:8000` in your default browser.
   - **Method B (Terminal / Cross-Platform)**:
     ```bash
     python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
     ```

5. **Stopping the Site**:
   - **Method A (One-Click Windows)**: Double-click [`stop.bat`](file:///d:/movies/aiml/aiblend/stop.bat). It instantly locates and terminates any server process listening on port 8000.
   - **Method B (Terminal)**: Press `Ctrl + C` in the terminal window where the server is running.
   - **Method C (PowerShell Command)**:
     ```powershell
     Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
     ```

6. **Access the Dashboard**:
   Open `http://localhost:8000` in your web browser.

---

## Verification & Testing

### 1. End-to-End Verification Pipeline
Run the automated end-to-end test suite:
```bash
python test_pipeline.py
```
This validates configuration directories, demo video synthesis, YOLOv8 pose inference, kinematic tracker state machines, and all REST endpoints.

### 2. Batch Surveillance Dataset Evaluation
Run the automated batch test suite across benchmark surveillance videos:
```bash
python batch_test.py --count 10 --step 1
```
Computes Accuracy, Precision, Recall, and F1-score and outputs a summary CSV report.

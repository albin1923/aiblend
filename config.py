"""
Configuration settings for SENTINEL-CV Shoplifting Detection.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
SNAPSHOT_DIR = BASE_DIR / "snapshots"

# Ensure runtime directories exist
for directory in (DATA_DIR, UPLOAD_DIR, SNAPSHOT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# YOLO Pose Model Configuration
MODEL_NAME = "yolov8n-pose.pt"
YOLO_DETECTION_CONF = 0.35
POSE_KEYPOINT_CONF = 0.50

# Behavioral Heuristic Thresholds & Proportions
POCKET_ZONE_DIST_RATIO = 0.25
CHEST_ZONE_DIST_RATIO = 0.35
FPS_ASSUMPTION = 30.0
POCKET_DWELL_TIME_SEC = 0.8
REACH_HEIGHT_RATIO = 0.15
CONCEAL_SEQUENCE_WINDOW_SEC = 3.0
CROUCH_DURATION_SEC = 1.0

# Threat Level Classifications
THREAT_SUSPICIOUS = 0.50
THREAT_NORMAL = 0.20

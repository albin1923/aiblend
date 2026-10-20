"""
Comprehensive Pipeline and Endpoint Verification Suite for SENTINEL-CV.
Runs end-to-end tests covering:
1. Configuration & runtime directories
2. Synthetic retail video generation
3. YOLOv8-Pose detection & HUD rendering
4. Kinematic behavior scoring & tracking state machine
5. FastAPI REST API endpoints
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import config
from tracker import MultiPersonTrackerManager, PersonTracker
from detector import ShopliftingDetector
from test_video_generator import generate_shoplifting_demo_video
from app import app
from starlette.testclient import TestClient


def test_1_configuration():
    print("\n[TEST 1] Verifying configuration & runtime directories...")
    assert config.DATA_DIR.exists(), "DATA_DIR does not exist"
    assert config.UPLOAD_DIR.exists(), "UPLOAD_DIR does not exist"
    assert config.SNAPSHOT_DIR.exists(), "SNAPSHOT_DIR does not exist"
    print("  -> PASS: All runtime directories verified.")


def test_2_video_generator():
    print("\n[TEST 2] Verifying synthetic retail CCTV demo video generation...")
    demo_path = config.DATA_DIR / "demo_test_clip.mp4"
    if demo_path.exists():
        demo_path.unlink()
    generate_shoplifting_demo_video(str(demo_path), duration_sec=2, fps=15)
    assert demo_path.exists(), "Demo video was not created"
    
    cap = cv2.VideoCapture(str(demo_path))
    assert cap.isOpened(), "Generated video cannot be decoded by OpenCV"
    ret, frame = cap.read()
    cap.release()
    assert ret and frame is not None, "Failed to read first frame of generated video"
    assert frame.shape == (480, 640, 3), f"Unexpected frame shape: {frame.shape}"
    demo_path.unlink()  # Cleanup test clip
    print("  -> PASS: Video generation and OpenCV decoding verified.")


def test_3_tracker_kinematics():
    print("\n[TEST 3] Verifying MultiPersonTrackerManager and behavioral scoring...")
    manager = MultiPersonTrackerManager()
    
    # Create mock keypoints for a person (17, 3)
    kpts = np.zeros((17, 3), dtype=np.float32)
    # Confidences
    kpts[:, 2] = 0.90
    # Shoulders
    kpts[5] = [200, 150, 0.9]  # L Shoulder
    kpts[6] = [240, 150, 0.9]  # R Shoulder
    # Hips
    kpts[11] = [205, 250, 0.9] # L Hip
    kpts[12] = [235, 250, 0.9] # R Hip
    # Wrists near hips (pocket concealment)
    kpts[9] = [205, 260, 0.9]  # L Wrist in pocket
    kpts[10] = [235, 260, 0.9] # R Wrist in pocket
    
    bbox = [180, 100, 260, 380]
    now = time.time()
    
    # Feed multiple frames to simulate dwell time
    alert_triggered = False
    for step in range(35):
        alerts = manager.update_tracks([{'id': 1, 'bbox': bbox, 'keypoints': kpts}], current_time=now + step * 0.04)
        if alerts:
            alert_triggered = True
            
    telemetry = manager.get_active_telemetry()
    assert len(telemetry) == 1, "Expected exactly 1 tracked person"
    assert telemetry[0]['pocket_conf'] > 40, f"Expected elevated pocket confidence, got {telemetry[0]['pocket_conf']}"
    
    # Test reset
    manager.reset()
    assert len(manager.trackers) == 0, "Tracker manager reset failed"
    print("  -> PASS: Kinematic state machine & dwell calculation verified.")


def test_4_detector_pipeline():
    print("\n[TEST 4] Verifying ShopliftingDetector and HUD rendering...")
    detector = ShopliftingDetector()
    assert detector.model is not None, "Model failed to initialize"
    
    # Generate test frame
    dummy_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    # Draw simple person rectangle
    cv2.rectangle(dummy_frame, (200, 100), (300, 400), (60, 60, 180), -1)
    
    annotated, alerts, telemetry = detector.process_frame(dummy_frame, time.time())
    assert annotated.shape == dummy_frame.shape, "Annotated frame shape mismatch"
    assert "fps" in telemetry, "Telemetry missing FPS field"
    assert "system_status" in telemetry, "Telemetry missing system_status field"
    
    # Test reset
    detector.reset()
    print("  -> PASS: Pose detector inference and HUD rendering pipeline verified.")


def test_5_fastapi_endpoints():
    print("\n[TEST 5] Verifying FastAPI HTTP endpoints...")
    client = TestClient(app)
    
    # 1. Dashboard UI
    res = client.get("/")
    assert res.status_code == 200, f"Dashboard index returned {res.status_code}"
    assert "SENTINEL" in res.text, "Dashboard HTML missing SENTINEL branding"
    
    # 2. System Status API
    res = client.get("/api/status")
    assert res.status_code == 200, f"/api/status returned {res.status_code}"
    data = res.json()
    assert data["status"] == "ONLINE", f"Expected ONLINE, got {data['status']}"
    assert "active_source" in data, "Status missing active_source"
    
    # 3. Incidents API
    res = client.get("/api/incidents")
    assert res.status_code == 200, f"/api/incidents returned {res.status_code}"
    assert "incidents" in res.json(), "Incidents JSON missing 'incidents' key"
    
    # 4. Export JSON
    res = client.get("/api/export?format=json")
    assert res.status_code == 200, f"/api/export?format=json returned {res.status_code}"
    
    # 5. Export CSV
    res = client.get("/api/export?format=csv")
    assert res.status_code == 200, f"/api/export?format=csv returned {res.status_code}"
    assert "text/csv" in res.headers["content-type"], "CSV export missing text/csv header"
    
    # 6. Source Switching API (Valid & Invalid)
    res = client.post("/api/switch_source", data={"source": "demo"})
    assert res.status_code == 200, f"Source switch returned {res.status_code}"
    
    res = client.post("/api/switch_source", data={"source": "invalid_source"})
    assert res.status_code == 400, "Expected 400 for invalid source"

    print("  -> PASS: All REST endpoints and exports verified.")


if __name__ == "__main__":
    print("===============================================================")
    print("     SENTINEL-CV END-TO-END PIPELINE VERIFICATION SUITE       ")
    print("===============================================================")
    try:
        test_1_configuration()
        test_2_video_generator()
        test_3_tracker_kinematics()
        test_4_detector_pipeline()
        test_5_fastapi_endpoints()
        print("\n===============================================================")
        print("     >>> ALL 5/5 VERIFICATION TESTS PASSED SUCCESSFULLY! <<<   ")
        print("===============================================================\n")
    except Exception as e:
        print(f"\n[FAILED] Verification error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

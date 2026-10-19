"""
FastAPI Surveillance Web Application for Real-Time Shoplifting Detection.
Provides MJPEG video streaming, WebSocket telemetry, source switching,
incident logging, and REST management endpoints.
"""

import os
import cv2
import time
import json
import uuid
import asyncio
import threading
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
from detector import ShopliftingDetector
from test_video_generator import generate_shoplifting_demo_video

app = FastAPI(title="SENTINEL-CV | AI Shoplifting Detection")

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(config.BASE_DIR / "static")), name="static")
app.mount("/snapshots", StaticFiles(directory=str(config.SNAPSHOT_DIR)), name="snapshots")
templates = Jinja2Templates(directory=str(config.BASE_DIR / "templates"))

# Global State & Video Manager
class VideoStreamManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_source_type = "demo"  # "demo", "webcam", "file"
        self.current_file_path = None
        self.cap = None
        self.running = True
        self.detector = ShopliftingDetector()
        
        self.latest_raw_frame = None
        self.latest_annotated_frame = None
        self.latest_telemetry = {
            "fps": 0.0,
            "active_targets": 0,
            "system_status": "NORMAL",
            "max_threat": 0.0,
            "persons": [],
            "source": "demo"
        }
        self.active_alerts = []
        self.all_incidents = []
        self.connected_websockets: List[WebSocket] = []
        
        # Ensure demo video exists
        self.demo_video_path = config.DATA_DIR / "demo_shoplifting.mp4"
        if not self.demo_video_path.exists():
            print("[StreamManager] Creating demo surveillance video...")
            generate_shoplifting_demo_video(str(self.demo_video_path))
            
        self._open_source("demo")
        
        # Start background processing thread
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()

    def _open_source(self, source_type: str, file_path: str = None):
        """Open or change video capture source."""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                
            self.current_source_type = source_type
            
            if source_type == "webcam":
                # Try opening index 0
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    print("[StreamManager] Warning: Webcam 0 failed to open. Falling back to demo video.")
                    self.current_source_type = "demo"
                    self.cap = cv2.VideoCapture(str(self.demo_video_path))
            elif source_type == "file" and file_path and os.path.exists(file_path):
                self.current_file_path = file_path
                self.cap = cv2.VideoCapture(file_path)
            else:
                self.current_source_type = "demo"
                self.cap = cv2.VideoCapture(str(self.demo_video_path))
                
            print(f"[StreamManager] Switched source to: {self.current_source_type}")

    def switch_source(self, source_type: str, file_path: str = None):
        self._open_source(source_type, file_path)

    def _processing_loop(self):
        """Continuous video frame acquisition, inference, and alert handling."""
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                # Loop video files seamlessly
                if self.current_source_type in ("demo", "file"):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        time.sleep(0.05)
                        continue
                else:
                    time.sleep(0.05)
                    continue
                    
            # Resize large frames for speed & consistent inference scale
            h, w = frame.shape[:2]
            if w > 854:
                scale = 854.0 / w
                frame = cv2.resize(frame, (854, int(h * scale)))
                
            now = time.time()
            annotated_frame, new_alerts, telemetry = self.detector.process_frame(frame, now)
            telemetry["source"] = self.current_source_type
            
            # Handle new alert events (save snapshots & record incident)
            for alert in new_alerts:
                snapshot_id = f"inc_{int(now)}_{uuid.uuid4().hex[:6]}"
                snapshot_filename = f"{snapshot_id}.jpg"
                snapshot_path = config.SNAPSHOT_DIR / snapshot_filename
                
                # Save high-res evidence snapshot
                cv2.imwrite(str(snapshot_path), annotated_frame)
                
                incident_record = {
                    "id": snapshot_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(alert["timestamp"])),
                    "person_id": alert["person_id"],
                    "threat_score": alert["threat_score"],
                    "threat_level": alert["threat_level"],
                    "pocket_conf": alert["pocket_conf"],
                    "chest_conf": alert["chest_conf"],
                    "sequence_conf": alert["sequence_conf"],
                    "crouch_conf": alert["crouch_conf"],
                    "reasons": alert["reasons"],
                    "snapshot_url": f"/snapshots/{snapshot_filename}"
                }
                self.all_incidents.insert(0, incident_record)
                if len(self.all_incidents) > 100:
                    self.all_incidents.pop()
                    
                # Broadcast alert to WebSockets asynchronously
                self._broadcast_event({
                    "type": "NEW_ALERT",
                    "incident": incident_record
                })

            with self.lock:
                self.latest_raw_frame = frame
                self.latest_annotated_frame = annotated_frame
                self.latest_telemetry = telemetry

            # Modulate frame rate if running on pre-recorded demo video to keep real-time ~25 FPS
            if self.current_source_type in ("demo", "file"):
                time.sleep(0.028)
            else:
                time.sleep(0.005)

    def _broadcast_event(self, data: dict):
        """Broadcast payload to connected websockets."""
        if not self.connected_websockets:
            return
            
        async def _send():
            msg = json.dumps(data)
            for ws in list(self.connected_websockets):
                try:
                    await ws.send_text(msg)
                except Exception:
                    if ws in self.connected_websockets:
                        self.connected_websockets.remove(ws)
                        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(_send())
            else:
                loop.run_until_complete(_send())
        except Exception:
            pass

    def get_jpeg_frame(self) -> bytes:
        """Encode current annotated frame into JPEG for MJPEG stream."""
        with self.lock:
            frame = self.latest_annotated_frame
            if frame is None:
                # Return placeholder frame if not ready
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "INITIALIZING CV PIPELINE...", (140, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return jpeg.tobytes() if ret else b""


# Singleton Video Stream Manager
stream_mgr = VideoStreamManager()


# -------------------------------------------------------------
# Streaming Routes
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return templates.TemplateResponse("index.html", {"request": {}})


def mjpeg_generator():
    """Generator for streaming multipart JPEG frames."""
    while True:
        frame_bytes = stream_mgr.get_jpeg_frame()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.035)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# -------------------------------------------------------------
# WebSocket Telemetry
# -------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stream_mgr.connected_websockets.append(websocket)
    try:
        # Initial greeting with recent incidents
        await websocket.send_text(json.dumps({
            "type": "INIT",
            "incidents": stream_mgr.all_incidents[:15],
            "config": {
                "show_skeleton": stream_mgr.detector.show_skeleton,
                "show_zones": stream_mgr.detector.show_zones,
                "show_trajectory": stream_mgr.detector.show_trajectory,
                "show_hud": stream_mgr.detector.show_hud,
                "detection_conf": stream_mgr.detector.detection_conf,
                "source": stream_mgr.current_source_type
            }
        }))
        
        while True:
            # Send periodic telemetry updates
            data = {
                "type": "TELEMETRY",
                "telemetry": stream_mgr.latest_telemetry
            }
            await websocket.send_text(json.dumps(data))
            
            # Non-blocking receive with timeout
            try:
                incoming_text = await asyncio.wait_for(websocket.receive_text(), timeout=0.15)
                msg = json.loads(incoming_text)
                action = msg.get("action")
                
                if action == "TOGGLE_FEATURE":
                    feature = msg.get("feature")
                    val = msg.get("value")
                    if feature == "skeleton":
                        stream_mgr.detector.show_skeleton = val
                    elif feature == "zones":
                        stream_mgr.detector.show_zones = val
                    elif feature == "trajectory":
                        stream_mgr.detector.show_trajectory = val
                    elif feature == "hud":
                        stream_mgr.detector.show_hud = val
                        
                elif action == "SET_CONFIDENCE":
                    conf_val = float(msg.get("value", 0.35))
                    stream_mgr.detector.detection_conf = conf_val
                    
                elif action == "SWITCH_SOURCE":
                    src = msg.get("source", "demo")
                    stream_mgr.switch_source(src)
                    
            except asyncio.TimeoutError:
                pass
                
    except WebSocketDisconnect:
        if websocket in stream_mgr.connected_websockets:
            stream_mgr.connected_websockets.remove(websocket)
    except Exception as e:
        if websocket in stream_mgr.connected_websockets:
            stream_mgr.connected_websockets.remove(websocket)


# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------
@app.post("/api/switch_source")
async def switch_source_endpoint(source: str = Form(...)):
    """Switch active input between 'demo' and 'webcam'."""
    if source not in ("demo", "webcam"):
        raise HTTPException(status_code=400, detail="Invalid source")
    stream_mgr.switch_source(source)
    return {"status": "success", "active_source": source}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload custom video for shoplifting analysis."""
    safe_name = f"{int(time.time())}_{file.filename}"
    save_path = config.UPLOAD_DIR / safe_name
    
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    # Switch to uploaded video immediately
    stream_mgr.switch_source("file", str(save_path))
    return {"status": "success", "filename": safe_name, "message": "Video loaded and playing"}


@app.get("/api/incidents")
async def get_incidents():
    """Retrieve full incident history with snapshot URLs."""
    return {"incidents": stream_mgr.all_incidents}


@app.get("/api/export")
async def export_incidents(format: str = "json"):
    """Export incident history as JSON or CSV."""
    if format == "csv":
        import io
        import csv
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Incident ID", "Timestamp", "Person ID", "Threat Level", "Risk Score %", "Pocket %", "Jacket %", "Sequence %", "Crouch %", "Reasons"])
        for inc in stream_mgr.all_incidents:
            writer.writerow([
                inc["id"],
                inc["timestamp"],
                inc["person_id"],
                inc["threat_level"],
                inc["threat_score"],
                inc["pocket_conf"],
                inc["chest_conf"],
                inc["sequence_conf"],
                inc["crouch_conf"],
                "; ".join(inc["reasons"])
            ])
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=shoplifting_incidents_report.csv"}
        )
    else:
        return JSONResponse(content=stream_mgr.all_incidents)


@app.get("/api/status")
async def get_status():
    """System health check and diagnostic telemetry."""
    import torch
    return {
        "status": "ONLINE",
        "gpu_available": torch.cuda.is_available() if "torch" in globals() else False,
        "device": stream_mgr.detector.device,
        "active_source": stream_mgr.current_source_type,
        "fps": stream_mgr.latest_telemetry["fps"],
        "total_incidents": len(stream_mgr.all_incidents)
    }

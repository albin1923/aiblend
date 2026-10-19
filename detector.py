"""
Computer Vision Inference & Surveillance HUD Rendering Engine.
Combines YOLOv8-Pose with multi-person tracking and behavioral risk overlays.
"""

import cv2
import numpy as np
import time
import math
from typing import List, Dict, Any, Tuple
import config
from tracker import MultiPersonTrackerManager


# Color definitions (BGR for OpenCV)
COLOR_SAFE = (74, 222, 128)        # Neon Emerald Green
COLOR_WARNING = (250, 204, 21)     # Amber Yellow
COLOR_ALERT = (38, 38, 250)        # Bright Alert Red
COLOR_CYAN = (235, 206, 135)       # Tactical Cyan
COLOR_PURPLE = (216, 112, 147)     # Zone Purple
COLOR_BG_DARK = (20, 20, 26)       # Dark HUD panel background

# COCO Keypoint pairs for skeleton bones
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),               # Head / Face
    (5, 6),                                       # Shoulders
    (5, 7), (7, 9),                               # Left arm: Shoulder -> Elbow -> Wrist
    (6, 8), (8, 10),                              # Right arm: Shoulder -> Elbow -> Wrist
    (5, 11), (6, 12),                             # Torso sides
    (11, 12),                                     # Hips
    (11, 13), (13, 15),                           # Left leg: Hip -> Knee -> Ankle
    (12, 14), (14, 16)                            # Right leg: Hip -> Knee -> Ankle
]


class ShopliftingDetector:
    """Manages YOLO Pose inference, tracking, behavior scoring, and frame HUD annotation."""
    def __init__(self, model_name: str = config.MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.device = "cpu"
        self.tracker_manager = MultiPersonTrackerManager()
        self.show_skeleton = True
        self.show_zones = True
        self.show_trajectory = True
        self.show_hud = True
        self.detection_conf = config.YOLO_DETECTION_CONF
        
        # Performance tracking
        self.fps = 0.0
        self.prev_frame_time = time.time()
        self.frame_count = 0
        
        # Load model lazily or on init
        self._init_model()

    def _init_model(self):
        """Initialize Ultralytics YOLOv8-Pose model with GPU/CPU auto-selection."""
        try:
            import torch
            from ultralytics import YOLO
            
            if torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"
                
            print(f"[Detector] Loading {self.model_name} on device: {self.device}")
            self.model = YOLO(self.model_name)
            # Warm up model with a dummy blank image
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model(dummy, device=self.device, verbose=False)
            print("[Detector] Model successfully initialized and warmed up.")
        except Exception as e:
            print(f"[Detector] Warning during model initialization: {e}")
            self.model = None

    def process_frame(self, frame: np.ndarray, current_time: float = None) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Processes a single video frame:
        1. Runs YOLOv8-Pose multi-object tracking.
        2. Computes behavioral threat scores.
        3. Renders high-tech tactical surveillance HUD.
        Returns:
            annotated_frame: np.ndarray
            alerts: list of newly triggered alert events
            telemetry: frame-level summary stats for dashboard
        """
        if current_time is None:
            current_time = time.time()
            
        self.frame_count += 1
        now = time.time()
        dt = now - self.prev_frame_time
        if dt > 0:
            current_fps = 1.0 / dt
            self.fps = 0.9 * self.fps + 0.1 * current_fps
        self.prev_frame_time = now
        
        h, w = frame.shape[:2]
        detections = []
        
        if self.model is not None:
            try:
                # Track persons across frames with ByteTrack
                results = self.model.track(
                    source=frame,
                    persist=True,
                    conf=self.detection_conf,
                    device=self.device,
                    verbose=False,
                    classes=[0]  # Class 0 is 'person' in COCO
                )
                
                if results and len(results) > 0:
                    res = results[0]
                    boxes = res.boxes
                    keypoints_data = res.keypoints
                    
                    if boxes is not None and len(boxes) > 0 and keypoints_data is not None:
                        # Extract boxes and track IDs
                        xyxy = boxes.xyxy.cpu().numpy()
                        ids = boxes.id.cpu().numpy() if boxes.id is not None else np.arange(len(boxes))
                        kpts = keypoints_data.data.cpu().numpy()  # shape (N, 17, 3) [x, y, conf]
                        
                        for i in range(len(xyxy)):
                            pid = int(ids[i])
                            bbox = xyxy[i].tolist()
                            kp = kpts[i]
                            detections.append({
                                'id': pid,
                                'bbox': bbox,
                                'keypoints': kp
                            })
            except Exception as e:
                # Fallback on tracking error
                pass

        # Update behavioral state trackers
        alerts = self.tracker_manager.update_tracks(detections, current_time)
        
        # Render surveillance HUD on frame copy
        annotated = frame.copy()
        self._render_hud(annotated, alerts)
        
        # Frame telemetry summary
        active_telemetry = self.tracker_manager.get_active_telemetry()
        max_threat = max([p['threat_score'] for p in active_telemetry], default=0.0)
        
        system_status = "NORMAL"
        if max_threat >= config.THREAT_SUSPICIOUS * 100:
            system_status = "HIGH ALERT"
        elif max_threat >= config.THREAT_NORMAL * 100:
            system_status = "SUSPICIOUS"
            
        telemetry = {
            "fps": round(self.fps, 1),
            "active_targets": len(active_telemetry),
            "system_status": system_status,
            "max_threat": round(max_threat, 1),
            "persons": active_telemetry,
            "timestamp": current_time
        }
        
        return annotated, alerts, telemetry

    def _render_hud(self, frame: np.ndarray, new_alerts: List[Dict[str, Any]]):
        """Draws tactical HUD, skeletons, concealment zones, and warning banners."""
        h, w = frame.shape[:2]
        
        # 1. Render person annotations
        for pid, tracker in self.tracker_manager.trackers.items():
            bbox = tracker.bbox
            threat = tracker.threat_score
            threat_level = tracker.threat_level
            
            # Select color palette based on threat level
            if threat_level == "ALERT":
                color = COLOR_ALERT
            elif threat_level == "SUSPICIOUS":
                color = COLOR_WARNING
            else:
                color = COLOR_SAFE
                
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            
            # Bounding Box with tactical corner brackets
            self._draw_tactical_box(frame, x1, y1, x2, y2, color, thickness=2)
            
            # Draw Concealment Target Zones (Pocket & Chest)
            if self.show_zones:
                if tracker.left_pocket_zone:
                    px, py, pr = tracker.left_pocket_zone
                    cv2.circle(frame, (px, py), pr, COLOR_PURPLE, 1, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 3, COLOR_PURPLE, -1)
                if tracker.right_pocket_zone:
                    px, py, pr = tracker.right_pocket_zone
                    cv2.circle(frame, (px, py), pr, COLOR_PURPLE, 1, cv2.LINE_AA)
                    cv2.circle(frame, (px, py), 3, COLOR_PURPLE, -1)
                if tracker.chest_zone:
                    cx, cy, cr = tracker.chest_zone
                    cv2.circle(frame, (cx, cy), cr, (180, 105, 255), 1, cv2.LINE_AA)
                    
            # Draw Wrist Trajectory Trails
            if self.show_trajectory and len(tracker.wrist_trajectory) > 1:
                traj = list(tracker.wrist_trajectory)
                for i in range(1, len(traj)):
                    p_start = (traj[i-1][0], traj[i-1][1])
                    p_end = (traj[i][0], traj[i][1])
                    alpha = i / len(traj)
                    t_color = (int(color[0]*alpha), int(color[1]*alpha), int(color[2]*alpha))
                    cv2.line(frame, p_start, p_end, t_color, 2, cv2.LINE_AA)

            # Draw Pose Skeleton
            if self.show_skeleton and len(tracker.history) > 0:
                kpts = tracker.history[-1].keypoints
                if len(kpts) >= 17:
                    # Draw bones
                    for pt1_idx, pt2_idx in SKELETON_CONNECTIONS:
                        p1, p2 = kpts[pt1_idx], kpts[pt2_idx]
                        if p1[2] > config.POSE_KEYPOINT_CONF and p2[2] > config.POSE_KEYPOINT_CONF:
                            pt1 = (int(p1[0]), int(p1[1]))
                            pt2 = (int(p2[0]), int(p2[1]))
                            
                            # Check if bone is involved in active suspicious gesture
                            is_suspicious_joint = (pt1_idx, pt2_idx) in tracker.active_suspicious_joints or \
                                                  (pt2_idx, pt1_idx) in tracker.active_suspicious_joints
                            bone_color = COLOR_ALERT if is_suspicious_joint else (160, 220, 160)
                            bone_thick = 3 if is_suspicious_joint else 1
                            cv2.line(frame, pt1, pt2, bone_color, bone_thick, cv2.LINE_AA)
                            
                    # Draw joint keypoints
                    for i, kp in enumerate(kpts):
                        if kp[2] > config.POSE_KEYPOINT_CONF:
                            pt = (int(kp[0]), int(kp[1]))
                            # Highlight wrists prominently
                            if i in (9, 10):  # Wrists
                                cv2.circle(frame, pt, 6, color, -1, cv2.LINE_AA)
                                cv2.circle(frame, pt, 8, (255, 255, 255), 1, cv2.LINE_AA)
                            else:
                                cv2.circle(frame, pt, 3, (220, 220, 220), -1, cv2.LINE_AA)
                                
            # Draw Floating Person Telemetry HUD Card above bbox
            if self.show_hud:
                self._draw_person_card(frame, x1, y1, tracker, color)
                
        # 2. Render Top Surveillance Status Bar
        self._draw_status_bar(frame)
        
        # 3. Flashing Border if any Active Alert
        has_critical_alert = any(t.threat_level == "ALERT" for t in self.tracker_manager.trackers.values())
        if has_critical_alert:
            # Pulsing red frame border
            pulse_thick = 6 + int(2 * math.sin(time.time() * 8))
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_ALERT, pulse_thick)
            
            # Top flashing alert banner
            banner_text = "!!! SUSPICIOUS CONCEALMENT DETECTED - SECURITY INTERCEPT !!!"
            (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            bx1 = (w - tw) // 2 - 15
            bx2 = (w + tw) // 2 + 15
            cv2.rectangle(frame, (bx1, 46), (bx2, 46 + th + 14), (15, 15, 180), -1)
            cv2.putText(frame, banner_text, ((w - tw) // 2, 46 + th + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_tactical_box(self, img, x1, y1, x2, y2, color, thickness=2):
        """Draws semi-transparent bounding box with tactical corner crosshairs."""
        # Main box with subtle line
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        
        # Corner brackets
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        c_thick = thickness + 1
        
        # Top-left
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, c_thick)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, c_thick)
        # Top-right
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, c_thick)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, c_thick)
        # Bottom-left
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, c_thick)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, c_thick)
        # Bottom-right
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, c_thick)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, c_thick)

    def _draw_person_card(self, frame, x1, y1, tracker, color):
        """Draws floating telemetry card above person bounding box."""
        card_w = 175
        card_h = 60
        card_x = max(10, min(frame.shape[1] - card_w - 10, x1))
        card_y = max(45, y1 - card_h - 8)
        
        # Dark HUD background
        sub_img = frame[card_y:card_y+card_h, card_x:card_x+card_w]
        black_rect = np.full(sub_img.shape, 20, dtype=np.uint8)
        res = cv2.addWeighted(sub_img, 0.25, black_rect, 0.75, 0)
        frame[card_y:card_y+card_h, card_x:card_x+card_w] = res
        
        # Border
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), color, 1)
        
        # Header: ID & Status
        header_text = f"SUSPECT #{tracker.id} | {tracker.threat_level}"
        cv2.putText(frame, header_text, (card_x + 8, card_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        
        # Threat Gauge Bar
        score_pct = int(tracker.threat_score * 100)
        cv2.putText(frame, f"RISK: {score_pct}%", (card_x + 8, card_y + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)
        
        # Progress Bar Background
        bar_x = card_x + 75
        bar_y = card_y + 24
        bar_w = 90
        bar_h = 10
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        # Filled bar
        fill_w = int(bar_w * tracker.threat_score)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
            
        # Metric indicators (Pocket % / Reach-Conceal %)
        sub_info = f"PKT: {int(tracker.pocket_conf*100)}% | JKT: {int(tracker.chest_conf*100)}%"
        cv2.putText(frame, sub_info, (card_x + 8, card_y + 51),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 180, 190), 1, cv2.LINE_AA)

    def _draw_status_bar(self, frame):
        """Draws futuristic top status banner with camera, FPS, and alert statistics."""
        w = frame.shape[1]
        banner_h = 36
        
        # Dark top strip
        sub_img = frame[0:banner_h, 0:w]
        black_rect = np.full(sub_img.shape, 15, dtype=np.uint8)
        frame[0:banner_h, 0:w] = cv2.addWeighted(sub_img, 0.15, black_rect, 0.85, 0)
        cv2.line(frame, (0, banner_h), (w, banner_h), (60, 60, 70), 1)
        
        # Status Items
        cv2.circle(frame, (18, 18), 5, (0, 255, 0), -1)
        cv2.putText(frame, "CAM-01 [LIVE]", (30, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
                    
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (180, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 220), 1, cv2.LINE_AA)
                    
        targets_text = f"TARGETS: {len(self.tracker_manager.trackers)}"
        cv2.putText(frame, targets_text, (290, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 220), 1, cv2.LINE_AA)
                    
        total_alerts = len(self.tracker_manager.all_time_alerts)
        alert_col = COLOR_ALERT if total_alerts > 0 else (180, 200, 220)
        alerts_text = f"INCIDENTS: {total_alerts}"
        cv2.putText(frame, alerts_text, (410, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, alert_col, 1, cv2.LINE_AA)
                    
        # Device badge
        dev_text = f"ENGINE: YOLOv8-POSE ({self.device.upper()})"
        (tw, _), _ = cv2.getTextSize(dev_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(frame, dev_text, (w - tw - 15, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 160, 180), 1, cv2.LINE_AA)

"""
Behavior Tracker and State Machine for Shoplifting Detection.
Tracks human keypoint kinematics, concealment zones, temporal dwell time,
and complex multi-stage reach-and-pocket behaviors.
"""

import time
import math
import numpy as np
from collections import deque
import config


def euclidean_dist(p1, p2):
    """Euclidean distance between two 2D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class KeypointHistory:
    """Snapshot of keypoints and posture at a given timestamp."""
    def __init__(self, keypoints, bbox, timestamp):
        self.keypoints = keypoints  # array shape (17, 3) [x, y, conf]
        self.bbox = bbox            # [x1, y1, x2, y2]
        self.timestamp = timestamp
        
        # Extract landmark points if valid
        # COCO: 5=LShoulder, 6=RShoulder, 9=LWrist, 10=RWrist, 11=LHip, 12=RHip, 13=LKnee, 14=RKnee
        self.valid = len(keypoints) >= 17
        
        if self.valid:
            self.l_shoulder = keypoints[5][:2]
            self.r_shoulder = keypoints[6][:2]
            self.l_wrist = keypoints[9][:2]
            self.r_wrist = keypoints[10][:2]
            self.l_hip = keypoints[11][:2]
            self.r_hip = keypoints[12][:2]
            self.l_knee = keypoints[13][:2]
            self.r_knee = keypoints[14][:2]
            
            self.l_wrist_conf = keypoints[9][2]
            self.r_wrist_conf = keypoints[10][2]
            self.l_hip_conf = keypoints[11][2]
            self.r_hip_conf = keypoints[12][2]
            
            # Torso length (scale-normalization factor)
            mid_shoulder = ((self.l_shoulder[0] + self.r_shoulder[0]) / 2.0,
                            (self.l_shoulder[1] + self.r_shoulder[1]) / 2.0)
            mid_hip = ((self.l_hip[0] + self.r_hip[0]) / 2.0,
                       (self.l_hip[1] + self.r_hip[1]) / 2.0)
            
            t_len = euclidean_dist(mid_shoulder, mid_hip)
            bbox_h = max(1.0, bbox[3] - bbox[1])
            self.torso_len = t_len if t_len > 15 else (bbox_h * 0.45)
            self.mid_shoulder = mid_shoulder
            self.mid_hip = mid_hip
        else:
            self.torso_len = max(1.0, (bbox[3] - bbox[1]) * 0.45)
            self.mid_shoulder = (0, 0)
            self.mid_hip = (0, 0)


class PersonTracker:
    """Tracks a single individual's behavior and risk profile over time."""
    def __init__(self, person_id: int):
        self.id = person_id
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.bbox = [0, 0, 0, 0]
        
        # History queue of recent frames (~3 seconds at 25 fps)
        self.history = deque(maxlen=75)
        
        # Kinematic states
        self.left_pocket_dwell_frames = 0
        self.right_pocket_dwell_frames = 0
        self.chest_dwell_frames = 0
        self.crouch_dwell_frames = 0
        
        # Sequence detection state (Shelf Reach -> Pocket Conceal)
        self.reach_state = False
        self.reach_timestamp = 0.0
        self.reach_hand = None
        
        # Confidence scores (0.0 to 1.0)
        self.pocket_conf = 0.0
        self.chest_conf = 0.0
        self.sequence_conf = 0.0
        self.crouch_conf = 0.0
        self.threat_score = 0.0
        self.threat_level = "NORMAL"  # "NORMAL", "SUSPICIOUS", "ALERT"
        
        # Visual zones for drawing
        self.left_pocket_zone = None
        self.right_pocket_zone = None
        self.chest_zone = None
        self.active_suspicious_joints = []
        
        # Alert trigger tracker
        self.last_alert_time = 0.0
        self.alert_cooldown = 3.0  # seconds between repeated alert events
        self.event_history = []
        
        # Trajectory of wrists (past 20 frames)
        self.wrist_trajectory = deque(maxlen=20)

    def update(self, keypoints, bbox, current_time=None):
        """Update tracker with new frame keypoints and bounding box."""
        if current_time is None:
            current_time = time.time()
            
        self.last_seen = current_time
        self.bbox = bbox
        
        snapshot = KeypointHistory(keypoints, bbox, current_time)
        self.history.append(snapshot)
        
        if not snapshot.valid:
            return self._finalize_scores()
            
        torso = snapshot.torso_len
        
        # Define anatomical concealment zones
        # Pocket zone: near hips, extending slightly downward
        l_hip = snapshot.l_hip
        r_hip = snapshot.r_hip
        l_pocket_center = (l_hip[0], l_hip[1] + torso * 0.15)
        r_pocket_center = (r_hip[0], r_hip[1] + torso * 0.15)
        pocket_radius = torso * config.POCKET_ZONE_DIST_RATIO
        
        self.left_pocket_zone = (int(l_pocket_center[0]), int(l_pocket_center[1]), int(pocket_radius))
        self.right_pocket_zone = (int(r_pocket_center[0]), int(r_pocket_center[1]), int(pocket_radius))
        
        # Chest / Jacket inner pocket zone: upper torso center
        chest_center = (snapshot.mid_shoulder[0], snapshot.mid_shoulder[1] + torso * 0.35)
        chest_radius = torso * config.CHEST_ZONE_DIST_RATIO
        self.chest_zone = (int(chest_center[0]), int(chest_center[1]), int(chest_radius))
        
        # Wrists
        l_wrist = snapshot.l_wrist
        r_wrist = snapshot.r_wrist
        l_w_conf = snapshot.l_wrist_conf
        r_w_conf = snapshot.r_wrist_conf
        
        # Save wrist trajectory points
        if r_w_conf > config.POSE_KEYPOINT_CONF:
            self.wrist_trajectory.append((int(r_wrist[0]), int(r_wrist[1]), 'right'))
        if l_w_conf > config.POSE_KEYPOINT_CONF:
            self.wrist_trajectory.append((int(l_wrist[0]), int(l_wrist[1]), 'left'))
            
        self.active_suspicious_joints = []
        
        # ----------------------------------------------------
        # 1. POCKET CONCEALMENT DETECTION
        # ----------------------------------------------------
        left_in_pocket = False
        right_in_pocket = False
        
        # Check Left Wrist -> Left Hip or Right Hip (cross-body concealment)
        if l_w_conf > config.POSE_KEYPOINT_CONF:
            d_l_pocket = euclidean_dist(l_wrist, l_pocket_center) / torso
            d_l_to_r = euclidean_dist(l_wrist, r_pocket_center) / torso
            if min(d_l_pocket, d_l_to_r) < config.POCKET_ZONE_DIST_RATIO:
                left_in_pocket = True
                self.active_suspicious_joints.append((9, 11))
                
        # Check Right Wrist -> Right Hip or Left Hip (cross-body concealment)
        if r_w_conf > config.POSE_KEYPOINT_CONF:
            d_r_pocket = euclidean_dist(r_wrist, r_pocket_center) / torso
            d_r_to_l = euclidean_dist(r_wrist, l_pocket_center) / torso
            if min(d_r_pocket, d_r_to_l) < config.POCKET_ZONE_DIST_RATIO:
                right_in_pocket = True
                self.active_suspicious_joints.append((10, 12))
                
        # Dwell time accumulator
        if left_in_pocket:
            self.left_pocket_dwell_frames += 1
        else:
            self.left_pocket_dwell_frames = max(0, self.left_pocket_dwell_frames - 2)
            
        if right_in_pocket:
            self.right_pocket_dwell_frames += 1
        else:
            self.right_pocket_dwell_frames = max(0, self.right_pocket_dwell_frames - 2)
            
        max_dwell = max(self.left_pocket_dwell_frames, self.right_pocket_dwell_frames)
        fps = config.FPS_ASSUMPTION
        dwell_sec = max_dwell / fps
        
        if dwell_sec >= config.POCKET_DWELL_TIME_SEC:
            # High pocket confidence scales with dwell duration
            self.pocket_conf = min(0.96, 0.50 + 0.35 * min(1.0, (dwell_sec - config.POCKET_DWELL_TIME_SEC) / 1.5))
        elif dwell_sec > 0.3:
            self.pocket_conf = 0.25 + 0.25 * (dwell_sec / config.POCKET_DWELL_TIME_SEC)
        else:
            self.pocket_conf = max(0.0, self.pocket_conf * 0.85)

        # ----------------------------------------------------
        # 2. JACKET / INNER TORSO CONCEALMENT DETECTION
        # ----------------------------------------------------
        hand_in_chest = False
        if l_w_conf > config.POSE_KEYPOINT_CONF:
            d_l_chest = euclidean_dist(l_wrist, chest_center) / torso
            if d_l_chest < config.CHEST_ZONE_DIST_RATIO:
                hand_in_chest = True
                self.active_suspicious_joints.append((9, 5))
                
        if r_w_conf > config.POSE_KEYPOINT_CONF:
            d_r_chest = euclidean_dist(r_wrist, chest_center) / torso
            if d_r_chest < config.CHEST_ZONE_DIST_RATIO:
                hand_in_chest = True
                self.active_suspicious_joints.append((10, 6))
                
        if hand_in_chest:
            self.chest_dwell_frames += 1
        else:
            self.chest_dwell_frames = max(0, self.chest_dwell_frames - 2)
            
        chest_dwell_sec = self.chest_dwell_frames / fps
        if chest_dwell_sec >= 0.6:
            self.chest_conf = min(0.92, 0.45 + 0.40 * min(1.0, (chest_dwell_sec - 0.6) / 1.2))
        else:
            self.chest_conf = max(0.0, self.chest_conf * 0.85)

        # ----------------------------------------------------
        # 3. REACH -> CONCEAL SEQUENCE (Shelf Pick -> Pocket)
        # ----------------------------------------------------
        # Detect if hand was previously raised (reaching for merchandise)
        # In image coords, smaller y = higher elevation (above shoulder)
        shoulder_y = min(snapshot.l_shoulder[1], snapshot.r_shoulder[1])
        
        # Check if reaching shelf currently
        l_reaching = (l_w_conf > config.POSE_KEYPOINT_CONF) and (l_wrist[1] < shoulder_y + torso * config.REACH_HEIGHT_RATIO)
        r_reaching = (r_w_conf > config.POSE_KEYPOINT_CONF) and (r_wrist[1] < shoulder_y + torso * config.REACH_HEIGHT_RATIO)
        
        if l_reaching or r_reaching:
            self.reach_state = True
            self.reach_timestamp = current_time
            self.reach_hand = 'left' if l_reaching else 'right'
            
        # Check if sequence is completed: reach followed by pocket within time window
        if self.reach_state and (current_time - self.reach_timestamp <= config.CONCEAL_SEQUENCE_WINDOW_SEC):
            if (self.reach_hand == 'right' and right_in_pocket) or \
               (self.reach_hand == 'left' and left_in_pocket) or \
               (hand_in_chest):
                # Reach followed by pocket dwell! Very high shoplifting probability
                self.sequence_conf = min(0.98, 0.75 + 0.20 * min(1.0, dwell_sec / 0.8))
        else:
            if current_time - self.reach_timestamp > config.CONCEAL_SEQUENCE_WINDOW_SEC:
                self.reach_state = False
                self.sequence_conf = max(0.0, self.sequence_conf * 0.88)
                
        # ----------------------------------------------------
        # 4. CROUCH / LOITER DETECTION
        # ----------------------------------------------------
        bbox_w = max(1.0, bbox[2] - bbox[0])
        bbox_h = max(1.0, bbox[3] - bbox[1])
        aspect_ratio = bbox_w / bbox_h
        
        # Check knee flexion or compressed aspect ratio
        is_crouching = False
        if aspect_ratio > 0.70:  # Wide bounding box indicates crouching or ducking
            is_crouching = True
            
        # Also check knee angle if keypoints present
        if snapshot.l_hip_conf > 0.3 and snapshot.l_wrist_conf > 0.3:
            # Check if wrist is near ground level
            if l_wrist[1] > snapshot.l_hip[1] + torso * 0.8:
                is_crouching = True
                
        if is_crouching:
            self.crouch_dwell_frames += 1
        else:
            self.crouch_dwell_frames = max(0, self.crouch_dwell_frames - 1)
            
        crouch_sec = self.crouch_dwell_frames / fps
        if crouch_sec >= config.CROUCH_DURATION_SEC:
            self.crouch_conf = min(0.85, 0.45 + 0.35 * min(1.0, (crouch_sec - 1.0) / 2.0))
        else:
            self.crouch_conf = max(0.0, self.crouch_conf * 0.90)

        return self._finalize_scores(current_time)

    def _finalize_scores(self, current_time=None):
        """Aggregate behavioral confidences into an overall Threat Score."""
        if current_time is None:
            current_time = time.time()
            
        # Dynamic fusion of confidence values
        raw_threat = max(
            self.sequence_conf,
            self.pocket_conf * 0.90,
            self.chest_conf * 0.88,
            self.crouch_conf * 0.75,
            (self.pocket_conf * 0.6 + self.crouch_conf * 0.5)
        )
        
        # Exponential moving average filter for smooth stability
        if raw_threat > self.threat_score:
            # Fast attack
            self.threat_score = 0.5 * self.threat_score + 0.5 * raw_threat
        else:
            # Slower decay
            self.threat_score = 0.85 * self.threat_score + 0.15 * raw_threat
            
        self.threat_score = float(np.clip(self.threat_score, 0.0, 1.0))
        
        # Classify Threat Level
        if self.threat_score >= config.THREAT_SUSPICIOUS:
            self.threat_level = "ALERT"
        elif self.threat_score >= config.THREAT_NORMAL:
            self.threat_level = "SUSPICIOUS"
        else:
            self.threat_level = "NORMAL"
            
        # Check for alert trigger
        alert_event = None
        if self.threat_level == "ALERT" and (current_time - self.last_alert_time >= self.alert_cooldown):
            self.last_alert_time = current_time
            
            reasons = []
            if self.sequence_conf >= 0.60:
                reasons.append(f"Reach & Pocket sequence ({int(self.sequence_conf*100)}%)")
            if self.pocket_conf >= 0.55:
                reasons.append(f"Pocket concealment dwell ({int(self.pocket_conf*100)}%)")
            if self.chest_conf >= 0.55:
                reasons.append(f"Jacket/chest concealment ({int(self.chest_conf*100)}%)")
            if self.crouch_conf >= 0.55:
                reasons.append(f"Suspicious crouching/loitering ({int(self.crouch_conf*100)}%)")
                
            if not reasons:
                reasons.append(f"High risk posture anomaly ({int(self.threat_score*100)}%)")
                
            alert_event = {
                "person_id": self.id,
                "timestamp": current_time,
                "threat_score": round(self.threat_score * 100, 1),
                "threat_level": self.threat_level,
                "pocket_conf": round(self.pocket_conf * 100, 1),
                "chest_conf": round(self.chest_conf * 100, 1),
                "sequence_conf": round(self.sequence_conf * 100, 1),
                "crouch_conf": round(self.crouch_conf * 100, 1),
                "reasons": reasons,
                "bbox": [int(x) for x in self.bbox]
            }
            self.event_history.append(alert_event)
            
        return alert_event

    def to_dict(self):
        """Serialize person telemetry for WebSocket transmission."""
        return {
            "id": self.id,
            "threat_score": round(self.threat_score * 100, 1),
            "threat_level": self.threat_level,
            "pocket_conf": round(self.pocket_conf * 100, 1),
            "chest_conf": round(self.chest_conf * 100, 1),
            "sequence_conf": round(self.sequence_conf * 100, 1),
            "crouch_conf": round(self.crouch_conf * 100, 1),
            "bbox": [int(x) for x in self.bbox],
            "first_seen": self.first_seen,
            "dwell_sec": round(max(self.left_pocket_dwell_frames, self.right_pocket_dwell_frames) / config.FPS_ASSUMPTION, 1)
        }


class MultiPersonTrackerManager:
    """Manages active person tracks and cleans up lost tracks."""
    def __init__(self, track_timeout_sec=2.5):
        self.trackers = {}
        self.track_timeout_sec = track_timeout_sec
        self.all_time_alerts = []

    def update_tracks(self, detections, current_time=None):
        """
        detections: list of dicts: {'id': int, 'bbox': list, 'keypoints': ndarray}
        returns: list of triggered alert events in this frame
        """
        if current_time is None:
            current_time = time.time()
            
        alerts_in_frame = []
        active_ids = set()
        
        for det in detections:
            pid = det.get('id', 0)
            active_ids.add(pid)
            
            if pid not in self.trackers:
                self.trackers[pid] = PersonTracker(pid)
                
            tracker = self.trackers[pid]
            alert = tracker.update(det['keypoints'], det['bbox'], current_time)
            if alert:
                alerts_in_frame.append(alert)
                self.all_time_alerts.append(alert)
                
        # Remove expired tracks
        stale_ids = [pid for pid, t in self.trackers.items() 
                     if (current_time - t.last_seen) > self.track_timeout_sec]
        for pid in stale_ids:
            del self.trackers[pid]
            
        return alerts_in_frame

    def get_active_telemetry(self):
        """Get live states for all actively tracked individuals."""
        return [t.to_dict() for t in self.trackers.values()]

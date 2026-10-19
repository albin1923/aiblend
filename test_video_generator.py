"""
Generates a realistic synthetic CCTV retail store surveillance video with shoplifting behavior.
Contains:
1. Store aisle background with shelves, items, aisle perspective.
2. Animated Person 1: Normal customer browsing items into shopping basket.
3. Animated Person 2: Shoplifter performing Reach -> Grab -> Pocket Concealment dwell sequence.
Used for immediate testing and verification without requiring external video files.
"""

import cv2
import numpy as np
import math
import os
from pathlib import Path


def create_retail_background(width=640, height=480):
    """Draws a convincing retail supermarket aisle background."""
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Store floor (polished linoleum tiles with perspective)
    for y in range(int(height * 0.45), height):
        ratio = (y - height * 0.45) / (height * 0.55)
        # Floor gradient
        color = (int(160 + 30 * ratio), int(165 + 30 * ratio), int(170 + 30 * ratio))
        cv2.line(bg, (0, y), (width, y), color, 1)
        
    # Floor tile grid lines (perspective lines towards vanishing point)
    vanish_x, vanish_y = width // 2, int(height * 0.40)
    for fx in range(-width, width * 2, 80):
        cv2.line(bg, (vanish_x, vanish_y), (fx, height), (130, 135, 140), 1, cv2.LINE_AA)
        
    # Ceiling & fluorescent light panels
    for y in range(0, int(height * 0.40)):
        ratio = y / (height * 0.40)
        c = int(80 + 40 * ratio)
        cv2.line(bg, (0, y), (width, y), (c, c, c), 1)
        
    # Fluorescent lights
    cv2.rectangle(bg, (int(width * 0.35), 10), (int(width * 0.65), 25), (240, 245, 255), -1)
    
    # Left Shelf Rack (Perspectival retail shelves with products)
    shelf_pts_left = np.array([
        [0, int(height * 0.20)],
        [int(width * 0.30), int(height * 0.38)],
        [int(width * 0.30), int(height * 0.85)],
        [0, height]
    ], np.int32)
    cv2.fillPoly(bg, [shelf_pts_left], (70, 75, 80))
    cv2.polylines(bg, [shelf_pts_left], True, (50, 55, 60), 2)
    
    # Shelves and products on left rack
    shelf_levels = [0.35, 0.50, 0.65, 0.80]
    product_colors = [(50, 50, 200), (200, 100, 50), (50, 180, 50), (200, 200, 50), (180, 50, 180)]
    
    for lvl in shelf_levels:
        y_start = int(height * (0.20 + (1.0 - 0.20) * lvl))
        y_end = int(height * (0.38 + (0.85 - 0.38) * lvl))
        cv2.line(bg, (0, y_start), (int(width * 0.30), y_end), (110, 115, 120), 4)
        
        # Draw merchandise boxes on shelf
        for bx in range(10, int(width * 0.26), 25):
            t = bx / (width * 0.30)
            by = int(y_start + (y_end - y_start) * t)
            col = product_colors[(bx // 25 + int(lvl*10)) % len(product_colors)]
            cv2.rectangle(bg, (bx, by - 22), (bx + 18, by - 2), col, -1)
            cv2.rectangle(bg, (bx, by - 22), (bx + 18, by - 2), (30, 30, 30), 1)

    # Right Shelf Rack
    shelf_pts_right = np.array([
        [width, int(height * 0.20)],
        [int(width * 0.70), int(height * 0.38)],
        [int(width * 0.70), int(height * 0.85)],
        [width, height]
    ], np.int32)
    cv2.fillPoly(bg, [shelf_pts_right], (70, 75, 80))
    cv2.polylines(bg, [shelf_pts_right], True, (50, 55, 60), 2)
    
    for lvl in shelf_levels:
        y_start = int(height * (0.20 + (1.0 - 0.20) * lvl))
        y_end = int(height * (0.38 + (0.85 - 0.38) * lvl))
        cv2.line(bg, (width, y_start), (int(width * 0.70), y_end), (110, 115, 120), 4)
        
        for bx in range(int(width * 0.72), width - 15, 25):
            t = (bx - width * 0.70) / (width * 0.30)
            by = int(y_end + (y_start - y_end) * t)
            col = product_colors[(bx // 25 + int(lvl*10)) % len(product_colors)]
            cv2.rectangle(bg, (bx, by - 22), (bx + 18, by - 2), col, -1)
            cv2.rectangle(bg, (bx, by - 22), (bx + 18, by - 2), (30, 30, 30), 1)
            
    # Store aisle sign
    cv2.rectangle(bg, (int(width * 0.42), int(height * 0.30)), (int(width * 0.58), int(height * 0.36)), (30, 120, 200), -1)
    cv2.putText(bg, "AISLE 4 - SNACKS", (int(width * 0.43), int(height * 0.34)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return bg


def draw_humanoid(img, cx, cy, scale=1.0, pose="idle", hand_pos=(0, 0), shirt_color=(40, 120, 180), pants_color=(50, 50, 60)):
    """
    Renders an anatomically articulated human figure so that YOLOv8-pose can reliably detect it.
    """
    head_r = int(14 * scale)
    torso_w = int(24 * scale)
    torso_h = int(55 * scale)
    leg_len = int(60 * scale)
    
    head_y = cy - torso_h - head_r
    
    # Shadow
    cv2.ellipse(img, (cx, cy + leg_len), (int(25 * scale), int(8 * scale)), 0, 0, 360, (60, 65, 70), -1)
    
    # Legs / Pants
    hip_y = cy
    l_foot_x, r_foot_x = cx - int(10 * scale), cx + int(10 * scale)
    cv2.line(img, (cx - int(7 * scale), hip_y), (l_foot_x, cy + leg_len), pants_color, int(10 * scale))
    cv2.line(img, (cx + int(7 * scale), hip_y), (r_foot_x, cy + leg_len), pants_color, int(10 * scale))
    
    # Shoes
    cv2.rectangle(img, (l_foot_x - int(8*scale), cy + leg_len - int(4*scale)), (l_foot_x + int(4*scale), cy + leg_len + int(4*scale)), (20, 20, 20), -1)
    cv2.rectangle(img, (r_foot_x - int(4*scale), cy + leg_len - int(4*scale)), (r_foot_x + int(8*scale), cy + leg_len + int(4*scale)), (20, 20, 20), -1)
    
    # Torso / Jacket
    cv2.rectangle(img, (cx - torso_w // 2, cy - torso_h), (cx + torso_w // 2, hip_y), shirt_color, -1)
    
    # Neck & Head
    cv2.line(img, (cx, cy - torso_h), (cx, head_y + head_r), (210, 180, 150), int(6 * scale))
    cv2.circle(img, (cx, head_y), head_r, (215, 185, 155), -1)  # skin
    cv2.ellipse(img, (cx, head_y - int(4 * scale)), (head_r, int(head_r * 0.7)), 0, 180, 360, (40, 30, 20), -1)  # hair
    
    # Arms
    shoulder_y = cy - torso_h + int(6 * scale)
    l_shoulder_x = cx - torso_w // 2
    r_shoulder_x = cx + torso_w // 2
    
    # Left arm (Idle at side or carrying basket)
    if pose == "basket":
        # Hand holding basket at hip
        elbow_l = (l_shoulder_x - int(12 * scale), shoulder_y + int(24 * scale))
        wrist_l = (l_shoulder_x - int(8 * scale), hip_y)
        cv2.line(img, (l_shoulder_x, shoulder_y), elbow_l, shirt_color, int(6 * scale))
        cv2.line(img, elbow_l, wrist_l, (215, 185, 155), int(5 * scale))
        # Shopping basket
        cv2.rectangle(img, (wrist_l[0] - int(15*scale), wrist_l[1]), (wrist_l[0] + int(15*scale), wrist_l[1] + int(20*scale)), (20, 150, 50), -1)
    else:
        elbow_l = (l_shoulder_x - int(8 * scale), shoulder_y + int(26 * scale))
        wrist_l = (l_shoulder_x - int(6 * scale), hip_y + int(6 * scale))
        cv2.line(img, (l_shoulder_x, shoulder_y), elbow_l, shirt_color, int(6 * scale))
        cv2.line(img, elbow_l, wrist_l, (215, 185, 155), int(5 * scale))
        
    # Right arm (Dynamic: Idle, Shelf Reaching, or Pocket Concealing)
    if pose == "reach":
        # Arm extended towards shelf (up & right)
        wrist_r = hand_pos
        elbow_r = ((r_shoulder_x + wrist_r[0]) // 2 + int(6*scale), (shoulder_y + wrist_r[1]) // 2 - int(8*scale))
        cv2.line(img, (r_shoulder_x, shoulder_y), elbow_r, shirt_color, int(6 * scale))
        cv2.line(img, elbow_r, wrist_r, (215, 185, 155), int(5 * scale))
        cv2.circle(img, wrist_r, int(5 * scale), (215, 185, 155), -1)
    elif pose == "pocket":
        # Hand inside pocket / waistband at hip
        wrist_r = hand_pos
        elbow_r = (r_shoulder_x + int(12 * scale), (shoulder_y + wrist_r[1]) // 2)
        cv2.line(img, (r_shoulder_x, shoulder_y), elbow_r, shirt_color, int(6 * scale))
        cv2.line(img, elbow_r, wrist_r, (215, 185, 155), int(5 * scale))
        cv2.circle(img, wrist_r, int(5 * scale), (215, 185, 155), -1)
    elif pose == "jacket":
        # Hand inside jacket flap
        wrist_r = hand_pos
        elbow_r = (r_shoulder_x + int(14 * scale), shoulder_y + int(16 * scale))
        cv2.line(img, (r_shoulder_x, shoulder_y), elbow_r, shirt_color, int(6 * scale))
        cv2.line(img, elbow_r, wrist_r, (215, 185, 155), int(5 * scale))
    else:
        # Idle arm
        elbow_r = (r_shoulder_x + int(8 * scale), shoulder_y + int(26 * scale))
        wrist_r = (r_shoulder_x + int(6 * scale), hip_y + int(6 * scale))
        cv2.line(img, (r_shoulder_x, shoulder_y), elbow_r, shirt_color, int(6 * scale))
        cv2.line(img, elbow_r, wrist_r, (215, 185, 155), int(5 * scale))


def generate_shoplifting_demo_video(output_path="data/demo_shoplifting.mp4", duration_sec=14, fps=25):
    """
    Generates a full MP4 surveillance video demonstrating both normal customer behavior
    and a shoplifter performing a reach-and-pocket concealment sequence.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    width, height = 640, 480
    total_frames = int(duration_sec * fps)
    
    # Try H.264 or MP4V codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    bg = create_retail_background(width, height)
    
    print(f"[DemoGenerator] Generating {total_frames} frames to {output_path}...")
    
    for frame_idx in range(total_frames):
        frame = bg.copy()
        t = frame_idx / fps  # Current second
        
        # -------------------------------------------------------------
        # PERSON 1: Regular Shopper (Left-Center aisle)
        # Browses peacefully, holds green shopping basket
        # -------------------------------------------------------------
        p1_x = int(width * 0.36 + 15 * math.sin(t * 0.8))
        p1_y = int(height * 0.62)
        draw_humanoid(frame, p1_x, p1_y, scale=1.1, pose="basket", 
                      shirt_color=(50, 130, 90), pants_color=(60, 60, 70))
        
        # -------------------------------------------------------------
        # PERSON 2: Shoplifter (Right aisle near snack shelves)
        # Timeline:
        # 0s - 3s:   Walks into frame, looks around (Idle)
        # 3s - 5.5s: Reaches up to right shelf (Reach State)
        # 5.5s - 6.5s: Quick pull-back into coat/pocket
        # 6.5s - 10.5s: Hand stays in pocket, lingers suspiciously (Concealment Dwell)
        # 10.5s - 14s: Walks away towards exit with hand concealed
        # -------------------------------------------------------------
        if t < 3.0:
            # Walking in from right
            p2_x = int(width * 0.68 - (t / 3.0) * 40)
            p2_y = int(height * 0.65)
            draw_humanoid(frame, p2_x, p2_y, scale=1.18, pose="idle",
                          shirt_color=(160, 45, 45), pants_color=(35, 35, 40))
        elif t < 5.5:
            # Reaching for item on shelf
            p2_x = int(width * 0.68 - 40)
            p2_y = int(height * 0.65)
            reach_progress = min(1.0, (t - 3.0) / 1.0)
            target_wrist_x = int(p2_x + 35 * reach_progress)
            target_wrist_y = int(p2_y - 75 * reach_progress)
            draw_humanoid(frame, p2_x, p2_y, scale=1.18, pose="reach",
                          hand_pos=(target_wrist_x, target_wrist_y),
                          shirt_color=(160, 45, 45), pants_color=(35, 35, 40))
        elif t < 6.5:
            # Rapid retraction from shelf to pocket
            p2_x = int(width * 0.68 - 40)
            p2_y = int(height * 0.65)
            retract_progress = (t - 5.5) / 1.0
            start_x, start_y = p2_x + 35, p2_y - 75
            end_x, end_y = p2_x + 12, p2_y + 12
            cur_wrist_x = int(start_x + (end_x - start_x) * retract_progress)
            cur_wrist_y = int(start_y + (end_y - start_y) * retract_progress)
            draw_humanoid(frame, p2_x, p2_y, scale=1.18, pose="pocket",
                          hand_pos=(cur_wrist_x, cur_wrist_y),
                          shirt_color=(160, 45, 45), pants_color=(35, 35, 40))
        elif t < 10.5:
            # Prolonged concealment in pocket (Dwell state)
            p2_x = int(width * 0.68 - 40 + 4 * math.sin(t * 2))
            p2_y = int(height * 0.65)
            pocket_pos = (p2_x + 12, p2_y + 12)
            draw_humanoid(frame, p2_x, p2_y, scale=1.18, pose="pocket",
                          hand_pos=pocket_pos,
                          shirt_color=(160, 45, 45), pants_color=(35, 35, 40))
        else:
            # Walking away with hand still concealed in pocket
            walk_progress = (t - 10.5) / 3.5
            p2_x = int(width * 0.68 - 40 - walk_progress * 120)
            p2_y = int(height * 0.65 + walk_progress * 15)
            pocket_pos = (p2_x + 12, p2_y + 12)
            draw_humanoid(frame, p2_x, p2_y, scale=1.18 + walk_progress * 0.1, pose="pocket",
                          hand_pos=pocket_pos,
                          shirt_color=(160, 45, 45), pants_color=(35, 35, 40))
                          
        # Add subtle CCTV camera timestamp and scanline effect
        timestamp_str = f"2026-10-19 15:24:{int(t):02d}:{int((t % 1.0)*100):02d}"
        cv2.putText(frame, timestamp_str, (15, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, "CAM 04 - AISLE 04 NORTH", (width - 210, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
                    
        out.write(frame)
        
    out.release()
    print(f"[DemoGenerator] Demo video generated successfully: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    generate_shoplifting_demo_video()

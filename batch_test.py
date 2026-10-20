"""
Automated Batch Evaluation Suite for SENTINEL-CV on Archive Surveillance Videos.
Evaluates both 'normal' and 'shoplifting' video folders from 'archive (1)'.
Calculates:
- True Positives, False Positives, True Negatives, False Negatives
- Accuracy, Precision, Recall, F1-Score
- Generates detailed CSV summary report
"""

import os
import sys
import time
import csv
import cv2
from pathlib import Path
from typing import List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from detector import ShopliftingDetector
import config


def evaluate_video(detector: ShopliftingDetector, video_path: Path, frame_step: int = 2) -> Dict[str, Any]:
    """Evaluates a single video file through the shoplifting detection pipeline."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "file": video_path.name,
            "error": "Failed to open video",
            "max_threat": 0.0,
            "alert_count": 0,
            "reasons": []
        }
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    detector.reset()
    max_threat = 0.0
    all_reasons = []
    alerts_count = 0
    frame_idx = 0
    processed_count = 0
    
    t_start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        # Subsample frames for speed if frame_step > 1
        if frame_step > 1 and (frame_idx % frame_step != 0):
            continue
            
        processed_count += 1
        sim_time = frame_idx / fps
        
        _, alerts, telemetry = detector.process_frame(frame, sim_time)
        if telemetry["max_threat"] > max_threat:
            max_threat = telemetry["max_threat"]
            
        if alerts:
            alerts_count += len(alerts)
            for a in alerts:
                all_reasons.extend(a.get("reasons", []))
                
    cap.release()
    elapsed = max(0.001, time.time() - t_start)
    
    # Classify prediction based on threat threshold
    is_detected_shoplifting = max_threat >= (config.THREAT_SUSPICIOUS * 100)
    
    return {
        "file": video_path.name,
        "frames_total": total_frames,
        "frames_processed": processed_count,
        "duration_sec": round(total_frames / fps, 1) if fps > 0 else 0,
        "elapsed_sec": round(elapsed, 2),
        "fps": round(processed_count / elapsed, 1),
        "max_threat": round(max_threat, 1),
        "predicted_alert": is_detected_shoplifting,
        "alerts_count": alerts_count,
        "reasons": list(set(all_reasons))
    }


def run_batch_evaluation(max_videos_per_class: int = 10, frame_step: int = 2, output_csv: str = "data/evaluation_report.csv"):
    """
    Runs batch evaluation across normal and shoplifting folders.
    If max_videos_per_class <= 0, evaluates ALL videos in the folders.
    """
    archive_dir = ROOT_DIR / "archive (1)"
    normal_dir = archive_dir / "normal"
    shop_dir = archive_dir / "shoplifting"
    
    if not archive_dir.exists():
        print(f"[Error] Dataset directory '{archive_dir}' not found.")
        return
        
    normal_files = sorted(list(normal_dir.glob("*.mp4")))
    shop_files = sorted(list(shop_dir.glob("*.mp4")))
    
    if max_videos_per_class > 0:
        normal_files = normal_files[:max_videos_per_class]
        shop_files = shop_files[:max_videos_per_class]
        
    total_eval = len(normal_files) + len(shop_files)
    print("=" * 70)
    print(f"  SENTINEL-CV BATCH SURVEILLANCE VIDEO EVALUATION")
    print(f"  Evaluating {len(shop_files)} Shoplifting + {len(normal_files)} Normal videos (Total: {total_eval})")
    print(f"  Frame Step: {frame_step} | Alert Threshold: {int(config.THREAT_SUSPICIOUS * 100)}%")
    print("=" * 70)
    
    detector = ShopliftingDetector()
    results = []
    
    # Confusion Matrix counts
    tp, fp, tn, fn = 0, 0, 0, 0
    
    # 1. Test Shoplifting Videos (Positive class)
    print("\n[PHASE 1/2] Testing Shoplifting Videos (Ground Truth: Positive)...")
    for i, vpath in enumerate(shop_files, 1):
        res = evaluate_video(detector, vpath, frame_step=frame_step)
        res["ground_truth"] = "shoplifting"
        is_pos = res["predicted_alert"]
        if is_pos:
            tp += 1
            status = "TRUE POSITIVE (FLAGGED)"
        else:
            fn += 1
            status = "FALSE NEGATIVE (MISSED)"
        results.append(res)
        reasons_str = "; ".join(res["reasons"][:2]) if res["reasons"] else "None"
        print(f"  [{i:02d}/{len(shop_files):02d}] {res['file']:<20} Max Threat: {res['max_threat']:>5.1f}% -> {status} [{reasons_str}]")
        
    # 2. Test Normal Videos (Negative class)
    print("\n[PHASE 2/2] Testing Normal Customer Videos (Ground Truth: Negative)...")
    for i, vpath in enumerate(normal_files, 1):
        res = evaluate_video(detector, vpath, frame_step=frame_step)
        res["ground_truth"] = "normal"
        is_pos = res["predicted_alert"]
        if is_pos:
            fp += 1
            status = "FALSE POSITIVE (FALSE ALARM)"
        else:
            tn += 1
            status = "TRUE NEGATIVE (CLEAN)"
        results.append(res)
        print(f"  [{i:02d}/{len(normal_files):02d}] {res['file']:<20} Max Threat: {res['max_threat']:>5.1f}% -> {status}")
        
    # Compute Metrics
    accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2 * (precision * recall) / max(0.0001, (precision + recall))
    
    # Save CSV Report
    out_path = ROOT_DIR / output_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Video File", "Ground Truth", "Predicted Alert", "Max Threat %", "Alert Count", "Duration (s)", "Processed FPS", "Key Reasons"])
        for r in results:
            writer.writerow([
                r["file"],
                r["ground_truth"],
                "ALERT" if r.get("predicted_alert") else "SAFE",
                r["max_threat"],
                r["alerts_count"],
                r["duration_sec"],
                r["fps"],
                "; ".join(r["reasons"])
            ])
            
    print("\n" + "=" * 70)
    print("                     EVALUATION SUMMARY REPORT                        ")
    print("=" * 70)
    print(f"  Total Videos Tested:       {len(results)}")
    print(f"  True Positives  (TP):      {tp:>3} (Correctly caught shoplifting)")
    print(f"  False Positives (FP):      {fp:>3} (False alarms on normal shoppers)")
    print(f"  True Negatives  (TN):      {tn:>3} (Correctly ignored normal shoppers)")
    print(f"  False Negatives (FN):      {fn:>3} (Missed shoplifting incidents)")
    print("-" * 70)
    print(f"  Accuracy:                  {accuracy * 100:.1f}%")
    print(f"  Recall (Detection Rate):   {recall * 100:.1f}%")
    print(f"  Precision:                 {precision * 100:.1f}%")
    print(f"  F1-Score:                  {f1 * 100:.1f}%")
    print(f"  Detailed Report Saved to:  {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate SENTINEL-CV on archive videos.")
    parser.add_argument("--count", type=int, default=10, help="Number of videos per category to test (0 = test all 182).")
    parser.add_argument("--step", type=int, default=2, help="Frame step for inference speedup (1 = full frames, 2 = 15 FPS).")
    parser.add_argument("--out", type=str, default="data/evaluation_report.csv", help="Output CSV filepath.")
    args = parser.parse_args()
    
    run_batch_evaluation(max_videos_per_class=args.count, frame_step=args.step, output_csv=args.out)

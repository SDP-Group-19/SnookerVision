"""Auto-label snooker ball frames using HSV color detection.

Tuned for the specific camera/table/LED setup observed in the captured frames:
- Overhead camera at 1280x720
- Bright green table cloth
- LED strip cluster in lower-left (cyan dots) — explicitly rejected
- Balls are ~20-35px diameter

Class mapping (matches bbox_colors order):
  0=red  1=white  2=yellow  3=green  4=blue  5=brown  6=pink  7=black
"""
import argparse
import os
import glob
import random
import cv2
import numpy as np
from collections import defaultdict

CLASS_NAMES = ["red", "white", "yellow", "green", "blue", "brown", "pink", "black"]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

DRAW_COLORS = {
    "red":    (0, 0, 255),
    "white":  (255, 255, 255),
    "yellow": (0, 255, 255),
    "green":  (0, 200, 0),
    "blue":   (255, 0, 0),
    "brown":  (42, 42, 165),
    "pink":   (203, 192, 255),
    "black":  (120, 120, 120),
}


def is_cyan_led(h, s, v):
    """Reject cyan LED dots. LEDs are bright vivid cyan, NOT dark navy blue."""
    # Cyan LEDs: H ~80-105, very bright V>180, saturated
    if 75 < h < 110 and s > 80 and v > 175:
        return True
    return False


def is_skin(h, s, v):
    """Detect skin tones to reject hands."""
    if 0 < h < 25 and 30 < s < 180 and v > 100:
        return True
    return False


def classify_ball_color(hsv_mean, bgr_mean):
    """Classify a snooker ball by its mean color. Tuned for this specific table."""
    h, s, v = hsv_mean
    b, g, r = bgr_mean

    # Reject LEDs first
    if is_cyan_led(h, s, v):
        return None

    # Reject skin (hands)
    if is_skin(h, s, v):
        return None

    # Black: very dark, low saturation
    if v < 65 and s < 120:
        return "black"

    # White: very low saturation, bright
    if s < 40 and v > 155:
        return "white"

    # Red: hue wraps around 0/180, saturated, not too dark
    if (h < 12 or h > 160) and s > 80 and v > 70:
        # Distinguish from pink: red is more saturated and deeper
        if s > 100 and v < 220:
            return "red"
        # Could be pink if less saturated and brighter
        if s < 130 and v > 150:
            return "pink"
        return "red"

    # Yellow: warm hue, very bright and saturated
    if 15 < h < 35 and s > 80 and v > 140:
        return "yellow"

    # Brown: warm hue like yellow but darker
    if 8 < h < 25 and s > 40 and 50 < v < 160:
        return "brown"

    # Blue (DARK navy, not cyan LED): H~100-130, NOT bright
    if 100 < h < 135 and s > 60 and v < 180:
        return "blue"

    # Green ball (not table — ball is brighter/smaller circle on green table)
    # This is tricky. Green ball will have similar hue to table but be a raised circle.
    if 35 < h < 85 and s > 80 and v > 90:
        return "green"

    # Pink: reddish-purple, less saturated than red, bright
    if 140 < h < 175 and s > 20 and v > 130:
        return "pink"

    return None


def detect_balls(frame, min_area=200, max_area=3000):
    """Detect snooker balls using HSV + contour analysis."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_img, w_img = frame.shape[:2]

    # --- Step 1: Mask out the green table to find non-green objects ---
    table_lower = np.array([30, 25, 25])
    table_upper = np.array([92, 255, 230])
    table_mask = cv2.inRange(hsv, table_lower, table_upper)
    non_table = cv2.bitwise_not(table_mask)

    # --- Step 2: Also mask out areas outside the table ---
    # The table occupies roughly the center of the frame
    # Mask a border to ignore cushions, electronics, people at edges
    border_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    # Table region (approximate, generous): x=180..1080, y=80..640
    cv2.rectangle(border_mask, (180, 80), (1080, 640), 255, -1)
    non_table = cv2.bitwise_and(non_table, border_mask)

    # --- Step 3: Clean up noise ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    non_table = cv2.morphologyEx(non_table, cv2.MORPH_OPEN, kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    non_table = cv2.morphologyEx(non_table, cv2.MORPH_CLOSE, kernel2)

    contours, _ = cv2.findContours(non_table, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        # Circularity: balls are round
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.60:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)

        # Aspect ratio: balls are roughly square
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect > 1.6:
            continue

        # Min enclosing circle fill check
        (_, _), enc_radius = cv2.minEnclosingCircle(contour)
        enc_area = np.pi * enc_radius * enc_radius
        fill_ratio = area / (enc_area + 1e-6)
        if fill_ratio < 0.50:
            continue

        # Sample color from center of the ball
        cx, cy = x + bw // 2, y + bh // 2
        radius = max(2, min(bw, bh) // 5)
        color_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.circle(color_mask, (cx, cy), radius, 255, -1)
        mean_hsv = cv2.mean(hsv, mask=color_mask)[:3]
        mean_bgr = cv2.mean(frame, mask=color_mask)[:3]

        # Reject bright point sources (LED glow)
        _, s_val, v_val = mean_hsv
        if v_val > 240 and s_val < 50:
            continue

        color_name = classify_ball_color(mean_hsv, mean_bgr)
        if color_name is None:
            continue

        # Expand bbox slightly for training
        pad = int(max(bw, bh) * 0.2)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + bw + pad)
        y2 = min(h_img, y + bh + pad)

        detections.append({
            "class": color_name,
            "class_id": CLASS_MAP[color_name],
            "bbox": (x1, y1, x2, y2),
            "center": (cx, cy),
            "area": area,
            "circularity": circularity,
            "hsv": mean_hsv,
        })

    # --- Step 4: Post-filtering ---
    detections = _deduplicate(detections)
    detections = _reject_led_clusters(detections)
    detections = _enforce_snooker_limits(detections)

    return detections


def _deduplicate(detections, iou_thresh=0.5):
    """Remove overlapping detections, keep the rounder one."""
    keep = []
    used = set()
    for i, d in enumerate(detections):
        if i in used:
            continue
        best = d
        for j, other in enumerate(detections):
            if j <= i or j in used:
                continue
            if _iou(d["bbox"], other["bbox"]) > iou_thresh:
                used.add(j)
                if other["circularity"] > best["circularity"]:
                    best = other
        keep.append(best)
    return keep


def _reject_led_clusters(detections, y_tolerance=30, min_in_line=3):
    """Remove detections that form a line — these are LED strips."""
    if len(detections) < min_in_line:
        return detections

    # Group all detections by Y proximity (LEDs sit in horizontal rows)
    reject = set()
    centers = [(i, d["center"]) for i, d in enumerate(detections)]

    for i in range(len(centers)):
        line = [centers[i][0]]
        for j in range(i + 1, len(centers)):
            if abs(centers[j][1][1] - centers[i][1][1]) <= y_tolerance:
                line.append(centers[j][0])
        # If 3+ objects in a horizontal line AND they're all small, reject
        if len(line) >= min_in_line:
            all_small = all(detections[idx]["area"] < 500 for idx in line)
            if all_small:
                reject.update(line)

    return [d for i, d in enumerate(detections) if i not in reject]


def _enforce_snooker_limits(detections):
    """Enforce snooker ball count limits: 1 of each colour, up to 15 reds."""
    limits = {
        "white": 1, "black": 1, "yellow": 1, "green": 1,
        "blue": 1, "brown": 1, "pink": 1, "red": 15,
    }
    # Sort by circularity (best first)
    detections.sort(key=lambda d: d["circularity"], reverse=True)
    counts = defaultdict(int)
    keep = []
    for d in detections:
        c = d["class"]
        if counts[c] < limits.get(c, 1):
            keep.append(d)
            counts[c] += 1
    return keep


def _iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def to_yolo_line(det, img_w, img_h):
    x1, y1, x2, y2 = det["bbox"]
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{det['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def draw_detections(frame, detections):
    vis = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = DRAW_COLORS.get(d["class"], (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        h, s, v = d["hsv"]
        label = f"{d['class']} c={d['circularity']:.2f} h={h:.0f}"
        cv2.putText(vis, label, (x1, y1 - 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="final/scripts/dataset/raw_frames")
    parser.add_argument("--out", default="final/scripts/dataset")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--min-area", type=int, default=200)
    parser.add_argument("--max-area", type=int, default=3000)
    parser.add_argument("--review", action="store_true",
                        help="Visual review: SPACE=keep, D=discard, Q=quit")
    args = parser.parse_args()

    frame_paths = sorted(
        glob.glob(os.path.join(args.frames, "*.jpg"))
        + glob.glob(os.path.join(args.frames, "*.png"))
    )
    if not frame_paths:
        print(f"No images found in {args.frames}/")
        return

    for split in ["train", "val"]:
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

    random.shuffle(frame_paths)
    val_count = max(1, int(len(frame_paths) * args.val_split))
    val_paths = set(frame_paths[:val_count])

    total_labels = 0
    total_frames_used = 0
    color_counts = {name: 0 for name in CLASS_NAMES}

    for i, fpath in enumerate(frame_paths):
        frame = cv2.imread(fpath)
        if frame is None:
            continue

        detections = detect_balls(frame, min_area=args.min_area, max_area=args.max_area)
        h_img, w_img = frame.shape[:2]
        lines = [to_yolo_line(d, w_img, h_img) for d in detections]

        if args.review:
            vis = draw_detections(frame, detections)
            info = f"[{i+1}/{len(frame_paths)}] {len(detections)} balls"
            cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(vis, "SPACE=keep  D=discard  Q=quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.imshow("Review Labels", vis)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                continue

        # Include frames with 0 detections as negative examples (empty label file)
        split = "val" if fpath in val_paths else "train"
        name = os.path.splitext(os.path.basename(fpath))[0]

        dst_img = os.path.join(args.out, "images", split, f"{name}.jpg")
        cv2.imwrite(dst_img, frame)

        dst_lbl = os.path.join(args.out, "labels", split, f"{name}.txt")
        with open(dst_lbl, "w") as f:
            if lines:
                f.write("\n".join(lines) + "\n")
            # Empty file for no-ball frames (negative examples)

        for d in detections:
            color_counts[d["class"]] += 1
        total_labels += len(lines)
        total_frames_used += 1

        balls_str = ", ".join(d["class"] for d in detections) if detections else "(empty)"
        print(f"  [{i+1}/{len(frame_paths)}] {name}: {balls_str}")

    if args.review:
        cv2.destroyAllWindows()

    # Write data.yaml
    yaml_path = os.path.join(args.out, "data.yaml")
    abs_out = os.path.abspath(args.out).replace("\\", "/")
    with open(yaml_path, "w") as f:
        f.write(f"path: {abs_out}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    print(f"\n{'='*50}")
    print(f"Dataset: {args.out}/")
    print(f"Frames used: {total_frames_used}")
    print(f"Total labels: {total_labels}")
    print(f"Per class: {color_counts}")
    print(f"data.yaml: {yaml_path}")
    print(f"\nNext: python final/scripts/train_model.py")


if __name__ == "__main__":
    main()

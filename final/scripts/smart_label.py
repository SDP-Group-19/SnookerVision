"""Smart auto-labeler: uses existing YOLO model for ball LOCATION + HSV for COLOR.

The existing best_color.pt model knows what balls look like (shape/size),
but struggles with colors on the new table. This script:
1. Runs the model at very low confidence to find ALL ball candidates
2. Samples the actual pixel color at each detection center
3. Reclassifies by HSV color analysis
4. Rejects LEDs (cyan, bright point sources, collinear clusters)

Usage:
  python final/scripts/smart_label.py
  python final/scripts/smart_label.py --review
"""
import argparse
import os
import glob
import random
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

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
    "reject": (0, 0, 100),
}


def classify_by_hsv(hsv_mean, bgr_mean):
    """Classify snooker ball color from sampled pixel values."""
    h, s, v = hsv_mean
    b, g, r = bgr_mean

    # --- Reject non-ball objects ---
    # Cyan LED: bright, vivid cyan
    if 75 < h < 115 and s > 70 and v > 170:
        return None
    # Skin (hands)
    if 0 < h < 25 and 30 < s < 180 and v > 100:
        return None
    # Very bright white glow (LED reflection)
    if v > 245 and s < 30:
        return None
    # Green table cloth (not a ball)
    if 35 < h < 85 and s > 40 and v > 60:
        # Green BALL would be detected by YOLO as a ball shape,
        # but need to distinguish from table bleed-through.
        # If it looks like the table, reject. Real green ball is brighter.
        if v < 150 and s > 60:
            return None  # Likely table cloth, not green ball
        return "green"

    # --- Classify ball colors ---
    if v < 60 and s < 100:
        return "black"
    if s < 40 and v > 150:
        return "white"
    if (h < 10 or h > 160) and s > 80 and v > 70:
        if s > 100:
            return "red"
        if v > 150 and s < 120:
            return "pink"
        return "red"
    if 15 < h < 35 and s > 70 and v > 130:
        return "yellow"
    if 8 < h < 25 and s > 40 and 45 < v < 155:
        return "brown"
    if 100 < h < 135 and s > 50 and v < 185:
        return "blue"
    if 140 < h < 178 and s > 20 and v > 120:
        return "pink"

    return None


def sample_center_color(frame, cx, cy, radius=5):
    """Sample mean HSV and BGR from a small circle at the detection center."""
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = int(np.clip(cx, 0, w - 1))
    cy = int(np.clip(cy, 0, h - 1))
    radius = max(2, radius)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv, mask=mask)[:3]
    mean_bgr = cv2.mean(frame, mask=mask)[:3]
    return mean_hsv, mean_bgr


def detect_and_relabel(model, frame, conf_threshold=0.05):
    """Run YOLO at low confidence, then reclassify each detection by color."""
    results = model.predict(
        source=frame,
        verbose=False,
        conf=conf_threshold,
        iou=0.4,
        imgsz=640,
        stream=False,
    )

    result = results[0] if results else None
    if result is None or result.boxes is None:
        return []

    boxes = result.boxes
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
    h_img, w_img = frame.shape[:2]

    detections = []
    for bbox in xyxy:
        xmin, ymin, xmax, ymax = bbox.tolist()
        bw = xmax - xmin
        bh = ymax - ymin
        area = bw * bh

        # Size filter: reject very small (noise) and very large (hands/table)
        if area < 100 or area > 5000:
            continue

        # Aspect ratio: balls are roughly square
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect > 2.0:
            continue

        cx = (xmin + xmax) // 2
        cy = (ymin + ymax) // 2

        # Sample actual color
        radius = max(2, min(bw, bh) // 5)
        mean_hsv, mean_bgr = sample_center_color(frame, cx, cy, radius)

        color_name = classify_by_hsv(mean_hsv, mean_bgr)
        if color_name is None:
            continue

        detections.append({
            "class": color_name,
            "class_id": CLASS_MAP[color_name],
            "bbox": (xmin, ymin, xmax, ymax),
            "center": (cx, cy),
            "area": area,
            "hsv": mean_hsv,
        })

    # Post-process
    detections = _deduplicate(detections)
    detections = _reject_led_clusters(detections)
    detections = _enforce_snooker_limits(detections)

    return detections


def _deduplicate(detections, iou_thresh=0.5):
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
                if other["area"] > best["area"]:
                    best = other
        keep.append(best)
    return keep


def _reject_led_clusters(detections, y_tolerance=30, min_in_line=3):
    """Remove detections forming horizontal lines (LED strips)."""
    if len(detections) < min_in_line:
        return detections
    reject = set()
    for i in range(len(detections)):
        line = [i]
        for j in range(i + 1, len(detections)):
            if abs(detections[j]["center"][1] - detections[i]["center"][1]) <= y_tolerance:
                line.append(j)
        if len(line) >= min_in_line:
            all_small = all(detections[idx]["area"] < 600 for idx in line)
            if all_small:
                reject.update(line)
    return [d for i, d in enumerate(detections) if i not in reject]


def _enforce_snooker_limits(detections):
    limits = {"white": 1, "black": 1, "yellow": 1, "green": 1,
              "blue": 1, "brown": 1, "pink": 1, "red": 15}
    detections.sort(key=lambda d: d["area"], reverse=True)
    counts = defaultdict(int)
    keep = []
    for d in detections:
        if counts[d["class"]] < limits.get(d["class"], 1):
            keep.append(d)
            counts[d["class"]] += 1
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
        hh, ss, vv = d["hsv"]
        label = f"{d['class']} H={hh:.0f} S={ss:.0f} V={vv:.0f}"
        cv2.putText(vis, label, (x1, max(y1 - 5, 12)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="final/scripts/dataset/raw_frames")
    parser.add_argument("--out", default="final/scripts/dataset")
    parser.add_argument("--model", default="final/src/snookervision/data/model/best_color.pt")
    parser.add_argument("--conf", type=float, default=0.05,
                        help="Very low confidence to catch all ball candidates")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--review", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        return

    frame_paths = sorted(
        glob.glob(os.path.join(args.frames, "*.jpg"))
        + glob.glob(os.path.join(args.frames, "*.png"))
    )
    if not frame_paths:
        print(f"No images in {args.frames}/")
        return

    print(f"Loading model {args.model}...")
    model = YOLO(args.model, task="detect")
    print(f"Model classes: {model.names}")
    print(f"Processing {len(frame_paths)} frames at conf={args.conf}...")

    for split in ["train", "val"]:
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

    random.seed(42)
    random.shuffle(frame_paths)
    val_count = max(1, int(len(frame_paths) * args.val_split))
    val_paths = set(frame_paths[:val_count])

    total_labels = 0
    color_counts = defaultdict(int)

    for i, fpath in enumerate(frame_paths):
        frame = cv2.imread(fpath)
        if frame is None:
            continue

        detections = detect_and_relabel(model, frame, conf_threshold=args.conf)
        h_img, w_img = frame.shape[:2]
        lines = [to_yolo_line(d, w_img, h_img) for d in detections]

        if args.review:
            vis = draw_detections(frame, detections)
            info = f"[{i+1}/{len(frame_paths)}] {len(detections)} balls"
            cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, "SPACE=keep D=discard Q=quit", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("Smart Label Review", vis)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                continue

        split = "val" if fpath in val_paths else "train"
        name = os.path.splitext(os.path.basename(fpath))[0]

        cv2.imwrite(os.path.join(args.out, "images", split, f"{name}.jpg"), frame)
        with open(os.path.join(args.out, "labels", split, f"{name}.txt"), "w") as f:
            if lines:
                f.write("\n".join(lines) + "\n")

        for d in detections:
            color_counts[d["class"]] += 1
        total_labels += len(lines)

        balls = ", ".join(d["class"] for d in detections) if detections else "(empty)"
        print(f"  [{i+1}/{len(frame_paths)}] {name}: {balls}")

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
    print(f"Total labels: {total_labels}")
    print(f"Per class: {dict(color_counts)}")
    print(f"Dataset: {args.out}/")
    print(f"\nNext: python final/scripts/train_model.py")


if __name__ == "__main__":
    main()

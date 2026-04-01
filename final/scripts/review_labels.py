"""Review YOLO labels visually — draws bounding boxes on each frame.

Usage:
  python final/scripts/review_labels.py

Controls:
  SPACE / RIGHT ARROW = next frame
  LEFT ARROW          = previous frame
  D                   = delete this label file (mark as bad)
  Q                   = quit
"""
import os
import glob
import cv2

CLASS_NAMES = ["red", "white", "yellow", "green", "blue", "brown", "pink", "black"]
COLORS = {
    "red":    (0, 0, 255),
    "white":  (255, 255, 255),
    "yellow": (0, 255, 255),
    "green":  (0, 200, 0),
    "blue":   (255, 0, 0),
    "brown":  (42, 42, 165),
    "pink":   (203, 192, 255),
    "black":  (120, 120, 120),
}

FRAMES_DIR = "final/scripts/dataset/raw_frames"


def load_labels(txt_path, img_w, img_h):
    labels = []
    if not os.path.exists(txt_path):
        return labels
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            # Convert normalized to pixel
            px_cx = int(cx * img_w)
            px_cy = int(cy * img_h)
            px_w = int(w * img_w)
            px_h = int(h * img_h)
            x1 = px_cx - px_w // 2
            y1 = px_cy - px_h // 2
            x2 = px_cx + px_w // 2
            y2 = px_cy + px_h // 2
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
            labels.append({"name": name, "bbox": (x1, y1, x2, y2)})
    return labels


def draw(frame, labels, idx, total, fname):
    vis = frame.copy()
    for lbl in labels:
        x1, y1, x2, y2 = lbl["bbox"]
        color = COLORS.get(lbl["name"], (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, lbl["name"], (x1, y1 - 6),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Info bar
    info = f"[{idx+1}/{total}] {fname}  |  {len(labels)} labels  |  SPACE=next  LEFT=prev  D=delete  Q=quit"
    cv2.putText(vis, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    return vis


def main():
    images = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
    if not images:
        print(f"No images in {FRAMES_DIR}")
        return

    print(f"Found {len(images)} images. Opening review window...")
    cv2.namedWindow("Review Labels", cv2.WINDOW_NORMAL)

    idx = 0
    deleted = 0
    while 0 <= idx < len(images):
        img_path = images[idx]
        fname = os.path.basename(img_path)
        txt_path = img_path.replace(".jpg", ".txt").replace(".png", ".txt")

        frame = cv2.imread(img_path)
        if frame is None:
            idx += 1
            continue
        h, w = frame.shape[:2]
        labels = load_labels(txt_path, w, h)
        vis = draw(frame, labels, idx, len(images), fname)
        cv2.imshow("Review Labels", vis)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            if os.path.exists(txt_path):
                os.remove(txt_path)
                deleted += 1
                print(f"  Deleted {txt_path}")
            idx += 1
        elif key == 81 or key == 2:  # left arrow
            idx = max(0, idx - 1)
        else:  # space, right arrow, any other key
            idx += 1

    cv2.destroyAllWindows()
    print(f"\nReview done. Deleted {deleted} label files.")


if __name__ == "__main__":
    main()

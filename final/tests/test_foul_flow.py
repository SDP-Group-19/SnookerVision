"""Continuous visual test for foul -> reposition flow.

Runs in a loop: monitors the table, detects shots via ball movement,
and when a foul is triggered (press F) shows FOUL overlay, waits for
potted balls to be off table, then guides reposition one by one.

No ESP32 needed — everything is drawn on the OpenCV window.

Usage:
  python final/tests/test_foul_flow.py ^
      --camera-source http://192.168.137.172:8000/stream.mjpg

Controls:
  f = trigger foul (snapshots current positions, enters foul mode)
  s = skip current step
  r = restart (go back to monitoring)
  q = quit
"""

import sys
import time
import math
import argparse
import logging
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snookervision.detection import DetectionModel
from snookervision.core import config

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BALL_BGR = {
    "white":  (255, 255, 255),
    "red":    (0, 0, 255),
    "yellow": (0, 255, 255),
    "green":  (0, 255, 0),
    "brown":  (19, 69, 139),
    "blue":   (255, 0, 0),
    "pink":   (180, 105, 255),
    "black":  (100, 100, 100),
}

BALL_NAMES = {"white", "red", "yellow", "green", "brown", "blue", "pink", "black"}

WIN = "Foul Flow Test"

# States
MONITORING = "MONITORING"
FOUL_DISPLAY = "FOUL_DISPLAY"
FOUL_WAIT_OFF = "FOUL_WAIT_OFF"
REPOSITIONING = "REPOSITIONING"
REPOSITION_HOLD = "REPOSITION_HOLD"


def parse_args():
    p = argparse.ArgumentParser(description="Continuous foul -> reposition visual test")
    p.add_argument("--camera-source", type=str, default=None)
    p.add_argument("--camera-port", type=int, default=1)
    p.add_argument("--camera-width", type=int, default=1280)
    p.add_argument("--camera-height", type=int, default=720)
    p.add_argument("--threshold", type=int, default=40,
                    help="Pixel distance to confirm placement (default 40)")
    p.add_argument("--confirm-frames", type=int, default=5)
    p.add_argument("--foul-display-secs", type=float, default=3.0,
                    help="Minimum seconds to show FOUL overlay (default 3)")
    p.add_argument("--hold-secs", type=float, default=3.0,
                    help="Seconds to hold green confirmation (default 3)")
    p.add_argument("--detector-imgsz", type=int, default=640)
    return p.parse_args()


def open_camera(args):
    if args.camera_source:
        logger.info(f"Opening camera: {args.camera_source}")
        cap = cv2.VideoCapture(args.camera_source, apiPreference=cv2.CAP_ANY)
    else:
        logger.info(f"Opening local camera port {args.camera_port}")
        cap = cv2.VideoCapture(args.camera_port)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    if not cap.isOpened():
        logger.error("Failed to open camera")
        sys.exit(1)
    for _ in range(10):
        cap.read()
    return cap


def detect_balls(model, frame):
    detections, _ = model.detect(frame)
    if not detections:
        return {}
    balls = {}
    for det in detections:
        label = det["label"]
        if label in BALL_NAMES:
            cx, cy = det["center"]
            balls.setdefault(label, []).append((int(cx), int(cy)))
    return balls


# ---- Drawing helpers ----

def draw_status_bar(frame, text):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2)


def draw_ball_labels(frame, balls):
    for color, positions in balls.items():
        bgr = BALL_BGR.get(color, (255, 255, 255))
        for (x, y) in positions:
            cv2.circle(frame, (x, y), 12, bgr, 2)
            cv2.putText(frame, color, (x + 15, y + 5),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1)


def draw_foul_overlay(frame):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (0, h // 3), (w, 2 * h // 3), (0, 0, 200), -1)
    frame_out = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "FOUL"
    scale = h / 200.0
    thickness = max(3, int(scale * 3))
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    tx = (w - text_size[0]) // 2
    ty = (h + text_size[1]) // 2
    cv2.putText(frame_out, text, (tx, ty), font, scale, (255, 255, 255), thickness + 4)
    cv2.putText(frame_out, text, (tx, ty), font, scale, (0, 0, 255), thickness)
    return frame_out


def draw_target_circle(frame, x, y, color_bgr, radius=25, label=""):
    pulse = int(10 * math.sin(time.time() * 6))
    r = radius + pulse
    cv2.circle(frame, (x, y), r + 4, (255, 255, 255), 2)
    cv2.circle(frame, (x, y), r, color_bgr, 3)
    cv2.line(frame, (x - r, y), (x + r, y), color_bgr, 1)
    cv2.line(frame, (x, y - r), (x, y + r), color_bgr, 1)
    if label:
        cv2.putText(frame, label, (x + r + 8, y + 5),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)


def draw_confirmed(frame, x, y, color_name):
    cv2.circle(frame, (x, y), 30, (0, 255, 0), 3)
    cv2.putText(frame, f"{color_name.upper()} OK", (x + 35, y + 5),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def main():
    args = parse_args()
    config.detector_imgsz = args.detector_imgsz
    config.process_every_n_frames = 1

    cap = open_camera(args)

    logger.info("Loading YOLO model...")
    model = DetectionModel()
    if model.model is None:
        logger.error("Failed to load detection model")
        sys.exit(1)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    # State
    state = MONITORING
    snapshot_positions = {}      # ball positions when foul was triggered
    reposition_queue = []        # list of {color_name, x, y, bgr}
    reposition_index = 0
    confirm_count = 0
    foul_start_time = 0.0
    hold_start_time = 0.0
    foul_colors_to_check = set()

    print("=" * 60)
    print("MONITORING — press F to trigger a foul at any time")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Camera read failed")
            break

        current_balls = detect_balls(model, frame)
        display = frame.copy()
        key = cv2.waitKey(1) & 0xFF

        # Global keys
        if key == ord("q"):
            break
        if key == ord("r"):
            # Reset to monitoring
            state = MONITORING
            reposition_queue = []
            reposition_index = 0
            confirm_count = 0
            logger.info("Reset to MONITORING")
            print("\n" + "=" * 60)
            print("MONITORING — press F to trigger a foul")
            print("=" * 60)
            continue

        # ---- MONITORING: show detected balls, wait for F ----
        if state == MONITORING:
            draw_ball_labels(display, current_balls)
            n = sum(len(v) for v in current_balls.values())
            draw_status_bar(display, f"MONITORING ({n} balls) | f=foul  q=quit")

            if key == ord("f"):
                # Snapshot current positions
                snapshot_positions = current_balls.copy()
                if not snapshot_positions:
                    logger.warning("No balls detected — cannot foul")
                else:
                    # Build reposition queue from ALL detected balls
                    reposition_queue = []
                    for color, positions in snapshot_positions.items():
                        for (x, y) in positions:
                            reposition_queue.append({
                                "color_name": color,
                                "x": x, "y": y,
                                "bgr": BALL_BGR.get(color, (255, 255, 255)),
                            })
                    foul_colors_to_check = set(snapshot_positions.keys()) - {"red"}
                    foul_start_time = time.time()
                    state = FOUL_DISPLAY
                    logger.info(f"FOUL triggered! {len(reposition_queue)} balls to reposition")
                    print(f"\nFOUL! Recorded {len(reposition_queue)} ball positions")
                    print("Now remove the balls from the table...")

        # ---- FOUL_DISPLAY: show FOUL overlay for minimum time ----
        elif state == FOUL_DISPLAY:
            display = draw_foul_overlay(display)
            elapsed = time.time() - foul_start_time
            remaining = max(0, args.foul_display_secs - elapsed)
            draw_status_bar(display,
                f"FOUL ({remaining:.1f}s) — remove balls from table | s=skip  r=reset")

            if key == ord("s") or elapsed >= args.foul_display_secs:
                state = FOUL_WAIT_OFF
                logger.info("Waiting for balls to be off table...")

        # ---- FOUL_WAIT_OFF: keep FOUL until potted balls gone from CV ----
        elif state == FOUL_WAIT_OFF:
            still_on = [c for c in foul_colors_to_check
                        if c in current_balls and current_balls[c]]
            display = draw_foul_overlay(display)
            if still_on:
                h = display.shape[0]
                text = "Still on table: " + ", ".join(c.upper() for c in still_on)
                cv2.putText(display, text, (10, h - 20),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            draw_status_bar(display,
                f"FOUL — waiting for {', '.join(c.upper() for c in foul_colors_to_check)} off table | s=skip  r=reset")

            if key == ord("s"):
                still_on = []

            if not still_on:
                logger.info("All potted balls off table — starting reposition")
                state = REPOSITIONING
                reposition_index = 0
                confirm_count = 0
                if reposition_queue:
                    t = reposition_queue[0]
                    print(f"\n[1/{len(reposition_queue)}] Place {t['color_name'].upper()} at ({t['x']}, {t['y']})")

        # ---- REPOSITIONING: show target, wait for ball placement ----
        elif state == REPOSITIONING:
            if reposition_index >= len(reposition_queue):
                # All done
                state = MONITORING
                logger.info("All balls repositioned!")
                print("\n" + "=" * 60)
                print("ALL BALLS REPOSITIONED! Back to monitoring.")
                print("Press F for next foul.")
                print("=" * 60)
                continue

            target = reposition_queue[reposition_index]
            color = target["color_name"]
            tx, ty = target["x"], target["y"]
            bgr = target["bgr"]
            total = len(reposition_queue)
            idx = reposition_index + 1

            # Draw target
            draw_target_circle(display, tx, ty, bgr,
                               label=f"{color.upper()} HERE")
            cv2.circle(display, (tx, ty), args.threshold, bgr, 1)
            draw_status_bar(display,
                f"[{idx}/{total}] Place {color.upper()} at ({tx},{ty}) | s=skip  r=reset")

            # Check placement
            detected = current_balls.get(color, [])
            near = False
            for (dx, dy) in detected:
                dist = math.hypot(dx - tx, dy - ty)
                if dist <= args.threshold:
                    near = True
                    cv2.putText(display, f"{dist:.0f}px",
                                 (dx + 15, dy - 10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    break

            if near:
                confirm_count += 1
                progress = min(confirm_count / args.confirm_frames, 1.0)
                angle = int(360 * progress)
                cv2.ellipse(display, (tx, ty), (35, 35), -90, 0, angle,
                             (0, 255, 0), 3)
                if confirm_count >= args.confirm_frames:
                    logger.info(f"{color.upper()} placed correctly!")
                    hold_start_time = time.time()
                    state = REPOSITION_HOLD
            else:
                confirm_count = 0

            if key == ord("s"):
                logger.info(f"Skipping {color.upper()}")
                reposition_index += 1
                confirm_count = 0
                if reposition_index < len(reposition_queue):
                    t = reposition_queue[reposition_index]
                    print(f"\n[{reposition_index+1}/{total}] Place {t['color_name'].upper()} at ({t['x']}, {t['y']})")

        # ---- REPOSITION_HOLD: green confirmation for hold_secs ----
        elif state == REPOSITION_HOLD:
            target = reposition_queue[reposition_index]
            color = target["color_name"]
            tx, ty = target["x"], target["y"]
            total = len(reposition_queue)
            idx = reposition_index + 1

            draw_confirmed(display, tx, ty, color)
            remaining = max(0, args.hold_secs - (time.time() - hold_start_time))
            draw_status_bar(display,
                f"[{idx}/{total}] {color.upper()} OK! Next in {remaining:.1f}s")

            if time.time() - hold_start_time >= args.hold_secs:
                reposition_index += 1
                confirm_count = 0
                state = REPOSITIONING
                if reposition_index < len(reposition_queue):
                    t = reposition_queue[reposition_index]
                    print(f"\n[{reposition_index+1}/{total}] Place {t['color_name'].upper()} at ({t['x']}, {t['y']})")

        cv2.imshow(WIN, display)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

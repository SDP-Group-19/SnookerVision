"""Track a red ball and light up LEDs following it via MQTT.

On startup, click the 4 corners of the table (top-left, top-right,
bottom-left, bottom-right). All ball coordinates are then warped
to a clean top-down view before being sent to the LEDs.

Usage:
  python final/src/snookervision/app/track_red.py ^
      --camera-source http://192.168.137.172:8000/stream.mjpg

Controls:
  q = quit
"""
import sys
import time
import argparse
import logging
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from pathlib import Path
from collections import deque

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from snookervision.detection import DetectionModel
from snookervision.core import config

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---- MQTT Setup ----
BROKER = "d35b0b7f3a0e43aeaa83cbe44fe9e73b.s1.eu.hivemq.cloud"
PORT = 8883
USERNAME = "SDPgroup19"
PASSWORD = "SDPgroup543"
TOPIC = "esp32/master"


def connect_mqtt():
    client = mqtt.Client(client_id="PythonController", protocol=mqtt.MQTTv311)
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()
    client.tls_insecure_set(True)
    client.connect(BROKER, PORT)
    client.loop_start()
    logger.info(f"MQTT connected to {BROKER}:{PORT}")
    return client


def send_resize(client, width, height):
    message = f"resize: {width}, {height}"
    client.publish(TOPIC, message)
    logger.info(f"Sent: {message}")


def send_ball(client, x, y):
    message = f"ball: {x},{y},255,0,0"
    client.publish(TOPIC, message)


def send_clear(client):
    client.publish(TOPIC, "clear")


# ---- Table corner selection ----
click_points = []


def _mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(click_points) < 4:
        click_points.append((x, y))
        logger.info(f"Point {len(click_points)}: ({x}, {y})")


def select_table_corners(cap):
    """Let user click 4 corners: top-left, top-right, bottom-left, bottom-right."""
    global click_points
    click_points = []

    win = "Select 4 Table Corners"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _mouse_callback)

    labels = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-RIGHT"]

    print("\n" + "=" * 50)
    print("Click the 4 corners of the table in this order:")
    print("  1. TOP-LEFT")
    print("  2. TOP-RIGHT")
    print("  3. BOTTOM-LEFT")
    print("  4. BOTTOM-RIGHT")
    print("=" * 50)

    while len(click_points) < 4:
        ret, frame = cap.read()
        if not ret:
            continue
        display = frame.copy()

        # Draw instruction
        idx = len(click_points)
        cv2.putText(display, f"Click {labels[idx]} ({idx+1}/4)",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Draw already-clicked points
        for i, pt in enumerate(click_points):
            cv2.circle(display, pt, 8, (0, 255, 0), -1)
            cv2.putText(display, labels[i], (pt[0] + 10, pt[1] - 10),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw lines between clicked points
        if len(click_points) >= 2:
            cv2.line(display, click_points[0], click_points[1], (0, 255, 0), 2)
        if len(click_points) >= 3:
            cv2.line(display, click_points[0], click_points[2], (0, 255, 0), 2)
        if len(click_points) >= 4:
            cv2.line(display, click_points[1], click_points[3], (0, 255, 0), 2)
            cv2.line(display, click_points[2], click_points[3], (0, 255, 0), 2)

        cv2.imshow(win, display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyWindow(win)
            return None

    # Show final selection for 1 second
    ret, frame = cap.read()
    if ret:
        display = frame.copy()
        pts = np.array(click_points, dtype=np.int32)
        order = [0, 1, 3, 2]  # TL, TR, BR, BL for polygon
        cv2.polylines(display, [pts[order]], True, (0, 255, 0), 2)
        for i, pt in enumerate(click_points):
            cv2.circle(display, pt, 8, (0, 255, 0), -1)
            cv2.putText(display, labels[i], (pt[0] + 10, pt[1] - 10),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(display, "Table selected! Starting...",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(win, display)
        cv2.waitKey(1000)

    cv2.destroyWindow(win)

    # Return as numpy array: [TL, TR, BL, BR]
    return np.float32(click_points)


# ---- Args ----
def parse_args():
    p = argparse.ArgumentParser(description="Track red ball with LED follow")
    p.add_argument("--camera-source", type=str, default=None)
    p.add_argument("--camera-port", type=int, default=1)
    p.add_argument("--camera-width", type=int, default=1280)
    p.add_argument("--camera-height", type=int, default=720)
    p.add_argument("--detector-imgsz", type=int, default=640)
    p.add_argument("--table-width", type=int, default=700,
                    help="LED table width to send in resize")
    p.add_argument("--table-height", type=int, default=400,
                    help="LED table height to send in resize")
    p.add_argument("--no-window", action="store_true", help="Hide OpenCV window")
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
    for _ in range(5):
        cap.read()
    return cap


def main():
    args = parse_args()
    config.detector_imgsz = args.detector_imgsz
    config.process_every_n_frames = 1

    table_w = args.table_width
    table_h = args.table_height

    # Open camera
    cap = open_camera(args)

    # Select table corners
    src_pts = select_table_corners(cap)
    if src_pts is None:
        logger.error("Table selection cancelled")
        cap.release()
        return

    # Destination rectangle: top-down view [TL, TR, BL, BR]
    dst_pts = np.float32([
        [0, 0],
        [table_w, 0],
        [0, table_h],
        [table_w, table_h],
    ])

    # Compute perspective transform
    homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
    logger.info(f"Perspective transform ready ({table_w}x{table_h})")

    # Load YOLO model
    logger.info("Loading YOLO model...")
    model = DetectionModel()
    if model.model is None:
        logger.error("Failed to load detection model")
        sys.exit(1)

    # Connect MQTT
    client = connect_mqtt()
    send_resize(client, table_w, table_h)

    if not args.no_window:
        cv2.namedWindow("Track Red", cv2.WINDOW_NORMAL)

    times = deque(maxlen=30)
    last_red_pos = None
    last_send_time = 0

    logger.info("Tracking red ball — press Q to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Camera read failed")
                break

            # Warp frame to top-down view for detection
            warped = cv2.warpPerspective(frame, homography, (table_w, table_h))

            # Detect on warped frame (clean top-down, no distortion)
            detections, _ = model.detect(warped)

            # Find red ball(s)
            red_positions = []
            if detections:
                for det in detections:
                    if det["label"] == "red":
                        cx, cy = det["center"]
                        red_positions.append((int(cx), int(cy)))

            # Send the first red ball position to LED (max every 0.3s)
            # Coordinates are already in table space (0,0)=top-left
            # Flip to LED space: (0,0)=bottom-right
            now = time.time()
            if red_positions and now - last_send_time >= 0.3:
                tx, ty = red_positions[0]
                tx = max(0, min(tx, table_w))
                ty = max(0, min(ty, table_h))
                led_x = table_w - tx
                led_y = table_h - ty
                send_ball(client, led_x, led_y)
                last_red_pos = (tx, ty)
                last_send_time = now

            # Draw on warped frame
            if not args.no_window:
                display = warped.copy()

                if detections:
                    for det in detections:
                        cx, cy = det["center"]
                        label = det["label"]
                        if label == "red":
                            cv2.circle(display, (int(cx), int(cy)), 15, (0, 0, 255), 3)
                            cv2.putText(display, f"RED ({int(cx)},{int(cy)})",
                                        (int(cx) + 20, int(cy)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        else:
                            cv2.circle(display, (int(cx), int(cy)), 8, (150, 150, 150), 1)

                now = time.time()
                times.append(now)
                fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0
                cv2.putText(display, f"FPS: {fps:.1f} | Reds: {len(red_positions)}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if last_red_pos:
                    led_x = table_w - last_red_pos[0]
                    led_y = table_h - last_red_pos[1]
                    cv2.putText(display, f"Table: ({last_red_pos[0]},{last_red_pos[1]}) -> LED: ({led_x},{led_y})",
                                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.imshow("Track Red", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        send_clear(client)
        client.loop_stop()
        client.disconnect()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Disconnected and cleaned up")


if __name__ == "__main__":
    main()

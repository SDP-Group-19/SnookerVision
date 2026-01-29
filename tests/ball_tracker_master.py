#!/usr/bin/env python3
"""
Master ball tracking test: works with a live camera or a prerecorded video.

Examples:
  python3 tests/ball_tracker_master.py --camera 0
  python3 tests/ball_tracker_master.py --video path/to/video.mp4
"""

import argparse
import time

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Master ball tracking test")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int, help="Camera index, e.g., 0")
    src.add_argument("--video", type=str, help="Path to prerecorded video")

    parser.add_argument("--width", type=int, default=1280, help="Capture width (camera only)")
    parser.add_argument("--height", type=int, default=720, help="Capture height (camera only)")
    parser.add_argument("--min-area", type=int, default=150, help="Minimum contour area for detection")
    parser.add_argument("--circularity", type=float, default=0.85, help="Min circularity (0-1)")
    parser.add_argument("--show-mask", action="store_true", help="Show threshold mask windows")
    return parser.parse_args()


def build_masks(hsv: np.ndarray):
    # Red wraps around HSV, so we use two ranges.
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # White: low saturation, high value.
    lower_white = np.array([0, 0, 0])
    upper_white = np.array([180, 120, 255])

    # Black: low value.
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

    return red_mask, white_mask, black_mask


def find_ball_contours(mask: np.ndarray, min_area: int, min_circularity: float):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    balls = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < min_circularity:
            continue
        balls.append(cnt)
    return balls


def draw_ball(frame, cnt, color, center_color=None):
    ((x, y), radius) = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    cv2.circle(frame, center, int(radius), color, 2)
    cv2.circle(frame, center, 3, center_color or color, -1)


def main() -> None:
    args = parse_args()

    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera index {args.camera}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    else:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video {args.video}")

    cv2.namedWindow("Ball Tracker", cv2.WINDOW_NORMAL)

    if args.show_mask:
        cv2.namedWindow("Red Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("White Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Black Mask", cv2.WINDOW_NORMAL)

    last_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream or frame capture failed.")
            break

        blur = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        red_mask, white_mask, black_mask = build_masks(hsv)

        red_balls = find_ball_contours(red_mask, args.min_area, args.circularity)
        white_balls = find_ball_contours(white_mask, args.min_area, args.circularity)
        black_balls = find_ball_contours(black_mask, args.min_area, args.circularity)

        # Draw all red balls
        for cnt in red_balls:
            draw_ball(frame, cnt, (0, 0, 255))

        # Draw one white ball (largest)
        if white_balls:
            draw_ball(frame, max(white_balls, key=cv2.contourArea), (255, 255, 255))

        # Draw one black ball (largest)
        if black_balls:
            draw_ball(frame, max(black_balls, key=cv2.contourArea), (0, 0, 0), center_color=(255, 255, 255))

        now = time.time()
        dt = now - last_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Ball Tracker", frame)
        if args.show_mask:
            cv2.imshow("Red Mask", red_mask)
            cv2.imshow("White Mask", white_mask)
            cv2.imshow("Black Mask", black_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

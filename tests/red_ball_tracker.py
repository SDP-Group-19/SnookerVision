#!/usr/bin/env python3
"""
Quick test to detect/track a small red ball from a USB camera.

Usage:
  python3 tests/red_ball_tracker.py --camera 0
"""

import argparse
import time

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Red ball detection test")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument("--min-area", type=int, default=150, help="Minimum contour area for detection")
    parser.add_argument("--circularity", type=float, default=0.65, help="Min circularity (0-1) for ball-like contours")
    parser.add_argument("--show-mask", action="store_true", help="Show threshold mask window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Red wraps around HSV, so we use two ranges.
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # White: low saturation, high value.
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])

    # Black: low value.
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])

    last_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed; retrying...")
            continue

        blur = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        black_mask = cv2.inRange(hsv, lower_black, upper_black)

        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

        def find_ball_contours(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            balls = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < args.min_area:
                    continue
                perimeter = cv2.arcLength(cnt, True)
                if perimeter <= 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity < args.circularity:
                    continue
                balls.append(cnt)
            return balls

        red_balls = find_ball_contours(red_mask)
        white_balls = find_ball_contours(white_mask)
        black_balls = find_ball_contours(black_mask)

        # Draw all red balls
        for cnt in red_balls:
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            cv2.circle(frame, center, int(radius), (0, 0, 255), 2)
            cv2.circle(frame, center, 3, (0, 0, 255), -1)

        # Draw one white ball (largest)
        if white_balls:
            largest_white = max(white_balls, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(largest_white)
            center = (int(x), int(y))
            cv2.circle(frame, center, int(radius), (255, 255, 255), 2)
            cv2.circle(frame, center, 3, (255, 255, 255), -1)

        # Draw one black ball (largest)
        if black_balls:
            largest_black = max(black_balls, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(largest_black)
            center = (int(x), int(y))
            cv2.circle(frame, center, int(radius), (0, 0, 0), 2)
            cv2.circle(frame, center, 3, (255, 255, 255), -1)

        # FPS display
        now = time.time()
        dt = now - last_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Red Ball Tracker", frame)
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

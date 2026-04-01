"""Capture frames from the Pi camera stream for training data.

Usage:
  python final/scripts/collect_frames.py --source http://192.168.137.225:8000/stream.mjpg

  - Frames auto-save every 2 seconds
  - Press SPACE to save immediately
  - Press Q to quit
"""
import argparse
import os
import time
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Camera URL or device index")
    parser.add_argument("--out", default="final/scripts/dataset/raw_frames", help="Output directory")
    parser.add_argument("--interval", type=float, default=2.0, help="Auto-save interval in seconds")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=200, help="Stop after N frames")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Continue numbering from existing files
    existing = [f for f in os.listdir(args.out) if f.startswith("frame_") and f.endswith(".jpg")]
    start = len(existing)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Cannot open {args.source}")
        return

    count = start
    last_save = 0
    print(f"Saving frames to {args.out}/")
    print("SPACE=save now  Q=quit")
    print("Move balls around between saves for variety!\n")

    target = start + args.max_frames
    print(f"Starting from frame {start} (target: {args.max_frames} new frames)")
    while count < target:
        ret, frame = cap.read()
        if not ret:
            print("Lost camera feed, retrying...")
            time.sleep(1)
            continue

        now = time.time()
        save = False

        cv2.putText(frame, f"Saved: {count - start}/{args.max_frames} (total: {count})", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Collect Frames", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            save = True

        if save:
            path = os.path.join(args.out, f"frame_{count:04d}.jpg")
            cv2.imwrite(path, frame)
            count += 1
            last_save = now
            print(f"  [{count}/{args.max_frames}] saved {path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone — {count} frames saved to {args.out}/")


if __name__ == "__main__":
    main()

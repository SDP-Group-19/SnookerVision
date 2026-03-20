import argparse
import logging
import time
from threading import Condition, Thread

import cv2
from flask import Flask, Response

try:
    from picamera2 import Picamera2
except ImportError as exc:
    raise SystemExit(
        "picamera2 is required on the Raspberry Pi. Install it with: sudo apt install -y python3-picamera2"
    ) from exc


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)


class FrameBuffer:
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, frame_bytes):
        with self.condition:
            self.frame = frame_bytes
            self.condition.notify_all()

    def get(self):
        with self.condition:
            while self.frame is None:
                self.condition.wait()
            return self.frame


frame_buffer = FrameBuffer()


def capture_loop(camera, quality):
    while True:
        frame = camera.capture_array()
        ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if ok:
            frame_buffer.write(encoded.tobytes())
        time.sleep(0.001)


@app.route("/")
def index():
    return {
        "status": "ok",
        "stream": "/stream.mjpg",
    }


@app.route("/stream.mjpg")
def stream():
    def generate():
        while True:
            frame = frame_buffer.get()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--quality", type=int, default=85)
    args = parser.parse_args()

    camera = Picamera2()
    config = camera.create_video_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}
    )
    camera.configure(config)
    camera.start()
    logger.info(
        "Streaming camera on http://%s:%s/stream.mjpg at %sx%s",
        args.host,
        args.port,
        args.width,
        args.height,
    )

    worker = Thread(target=capture_loop, args=(camera, args.quality), daemon=True)
    worker.start()
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()

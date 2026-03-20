# Raspberry Pi Remote Camera

This project can now read frames from a Raspberry Pi camera stream instead of a locally attached USB camera.

## 1. Run capture on the Raspberry Pi

Install the required packages:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-flask
```

Start the MJPEG server:

```bash
python3 final/scripts/raspi_camera_mjpeg_server.py --host 0.0.0.0 --port 8000 --width 1280 --height 720
```

If your Raspberry Pi hostname is `raspberrypi.local`, the stream URL will be:

```text
http://raspberrypi.local:8000/stream.mjpg
```

## 2. Run detection on the computer

Start SnookerVision with the remote stream:

```bash
python3 final/src/snookervision/app/main.py --camera-source http://raspberrypi.local:8000/stream.mjpg
```

If calibration or table-point selection becomes inconvenient while testing, you can temporarily disable them:

```bash
python3 final/src/snookervision/app/main.py \
  --camera-source http://raspberrypi.local:8000/stream.mjpg \
  --no-calibration
```

For higher FPS, start with these settings on the computer side:

```bash
python3 final/src/snookervision/app/main.py \
  --camera-source http://raspberrypi.local:8000/stream.mjpg \
  --camera-width 1280 \
  --camera-height 720 \
  --process-every-n-frames 1 \
  --detector-imgsz 640 \
  --detector-device auto \
  --no-calibration
```

Final
```bash
python .\final\src\snookervision\app\main.py --camera-source http://192.168.137.225:8000/stream.mjpg --camera-width 1280 --camera-width 720 --detector-imgsz 512 --process-every-n-frames 1  --no-draw-results --no-interface --fast-mode --no-calibration --no-table-pts
```

If throughput is still too low, try these in order:

- Lower Raspberry Pi stream resolution to `960x540` or `640x480`
- Lower `--detector-imgsz` from `640` to `512`
- Run with `--hide-windows --no-draw-results` if you do not need live overlays
- Increase `--process-every-n-frames` to `2` or `3` to trade temporal precision for FPS
- If you are on Apple Silicon or NVIDIA, the app now prefers `MPS/CUDA` automatically when available

## Notes

- The computer still performs all detection and game-state logic.
- The Raspberry Pi only captures and serves frames.
- If latency is high, lower the Raspberry Pi stream resolution first.

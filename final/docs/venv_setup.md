# Virtual Environment Setup

This repository has two Python environments:

- Desktop environment: runs SnookerVision inference on your computer
- Raspberry Pi environment: runs camera capture/streaming on the Pi

## External Libraries Used

The codebase imports these non-stdlib libraries:

- `opencv-contrib-python`
- `numpy`
- `ultralytics`
- `torch`
- `python-socketio`
- `liveconfig`
- `paho-mqtt`
- `flask` for the Raspberry Pi MJPEG server
- `picamera2` for the Raspberry Pi camera
- `tensorflow` / `keras` only if you later enable obstruction detection

## Desktop Setup

```bash
cd /Users/liyuchen/Desktop/SnookerVision
chmod +x final/scripts/setup_desktop_venv.sh
./final/scripts/setup_desktop_venv.sh
```

Activate the environment later with:

```bash
source /Users/liyuchen/Desktop/SnookerVision/.venv/bin/activate
```

Check installed libraries and GPU visibility:

```bash
python final/scripts/check_env.py
```

Run the main app:

```bash
python final/src/snookervision/app/main.py --no-interface
```

## Raspberry Pi Setup

On the Raspberry Pi:

```bash
cd /path/to/SnookerVision
sudo apt update
sudo apt install -y python3-venv python3-picamera2 python3-libcamera
chmod +x final/scripts/setup_raspi_venv.sh
./final/scripts/setup_raspi_venv.sh
```

Activate the environment later with:

```bash
source /path/to/SnookerVision/.venv-raspi/bin/activate
```

Start the MJPEG stream:

```bash
python final/scripts/raspi_camera_mjpeg_server.py --host 0.0.0.0 --port 8000 --width 1280 --height 720
```

## Recommended Workflow

1. Create the desktop environment on your computer
2. Run `python final/scripts/check_env.py`
3. Create the Raspberry Pi environment on the Pi
4. Start the Pi camera stream
5. Run SnookerVision on the computer and connect to the stream URL

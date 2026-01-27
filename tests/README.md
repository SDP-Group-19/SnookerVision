# Tests

This folder contains quick validation scripts for camera ingest and ball tracking.

## Master ball tracking test
Script: `tests/ball_tracker_master.py`

### Requirements
- Python 3
- OpenCV (`opencv-python`) and NumPy installed
- A USB camera or a prerecorded video file

### Run with a live camera
```bash
python3 tests/ball_tracker_master.py --camera 0
```

### Run with a prerecorded video
```bash
python3 tests/ball_tracker_master.py --video path/to/video.mp4
```

### Helpful flags
- `--show-mask` to view the red/white/black threshold masks
- `--min-area 50` if the balls are small or far away
- `--circularity 0.6` to loosen the filled-circle filter

### Quit
Press `q` or `Esc` to exit.

### Notes
- Camera indices can change between boots; if you don’t see the USB feed, try `--camera 1` or `--camera 2`.
- Lighting matters. If detection is weak, increase light or adjust HSV ranges in the script.

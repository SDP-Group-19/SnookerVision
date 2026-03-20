# Arduino

This folder is split into two parts:

- `sketches/` contains the Arduino projects you upload to a board.
- `host/` contains small Python tools for talking to a board over serial.

## Layout

- `sketches/display_controller/display_controller.ino`
- `sketches/pocket_sensors/pocket_sensors.ino`
- `host/arduino_monitor.py`
- `requirements.txt`

## Sketches

### `display_controller`

Controls two TM1637 score displays and local score buttons.

Serial protocol:

- `SET <display> <score>`
- `ADD <display> <delta>`
- `RESET`
- `STATUS`
- legacy `1=1234` / `2=5678`

Example responses:

- `READY DISPLAY`
- `SCORE 1 0`
- `SET 1 35`
- `ADD 2 -1`

### `pocket_sensors`

Reads six pocket sensors using `INPUT_PULLUP` and emits debounced events.

Example responses:

- `READY POCKETS`
- `POCKET_STATE 1 IDLE`
- `POCKET 3`
- `POCKET_CLEAR 3`

## How To Run

### 1. Upload a sketch

Open one of these folders in the Arduino IDE and upload it to the correct board:

- `final/src/snookervision/arduino/sketches/display_controller`
- `final/src/snookervision/arduino/sketches/pocket_sensors`

The sketch filename matches the folder name because the Arduino IDE expects that layout.

### 2. Install Python dependency

From the project root:

```bash
python3 -m pip install -r final/src/snookervision/arduino/requirements.txt
```

### 3. Find your serial port

On macOS:

```bash
ls /dev/cu.*
```

Typical examples:

- `/dev/cu.usbmodem1101`
- `/dev/cu.usbserial-0001`

### 4. Monitor a board

```bash
python3 final/src/snookervision/arduino/host/arduino_monitor.py \
  --port /dev/cu.usbmodem1101 \
  --mode monitor
```

### 5. Send a display command

```bash
python3 final/src/snookervision/arduino/host/arduino_monitor.py \
  --port /dev/cu.usbmodem1101 \
  --mode command \
  --command "STATUS"
```

More examples:

```bash
python3 final/src/snookervision/arduino/host/arduino_monitor.py --port /dev/cu.usbmodem1101 --mode command --command "SET 1 24"
python3 final/src/snookervision/arduino/host/arduino_monitor.py --port /dev/cu.usbmodem1101 --mode command --command "ADD 2 1"
python3 final/src/snookervision/arduino/host/arduino_monitor.py --port /dev/cu.usbmodem1101 --mode command --command "RESET"
```

## Notes

- `display_controller` and `pocket_sensors` currently reuse some of the same pin numbers, so they are intended for separate Arduino boards unless you merge and remap the pins.
- The `nmt_env/` folder is left in place but ignored by git. It is not required for using the Arduino sketches.

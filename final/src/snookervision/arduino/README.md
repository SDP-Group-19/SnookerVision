# Arduino

This folder is split into two parts:

- `sketches/` contains the Arduino projects you upload to a board.
- `host/` contains small Python tools for talking to a board over serial.

## Layout

- `sketches/display_controller/display_controller.ino`
- `sketches/esp_display_node_left/esp_display_node_left.ino`
- `sketches/esp_display_node_right/esp_display_node_right.ino`
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

### `esp_display_node_left`

ESP8266 Wi-Fi scoreboard node for:

- left score display
- right score display
- left score up/down buttons
- last position button
- LCD screen

Exposes:

- `GET /status`
- `POST /command`
- `GET /command?cmd=...`

### `esp_display_node_right`

ESP8266 Wi-Fi control node for:

- right score up/down buttons
- player lights
- change player button
- full reset button

Exposes:

- `GET /status`
- `POST /command`
- `GET /command?cmd=...`

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
- `final/src/snookervision/arduino/sketches/esp_display_node_left`
- `final/src/snookervision/arduino/sketches/esp_display_node_right`
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

## ESP8266 Wi-Fi Display Setup

1. Open `esp_display_node_left.ino` and `esp_display_node_right.ino`.
2. Set `kWifiSsid` and `kWifiPassword` in both files.
3. Adjust pin numbers if your wiring differs.
4. Upload the left sketch to the ESP that owns display 1.
5. Upload the right sketch to the ESP that owns display 2.

Suggested split:

- `esp_display_node_left`
  - left TM1637 display on D2/D3
  - right TM1637 display on D4/D5
  - left score up/down buttons on D6/D7
  - last position button on D8
  - LCD on I2C pins
- `esp_display_node_right`
  - right score up/down buttons on D2/D3
  - chainable turn lights on D4/D5
  - change player button on D6
  - full reset button on D7

Default pin map in the sketches:

- `esp_display_node_left`
  - left display: `CLK=D2/GPIO13`, `DIO=D3/GPIO10`
  - right display: `CLK=D4/GPIO15`, `DIO=D5/GPIO2`
  - left up button: `D6/GPIO5`
  - left down button: `D7/GPIO4`
  - last position button: `D8/GPIO0`
  - LCD I2C: dedicated `SDA` / `SCL` pins from the board variant, not the `D` pin map
- `esp_display_node_right`
  - right up button: `D2/GPIO13`
  - right down button: `D3/GPIO10`
  - chainable LEDs: `DATA=D4/GPIO15`, `CLK=D5/GPIO2`
  - change player button: `D6/GPIO5`
  - full reset button: `D7/GPIO4`

Libraries you need in Arduino IDE:

- `TM1637Display`
- `rgb_lcd`
- `ChainableLED`
- ESP8266 board support package

After boot, the nodes will appear at:

- `http://snooker-display-1.local`
- `http://snooker-display-2.local`

Quick checks:

```bash
curl http://snooker-display-1.local/status
curl "http://snooker-display-2.local/command?cmd=SET%202%2034"
curl "http://snooker-display-1.local/command?cmd=LCD%20Player%201%7CAt%20table"
curl "http://snooker-display-1.local/command?cmd=LIGHT%201%2040%2040%2040"
curl "http://snooker-display-2.local/command?cmd=LIGHT%202%2040%2040%2040"
```

To run the SnookerVision app against both nodes:

```bash
python3 final/src/snookervision/app/main.py \
  --arduino-port http://snooker-display-1.local,http://snooker-display-2.local
```

The Python bridge will fan commands out to both endpoints. Each ESP ignores commands meant for the other display.

## Upload And Test Flow

1. In Arduino IDE, install the ESP8266 board package.
2. Connect the left ESP by USB.
3. Open `sketches/esp_display_node_left/esp_display_node_left.ino`.
4. Select the correct ESP8266 board and port.
5. Upload.
6. Open Serial Monitor at `115200` and wait for the IP / status line.
7. Repeat for the right ESP with `esp_display_node_right.ino`.
8. Once both are on Wi-Fi, test each one with `curl`.
9. Then run the Python app with both `.local` endpoints in `--arduino-port`.

Expected behavior:

- `SET 1 ...` updates the left display on `esp_display_node_left`.
- `SET 2 ...` updates the right display on `esp_display_node_left`.
- `LCD ...` only updates the LCD on `esp_display_node_left`.
- `LIGHT 1 ...` and `LIGHT 2 ...` update the chainable LEDs on `esp_display_node_right`.
- Right-side physical buttons on `esp_display_node_right` forward score changes to `esp_display_node_left` over Wi-Fi.
- Left-side physical buttons on `esp_display_node_left` update the left display locally.
- `LIGHTOFF` turns both player lights off.

## Important Notes

- `D3/GPIO0` and `D8/GPIO15` are ESP8266 boot-strapping pins. If those lines are held in the wrong state at boot, the board may fail to start.
- The `last position` button on `D8` is configured as active-high because `GPIO15` normally needs to stay low during boot.
- The LCD now uses the board's `SDA` and `SCL` definitions, so it does not clash with your custom `D`-pin GPIO mapping.

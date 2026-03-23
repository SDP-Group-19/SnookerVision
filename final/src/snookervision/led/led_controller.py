import socket
import logging
import threading
import time

logger = logging.getLogger(__name__)

# TCP protocol matching the ESP32 Arduino firmware:
#   "ball: x,y,r,g,b\n"   - light LEDs at table position (x,y) with colour
#   "resize: w,h\n"        - set table dimensions for coordinate scaling
#   "clear\n"              - turn off all LEDs


class LEDController:
    def __init__(self, arduino_ip, arduino_port):
        self.arduino_ip = arduino_ip
        self.arduino_port = arduino_port
        self._sock = None
        self._lock = threading.Lock()
        self._pulse_thread = None
        self._pulse_stop = threading.Event()

    def connect(self):
        with self._lock:
            if self._sock is not None:
                return True
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3.0)
                sock.connect((self.arduino_ip, self.arduino_port))
                sock.settimeout(None)
                self._sock = sock
                logger.info(f"LED connected to {self.arduino_ip}:{self.arduino_port}")
                return True
            except OSError as e:
                logger.error(f"LED connection failed: {e}")
                self._sock = None
                return False

    def _reconnect(self):
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None
        return self.connect()

    def _send(self, message):
        with self._lock:
            if self._sock is None:
                return False
            try:
                self._sock.send(message.encode("ascii"))
                return True
            except OSError as e:
                logger.warning(f"LED send failed: {e}")
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None
                return False

    def send_ball(self, x, y, r, g, b):
        msg = f"ball: {x},{y},{r},{g},{b}\n"
        if not self._send(msg):
            if self._reconnect():
                self._send(msg)

    def send_resize(self, width, height):
        msg = f"resize: {width},{height}\n"
        if not self._send(msg):
            if self._reconnect():
                self._send(msg)

    def send_clear(self):
        if not self._send("clear\n"):
            if self._reconnect():
                self._send("clear\n")

    def start_pulse(self, x, y, color=(255, 0, 0), on_time=0.6, off_time=0.4):
        """Pulse LEDs at position (x,y) on/off in a background thread."""
        self.stop_pulse()
        self._pulse_stop.clear()
        r, g, b = color
        self._pulse_thread = threading.Thread(
            target=self._pulse_loop,
            args=(x, y, r, g, b, on_time, off_time),
            daemon=True,
        )
        self._pulse_thread.start()

    def _pulse_loop(self, x, y, r, g, b, on_time, off_time):
        while not self._pulse_stop.is_set():
            self.send_ball(x, y, r, g, b)
            if self._pulse_stop.wait(on_time):
                break
            self.send_clear()
            if self._pulse_stop.wait(off_time):
                break

    def stop_pulse(self):
        self._pulse_stop.set()
        if self._pulse_thread is not None and self._pulse_thread.is_alive():
            self._pulse_thread.join(timeout=2.0)
        self._pulse_thread = None

    def close(self):
        self.stop_pulse()
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None

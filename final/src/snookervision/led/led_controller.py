import socket
import logging
import threading

logger = logging.getLogger(__name__)

# TCP protocol matching the ESP32 Arduino firmware:
#   "ball: x,y,r,g,b\n"   - light LEDs nearest to table position (x,y)
#   "resize: w,h\n"        - set table dimensions for coordinate scaling
#   "clear\n"              - turn off all LEDs


class LEDController:
    def __init__(self, arduino_ip, arduino_port):
        self.arduino_ip = arduino_ip
        self.arduino_port = arduino_port
        self._sock = None
        self._lock = threading.Lock()

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

    def close(self):
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None

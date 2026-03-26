import socket
import logging
import threading

logger = logging.getLogger(__name__)

# TCP protocol for ESP32 LED master:
#   Connect → send "CTRL\n" to identify as controller
#   "ball: x,y,r,g,b\n"   - light LEDs nearest to table position (x,y)
#   "resize: w,h\n"        - set table dimensions for coordinate scaling
#   "clear\n"              - turn off all LEDs


class LEDController:
    def __init__(self, arduino_ip, arduino_port):
        self.arduino_ip = arduino_ip
        self.arduino_port = arduino_port
        self._sock = None
        self._lock = threading.Lock()
        self.table_width = 1200
        self.table_height = 600

    def connect(self):
        with self._lock:
            if self._sock is not None:
                return True
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect((self.arduino_ip, self.arduino_port))
                sock.sendall(b"CTRL\n")
                sock.settimeout(None)
                self._sock = sock
                logger.info(f"LED connected to {self.arduino_ip}:{self.arduino_port} (CTRL)")
                return True
            except OSError as e:
                logger.error(f"LED connection failed: {e}")
                self._sock = None
                return False

    def _send(self, message):
        with self._lock:
            if self._sock is None:
                return False
            try:
                self._sock.sendall(message.encode("ascii"))
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
        self._send(f"ball: {x},{y},{r},{g},{b}\n")

    def send_resize(self, width, height):
        self.table_width = width
        self.table_height = height
        self._send(f"resize: {width},{height}\n")

    def send_foul(self):
        """Light up red LEDs along both long sides of the table to indicate FOUL."""
        w = self.table_width
        h = self.table_height
        step = 50
        for x in range(0, w + 1, step):
            self._send(f"ball: {x},0,255,0,0\n")
            self._send(f"ball: {x},{h},255,0,0\n")

    def send_clear(self):
        self._send("clear\n")

    def close(self):
        with self._lock:
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None

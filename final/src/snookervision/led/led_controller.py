import ssl
import math
import logging
import threading
import numpy as np

logger = logging.getLogger(__name__)

# MQTT protocol for ESP32 LED master (topic: "esp32/master"):
#   "ball: x,y,r,g,b"   - light LEDs nearest to table position (x,y)
#   "resize: w,h"        - set LED coordinate range (w = x-axis, h = y-axis)
#   "clear"              - turn off all LEDs
#
# CV coordinate system:  (0,0) = top-left,     x+ right, y+ down
# LED coordinate system: (0,0) = bottom-right, x+ left,  y+ up
#
# Table dimensions for LED are computed from the selected table_pts
# (physical pixel distances in the original camera frame).
# Ball coords (in output_dimensions space) are scaled to those
# dimensions before flipping.


def compute_table_dimensions(table_pts):
    """Compute physical table width and height from 4 corner points.

    table_pts: [[TL], [TR], [BL], [BR]] in original camera pixel coords.
    Returns (width, height) as average edge lengths.
    """
    pts = np.array(table_pts, dtype=np.float64)
    tl, tr, bl, br = pts[0], pts[1], pts[2], pts[3]
    top_w = math.hypot(tr[0] - tl[0], tr[1] - tl[1])
    bot_w = math.hypot(br[0] - bl[0], br[1] - bl[1])
    left_h = math.hypot(bl[0] - tl[0], bl[1] - tl[1])
    right_h = math.hypot(br[0] - tr[0], br[1] - tr[1])
    return int((top_w + bot_w) / 2), int((left_h + right_h) / 2)


class LEDController:
    def __init__(self, broker, port, username, password, topic="esp32/master"):
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.topic = topic
        self._client = None
        self._lock = threading.Lock()
        # LED table dimensions (set by send_resize, computed from table_pts)
        self.table_width = 800
        self.table_height = 500
        # CV source dimensions (output_dimensions, for scaling)
        self.cv_width = 1200
        self.cv_height = 600

    def connect(self):
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            logger.warning("paho-mqtt not installed — LED controller disabled")
            return False

        try:
            client = mqtt.Client(client_id="LEDController", protocol=mqtt.MQTTv311)
            client.username_pw_set(self.username, self.password)
            client.tls_set(cert_reqs=ssl.CERT_NONE)
            client.tls_insecure_set(True)
            client.connect(self.broker, self.port)
            client.loop_start()
            self._client = client
            logger.info(f"LED controller connected via MQTT — topic: {self.topic}")
            return True
        except Exception as e:
            logger.error(f"LED MQTT connection failed: {e}")
            self._client = None
            return False

    def _publish(self, message):
        with self._lock:
            if self._client is None:
                return False
            try:
                self._client.publish(self.topic, message)
                return True
            except Exception as e:
                logger.warning(f"LED publish failed: {e}")
                return False

    def _cv_to_led(self, cv_x, cv_y):
        """Convert CV coordinates to LED coordinates.

        1. Scale from CV space (output_dimensions) to LED space (table_pts dimensions)
        2. Flip: LED (0,0) = bottom-right, x+ left, y+ up
        """
        scaled_x = cv_x * self.table_width / self.cv_width
        scaled_y = cv_y * self.table_height / self.cv_height
        led_x = int(self.table_width - scaled_x)
        led_y = int(self.table_height - scaled_y)
        return led_x, led_y

    def send_ball(self, x, y, r, g, b):
        led_x, led_y = self._cv_to_led(x, y)
        self._publish(f"ball: {led_x},{led_y},{r},{g},{b}")

    def send_resize(self, table_width, table_height, cv_width, cv_height):
        """Set LED table dimensions and CV source dimensions.

        table_width/height: computed from table_pts (physical proportions).
        cv_width/height: output_dimensions (ball coordinate range).
        """
        self.table_width = table_width
        self.table_height = table_height
        self.cv_width = cv_width
        self.cv_height = cv_height
        self._publish(f"resize: {table_width},{table_height}")
        logger.info(
            f"LED resize: {table_width}x{table_height} "
            f"(CV source: {cv_width}x{cv_height})"
        )

    def send_foul(self):
        """Light up red LEDs along both long sides of the table to indicate FOUL."""
        w = self.table_width
        h = self.table_height
        step = 50
        for led_x in range(0, w + 1, step):
            self._publish(f"ball: {led_x},0,255,0,0")
            self._publish(f"ball: {led_x},{h},255,0,0")

    def send_clear(self):
        self._publish("clear")

    def close(self):
        with self._lock:
            if self._client is not None:
                try:
                    self._client.loop_stop()
                    self._client.disconnect()
                except Exception:
                    pass
                self._client = None

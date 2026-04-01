import ssl
import logging
import threading

logger = logging.getLogger(__name__)

# MQTT protocol for ESP32 LED master (topic: "esp32/master"):
#   "ball: x,y,r,g,b"   - light LEDs nearest to table position (x,y)
#   "resize: w,h"        - set LED coordinate range (w = x-axis, h = y-axis)
#   "clear"              - turn off all LEDs
#
# LED coordinate system: (0,0) = bottom-right of table
#   led_x increases leftward  (= table_width - CV x)
#   led_y increases upward    (= table_height - CV y)


class LEDController:
    def __init__(self, broker, port, username, password, topic="esp32/master"):
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.topic = topic
        self._client = None
        self._lock = threading.Lock()
        self.table_width = 1200
        self.table_height = 600

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

        LED: (0,0) = bottom-right, x left, y up.
        CV:  (0,0) = top-left,     x right, y down.
        """
        led_x = self.table_width - cv_x
        led_y = self.table_height - cv_y
        return led_x, led_y

    def send_ball(self, x, y, r, g, b):
        led_x, led_y = self._cv_to_led(x, y)
        self._publish(f"ball: {led_x},{led_y},{r},{g},{b}")

    def send_resize(self, width, height):
        self.table_width = width
        self.table_height = height
        self._publish(f"resize: {width},{height}")

    def send_foul(self):
        """Light up red LEDs along both long sides of the table to indicate FOUL."""
        w = self.table_width
        h = self.table_height
        step = 50
        # Long sides of the table (top edge y=0, bottom edge y=h in CV)
        for cv_x in range(0, w + 1, step):
            lx0, ly0 = self._cv_to_led(cv_x, 0)
            lx1, ly1 = self._cv_to_led(cv_x, h)
            self._publish(f"ball: {lx0},{ly0},255,0,0")
            self._publish(f"ball: {lx1},{ly1},255,0,0")

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

"""MQTT pocket trigger sensor -- receives hardware pot confirmations from ESP32.

Each pocket has an IR sensor that fires when a ball passes through.
Messages arrive on the "pockets/esp" topic as JSON:
    {"pocket": 1, "state": "active"}   -- ball entered pocket 1
    {"pocket": 1, "state": "clear"}    -- pocket 1 cleared

The pocket numbers (1-6) are mapped to internal pocket indices (0-5).
Default mapping (configurable):
    1 -> 0 (top_left)
    2 -> 1 (top_middle)
    3 -> 2 (top_right)
    4 -> 3 (bottom_left)
    5 -> 4 (bottom_middle)
    6 -> 5 (bottom_right)
"""
import ssl
import json
import time
import logging
import threading
from collections import deque

logger = logging.getLogger(__name__)


class PocketSensor:
    """Receives hardware pocket trigger events via MQTT."""

    def __init__(self, config, pocket_map=None):
        """
        Args:
            config: Config object with mqtt_broker, mqtt_port, etc.
            pocket_map: dict mapping ESP pocket number (int) to internal
                        pocket index (0-5). Defaults to {1:0, 2:1, ..., 6:5}.
        """
        self.config = config
        self.pocket_map = pocket_map or {i: i - 1 for i in range(1, 7)}
        self._client = None
        self._lock = threading.Lock()
        self._events = deque(maxlen=50)
        self._topic = getattr(config, "pocket_sensor_topic", "pockets/esp")

    def connect(self):
        """Connect to the MQTT broker and start listening."""
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            logger.warning("paho-mqtt not installed -- pocket sensor disabled")
            return False

        try:
            client = mqtt.Client()
            client.username_pw_set(self.config.mqtt_username, self.config.mqtt_password)
            client.tls_set(cert_reqs=ssl.CERT_NONE)
            client.tls_insecure_set(True)
            client.on_connect = self._on_connect
            client.on_message = self._on_message
            client.connect(self.config.mqtt_broker, self.config.mqtt_port)
            client.loop_start()
            self._client = client
            logger.info(f"Pocket sensor connected -- topic: {self._topic}")
            return True
        except Exception as e:
            logger.error(f"Pocket sensor MQTT connect failed: {e}")
            return False

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(self._topic)
            logger.info(f"Pocket sensor subscribed to {self._topic}")
        else:
            logger.error(f"Pocket sensor MQTT connect rc={rc}")

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            pocket_num = data.get("pocket")
            state = data.get("state", "").lower()

            if pocket_num is None or state not in ("active", "clear"):
                return

            pocket_idx = self.pocket_map.get(pocket_num)
            if pocket_idx is None:
                logger.warning(f"Unknown pocket number: {pocket_num}")
                return

            event = {
                "pocket_idx": pocket_idx,
                "state": state,
                "time": time.time(),
            }

            with self._lock:
                self._events.append(event)

            if state == "active":
                logger.info(f"[POCKET] Pocket {pocket_num} (idx {pocket_idx}) TRIGGERED")

        except Exception as e:
            logger.warning(f"Pocket sensor bad message: {e}")

    def drain_events(self):
        """Return and clear all pending pocket trigger events.

        Returns list of dicts: [{"pocket_idx": int, "state": str, "time": float}]
        """
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events

    def close(self):
        if self._client:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
            self._client = None

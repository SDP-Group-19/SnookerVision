from __future__ import annotations

from dataclasses import dataclass
import copy
import logging
import math
from typing import Dict, Iterable, List, Optional

try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover
    mqtt = None


logger = logging.getLogger(__name__)

BALL_LED_COLORS = {
    "white": (255, 255, 255),
    "yellow": (255, 255, 0),
    "green": (0, 255, 0),
    "brown": (150, 75, 0),
    "blue": (0, 0, 255),
    "pink": (255, 105, 180),
    "black": (80, 80, 80),
    "red": (255, 0, 0),
}

BALL_REPOSITION_ORDER = [
    "yellow",
    "green",
    "brown",
    "blue",
    "pink",
    "black",
    "white",
]


@dataclass
class RepositionTarget:
    color_name: str
    x: int
    y: int
    led_color: tuple[int, int, int]


def snapshot_positions(ball_positions: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    return {
        color: [copy.deepcopy(position) for position in positions]
        for color, positions in (ball_positions or {}).items()
    }


def build_reposition_queue(
    shot_start_positions: Dict[str, List[dict]],
    current_positions: Dict[str, List[dict]],
    threshold_px: int,
) -> List[RepositionTarget]:
    queue: List[RepositionTarget] = []

    for color_name in BALL_REPOSITION_ORDER:
        starting_positions = shot_start_positions.get(color_name, [])
        if not starting_positions:
            continue

        current_for_color = current_positions.get(color_name, [])
        for start_position in starting_positions:
            nearest = _nearest_distance(start_position, current_for_color)
            if nearest is not None and nearest <= threshold_px:
                continue

            queue.append(
                RepositionTarget(
                    color_name=color_name,
                    x=int(start_position["x"]),
                    y=int(start_position["y"]),
                    led_color=BALL_LED_COLORS.get(color_name, (255, 255, 255)),
                )
            )

    return queue


def _nearest_distance(target: dict, candidates: Iterable[dict]) -> Optional[float]:
    best_distance = None
    for candidate in candidates:
        distance = math.hypot(target["x"] - candidate["x"], target["y"] - candidate["y"])
        if best_distance is None or distance < best_distance:
            best_distance = distance
    return best_distance


class FoulIndicatorClient:
    def __init__(
        self,
        broker: str,
        port: int,
        topic: str,
        username: str = "",
        password: str = "",
        tls_enabled: bool = True,
        tls_insecure: bool = True,
    ) -> None:
        self.broker = broker
        self.port = port
        self.topic = topic
        self.username = username
        self.password = password
        self.tls_enabled = tls_enabled
        self.tls_insecure = tls_insecure
        self.client = None

    def connect(self) -> bool:
        if mqtt is None:
            logger.warning("paho-mqtt is not installed; foul indicator is disabled.")
            return False
        if not self.broker or not self.topic:
            logger.warning("Foul indicator broker/topic missing; indicator is disabled.")
            return False

        client = mqtt.Client(client_id="SnookerVisionFoulLogic", protocol=mqtt.MQTTv311)
        if self.username:
            client.username_pw_set(self.username, self.password)
        if self.tls_enabled:
            client.tls_set()
            client.tls_insecure_set(self.tls_insecure)

        try:
            client.connect(self.broker, self.port)
            client.loop_start()
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not connect foul indicator MQTT client: %s", exc)
            return False

        self.client = client
        return True

    def highlight_target(self, target: RepositionTarget) -> bool:
        if self.client is None:
            return False
        r, g, b = target.led_color
        return self._publish(f"ball: {target.x},{target.y},{r},{g},{b}")

    def clear(self) -> bool:
        if self.client is None:
            return False
        return self._publish("clear")

    def close(self) -> None:
        if self.client is None:
            return
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass
        finally:
            self.client = None

    def _publish(self, payload: str) -> bool:
        try:
            result = self.client.publish(self.topic, payload)
        except Exception as exc:  # pragma: no cover
            logger.warning("Foul indicator publish failed for %r: %s", payload, exc)
            return False
        return result.rc == mqtt.MQTT_ERR_SUCCESS

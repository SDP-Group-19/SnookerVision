from __future__ import annotations

import logging
import time
from urllib import error, request

try:
    import serial
    from serial import SerialException
except ImportError:  # pragma: no cover - optional dependency
    serial = None
    SerialException = Exception


logger = logging.getLogger(__name__)


class ArduinoDisplayBridge:
    def __init__(self, port: str, baud: int = 115200) -> None:
        self.port = port.strip()
        self.baud = baud
        self.connection = None
        self.http_endpoints = []

        if self.port.startswith(("http://", "https://")):
            self.http_endpoints = [
                endpoint.strip().rstrip("/")
                for endpoint in self.port.split(",")
                if endpoint.strip()
            ]

    @property
    def is_available(self) -> bool:
        if self.http_endpoints:
            return bool(self.http_endpoints)
        return self.connection is not None

    def connect(self) -> bool:
        if self.http_endpoints:
            reachable = 0
            for endpoint in self.http_endpoints:
                try:
                    with request.urlopen(f"{endpoint}/status", timeout=1.5) as response:
                        payload = response.read().decode("utf-8", errors="replace").strip()
                except (error.URLError, TimeoutError, ValueError) as exc:
                    logger.warning("Could not reach ESP display endpoint %s: %s", endpoint, exc)
                    continue

                reachable += 1
                if payload:
                    logger.info("Connected to ESP display endpoint %s: %s", endpoint, payload)
                else:
                    logger.info("Connected to ESP display endpoint %s", endpoint)

            return reachable > 0

        if serial is None:
            logger.warning("pyserial is not installed; Arduino display bridge is disabled.")
            return False

        try:
            self.connection = serial.Serial(self.port, baudrate=self.baud, timeout=0.2)
        except SerialException as exc:
            logger.warning("Could not open Arduino port %s: %s", self.port, exc)
            self.connection = None
            return False

        time.sleep(2.0)
        self.connection.reset_input_buffer()
        logger.info("Connected to Arduino display on %s at %s baud.", self.port, self.baud)
        return True

    def close(self) -> None:
        if self.http_endpoints:
            return

        if self.connection is None:
            return

        try:
            self.connection.close()
        finally:
            self.connection = None

    def send_command(self, command: str) -> bool:
        if self.http_endpoints:
            return self._send_http_command(command)

        if self.connection is None:
            return False

        payload = command.strip()
        if not payload:
            return False

        try:
            self.connection.write((payload + "\n").encode("utf-8"))
            self.connection.flush()
        except SerialException as exc:
            logger.warning("Arduino write failed for %r: %s", payload, exc)
            self.close()
            return False

        logger.info("[ARDUINO] %s", payload)
        return True

    def _send_http_command(self, command: str) -> bool:
        payload = command.strip()
        if not payload or not self.http_endpoints:
            return False

        delivered = False
        body = payload.encode("utf-8")
        for endpoint in self.http_endpoints:
            http_request = request.Request(
                f"{endpoint}/command",
                data=body,
                headers={"Content-Type": "text/plain; charset=utf-8"},
                method="POST",
            )

            try:
                with request.urlopen(http_request, timeout=1.5) as response:
                    response_body = response.read().decode("utf-8", errors="replace").strip()
            except (error.URLError, TimeoutError, ValueError) as exc:
                logger.warning("ESP command failed for %s via %s: %s", payload, endpoint, exc)
                continue

            delivered = True
            if response_body:
                logger.info("[ESP %s] %s -> %s", endpoint, payload, response_body)
            else:
                logger.info("[ESP %s] %s", endpoint, payload)

        return delivered

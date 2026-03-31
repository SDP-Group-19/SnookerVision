from __future__ import annotations

import logging
import time

try:
    import serial
    from serial import SerialException
except ImportError:  # pragma: no cover - optional dependency
    serial = None
    SerialException = Exception


logger = logging.getLogger(__name__)


class ArduinoDisplayBridge:
    def __init__(self, port: str, baud: int = 115200) -> None:
        self.port = port
        self.baud = baud
        self.connection = None

    @property
    def is_available(self) -> bool:
        return self.connection is not None

    def connect(self) -> bool:
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
        if self.connection is None:
            return

        try:
            self.connection.close()
        finally:
            self.connection = None

    def send_command(self, command: str) -> bool:
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

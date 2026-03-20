#!/usr/bin/env python3
"""Serial monitor and command sender for SnookerVision Arduino boards."""

from __future__ import annotations

import argparse
import sys
import time

import serial
from serial import SerialException


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor pocket events or send display commands over serial."
    )
    parser.add_argument("--port", required=True, help="Serial port, for example /dev/cu.usbmodem1101")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate. Default: 115200")
    parser.add_argument(
        "--mode",
        choices=("monitor", "command"),
        default="monitor",
        help="Monitor serial output continuously or send one command.",
    )
    parser.add_argument(
        "--command",
        help="Command to send in command mode, for example 'STATUS' or 'SET 1 12'.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Optional monitor duration in seconds. Default 0 means run until Ctrl+C.",
    )
    return parser.parse_args()


def open_serial(port: str, baud: int) -> serial.Serial:
    try:
        connection = serial.Serial(port, baudrate=baud, timeout=0.2)
    except SerialException as exc:
        raise SystemExit(f"Could not open serial port {port}: {exc}") from exc

    time.sleep(2.0)
    connection.reset_input_buffer()
    return connection


def run_monitor(connection: serial.Serial, duration: float) -> None:
    deadline = time.monotonic() + duration if duration > 0 else None
    print(f"Monitoring {connection.port} at {connection.baudrate} baud. Press Ctrl+C to stop.")

    try:
        while True:
            raw_line = connection.readline()
            if raw_line:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line:
                    print(line)

            if deadline is not None and time.monotonic() >= deadline:
                break
    except KeyboardInterrupt:
        pass


def run_command(connection: serial.Serial, command: str | None) -> None:
    if not command:
        raise SystemExit("--command is required when --mode command is used.")

    payload = command.strip()
    if not payload:
        raise SystemExit("Command must not be empty.")

    connection.write((payload + "\n").encode("utf-8"))
    connection.flush()
    time.sleep(0.3)

    received = False
    while True:
        raw_line = connection.readline()
        if not raw_line:
            break

        line = raw_line.decode("utf-8", errors="replace").strip()
        if line:
            received = True
            print(line)

    if not received:
        print("No response received.")


def main() -> int:
    args = parse_args()
    connection = open_serial(args.port, args.baud)

    try:
        if args.mode == "monitor":
            run_monitor(connection, args.duration)
        else:
            run_command(connection, args.command)
    finally:
        connection.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

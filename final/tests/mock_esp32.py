"""Mock ESP32 TCP server — prints LED commands to terminal.

Run this instead of the real Arduino to see what gets sent:
    python final/tests/mock_esp32.py

Then run the test script in another terminal:
    python final/tests/test_led_reposition.py
"""
import socket
import threading

HOST = "127.0.0.1"
PORT = 4210


def handle_client(conn, addr):
    print(f"[CONNECTED] {addr}")
    buf = ""
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            buf += data.decode("ascii", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                if line.startswith("ball:"):
                    parts = line.split(":")[1].strip().split(",")
                    x, y, r, g, b = parts
                    print(f"  [LED ON]  position=({x},{y})  color=({r},{g},{b})")
                elif line == "clear":
                    print(f"  [CLEAR]   all LEDs off")
                elif line.startswith("resize:"):
                    dims = line.split(":")[1].strip()
                    print(f"  [RESIZE]  table={dims}")
                else:
                    print(f"  [???]     {line}")
    except (ConnectionResetError, BrokenPipeError):
        pass
    print(f"[DISCONNECTED] {addr}")
    conn.close()


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(2)
    print(f"Mock ESP32 listening on {HOST}:{PORT}")
    print("Waiting for connections...\n")
    try:
        while True:
            conn, addr = server.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        print("\nShutting down.")
    server.close()


if __name__ == "__main__":
    main()

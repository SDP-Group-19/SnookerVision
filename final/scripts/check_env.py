import importlib
import platform
import sys


def check_module(name):
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] {name} {version}")
        return module
    except Exception as exc:
        print(f"[MISSING] {name}: {exc}")
        return None


def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    cv2 = check_module("cv2")
    np = check_module("numpy")
    torch = check_module("torch")
    ultralytics = check_module("ultralytics")
    socketio = check_module("socketio")
    liveconfig = check_module("liveconfig")
    mqtt = check_module("paho.mqtt")

    if torch is not None:
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None:
            print(f"torch.backends.mps.is_built(): {mps_backend.is_built()}")
            print(f"torch.backends.mps.is_available(): {mps_backend.is_available()}")

    if cv2 is not None:
        print(f"OpenCV version: {cv2.__version__}")

    print("Environment check complete.")


if __name__ == "__main__":
    main()

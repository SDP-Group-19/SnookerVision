from liveconfig import liveinstance, trigger
from snookervision.state import StateManager
from snookervision.config import Config
import argparse
import logging
import cv2
import time
import os
import platform
import threading
from random import randint

config = liveinstance("config")(Config())
logger = logging.getLogger(__name__)


class State:
    def __init__(self):
        self.network = None
        self.autoencoder = None


# Create the state instance
state = State()

if config.use_obstruction_detection:
    from snookervision.detection import AutoEncoder
    state.autoencoder = AutoEncoder()

# Initialize networking if needed
if config.use_networking:
    from snookervision.networking import Network
    state.network = Network()
    state.network.connect()

# Initialize the state manager
state_manager = StateManager()
state_manager.initialize(config, state)


class ThreadedCamera:
    """
    Continuously grabs frames and keeps only the newest one to avoid latency buildup.
    """

    def __init__(self, camera):
        self.camera = camera
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.running = True
        self.latest_frame = None
        self.latest_ok = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ok, frame = self.camera.read()
            with self.condition:
                self.latest_ok = ok
                self.latest_frame = frame if ok else None
                self.condition.notify_all()
            if not ok:
                time.sleep(0.01)

    def read(self, timeout=None):
        with self.condition:
            if timeout is not None and self.latest_frame is None:
                end_time = time.time() + timeout
                while self.running and self.latest_frame is None:
                    remaining = end_time - time.time()
                    if remaining <= 0:
                        break
                    self.condition.wait(timeout=remaining)
            if self.latest_frame is None:
                return False, None
            return self.latest_ok, self.latest_frame.copy()

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.camera.release()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stream",
        type=str,
        help="Network stream URL (e.g. tcp://192.168.1.10:8888)"
    )

    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="The path to the image file."
    )

    parser.add_argument(
        "--no-interface",
        action="store_true",
        help="Disable the interface.",
        default=False
    )

    parser.add_argument(
        "--interface-port",
        type=int,
        default=5000,
        help="Port for the web interface (default: 5000)."
    )

    parser.add_argument(
        "--camera-port",
        type=int,
        default=config.camera_port,
        help="The camera port to use."
    )

    parser.add_argument(
        "--camera-source",
        type=str,
        default=config.camera_source,
        help="Camera source URL/device path, e.g. http://raspberrypi.local:8000/stream.mjpg"
    )

    parser.add_argument(
        "--camera-width",
        type=int,
        default=config.camera_width,
        help="Requested capture width."
    )

    parser.add_argument(
        "--camera-height",
        type=int,
        default=config.camera_height,
        help="Requested capture height."
    )

    parser.add_argument(
        "--camera-fps",
        type=int,
        default=config.camera_fps,
        help="Requested camera FPS."
    )

    parser.add_argument(
        "--process-every-n-frames",
        type=int,
        default=config.process_every_n_frames,
        help="Run YOLO every Nth frame and reuse the last detection on skipped frames."
    )

    parser.add_argument(
        "--detector-imgsz",
        type=int,
        default=config.detector_imgsz,
        help="YOLO inference image size. Lower values usually increase FPS."
    )

    parser.add_argument(
        "--detector-device",
        type=str,
        default=config.detector_device,
        choices=("auto", "cpu", "cuda", "mps"),
        help="Inference device selection."
    )

    parser.add_argument(
        "--hide-windows",
        action="store_true",
        default=config.hide_windows,
        help="Do not open OpenCV display windows."
    )

    parser.add_argument(
        "--no-draw-results",
        action="store_true",
        default=False,
        help="Run detection without drawing boxes or overlays."
    )

    parser.add_argument(
        "--show-generated-table",
        action="store_true",
        default=config.show_generated_table,
        help="Show the generated table view."
    )

    parser.add_argument(
        "--fast-mode",
        action="store_true",
        default=config.fast_mode,
        help="Skip non-essential game logic and overlays to maximize throughput."
    )

    parser.add_argument(
        "--use-calibration",
        action="store_true",
        default=config.use_calibration,
        help="Enable camera undistortion using saved calibration parameters."
    )

    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip camera calibration/undistortion.",
        default=False
    )

    parser.add_argument(
        "--no-table-pts",
        action="store_true",
        help="Skip table point selection and perspective transform.",
        default=False
    )

    parser.add_argument(
        "--select-table-pts",
        action="store_true",
        help="Force selecting table points again and overwrite saved points.",
        default=False
    )

    parser.add_argument(
        "--overlay-only",
        action="store_true",
        help="Show only the generated table overlay window, not the live camera view.",
        default=False
    )

    parser.add_argument(
        "--arduino-port",
        type=str,
        default=None,
        help="Optional display controller target. Use a serial port like /dev/cu.usbmodem1101 or comma-separated HTTP endpoints like http://snooker-display-1.local,http://snooker-display-2.local.",
    )

    parser.add_argument(
        "--arduino-baud",
        type=int,
        default=115200,
        help="Baud rate for the Arduino display controller. Default: 115200.",
    )

    parser.add_argument(
        "--led-enabled",
        action="store_true",
        default=config.led_enabled,
        help="Enable LED strip reposition indicator via ESP32."
    )

    parser.add_argument(
        "--led-ip",
        type=str,
        default=config.led_arduino_ip,
        help="ESP32 IP address for LED control (default: 192.168.1.42)."
    )

    parser.add_argument(
        "--led-port",
        type=int,
        default=config.led_arduino_port,
        help="ESP32 TCP port for LED control (default: 4210)."
    )

    parser.add_argument(
        "--show-trajectory",
        action="store_true",
        default=config.show_trajectory,
        help="Show predicted cue ball trajectory lines."
    )

    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        default=False,
        help="Disable trajectory prediction."
    )

    return parser.parse_args()


def load_camera():
    """
    This function loads the camera from a configured network source or local camera port.
    """
    try:
        logger.info("Starting camera...")
        if config.camera_source:
            logger.info(f"Opening remote camera source: {config.camera_source}")
            camera = cv2.VideoCapture(config.camera_source, apiPreference=cv2.CAP_ANY)
            if not camera.isOpened():
                logger.error(
                    f"Could not open camera source {config.camera_source}."
                )
                return None
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                camera.set(cv2.CAP_PROP_BUFFERSIZE, config.camera_buffer_size)
            return ThreadedCamera(camera)

        system = platform.system()
        if system == "Darwin":
            backend = cv2.CAP_AVFOUNDATION
        elif system == "Windows":
            backend = cv2.CAP_MSMF
        elif hasattr(cv2, "CAP_V4L2"):
            backend = cv2.CAP_V4L2
        else:
            backend = cv2.CAP_ANY

        camera = cv2.VideoCapture(config.camera_port, apiPreference=backend)
        if not camera.isOpened():
            logger.error(
                f"Could not open camera index {config.camera_port} with backend {backend}."
            )
            return None

        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            camera.set(cv2.CAP_PROP_BUFFERSIZE, config.camera_buffer_size)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
        camera.set(cv2.CAP_PROP_FPS, config.camera_fps)
        time.sleep(2.0)
        return ThreadedCamera(camera)
    except Exception as e:
        logger.error(f"Error starting camera: {e}")
        return


def capture_frame(path, frame) -> None:
    """
    This function captures a the current frame and saves it to the specified path.
    """
    if config.collect_model_images:
        path = config.model_image_path
        if not os.path.exists(path):
            os.makedirs(path)
        _capture_and_save(path, frame)

    if config.collect_ae_data:
        path = config.ae_data_path
        if not os.path.exists(path):
            os.makedirs(path)
        _capture_and_save(path, frame)


def _capture_and_save(path, frame):
    if cv2.waitKey(1) & 0xFF == ord('t'):
        num = randint(0, 10000)
        filename = f"{path}image_{num}.jpg"
        cv2.imwrite(filename, frame)
        time.sleep(0.1)
        logger.info(f"Image {num} saved")


@trigger
def start_network():
    if config.use_networking and state.network is None:
        from snookervision.networking import Network
        logger.info("Starting network...")
        state.network = Network()
        state.network.connect()


@trigger
def stop_network():
    if config.use_networking and state.network is not None:
        logger.info("Stopping network...")
        state.network.disconnect()
        state.network = None

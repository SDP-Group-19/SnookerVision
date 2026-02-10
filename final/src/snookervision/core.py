from liveconfig import liveinstance, trigger
from final.src.snookervision.state import StateManager
from final.src.snookervision.config import Config
import argparse
import logging
import cv2
import time
import os
import platform
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
    from final.src.snookervision.detection import AutoEncoder
    state.autoencoder = AutoEncoder()

# Initialize networking if needed
if config.use_networking:
    from final.src.snookervision.networking import Network
    state.network = Network()
    state.network.connect()

# Initialize the state manager
state_manager = StateManager()
state_manager.initialize(config, state)


def parse_args():
    parser = argparse.ArgumentParser()
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

    return parser.parse_args()


def load_camera():
    """
    This function loads the camera from the camera port specified in the config.
    """
    try:
        logger.info("Starting camera...")
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

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        camera.set(cv2.CAP_PROP_FPS, 30)
        time.sleep(2.0)
        return camera
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
        from final.src.snookervision.networking import Network
        logger.info("Starting network...")
        state.network = Network()
        state.network.connect()


@trigger
def stop_network():
    if config.use_networking and state.network is not None:
        logger.info("Stopping network...")
        state.network.disconnect()
        state.network = None

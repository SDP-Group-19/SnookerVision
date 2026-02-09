import time
import numpy as np
import cv2
import logging
from liveconfig import LiveConfig, start_interface
from snookervision.processing import (
    get_top_down_view,
    handle_calibration,
    undistort_frame,
    manage_point_selection,
)

from snookervision.detection import DetectionModel
from snookervision import config, state, load_camera, parse_args, capture_frame
from snookervision.game_logic.game_logic import GameState, RuleEngine
from snookervision.perception.event_detector import EventDetector
from snookervision.state import StateManager



LiveConfig("./tests/src/data")

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s (%(filename)s:%(lineno)d)",
    handlers=[
        logging.StreamHandler()
    ]
)


def main():
    from collections import deque
    times = deque(maxlen=30)
    fps = 0.0
    args = parse_args()
    config.camera_port = args.camera_port
    if not args.no_interface:
        start_interface("web")

    if args.file is not None:
        camera = cv2.VideoCapture(args.file)
        ret, frame = camera.read()
        if not ret:
            logger.error("Failed to read from video file.")
            return
    else:
        camera = load_camera()
        ret, frame = camera.read()
        if not ret:
            logger.error("Failed to read from camera.")
            return

    processed_frame = frame
    
    # Handle calibration if not disabled
    homography_matrix = None
    if not args.no_calibration:
        mtx, dist, newcameramtx, roi = handle_calibration(frame)
        processed_frame = undistort_frame(frame, mtx, dist, newcameramtx, roi)
    
    # Handle table point selection if not disabled
    if not args.no_table_pts:
        table_pts = manage_point_selection(processed_frame)
        if table_pts is None:
            logger.error("Table points not selected. Continuing without.")
            config.use_table_pts = False
        else:
            table_rect = np.float32([
                [0, 0],
                [config.output_dimensions[0], 0],
                [0, config.output_dimensions[1]],
                [config.output_dimensions[0], config.output_dimensions[1]]
            ])
            homography_matrix = cv2.getPerspectiveTransform(table_pts, table_rect)
            processed_frame = get_top_down_view(processed_frame, homography_matrix)
    else:
        config.use_table_pts = False


    detection_model = DetectionModel()
    if detection_model.model is None:
        return

    # Game logic and event detector integration
    game_state = GameState()
    game_state.start_frame()
    rule_engine = RuleEngine(game_state)
    event_detector = EventDetector()
    state_manager = StateManager()
    state_manager.initialize(config, state)
    state_manager.set_game_logic(rule_engine, event_detector)

    # Create resizable window for fullscreen capability
    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)

    import logging
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
    last_fps_log_time = time.time()
    fps_log_interval = 2.0  # seconds

    while True:

        ret, frame = camera.read()
        if not ret or frame is None:
            if args.file is not None:
                logger.info("Video finished.")
            else:
                logger.error("Failed to read from camera.")
            break

        processed_frame = frame

        if config.use_calibration:
            processed_frame = undistort_frame(
                frame, mtx, dist, newcameramtx, roi)
        if config.use_table_pts:
            processed_frame = get_top_down_view(
                processed_frame, homography_matrix)
        if config.collect_model_images or config.collect_ae_data:
            capture_frame(processed_frame)

        detections, labels = detection_model.handle_detection(processed_frame, fps)
        state_manager.update(detections, labels)

        if state.autoencoder is not None \
                and config.use_obstruction_detection \
                and config.use_model:
            table_only = detection_model.extract_bounding_boxes(
                processed_frame,
                detections)
            state.autoencoder.handle_obstruction_detection(table_only)

        now = time.time()
        times.append(now)
        if len(times) > 1:
            fps = (len(times) - 1) / (times[-1] - times[0])
        else:
            fps = 0.0

        # Log FPS every fps_log_interval seconds
        if now - last_fps_log_time >= fps_log_interval:
            logging.info(f"Current FPS: {fps:.2f}")
            last_fps_log_time = now

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    if config.use_networking:
        state.network.disconnect()


if __name__ == "__main__":
    main()
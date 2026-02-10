import sys
from pathlib import Path
import time
import numpy as np
import cv2
import logging
from liveconfig import LiveConfig, start_interface

# Allow running this file directly: `python final/src/snookervision/app/main.py`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from final.src.snookervision.processing import (
    get_top_down_view,
    handle_calibration,
    undistort_frame,
    manage_point_selection,
)

from final.src.snookervision.detection import DetectionModel
from final.src.snookervision import config, state, load_camera, parse_args, capture_frame
from final.src.snookervision.state import StateManager
from final.src.snookervision.visualization import GeneratedTableRenderer



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
        start_interface("web", port=args.interface_port)

    if args.file is not None:
        camera = cv2.VideoCapture(args.file)
        ret, frame = camera.read()
        if not ret:
            logger.error("Failed to read from video file.")
            return
    else:
        camera = load_camera()
        if camera is None:
            logger.error(
                "Camera initialization failed. Try --camera-port 1 (or 2) and --no-interface."
            )
            return
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
        table_pts = manage_point_selection(
            processed_frame,
            force_reselect=args.select_table_pts,
        )
        if table_pts is None:
            logger.error("Table points not selected. Continuing without.")
            config.use_table_pts = False
        else:
            config.use_table_pts = True
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
    table_renderer = GeneratedTableRenderer(config.generated_table_size)

    state_manager = StateManager()
    state_manager.initialize(config, state)

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
            capture_frame(None, processed_frame)

        overlay_lines = state_manager.get_overlay_lines()
        detections, labels = detection_model.handle_detection(
            processed_frame,
            fps,
            overlay_lines=overlay_lines,
        )
        state_manager.update(detections, labels)

        if config.show_generated_table and not config.hide_windows:
            markers = detection_model.get_ball_markers(processed_frame, detections)
            generated_table = table_renderer.render(processed_frame.shape, markers)
            for idx, text in enumerate(overlay_lines):
                y = 22 + (idx * 18)
                cv2.putText(
                    generated_table,
                    text,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                )
            cv2.imshow(config.generated_table_window_name, generated_table)
        else:
            try:
                if cv2.getWindowProperty(config.generated_table_window_name, cv2.WND_PROP_VISIBLE) >= 0:
                    cv2.destroyWindow(config.generated_table_window_name)
            except cv2.error:
                pass

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

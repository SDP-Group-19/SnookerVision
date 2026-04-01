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

from snookervision.processing import (
    get_top_down_view,
    handle_calibration,
    undistort_frame,
    manage_point_selection,
)

from snookervision.detection import DetectionModel
from snookervision import config, state, load_camera, parse_args, capture_frame
from snookervision.state import StateManager
from snookervision.visualization import GeneratedTableRenderer
from snookervision.trajectory import TrajectoryPredictor, CueDetector, TrajectoryRenderer



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
    config.camera_source = args.camera_source
    config.camera_width = args.camera_width
    config.camera_height = args.camera_height
    config.camera_fps = args.camera_fps
    config.process_every_n_frames = max(1, args.process_every_n_frames)
    config.detector_imgsz = max(128, args.detector_imgsz)
    config.detector_device = args.detector_device
    config.fast_mode = args.fast_mode
    config.hide_windows = args.hide_windows
    config.draw_results = not args.no_draw_results
    config.show_generated_table = args.show_generated_table
    config.use_calibration = args.use_calibration
    config.led_enabled = args.led_enabled
    config.led_arduino_ip = args.led_ip
    config.led_arduino_port = args.led_port
    config.show_trajectory = args.show_trajectory and not args.no_trajectory
    if not args.no_interface:
        start_interface("web", port=args.interface_port)

    if hasattr(args, "stream") and args.stream is not None:
        logger.info(f"Connecting to stream: {args.stream}")
        camera = cv2.VideoCapture(args.stream, cv2.CAP_FFMPEG)

        # 减少延迟
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 等待流稳定
        time.sleep(1)

        ret, frame = camera.read()
        if not ret:
            logger.error("Failed to read from stream.")
            return

    elif args.file is not None:
        camera = cv2.VideoCapture(args.file)
        ret, frame = camera.read()
        if not ret:
            logger.error("Failed to read from video file.")
            return

    else:
        camera = load_camera()
        if camera is None:
            logger.error(
                "Camera initialization failed. Try --camera-source http://<raspberrypi>:8000/stream.mjpg or --camera-port 1 and --no-interface."
            )
            return
        ret, frame = camera.read(timeout=5.0)
        if not ret:
            logger.error(
                "Failed to read the first frame from camera within 5 seconds. "
                "If you are using a Raspberry Pi stream, verify the Pi server is running and the stream URL opens in a browser."
            )
            return

    processed_frame = frame
    
    # Handle calibration if not disabled
    homography_matrix = None
    if not args.no_calibration:
        mtx, dist, newcameramtx, roi = handle_calibration(frame)
        processed_frame = undistort_frame(frame, mtx, dist, newcameramtx, roi)
    
    # Handle table point selection if not disabled
    if not args.no_table_pts:
        force_reselect = args.select_table_pts or args.file is not None
        if args.file is not None and not args.select_table_pts:
            logger.info(
                "Video file input detected; please select 4 table corners for this clip."
            )
        table_pts = manage_point_selection(
            processed_frame,
            force_reselect=force_reselect,
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
    table_renderer = None
    if config.show_generated_table and not config.hide_windows:
        table_renderer = GeneratedTableRenderer(config.generated_table_size)
    show_live_view = not args.overlay_only

    # Trajectory prediction
    trajectory_predictor = None
    cue_detector = None
    trajectory_renderer = None
    if config.show_trajectory:
        tw, th = config.output_dimensions
        trajectory_predictor = TrajectoryPredictor(tw, th)
        cue_detector = CueDetector()
        trajectory_renderer = TrajectoryRenderer()

    state_manager = StateManager()
    state_manager.initialize(
        config,
        state,
        arduino_port=args.arduino_port,
        arduino_baud=args.arduino_baud,
    )

    # Create resizable window for fullscreen capability
    if not config.hide_windows and show_live_view:
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

        overlay_lines = None if config.fast_mode else state_manager.get_overlay_lines()
        detections, labels = detection_model.handle_detection(
            processed_frame,
            fps,
            overlay_lines=overlay_lines,
            show_live_view=show_live_view,
        )
        if not config.fast_mode:
            state_manager.update(detections)

        # --- Trajectory prediction ---
        if (
            config.show_trajectory
            and trajectory_predictor is not None
            and detections
            and not state_manager.shot_active
        ):
            white_pos = None
            other_balls = []
            for det in detections:
                if det["label"] == "white":
                    white_pos = det["center"]
                elif det["label"] in detection_model.ball_class_names:
                    cx, cy = det["center"]
                    other_balls.append({"x": cx, "y": cy, "color": det["label"]})

            if white_pos is not None:
                aim = cue_detector.detect_aim(processed_frame, white_pos, detections)
                if aim is not None:
                    prediction = trajectory_predictor.predict(
                        cue_pos=white_pos,
                        aim_direction=aim,
                        balls=other_balls,
                    )
                    if prediction is not None:
                        if not config.hide_windows:
                            trajectory_renderer.draw_on_frame(
                                processed_frame, prediction)
                        if config.trajectory_led and state_manager.led_controller:
                            trajectory_renderer.send_to_leds(
                                state_manager.led_controller, prediction)

        if (
            not config.fast_mode
            and config.show_generated_table
            and not config.hide_windows
            and table_renderer is not None
        ):
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

        if not config.fast_mode and state.autoencoder is not None \
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

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r") and state_manager.foul_reposition_active:
            # Skip current ball or clear if last one
            state_manager.skip_reposition_target()

    camera.release()
    cv2.destroyAllWindows()
    state_manager.clear_foul_leds()
    if state_manager.led_controller:
        state_manager.led_controller.close()
    state_manager.shutdown()
    if config.use_networking:
        state.network.disconnect()


if __name__ == "__main__":
    main()

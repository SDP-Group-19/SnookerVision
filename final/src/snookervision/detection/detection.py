import cv2
import logging
import os
from collections import defaultdict
from ultralytics import YOLO
import numpy as np
import torch

from snookervision.core import config

logger = logging.getLogger(__name__)


def resolve_torch_device():
    requested = getattr(config, "detector_device", "auto")
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable, falling back to CPU.")
            return "cpu"
        if requested == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is None or not mps_backend.is_available():
                mps_built = mps_backend is not None and mps_backend.is_built()
                logger.warning(
                    "MPS requested but unavailable, falling back to CPU. "
                    f"mps_built={mps_built}"
                )
                return "cpu"
        return requested

    if torch.cuda.is_available():
        logger.info("Auto-selected CUDA for detection.")
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        logger.info("Auto-selected MPS for detection.")
        return "mps"

    logger.info("No GPU backend available, using CPU for detection.")
    return "cpu"


class DetectionModel:
    def __init__(self):
        self.device = resolve_torch_device()
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
        self.model = self.load_model()
        self.labels = self._normalize_labels(
            self.model.names if self.model is not None else {}
        )
        self.total_objects = 0
        self.total_balls = 0
        self.hole_positions = self._init_pocket_positions()
        self.frame_count = 0
        self.found_holes = []
        self.last_result = None
        self.ball_class_names = {
            "white", "black", "red", "yellow", "green", "brown", "blue", "pink"
        }
        self._last_pocket_filter_log = {}  # throttle pocket filter logging

    def _init_pocket_positions(self):
        if config.pocket_pts is not None:
            return list(config.pocket_pts)
        return [
            (0, 0),
            (config.output_dimensions[0] // 2, 0),
            (config.output_dimensions[0], 0),
            (0, config.output_dimensions[1]),
            (config.output_dimensions[0] // 2, config.output_dimensions[1]),
            (config.output_dimensions[0], config.output_dimensions[1]),
        ]

    def _is_near_pocket(self, cx, cy):
        radius = config.pocket_filter_radius_px
        for px, py in self.hole_positions:
            if (cx - px) ** 2 + (cy - py) ** 2 <= radius ** 2:
                return True
        return False

    def _normalize_labels(self, names):
        """Normalize model class names so the rest of the codebase works.

        Handles models that use 'black-ball' style names by stripping '-ball',
        and maps 'pocket' to 'hole' for compatibility.
        """
        normalized = {}
        for idx, name in names.items():
            n = name.lower().replace("-ball", "").replace("_ball", "")
            if n == "pocket":
                n = "hole"
            normalized[idx] = n
        return normalized

    def _label_to_bbox_color(self, classname):
        """Map a normalized class name to a bbox color."""
        color_map = {
            "red": (0, 0, 255),
            "white": (255, 255, 255),
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "brown": (42, 42, 165),
            "pink": (203, 192, 255),
            "black": (0, 0, 0),
            "hole": (128, 128, 128),
            "arm": (0, 165, 255),
        }
        return color_map.get(classname, (255, 255, 255))

    def load_model(self):
        if not os.path.exists(config.detection_model_path):
            logger.error(
                f"Model file not found at {config.detection_model_path}.")
            return None
        else:
            model = YOLO(config.detection_model_path, task="detect")
            model.to(self.device)
            logger.info(f"Detection model running on device: {self.device}")
            return model

    # Can have as a trigger function to change the model during runtime. Not used yet, waiting for liveconfig to be updated.
    # @trigger
    def change_model(self, path=None):
        if path is None:
            path = config.detection_model_path
        if os.path.exists(path):
            logger.info(f"Loading model from {path}.")
            self.model = YOLO(path, task="detect")
        else:
            logger.error(
                f"Model file not found at {path}. Continuing with existing model.")
            return None

    def detect(self, frame):
        self.frame_count += 1
        if self.frame_count % config.process_every_n_frames != 0:
            return self.last_result, self.labels

        use_half = self.device == "cuda"

        with torch.inference_mode():
            results = self.model.predict(
                source=frame,
                verbose=False,
                conf=config.conf_threshold,
                iou=0.40,
                device=self.device,
                half=use_half,
                imgsz=config.detector_imgsz,
                stream=False,
            )

        result = results[0] if results else None
        if result is None or result.boxes is None:
            return None, None

        parsed_results = self._parse_result_boxes(result)
        parsed_results.sort(key=lambda item: item["conf"], reverse=True)
        filtered_results = self._filter_results(parsed_results)
        self.last_result = filtered_results
        return self.last_result, self.labels

    def _parse_result_boxes(self, result):
        boxes = result.boxes
        xyxy = boxes.xyxy.detach().to("cpu").numpy().astype(int)
        class_indices = boxes.cls.detach().to("cpu").numpy().astype(int)
        confidences = boxes.conf.detach().to("cpu").numpy()

        parsed = []
        for bbox, classidx, conf in zip(xyxy, class_indices, confidences):
            xmin, ymin, xmax, ymax = bbox.tolist()
            classname = self.labels[classidx]
            parsed.append(
                {
                    "classidx": classidx,
                    "label": classname,
                    "color": self._label_to_bbox_color(classname),
                    "bbox": (xmin, ymin, xmax, ymax),
                    "center": ((xmin + xmax) // 2, (ymin + ymax) // 2),
                    "conf": float(conf),
                }
            )
        return parsed

    def _filter_results(self, all_results):
        filtered_results = []
        self.found_holes = []
        counts = defaultdict(int)
        self.total_balls = 0

        class_limits = {
            "white": 1,
            "black": 1,
            "red": 15,  # Up to 15 reds in snooker
            "yellow": 1,
            "green": 1,
            "brown": 1,
            "blue": 1,
            "pink": 1,
            "hole": 6,
            "arm": 3}

        for result in all_results:
            classname = result["label"]
            xmin, ymin, xmax, ymax = result["bbox"]
            area = (xmax - xmin) * (ymax - ymin)

            if classname in class_limits \
                    and counts[classname] < class_limits[classname]:

                if classname in {"white", "black", "red", "yellow", "green", "brown", "blue", "pink"} \
                        and self._is_likely_ball(area):
                    cx, cy = result["center"]
                    if self._is_near_pocket(cx, cy):
                        if self._last_pocket_filter_log.get(classname) != self.frame_count - 1:
                            logger.info(f"[POCKET] {classname} filtered at ({cx}, {cy}) — inside pocket zone")
                        self._last_pocket_filter_log[classname] = self.frame_count
                        continue
                    counts[classname] += 1
                    filtered_results.append(result)
                    self.total_balls += 1

                elif classname == "hole" \
                        and self._is_likely_hole(xmin, ymin, xmax, ymax):
                    counts[classname] += 1
                    filtered_results.append(result)

                elif classname == "arm" \
                        and self._is_likely_arm(area):
                    counts[classname] += 1
                    filtered_results.append(result)

        return filtered_results

    def _is_likely_ball(self, area):
        if config.fast_mode or not config.use_table_pts:
            return area > 0
        return area > config.ball_area_range[0] \
            and area < config.ball_area_range[1]

    def _is_likely_arm(self, area):
        if config.fast_mode or not config.use_table_pts:
            return area > 0
        return area > config.arm_area_range[0] \
            and area < config.arm_area_range[1]

    def _is_likely_hole(self, xmin, ymin, xmax, ymax):
        if config.fast_mode or not config.use_table_pts:
            return True
        middlex = int((xmin + xmax) / 2)
        middley = int((ymin + ymax) / 2)

        if self._is_near_existing_hole(middlex, middley):
            return False
        if self._hole_is_near_expected_position(middlex, middley):
            self.found_holes.append((middlex, middley))
            return True

    def _hole_is_near_expected_position(self, x, y):
        for hole in self.hole_positions:
            if abs(hole[0] - x) < config.hole_threshold \
                    and abs(hole[1] - y) < config.hole_threshold:
                return True
        return False

    def _is_near_existing_hole(self, x, y):
        for hole in self.found_holes:
            if abs(hole[0] - x) < config.hole_threshold \
                    and abs(hole[1] - y) < config.hole_threshold:
                return True
        return False

    def _sample_ball_color(self, frame, center_x, center_y, radius):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        mean_bgr = cv2.mean(frame, mask=mask)[:3]
        return tuple(int(c) for c in mean_bgr)

    def get_ball_markers(self, frame, filtered_results):
        markers = []
        if not filtered_results:
            return markers

        frame_height, frame_width = frame.shape[:2]

        for result in filtered_results:
            classname = result["label"]
            xmin, ymin, xmax, ymax = result["bbox"]
            if classname not in self.ball_class_names:
                continue

            center_x, center_y = result["center"]
            center_x = int(np.clip(center_x, 0, frame_width - 1))
            center_y = int(np.clip(center_y, 0, frame_height - 1))

            radius = max(2, min((xmax - xmin), (ymax - ymin)) // 4)
            color = self._sample_ball_color(frame, center_x, center_y, radius)
            markers.append(
                {
                    "label": classname,
                    "center": (center_x, center_y),
                    "color": color,
                }
            )

        return markers

    def draw(self, frame, filtered_results, fps=0, overlay_lines=None):
        frame_height, frame_width = frame.shape[:2]
        
        # Draw FPS in top right corner (small text)
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (frame_width - 90, 24), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        
        if not filtered_results:
            self._draw_notifications(frame, overlay_lines)
            return

        self.total_objects = 0

        object_data = []
        for result in filtered_results:
            classname = result["label"]
            color = result["color"]
            xmin, ymin, xmax, ymax = result["bbox"]
            conf = result["conf"]

            if conf > config.conf_threshold:
                object_data.append(
                    (classname, color, xmin, ymin, xmax, ymax, conf))

        for classname, color, xmin, ymin, xmax, ymax, conf in object_data:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.circle(
                frame,
                ((xmin + xmax) // 2, (ymin + ymax) // 2),
                4, (0, 0, 255), -1)

            label = f"{classname}: {int(conf * 100)}%"
            label_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                config.font_scale,
                config.font_thickness)
            label_ymin = max(ymin, label_size[1] + 10)

            cv2.putText(
                frame,
                label,
                (xmin, label_ymin - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.font_scale,
                config.font_color,
                config.font_thickness)

            self.total_objects += 1

        # Summary text (half size, under FPS on right side)
        summary_text = [
            f'Objects: {self.total_objects}',
            f'Balls: {self.total_balls}'
        ]
        for i, text in enumerate(summary_text):
            cv2.putText(
                frame,
                text, (frame_width - 90, 40 + i * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1)

        # Draw detected colors box in top right (1/3 size), below summary
        detected_colors = sorted(set([classname for classname, _, _, _, _, _, _ in object_data]))
        if detected_colors:
            # Calculate box dimensions (1/3 original size)
            box_width = 100
            box_height = 22 + len(detected_colors) * 18
            box_x = frame_width - box_width - 10
            box_y = 70  # Below FPS and summary
            
            # Draw semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Draw border
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), 1)
            
            # Draw title (slightly bigger text)
            cv2.putText(frame, "Colors:", (box_x + 5, box_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
            # Draw each detected color (slightly bigger text)
            for i, color_name in enumerate(detected_colors):
                cv2.putText(frame, f"{color_name.capitalize()}", 
                           (box_x + 5, box_y + 32 + i * 18), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        self._draw_notifications(frame, overlay_lines)

    def _draw_notifications(self, frame, overlay_lines):
        if not overlay_lines:
            return
        padding = 8
        line_h = 20
        box_width = 420
        box_height = padding * 2 + len(overlay_lines) * line_h
        x0, y0 = 10, 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_width, y0 + box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + box_width, y0 + box_height), (255, 255, 255), 1)

        for i, text in enumerate(overlay_lines):
            y = y0 + padding + 14 + i * line_h
            cv2.putText(
                frame,
                text,
                (x0 + padding, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    def extract_bounding_boxes(self, frame, results):
        bounding_boxes = []
        if not results:
            return None
        for result in results:
            bounding_boxes.append(result["bbox"])

        mask = np.zeros_like(frame[:, :, 0])
        for (xmin, ymin, xmax, ymax) in bounding_boxes:
            mask[ymin:ymax, xmin:xmax] = 255

        return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

    def handle_detection(self, frame, fps=0, overlay_lines=None, show_live_view=True):
        detections = None
        labels = None
        if not show_live_view:
            self.destroy_camera_frame_window()
            self.destroy_detection_drawing_window()
        if not config.use_model:
            if not config.hide_windows and show_live_view:
                self.destroy_detection_drawing_window()
                cv2.imshow("Camera Frame", frame)
        else:
            detections, labels = self.detect(frame)
            if config.draw_results and not config.hide_windows and show_live_view:
                self.destroy_camera_frame_window()
                drawing_frame = frame.copy()
                self.draw(drawing_frame, detections, fps, overlay_lines=overlay_lines)
                cv2.imshow("Detection", drawing_frame)
            elif not config.hide_windows and show_live_view:
                self.destroy_detection_drawing_window()
                cv2.imshow("Camera Frame", frame)
            elif config.hide_windows:
                self.destroy_camera_frame_window()
                self.destroy_detection_drawing_window()

        return detections, labels

    def destroy_camera_frame_window(self):
        try:
            if cv2.getWindowProperty("Camera Frame", cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow("Camera Frame")
        except cv2.error as e:
            pass

    def destroy_detection_drawing_window(self):
        try:
            if cv2.getWindowProperty("Detection", cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow("Detection")
        except cv2.error as e:
            pass

"""Detect the cue stick and determine its aiming direction.

Uses the "arm" detection from YOLO + line detection (HoughLines) to find
the cue stick direction. The aim line passes through the cue ball.

Two approaches combined:
1. Hough line detection on the arm/cue region
2. If the cue tip is visible, use it directly with the white ball position
"""
import cv2
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)


class CueDetector:
    def __init__(self):
        self.last_aim_direction = None
        self.last_cue_tip = None
        self.smoothing = 0.3  # low-pass filter for stability

    def detect_aim(self, frame, white_ball_pos, detections=None):
        """Detect cue aim direction from the frame.

        Args:
            frame: the current video frame (BGR)
            white_ball_pos: (x, y) of the white/cue ball
            detections: list of YOLO detections (may contain "arm" class)

        Returns:
            (dx, dy) normalized aim direction, or None if not detected
        """
        if white_ball_pos is None:
            return None

        wx, wy = white_ball_pos

        # Method 1: Use arm detection bounding box to narrow the search area
        arm_region = None
        if detections:
            for det in detections:
                if det.get("label") == "arm":
                    arm_region = det["bbox"]
                    break

        # Method 2: Hough line detection near the white ball
        aim = self._detect_cue_line(frame, wx, wy, arm_region)

        if aim is not None:
            if self.last_aim_direction is not None:
                # Smooth the direction to reduce jitter
                a = self.smoothing
                px, py = self.last_aim_direction
                aim = (
                    a * aim[0] + (1 - a) * px,
                    a * aim[1] + (1 - a) * py,
                )
                length = math.hypot(aim[0], aim[1])
                if length > 1e-6:
                    aim = (aim[0] / length, aim[1] / length)
            self.last_aim_direction = aim

        return aim

    def _detect_cue_line(self, frame, wx, wy, arm_bbox=None):
        """Find the cue stick line using edge detection + Hough transform."""
        h, w = frame.shape[:2]

        # Define region of interest around the white ball
        margin = 150
        if arm_bbox is not None:
            # Expand search to include the arm region
            ax1, ay1, ax2, ay2 = arm_bbox
            x1 = max(0, min(int(wx) - margin, ax1))
            y1 = max(0, min(int(wy) - margin, ay1))
            x2 = min(w, max(int(wx) + margin, ax2))
            y2 = min(h, max(int(wy) + margin, ay2))
        else:
            x1 = max(0, int(wx) - margin)
            y1 = max(0, int(wy) - margin)
            x2 = min(w, int(wx) + margin)
            y2 = min(h, int(wy) + margin)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Edge detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Hough lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=30,
            minLineLength=40,
            maxLineGap=15,
        )

        if lines is None:
            return None

        # Find the line closest to the white ball that looks like a cue
        best_line = None
        best_score = float("inf")
        local_wx = wx - x1
        local_wy = wy - y1

        for line in lines:
            lx1, ly1, lx2, ly2 = line[0]
            line_len = math.hypot(lx2 - lx1, ly2 - ly1)
            if line_len < 30:
                continue

            # Distance from white ball to this line
            dist = self._point_to_line_dist(local_wx, local_wy, lx1, ly1, lx2, ly2)

            # The cue should be close to the white ball
            # Score: distance to white ball (lower is better)
            if dist < best_score and dist < 50:
                best_score = dist
                best_line = (lx1, ly1, lx2, ly2)

        if best_line is None:
            return None

        lx1, ly1, lx2, ly2 = best_line

        # Convert to global coords
        gx1, gy1 = lx1 + x1, ly1 + y1
        gx2, gy2 = lx2 + x1, ly2 + y1

        # Direction: from the farther end toward the white ball
        d1 = math.hypot(gx1 - wx, gy1 - wy)
        d2 = math.hypot(gx2 - wx, gy2 - wy)

        if d1 > d2:
            # Point 1 is the handle end, point 2 is near the tip
            dx, dy = gx2 - gx1, gy2 - gy1
        else:
            dx, dy = gx1 - gx2, gy1 - gy2

        # Normalize: direction from cue handle TOWARD the white ball and beyond
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return None

        return (dx / length, dy / length)

    def _point_to_line_dist(self, px, py, x1, y1, x2, y2):
        """Perpendicular distance from point (px,py) to line segment."""
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-6:
            return math.hypot(px - x1, py - y1)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.hypot(px - proj_x, py - proj_y)

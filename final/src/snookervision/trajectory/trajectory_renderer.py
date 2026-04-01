"""Draw the predicted object ball path on the frame and send to LED strips."""
import cv2
import math


class TrajectoryRenderer:
    AIM_COLOR = (255, 255, 0)       # cyan — cue ball aim line
    OBJECT_PATH_COLOR = (0, 255, 255)  # yellow — where the object ball goes
    GHOST_COLOR = (200, 200, 200)    # gray — ghost ball

    def draw_on_frame(self, frame, prediction):
        """Draw aim line + object ball predicted path."""
        if not prediction:
            return

        # Aim line: cue ball → ghost ball (dashed cyan)
        if prediction["aim_line"]:
            (x1, y1), (x2, y2) = prediction["aim_line"]
            self._draw_dashed_line(frame,
                                    (int(x1), int(y1)), (int(x2), int(y2)),
                                    self.AIM_COLOR, thickness=2, dash_len=8)

        # Ghost ball circle
        if prediction["ghost_ball"]:
            gx, gy = int(prediction["ghost_ball"][0]), int(prediction["ghost_ball"][1])
            cv2.circle(frame, (gx, gy), 12, self.GHOST_COLOR, 1)

        # Object ball path: where it's going (solid yellow arrow)
        if prediction["object_path"]:
            (x1, y1), (x2, y2) = prediction["object_path"]
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            cv2.arrowedLine(frame, pt1, pt2,
                            self.OBJECT_PATH_COLOR, 2, tipLength=0.03)

        # Label
        if prediction["hit_ball"] and prediction["hit_point"]:
            hx, hy = int(prediction["hit_point"][0]), int(prediction["hit_point"][1])
            cv2.putText(frame, f"-> {prediction['hit_ball'].upper()}",
                        (hx + 15, hy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, self.OBJECT_PATH_COLOR, 1)

    def send_to_leds(self, led_controller, prediction, color=(0, 255, 255)):
        """Light up the object ball's predicted path on the LED strips."""
        if not led_controller or not prediction or not prediction["object_path"]:
            return

        led_controller.send_clear()
        r, g, b = color

        (x1, y1), (x2, y2) = prediction["object_path"]
        length = math.hypot(x2 - x1, y2 - y1)
        num_points = max(1, int(length / 25))

        for i in range(num_points + 1):
            t = i / max(1, num_points)
            px = int(x1 + (x2 - x1) * t)
            py = int(y1 + (y2 - y1) * t)
            led_controller.send_ball(px, py, r, g, b)

    def _draw_dashed_line(self, frame, pt1, pt2, color, thickness=2, dash_len=10):
        x1, y1 = pt1
        x2, y2 = pt2
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 1:
            return
        dx = (x2 - x1) / dist
        dy = (y2 - y1) / dist
        for i in range(0, int(dist / dash_len), 2):
            sx = int(x1 + dx * dash_len * i)
            sy = int(y1 + dy * dash_len * i)
            ex = int(x1 + dx * dash_len * min(i + 1, int(dist / dash_len)))
            ey = int(y1 + dy * dash_len * min(i + 1, int(dist / dash_len)))
            cv2.line(frame, (sx, sy), (ex, ey), color, thickness)

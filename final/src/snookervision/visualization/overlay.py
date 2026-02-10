import cv2
import numpy as np


class GeneratedTableRenderer:
    def __init__(self, table_size=(280, 560)):
        width = int(table_size[0])
        height = int(table_size[1])
        # Keep generated table in landscape orientation.
        if height > width:
            width, height = height, width
        self.width = width
        self.height = height

    def _draw_table_base(self):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:, :] = (0, 180, 10)

        line_color = (50, 255, 50)
        if self.width >= self.height:
            # Horizontal table: baulk line is vertical near the right end.
            baulk_x = int(self.width * 4 / 5)
            d_radius = int(self.height / 6)
            d_center = (baulk_x, int(self.height / 2))
            cv2.circle(img, d_center, d_radius, line_color, 1)
            img[:, baulk_x:self.width] = (0, 180, 10)
            cv2.line(img, (baulk_x, 0), (baulk_x, self.height), line_color, 1)
        else:
            # Fallback for portrait tables.
            baulk_y = int(self.height / 5)
            d_radius = int(self.width / 6)
            d_center = (int(self.width / 2), baulk_y)
            cv2.circle(img, d_center, d_radius, line_color, 1)
            img[baulk_y:self.height, 0:self.width] = (0, 180, 10)
            cv2.line(img, (0, baulk_y), (self.width, baulk_y), line_color, 1)

        border_color = (0, 140, 255)
        cv2.rectangle(img, (0, 0), (self.width - 1, self.height - 1), border_color, 2)

        outer = (190, 190, 190)
        inner = (120, 120, 120)
        if self.width >= self.height:
            # In landscape, long rails are top/bottom.
            pockets = [
                ((0, 0), 11, 9),
                ((self.width - 1, 0), 11, 9),
                ((0, self.height - 1), 11, 9),
                ((self.width - 1, self.height - 1), 11, 9),
                ((int(self.width / 2), 0), 8, 6),
                ((int(self.width / 2), self.height - 1), 8, 6),
            ]
        else:
            # In portrait, long rails are left/right.
            pockets = [
                ((0, 0), 11, 9),
                ((self.width - 1, 0), 11, 9),
                ((0, self.height - 1), 11, 9),
                ((self.width - 1, self.height - 1), 11, 9),
                ((0, int(self.height / 2)), 8, 6),
                ((self.width - 1, int(self.height / 2)), 8, 6),
            ]
        for center, r1, r2 in pockets:
            cv2.circle(img, center, r1, outer, -1)
            cv2.circle(img, center, r2, inner, -1)

        return img

    def render(self, frame_shape, ball_markers):
        canvas = self._draw_table_base()
        frame_h, frame_w = frame_shape[:2]
        if frame_h <= 0 or frame_w <= 0:
            return canvas

        for marker in ball_markers:
            src_x, src_y = marker["center"]
            color = marker["color"]

            x = int((src_x / frame_w) * (self.width - 1))
            y = int((src_y / frame_h) * (self.height - 1))
            x = int(np.clip(x, 0, self.width - 1))
            y = int(np.clip(y, 0, self.height - 1))

            cv2.circle(canvas, (x, y), 7, color, -1)
            cv2.circle(canvas, (x, y), 7, (0, 0, 0), 1)
            cv2.circle(canvas, (max(0, x - 2), max(0, y - 2)), 2, (255, 255, 255), -1)

        return canvas

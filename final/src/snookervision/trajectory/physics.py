"""Predict where the object ball goes after being hit by the cue ball.

Simple geometry: the object ball travels along the line connecting
the cue ball's contact point (ghost ball) to the object ball center.
"""
import math


class TrajectoryPredictor:
    def __init__(self, table_width, table_height):
        self.tw = table_width
        self.th = table_height

    def predict(self, cue_pos, aim_direction, balls, path_length=800):
        """Predict which object ball gets hit and where it goes.

        Args:
            cue_pos: (x, y) of the cue ball center
            aim_direction: (dx, dy) unit vector of the cue aim
            balls: list of {"x", "y", "color"} for all other balls
            path_length: how far to extend the object ball path (px)

        Returns:
            {
                "hit_ball": "red" or None,
                "hit_point": (x, y) — object ball center,
                "ghost_ball": (x, y) — where cue ball contacts,
                "object_path": [(x1,y1), (x2,y2)] — predicted object ball path,
                "aim_line": [(x1,y1), (x2,y2)] — cue ball aim line to contact,
            }
        """
        dx, dy = aim_direction
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return None
        dx, dy = dx / length, dy / length

        # Find which ball the cue ball will hit first
        hit_dist, hit_ball = self._find_target(cue_pos, dx, dy, balls)
        if hit_ball is None:
            return None

        cx, cy = cue_pos
        # Ghost ball position (where cue ball center is at moment of contact)
        gx = cx + dx * hit_dist
        gy = cy + dy * hit_dist

        bx, by = hit_ball["x"], hit_ball["y"]

        # Object ball direction: from ghost ball toward object ball center
        obj_dx = bx - gx
        obj_dy = by - gy
        obj_len = math.hypot(obj_dx, obj_dy)
        if obj_len < 1e-6:
            return None
        obj_dx, obj_dy = obj_dx / obj_len, obj_dy / obj_len

        # Extend the object ball path
        end_x = bx + obj_dx * path_length
        end_y = by + obj_dy * path_length

        # Clip to table bounds
        end_x = max(0, min(self.tw, end_x))
        end_y = max(0, min(self.th, end_y))

        return {
            "hit_ball": hit_ball["color"],
            "hit_point": (bx, by),
            "ghost_ball": (gx, gy),
            "object_path": [(bx, by), (end_x, end_y)],
            "aim_line": [(cx, cy), (gx, gy)],
        }

    def _find_target(self, cue_pos, dx, dy, balls):
        """Find the first ball the cue ball will hit along the aim direction.

        Uses ray-circle intersection (ghost ball method):
        the cue ball contacts when its center is 2*radius from the target center.
        We estimate ball radius from the detection bounding box or use a default.
        """
        cx, cy = cue_pos
        best_dist = float("inf")
        best_ball = None
        # Approximate ball radius — snooker ball on a 1200px table
        hit_radius = self.tw * 0.019 * 2  # 2 * ball_radius

        for ball in balls:
            bx, by = ball["x"], ball["y"]
            # Vector from cue to ball
            ex, ey = bx - cx, by - cy

            # Project onto aim direction
            proj = ex * dx + ey * dy
            if proj <= 0:
                continue  # Behind us

            # Perpendicular distance squared
            perp_sq = (ex * ex + ey * ey) - proj * proj
            if perp_sq > hit_radius * hit_radius:
                continue  # Miss

            # Distance to contact
            offset = math.sqrt(max(0, hit_radius * hit_radius - perp_sq))
            t = proj - offset

            if t > 0.1 and t < best_dist:
                best_dist = t
                best_ball = ball

        return best_dist, best_ball

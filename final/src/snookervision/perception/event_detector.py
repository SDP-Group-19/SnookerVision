import time
import math
import logging
from snookervision.game_logic.game_logic import Event, EventType, BallType

logger = logging.getLogger(__name__)


class EventDetector:
    def __init__(self):
        self.prev_balls = {}
        self.shot_active = False
        self.first_contact_done = False
        self.last_motion_time = 0.0
        self.motion_start_time = {}
        self.min_shot_gap = 0.4  # seconds

    def update(self, balls: dict):
        now = time.time()
        events = []

        # ---------- SHOT START (debounced) ----------
        if not self.shot_active and self._cue_moved(balls):
            if now - self.last_motion_time > self.min_shot_gap:
                events.append(Event(now, EventType.SHOT_START))
                self.shot_active = True
                self.first_contact_done = False
                self.motion_start_time = {}
                logger.info("[EVENT] SHOT_START")

        # ---------- FIRST CONTACT (earliest mover) ----------
        if self.shot_active and not self.first_contact_done:
            hit = self._detect_first_contact(balls)
            if hit:
                events.append(
                    Event(now, EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": hit})
                )
                self.first_contact_done = True
                logger.info(f"[EVENT] FIRST_CONTACT {hit.name}")

        # ---------- BALL POTTED (with pocket confirmation) ----------
        for b in self._detect_pots(balls):
            events.append(Event(now, EventType.BALL_POTTED, {"ball": b}))
            logger.info(f"[EVENT] BALL_POTTED {b.name}")

        # ---------- SHOT END ----------
        if self.shot_active and self._all_stopped(balls):
            events.append(Event(now, EventType.SHOT_END))
            self.shot_active = False
            logger.info("[EVENT] SHOT_END")

        # ---------- NO REDS ----------
        if "red" in self.prev_balls and not balls.get("red"):
            events.append(Event(now, EventType.NO_REDS_REMAINING))
            logger.info("[EVENT] NO_REDS_REMAINING")

        self.prev_balls = self._copy(balls)
        self.last_motion_time = now
        return events

    # ================= helpers =================

    def _cue_moved(self, balls):
        return self._ball_moved("white", balls, threshold=30)

    def _detect_first_contact(self, balls):
        earliest = None
        earliest_time = float("inf")

        for colour, btype in self._colour_map().items():
            if self._ball_moved(colour, balls, threshold=10):
                t = self.motion_start_time.get(colour)
                if t is None:
                    t = time.time()
                    self.motion_start_time[colour] = t
                if t < earliest_time:
                    earliest_time = t
                    earliest = btype

        return earliest

    def _detect_pots(self, balls):
        potted = []
        for colour in self.prev_balls:
            prev = self.prev_balls.get(colour, [])
            curr = balls.get(colour, [])
            if len(curr) < len(prev):
                if self._near_pocket(prev):
                    b = self._colour_map().get(colour)
                    if b:
                        potted.append(b)
        return potted

    def _near_pocket(self, prev_positions):
        for p in prev_positions:
            if (
                p["x"] < 40 or p["x"] > 960 or
                p["y"] < 40 or p["y"] > 520
            ):
                return True
        return False

    def _all_stopped(self, balls):
        for colour in balls:
            if self._ball_moved(colour, balls, threshold=2):
                return False
        return True

    def _ball_moved(self, colour, balls, threshold):
        if colour not in balls or colour not in self.prev_balls:
            return False
        if not balls[colour] or not self.prev_balls[colour]:
            return False

        a = balls[colour][0]
        b = self.prev_balls[colour][0]
        return math.hypot(a["x"] - b["x"], a["y"] - b["y"]) > threshold

    def _copy(self, balls):
        return {c: [dict(p) for p in v] for c, v in balls.items()}

    def _colour_map(self):
        return {
            "white": BallType.CUE,
            "red": BallType.RED,
            "yellow": BallType.YELLOW,
            "green": BallType.GREEN,
            "brown": BallType.BROWN,
            "blue": BallType.BLUE,
            "pink": BallType.PINK,
            "black": BallType.BLACK,
        }

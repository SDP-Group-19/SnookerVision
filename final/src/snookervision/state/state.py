import time
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


class StateManager():
    def __init__(self):
        self.previous_state = None
        self.time_since_last_update = None
        self.end_of_turn = False
        self.not_moved_counter = 0
        self.config = None
        self.state = None
        self.ball_tracks = {}
        self.next_track_id = 1
        self.pot_notifications = []
        self.pot_counter = 0
        self.tracking_tick = 0
        self.overlay_notifications = []
        self.pocket_names = [
            "top_left",
            "top_middle",
            "top_right",
            "bottom_left",
            "bottom_middle",
            "bottom_right",
        ]

    def initialize(self, config, state):
        """Initialize the StateManager with configuration and state objects"""
        self.config = config
        self.state = state
        self.time_since_last_update = time.time() - config.network_update_interval
        self.x_ratio = np.divide(self.config.output_dimensions[0], (
            self.config.output_dimensions[0] - (2 * self.config.gantry_effective_range_x_px[0])))
        self.y_ratio = np.divide(self.config.output_dimensions[1], (
            self.config.output_dimensions[1] - (2 * self.config.gantry_effective_range_y_px[0])))

    def update(self, detections, labels):
        if not self.config or not self.state:
            logger.error(
                "StateManager not initialized. Call initialize() first.")
            return

        if self.state.network and self.state.network.positions_requested:
            self.previous_state = None
            self.state.network.positions_requested = False

        current_time = time.time()
        self.tracking_tick += 1
        self._advance_overlay_notifications()

        if current_time - self.time_since_last_update < self.config.network_update_interval:
            return

        balls = {}
        corrected_white_ball = {}
        num_balls = 0
        self.not_moved_counter = 0

        if not detections or not detections[0].boxes:
            pot_notifications = self._update_tracks_and_detect_pots(balls, current_time)
            self._notify_pots(pot_notifications)
            return

        for ball in detections[0].boxes:
            classname, middlex, middley = self._get_ball_info(ball, labels)
            if classname in {"arm", "hole"}:
                continue

            num_balls += 1

            if classname == "white":
                corrected_middlex, corrected_middley = self._handle_offset(
                    middlex, middley, self.x_ratio, self.y_ratio)
                corrected_white_ball.update({
                    "x": corrected_middlex,
                    "y": corrected_middley})

            if self.previous_state and classname in self.previous_state:
                for prev_ball in self.previous_state[classname]:
                    if self._near_previous_position(prev_ball, middlex, middley):
                        self.not_moved_counter += 1
                        prev_ball["x"] = middlex
                        prev_ball["y"] = middley
                        break

            balls.setdefault(classname, []).append(
                {"x": middlex,
                 "y": middley})

        if self.not_moved_counter == num_balls:
            self.previous_state = balls

        pot_notifications = self._update_tracks_and_detect_pots(balls, current_time)
        self._notify_pots(pot_notifications)

        self._update_and_send_balls(balls, corrected_white_ball, current_time)

    def _get_ball_info(self, ball, labels):
        xyxy_tensor = ball.xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = map(int, xyxy.astype(int))
        classidx: int = int(ball.cls.item())
        classname: str = labels[classidx]
        _middlex: int = int((xmin + xmax) // 2)
        _middley: int = int((ymin + ymax) // 2)

        middlex, middley = self._coords_clamped(_middlex, _middley)
        return classname, middlex, middley

    def _distance(self, a, b):
        return math.hypot(a["x"] - b["x"], a["y"] - b["y"])

    def _create_track(self, colour, pos, now):
        track_id = self.next_track_id
        self.next_track_id += 1
        self.ball_tracks[track_id] = {
            "id": track_id,
            "colour": colour,
            "x": pos["x"],
            "y": pos["y"],
            "last_seen": now,
            "missing_since": None,
            "pocket_idx": None,
            "potted": False,
            "potted_at": None,
        }
        return track_id

    def _match_colour_tracks(self, tracks, observations, threshold):
        unmatched_track_ids = {track_id for track_id, _ in tracks}
        unmatched_obs_idxs = set(range(len(observations)))
        matched_pairs = []

        while unmatched_track_ids and unmatched_obs_idxs:
            best = None
            best_dist = float("inf")
            for track_id, track in tracks:
                if track_id not in unmatched_track_ids:
                    continue
                for obs_idx in unmatched_obs_idxs:
                    obs = observations[obs_idx]
                    d = self._distance(track, obs)
                    if d < best_dist:
                        best_dist = d
                        best = (track_id, obs_idx)

            if best is None or best_dist > threshold:
                break

            track_id, obs_idx = best
            matched_pairs.append((track_id, obs_idx))
            unmatched_track_ids.remove(track_id)
            unmatched_obs_idxs.remove(obs_idx)

        return matched_pairs, unmatched_track_ids, unmatched_obs_idxs

    def _pocket_centers(self):
        w, h = self.config.output_dimensions
        return [
            {"x": 0, "y": 0},
            {"x": int(w / 2), "y": 0},
            {"x": w, "y": 0},
            {"x": 0, "y": h},
            {"x": int(w / 2), "y": h},
            {"x": w, "y": h},
        ]

    def _nearest_pocket(self, pos):
        pockets = self._pocket_centers()
        best_idx = None
        best_dist = float("inf")
        for idx, pocket in enumerate(pockets):
            d = self._distance(pos, pocket)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx, best_dist

    def _update_tracks_and_detect_pots(self, balls, now):
        if not self.config.enable_pot_notifications:
            return []

        events = []
        match_threshold = max(8, int(self.config.pot_tracking_match_px))
        pocket_threshold = max(10, int(self.config.pot_pocket_radius_px))
        missing_seconds = max(0.2, float(self.config.pot_missing_seconds))
        stale_seconds = max(missing_seconds + 0.5, float(self.config.pot_track_stale_seconds))

        colours = set(
            [c for c in balls.keys() if c not in {"arm", "hole"}]
            + [t["colour"] for t in self.ball_tracks.values() if not t["potted"]]
        )

        for colour in colours:
            observations = [dict(p) for p in balls.get(colour, [])]
            tracks = [
                (track_id, track)
                for track_id, track in self.ball_tracks.items()
                if (not track["potted"]) and track["colour"] == colour
            ]

            matched_pairs, unmatched_track_ids, unmatched_obs_idxs = self._match_colour_tracks(
                tracks, observations, match_threshold
            )

            for track_id, obs_idx in matched_pairs:
                track = self.ball_tracks[track_id]
                obs = observations[obs_idx]
                track["x"] = obs["x"]
                track["y"] = obs["y"]
                track["last_seen"] = now
                track["missing_since"] = None
                track["pocket_idx"] = None

            for track_id in unmatched_track_ids:
                track = self.ball_tracks[track_id]
                if track["missing_since"] is None:
                    track["missing_since"] = now
                    pocket_idx, pocket_dist = self._nearest_pocket(track)
                    if pocket_idx is not None and pocket_dist <= pocket_threshold:
                        track["pocket_idx"] = pocket_idx
                    else:
                        track["pocket_idx"] = None

            for obs_idx in unmatched_obs_idxs:
                self._create_track(colour, observations[obs_idx], now)

        stale_track_ids = []
        for track_id, track in self.ball_tracks.items():
            age = now - track["last_seen"]
            if track["potted"]:
                if age > stale_seconds:
                    stale_track_ids.append(track_id)
                continue

            if track["missing_since"] is not None and track["pocket_idx"] is not None:
                missing_duration = now - track["missing_since"]
                if missing_duration >= missing_seconds:
                    self.pot_counter += 1
                    events.append({
                        "order": self.pot_counter,
                        "track_id": track["id"],
                        "colour": track["colour"],
                        "pocket": self.pocket_names[track["pocket_idx"]],
                        "missing_seconds": missing_duration,
                    })
                    track["potted"] = True
                    track["potted_at"] = now
                    continue

            if age > stale_seconds:
                stale_track_ids.append(track_id)

        for track_id in stale_track_ids:
            self.ball_tracks.pop(track_id, None)

        events.sort(key=lambda e: e["order"])
        return events

    def _notify_pots(self, pot_notifications):
        if not pot_notifications:
            return

        for n in pot_notifications:
            logger.info(
                f"[POT] #{n['order']} {n['colour'].upper()} "
                f"(track {n['track_id']}) potted at {n['pocket']} "
                f"after missing {n['missing_seconds']:.2f}s"
            )

        if len(pot_notifications) > 1:
            ordered = " -> ".join([n["colour"] for n in pot_notifications])
            logger.info(f"[POT] Sequence: {ordered}")

        self.pot_notifications.extend(pot_notifications)
        ttl = max(1, int(self.config.pot_overlay_ttl_frames))
        for n in pot_notifications:
            self.overlay_notifications.append(
                {
                    "text": (
                        f"Pot #{n['order']}: {n['colour'].upper()} "
                        f"T{n['track_id']} ({n['pocket']})"
                    ),
                    "ttl": ttl,
                }
            )
        if len(pot_notifications) > 1:
            ordered = " -> ".join([n["colour"].upper() for n in pot_notifications])
            self.overlay_notifications.append(
                {
                    "text": f"Sequence: {ordered}",
                    "ttl": ttl,
                }
            )

    def _advance_overlay_notifications(self):
        updated = []
        for item in self.overlay_notifications:
            remaining = int(item["ttl"]) - 1
            if remaining > 0:
                updated.append({"text": item["text"], "ttl": remaining})
        self.overlay_notifications = updated

    def get_overlay_lines(self, max_lines=4):
        lines = [item["text"] for item in self.overlay_notifications]
        return lines[-max_lines:]

    def _coords_clamped(self, x, y):
        x = max(0, min(x, self.config.output_dimensions[0]))
        y = max(0, min(y, self.config.output_dimensions[1]))
        return int(x), int(y)

    def _near_previous_position(self, prev_ball, x, y):
        return abs(prev_ball["x"] - x) < self.config.position_threshold \
            and abs(prev_ball["y"] - y) < self.config.position_threshold

    def _handle_end_of_turn(self):
        if self.end_of_turn:
            return
        self.end_of_turn = True
        if self.config.use_networking and self.state.network:
            self.state.network.send_end_of_turn("true")
        else:
            logger.info("No movement detected, end of turn.")

    def _update_and_send_balls(self, balls, corrected_white_ball, current_time):
        if not balls:
            return

        self.previous_state = balls
        self.time_since_last_update = current_time
        self.end_of_turn = False

        if self.config.use_networking and self.state.network:
            self.state.network.send_balls({"balls": balls})
            if corrected_white_ball:
                self.state.network.send_corrected_white_ball(
                    corrected_white_ball)
        else:
            logger.info(f"Sending balls: {balls}")
            if corrected_white_ball:
                logger.info(
                    f"Sending corrected white ball: {corrected_white_ball}")

    def _handle_offset(self, middlex, middley, x_ratio, y_ratio):
        corrected_middlex = self._handle_x_offset(middlex, x_ratio)
        corrected_middley = self._handle_y_offset(middley, y_ratio)
        corrected_middlex, corrected_middley = self._coords_clamped(
            corrected_middlex, corrected_middley)
        return corrected_middlex, corrected_middley

    def _handle_x_offset(self, middlex, x_ratio):
        if middlex > self.config.gantry_effective_range_x_px[1]:
            return self.config.output_dimensions[0]
        elif middlex < self.config.gantry_effective_range_x_px[0]:
            return 0
        return (middlex - self.config.gantry_effective_range_x_px[0]) * x_ratio

    def _handle_y_offset(self, middley, y_ratio):
        if middley > self.config.gantry_effective_range_y_px[1]:
            return self.config.output_dimensions[1]
        elif middley < self.config.gantry_effective_range_y_px[0]:
            return 0
        return (middley - self.config.gantry_effective_range_y_px[0]) * y_ratio

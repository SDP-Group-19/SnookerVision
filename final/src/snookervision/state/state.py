import time
import logging
import math
import numpy as np
from snookervision.game_logic.game_logic import (
    BallType,
    Event,
    EventType,
    GameState,
    RuleEngine,
)

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
        self.recent_non_red_pots = []
        self.pot_counter = 0
        self.tracking_tick = 0
        self.overlay_notifications = []
        self.shot_active = False
        self.shot_last_motion_time = None
        self.shot_stopped_at = None
        self.first_hit_colour = None
        self.second_hit_colour = None
        self.live_overlay_lines = []
        self.last_potted_text = "-"
        self.last_foul_text = "-"
        self.game_state = GameState()
        self.game_state.start_frame()
        self.rule_engine = RuleEngine(self.game_state)
        self.last_shot_active = False
        self.game_shot_open = False
        self.first_contact_sent_this_shot = False
        self.no_reds_announced = False
        self.red_potted_ever = False
        self.zero_red_since = None
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
        self._rebuild_overlay_lines()

    def update(self, detections, labels=None):
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

        if not detections:
            self._update_hit_order({}, current_time)
            pot_notifications = self._update_tracks_and_detect_pots(balls, current_time)
            self._feed_game_logic(balls, pot_notifications, current_time)
            self._notify_pots(pot_notifications)
            self._rebuild_overlay_lines()
            return

        for ball in detections:
            classname, middlex, middley = self._get_ball_info(ball)
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

        self._update_hit_order(balls, current_time)
        pot_notifications = self._update_tracks_and_detect_pots(balls, current_time)
        self._feed_game_logic(balls, pot_notifications, current_time)
        self._notify_pots(pot_notifications)
        self._rebuild_overlay_lines()

        self._update_and_send_balls(balls, corrected_white_ball, current_time)

    def _get_ball_info(self, ball):
        classname = ball["label"]
        _middlex, _middley = ball["center"]

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
        non_red_cooldown = max(
            0.2, float(getattr(self.config, "non_red_pot_cooldown_seconds", 3.0))
        )
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
                    pocket_name = self.pocket_names[track["pocket_idx"]]
                    allow_emit = True
                    for recent in self.recent_non_red_pots:
                        if recent["colour"] != track["colour"]:
                            continue
                        if recent["pocket"] != pocket_name:
                            continue
                        if now - recent["time"] <= non_red_cooldown:
                            allow_emit = False
                            break

                    if allow_emit:
                        self.pot_counter += 1
                        events.append({
                            "order": self.pot_counter,
                            "track_id": track["id"],
                            "colour": track["colour"],
                            "pocket": pocket_name,
                            "missing_seconds": missing_duration,
                        })
                        self.recent_non_red_pots.append(
                            {"colour": track["colour"], "pocket": pocket_name, "time": now}
                        )
                    track["potted"] = True
                    track["potted_at"] = now
                    continue

            if age > stale_seconds:
                stale_track_ids.append(track_id)

        for track_id in stale_track_ids:
            self.ball_tracks.pop(track_id, None)

        self.recent_non_red_pots = [
            p for p in self.recent_non_red_pots if now - p["time"] <= non_red_cooldown
        ]

        events.sort(key=lambda e: e["order"])
        return events

    def _notify_pots(self, pot_notifications):
        if not pot_notifications:
            return

        self.pot_notifications.extend(pot_notifications)
        ttl = max(1, int(self.config.pot_overlay_ttl_frames))
        for n in pot_notifications:
            logger.info(
                f"[POT] #{n['order']} {n['colour'].upper()} "
                f"(track {n['track_id']}) potted at {n['pocket']} "
                f"after missing {n['missing_seconds']:.2f}s"
            )
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
            self.last_potted_text = ordered
            self.overlay_notifications.append(
                {
                    "text": f"Sequence: {ordered}",
                    "ttl": ttl,
                }
            )
        else:
            n = pot_notifications[-1]
            self.last_potted_text = f"{n['colour'].upper()} ({n['pocket']})"

    def _colour_to_ball_type(self, colour):
        mapping = {
            "white": BallType.CUE,
            "red": BallType.RED,
            "yellow": BallType.YELLOW,
            "green": BallType.GREEN,
            "brown": BallType.BROWN,
            "blue": BallType.BLUE,
            "pink": BallType.PINK,
            "black": BallType.BLACK,
        }
        return mapping.get((colour or "").lower())

    def _push_game_outputs(self, outputs):
        if not outputs:
            return
        ttl = max(1, int(self.config.pot_overlay_ttl_frames))
        for msg in outputs:
            logger.info(f"[GAME] {msg}")
            if msg.startswith("FOUL"):
                self.last_foul_text = msg
            self.overlay_notifications.append(
                {"text": f"Game: {msg}", "ttl": ttl}
            )

    def _feed_game_logic(self, balls, pot_notifications, now):
        if any((n.get("colour") or "").lower() == "red" for n in pot_notifications):
            self.red_potted_ever = True

        # Shot lifecycle for rules: keep shot open until pot confirmation window has passed.
        shot_end_grace = max(0.2, float(getattr(self.config, "pot_missing_seconds", 2.0)))

        if self.shot_active and not self.game_shot_open:
            outputs = self.rule_engine.on_event(Event(now, EventType.SHOT_START))
            self._push_game_outputs(outputs)
            self.game_shot_open = True
            self.first_contact_sent_this_shot = False

        can_end_shot = (
            self.game_shot_open
            and (not self.shot_active)
            and self.shot_stopped_at is not None
            and (now - self.shot_stopped_at) >= shot_end_grace
            and (not self._has_pending_pot_confirmations(now))
        )
        if can_end_shot:
            outputs = self.rule_engine.on_event(Event(now, EventType.SHOT_END))
            self._push_game_outputs(outputs)
            self.game_shot_open = False
            self.first_contact_sent_this_shot = False

        # First contact: cue -> first non-white hit colour.
        if self.game_shot_open and self.shot_active and not self.first_contact_sent_this_shot:
            hit_colour = None
            if self.first_hit_colour and self.first_hit_colour.lower() != "white":
                hit_colour = self.first_hit_colour
            elif self.second_hit_colour:
                hit_colour = self.second_hit_colour

            hit_ball = self._colour_to_ball_type(hit_colour)
            if hit_ball is not None and hit_ball != BallType.CUE:
                outputs = self.rule_engine.on_event(
                    Event(
                        now,
                        EventType.FIRST_CONTACT,
                        {"a": BallType.CUE, "b": hit_ball},
                    )
                )
                self._push_game_outputs(outputs)
                self.first_contact_sent_this_shot = True

        # Pot events from already-confirmed live pot notifications (frame delta only).
        if self.game_shot_open:
            for n in pot_notifications:
                ball = self._colour_to_ball_type(n.get("colour"))
                if ball is None:
                    continue
                outputs = self.rule_engine.on_event(
                    Event(now, EventType.BALL_POTTED, {"ball": ball})
                )
                self._push_game_outputs(outputs)

        # Optional phase event when all reds gone.
        no_reds_confirm_seconds = max(
            0.2, float(getattr(self.config, "no_reds_confirm_seconds", 1.0))
        )
        red_count = len(balls.get("red", []))
        if red_count > 0:
            self.no_reds_announced = False
            self.zero_red_since = None
        elif not self.shot_active and self.red_potted_ever:
            if self.zero_red_since is None:
                self.zero_red_since = now
            stable_zero_reds = (now - self.zero_red_since) >= no_reds_confirm_seconds
            if stable_zero_reds and not self.no_reds_announced:
                outputs = self.rule_engine.on_event(Event(now, EventType.NO_REDS_REMAINING))
                self._push_game_outputs(outputs)
                self.no_reds_announced = True

        self.last_shot_active = self.shot_active

    def _colour_is_moving(self, colour, balls, threshold):
        if not self.previous_state:
            return False
        curr_positions = balls.get(colour, [])
        prev_positions = self.previous_state.get(colour, [])
        if not curr_positions or not prev_positions:
            return False

        for curr in curr_positions:
            nearest = min(
                math.hypot(curr["x"] - prev["x"], curr["y"] - prev["y"])
                for prev in prev_positions
            )
            if nearest > threshold:
                return True
        return False

    def _update_hit_order(self, balls, now):
        move_threshold = max(2, int(getattr(self.config, "hit_motion_threshold_px", 10)))
        reset_seconds = max(0.2, float(getattr(self.config, "hit_stationary_reset_seconds", 1.0)))

        moving_colours = []
        for colour in balls.keys():
            if colour in {"arm", "hole"}:
                continue
            if self._colour_is_moving(colour, balls, move_threshold):
                moving_colours.append(colour)

        white_moving = "white" in moving_colours
        any_moving = len(moving_colours) > 0

        if not self.shot_active and any_moving:
            self.shot_active = True
            self.shot_last_motion_time = now
            self.shot_stopped_at = None
            self.first_hit_colour = None
            self.second_hit_colour = None

        if self.shot_active:
            if any_moving:
                self.shot_last_motion_time = now

            for colour in moving_colours:
                if self.first_hit_colour is None:
                    self.first_hit_colour = colour
                    continue
                if self.second_hit_colour is None and colour != self.first_hit_colour:
                    self.second_hit_colour = colour

            if self.shot_last_motion_time is not None and (now - self.shot_last_motion_time) >= reset_seconds:
                self.shot_active = False
                self.shot_stopped_at = now
                self.shot_last_motion_time = None

    def _has_pending_pot_confirmations(self, now):
        missing_seconds = max(0.2, float(self.config.pot_missing_seconds))
        for track in self.ball_tracks.values():
            if track.get("potted"):
                continue
            if track.get("missing_since") is None:
                continue
            if track.get("pocket_idx") is None:
                continue
            if (now - track["missing_since"]) < missing_seconds:
                return True
        return False

    def _rebuild_overlay_lines(self):
        first_txt = self.first_hit_colour.upper() if self.first_hit_colour else "-"
        second_txt = self.second_hit_colour.upper() if self.second_hit_colour else "-"
        player1 = self.game_state.player1
        player2 = self.game_state.player2
        frame = self.game_state.current_frame
        turn_name = frame.activePlayer.name if frame is not None else "-"
        target_name = frame.activePlayer.target if frame is not None else "-"

        self.live_overlay_lines = [
            "[SHOT]",
            f"First hit: {first_txt}",
            f"Second hit: {second_txt}",
            f"Ball potted: {self.last_potted_text}",
            "[FOUL]",
            f"Last foul: {self.last_foul_text}",
            "[POINTS]",
            f"{player1.name}: {player1.score}",
            f"{player2.name}: {player2.score}",
            f"Turn: {turn_name}",
            f"Target: {target_name}",
        ]

    def _advance_overlay_notifications(self):
        updated = []
        for item in self.overlay_notifications:
            remaining = int(item["ttl"]) - 1
            if remaining > 0:
                updated.append({"text": item["text"], "ttl": remaining})
        self.overlay_notifications = updated

    def get_overlay_lines(self, max_lines=14):
        lines = [item["text"] for item in self.overlay_notifications]
        if len(self.live_overlay_lines) >= max_lines:
            return self.live_overlay_lines[:max_lines]

        remaining = max_lines - len(self.live_overlay_lines)
        return self.live_overlay_lines + lines[-remaining:]

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

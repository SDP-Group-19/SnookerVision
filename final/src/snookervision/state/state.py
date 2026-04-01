import time
import logging
import math
import numpy as np
from snookervision.arduino.host.display_bridge import ArduinoDisplayBridge
from snookervision.foul_logic import build_reposition_queue, snapshot_positions
from snookervision.state.pocket_sensor import PocketSensor
from snookervision.game_logic.game_logic import (
    BallType,
    Event,
    EventType,
    GameState,
    RuleEngine,
)
from snookervision.state.pocket_sensor import PocketSensor

logger = logging.getLogger(__name__)

# BallType → detection colour name
_BALLTYPE_TO_NAME = {
    BallType.CUE: "white",
    BallType.RED: "red",
    BallType.YELLOW: "yellow",
    BallType.GREEN: "green",
    BallType.BROWN: "brown",
    BallType.BLUE: "blue",
    BallType.PINK: "pink",
    BallType.BLACK: "black",
}

# LED colours per ball (RGB for the LED strip)
_BALL_LED_COLORS = {
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "yellow": (255, 255, 0),
    "green": (0, 255, 0),
    "brown": (139, 69, 19),
    "blue": (0, 0, 255),
    "pink": (255, 105, 180),
    "black": (100, 100, 100),
}


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
        self.first_object_hit_colour = None
        self.second_object_hit_colour = None
        self.shot_moved_colours = set()  # all colours that moved during current shot
        self.live_overlay_lines = []
        self.last_potted_text = "-"
        self.last_ball_hit_text = "-"
        self.last_foul_text = "-"
        self.show_foul_until = 0.0
        self.game_state = GameState()
        self.game_state.start_frame()
        self.rule_engine = RuleEngine(self.game_state)
        self.last_shot_active = False
        self.game_shot_open = False
        self.first_contact_sent_this_shot = False
        self.no_reds_announced = False
        self.red_potted_ever = False
        self.zero_red_since = None
        self.arduino = None
        self.arduino_state = None
        self.pocket_sensor = None
        self.pending_pocket_triggers = []
        self.current_balls_snapshot = {}
        self.pocket_names = [
            "top_left",
            "top_middle",
            "top_right",
            "bottom_left",
            "bottom_middle",
            "bottom_right",
        ]
        # LED reposition queue
        self.led_controller = None
        self.shot_start_positions = {}
        self.reposition_queue = []
        self.reposition_index = 0
        self.reposition_confirm_count = 0
        self.foul_reposition_active = False
        self.reposition_hold_active = False
        self.reposition_hold_start = None
        self.foul_flash_active = False
        self.foul_flash_start = None
        self.foul_potted_colors = set()
        # Pocket sensor (hardware trigger confirmation)
        self.pocket_sensor = None
        self.trigger_pot_events = []  # pot events confirmed by hardware trigger

    def initialize(self, config, state, arduino_port=None, arduino_baud=115200):
        """Initialize the StateManager with configuration and state objects"""
        self.config = config
        self.state = state
        self.time_since_last_update = time.time() - config.network_update_interval
        self.x_ratio = np.divide(self.config.output_dimensions[0], (
            self.config.output_dimensions[0] - (2 * self.config.gantry_effective_range_x_px[0])))
        self.y_ratio = np.divide(self.config.output_dimensions[1], (
            self.config.output_dimensions[1] - (2 * self.config.gantry_effective_range_y_px[0])))
        if arduino_port:
            self.arduino = ArduinoDisplayBridge(arduino_port, arduino_baud)
            self.arduino.connect()
        if config.led_enabled:
            from snookervision.led import LEDController
            from snookervision.led.led_controller import compute_table_dimensions
            self.led_controller = LEDController(
                config.mqtt_broker, config.mqtt_port,
                config.mqtt_username, config.mqtt_password,
            )
            if self.led_controller.connect():
                cv_w, cv_h = config.output_dimensions
                # Compute LED table dimensions from table_pts if available
                if config.pocket_pts is not None:
                    from snookervision.processing.camera_processing import load_table_pts
                    table_pts = load_table_pts()
                    if table_pts is not None:
                        led_w, led_h = compute_table_dimensions(table_pts.tolist())
                    else:
                        led_w, led_h = cv_w, cv_h
                else:
                    led_w, led_h = cv_w, cv_h
                self.led_controller.send_resize(led_w, led_h, cv_w, cv_h)
        if getattr(config, "pocket_sensor_enabled", False):
            self.pocket_sensor = PocketSensor(config)
            if self.pocket_sensor.connect():
                logger.info("Pocket sensor connected — pots require hardware trigger confirmation")
            else:
                logger.warning("Pocket sensor connection failed — falling back to CV-only pot detection")
                self.pocket_sensor = None
        self._rebuild_overlay_lines()
        self._sync_arduino_display(force=True)

    def _drain_pocket_triggers(self, now):
        """Drain MQTT events. Trigger fired = ball potted. Find which hit ball disappeared."""
        if self.pocket_sensor is None:
            return
        for event in self.pocket_sensor.drain_events():
            if event["state"] != "active":
                continue
            pocket_idx = event["pocket_idx"]
            pocket_name = self.pocket_names[pocket_idx] if pocket_idx < len(self.pocket_names) else str(pocket_idx)
            logger.info(f"[TRIGGER] Pocket {pocket_idx + 1} ({pocket_name}) sensor fired")

            pocket_center = self._pocket_centers()[pocket_idx]

            # Priority 1: ball that was HIT (moved this shot) and is now MISSING
            best_hit_missing_id = None
            best_hit_missing_dist = float("inf")
            # Priority 2: any ball that is MISSING near this pocket
            best_missing_id = None
            best_missing_dist = float("inf")
            # Priority 3: any non-potted track nearest to pocket (last resort)
            best_any_id = None
            best_any_dist = float("inf")

            for track_id, track in self.ball_tracks.items():
                if track["potted"] or track["colour"] == "white":
                    continue
                d = self._distance(track, pocket_center)
                is_missing = track["missing_since"] is not None
                was_hit = track["colour"] in self.shot_moved_colours

                if was_hit and is_missing and d < best_hit_missing_dist:
                    best_hit_missing_dist = d
                    best_hit_missing_id = track_id

                if is_missing and d < best_missing_dist:
                    best_missing_dist = d
                    best_missing_id = track_id

                if d < best_any_dist:
                    best_any_dist = d
                    best_any_id = track_id

            # Pick best candidate by priority
            if best_hit_missing_id is not None:
                chosen_id, chosen_dist, source = best_hit_missing_id, best_hit_missing_dist, "hit+missing"
            elif best_missing_id is not None:
                chosen_id, chosen_dist, source = best_missing_id, best_missing_dist, "missing"
            elif best_any_id is not None:
                chosen_id, chosen_dist, source = best_any_id, best_any_dist, "closest"
            else:
                chosen_id = None

            if chosen_id is not None:
                track = self.ball_tracks[chosen_id]
                colour = track["colour"]

                # Cooldown check
                allow_emit = True
                non_red_cooldown = max(
                    0.2, float(getattr(self.config, "non_red_pot_cooldown_seconds", 3.0))
                )
                for recent in self.recent_non_red_pots:
                    if recent["colour"] == colour and recent["pocket"] == pocket_name:
                        if now - recent["time"] <= non_red_cooldown:
                            allow_emit = False
                            break

                if allow_emit:
                    self.pot_counter += 1
                    self.trigger_pot_events.append({
                        "order": self.pot_counter,
                        "track_id": track["id"],
                        "colour": colour,
                        "pocket": pocket_name,
                        "missing_seconds": 0.0,
                    })
                    self.recent_non_red_pots.append(
                        {"colour": colour, "pocket": pocket_name, "time": now}
                    )
                track["potted"] = True
                track["potted_at"] = now
                logger.info(
                    f"[POT] {colour.upper()} potted at {pocket_name} "
                    f"(trigger, {source}, dist={chosen_dist:.0f}px)"
                )
            else:
                logger.warning(f"[TRIGGER] Pocket {pocket_idx + 1} ({pocket_name}) fired but no ball track found")

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
        self._drain_pocket_triggers(current_time)
        self._advance_overlay_notifications()

        if current_time - self.time_since_last_update < self.config.network_update_interval:
            return

        balls = {}
        corrected_white_ball = {}
        num_balls = 0
        self.not_moved_counter = 0

        if not detections:
            self.current_balls_snapshot = {}
            self._update_hit_order({}, current_time)
            pot_notifications = self._update_tracks_and_detect_pots(balls, current_time)
            pot_notifications.extend(self.trigger_pot_events)
            self.trigger_pot_events.clear()
            self._feed_game_logic(balls, pot_notifications, current_time)
            self._notify_pots(pot_notifications)
            self._poll_display_events()
            self._rebuild_overlay_lines()
            self._sync_arduino_display()
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
        pot_notifications.extend(self.trigger_pot_events)
        self.trigger_pot_events.clear()
        self._feed_game_logic(balls, pot_notifications, current_time)
        self._notify_pots(pot_notifications)
        self.current_balls_snapshot = snapshot_positions(balls)
        if self.foul_flash_active:
            self._check_foul_flash(current_time, balls)
        if self.foul_reposition_active:
            self._check_reposition_progress(balls, current_time)
        self._poll_display_events()
        self._rebuild_overlay_lines()
        self._sync_arduino_display()

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
        if self.config.pocket_pts is not None:
            return [{"x": int(p[0]), "y": int(p[1])} for p in self.config.pocket_pts]
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

        cv_candidates = []
        events = []
        match_threshold = max(8, int(self.config.pot_tracking_match_px))
        pocket_threshold = max(10, int(self.config.pot_pocket_radius_px))
        pocket_filter = max(10, int(self.config.pocket_filter_radius_px))
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
                obs = observations[obs_idx]
                _, pocket_dist = self._nearest_pocket(obs)
                if pocket_dist <= pocket_filter:
                    continue
                self._create_track(colour, obs, now)

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
                    # When pocket sensor is active, pots are only confirmed by trigger
                    # (handled in _drain_pocket_triggers). CV-only fallback when no sensor.
                    if self.pocket_sensor is not None:
                        continue

                    pocket_idx = track["pocket_idx"]
                    pocket_name = self.pocket_names[pocket_idx]
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
                        cv_candidates.append({
                            "track_id": track["id"],
                            "colour": track["colour"],
                            "pocket_idx": track["pocket_idx"],
                            "pocket": pocket_name,
                            "missing_seconds": missing_duration,
                        })
                    else:
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

        events.extend(self._resolve_pot_candidates(cv_candidates, now))
        events.sort(key=lambda e: e["order"])
        return events

    def _poll_pocket_sensor_events(self, now):
        if self.pocket_sensor is None:
            return

        match_window = max(0.2, float(getattr(self.config, "pocket_sensor_match_seconds", 2.5)))
        for event in self.pocket_sensor.drain_events():
            if event.get("state") != "active":
                continue
            self.pending_pocket_triggers.append(event)

        self.pending_pocket_triggers = [
            event for event in self.pending_pocket_triggers
            if now - float(event.get("time", now)) <= match_window
        ]

    def _resolve_pot_candidates(self, cv_candidates, now):
        if self.pocket_sensor is None:
            return [
                self._finalize_cv_pot_candidate(candidate, candidate["pocket_idx"], now)
                for candidate in cv_candidates
            ]

        self._poll_pocket_sensor_events(now)
        sensor_triggers = self.pending_pocket_triggers
        self.pending_pocket_triggers = []

        if cv_candidates and not sensor_triggers:
            for candidate in cv_candidates:
                self._reject_cv_pot_candidate(candidate["track_id"])
            return []

        events = []
        used_sensor_count = 0
        for candidate in cv_candidates:
            if used_sensor_count >= len(sensor_triggers):
                self._reject_cv_pot_candidate(candidate["track_id"])
                continue

            trigger = sensor_triggers[used_sensor_count]
            used_sensor_count += 1
            events.append(self._finalize_cv_pot_candidate(candidate, trigger["pocket_idx"], now))

        for trigger in sensor_triggers[used_sensor_count:]:
            fallback_event = self._create_sensor_only_pot_event(trigger["pocket_idx"], now)
            if fallback_event is not None:
                events.append(fallback_event)

        return events

    def _reject_cv_pot_candidate(self, track_id):
        track = self.ball_tracks.get(track_id)
        if track is None:
            return
        track["missing_since"] = None
        track["pocket_idx"] = None

    def _finalize_cv_pot_candidate(self, candidate, pocket_idx, now):
        track = self.ball_tracks.get(candidate["track_id"])
        if track is not None:
            track["potted"] = True
            track["potted_at"] = now

        pocket_name = self.pocket_names[pocket_idx]
        self.pot_counter += 1
        event = {
            "order": self.pot_counter,
            "track_id": candidate["track_id"],
            "colour": candidate["colour"],
            "pocket": pocket_name,
            "missing_seconds": candidate["missing_seconds"],
            "source": "cv+sensor",
        }
        self.recent_non_red_pots.append(
            {"colour": candidate["colour"], "pocket": pocket_name, "time": now}
        )
        return event

    def _create_sensor_only_pot_event(self, pocket_idx, now):
        colour = (self.first_object_hit_colour or "").lower()
        if not colour or colour in {"white", "arm", "hole"}:
            logger.info("[POCKET] Sensor trigger ignored: no usable last-hit ball")
            return None

        track_id = self._claim_missing_track_for_sensor_pot(colour, pocket_idx, now)
        if track_id is None:
            track_id = 0

        pocket_name = self.pocket_names[pocket_idx]
        self.pot_counter += 1
        event = {
            "order": self.pot_counter,
            "track_id": track_id,
            "colour": colour,
            "pocket": pocket_name,
            "missing_seconds": 0.0,
            "source": "sensor_only",
        }
        self.recent_non_red_pots.append(
            {"colour": colour, "pocket": pocket_name, "time": now}
        )
        return event

    def _claim_missing_track_for_sensor_pot(self, colour, pocket_idx, now):
        candidates = []
        for track_id, track in self.ball_tracks.items():
            if track.get("potted"):
                continue
            if track.get("colour") != colour:
                continue
            if track.get("missing_since") is None:
                continue
            score = 0 if track.get("pocket_idx") == pocket_idx else 1
            candidates.append((score, track.get("missing_since"), track_id))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1]))
        track_id = candidates[0][2]
        track = self.ball_tracks.get(track_id)
        if track is not None:
            track["potted"] = True
            track["potted_at"] = now
        return track_id

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
                self.show_foul_until = time.time() + 2.5
            if msg.startswith("FIRST_CONTACT"):
                self.last_ball_hit_text = msg.replace("FIRST_CONTACT", "").strip()
            self.overlay_notifications.append(
                {"text": f"Game: {msg}", "ttl": ttl}
            )
        self._sync_arduino_display(force=True, latest_outputs=outputs)

    # ---- LED foul flash + reposition queue ------------------------------------

    def _start_foul_flash(self, now):
        """Send the FOUL command to the LED strip and start the flash timer.

        Also records which colour balls were potted so we can wait for CV
        to confirm they are off the table before starting reposition.
        """
        if self.led_controller:
            logger.info("[LED] Sending FOUL command to ESP32")
            self.led_controller.send_foul()
        else:
            logger.warning("[LED] No led_controller — cannot send FOUL (led_enabled=%s)",
                           getattr(self.config, "led_enabled", None))
        # Record the potted ball colours that need to be confirmed off-table
        fs = self.game_state.current_frame
        potted_names = set()
        if fs is not None:
            for ball_type in fs.ctx.potted:
                name = _BALLTYPE_TO_NAME.get(ball_type)
                if name:
                    potted_names.add(name)
            if fs.ctx.cue_potted:
                potted_names.add("white")
        self.foul_potted_colors = potted_names
        self.foul_flash_active = True
        self.foul_flash_start = now
        logger.info("[LED] FOUL flash started — waiting for %s to be off table",
                    ", ".join(c.upper() for c in potted_names) or "none")

    def _check_foul_flash(self, now, balls):
        """Keep FOUL lit until CV confirms potted balls are off the table, then reposition."""
        # Minimum display time so the foul is visible even if balls vanish instantly
        min_duration = float(getattr(self.config, "led_foul_flash_seconds", 3.0))
        if now - self.foul_flash_start < min_duration:
            return

        # Check CV: each potted colour must NOT be detected on the table
        pocket_threshold = max(10, int(self.config.pocket_filter_radius_px))
        for color in self.foul_potted_colors:
            detected = balls.get(color, [])
            if color == "red":
                # Reds: we only care that the count decreased, but the simplest
                # check is that at least one red is gone.  Since we can't know
                # the exact count that was potted here, just skip reds — they
                # aren't re-spotted anyway.
                continue
            # Ignore detections near pocket positions (likely false positives)
            real_detections = [
                d for d in detected
                if self._nearest_pocket(d)[1] > pocket_threshold
            ]
            if real_detections:
                # Ball still visible on table — keep FOUL lit
                return

        # All potted balls confirmed off table
        if self.led_controller:
            self.led_controller.send_clear()
        self.foul_flash_active = False
        self.foul_flash_start = None
        self.foul_potted_colors = set()
        logger.info("[LED] FOUL flash ended — potted balls confirmed off table, starting reposition")
        self._build_reposition_queue()

    def _build_reposition_queue(self):
        """Build ordered list of balls to reposition after a foul.

        Uses the shot-start snapshot to know where each ball was before
        the foul stroke displaced them.
        """
        fs = self.game_state.current_frame
        if fs is None:
            return
        ctx = fs.ctx
        queue = []

        # Colour balls that were potted illegally (re-spotted to their old position)
        for ball_type in ctx.potted:
            name = _BALLTYPE_TO_NAME.get(ball_type)
            if name and name != "white":
                positions = self.shot_start_positions.get(name, [])
                if positions:
                    pos = positions[0]
                    queue.append({
                        "color_name": name,
                        "x": int(pos["x"]),
                        "y": int(pos["y"]),
                        "led_color": _BALL_LED_COLORS.get(name, (255, 255, 255)),
                    })

        # Cue ball potted (ball-in-hand) — show where it was
        if ctx.cue_potted:
            whites = self.shot_start_positions.get("white", [])
            if whites:
                pos = whites[0]
                queue.append({
                    "color_name": "white",
                    "x": int(pos["x"]),
                    "y": int(pos["y"]),
                    "led_color": _BALL_LED_COLORS["white"],
                })

        if not queue:
            return

        self.reposition_queue = queue
        self.reposition_index = 0
        self.reposition_confirm_count = 0
        self.foul_reposition_active = True
        logger.info(
            f"[LED] Reposition queue: "
            + ", ".join(f"{q['color_name']}@({q['x']},{q['y']})" for q in queue)
        )
        self._light_current_reposition_target()

    def start_last_position_reposition(self):
        if not self.shot_start_positions:
            logger.info("[LED] No shot-start snapshot available for last-position restore")
            return False

        threshold = max(10, int(getattr(self.config, "led_reposition_threshold_px", 40)))
        queue = build_reposition_queue(
            self.shot_start_positions,
            self.current_balls_snapshot,
            threshold,
        )
        if not queue:
            logger.info("[LED] No displaced balls found for last-position restore")
            return False

        self.reposition_queue = [
            {
                "color_name": target.color_name,
                "x": target.x,
                "y": target.y,
                "led_color": target.led_color,
            }
            for target in queue
        ]
        self.reposition_index = 0
        self.reposition_confirm_count = 0
        self.foul_reposition_active = True
        self.reposition_hold_active = False
        self.reposition_hold_start = None
        logger.info(
            "[LED] Last-position queue: %s",
            ", ".join(f"{q['color_name']}@({q['x']},{q['y']})" for q in self.reposition_queue),
        )
        if self.led_controller:
            self.led_controller.send_clear()
        self._light_current_reposition_target()
        self._rebuild_overlay_lines()
        self._sync_arduino_display(force=True)
        return True

    def _light_current_reposition_target(self):
        """Send the current queue entry's position+colour to the LED strips."""
        if not self.led_controller:
            return
        if self.reposition_index >= len(self.reposition_queue):
            return
        target = self.reposition_queue[self.reposition_index]
        r, g, b = target["led_color"]
        self.led_controller.send_ball(target["x"], target["y"], r, g, b)
        logger.info(
            f"[LED] Lighting {target['color_name'].upper()} "
            f"at ({target['x']}, {target['y']}) "
            f"[{self.reposition_index + 1}/{len(self.reposition_queue)}]"
        )

    def _check_reposition_progress(self, balls, now):
        """Check if the current target ball has been placed near its position.

        When confirmed for enough frames, hold the LED for a few seconds,
        then advance to the next ball in the queue.
        """
        if self.reposition_index >= len(self.reposition_queue):
            self.clear_foul_leds()
            return

        # ---- Hold phase: LED stays lit after confirmed placement ----
        if self.reposition_hold_active:
            hold_secs = float(getattr(self.config, "led_reposition_hold_seconds", 3.0))
            if now - self.reposition_hold_start >= hold_secs:
                self.reposition_hold_active = False
                self.reposition_hold_start = None
                self.reposition_index += 1
                self.reposition_confirm_count = 0

                if self.reposition_index >= len(self.reposition_queue):
                    self.clear_foul_leds()
                    logger.info("[LED] All balls repositioned")
                else:
                    if self.led_controller:
                        self.led_controller.send_clear()
                    self._light_current_reposition_target()
            return

        # ---- Check phase: is the ball near the target? ----
        target = self.reposition_queue[self.reposition_index]
        threshold = max(10, int(self.config.led_reposition_threshold_px))
        needed_frames = max(1, int(self.config.led_reposition_confirm_frames))

        detected_positions = balls.get(target["color_name"], [])
        placed = False
        for pos in detected_positions:
            dist = math.hypot(pos["x"] - target["x"], pos["y"] - target["y"])
            if dist <= threshold:
                placed = True
                break

        if placed:
            self.reposition_confirm_count += 1
            if self.reposition_confirm_count >= needed_frames:
                logger.info(
                    f"[LED] {target['color_name'].upper()} repositioned OK "
                    f"[{self.reposition_index + 1}/{len(self.reposition_queue)}] "
                    f"— holding {getattr(self.config, 'led_reposition_hold_seconds', 3.0)}s"
                )
                self.reposition_hold_active = True
                self.reposition_hold_start = now
        else:
            self.reposition_confirm_count = 0

    def clear_foul_leds(self):
        """Stop all LED indication and reset foul flash + reposition queue."""
        if self.led_controller:
            self.led_controller.send_clear()
        self.foul_flash_active = False
        self.foul_flash_start = None
        self.foul_potted_colors = set()
        self.foul_reposition_active = False
        self.reposition_hold_active = False
        self.reposition_hold_start = None
        self.reposition_queue = []
        self.reposition_index = 0
        self.reposition_confirm_count = 0
        logger.info("[LED] Foul indicator cleared")

    def skip_reposition_target(self):
        """Skip the current reposition target and move to the next, or finish."""
        if not self.foul_reposition_active:
            return
        self.reposition_index += 1
        self.reposition_confirm_count = 0
        if self.reposition_index >= len(self.reposition_queue):
            self.clear_foul_leds()
            logger.info("[LED] Reposition skipped — all done")
        else:
            if self.led_controller:
                self.led_controller.send_clear()
            self._light_current_reposition_target()

    def _poll_display_events(self):
        if self.arduino is None or not hasattr(self.arduino, "poll_events"):
            return

        for event in self.arduino.poll_events():
            self.handle_display_event(event)

    def handle_display_event(self, event):
        normalized = (event or "").strip().upper()
        if not normalized:
            return

        logger.info("[DISPLAY] Event: %s", normalized)
        if normalized == "DISPLAY_1_UP":
            self._adjust_player_score(self._player_index_for_display(1), 1)
        elif normalized == "DISPLAY_1_DOWN":
            self._adjust_player_score(self._player_index_for_display(1), -1)
        elif normalized == "DISPLAY_2_UP":
            self._adjust_player_score(self._player_index_for_display(2), 1)
        elif normalized == "DISPLAY_2_DOWN":
            self._adjust_player_score(self._player_index_for_display(2), -1)
        elif normalized == "CHANGE_PLAYER":
            self._toggle_active_player()
        elif normalized == "FULL_RESET":
            self._reset_game_from_display()
        elif normalized == "LAST_POSITION":
            if self.foul_reposition_active:
                self.skip_reposition_target()
            else:
                self.start_last_position_reposition()

    def _adjust_player_score(self, player_index, delta):
        player = self.game_state.player1 if player_index == 1 else self.game_state.player2
        player.score = max(0, player.score + int(delta))
        self.last_potted_text = "-"
        self._rebuild_overlay_lines()
        self._sync_arduino_display(force=True)

    def _toggle_active_player(self):
        frame = self.game_state.current_frame
        if frame is None:
            return

        frame.swap_players()
        self._rebuild_overlay_lines()
        self._sync_arduino_display(force=True)

    def _reset_game_from_display(self):
        self.clear_foul_leds()
        self.game_state.start_frame()
        self.rule_engine = RuleEngine(self.game_state)
        self.previous_state = None
        self.ball_tracks = {}
        self.next_track_id = 1
        self.pot_notifications = []
        self.recent_non_red_pots = []
        self.pot_counter = 0
        self.overlay_notifications = []
        self.shot_active = False
        self.shot_last_motion_time = None
        self.shot_stopped_at = None
        self.first_object_hit_colour = None
        self.second_object_hit_colour = None
        self.shot_moved_colours = set()
        self.last_potted_text = "-"
        self.last_ball_hit_text = "-"
        self.last_foul_text = "-"
        self.show_foul_until = 0.0
        self.last_shot_active = False
        self.game_shot_open = False
        self.first_contact_sent_this_shot = False
        self.no_reds_announced = False
        self.red_potted_ever = False
        self.zero_red_since = None
        self.current_balls_snapshot = {}
        self.shot_start_positions = {}
        self._rebuild_overlay_lines()
        self._sync_arduino_display(force=True)

    def _feed_game_logic(self, balls, pot_notifications, now):
        if any((n.get("colour") or "").lower() == "red" for n in pot_notifications):
            self.red_potted_ever = True

        # Shot lifecycle for rules: keep shot open until pot confirmation window has passed.
        shot_end_grace = max(0.2, float(getattr(self.config, "pot_missing_seconds", 2.0)))

        if self.shot_active and not self.game_shot_open:
            if self.foul_flash_active or self.foul_reposition_active:
                self.clear_foul_leds()
            # Snapshot ball positions before the shot changes anything
            if self.previous_state:
                self.shot_start_positions = {
                    c: [dict(p) for p in positions]
                    for c, positions in self.previous_state.items()
                }
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
            # Flash FOUL on the table, then reposition after the flash
            if any(msg.startswith("FOUL") for msg in outputs):
                self._start_foul_flash(now)
            self.game_shot_open = False
            self.first_contact_sent_this_shot = False

        # First contact: always assume the cue ball is first and only detect
        # the first non-white ball as the object ball.
        if self.game_shot_open and self.shot_active and not self.first_contact_sent_this_shot:
            hit_ball = self._colour_to_ball_type(self.first_object_hit_colour)
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
        if not prev_positions:
            return False
        # Ball disappeared — don't treat as "stopped"; it might be occluded or entering pocket
        if not curr_positions:
            # If this colour was already moving this shot, keep it as "moving"
            # so the shot doesn't end prematurely due to occlusion
            return colour in self.shot_moved_colours

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

        if not self.shot_active and white_moving:
            self.shot_active = True
            self.shot_last_motion_time = now
            self.shot_stopped_at = None
            self.first_object_hit_colour = None
            self.second_object_hit_colour = None
            self.shot_moved_colours = set()

        if self.shot_active:
            if any_moving:
                self.shot_last_motion_time = now

            for colour in moving_colours:
                self.shot_moved_colours.add(colour)
                if colour == "white":
                    continue
                if self.first_object_hit_colour is None:
                    self.first_object_hit_colour = colour
                    logger.info(f"[HIT] 1st ball hit: {colour.upper()}")
                elif self.second_object_hit_colour is None and colour != self.first_object_hit_colour:
                    self.second_object_hit_colour = colour
                    logger.info(f"[HIT] 2nd ball hit: {colour.upper()}")

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
        player1 = self.game_state.player1
        player2 = self.game_state.player2
        frame = self.game_state.current_frame
        turn_name = frame.activePlayer.name if frame is not None else "-"
        target_name = frame.activePlayer.target if frame is not None else "-"

        self.live_overlay_lines = [
            "[SHOT]",
            f"Ball hit: {self.last_ball_hit_text}",
            f"Ball potted: {self.last_potted_text}",
            "[FOUL]",
            f"Last foul: {self.last_foul_text}",
            "[POINTS]",
            f"{player1.name}: {player1.score}",
            f"{player2.name}: {player2.score}",
            f"Turn: {turn_name}",
            f"Target: {target_name}",
        ]
        if self.foul_reposition_active and self.reposition_index < len(self.reposition_queue):
            target = self.reposition_queue[self.reposition_index]
            total = len(self.reposition_queue)
            idx = self.reposition_index + 1
            self.live_overlay_lines.append(
                f"[REPOSITION {idx}/{total}] Place {target['color_name'].upper()} "
                f"at ({target['x']}, {target['y']}) - R to skip"
            )

    def _sanitize_lcd_text(self, text, limit=16):
        safe = (text or "").replace("|", "/").replace("\n", " ").replace("\r", " ").strip()
        if len(safe) > limit:
            return safe[:limit]
        return safe

    def _display_index_for_player(self, player_index):
        return 2 if player_index == 1 else 1

    def _player_index_for_display(self, display_index):
        return 1 if display_index == 2 else 2

    def _active_player_index(self):
        frame = self.game_state.current_frame
        if frame is None:
            return 2
        if frame.activePlayer is self.game_state.player1:
            return self._display_index_for_player(1)
        return self._display_index_for_player(2)

    def _format_target_name(self, target):
        if not target:
            return "-"
        if target == "COLOUR":
            return "Colour"
        return target.capitalize()

    def _build_lcd_lines(self, latest_outputs=None):
        frame = self.game_state.current_frame
        if frame is None:
            return "SnookerVision", "No frame"

        if self.foul_reposition_active and self.reposition_index < len(self.reposition_queue):
            target = self.reposition_queue[self.reposition_index]
            line1 = f"Replace {target['color_name']}"
            line2 = f"{self.reposition_index + 1}/{len(self.reposition_queue)}"
            return self._sanitize_lcd_text(line1), self._sanitize_lcd_text(line2)

        if time.time() < self.show_foul_until:
            return self._sanitize_lcd_text("Foul!!!"), self._sanitize_lcd_text("")

        player_label = "Player 1" if frame.activePlayer is self.game_state.player1 else "Player 2"
        target_label = f"Target: {self._format_target_name(frame.activePlayer.target)}"
        return self._sanitize_lcd_text(player_label), self._sanitize_lcd_text(target_label)

    def _sync_arduino_display(self, force=False, latest_outputs=None):
        if self.arduino is None or not self.arduino.is_available:
            return

        frame = self.game_state.current_frame
        score1 = self.game_state.player1.score
        score2 = self.game_state.player2.score
        active_player = self._active_player_index()
        lcd_line1, lcd_line2 = self._build_lcd_lines(latest_outputs=latest_outputs)

        next_state = {
            "score1": score1,
            "score2": score2,
            "active_player": active_player,
            "lcd_line1": lcd_line1,
            "lcd_line2": lcd_line2,
        }

        if not force and next_state == self.arduino_state:
            return

        player1_display = self._display_index_for_player(1)
        player2_display = self._display_index_for_player(2)

        if force or self.arduino_state is None or self.arduino_state["score1"] != score1:
            self.arduino.send_command(f"SET {player1_display} {score1}")
        if force or self.arduino_state is None or self.arduino_state["score2"] != score2:
            self.arduino.send_command(f"SET {player2_display} {score2}")

        if force or self.arduino_state is None or self.arduino_state["active_player"] != active_player:
            self.arduino.send_command("LIGHTOFF")
            self.arduino.send_command(f"LIGHT {active_player} 40 40 40")

        if (
            force
            or self.arduino_state is None
            or self.arduino_state["lcd_line1"] != lcd_line1
            or self.arduino_state["lcd_line2"] != lcd_line2
        ):
            self.arduino.send_command(f"LCD {lcd_line1}|{lcd_line2}")

        self.arduino_state = next_state

    def shutdown(self):
        if self.pocket_sensor is not None:
            self.pocket_sensor.close()
        if self.arduino is not None:
            self.arduino.close()

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
            logger.debug(f"Sending balls: {balls}")
            if corrected_white_ball:
                logger.debug(
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

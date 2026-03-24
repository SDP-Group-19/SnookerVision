"""Test the LED reposition queue without a camera.

Simulates a foul where red, blue, and cue ball are potted,
then fakes placing each ball back to verify the one-by-one LED flow.

Usage:
  1. In terminal 1:  python final/tests/mock_esp32.py
  2. In terminal 2:  python final/tests/test_led_reposition.py

Watch terminal 1 to see LED commands arrive in sequence.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snookervision.game_logic.game_logic import (
    BallType, Event, EventType, GameState, RuleEngine,
)
from snookervision.state.state import StateManager, _BALL_LED_COLORS


def make_config():
    class FakeConfig:
        pass
    c = FakeConfig()
    c.output_dimensions = (1200, 600)
    c.network_update_interval = 0.0
    c.gantry_effective_range_x_px = (100, 1100)
    c.gantry_effective_range_y_px = (84, 516)
    c.position_threshold = 6
    c.enable_pot_notifications = True
    c.pot_missing_seconds = 0.3
    c.pot_track_stale_seconds = 2.0
    c.pot_tracking_match_px = 22
    c.pot_pocket_radius_px = 80
    c.non_red_pot_cooldown_seconds = 0.5
    c.no_reds_confirm_seconds = 0.5
    c.hit_motion_threshold_px = 10
    c.hit_stationary_reset_seconds = 0.3
    c.pot_overlay_ttl_frames = 60
    c.use_networking = False
    c.fast_mode = False
    # LED — point at mock server
    c.led_enabled = True
    c.led_arduino_ip = "127.0.0.1"
    c.led_arduino_port = 4210
    c.led_reposition_threshold_px = 40
    c.led_reposition_stationary_seconds = 3.0
    return c


def make_state():
    class FakeState:
        network = None
        autoencoder = None
    return FakeState()


def make_detections(balls_dict):
    detections = []
    for color, positions in balls_dict.items():
        for (x, y) in positions:
            detections.append({
                "label": color,
                "center": (x, y),
                "bbox": (x - 20, y - 20, x + 20, y + 20),
                "conf": 0.95,
                "classidx": 0,
                "color": (0, 0, 255),
            })
    return detections


def main():
    config = make_config()
    state = make_state()

    sm = StateManager()
    sm.initialize(config, state)

    print("=" * 60)
    print("TEST: LED Reposition Queue (3s stationary)")
    print("=" * 60)

    # --- Setup: force a foul via the rule engine ---
    sm.shot_start_positions = {
        "white": [{"x": 600, "y": 300}],
        "red":   [{"x": 400, "y": 200}],
        "blue":  [{"x": 300, "y": 150}],
        "pink":  [{"x": 500, "y": 100}],
    }
    # Need previous_state for _colour_is_moving checks
    sm.previous_state = {
        "pink": [{"x": 500, "y": 100}],
    }

    gs = sm.game_state
    gs.start_frame()
    gs.current_frame.reds_left = 10
    engine = sm.rule_engine

    engine.on_event(Event(time.time(), EventType.SHOT_START))
    engine.on_event(Event(time.time(), EventType.FIRST_CONTACT,
                          {"a": BallType.CUE, "b": BallType.RED}))
    engine.on_event(Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.RED}))
    engine.on_event(Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.BLUE}))
    engine.on_event(Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.CUE}))
    outputs = engine.on_event(Event(time.time(), EventType.SHOT_END))

    print(f"\nRule engine outputs: {outputs}")
    is_foul = any(msg.startswith("FOUL") for msg in outputs)
    print(f"Foul detected: {is_foul}")

    if is_foul:
        sm._push_game_outputs(outputs)
        sm._build_reposition_queue()

    print(f"\nReposition queue ({len(sm.reposition_queue)} balls):")
    for i, item in enumerate(sm.reposition_queue):
        print(f"  {i+1}. {item['color_name'].upper()} at ({item['x']},{item['y']}) "
              f"LED=rgb{item['led_color']}")

    if not sm.reposition_queue:
        print("\nNo balls in queue.")
        return

    print("\n" + "=" * 60)
    print("Watch the mock ESP32 terminal for LED commands!")
    print("Each ball lights up, stays for 3s stationary, then next.")
    print("=" * 60)

    on_table = {"pink": [(500, 100)]}

    for qi, target in enumerate(list(sm.reposition_queue)):
        print(f"\n--- [{qi+1}/{len(sm.reposition_queue)}] "
              f"{target['color_name'].upper()} at ({target['x']},{target['y']}) ---")
        print(f"  LED is now on. Simulating ball placement...")

        # Place the ball at the target position
        on_table[target["color_name"]] = [(target["x"], target["y"])]
        sm.previous_state = {
            c: [{"x": p[0], "y": p[1]} for p in positions]
            for c, positions in on_table.items()
        }

        # Feed stationary frames for 3+ seconds
        start = time.time()
        while sm.foul_reposition_active and sm.reposition_index == qi:
            sm.update(make_detections(on_table))
            elapsed = time.time() - start
            if elapsed > 5.0:
                print(f"  Timeout — skipping")
                sm.skip_reposition_target()
                break
            time.sleep(0.05)

        if sm.reposition_index > qi:
            print(f"  Confirmed after {time.time() - start:.1f}s")

    print("\n" + "=" * 60)
    if not sm.foul_reposition_active:
        print("All balls repositioned! LEDs cleared.")
    else:
        sm.clear_foul_leds()
    print("=" * 60)

    if sm.led_controller:
        sm.led_controller.close()


if __name__ == "__main__":
    main()

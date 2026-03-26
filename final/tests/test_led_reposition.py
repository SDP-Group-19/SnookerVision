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

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snookervision.game_logic.game_logic import (
    BallType, Event, EventType, GameState, RuleEngine,
)
from snookervision.state.state import StateManager, _BALL_LED_COLORS


def make_config():
    """Minimal config object with the fields StateManager needs."""
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
    # LED settings — point at mock server
    c.led_enabled = True
    c.led_arduino_ip = "127.0.0.1"
    c.led_arduino_port = 4210
    c.led_foul_flash_seconds = 3.0
    c.led_reposition_threshold_px = 40
    c.led_reposition_confirm_frames = 3
    return c


def make_state():
    class FakeState:
        network = None
        autoencoder = None
    return FakeState()


def make_detections(balls_dict):
    """Convert {"red": [(x,y)], "white": [(x,y)]} to detection list."""
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
    print("TEST: LED Reposition Queue")
    print("=" * 60)

    # --- Phase 1: Establish ball positions (stationary) ---
    print("\n[Phase 1] Establishing ball positions...")
    initial_balls = {
        "white": [(600, 300)],
        "red":   [(400, 200)],
        "blue":  [(300, 150)],
        "pink":  [(500, 100)],
    }
    detections = make_detections(initial_balls)
    for _ in range(10):
        sm.update(detections)
        time.sleep(0.02)
    print(f"  Balls on table: white@(600,300) red@(400,200) blue@(300,150) pink@(500,100)")

    # --- Phase 2: Simulate a shot (cue ball moves) ---
    print("\n[Phase 2] Simulating shot — cue ball moving...")
    for i in range(5):
        moving_balls = {
            "white": [(600 + i * 30, 300)],
            "red":   [(400, 200)],
            "blue":  [(300, 150)],
            "pink":  [(500, 100)],
        }
        sm.update(make_detections(moving_balls))
        time.sleep(0.05)

    # --- Phase 3: Balls collide and some get potted ---
    print("[Phase 3] Red and blue disappear (potted)...")
    # White hits red, red and blue go to pockets
    for i in range(5):
        after_hit = {
            "white": [(750, 300)],
            # red moving toward pocket
            "red":   [(400 + i * 40, 200 - i * 40)] if i < 3 else [],
            # blue moving toward pocket
            "blue":  [(300 - i * 30, 150 - i * 30)] if i < 3 else [],
            "pink":  [(500, 100)],
        }
        # Remove empty lists
        after_hit = {k: v for k, v in after_hit.items() if v}
        sm.update(make_detections(after_hit))
        time.sleep(0.05)

    # --- Phase 4: Everything stops, white also potted ---
    print("[Phase 4] White also potted (in-off)...")
    stopped = {
        "pink": [(500, 100)],
    }
    for _ in range(15):
        sm.update(make_detections(stopped))
        time.sleep(0.05)

    # At this point the rule engine should detect a foul (no first contact on legal ball,
    # or cue potted). Let's force the foul through the game logic directly.
    print("\n[Phase 5] Forcing foul through rule engine...")
    # Manually trigger the shot lifecycle since our simple simulation
    # may not perfectly trigger the state manager's shot detection.
    sm.shot_start_positions = {
        "white": [{"x": 600, "y": 300}],
        "red":   [{"x": 400, "y": 200}],
        "blue":  [{"x": 300, "y": 150}],
        "pink":  [{"x": 500, "y": 100}],
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

    print(f"  Rule engine outputs: {outputs}")
    is_foul = any(msg.startswith("FOUL") for msg in outputs)
    print(f"  Foul detected: {is_foul}")

    if is_foul:
        sm._push_game_outputs(outputs)
        sm._build_reposition_queue()

    print(f"\n  Reposition queue ({len(sm.reposition_queue)} balls):")
    for i, item in enumerate(sm.reposition_queue):
        print(f"    {i+1}. {item['color_name'].upper()} at ({item['x']},{item['y']}) "
              f"LED={item['led_color']}")

    if not sm.reposition_queue:
        print("\n  No balls in queue — check if foul was detected.")
        return

    # --- Phase 6: Simulate placing balls back one by one ---
    print("\n" + "=" * 60)
    print("Now watch the mock ESP32 terminal!")
    print("LEDs should pulse for each ball, one at a time.")
    print("=" * 60)

    confirm_frames = config.led_reposition_confirm_frames

    for qi, target in enumerate(list(sm.reposition_queue)):
        print(f"\n[Reposition {qi+1}/{len(sm.reposition_queue)}] "
              f"Waiting for {target['color_name'].upper()} "
              f"to be placed at ({target['x']},{target['y']})...")
        time.sleep(2.0)  # Let the pulse blink a few times

        print(f"  Placing {target['color_name']} back...")
        placed_balls = dict(stopped)  # pink still on table
        placed_balls[target["color_name"]] = [(target["x"], target["y"])]

        # Feed frames with the ball at the target position
        for frame in range(confirm_frames + 2):
            sm.update(make_detections(placed_balls))
            time.sleep(0.05)

            if not sm.foul_reposition_active:
                break
            if sm.reposition_index > qi:
                break

        if sm.foul_reposition_active and sm.reposition_index <= qi:
            print(f"  (Ball not confirmed — pressing R to skip)")
            sm.skip_reposition_target()

        # Add this ball to the "on table" set for subsequent iterations
        stopped[target["color_name"]] = [(target["x"], target["y"])]

    print("\n" + "=" * 60)
    if not sm.foul_reposition_active:
        print("All balls repositioned! LEDs cleared.")
    else:
        print(f"Still active — index {sm.reposition_index}/{len(sm.reposition_queue)}")
        sm.clear_foul_leds()
    print("=" * 60)

    # Cleanup
    if sm.led_controller:
        sm.led_controller.close()


if __name__ == "__main__":
    main()

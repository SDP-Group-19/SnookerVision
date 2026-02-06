import unittest
from backend.gameLogic.game_logic import Event, EventType, BallType

def test_pot_red_scores_1_and_continues(engine):
    events = [
        Event(None, EventType.SHOT_START),
        Event(None, EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.RED}),
        Event(None, EventType.BALL_POTTED, {"ball": BallType.RED}),
        Event(None, EventType.SHOT_END),
    ]

    outputs = []
    for e in events:
        outputs += engine.on_event(e)

    fs = engine.gameState.current_frame
    p1 = fs.activePlayer

    assert p1.score == 1
    assert fs.phase.name == "IDLE"
    assert "LEGAL" in " ".join(outputs)
    assert p1.target == "COLOUR"

if __name__ == '__main__':
    unittest.main()

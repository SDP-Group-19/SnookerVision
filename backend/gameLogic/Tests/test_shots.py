import unittest
from backend.gameLogic.game_logic import Event, EventType, BallType
from utils import *

def test_pot_red_scores_1_and_continues(engine):
    events = [
        Event(None, EventType.SHOT_START),
        Event(None, EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.RED}),
        Event(None, EventType.BALL_POTTED, {"ball": BallType.RED}),
        Event(None, EventType.SHOT_END),
    ]

    outputs = run(engine, *events)

    fs = engine.gameState.current_frame
    p1 = fs.activePlayer

    assert p1.score == 1
    assert fs.phase.name == "IDLE"
    assert "LEGAL" in " ".join(outputs)
    assert p1.target == "COLOUR"

def test_no_pot_and_continues(engine):
    events = [
        Event(None, EventType.SHOT_START),
        Event(None, EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.RED}),
        Event(None, EventType.SHOT_END),
    ]

    fs = engine.gameState.current_frame
    p1 = fs.activePlayer
    p2 = fs.opponent

    outputs = run(engine, *events)

    assert p1.score == 0
    assert p2.score == 0
    assert fs.phase.name == "IDLE"
    assert "LEGAL" in " ".join(outputs)
    assert fs.activePlayer == p2

if __name__ == '__main__':
    unittest.main()

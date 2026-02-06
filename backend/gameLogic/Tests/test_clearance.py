import unittest
from backend.gameLogic.game_logic import Event, EventType, BallType
from utils import *

def pot_colour(colour):
    return [
        Event(None, EventType.SHOT_START),
        Event(None, EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": colour}),
        Event(None, EventType.BALL_POTTED, {"ball": colour}),
        Event(None, EventType.SHOT_END),
    ]

def test_colour_clearance_order(engine):
    fs = engine.gameState.current_frame

    engine.on_event(Event(None, EventType.NO_REDS_REMAINING))
    fs.activePlayer.target = "YELLOW"

    run(engine, *pot_colour(BallType.YELLOW))

    assert fs.activePlayer.target == "GREEN"
    assert fs.tempBallOrder[0] == "GREEN"

if __name__ == '__main__':
    unittest.main()

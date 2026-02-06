import unittest
from backend.gameLogic.game_logic import Event, EventType, BallType
from utils import *

def test_foul_wrong_first_contact(engine):
    fs = engine.gameState.current_frame
    opponent = fs.opponent

    events = [
        Event(None, EventType.SHOT_START),
        Event(None, EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.BLACK}),
        Event(None, EventType.SHOT_END),
    ]

    run(engine, *events)

    assert opponent.score == 7  # BLACK foul value
    assert fs.activePlayer == opponent  # turn swapped


if __name__ == '__main__':
    unittest.main()

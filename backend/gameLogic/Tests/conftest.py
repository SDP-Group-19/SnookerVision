import pytest
from backend.gameLogic.game_logic import GameState, RuleEngine

@pytest.fixture
def engine():
    gs = GameState()
    gs.start_frame()
    return RuleEngine(gs)
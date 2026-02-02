import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

class BallType(Enum):
    RED = auto()
    YELLOW = auto()
    GREEN = auto()
    BROWN = auto()
    BLUE = auto()
    PINK = auto()
    BLACK = auto()
    CUE = auto()

BALL_ORDER = [
    "YELLOW",
    "GREEN",
    "BROWN",
    "BLUE",
    "PINK",
    "BLACK"
]

BALL_VALUE = {
    BallType.RED: 1,
    BallType.YELLOW: 2,
    BallType.GREEN: 3,
    BallType.BROWN: 4,
    BallType.BLUE: 5,
    BallType.PINK: 6,
    BallType.BLACK: 7,
    BallType.CUE: 0,
}

class EventType(Enum):
    SHOT_START = auto()
    SHOT_END = auto()
    FIRST_CONTACT = auto()
    BALL_POTTED = auto()
    NO_REDS_REMAINING = auto()

@dataclass
class Event:
    t: float
    type: EventType
    data: dict = field(default_factory=dict)

@dataclass
class ShotContext:
    first_contact: Optional[Tuple[BallType, BallType]] = None
    potted: List[BallType] = field(default_factory=list)
    cue_potted: bool = False

class Phase(Enum):
    IDLE = auto()
    IN_SHOT = auto()

'''
- TODO PAUSE GAME ON FOUL TO REQUEST PLAYER IF THEY WANT TO FORCE 
OPPONENT TO RETAKE SHOT
- END GAME WHEN DIFFERENCE OF > 7 POINTS, WHEN BLACK IS ONLY BALL LEFT
- SET AMOUNT OF FRAMES/GAMES
- PLAYER STARTING ALTERNATES BETWEEN FRAMES
- COLOURS MUST BE POT IN ASCENDING ORDER AFTER REDS ARE GONE
'''

@dataclass
class GameState:
    games: int = 1 # default 1 for now
    frames: int = 3 # default 3 for now
    firstTurn: int = 0 # player who starts first

@dataclass
class FrameState:
    turn: int = 0  # 0 for P1, 1 for P2
    score: List[int] = field(default_factory=lambda: [0, 0])
    # YELLOW, GREEN, BROWN, BLUE... ---> WHEN REDS ARE GONE
    colourClearance: bool = False # for when reds are gone
    target: str = "RED"  # "RED" or "COLOR" (simplify)
    phase: Phase = Phase.IDLE
    ctx: ShotContext = field(default_factory=ShotContext)

class RuleEngine:
    def __init__(self) -> None:
        self.gameState = GameState()
        self.frameState = FrameState()

    def on_event(self, e: Event) -> List[str]:
        gs = self.gameState
        fs = self.frameState
        outputs: List[str] = []

        if e.type == EventType.SHOT_START and fs.phase == Phase.IDLE:
            fs.phase = Phase.IN_SHOT
            fs.ctx = ShotContext()
            outputs.append("SHOT_START")

        elif e.type == EventType.NO_REDS_REMAINING and fs.phase == Phase.IDLE:
            outputs.append("NO_REDS_REMAINING")
            fs.colourClearance = True

        elif e.type == EventType.FIRST_CONTACT and fs.phase == Phase.IN_SHOT:
            if fs.ctx.first_contact is None:
                fs.ctx.first_contact = (e.data["a"], e.data["b"])
                outputs.append(f"FIRST_CONTACT {fs.ctx.first_contact}")

        elif e.type == EventType.BALL_POTTED and fs.phase == Phase.IN_SHOT:
            b = e.data["ball"]
            if b == BallType.CUE:
                fs.ctx.cue_potted = True
            else:
                fs.ctx.potted.append(b)
            outputs.append(f"BALL_POTTED {b.name}")

        elif e.type == EventType.SHOT_END and fs.phase == Phase.IN_SHOT:
            outputs.extend(self._resolve_shot())
            fs.phase = Phase.IDLE

        return outputs

    def _next_target(self):
        """
        LOGIC TO DECIDE NEXT SHOT TARGET,
        SHOULD USE BALL_ORDER
        """

        fs = self.frameState

        if fs.target in BALL_ORDER:
            index = BALL_ORDER.index(fs.target)
            index += 1 # get next colour in sequence
            if index < len(BALL_ORDER):
                fs.target = BALL_ORDER[index]


    def _resolve_shot(self) -> List[str]:
        gs = self.gameState
        fs = self.frameState
        ctx = fs.ctx
        out: List[str] = []

        foul = False
        foul_points = 0

        # MVP foul: cue ball potted
        if ctx.cue_potted:
            foul = True
            foul_points = max(foul_points, 4)

        # MVP foul: no contact detected (you can enable once you have cue ball)
        if ctx.first_contact is None:
            foul = True
            foul_points = max(foul_points, 4)

        if ctx.first_contact[0] != BallType.CUE:
            foul = True
            foul_points = max(foul_points, BALL_VALUE[ctx.first_contact[0]])

        if (ctx.first_contact[1] != BallType.RED) and (fs.target == "RED"):
            foul = True
            foul_points = max(foul_points, BALL_VALUE[ctx.first_contact[1]])

        if (ctx.first_contact[1] == BallType.RED) and (fs.target == "COLOR"):
            foul = True
            foul_points = max(foul_points, 4)

        if foul:
            opponent = 1 - fs.turn
            fs.score[opponent] += foul_points
            fs.turn = opponent
            out.append(f"FOUL +{foul_points} to P{opponent+1}, turn -> P{fs.turn+1}")
            # target stays same in MVP

            # target should return to red no?
            fs.target = "RED"

            return out

        # scoring: sum of potted (simplified)
        gained = sum(BALL_VALUE[b] for b in ctx.potted)
        fs.score[fs.turn] += gained
        out.append(f"LEGAL: P{fs.turn+1} +{gained} (score {fs.score})")

        # turn logic (simplified): continue if potted any, else switch
        if gained == 0:
            fs.turn = 1 - fs.turn
            # target should return to red no?
            fs.target = "RED"
            out.append(f"NO_POT: turn -> P{fs.turn+1}")
        else:
            out.append(f"CONTINUE: P{fs.turn+1} keeps turn")
            # target should alternate.
            if fs.target == "RED":
                fs.target = "COLOR"
            else:
                fs.target = "RED"

        return out

engine = RuleEngine()

events = [
    Event(time.time(), EventType.SHOT_START),
    Event(time.time(), EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.RED}),
    Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.RED}),
    Event(time.time(), EventType.SHOT_END),
    Event(time.time(), EventType.SHOT_START),
    Event(time.time(), EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.BLACK}),
    Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.BLACK}),
    Event(time.time(), EventType.SHOT_END),
    Event(time.time(), EventType.SHOT_START),
    Event(time.time(), EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.BLACK}),
    Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.BLACK}),
    Event(time.time(), EventType.SHOT_END),
]

for e in events:
    for msg in engine.on_event(e):
        print(msg)

print("Final score:", engine.frameState.score)
print("Turn:", engine.frameState.turn)

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

@dataclass
class GameState:
    turn: int = 0  # 0 for P1, 1 for P2
    score: List[int] = field(default_factory=lambda: [0, 0])
    target: str = "RED"  # "RED" or "COLOR" (simplify)
    phase: Phase = Phase.IDLE
    ctx: ShotContext = field(default_factory=ShotContext)

class RuleEngine:
    def __init__(self) -> None:
        self.state = GameState()

    def on_event(self, e: Event) -> List[str]:
        s = self.state
        outputs: List[str] = []

        if e.type == EventType.SHOT_START and s.phase == Phase.IDLE:
            s.phase = Phase.IN_SHOT
            s.ctx = ShotContext()
            outputs.append("SHOT_START")

        elif e.type == EventType.FIRST_CONTACT and s.phase == Phase.IN_SHOT:
            if s.ctx.first_contact is None:
                s.ctx.first_contact = (e.data["a"], e.data["b"])
                outputs.append(f"FIRST_CONTACT {s.ctx.first_contact}")

        elif e.type == EventType.BALL_POTTED and s.phase == Phase.IN_SHOT:
            b = e.data["ball"]
            if b == BallType.CUE:
                s.ctx.cue_potted = True
            else:
                s.ctx.potted.append(b)
            outputs.append(f"BALL_POTTED {b.name}")

        elif e.type == EventType.SHOT_END and s.phase == Phase.IN_SHOT:
            outputs.extend(self._resolve_shot())
            s.phase = Phase.IDLE

        return outputs

    def _resolve_shot(self) -> List[str]:
        s = self.state
        ctx = s.ctx
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

        if (ctx.first_contact[1] != BallType.RED) and (s.target == "RED"):
            foul = True
            foul_points = max(foul_points, BALL_VALUE[ctx.first_contact[1]])

        if (ctx.first_contact[1] == BallType.RED) and (s.target == "COLOR"):
            foul = True
            foul_points = max(foul_points, 4)

        if foul:
            opponent = 1 - s.turn
            s.score[opponent] += foul_points
            s.turn = opponent
            out.append(f"FOUL +{foul_points} to P{opponent+1}, turn -> P{s.turn+1}")
            # target stays same in MVP
            return out

        # scoring: sum of potted (simplified)
        gained = sum(BALL_VALUE[b] for b in ctx.potted)
        s.score[s.turn] += gained
        out.append(f"LEGAL: P{s.turn+1} +{gained} (score {s.score})")

        # turn logic (simplified): continue if potted any, else switch
        if gained == 0:
            s.turn = 1 - s.turn
            out.append(f"NO_POT: turn -> P{s.turn+1}")
        else:
            out.append(f"CONTINUE: P{s.turn+1} keeps turn")

        return out

engine = RuleEngine()

events = [
    Event(time.time(), EventType.SHOT_START),
    Event(time.time(), EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.RED}),
    Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.RED}),
    Event(time.time(), EventType.SHOT_END),
]

for e in events:
    for msg in engine.on_event(e):
        print(msg)

print("Final score:", engine.state.score)
print("Turn:", engine.state.turn)

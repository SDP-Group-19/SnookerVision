import time
from dataclasses import dataclass, field
from enum import Enum, auto
from idlelib.autocomplete import TRY_A
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
    GAME_FORFEITED = auto()
    NO_BALLS_REMAINING = auto()

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
class Player:
    name: str
    gamesWon: int = 0
    framesWon: int = 0
    score: int = 0
    # YELLOW, GREEN, BROWN, BLUE... ---> WHEN REDS ARE GONE
    target: str = "RED"  # "RED" or "COLOUR" (simplify)

@dataclass
class GameState: # PROBABLY JUST GOING TO LEAVE FOR NOW
    games: int = 1 # default 1 for now
    frames: int = 3 # default 3 for now
    player1: Player = field(default_factory=lambda: Player("Player1"))
    player2: Player = field(default_factory=lambda: Player("Player2"))
    firstTurn: Player = field(init=False) # player who starts first

    def __post_init__(self):
        self.firstTurn = self.player1

    def end_frame(self): # FOR NOW WERE ONLY PLAYING 1 FRAME AS MVP
        print("GAME HAS ENDED")

@dataclass
class FrameState:
    tempBallOrder: List[str] = field(default_factory=lambda: BALL_ORDER.copy())
    colourClearance: bool = False # for when reds are gone
    activePlayer: Player = field(default_factory=lambda: Player("Player1"))
    opponent: Player = field(default_factory=lambda: Player("Player2"))
    phase: Phase = Phase.IDLE
    ctx: ShotContext = field(default_factory=ShotContext)

    def swap_players(self):
        temp = self.activePlayer
        self.activePlayer = self.opponent
        self.opponent = temp

        #reset player target
        if not self.colourClearance:
            self.activePlayer.target = "RED"
        else:
            self.activePlayer.target = self.tempBallOrder[0] #ensure it's not empty

    def get_next_target(self, justPotRed: bool):
        #NORMAL PLAY
        if justPotRed and not self.colourClearance:
            self.activePlayer.target = "COLOUR"
        if not justPotRed and not self.colourClearance:
            self.activePlayer.target = "RED"

        #COLOUR CLEARANCE
        if not justPotRed and self.colourClearance and (len(self.tempBallOrder) > 0):
            self.activePlayer.target = self.tempBallOrder[0] #ensure it's not empty
        if justPotRed and self.colourClearance:
            self.activePlayer.target = "COLOUR"

        print(f"TEST NEXT TARGET: {self.activePlayer.target}")


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

        elif e.type == EventType.GAME_FORFEITED and fs.phase == Phase.IDLE:
            pass

        elif e.type == EventType.NO_BALLS_REMAINING and fs.phase == Phase.IDLE:
            gs.end_frame()

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

    def _resolve_shot(self) -> List[str]:
        gs = self.gameState
        fs = self.frameState
        ctx = fs.ctx
        out: List[str] = []

        foul = False
        foul_points = 4

        # MVP foul: cue ball potted
        if ctx.cue_potted:
            foul = True
            foul_points = max(foul_points, 4)

        # MVP foul: no contact detected (you can enable once you have cue ball)
        if ctx.first_contact is None:
            foul = True
            foul_points = max(foul_points, 4)
        else:
            if ctx.first_contact[0] != BallType.CUE:
                foul = True
                foul_points = max(foul_points, BALL_VALUE[ctx.first_contact[0]])

            if (ctx.first_contact[1] != BallType.RED) and (fs.activePlayer.target == "RED"):
                foul = True
                foul_points = max(foul_points, BALL_VALUE[ctx.first_contact[1]])

            if (ctx.first_contact[1] == BallType.RED) and (fs.activePlayer.target == "COLOUR"):
                foul = True
                foul_points = max(foul_points, 4)

            if (ctx.first_contact[1] != BallType.YELLOW) and (fs.activePlayer.target == "YELLOW"):
                foul = True
                foul_points = max(foul_points, BALL_VALUE[ctx.first_contact[1]])

            if (ctx.first_contact[1] != BallType.GREEN) and (fs.activePlayer.target == "GREEN"):
                foul = True
                foul_points = max(foul_points, BALL_VALUE[ctx.first_contact[1]])

            if (ctx.first_contact[1] != BallType.BROWN) and (fs.activePlayer.target == "BROWN"):
                foul = True
                foul_points = max(foul_points, BALL_VALUE[ctx.first_contact[1]])

            if (ctx.first_contact[1] != BallType.BLUE) and (fs.activePlayer.target == "BLUE"):
                foul = True
                foul_points = max(foul_points, BALL_VALUE[ctx.first_contact[1]])

            if (ctx.first_contact[1] != BallType.PINK) and (fs.activePlayer.target == "PINK"):
                foul = True
                foul_points = max(foul_points, BALL_VALUE[ctx.first_contact[1]])

            if (ctx.first_contact[1] != BallType.BLACK) and (fs.activePlayer.target == "BLACK"):
                foul = True
                foul_points = max(foul_points, BALL_VALUE[ctx.first_contact[1]])

            if (any(b != BallType.RED for b in fs.ctx.potted)) and (fs.activePlayer.target == "RED"):
                foul = True
                for b in fs.ctx.potted:
                    foul_points = max(foul_points, BALL_VALUE[b]) # foul is value of highest ball

            if (len(ctx.potted) > 1) and (fs.activePlayer.target == "COLOUR"):
                foul = True
                for b in fs.ctx.potted:
                    foul_points = max(foul_points, BALL_VALUE[b])

            if (any(b != BallType.YELLOW for b in fs.ctx.potted)) and (fs.activePlayer.target == "YELLOW"):
                foul = True
                for b in fs.ctx.potted:
                    foul_points = max(foul_points, BALL_VALUE[b])

            if (any(b != BallType.GREEN for b in fs.ctx.potted)) and (fs.activePlayer.target == "GREEN"):
                foul = True
                for b in fs.ctx.potted:
                    foul_points = max(foul_points, BALL_VALUE[b])

            if (any(b != BallType.BROWN for b in fs.ctx.potted)) and (fs.activePlayer.target == "BROWN"):
                foul = True
                for b in fs.ctx.potted:
                    foul_points = max(foul_points, BALL_VALUE[b])

            if (any(b != BallType.BLUE for b in fs.ctx.potted)) and (fs.activePlayer.target == "BLUE"):
                foul = True
                for b in fs.ctx.potted:
                    foul_points = max(foul_points, BALL_VALUE[b])

            if (any(b != BallType.PINK for b in fs.ctx.potted)) and (fs.activePlayer.target == "PINK"):
                foul = True
                for b in fs.ctx.potted:
                    foul_points = max(foul_points, BALL_VALUE[b])

            if (any(b != BallType.BLACK for b in fs.ctx.potted)) and (fs.activePlayer.target == "BLACK"):
                foul = True
                for b in fs.ctx.potted:
                    foul_points = max(foul_points, BALL_VALUE[b])

        if foul:
            fs.opponent.score += foul_points
            out.append(f"FOUL +{foul_points} to {fs.opponent.name}")

            fs.swap_players()
            return out

        # scoring: sum of potted (simplified)
        gained = sum(BALL_VALUE[b] for b in ctx.potted)
        fs.activePlayer.score += gained
        out.append(f"LEGAL: {fs.activePlayer.name} +{gained} (score {fs.activePlayer.score})")

        # turn logic (simplified): continue if potted any, else switch
        if gained == 0:
            out.append(f"NO_POT: turn -> {fs.opponent.name}")
            fs.swap_players()
        else:
            out.append(f"CONTINUE: {fs.activePlayer.name} keeps turn")
            #MUST HAVE POT RED IF NO FOULS AND TARGET WAS RED
            justPotRed = (fs.activePlayer.target == "RED")

            if not justPotRed and fs.activePlayer.target != "COLOUR" and fs.colourClearance:
                if fs.tempBallOrder:
                    fs.tempBallOrder.pop(0) #must have pot the colour in colour clearance
                if len(fs.tempBallOrder) == 0:
                    gs.end_frame()

            fs.get_next_target(justPotRed)

        return out

engine = RuleEngine()

events = [
    Event(time.time(), EventType.SHOT_START),
    Event(time.time(), EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.RED}),
    Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.BLACK}),
    Event(time.time(), EventType.SHOT_END),
    Event(time.time(), EventType.SHOT_START),
    Event(time.time(), EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.BLACK}),
    Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.BLACK}),
    Event(time.time(), EventType.SHOT_END),
    Event(time.time(), EventType.SHOT_START),
    Event(time.time(), EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.RED}),
    Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.RED}),
    Event(time.time(), EventType.SHOT_END),
    Event(time.time(), EventType.NO_REDS_REMAINING),
    Event(time.time(), EventType.SHOT_START),
    Event(time.time(), EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.BLACK}),
    Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.BLACK}),
    Event(time.time(), EventType.SHOT_END),
    Event(time.time(), EventType.SHOT_START),
    Event(time.time(), EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.RED}),
    Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.RED}),
    Event(time.time(), EventType.SHOT_END),
    Event(time.time(), EventType.SHOT_START),
    Event(time.time(), EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.YELLOW}),
    Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.YELLOW}),
    Event(time.time(), EventType.SHOT_END),
    Event(time.time(), EventType.SHOT_START),
    Event(time.time(), EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.GREEN}),
    Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.GREEN}),
    Event(time.time(), EventType.SHOT_END),
    Event(time.time(), EventType.SHOT_START),
    Event(time.time(), EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.BROWN}),
    Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.BROWN}),
    Event(time.time(), EventType.SHOT_END),
    Event(time.time(), EventType.SHOT_START),
    Event(time.time(), EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.BLUE}),
    Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.BLUE}),
    Event(time.time(), EventType.SHOT_END),
    Event(time.time(), EventType.SHOT_START),
    Event(time.time(), EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.PINK}),
    Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.PINK}),
    Event(time.time(), EventType.SHOT_END),
    Event(time.time(), EventType.SHOT_START),
    Event(time.time(), EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.BLACK}),
    Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.BLACK}),
    Event(time.time(), EventType.SHOT_END),
    Event(time.time(), EventType.NO_BALLS_REMAINING),
]

for e in events:
    for msg in engine.on_event(e):
        print(msg)

print("Final scores")
print(f"{engine.frameState.activePlayer.name} score: {engine.frameState.activePlayer.score}")
print(f"{engine.frameState.opponent.name} score: {engine.frameState.opponent.score}")

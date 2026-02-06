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

LEGAL_TARGETS = {
    "RED": {BallType.RED},
    "COLOUR": {
        BallType.YELLOW,
        BallType.GREEN,
        BallType.BROWN,
        BallType.BLUE,
        BallType.PINK,
        BallType.BLACK,
    },
    "YELLOW": {BallType.YELLOW},
    "GREEN": {BallType.GREEN},
    "BROWN": {BallType.BROWN},
    "BLUE": {BallType.BLUE},
    "PINK": {BallType.PINK},
    "BLACK": {BallType.BLACK},
}

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
class FrameState:
    activePlayer: Player = field()
    opponent: Player = field()
    tempBallOrder: List[str] = field(default_factory=lambda: BALL_ORDER.copy())
    colourClearance: bool = False # for when reds are gone
    phase: Phase = Phase.IDLE
    ctx: ShotContext = field(default_factory=ShotContext)

    def swap_players(self):
        self.activePlayer, self.opponent = self.opponent, self.activePlayer

        #reset player target
        if not self.colourClearance:
            self.activePlayer.target = "RED"
        else:
            self.activePlayer.target = self.tempBallOrder[0] #ensure it's not empty

    def get_next_target(self, just_potted_red: bool):
        #NORMAL PLAY
        if just_potted_red and not self.colourClearance:
            self.activePlayer.target = "COLOUR"
        if not just_potted_red and not self.colourClearance:
            self.activePlayer.target = "RED"

        #COLOUR CLEARANCE
        if not just_potted_red and self.colourClearance and (len(self.tempBallOrder) > 0):
            self.activePlayer.target = self.tempBallOrder[0] #ensure it's not empty
        if just_potted_red and self.colourClearance:
            self.activePlayer.target = "COLOUR"

        print(f"TEST NEXT TARGET: {self.activePlayer.target}")

@dataclass
class GameState: # PROBABLY JUST GOING TO LEAVE FOR NOW
    games: int = 1 # default 1 for now
    frames: int = 1 # default 3 for now
    player1: Player = field(default_factory=lambda: Player("Player1"))
    player2: Player = field(default_factory=lambda: Player("Player2"))
    firstTurn: Player = field(init=False) # player who starts first
    current_frame: FrameState | None = field(init=False, default=None)

    def __post_init__(self):
        self.firstTurn = self.player1

    def start_frame(self):
        self.current_frame = FrameState(
            activePlayer=self.firstTurn,
            opponent=self.player2 if self.firstTurn is self.player1 else self.player1
        )

    def end_frame(self): # FOR NOW WERE ONLY PLAYING 1 FRAME AS MVP
        print("FRAME HAS ENDED")
        print("Final scores")
        print(f"{self.player1.name} score: {self.player1.score}")
        print(f"{self.player2.name} score: {self.player2.score}")

        if self.player1.score > self.player2.score:
            self.player1.framesWon += 1
        else:
            self.player2.framesWon += 1

        if self.player1.framesWon == self.frames:
            self.player1.gamesWon += 1

            self.player1.framesWon = 0
            self.player2.framesWon = 0
        elif self.player2.framesWon == self.frames:
            self.player2.gamesWon += 1

            self.player1.framesWon = 0
            self.player2.framesWon = 0

        if self.player1.gamesWon == self.games:
            print("Game Over")
            print(f"{self.player1.name} HAS WON")
        elif self.player2.gamesWon == self.games:
            print("Game Over")
            print(f"{self.player2.name} HAS WON")

    def forfeit_frame(self, forfeit_player: Player):
        print("Frame HAS BEEN FORFEIT")
        print(f"{forfeit_player.name} FORFEIT")
        print("Final scores")
        print(f"{self.player1.name} score: {self.player1.score}")
        print(f"{self.player2.name} score: {self.player2.score}")

def foul_value(*balls: BallType) -> int:
    return max(4, *(BALL_VALUE[b] for b in balls if b in BALL_VALUE))

class RuleEngine:
    def __init__(self, game_state: GameState) -> None:
        self.gameState = game_state

    def on_event(self, e: Event) -> List[str]:
        gs = self.gameState
        fs = gs.current_frame
        outputs: List[str] = []

        if e.type == EventType.SHOT_START and fs.phase == Phase.IDLE:
            fs.phase = Phase.IN_SHOT
            fs.ctx = ShotContext()
            outputs.append("SHOT_START")

        elif e.type == EventType.NO_REDS_REMAINING and fs.phase == Phase.IDLE:
            outputs.append("NO_REDS_REMAINING")
            fs.colourClearance = True

        elif e.type == EventType.GAME_FORFEITED and fs.phase == Phase.IDLE:
            if e.data["player"] == 1:
                gs.forfeit_frame(gs.player1)
            else:
                gs.forfeit_frame(gs.player2)

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
        fs = self.gameState.current_frame
        ctx = fs.ctx
        out: List[str] = []

        foul = False
        foul_points = 4

        target = fs.activePlayer.target
        legal_targets = LEGAL_TARGETS[target]

        # --- basic fouls ---
        if ctx.cue_potted:
            foul = True
            foul_points = foul_value(BallType.CUE)

        if ctx.first_contact is None:
            foul = True
        else:
            cue, first = ctx.first_contact

            if cue != BallType.CUE:
                foul = True
                foul_points = foul_value(cue)

            if first not in legal_targets:
                foul = True
                foul_points = foul_value(first)

        # --- potting fouls ---
        if ctx.potted:
            if any(b not in legal_targets for b in ctx.potted):
                foul = True
                foul_points = foul_value(*ctx.potted)

            if target != "RED" and len(ctx.potted) > 1:
                foul = True
                foul_points = foul_value(*ctx.potted)

        # --- foul resolution ---
        if foul:
            fs.opponent.score += foul_points
            out.append(f"FOUL +{foul_points} to {fs.opponent.name}")
            fs.swap_players()
            return out

        # --- legal scoring ---
        gained = sum(BALL_VALUE[b] for b in ctx.potted)
        fs.activePlayer.score += gained
        out.append(f"LEGAL: {fs.activePlayer.name} +{gained} (score {fs.activePlayer.score})")

        if gained == 0:
            out.append(f"NO_POT: turn -> {fs.opponent.name}")
            fs.swap_players()
            return out

        if fs.colourClearance and fs.activePlayer.target != "COLOUR" and fs.tempBallOrder:
            fs.tempBallOrder.pop(0)

        # --- continue & update target ---
        just_potted_red = target == "RED"

        fs.get_next_target(just_potted_red)
        out.append(f"CONTINUE: {fs.activePlayer.name} keeps turn")

        return out

gs = GameState()
gs.start_frame()
engine = RuleEngine(gs)

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
    Event(time.time(), EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.RED}),
    Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.RED}),
    Event(time.time(), EventType.SHOT_END),

    Event(time.time(), EventType.NO_REDS_REMAINING),

    Event(time.time(), EventType.SHOT_START),
    Event(time.time(), EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.YELLOW}),
    Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.YELLOW}),
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

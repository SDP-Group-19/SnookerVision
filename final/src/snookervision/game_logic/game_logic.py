import logging
import time
from threading import Event

import paho.mqtt.publish as publish

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Set

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
    t: float | None
    type: EventType
    data: dict = field(default_factory=dict)

@dataclass
class ShotContext:
    first_contact: Optional[Tuple[BallType, BallType]] = None
    potted: List[BallType] = field(default_factory=list)
    cue_potted: bool = False

    # ===== Added for fuller rules support (no new classes / no new EventType needed) =====
    # If a ball is "forced off the table", emit EventType.BALL_POTTED with:
    #   {"ball": BallType.X, "off_table": True}
    cue_off_table: bool = False
    balls_off_table: List[BallType] = field(default_factory=list)

    # Optional nomination support when target == "COLOUR":
    # You can pass it via SHOT_START or FIRST_CONTACT:
    #   Event(..., SHOT_START, {"nominated": BallType.BLACK})
    nominated: Optional[BallType] = None

    # Optional "free ball" support (manual):
    #   Event(..., SHOT_START, {"free_ball": BallType.BLACK})
    free_ball: Optional[BallType] = None

class Phase(Enum):
    IDLE = auto()
    IN_SHOT = auto()

@dataclass
class Player:
    name: str
    gamesWon: int = 0
    framesWon: int = 0
    score: int = 0
    target: str = "RED"

def foul_value(*balls: BallType) -> int:
    return max(4, *(BALL_VALUE[b] for b in balls if b in BALL_VALUE))

def buzz_cue():
    try:
        publish.single(
            "team/vibrate",
            payload="buzz",
            hostname="broker.hivemq.com",
            connect_timeout=1,
        )
    except Exception:
        pass

def _is_colour(b: BallType) -> bool:
    return b in {
        BallType.YELLOW, BallType.GREEN, BallType.BROWN,
        BallType.BLUE, BallType.PINK, BallType.BLACK
    }

def _target_to_balltype(target: str) -> Optional[BallType]:
    if target == "RED":
        return BallType.RED
    if target == "COLOUR":
        return None
    try:
        return BallType[target]
    except KeyError:
        return None

@dataclass
class FrameState:
    activePlayer: Player = field()
    opponent: Player = field()
    tempBallOrder: List[str] = field(default_factory=lambda: BALL_ORDER.copy())
    colourClearance: bool = False
    phase: Phase = Phase.IDLE
    ctx: ShotContext = field(default_factory=ShotContext)

    # ===== Added: internal state so vision layer does NOT need NO_REDS_REMAINING =====
    # Canonical "ball on" for next stroke (RED / COLOUR / YELLOW...).
    ball_on: str = "RED"
    # Track remaining reds internally (default full frame = 15; set to 2 for your MVP tests).
    reds_left: int = 15
    reds_gone: bool = False
    # After last red is potted legally -> one final COLOUR shot required before clearance starts at YELLOW.
    final_colour_pending: bool = False

    def _sync_targets(self):
        # Keep player.target consistent with frame.ball_on.
        self.activePlayer.target = self.ball_on
        # Keep colourClearance consistent with ball_on.
        self.colourClearance = (self.ball_on in BALL_ORDER)

    def swap_players(self):
        self.activePlayer, self.opponent = self.opponent, self.activePlayer
        # Incoming player's target should be the current ball_on (can be COLOUR, not always RED).
        self._sync_targets()
        logger.debug(f"Next target: {self.activePlayer.target}")

    def get_next_target(self, just_potted_red: bool):
        # Backwards-compatible signature.
        # We now drive target from self.ball_on (updated by the rule engine).
        self._sync_targets()
        logger.debug(f"Next target: {self.activePlayer.target}")

@dataclass
class GameState:
    games: int = 1
    frames: int = 1
    player1: Player = field(default_factory=lambda: Player("Player1"))
    player2: Player = field(default_factory=lambda: Player("Player2"))
    firstTurn: Player = field(init=False)
    current_frame: FrameState | None = field(init=False, default=None)

    def __post_init__(self):
        self.firstTurn = self.player1

    def start_frame(self):
        # Reset per-frame basics
        self.player1.score = 0
        self.player2.score = 0
        self.player1.target = "RED"
        self.player2.target = "RED"
        self.current_frame = FrameState(
            activePlayer=self.firstTurn,
            opponent=self.player2 if self.firstTurn is self.player1 else self.player1
        )
        self.current_frame.ball_on = "RED"
        self.current_frame._sync_targets()

    def end_frame(self):
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

class RuleEngine:
    def __init__(self, game_state: GameState) -> None:
        self.gameState = game_state

    def on_event(self, e: Event) -> List[str]:
        gs = self.gameState
        fs = gs.current_frame
        outputs: List[str] = []
        if fs is None:
            return outputs

        if e.type == EventType.SHOT_START and fs.phase == Phase.IDLE:
            fs.phase = Phase.IN_SHOT
            fs.ctx = ShotContext()

            # Optional: nomination/free-ball can be passed here without changing the framework.
            if "nominated" in e.data:
                fs.ctx.nominated = e.data["nominated"]
            if "free_ball" in e.data:
                fs.ctx.free_ball = e.data["free_ball"]

            outputs.append("SHOT_START")

        elif e.type == EventType.NO_REDS_REMAINING and fs.phase == Phase.IDLE:
            # Compatibility only: we do NOT rely on vision to emit this.
            outputs.append("NO_REDS_REMAINING")

            fs.reds_left = 0
            fs.reds_gone = True

            # If we were still expecting RED, jump to clearance at YELLOW.
            if fs.ball_on == "RED":
                fs.final_colour_pending = False
                fs.tempBallOrder = BALL_ORDER.copy()
                fs.ball_on = fs.tempBallOrder[0]  # "YELLOW"
            # If we were already on COLOUR, treat it as the final colour before clearance.
            elif fs.ball_on == "COLOUR":
                fs.final_colour_pending = True

            fs._sync_targets()

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

                # Optional: nomination can also be passed here.
                if "nominated" in e.data and fs.ctx.nominated is None:
                    fs.ctx.nominated = e.data["nominated"]

                outputs.append(f"FIRST_CONTACT {fs.ctx.first_contact}")

        elif e.type == EventType.BALL_POTTED and fs.phase == Phase.IN_SHOT:
            b = e.data["ball"]
            off_table = bool(e.data.get("off_table", False))

            if off_table:
                if b == BallType.CUE:
                    fs.ctx.cue_off_table = True
                else:
                    fs.ctx.balls_off_table.append(b)
                # Keep output style unchanged for compatibility.
                outputs.append(f"BALL_POTTED {b.name}")
            else:
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
        assert fs is not None
        ctx = fs.ctx
        out: List[str] = []

        target = fs.ball_on  # canonical ball-on state

        # ------------------------------------------------------------
        # Update reds_left based on physical outcomes (regardless of foul).
        # Potted/off-table reds are removed (not replaced).
        # ------------------------------------------------------------
        potted_reds = sum(1 for b in ctx.potted if b == BallType.RED)
        off_reds = sum(1 for b in ctx.balls_off_table if b == BallType.RED)
        total_reds_removed = potted_reds + off_reds

        if total_reds_removed > 0:
            fs.reds_left = max(0, fs.reds_left - total_reds_removed)

        just_became_reds_gone = (not fs.reds_gone) and (fs.reds_left == 0)
        if fs.reds_left == 0:
            fs.reds_gone = True

        # ------------------------------------------------------------
        # Determine effective ball-on set for this stroke (nomination/free ball)
        # ------------------------------------------------------------
        free_ball = ctx.free_ball
        nominated = ctx.nominated

        if target == "COLOUR":
            # If nomination not provided, infer from first contact (typical CV approach).
            if nominated is None and ctx.first_contact is not None:
                _, first = ctx.first_contact
                if _is_colour(first):
                    nominated = first
            # If still None, infer from first colour potted.
            if nominated is None:
                for b in ctx.potted:
                    if _is_colour(b):
                        nominated = b
                        break

        if free_ball is not None:
            on_set: Set[BallType] = {free_ball}
        else:
            if target == "RED":
                on_set = {BallType.RED}
            elif target == "COLOUR":
                on_set = {nominated} if nominated is not None else set(LEGAL_TARGETS["COLOUR"])
            else:
                bt = _target_to_balltype(target)
                on_set = {bt} if bt is not None else set()

        # Keep inferred nomination for later checks / foul points
        ctx.nominated = nominated

        # ------------------------------------------------------------
        # FOUL detection + foul points
        # Foul points = max(4, values of involved balls: ball on, first hit, potted/off-table, cue)
        # ------------------------------------------------------------
        foul = False
        involved: List[BallType] = []

        # Add "ball on" (if known) for foul points
        if target == "RED":
            involved.append(BallType.RED)
        elif target in BALL_ORDER:
            bt = _target_to_balltype(target)
            if bt:
                involved.append(bt)
        elif target == "COLOUR" and nominated is not None:
            involved.append(nominated)

        # Cue ball in-off / off table
        if ctx.cue_potted or ctx.cue_off_table:
            foul = True
            involved.append(BallType.CUE)

        # First contact validity
        if ctx.first_contact is None:
            foul = True
        else:
            cue, first = ctx.first_contact
            if cue != BallType.CUE:
                foul = True
                involved.append(cue)

            if first not in on_set:
                foul = True
                involved.append(first)

        # Pots / off-table legality
        object_potted = list(ctx.potted)          # excludes cue
        object_off = list(ctx.balls_off_table)    # excludes cue
        involved.extend(object_potted)
        involved.extend(object_off)

        # Any object ball forced off the table is a foul.
        if object_off:
            foul = True

        def _is_legal_pot_sequence() -> bool:
            # Returns True if pots/off-table are legal given current "ball on" and free-ball
            all_objs = object_potted + object_off

            if free_ball is not None:
                # Simplified free-ball support:
                # - If original target is RED: allow free ball + reds only.
                # - Otherwise: allow only the free ball.
                if target == "RED":
                    for b in all_objs:
                        if b != BallType.RED and b != free_ball:
                            return False
                    return True
                else:
                    for b in all_objs:
                        if b != free_ball:
                            return False
                    return True

            if target == "RED":
                # Only reds may be potted/off-table; multiple reds are allowed.
                return all(b == BallType.RED for b in all_objs)

            if target == "COLOUR":
                # Only ONE colour ball may be potted/off-table; reds are illegal here.
                if not all_objs:
                    return True
                if any(b == BallType.RED for b in all_objs):
                    return False
                if any(not _is_colour(b) for b in all_objs):
                    return False
                if len(all_objs) != 1:
                    return False
                if nominated is not None and all_objs[0] != nominated:
                    return False
                return True

            # Clearance (specific colour): only ONE object ball, and it must be the target colour.
            bt = _target_to_balltype(target)
            if not all_objs:
                return True
            if len(all_objs) != 1:
                return False
            return bt is not None and all_objs[0] == bt

        if not _is_legal_pot_sequence():
            foul = True

        foul_points = foul_value(*involved) if involved else 4

        # ------------------------------------------------------------
        # FOUL branch
        # ------------------------------------------------------------
        if foul:
            fs.opponent.score += foul_points
            out.append(f"FOUL +{foul_points} to {fs.opponent.name}")
            buzz_cue()

            # If the LAST red disappears in THIS foul stroke -> jump straight to clearance at YELLOW.
            # (Your earlier requirement)
            if just_became_reds_gone:
                fs.final_colour_pending = False
                fs.tempBallOrder = BALL_ORDER.copy()
                fs.ball_on = fs.tempBallOrder[0]  # "YELLOW"
            else:
                # Revert to RED if we were on COLOUR during reds stage
                if not fs.reds_gone and target == "COLOUR":
                    fs.ball_on = "RED"
                # Final colour after last red missed/fouled → start clearance at YELLOW
                elif fs.reds_gone and fs.final_colour_pending and target == "COLOUR":
                    fs.final_colour_pending = False
                    fs.tempBallOrder = BALL_ORDER.copy()
                    fs.ball_on = fs.tempBallOrder[0]  # "YELLOW"
                else:
                    fs.ball_on = target

            fs._sync_targets()
            fs.swap_players()
            return out

        # ------------------------------------------------------------
        # LEGAL scoring
        # Note: colours potted during reds stage are re-spotted (state unchanged).
        # ------------------------------------------------------------
        gained = sum(BALL_VALUE[b] for b in ctx.potted)

        # Optional free-ball scoring:
        # If nominated free ball is potted legally, it scores as if it were the ball on.
        if free_ball is not None and free_ball in ctx.potted:
            gained -= BALL_VALUE[free_ball]
            if target == "RED":
                gained += BALL_VALUE[BallType.RED]
            elif target in BALL_ORDER:
                bt = _target_to_balltype(target)
                gained += BALL_VALUE.get(bt, 0)
            elif target == "COLOUR" and nominated is not None:
                gained += BALL_VALUE[nominated]

        fs.activePlayer.score += gained
        out.append(f"LEGAL: {fs.activePlayer.name} +{gained} (score {fs.activePlayer.score})")

        # ------------------------------------------------------------
        # No pot: turn passes, ball on stays the same
        # ------------------------------------------------------------
        if gained == 0:
            out.append(f"NO_POT: turn -> {fs.opponent.name}")
            # Revert to RED if we were on COLOUR during reds stage
            if not fs.reds_gone and target == "COLOUR":
                fs.ball_on = "RED"
            # Final colour after last red missed → start clearance at YELLOW
            elif fs.reds_gone and fs.final_colour_pending and target == "COLOUR":
                fs.final_colour_pending = False
                fs.tempBallOrder = BALL_ORDER.copy()
                fs.ball_on = fs.tempBallOrder[0]  # "YELLOW"
            else:
                fs.ball_on = target
            fs._sync_targets()
            fs.swap_players()
            return out

        # ------------------------------------------------------------
        # LEGAL state transitions (ball_on update)
        # ------------------------------------------------------------

        # (A) Last red potted legally -> next is COLOUR, final colour pending before clearance.
        if just_became_reds_gone and target == "RED":
            fs.final_colour_pending = True
            fs.ball_on = "COLOUR"
            fs._sync_targets()
            fs.get_next_target(just_potted_red=True)
            out.append(f"CONTINUE: {fs.activePlayer.name} keeps turn")
            return out

        # (B) Reds stage: RED -> COLOUR after pot
        if (not fs.reds_gone) and target == "RED":
            fs.ball_on = "COLOUR"
            fs._sync_targets()
            fs.get_next_target(just_potted_red=True)
            out.append(f"CONTINUE: {fs.activePlayer.name} keeps turn")
            return out

        # (C) Reds stage: COLOUR -> RED after pot
        if (not fs.reds_gone) and target == "COLOUR":
            fs.ball_on = "RED"
            fs._sync_targets()
            fs.get_next_target(just_potted_red=False)
            out.append(f"CONTINUE: {fs.activePlayer.name} keeps turn")
            return out

        # (D) Final colour after last red -> start clearance at YELLOW
        if fs.reds_gone and fs.final_colour_pending and target == "COLOUR":
            fs.final_colour_pending = False
            fs.tempBallOrder = BALL_ORDER.copy()
            fs.ball_on = fs.tempBallOrder[0]  # "YELLOW"
            fs._sync_targets()
            fs.get_next_target(just_potted_red=False)
            out.append(f"CONTINUE: {fs.activePlayer.name} keeps turn")
            return out

        # (E) Clearance: pop the expected colour and move to next
        if target in BALL_ORDER and fs.tempBallOrder:
            if fs.tempBallOrder[0] == target:
                fs.tempBallOrder.pop(0)

            if not fs.tempBallOrder:
                # BLACK potted legally -> frame ends
                out.append("NO_BALLS_REMAINING")
                self.gameState.end_frame()
                return out

            fs.ball_on = fs.tempBallOrder[0]
            fs._sync_targets()
            fs.get_next_target(just_potted_red=False)
            out.append(f"CONTINUE: {fs.activePlayer.name} keeps turn")
            return out

        # Fallback: keep current ball on
        fs.ball_on = target
        fs._sync_targets()
        fs.get_next_target(just_potted_red=False)
        out.append(f"CONTINUE: {fs.activePlayer.name} keeps turn")
        return out

if __name__ == "__main__":
    gs = GameState()
    gs.start_frame()
    engine = RuleEngine(gs)

    # MVP test: 2 reds
    gs.current_frame.reds_left = 2

    events = [
        Event(time.time(), EventType.SHOT_START),
        Event(time.time(), EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.RED}),
        Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.RED}),
        Event(time.time(), EventType.SHOT_END),

        # COLOUR shot (no pot) -> turn passes, next target remains COLOUR
        Event(time.time(), EventType.SHOT_START),
        Event(time.time(), EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.BLUE}),
        Event(time.time(), EventType.SHOT_END),

        # Opponent COLOUR shot: pot black -> next RED
        Event(time.time(), EventType.SHOT_START),
        Event(time.time(), EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.BLACK}),
        Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.BLACK}),
        Event(time.time(), EventType.SHOT_END),

        # Opponent RED shot: pot last red + cue in-off (foul) -> jump to YELLOW for incoming player
        Event(time.time(), EventType.SHOT_START),
        Event(time.time(), EventType.FIRST_CONTACT, {"a": BallType.CUE, "b": BallType.RED}),
        Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.RED}),
        Event(time.time(), EventType.BALL_POTTED, {"ball": BallType.CUE}),
        Event(time.time(), EventType.SHOT_END),
    ]

    for e in events:
        for msg in engine.on_event(e):
            print(msg)
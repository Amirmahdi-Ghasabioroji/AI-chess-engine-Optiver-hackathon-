"""AI Chessathon agent: numba PVS with classical + NNUE mix. See ROADMAP.md."""

from __future__ import annotations

import traceback

import chess

from engine.nnue import warmup as warmup_nnue
from engine.play import Engine, pick_move

warmup_nnue()
_engine = Engine()
_engine.warmup()


def get_move(fen: str, time_left_ms: int) -> str:
    try:
        return pick_move(fen, int(time_left_ms), _engine)
    except Exception:
        traceback.print_exc()
        try:
            return next(iter(chess.Board(fen).legal_moves)).uci()
        except Exception:
            return "a2a3"

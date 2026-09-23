"""The bitboard evaluation must agree with evaluate.py, and SEE must be sane.

evaluate.py is the reference the current engine plays with, so any disagreement
is a porting bug rather than a tuning choice.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess
import numpy as np

from engine import bb
from engine import bb_eval
from engine import evaluate
from engine.bbpos import from_board, move_from_chess, move_to_uci


def check_evaluate(seed: int = 3, games: int = 80) -> bool:
    rng = random.Random(seed)
    checked = 0
    worst = 0
    pc = bb_eval.new_pawn_cache()
    for _ in range(games):
        board = chess.Board()
        for _ in range(140):
            if board.is_game_over(claim_draw=False):
                break
            bbs, st, _ = from_board(board)
            ours = int(bb_eval.evaluate_classical(bbs, st, pc))
            theirs = evaluate.evaluate(board)
            if ours != theirs:
                print(f"FAIL eval {board.fen()}: bitboard {ours} vs reference {theirs}")
                return False
            worst = max(worst, abs(ours))
            checked += 1
            board.push(rng.choice(list(board.legal_moves)))
    print(f"ok  evaluation matches evaluate.py over {checked} positions (max |score| {worst})")
    return True


def _value(piece_type: int | None) -> int:
    return int(bb_eval.SEE_VALUE[piece_type - 1]) if piece_type else 0


def _swap(board: chess.Board, target: int, side: chess.Color) -> int:
    """Best gain for `side`, who may capture on `target` or decline.

    Deliberately pin-blind, like every engine's SEE: `attackers` ignores pins, and
    rebuilding the board each ply exposes x-rays for free.
    """
    attackers = board.attackers(side, target)
    if not attackers:
        return 0
    square = min(attackers, key=lambda s: _value(board.piece_type_at(s)))
    if board.piece_type_at(square) == chess.KING and board.attackers(not side, target):
        return 0
    gain = _value(board.piece_type_at(target))
    nxt = board.copy(stack=False)
    nxt.set_piece_at(target, nxt.piece_at(square))
    nxt.remove_piece_at(square)
    return max(0, gain - _swap(nxt, target, not side))


def _reference_see(board: chess.Board, move: chess.Move) -> int:
    """Independent SEE oracle: play the capture, then swap pieces off by hand."""
    target = move.to_square
    side = board.turn
    if board.is_en_passant(move):
        captured = _value(chess.PAWN)
    else:
        captured = _value(board.piece_type_at(target))

    after = board.copy(stack=False)
    if board.is_en_passant(move):
        after.remove_piece_at(target - 8 if side == chess.WHITE else target + 8)
    after.set_piece_at(target, after.piece_at(move.from_square))
    after.remove_piece_at(move.from_square)
    return captured - _swap(after, target, not side)


def check_see(seed: int = 5, games: int = 40) -> bool:
    rng = random.Random(seed)
    checked = 0
    for _ in range(games):
        board = chess.Board()
        for _ in range(140):
            if board.is_game_over(claim_draw=False):
                break
            bbs, _, mb = from_board(board)
            for move in board.legal_moves:
                if not board.is_capture(move) or move.promotion:
                    continue
                mv = move_from_chess(move, board)
                ours = int(bb_eval.see(bbs, mb, np.int64(mv)))
                theirs = _reference_see(board, move)
                if ours != theirs:
                    print(f"FAIL see {move_to_uci(mv)} in {board.fen()}: {ours} vs {theirs}")
                    return False
                checked += 1
            board.push(rng.choice(list(board.legal_moves)))
    print(f"ok  SEE matches a recursive oracle over {checked} captures")
    return True


def main() -> int:
    bb.init()
    ok = check_evaluate() and check_see()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Smoke checks: legality, mate-in-one, time bound, book probe."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess

import agent
from engine import book
from engine import evaluate


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {msg}")
    print("ok:", msg)


def test_startpos_legal() -> None:
    board = chess.Board()
    t0 = time.perf_counter()
    uci = agent.get_move(board.fen(), 2000)
    dt = time.perf_counter() - t0
    move = chess.Move.from_uci(uci)
    _assert(move in board.legal_moves, f"startpos move {uci} legal")
    _assert(dt < 3.5, f"startpos 2s budget finished in {dt:.2f}s")


def test_mate_in_one() -> None:
    # White mates with Qh7 or similar — classic back rank / scholar-style
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1")
    uci = agent.get_move(board.fen(), 3000)
    board.push_uci(uci)
    _assert(board.is_checkmate(), f"rook mate in one, played {uci}")


def test_hanging_queen() -> None:
    # Queen on a5, rook on a1: same file, unprotected.
    board = chess.Board("4k3/8/8/q7/8/8/8/R3K3 w Q - 0 1")
    uci = agent.get_move(board.fen(), 2500)
    _assert(uci == "a1a5", f"expected Rxa5, got {uci}")


def test_eval_symmetric() -> None:
    start = evaluate.evaluate(chess.Board())
    _assert(abs(start) < 40, f"start eval near 0, got {start}")


def test_book_start() -> None:
    m = book.probe(chess.Board())
    _assert(m is not None, "startpos is in the book")
    print("   (book", m.uci(), ")")


def main() -> int:
    test_eval_symmetric()
    test_book_start()
    test_startpos_legal()
    # Fresh process state: mate tests after startpos still share TT/game tracker.
    # Force a new game via an unreachable fen transition — get_move handles it.
    test_mate_in_one()
    test_hanging_queen()
    print("all smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

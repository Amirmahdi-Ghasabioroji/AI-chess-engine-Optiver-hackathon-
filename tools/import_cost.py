"""Reproduce the agent's import path step by step, with timings.

Import time is a hard failure mode: the platform allows 60s before the clock
starts, and every game pays it again.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def main() -> None:
    t0 = time.perf_counter()
    marks: list[tuple[str, float]] = []

    def mark(name: str) -> None:
        nonlocal t0
        now = time.perf_counter()
        marks.append((name, now - t0))
        t0 = now

    import chess  # noqa: F401

    mark("import chess")

    from engine import bb

    mark("import bb")

    from engine import bb_eval  # noqa: F401
    from engine import bb_search
    from engine import nnue

    mark("import bb_eval + bb_search")

    from engine import book  # noqa: F401
    from engine.bbpos import from_board

    mark("import book + bbpos")

    bb.init()
    mark("bb.init (magics)")
    nnue.load()
    mark("nnue.load")

    tt, ord32, rep, info = bb_search.new_tables()
    pc = bb_eval.new_pawn_cache()
    hist, moves, scores = bb.new_buffers()
    mark("allocate tables")

    board = chess.Board()
    bbs, st, mb = from_board(board)
    mark("from_board (compiles compute_hash)")

    rep[0] = st[bb.ST_HASH]
    info[bb_search.I_REP_BASE] = 0
    deadline = time.perf_counter() + 120.0
    bb_search.search(
        bbs, st, mb, hist, moves, scores, tt, ord32, pc, rep, info,
        np.int64(6), deadline, deadline,
    )
    mark("first search (whole JIT call graph)")

    deadline = time.perf_counter() + 2.0
    bb_search.search(
        bbs, st, mb, hist, moves, scores, tt, ord32, pc, rep, info,
        np.int64(63), deadline, deadline,
    )
    mark("2s warm search")

    total = sum(dt for _, dt in marks)
    for name, dt in marks:
        print(f"{dt:7.2f}s  {name}")
    print(f"{total:7.2f}s  TOTAL")
    print(
        f"        warm rate {int(info[bb_search.I_NODES]) / 2000:.0f} knps, "
        f"depth {int(info[bb_search.I_DEPTH_DONE])}"
    )


if __name__ == "__main__":
    main()

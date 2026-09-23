"""Correctness gate for v7.1.1.1.

    python experiments/test_v71111_nnue
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V71111 = ROOT / "local" / "engine_snapshots" / "v7.1.1.1"
sys.path.insert(0, str(V71111))

import chess
import numpy as np

from engine import bb
from engine import bb_eval
from engine import bb_search
from engine.bbpos import from_board
from engine.nnue import blend_residual, evaluate_nnue_bbs

START = chess.STARTING_FEN
CASTLE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
EP = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
RUY = "r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w - - 0 9"


def perft(fen: str, depth: int) -> int:
    board = chess.Board(fen)
    bbs, st, mb = from_board(board)
    hist, moves, _ = bb.new_buffers()
    return int(bb.perft(bbs, st, mb, hist, moves, np.int64(0), np.int64(depth)))


def check_mix_startpos() -> str | None:
    board = chess.Board()
    bbs, st, _mb = from_board(board)
    pc = bb_eval.new_pawn_cache()
    classical = int(bb_eval.evaluate_classical(bbs, st, pc))
    nnue = int(evaluate_nnue_bbs(bbs, st[bb.ST_SIDE]))
    mixed = int(bb_eval.evaluate(bbs, st, pc))
    rebuilt = int(blend_residual(bbs, st[bb.ST_SIDE], np.int64(classical)))
    if mixed != rebuilt:
        return f"eval {mixed} != blend {rebuilt}"
    if mixed != nnue:
        return f"eval {mixed} != 100% nnue {nnue} (C={classical})"
    stand_mid = int(bb_eval.evaluate_stand(bbs, st, pc, np.int64(-32000), np.int64(32000)))
    if stand_mid != mixed:
        return f"stand {stand_mid} != nnue {mixed}"
    return None


def nps_probe(fen: str, hard_s: float) -> tuple[int, int, int, int, float, float]:
    board = chess.Board(fen)
    bbs, st, mb = from_board(board)
    hist, moves, scores = bb.new_buffers()
    tt, ord32, rep, info = bb_search.new_tables()
    pc = bb_eval.new_pawn_cache()
    rep[0] = st[bb.ST_HASH]
    info[bb_search.I_REP_BASE] = 0
    deadline = time.perf_counter() + hard_s
    t0 = time.perf_counter()
    packed = int(
        bb_search.search(
            bbs, st, mb, hist, moves, scores, tt, ord32, pc, rep, info,
            np.int64(63), deadline, deadline,
        )
    )
    dt = time.perf_counter() - t0
    nodes = int(info[bb_search.I_NODES])
    depth = int(info[bb_search.I_DEPTH_DONE])
    sel = int(info[bb_search.I_SELDEPTH])
    nps = nodes / dt if dt > 0 else 0.0
    return packed, depth, sel, nodes, nps, dt


def main() -> int:
    bb.init()
    from engine.nnue import warmup

    warmup()
    failed = 0
    for name, fen, depth, expect in (
        ("start d3", START, 3, 8902),
        ("kiwipete d3", CASTLE, 3, 97862),
        ("ep d4", EP, 4, 43238),
    ):
        got = perft(fen, depth)
        ok = got == expect
        print(f"  perft {name}: {got} {'ok' if ok else f'FAIL expected {expect}'}")
        failed += 0 if ok else 1

    err = check_mix_startpos()
    if err:
        print(f"  mix: FAIL {err}")
        failed += 1
    else:
        print("  mix startpos: ok")

    hist, moves, scores = bb.new_buffers()
    tt, ord32, rep, info = bb_search.new_tables()
    pc = bb_eval.new_pawn_cache()
    board = chess.Board()
    bbs, st, mb = from_board(board)
    deadline = time.perf_counter() + 8.0
    packed = int(
        bb_search.search(
            bbs, st, mb, hist, moves, scores, tt, ord32, pc, rep, info,
            np.int64(9), deadline, deadline,
        )
    )
    if packed == 0:
        print("  search: FAIL no move")
        failed += 1
    else:
        print(f"  search move={packed} depth={int(info[bb_search.I_DEPTH_DONE])}")

    packed, depth, sel, nodes, nps, dt = nps_probe(START, 2.0)
    print(
        f"  nps startpos 2s: d{depth} sel{sel} n{nodes} {nps:.0f} nps "
        f"({dt:.2f}s) move={packed}"
    )
    packed, depth, sel, nodes, nps, dt = nps_probe(RUY, 2.0)
    print(
        f"  nps ruy 2s: d{depth} sel{sel} n{nodes} {nps:.0f} nps "
        f"({dt:.2f}s) move={packed}"
    )
    print("FAIL" if failed else "ok")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Correctness gate for v7.1.1.2 incremental mix NNUE.

    python experiments/test_v7112_nnue
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V7112 = ROOT / "local" / "engine_snapshots" / "v7.1.1.2"
sys.path.insert(0, str(V7112))

import chess
import numpy as np

from engine import bb
from engine import bb_eval
from engine import bb_search
from engine.bbpos import from_board, move_from_chess
from engine.nnue import (
    acc_make,
    acc_rebuild,
    acc_unmake,
    evaluate_acc,
    evaluate_nnue_bbs,
    mix_nnue,
    new_accumulators,
)

START = chess.STARTING_FEN
CASTLE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
EP = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"
PROMO = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"


def _acc_close(a: np.ndarray, b: np.ndarray, tol: float = 1e-4) -> bool:
    return bool(np.max(np.abs(a - b)) <= tol)


def check_rebuild_equals_incremental(fen: str, plies: int, rng: random.Random) -> str | None:
    board = chess.Board(fen)
    bbs, st, mb = from_board(board)
    hist, moves, _ = bb.new_buffers()
    acc = new_accumulators()
    acc_rebuild(acc, bbs)
    ref = new_accumulators()
    for _ in range(plies):
        if board.is_game_over(claim_draw=False):
            break
        legal = list(board.legal_moves)
        if not legal:
            break
        move = rng.choice(legal)
        packed = np.int64(move_from_chess(move, board))
        acc_make(acc, mb, st, packed)
        bb.make_move(bbs, st, mb, hist, np.int64(0), packed)
        board.push(move)
        acc_rebuild(ref, bbs)
        if not _acc_close(acc, ref):
            return f"acc drift after {move.uci()} in {board.fen()}"
        full = int(evaluate_nnue_bbs(bbs, st[bb.ST_SIDE]))
        incr = int(evaluate_acc(acc, st[bb.ST_SIDE]))
        if full != incr:
            return f"eval {full} vs incr {incr} after {move.uci()} {board.fen()}"
        bb.unmake_move(bbs, st, mb, hist, np.int64(0), packed)
        acc_unmake(acc, mb, st, packed)
        acc_rebuild(ref, bbs)
        if not _acc_close(acc, ref):
            return f"unmake drift {move.uci()} {board.fen()}"
        acc_make(acc, mb, st, packed)
        bb.make_move(bbs, st, mb, hist, np.int64(0), packed)
    return None


def perft(fen: str, depth: int) -> int:
    board = chess.Board(fen)
    bbs, st, mb = from_board(board)
    hist, moves, _ = bb.new_buffers()
    return int(bb.perft(bbs, st, mb, hist, moves, np.int64(0), np.int64(depth)))


def check_mix_startpos() -> str | None:
    board = chess.Board()
    bbs, st, mb = from_board(board)
    pc = bb_eval.new_pawn_cache()
    acc = new_accumulators()
    acc_rebuild(acc, bbs)
    classical = int(bb_eval.evaluate_classical(bbs, st, pc))
    nnue = int(evaluate_acc(acc, st[bb.ST_SIDE]))
    mixed = int(bb_eval.evaluate(bbs, st, pc, acc))
    expect = int(mix_nnue(np.int64(classical), np.int64(nnue)))
    if mixed != expect:
        return f"mix {mixed} != {expect} (C={classical} N={nnue})"
    from engine.nnue import MIX

    if int(MIX[0]) != 1:
        return f"expected residual MIX flag, got {int(MIX[0])}"
    half = classical + max(-400, min(400, nnue)) // 2
    if mixed != half:
        return f"50% residual mix {mixed} != C+R/2 {half}"
    return None


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

    rng = random.Random(0)
    for name, fen in (("start", START), ("castle", CASTLE), ("ep", EP), ("promo", PROMO)):
        err = check_rebuild_equals_incremental(fen, 40, rng)
        if err:
            print(f"  acc {name}: FAIL {err}")
            failed += 1
        else:
            print(f"  acc {name}: ok")

    err = check_mix_startpos()
    if err:
        print(f"  mix: FAIL {err}")
        failed += 1
    else:
        print("  mix startpos: ok")

    hist, moves, scores = bb.new_buffers()
    tt, ord32, rep, info = bb_search.new_tables()
    pc = bb_eval.new_pawn_cache()
    acc = new_accumulators()
    board = chess.Board()
    bbs, st, mb = from_board(board)
    deadline = time.perf_counter() + 8.0
    t0 = time.perf_counter()
    packed = int(
        bb_search.search(
            bbs, st, mb, hist, moves, scores, tt, ord32, pc, acc, rep, info,
            np.int64(9), deadline, deadline,
        )
    )
    elapsed = time.perf_counter() - t0
    nodes = int(info[bb_search.I_NODES])
    nps = int(nodes / elapsed) if elapsed > 0 else 0
    print(
        f"  search d{int(info[bb_search.I_DEPTH_DONE])} "
        f"nodes={nodes} nps={nps} move={packed}"
    )
    if packed == 0:
        print("  search: FAIL no move")
        failed += 1
    print("FAIL" if failed else "ok")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

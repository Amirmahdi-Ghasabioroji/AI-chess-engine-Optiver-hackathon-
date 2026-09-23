"""Correctness gate for v2.2_NNUE incremental mix NNUE.

    python experiments/test_v22_nnue
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V22 = ROOT / "local" / "engine_snapshots" / "v2.2_NNUE"
sys.path.insert(0, str(V22))

import chess
import numpy as np

import bb
import bb_eval
import bb_search
from bbpos import from_board, move_from_chess
from nnue import (
    MIX,
    acc_make,
    acc_rebuild,
    acc_unmake,
    evaluate_acc,
    evaluate_nnue_bbs,
    mix_nnue,
    new_accumulators,
    warmup,
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

    def walk(d: int, ply: int) -> int:
        if d == 0:
            return 1
        n = 0
        base = ply * bb.MAX_MOVES
        end = bb.gen_moves(bbs, st, moves, np.int64(base), False)
        for i in range(base, end):
            mv = np.int64(moves[i])
            side = st[bb.ST_SIDE]
            bb.make_move(bbs, st, mb, hist, np.int64(ply), mv)
            if not bb.attacked(
                bbs, bb.king_square(bbs, side), 1 - side, bbs[bb.OCC_W] | bbs[bb.OCC_B]
            ):
                n += walk(d - 1, ply + 1)
            bb.unmake_move(bbs, st, mb, hist, np.int64(ply), mv)
        return n

    return walk(depth, 0)


def mix_startpos() -> str | None:
    board = chess.Board()
    bbs, st, mb = from_board(board)
    pc = bb_eval.new_pawn_cache()
    acc = new_accumulators()
    acc_rebuild(acc, bbs)
    mixed = int(bb_eval.evaluate(bbs, st, pc, acc))
    classical = int(bb_eval.evaluate_classical(bbs, st, pc))
    nn = int(evaluate_acc(acc, st[bb.ST_SIDE]))
    hang = int(bb_eval._hanging_material(bbs))
    if int(st[bb.ST_SIDE]) != 0:
        hang = -hang
    expect = int(mix_nnue(np.int64(classical), np.int64(nn))) + hang // 2
    if mixed != expect:
        return f"mix {mixed} != {expect} (C={classical} N={nn} hang={hang})"
    if MIX[0] != 0:
        return "weights loaded as residual; v2.2_NNUE is absolute"
    return None


def main() -> int:
    bb.init()
    warmup()
    rng = random.Random(1)
    checks = [
        ("perft start d3", perft(START, 3) == 8902),
        ("perft kiwipete d3", perft(CASTLE, 3) == 97862),
        ("perft ep d4", perft(EP, 4) == 43238),
        ("acc start", check_rebuild_equals_incremental(START, 40, rng) is None),
        ("acc castle", check_rebuild_equals_incremental(CASTLE, 24, rng) is None),
        ("acc ep", check_rebuild_equals_incremental(EP, 24, rng) is None),
        ("acc promo", check_rebuild_equals_incremental(PROMO, 16, rng) is None),
        ("mix startpos", mix_startpos() is None),
    ]
    failed = False
    for name, ok in checks:
        print(f"  {name}: {'ok' if ok else 'FAIL'}")
        if not ok:
            failed = True
            extra = check_rebuild_equals_incremental(START, 8, random.Random(1))
            if extra:
                print("   ", extra)

    tt, ord32, rep, info = bb_search.new_tables()
    pc = bb_eval.new_pawn_cache()
    hist, moves, scores = bb.new_buffers()
    acc = new_accumulators()
    board = chess.Board()
    bbs, st, mb = from_board(board)
    rep[0] = st[bb.ST_HASH]
    info[bb_search.I_REP_BASE] = 0
    deadline = time.perf_counter() + 8.0
    packed = bb_search.search(
        bbs, st, mb, hist, moves, scores, tt, ord32, pc, acc, rep, info,
        np.int64(1), deadline, deadline,
    )
    print(f"  search move={int(packed)} depth={int(info[bb_search.I_DEPTH_DONE])}")
    if packed == 0:
        print("  search: FAIL")
        failed = True
    else:
        print("ok")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

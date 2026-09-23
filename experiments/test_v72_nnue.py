"""Correctness gate for v7.2 incremental mix NNUE.

    python experiments/test_v72_nnue
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V72 = ROOT / "local" / "engine_snapshots" / "v7.2"
sys.path.insert(0, str(V72))

import chess
import numpy as np

from engine import bb
from engine import bb_eval
from engine import bb_search
from engine.bbpos import from_board, move_from_chess
from engine.nnue import (
    QUANT_Q1,
    acc_make,
    acc_rebuild,
    acc_rebuild_quant,
    acc_unmake,
    blend_scores,
    evaluate_acc,
    evaluate_acc_quant,
    evaluate_nnue_bbs,
    load_nnue,
    mix_cap,
    new_accumulators,
    new_accumulators_i32,
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


def check_specials() -> list[str]:
    fails: list[str] = []
    rng = random.Random(3)
    for fen, n in ((START, 40), (CASTLE, 24), (EP, 20), (PROMO, 16)):
        err = check_rebuild_equals_incremental(fen, n, rng)
        if err:
            fails.append(err)
    return fails


def check_mix_formula() -> str | None:
    board = chess.Board()
    bbs, st, mb = from_board(board)
    pc = bb_eval.new_pawn_cache()
    acc = new_accumulators()
    acc_rebuild(acc, bbs)
    classical = int(bb_eval.evaluate_classical(bbs, st, pc))
    nnue = int(evaluate_acc(acc, st[bb.ST_SIDE]))
    mixed = int(bb_eval.evaluate(bbs, st, pc, acc, np.int64(-32000), np.int64(32000)))
    expect = int(blend_scores(np.int64(classical), np.int64(nnue), np.int64(24)))
    if mixed != expect:
        return f"mix {mixed} != blend {expect} (C={classical} N={nnue})"
    cap = int(mix_cap())
    if cap != 200:
        return f"mix_cap {cap} want 200"
    return None


def check_quant(n_games: int = 12) -> tuple[float, float]:
    rng = random.Random(5)
    errs: list[int] = []
    for _ in range(n_games):
        board = chess.Board()
        for _ply in range(30):
            if board.is_game_over(claim_draw=False):
                break
            bbs, st, _ = from_board(board)
            acc = new_accumulators()
            acc_rebuild(acc, bbs)
            qi = new_accumulators_i32()
            acc_rebuild_quant(qi, bbs)
            full = int(evaluate_acc(acc, st[bb.ST_SIDE]))
            quant = int(evaluate_acc_quant(qi, st[bb.ST_SIDE]))
            errs.append(abs(full - quant))
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
    arr = np.array(errs, dtype=np.float64)
    return float(arr.mean()), float(arr.max()) if len(arr) else (0.0, 0.0)


def check_perft() -> bool:
    ok = True
    cases = (
        (START, 3, 8902),
        (CASTLE, 3, 97862),
        (EP, 4, 43238),
    )
    hist, moves, _ = bb.new_buffers()
    for fen, depth, want in cases:
        bbs, st, mb = from_board(chess.Board(fen))
        got = int(bb.perft(bbs, st, mb, hist, moves, np.int64(0), np.int64(depth)))
        mark = "ok " if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  {mark} perft {depth} {got} want {want}  {fen[:40]}")
    return ok


def nps_probe() -> None:
    board = chess.Board()
    bbs, st, mb = from_board(board)
    tt, ord32, rep, info = bb_search.new_tables()
    pc = bb_eval.new_pawn_cache()
    acc = new_accumulators()
    hist, moves, scores = bb.new_buffers()
    rep[0] = st[bb.ST_HASH]
    compile_deadline = time.perf_counter() + 90.0
    bb_search.search(
        bbs, st, mb, hist, moves, scores, tt, ord32, pc, acc, rep, info,
        np.int64(4), compile_deadline, compile_deadline,
    )
    tt[:, :] = 0
    ord32[:] = 0
    info[:] = 0
    bbs, st, mb = from_board(board)
    acc_rebuild(acc, bbs)
    rep[0] = st[bb.ST_HASH]
    deadline = time.perf_counter() + 2.0
    t0 = time.perf_counter()
    packed = bb_search.search(
        bbs, st, mb, hist, moves, scores, tt, ord32, pc, acc, rep, info,
        np.int64(12), deadline, deadline,
    )
    dt = time.perf_counter() - t0
    nodes = int(info[bb_search.I_NODES])
    nps = nodes / dt if dt > 0 else 0.0
    print(
        f"  search depth {int(info[bb_search.I_DEPTH_DONE])} move {packed} "
        f"nodes {nodes} {nps:.0f} nps evals {int(info[bb_search.I_EVAL])} "
        f"tt {int(info[bb_search.I_TT_HIT])}/{int(info[bb_search.I_TT_PROBE])}"
    )


def main() -> int:
    load_nnue()
    bb.init()
    print("perft")
    perft_ok = check_perft()
    print("accumulator")
    fails = check_specials()
    for fail in fails:
        print(f"  FAIL {fail}")
    if not fails:
        print("  ok incremental == rebuild on start/castle/ep/promo walks")
    mix_err = check_mix_formula()
    if mix_err:
        print(f"  FAIL mix {mix_err}")
        fails.append(mix_err)
    else:
        print("  ok mix formula startpos")
    mae, mx = check_quant()
    print(f"  quant q1={int(QUANT_Q1)} mae={mae:.2f} max_err={mx:.0f} cp")
    print("search probe")
    nps_probe()
    return 0 if perft_ok and not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())

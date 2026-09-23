"""Correctness and diagnostic gate for v2.3_NNUE.

    python experiments/test_v23_nnue
"""

from __future__ import annotations

import hashlib
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V23 = ROOT / "local" / "engine_snapshots" / "v2.3_NNUE"
sys.path.insert(0, str(V23))

import chess
import numpy as np

import bb
import bb_eval
import bb_search
from bbpos import from_board, move_from_chess, move_to_uci
from nnue import (
    CORR_CLIP,
    EVAL_MODE,
    MIX,
    NNUE_PCT,
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
POS5 = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"
POS6 = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"

PERFT_CASES: tuple[tuple[str, int, int], ...] = (
    (START, 1, 20),
    (START, 2, 400),
    (START, 3, 8902),
    (START, 4, 197281),
    (CASTLE, 1, 48),
    (CASTLE, 2, 2039),
    (CASTLE, 3, 97862),
    (EP, 4, 43238),
    (PROMO, 3, 9467),
    (POS5, 3, 62379),
    (POS6, 3, 89890),
)

SEE_CASES: tuple[tuple[str, str, int], ...] = (
    ("4k3/8/8/4q3/3P4/8/8/4K3 w - - 0 1", "d4e5", 900),
    ("4k3/8/8/3p4/3Q4/8/8/4K3 w - - 0 1", "d4d5", 100),
    ("4k3/8/2p5/3p4/2Q5/8/8/4K3 w - - 0 1", "c4d5", -800),
    ("4k3/8/8/3p4/3R4/8/8/4K3 w - - 0 1", "d4d5", 100),
    ("4k3/8/8/3p4/2B5/8/8/4K3 w - - 0 1", "c4d5", 100),
    ("4k3/8/5n2/3p4/3R4/8/8/4K3 w - - 0 1", "d4d5", -400),
)

TACTICS: tuple[tuple[str, tuple[str, ...], str, float], ...] = (
    ("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", ("e1e8",), "back rank mate in 1", 1.5),
    ("4k3/8/8/q7/8/8/8/R3K3 w Q - 0 1", ("a1a5",), "hanging queen", 1.5),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1", ("f3f7",), "scholar mate", 1.5),
    ("6k1/8/6K1/8/8/8/8/4R3 w - - 0 1", ("e1e8",), "rook mate in 1", 1.0),
    ("8/8/8/8/8/8/6PP/5k1K w - - 0 1", ("h2h4", "g2g4", "h2h3", "g2g3"), "pawn race legal", 1.0),
)

ZUGZWANG = "8/8/8/8/8/4k3/8/4K3 w - - 0 1"


def _acc_close(a: np.ndarray, b: np.ndarray, tol: float = 1e-4) -> bool:
    return bool(np.max(np.abs(a - b)) <= tol)


def _tables() -> tuple:
    tt, ord32, rep, info = bb_search.new_tables()
    pc = bb_eval.new_pawn_cache()
    hist, moves, scores = bb.new_buffers()
    acc = new_accumulators()
    return tt, ord32, rep, info, pc, hist, moves, scores, acc


def _search(
    fen: str,
    seconds: float,
    max_depth: int = 63,
    node_limit: int = 0,
    *,
    tables: tuple | None = None,
) -> tuple[str, int, int, int]:
    board = chess.Board(fen)
    bbs, st, mb = from_board(board)
    if tables is None:
        tt, ord32, rep, info, pc, hist, moves, scores, acc = _tables()
    else:
        tt, ord32, rep, info, pc, hist, moves, scores, acc = tables
        tt[:, :] = 0
        info[:] = 0
        ord32[bb_search.KILLER_OFF : bb_search.KILLER_OFF + bb_search.KILLER_LEN] = 0
    rep[0] = st[bb.ST_HASH]
    info[bb_search.I_REP_BASE] = 0
    deadline = time.perf_counter() + seconds
    packed = int(
        bb_search.search(
            bbs, st, mb, hist, moves, scores, tt, ord32, pc, acc, rep, info,
            np.int64(max_depth), deadline, deadline, node_limit,
        )
    )
    uci = move_to_uci(packed) if packed else "0000"
    return (
        uci,
        int(info[bb_search.I_NODES]),
        int(info[bb_search.I_DEPTH_DONE]),
        int(info[bb_search.I_ROOT_SCORE]),
    )


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


def check_perft() -> list[tuple[str, bool]]:
    out: list[tuple[str, bool]] = []
    hist, moves, _ = bb.new_buffers()
    for fen, depth, expected in PERFT_CASES:
        bbs, st, mb = from_board(chess.Board(fen))
        got = int(bb.perft(bbs, st, mb, hist, moves, np.int64(0), np.int64(depth)))
        name = f"perft d{depth} {fen[:24]}"
        out.append((name, got == expected))
        if got != expected:
            print(f"    got {got} want {expected}")
    return out


def check_make_unmake_hash(seed: int = 11, games: int = 20) -> str | None:
    rng = random.Random(seed)
    hist, moves, _ = bb.new_buffers()
    for _ in range(games):
        board = chess.Board()
        for _ in range(80):
            if board.is_game_over(claim_draw=False):
                break
            bbs, st, mb = from_board(board)
            before = (bbs.copy(), st.copy(), mb.copy())
            end = int(bb.gen_moves(bbs, st, moves, np.int64(0), False))
            for i in range(end):
                mv = np.int64(int(moves[i]))
                if not bb.make_legal(bbs, st, mb, hist, np.int64(0), mv):
                    continue
                if int(bb.compute_hash(bbs, st)) != int(st[bb.ST_HASH]):
                    return f"hash after {move_to_uci(int(mv))} in {board.fen()}"
                bb.unmake_move(bbs, st, mb, hist, np.int64(0), mv)
                if (
                    not np.array_equal(bbs, before[0])
                    or not np.array_equal(st, before[1])
                    or not np.array_equal(mb, before[2])
                ):
                    return f"unmake {move_to_uci(int(mv))} {board.fen()}"
            board.push(rng.choice(list(board.legal_moves)))
    return None


def check_see() -> list[tuple[str, bool]]:
    out: list[tuple[str, bool]] = []
    for fen, uci, want in SEE_CASES:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            out.append((f"see {uci} legal", False))
            print(f"    illegal {uci} in {fen}")
            continue
        bbs, st, mb = from_board(board)
        packed = np.int64(move_from_chess(move, board))
        got = int(bb_eval.see(bbs, mb, packed))
        ok = abs(got - want) <= 50
        out.append((f"see {uci} {fen[:20]}", ok))
        if not ok:
            print(f"    see {uci}: got {got} want {want}")
    return out


def check_eval_modes() -> str | None:
    board = chess.Board()
    bbs, st, _mb = from_board(board)
    pc = bb_eval.new_pawn_cache()
    acc = new_accumulators()
    acc_rebuild(acc, bbs)
    classical = int(bb_eval.evaluate_classical(bbs, st, pc))
    nn = int(evaluate_acc(acc, st[bb.ST_SIDE]))
    hang = int(bb_eval._hanging_material(bbs))
    scores: dict[int, int] = {}
    for mode in (0, 1, 2, 3):
        scores[mode] = int(bb_eval.evaluate(bbs, st, pc, acc, np.int64(mode)))
    delta = nn - classical
    clip = int(CORR_CLIP[0])
    pct = int(NNUE_PCT[0])
    e1 = classical + max(-400, min(400, delta)) // 2 + hang // 2
    e2 = classical + max(-clip, min(clip, delta)) * pct // 100 + hang // 2
    e3 = classical + max(-clip, min(clip, delta)) + hang // 2
    print(
        f"    eval modes startpos C={classical} N={nn} "
        f"E0={scores[0]} E1={scores[1]} E2={scores[2]} E3={scores[3]} hang={hang}"
    )
    if scores[0] != classical + hang // 2:
        return f"E0 {scores[0]} != C+hang/2 {classical + hang // 2}"
    if scores[1] != e1:
        return f"E1 {scores[1]} != {e1}"
    if scores[2] != e2:
        return f"E2 {scores[2]} != {e2}"
    if scores[3] != e3:
        return f"E3 {scores[3]} != {e3}"
    if MIX[0] != 0:
        return "weights loaded as residual; v2.3 uses the absolute v2.2 net"
    return None


def check_mix_identity() -> str | None:
    board = chess.Board()
    bbs, st, _mb = from_board(board)
    pc = bb_eval.new_pawn_cache()
    acc = new_accumulators()
    acc_rebuild(acc, bbs)
    mode = int(EVAL_MODE[0])
    mixed = int(bb_eval.evaluate(bbs, st, pc, acc, np.int64(mode)))
    classical = int(bb_eval.evaluate_classical(bbs, st, pc))
    nn = int(evaluate_acc(acc, st[bb.ST_SIDE]))
    hang = int(bb_eval._hanging_material(bbs))
    if mode == 0:
        expect = classical + hang // 2
    else:
        expect = int(mix_nnue(np.int64(classical), np.int64(nn))) + hang // 2
    if mixed != expect:
        return f"mix {mixed} != {expect} (C={classical} N={nn} hang={hang} mode={mode})"
    return None


def main() -> int:
    bb.init()
    t0 = time.perf_counter()
    warmup()
    warm = time.perf_counter() - t0
    weights = V23 / "nnue.npz"
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()[:16]
    print("baseline")
    print(f"  engine     v2.3_NNUE")
    print(f"  python     {platform.python_version()}")
    print(f"  os         {platform.system()} {platform.release()}")
    print(f"  cpu        {platform.processor() or platform.machine()}")
    print(f"  threads    1")
    print(f"  eval_mode  {int(EVAL_MODE[0])}  nnue_pct={int(NNUE_PCT[0])} corr_clip={int(CORR_CLIP[0])}")
    print(f"  nnue sha   {sha}")
    print(f"  warmup     {warm:.1f}s")

    rng = random.Random(1)
    checks: list[tuple[str, bool]] = []
    checks.extend(check_perft())
    hash_err = check_make_unmake_hash()
    checks.append(("make/unmake/hash", hash_err is None))
    if hash_err:
        print("   ", hash_err)
    checks.append(("acc start", check_rebuild_equals_incremental(START, 40, rng) is None))
    checks.append(("acc castle", check_rebuild_equals_incremental(CASTLE, 24, rng) is None))
    checks.append(("acc ep", check_rebuild_equals_incremental(EP, 24, rng) is None))
    checks.append(("acc promo", check_rebuild_equals_incremental(PROMO, 16, rng) is None))
    checks.extend(check_see())
    mix_err = check_mix_identity()
    checks.append(("mix identity", mix_err is None))
    if mix_err:
        print("   ", mix_err)
    mode_err = check_eval_modes()
    checks.append(("eval modes", mode_err is None))
    if mode_err:
        print("   ", mode_err)

    tables = _tables()
    _search(START, 0.2, max_depth=2, tables=tables)
    t1 = time.perf_counter()
    uci, nodes, depth, score = _search(START, 2.0, tables=tables)
    compile_dt = time.perf_counter() - t1
    print(f"  first search startpos {uci} depth {depth} nodes {nodes} score {score} in {compile_dt:.1f}s")
    checks.append(("search startpos legal", uci != "0000"))

    for fen, want, label, budget in TACTICS:
        uci, nodes, depth, score = _search(fen, budget, tables=tables)
        board = chess.Board(fen)
        legal = chess.Move.from_uci(uci) in board.legal_moves if uci != "0000" else False
        ok = legal and uci in want
        checks.append((f"tactic {label}", ok))
        mark = "ok" if ok else "FAIL"
        print(f"    {mark} {label}: {uci} (want {want}) d{depth} n{nodes} s{score}")

    uci, nodes, depth, score = _search(ZUGZWANG, 1.0, tables=tables)
    board = chess.Board(ZUGZWANG)
    legal = chess.Move.from_uci(uci) in board.legal_moves if uci != "0000" else False
    checks.append(("zugzwang kp legal", legal))
    print(f"    zugzwang {uci} d{depth} n{nodes} s{score} legal={legal}")

    t2 = time.perf_counter()
    uci, nodes, depth, score = _search(START, 30.0, node_limit=50_000, tables=tables)
    dt = time.perf_counter() - t2
    nps = nodes / dt if dt > 0 else 0.0
    checks.append(("fixed 50k nodes", nodes > 0 and uci != "0000"))
    print(f"  fixed-node 50k: {uci} d{depth} n{nodes} {nps:.0f} nps score {score}")

    failed = False
    print("\nresults")
    for name, ok in checks:
        print(f"  {name}: {'ok' if ok else 'FAIL'}")
        if not ok:
            failed = True
    print("FAIL" if failed else "ok")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

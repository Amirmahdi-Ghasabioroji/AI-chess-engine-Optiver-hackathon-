"""Perft gate: the bitboard movegen must match python-chess exactly.

Also checks that make/unmake restores every field, that the incremental Zobrist
matches a from-scratch recompute, and that the move list equals python-chess's
legal move set position by position.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess
import numpy as np

from engine import bb
from engine.bbpos import from_board, move_to_uci

# (fen, [(depth, nodes), ...]) — the standard test set.
CASES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    (
        chess.STARTING_FEN,
        ((1, 20), (2, 400), (3, 8902), (4, 197281), (5, 4865609)),
    ),
    (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ((1, 48), (2, 2039), (3, 97862), (4, 4085603)),
    ),
    (
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ((1, 14), (2, 191), (3, 2812), (4, 43238), (5, 674624)),
    ),
    (
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        ((1, 6), (2, 264), (3, 9467), (4, 422333)),
    ),
    (
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        ((1, 44), (2, 1486), (3, 62379), (4, 2103487)),
    ),
    (
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        ((1, 46), (2, 2079), (3, 89890)),
    ),
)


def run_perft(fen: str, depth: int) -> int:
    bbs, st, mb = from_board(chess.Board(fen))
    hist, moves, _ = bb.new_buffers()
    return int(bb.perft(bbs, st, mb, hist, moves, np.int64(0), np.int64(depth)))


def check_perft() -> bool:
    ok = True
    for fen, expectations in CASES:
        for depth, expected in expectations:
            t0 = time.perf_counter()
            got = run_perft(fen, depth)
            dt = time.perf_counter() - t0
            mark = "ok " if got == expected else "FAIL"
            if got != expected:
                ok = False
            rate = got / dt / 1e6 if dt > 0 else 0.0
            print(f"{mark} depth {depth} {got:>10} (want {expected:>10}) {rate:6.2f} Mnps  {fen[:40]}")
    return ok


def check_move_sets(seed: int = 7, games: int = 60) -> bool:
    """Walk random games; at every position compare our legal moves to python-chess."""
    rng = random.Random(seed)
    hist, moves, _ = bb.new_buffers()
    positions = 0
    for _ in range(games):
        board = chess.Board()
        for _ in range(120):
            if board.is_game_over(claim_draw=False):
                break
            bbs, st, mb = from_board(board)

            end = int(bb.gen_moves(bbs, st, moves, np.int64(0), False))
            ours = set()
            for i in range(end):
                mv = int(moves[i])
                if bb.make_legal(bbs, st, mb, hist, np.int64(0), np.int64(mv)):
                    bb.unmake_move(bbs, st, mb, hist, np.int64(0), np.int64(mv))
                    ours.add(move_to_uci(mv))
            theirs = {m.uci() for m in board.legal_moves}
            if ours != theirs:
                print(f"FAIL movegen {board.fen()}")
                print(f"  missing {sorted(theirs - ours)}  extra {sorted(ours - theirs)}")
                return False

            # captures-only generation must be a subset that covers every capture
            cend = int(bb.gen_moves(bbs, st, moves, np.int64(0), True))
            caps = set()
            for i in range(cend):
                mv = int(moves[i])
                if bb.make_legal(bbs, st, mb, hist, np.int64(0), np.int64(mv)):
                    bb.unmake_move(bbs, st, mb, hist, np.int64(0), np.int64(mv))
                    caps.add(move_to_uci(mv))
            want_caps = {
                m.uci()
                for m in board.legal_moves
                if board.is_capture(m) or (m.promotion is not None and m.promotion != chess.PAWN)
            }
            # promotions without capture are generated only when they promote
            want_caps |= {m.uci() for m in board.legal_moves if m.promotion is not None}
            if not want_caps <= caps:
                print(f"FAIL captures {board.fen()}  missing {sorted(want_caps - caps)}")
                return False
            if not caps <= ours:
                print(f"FAIL captures not legal subset {board.fen()}")
                return False

            positions += 1
            board.push(rng.choice(list(board.legal_moves)))
    print(f"ok  movegen matches python-chess over {positions} positions")
    return True


def check_make_unmake(seed: int = 11, games: int = 40) -> bool:
    """make/unmake must restore every array, and the incremental hash must be exact."""
    rng = random.Random(seed)
    hist, moves, _ = bb.new_buffers()
    for _ in range(games):
        board = chess.Board()
        for _ in range(120):
            if board.is_game_over(claim_draw=False):
                break
            bbs, st, mb = from_board(board)
            before = (bbs.copy(), st.copy(), mb.copy())
            end = int(bb.gen_moves(bbs, st, moves, np.int64(0), False))
            for i in range(end):
                mv = np.int64(int(moves[i]))
                if not bb.make_legal(bbs, st, mb, hist, np.int64(0), mv):
                    continue
                fresh = int(bb.compute_hash(bbs, st))
                if fresh != int(st[bb.ST_HASH]):
                    print(f"FAIL hash after {move_to_uci(int(mv))} in {board.fen()}")
                    return False
                bb.unmake_move(bbs, st, mb, hist, np.int64(0), mv)
                if (
                    not np.array_equal(bbs, before[0])
                    or not np.array_equal(st, before[1])
                    or not np.array_equal(mb, before[2])
                ):
                    print(f"FAIL restore after {move_to_uci(int(mv))} in {board.fen()}")
                    return False
            board.push(rng.choice(list(board.legal_moves)))
    print("ok  make/unmake restores state and the incremental hash is exact")
    return True


def main() -> int:
    t0 = time.perf_counter()
    bb.init()
    print(f"magics built in {time.perf_counter() - t0:.2f}s")
    ok = check_move_sets() and check_make_unmake() and check_perft()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

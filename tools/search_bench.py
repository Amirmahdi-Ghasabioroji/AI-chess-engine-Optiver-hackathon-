"""Exercise the jitted search: legality, tactics, node rate, warm-up cost.

Also times the old python-chess engine on the same positions so the speedup is
measured rather than assumed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess
import numpy as np

from engine import bb
from engine import bb_eval
from engine import bb_search
from engine import nnue
from engine.bbpos import from_board, move_to_uci

POSITIONS = (
    ("startpos", chess.STARTING_FEN),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("midgame", "r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 0 9"),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
)

TACTICS = (
    # (fen, expected uci, description)
    ("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "e1e8", "back rank mate in one"),
    ("4k3/8/8/q7/8/8/8/R3K3 w Q - 0 1", "a1a5", "take the hanging queen"),
    ("2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1", "g3g6", "Qg6 mating net"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1", "f3f7", "scholar mate"),
)


class Engine:
    def __init__(self) -> None:
        nnue.load()
        bb.init()
        self.tt, self.ord32, self.rep, self.info = bb_search.new_tables()
        self.pc = bb_eval.new_pawn_cache()
        self.hist, self.moves, self.scores = bb.new_buffers()

    def go(self, board: chess.Board, seconds: float, max_depth: int = 64) -> tuple[str, int, int]:
        bbs, st, mb = from_board(board)
        self.rep[0] = st[bb.ST_HASH]
        self.info[bb_search.I_REP_BASE] = 0
        deadline = time.perf_counter() + seconds
        mv = int(
            bb_search.search(
                bbs, st, mb, self.hist, self.moves, self.scores, self.tt, self.ord32,
                self.pc, self.rep, self.info, np.int64(max_depth), deadline, deadline,
            )
        )
        return move_to_uci(mv), int(self.info[bb_search.I_NODES]), int(
            self.info[bb_search.I_DEPTH_DONE]
        )


def main() -> int:
    t0 = time.perf_counter()
    engine = Engine()
    build = time.perf_counter() - t0

    t0 = time.perf_counter()
    engine.go(chess.Board(), 0.5)
    warm = time.perf_counter() - t0
    print(f"table build {build:.2f}s, first search (JIT compile included) {warm:.2f}s")

    ok = True
    for fen, expected, label in TACTICS:
        board = chess.Board(fen)
        uci, nodes, depth = engine.go(board, 2.0)
        move = chess.Move.from_uci(uci)
        legal = move in board.legal_moves
        good = legal and uci == expected
        print(
            f"{'ok  ' if good else 'FAIL'} {label}: played {uci} "
            f"(want {expected}) depth {depth} nodes {nodes}"
        )
        if not legal:
            print("     !! ILLEGAL MOVE")
        ok = ok and good

    print()
    total_nodes = 0
    total_time = 0.0
    for label, fen in POSITIONS:
        board = chess.Board(fen)
        t0 = time.perf_counter()
        uci, nodes, depth = engine.go(board, 3.0)
        dt = time.perf_counter() - t0
        total_nodes += nodes
        total_time += dt
        legal = chess.Move.from_uci(uci) in board.legal_moves
        print(
            f"{'ok  ' if legal else 'FAIL'} {label:9} depth {depth:2} "
            f"nodes {nodes:>9} in {dt:.2f}s = {nodes / dt / 1000:7.0f} knps  best {uci}"
        )
        ok = ok and legal
    print(f"\nbitboard engine overall {total_nodes / total_time / 1000:.0f} knps")

    _reference_rate()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def _reference_rate() -> None:
    """Node rate of the existing python-chess engine, for the speedup ratio."""
    from archive import search as old_search

    engine = old_search.Engine()
    nodes = 0
    elapsed = 0.0
    for _, fen in POSITIONS:
        board = chess.Board(fen)
        t0 = time.perf_counter()
        try:
            engine.search(board, time_ms=3000, max_depth=64, soft_ms=3000)
        except Exception:
            pass
        elapsed += time.perf_counter() - t0
        nodes += engine.nodes
    print(f"python-chess engine overall {nodes / elapsed / 1000:.0f} knps")


if __name__ == "__main__":
    raise SystemExit(main())

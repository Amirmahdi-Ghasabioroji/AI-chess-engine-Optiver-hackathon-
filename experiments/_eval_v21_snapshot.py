"""STM classical eval snapshot for self_v2.1. Run via experiments._eval_v23_vs_v21."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local" / "engine_snapshots" / "self_v2.1"))

import chess

import bb
import bb_eval
from bbpos import from_board

FENS = [
    ("start", chess.STARTING_FEN),
    ("ruy", "r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w - - 1 9"),
    ("italian", "r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 0 6"),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
]


def main() -> None:
    bb.init()
    print("{:10} {:>7}".format("pos", "v2.1"))
    for name, fen in FENS:
        bbs, st, _mb = from_board(chess.Board(fen))
        pc = bb_eval.new_pawn_cache()
        print("{:10} {:7d}".format(name, int(bb_eval.evaluate(bbs, st, pc))))


if __name__ == "__main__":
    main()

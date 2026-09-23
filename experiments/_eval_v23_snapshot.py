"""STM eval snapshot for v2.3 mix modes. Run via experiments._eval_v23_vs_v21."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "local" / "engine_snapshots" / "v2.3_NNUE"))

import chess
import numpy as np

import bb
import bb_eval
from bbpos import from_board
from nnue import acc_rebuild, evaluate_acc, new_accumulators, warmup

FENS = [
    ("start", chess.STARTING_FEN),
    ("ruy", "r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w - - 1 9"),
    ("italian", "r1bqk2r/ppp2ppp/2np1n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 0 6"),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
]


def main() -> None:
    bb.init()
    warmup()
    header = "{:10} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>6}".format(
        "pos", "E0", "E1", "E2", "E3", "N", "C", "hang"
    )
    print(header)
    for name, fen in FENS:
        bbs, st, _mb = from_board(chess.Board(fen))
        pc = bb_eval.new_pawn_cache()
        acc = new_accumulators()
        acc_rebuild(acc, bbs)
        classical = int(bb_eval.evaluate_classical(bbs, st, pc))
        nn = int(evaluate_acc(acc, st[bb.ST_SIDE]))
        hang = int(bb_eval._hanging_material(bbs))
        scores = [int(bb_eval.evaluate(bbs, st, pc, acc, np.int64(mode))) for mode in range(4)]
        print(
            "{:10} {:7d} {:7d} {:7d} {:7d} {:7d} {:7d} {:6d}".format(
                name, scores[0], scores[1], scores[2], scores[3], nn, classical, hang
            )
        )


if __name__ == "__main__":
    main()

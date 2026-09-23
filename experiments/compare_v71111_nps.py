"""Compare v7.1.1 vs v7.1.1.1 depth/NPS after JIT is warm.

    python experiments/compare_v71111_nps
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
HARD_S = 4.8
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
RUY = "r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/1BP2N2/PP1P1PPP/RNBQR1K1 w - - 0 9"

WORKER = r'''
import sys, time
sys.path.insert(0, sys.argv[1])
hard_s = float(sys.argv[2])
start = sys.argv[3]
ruy = sys.argv[4]

import chess
import numpy as np
from engine import bb, bb_eval, bb_search
from engine.bbpos import from_board
from engine.nnue import warmup

bb.init()
warmup()
hist, moves, scores = bb.new_buffers()
tt, ord32, rep, info = bb_search.new_tables()
pc = bb_eval.new_pawn_cache()

def go(fen, seconds, clear):
    board = chess.Board(fen)
    bbs, st, mb = from_board(board)
    if clear:
        tt[:, :] = 0
        ord32[:] = 0
        info[:] = 0
    rep[0] = st[bb.ST_HASH]
    info[bb_search.I_REP_BASE] = 0
    deadline = time.perf_counter() + seconds
    t0 = time.perf_counter()
    packed = int(bb_search.search(
        bbs, st, mb, hist, moves, scores, tt, ord32, pc, rep, info,
        np.int64(63), deadline, deadline,
    ))
    dt = time.perf_counter() - t0
    nodes = int(info[bb_search.I_NODES])
    depth = int(info[bb_search.I_DEPTH_DONE])
    sel = int(info[bb_search.I_SELDEPTH])
    nps = nodes / dt if dt > 0 else 0.0
    return packed, depth, sel, nodes, nps, dt

print("warmup 8s startpos")
packed, depth, sel, nodes, nps, dt = go(start, 8.0, True)
print(f"  warm d{depth} sel{sel} n{nodes} {nps:.0f} nps ({dt:.2f}s)")
for name, fen in (("startpos", start), ("ruy", ruy)):
    packed, depth, sel, nodes, nps, dt = go(fen, hard_s, True)
    print(
        f"  {name} {hard_s:.1f}s: d{depth} sel{sel} n{nodes} "
        f"{nps:.0f} nps ({dt:.2f}s) move={packed}"
    )
'''


def run_one(label: str, engine_dir: Path) -> None:
    print(f"== {label} ==")
    proc = subprocess.run(
        [str(PY), "-c", WORKER, str(engine_dir), str(HARD_S), START, RUY],
        cwd=str(ROOT),
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> int:
    run_one("v7.1.1", ROOT / "local" / "engine_snapshots" / "v7.1.1")
    run_one("v7.1.1.1", ROOT / "local" / "engine_snapshots" / "v7.1.1.1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

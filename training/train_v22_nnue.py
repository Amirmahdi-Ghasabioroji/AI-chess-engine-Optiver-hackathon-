"""Train v2.2_NNUE absolute STM NNUE on all ready Stockfish-labelled packs.

Same data and architecture as v7.1.1.3 (5M Lichess + endgame + house).
Skips the 2.5M halves of the 5M file and the quiet residual cache.

    python -m training.train_v22_nnue --from-scratch --device cuda
"""

from __future__ import annotations

from pathlib import Path

from training import train_v71_nnue as base
from training.train_v7113_nnue import rows_from_npz

ROOT = Path(__file__).resolve().parents[1]
V22 = ROOT / "local" / "engine_snapshots" / "v2.2_NNUE"

base.DEFAULT_OUT = V22 / "weights" / "nnue.npz"
base.DEFAULT_DATA = [
    Path("data/lichess/nnue_lichess.npz"),
    Path("data/lichess/nnue_endgame.npz"),
    Path("data/nnue_train.npz"),
    Path("data/nnue_train_extra.npz"),
    Path("data/nnue_train_extra2.npz"),
    Path("data/nnue_train_play6.npz"),
    Path("data/v71_selfplay.npz"),
]
base.rows_from_npz = rows_from_npz


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())

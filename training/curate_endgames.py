"""Curate SF-labelled endgames from the packed Lichess NNUE dump.

NNUE should not train on hanging/tactical positions (arXiv:2412.17948). Endgames
are underrepresented in a uniform Lichess subsample and are the gap in play.
Piece-count ≤ 12 is a quiet-enough proxy that still uses the existing SF labels.

    python -m training.curate_endgames
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_SRC = Path("data/lichess/nnue_lichess.npz")
DEFAULT_OUT = Path("data/lichess/nnue_endgame.npz")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-pieces", type=int, default=12)
    parser.add_argument("--max-cp", type=float, default=1500.0)
    arguments = parser.parse_args()
    if not arguments.src.is_file():
        print(f"missing {arguments.src}")
        return 1
    blob = np.load(arguments.src)
    features = blob["features"]
    sf = blob["sf"].astype(np.float32)
    n_pieces = (features >= 0).sum(axis=1)
    keep = (n_pieces <= arguments.max_pieces) & (np.abs(sf) <= arguments.max_cp)
    n = int(keep.sum())
    print(
        f"{arguments.src} n={len(sf)}  pieces<={arguments.max_pieces} kept={n} "
        f"({100.0 * n / max(len(sf), 1):.1f}%)",
        flush=True,
    )
    if n < 1000:
        print("too few endgames")
        return 1
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        arguments.out,
        features=features[keep],
        sf=sf[keep],
        packed_stm=np.int8(1),
        min_depth=blob["min_depth"] if "min_depth" in blob.files else np.int32(16),
        max_pieces=np.int32(arguments.max_pieces),
    )
    print(f"wrote {arguments.out} n={n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

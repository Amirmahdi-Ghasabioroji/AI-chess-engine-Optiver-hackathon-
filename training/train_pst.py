"""Ridge-fit a linear piece-square residual on Stockfish - PeSTO labels.

The MLP was slower than its accuracy was worth. A 768-weight table is a PST
delta: one add per piece, same cost as PeSTO, and it ships as numpy.

    python -m training.train_pst --data data/nnue_train.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

N_FEAT = 12 * 64
MAX_PIECES = 32
OUT = Path(__file__).resolve().parents[1] / "weights" / "pst.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        default=[Path("data/nnue_train.npz")],
        help="one or more labelled npz files; duplicate FENs keep the first copy",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--ridge", type=float, default=80.0)
    parser.add_argument("--clip", type=float, default=48.0)
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args()

    feature_parts: list[np.ndarray] = []
    pesto_parts: list[np.ndarray] = []
    teacher_parts: list[np.ndarray] = []
    stm_parts: list[np.ndarray] = []
    seen: set[str] = set()
    for path in arguments.data:
        blob = np.load(path)
        key = "sf" if "sf" in blob.files else "search"
        feats = blob["features"].astype(np.int64)
        pesto_col = blob["pesto"].astype(np.float64)
        teacher_col = blob[key].astype(np.float64)
        stm_col = blob["stm"].astype(np.float64)
        if "fen" in blob.files:
            keep = []
            for i, fen in enumerate(blob["fen"]):
                text = str(fen)
                if text in seen:
                    continue
                seen.add(text)
                keep.append(i)
            if len(keep) != len(feats):
                idx = np.array(keep, dtype=np.int64)
                feats = feats[idx]
                pesto_col = pesto_col[idx]
                teacher_col = teacher_col[idx]
                stm_col = stm_col[idx]
        print(f"loaded {path} n={len(feats)}")
        feature_parts.append(feats)
        pesto_parts.append(pesto_col)
        teacher_parts.append(teacher_col)
        stm_parts.append(stm_col)
    features = np.concatenate(feature_parts)
    pesto = np.concatenate(pesto_parts)
    teacher = np.concatenate(teacher_parts)
    stm = np.concatenate(stm_parts)
    y = np.clip(teacher - pesto, -300.0, 300.0) * stm

    n = features.shape[0]
    order = np.random.default_rng(arguments.seed).permutation(n)
    split = max(1, int(n * 0.9))
    train_i, val_i = order[:split], order[split:]

    xtx = np.zeros((N_FEAT, N_FEAT), dtype=np.float64)
    xty = np.zeros(N_FEAT, dtype=np.float64)
    for idx in train_i:
        row = features[idx]
        active = row[row >= 0]
        if active.size == 0:
            continue
        xty[active] += y[idx]
        # Outer product of the one-hot piece vector: every pair of pieces co-occurs.
        xtx[np.ix_(active, active)] += 1.0

    xtx += arguments.ridge * np.eye(N_FEAT)
    weights = np.linalg.solve(xtx, xty)
    weights = np.clip(weights, -arguments.clip, arguments.clip)

    def predict(indices: np.ndarray) -> np.ndarray:
        out = np.zeros(len(indices), dtype=np.float64)
        for j, idx in enumerate(indices):
            row = features[idx]
            active = row[row >= 0]
            out[j] = weights[active].sum() if active.size else 0.0
        return out

    pred_val = predict(val_i)
    y_val = y[val_i]
    mae = float(np.mean(np.abs(pred_val - y_val)))
    zero = float(np.mean(np.abs(y_val)))
    print(f"n={n} val mae {mae:.1f}cp  zero {zero:.1f}cp  |w| mean {np.mean(np.abs(weights)):.2f}")
    if mae >= zero - 0.5:
        print("no useful generalisation; not writing weights")
        return 1

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(arguments.out, w=weights.astype(np.float32))
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Train v7.1.1.3 absolute STM NNUE on all ready Stockfish-labelled packs.

Uses the 5M Lichess cloud set, the ≤12-piece endgame slice, and the house
packs. Skips the 2.5M halves of the 5M file and the quiet residual cache
(wrong target). Absolute labels: clipped SF centipawns, mix is still N − C.

    python -m training.train_v7113_nnue --from-scratch --device cuda
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from training import train_v71_nnue as base

ROOT = Path(__file__).resolve().parents[1]
V7113 = ROOT / "local" / "engine_snapshots" / "v7.1.1.3"

base.DEFAULT_OUT = V7113 / "weights" / "nnue.npz"
base.DEFAULT_DATA = [
    Path("data/lichess/nnue_lichess.npz"),
    Path("data/lichess/nnue_endgame.npz"),
    Path("data/nnue_train.npz"),
    Path("data/nnue_train_extra.npz"),
    Path("data/nnue_train_extra2.npz"),
    Path("data/nnue_train_play6.npz"),
    Path("data/v71_selfplay.npz"),
]


_orig_rows = base.rows_from_npz


def rows_from_npz(paths: list[Path], max_cp: float) -> tuple[np.ndarray, np.ndarray]:
    """Treat any features+sf pack as packed STM, including house files."""
    seen: set[str] = set()
    feat_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    skipped_check = 0
    skipped_cp = 0
    import chess
    from training.v71_nnue_data import pack_stm

    for path in paths:
        if not path.is_file():
            print(f"missing {path}, skip")
            continue
        blob = np.load(path, allow_pickle=True)
        key = "sf" if "sf" in blob.files else "search"
        scores = blob[key].astype(np.float32)
        packed = "features" in blob.files
        print(f"reading {path} n={len(scores)} packed={packed}", flush=True)
        if packed:
            feats = blob["features"].astype(np.int32, copy=False)
            keep = np.abs(scores) <= max_cp
            skipped_cp += int((~keep).sum())
            feat_parts.append(feats[keep])
            label_parts.append(np.clip(scores[keep], -base.CLIP, base.CLIP))
            continue
        fens = blob["fen"]
        rows: list[np.ndarray] = []
        labs: list[float] = []
        for i, raw in enumerate(fens):
            fen = str(raw)
            if fen in seen:
                continue
            seen.add(fen)
            cp = float(scores[i])
            if abs(cp) > max_cp:
                skipped_cp += 1
                continue
            try:
                board = chess.Board(fen)
            except ValueError:
                continue
            if board.is_check() or board.is_game_over():
                skipped_check += 1
                continue
            rows.append(pack_stm(board))
            labs.append(float(np.clip(cp, -base.CLIP, base.CLIP)))
        if rows:
            feat_parts.append(np.stack(rows))
            label_parts.append(np.asarray(labs, dtype=np.float32))
    n = sum(len(p) for p in label_parts)
    print(f"kept {n}  skipped_check {skipped_check}  skipped_cp {skipped_cp}")
    if n == 0:
        raise SystemExit("no training rows")
    return np.concatenate(feat_parts), np.concatenate(label_parts)


base.rows_from_npz = rows_from_npz


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())

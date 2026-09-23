"""Train the residual NNUE with PyTorch on Stockfish labels.

    python -m training.train_nnue --data data/nnue_train.npz --epochs 1000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from engine.nnue import CLIP, H1, H2, N_FEAT, SCALE, WEIGHTS_NPZ, WEIGHTS_PT, load

MAX_PIECES = 32


class ResidualNNUE(nn.Module):
    """White-oriented residual in centipawns: SCALE * tanh(MLP(sum embeddings))."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(N_FEAT, H1)
        nn.init.normal_(self.embed.weight, 0.0, 0.02)
        self.b0 = nn.Parameter(torch.full((H1,), 0.2))
        self.fc1 = nn.Linear(H1, H2)
        self.fc2 = nn.Linear(H2, 1)
        nn.init.constant_(self.fc1.bias, 0.2)
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.fc1.weight, 0.0, 0.05)
        nn.init.normal_(self.fc2.weight, 0.0, 0.05)

    def raw(self, features: torch.Tensor) -> torch.Tensor:
        valid = features >= 0
        idx = features.clamp(min=0)
        emb = self.embed(idx) * valid.unsqueeze(-1).to(dtype=self.embed.weight.dtype)
        acc = self.b0 + emb.sum(dim=1)
        h1 = acc.clamp(0.0, 1.0)
        h2 = self.fc1(h1).clamp(0.0, 1.0)
        return self.fc2(h2).squeeze(-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return SCALE * torch.tanh(self.raw(features))


def arrays_from_module(model: ResidualNNUE) -> dict[str, np.ndarray]:
    return {
        "w0": model.embed.weight.detach().cpu().numpy().astype(np.float32),
        "b0": model.b0.detach().cpu().numpy().astype(np.float32),
        "w1": model.fc1.weight.detach().cpu().numpy().T.astype(np.float32),
        "b1": model.fc1.bias.detach().cpu().numpy().astype(np.float32),
        "w2": model.fc2.weight.detach().cpu().numpy().reshape(-1).astype(np.float32),
        "b2": model.fc2.bias.detach().cpu().numpy().astype(np.float32),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        default=[Path("data/nnue_train.npz")],
        help="one or more labelled npz files; duplicate FENs keep the first copy",
    )
    parser.add_argument("--out", type=Path, default=WEIGHTS_NPZ)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument(
        "--max-residual",
        type=float,
        default=250.0,
        help="drop positions whose |SF-PeSTO| exceeds this; those are tactics a bag-of-pieces net cannot learn",
    )
    arguments = parser.parse_args()

    n_threads = max(1, min(8, os.cpu_count() or 4))
    torch.set_num_threads(n_threads)
    torch.manual_seed(arguments.seed)
    feature_parts: list[np.ndarray] = []
    pesto_parts: list[np.ndarray] = []
    teacher_parts: list[np.ndarray] = []
    stm_parts: list[np.ndarray] = []
    seen: set[str] = set()
    for path in arguments.data:
        blob = np.load(path)
        key = "sf" if "sf" in blob.files else "search"
        feats = blob["features"].astype(np.int64)
        pesto_col = blob["pesto"].astype(np.float32)
        teacher_col = blob[key].astype(np.float32)
        stm_col = blob["stm"].astype(np.float32)
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
    raw_gap = teacher - pesto
    keep = np.abs(raw_gap) <= arguments.max_residual
    features = features[keep]
    pesto = pesto[keep]
    teacher = teacher[keep]
    stm = stm[keep]
    residual = np.clip(teacher - pesto, -CLIP, CLIP)
    # White-oriented net; labels are side-to-move, so flip when Black is to move.
    residual_white = residual * stm

    n = features.shape[0]
    print(f"kept {n} / {keep.size} positions with |SF-PeSTO| <= {arguments.max_residual:.0f}cp")
    order = np.random.default_rng(arguments.seed).permutation(n)
    split = max(1, int(n * 0.9))
    train_i, val_i = order[:split], order[split:]

    x_train = torch.from_numpy(features[train_i])
    y_train = torch.from_numpy(residual_white[train_i])
    x_val = torch.from_numpy(features[val_i])
    y_val = torch.from_numpy(residual_white[val_i])

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=arguments.batch,
        shuffle=True,
    )
    model = ResidualNNUE()
    opt = torch.optim.AdamW(model.parameters(), lr=arguments.lr, weight_decay=arguments.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=arguments.epochs)
    loss_fn = nn.SmoothL1Loss()

    def metrics(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
        with torch.no_grad():
            pred = model(x)
            mse = float(torch.mean((pred - y) ** 2))
            mae = float(torch.mean(torch.abs(pred - y)))
        return mse, mae

    _, base_mae = metrics(x_val, y_val)
    # Untrained net is near 0; the real baseline is "predict no residual".
    zero_mae = float(np.mean(np.abs(residual_white[val_i])))
    print(f"n={n} zero-residual val mae {zero_mae:.1f}cp  untrained {base_mae:.1f}cp")

    best_mae = zero_mae
    best_state: dict[str, np.ndarray] | None = None
    stale = 0

    for epoch in range(1, arguments.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred / SCALE, yb / SCALE)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.detach()) * len(xb)
            seen += len(xb)
        sched.step()
        model.eval()
        val_mse, val_mae = metrics(x_val, y_val)
        train_mse, train_mae = metrics(x_train, y_train)
        lr = sched.get_last_lr()[0]
        print(
            f"epoch {epoch:03d} train mae {train_mae:.1f}cp  "
            f"val mae {val_mae:.1f}cp  val mse {val_mse:.0f}  lr {lr:.5f}",
            flush=True,
        )
        if val_mae < best_mae - 0.05:
            best_mae = val_mae
            best_state = arrays_from_module(model)
            stale = 0
        else:
            stale += 1
            if stale >= arguments.patience:
                print(f"early stop at epoch {epoch}")
                break

    if best_state is None:
        print("no useful generalisation; not writing weights")
        return 1

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(
        {
            "embed.weight": torch.from_numpy(best_state["w0"]),
            "b0": torch.from_numpy(best_state["b0"]),
            "fc1.weight": torch.from_numpy(best_state["w1"].T.copy()),
            "fc1.bias": torch.from_numpy(best_state["b1"]),
            "fc2.weight": torch.from_numpy(best_state["w2"].reshape(1, -1)),
            "fc2.bias": torch.from_numpy(best_state["b2"].reshape(-1)),
        }
    )
    torch.save(model.state_dict(), WEIGHTS_PT)
    out_npz = WEIGHTS_NPZ if arguments.out == WEIGHTS_NPZ else arguments.out
    np.savez(out_npz, **best_state)
    load(out_npz)
    print(f"wrote {WEIGHTS_PT} and {out_npz} val mae {best_mae:.1f}cp  (zero {zero_mae:.1f}cp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

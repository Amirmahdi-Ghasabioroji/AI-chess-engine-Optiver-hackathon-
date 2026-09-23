"""Fine-tune v6 NNUE as a residual on v6 classical, targeting Stockfish labels.

The net must beat classical MAE on a held-out slice or this script refuses to
write weights. Inference is `classical + net` (see core.nnue residual=1).

    python -m training.finetune_v6_nnue
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "local" / "engine_snapshots" / "v6_NNUE"
if str(V6) not in sys.path:
    sys.path.insert(0, str(V6))

from core.board import Board  # noqa: E402
from core.eval import evaluate_classical  # noqa: E402
from core.nnue import WEIGHTS_PATH, fill_indices  # noqa: E402

N_FEATURES = 768
H1 = 128
H2 = 32
SCALE = 400.0
MAX_PIECES = 32
DEFAULT_DATA = [
    ROOT / "data" / "nnue_train.npz",
    ROOT / "data" / "nnue_train_extra.npz",
    ROOT / "data" / "nnue_train_extra2.npz",
    ROOT / "data" / "nnue_train_play6.npz",
]


class ResidualMLP(nn.Module):
    """Matches v6 `_forward`: sparse embed + ReLU + ReLU + linear * SCALE."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(N_FEATURES, H1)
        self.b1 = nn.Parameter(torch.zeros(H1))
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, 1)
        nn.init.normal_(self.embed.weight, 0.0, 0.02)
        nn.init.constant_(self.b1, 0.2)
        nn.init.constant_(self.fc2.bias, 0.2)
        nn.init.zeros_(self.fc3.bias)
        nn.init.normal_(self.fc2.weight, 0.0, 0.05)
        nn.init.normal_(self.fc3.weight, 0.0, 0.05)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        valid = (idx >= 0).unsqueeze(-1)
        safe = idx.clamp(min=0)
        acc = self.b1 + (self.embed(safe) * valid.to(self.embed.weight.dtype)).sum(dim=1)
        h1 = torch.relu(acc)
        h2 = torch.relu(self.fc2(h1))
        return SCALE * self.fc3(h2).squeeze(-1)


def load_pretrained(model: ResidualMLP, path: Path) -> None:
    blob = np.load(path)
    w1 = np.ascontiguousarray(blob["fc1_weight"].T, dtype=np.float32)
    if w1.shape != (N_FEATURES, H1):
        print(f"skip pretrained: fc1 {w1.shape} != {(N_FEATURES, H1)}")
        return
    model.embed.weight.data.copy_(torch.from_numpy(w1))
    model.b1.data.copy_(torch.from_numpy(np.ascontiguousarray(blob["fc1_bias"], dtype=np.float32)))
    model.fc2.weight.data.copy_(
        torch.from_numpy(np.ascontiguousarray(blob["fc2_weight"], dtype=np.float32))
    )
    model.fc2.bias.data.copy_(
        torch.from_numpy(np.ascontiguousarray(blob["fc2_bias"], dtype=np.float32))
    )
    model.fc3.weight.data.copy_(
        torch.from_numpy(np.ascontiguousarray(blob["fc3_weight"], dtype=np.float32))
    )
    model.fc3.bias.data.copy_(
        torch.from_numpy(np.ascontiguousarray(blob["fc3_bias"], dtype=np.float32))
    )
    print(f"loaded pretrained {path}")


def arrays_from_module(model: ResidualMLP) -> dict[str, np.ndarray]:
    return {
        "fc1_weight": model.embed.weight.detach().cpu().numpy().T.astype(np.float32),
        "fc1_bias": model.b1.detach().cpu().numpy().astype(np.float32),
        "fc2_weight": model.fc2.weight.detach().cpu().numpy().astype(np.float32),
        "fc2_bias": model.fc2.bias.detach().cpu().numpy().astype(np.float32),
        "fc3_weight": model.fc3.weight.detach().cpu().numpy().astype(np.float32),
        "fc3_bias": model.fc3.bias.detach().cpu().numpy().astype(np.float32),
    }


def extract_rows(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STM features + v6 classical + SF, keyed on unique FEN."""
    seen: set[str] = set()
    feats: list[np.ndarray] = []
    classical: list[int] = []
    teacher: list[int] = []
    buf = np.empty(MAX_PIECES, dtype=np.int32)
    for path in paths:
        if not path.is_file():
            print(f"missing {path}, skip")
            continue
        blob = np.load(path, allow_pickle=True)
        key = "sf" if "sf" in blob.files else "search"
        fens = blob["fen"]
        scores = blob[key].astype(np.int32)
        print(f"labelling {path.name} n={len(fens)}", flush=True)
        for i, raw in enumerate(fens):
            fen = str(raw)
            if fen in seen:
                continue
            seen.add(fen)
            try:
                pos = Board.from_fen(fen)
            except Exception:
                continue
            buf.fill(-1)
            n = fill_indices(pos, buf)
            if n == 0:
                continue
            row = buf.copy()
            if n < MAX_PIECES:
                row[n:] = -1
            feats.append(row)
            classical.append(int(evaluate_classical(pos)))
            teacher.append(int(scores[i]))
            if len(feats) % 10000 == 0:
                print(f"  {len(feats)} positions", flush=True)
    if not feats:
        raise SystemExit("no training rows")
    return np.stack(feats), np.asarray(classical, dtype=np.float32), np.asarray(teacher, dtype=np.float32)


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def train(
    model: ResidualMLP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    c_val: np.ndarray,
    t_val: np.ndarray,
    classical_mae: float,
    epochs: int,
    batch: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> tuple[dict[str, np.ndarray] | None, float]:
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    loss_fn = nn.SmoothL1Loss()
    best_mae = classical_mae
    best_state: dict[str, np.ndarray] | None = None
    stale = 0

    def combo_mae() -> float:
        model.eval()
        with torch.no_grad():
            residual = model(x_val).cpu().numpy()
        return mae(c_val + residual, t_val)

    start = combo_mae()
    print(f"start combo mae {start:.1f}cp  classical {classical_mae:.1f}cp", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred / SCALE, yb / SCALE)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        val_mae = combo_mae()
        print(
            f"epoch {epoch:03d} combo mae {val_mae:.1f}cp  lr {sched.get_last_lr()[0]:.6f}",
            flush=True,
        )
        if val_mae < best_mae - 0.15:
            best_mae = val_mae
            best_state = arrays_from_module(model)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {epoch}")
                break
    return best_state, best_mae


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=WEIGHTS_PATH)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--max-residual", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--from-scratch", action="store_true")
    arguments = parser.parse_args()

    torch.set_num_threads(max(1, min(8, os_cpu())))
    torch.manual_seed(arguments.seed)
    np.random.seed(arguments.seed)

    features, classical, teacher = extract_rows(list(arguments.data))
    residual = np.clip(teacher - classical, -arguments.max_residual, arguments.max_residual)
    keep = np.abs(teacher - classical) <= arguments.max_residual
    features, classical, teacher, residual = features[keep], classical[keep], teacher[keep], residual[keep]
    n = features.shape[0]
    print(f"kept {n} / {keep.size} with |SF-classical| <= {arguments.max_residual:.0f}cp")

    order = np.random.default_rng(arguments.seed).permutation(n)
    split = max(1, int(n * 0.9))
    train_i, val_i = order[:split], order[split:]
    x_train = torch.from_numpy(features[train_i].astype(np.int64))
    y_train = torch.from_numpy(residual[train_i])
    x_val = torch.from_numpy(features[val_i].astype(np.int64))
    y_val = torch.from_numpy(residual[val_i])
    c_val = classical[val_i]
    t_val = teacher[val_i]
    classical_mae = mae(c_val, t_val)
    zero_res = mae(np.zeros_like(t_val), t_val - c_val)
    print(f"val classical mae {classical_mae:.1f}cp  |residual| {zero_res:.1f}cp  n_val={len(val_i)}")

    model = ResidualMLP()
    if not arguments.from_scratch and arguments.out.is_file():
        load_pretrained(model, arguments.out)

    best_state, best_mae = train(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        c_val,
        t_val,
        classical_mae,
        arguments.epochs,
        arguments.batch,
        arguments.lr,
        arguments.weight_decay,
        arguments.patience,
    )

    if best_state is None or best_mae >= classical_mae:
        print("fine-tune missed classical; retrying from scratch")
        model = ResidualMLP()
        best_state, best_mae = train(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            c_val,
            t_val,
            classical_mae,
            arguments.epochs,
            arguments.batch,
            arguments.lr,
            arguments.weight_decay,
            arguments.patience,
        )

    if best_state is None or best_mae >= classical_mae:
        print(
            f"net does not beat classical ({best_mae:.1f} vs {classical_mae:.1f}); not writing"
        )
        return 1

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    if arguments.out.is_file():
        bak = arguments.out.with_suffix(".npz.bak")
        shutil.copy2(arguments.out, bak)
        print(f"backed up {arguments.out} -> {bak}")
    np.savez(
        arguments.out,
        **best_state,
        scale=np.float32(SCALE),
        clip=np.float32(arguments.max_residual),
        residual=np.int32(1),
        linear=np.int32(1),
        blend=np.float32(1.0),
        val_mae=np.float32(best_mae),
        classical_mae=np.float32(classical_mae),
    )
    print(
        f"wrote {arguments.out} combo mae {best_mae:.1f}cp  "
        f"(classical {classical_mae:.1f}cp, {classical_mae - best_mae:.1f}cp better)"
    )
    return 0


def os_cpu() -> int:
    import os

    return os.cpu_count() or 4


if __name__ == "__main__":
    raise SystemExit(main())

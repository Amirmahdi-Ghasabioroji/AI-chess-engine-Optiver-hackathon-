"""Train v7.1's STM NNUE on Stockfish centipawns (absolute, not residual).

Picks the learning rate by measuring validation MAE, then trains on GPU
when CUDA is available.

    python -m training.train_v71_nnue
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training.v71_nnue_data import N_FEATURES, V71_ROOT, pack_stm

H1 = 160
H2 = 32
SCALE = 400.0
CLIP = 2000.0
DEFAULT_OUT = V71_ROOT / "weights" / "nnue.npz"
DEFAULT_DATA = [
    Path("data/lichess/nnue_lichess.npz"),
    Path("data/nnue_train.npz"),
    Path("data/nnue_train_extra.npz"),
    Path("data/nnue_train_extra2.npz"),
    Path("data/nnue_train_play6.npz"),
    Path("data/v71_selfplay.npz"),
]


class STMLP(nn.Module):
    """Matches v7.1 `_forward`: sparse embed + ReLU + ReLU + linear * SCALE."""

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


def arrays_from_module(model: STMLP) -> dict[str, np.ndarray]:
    return {
        "fc1_weight": model.embed.weight.detach().cpu().numpy().T.astype(np.float32),
        "fc1_bias": model.b1.detach().cpu().numpy().astype(np.float32),
        "fc2_weight": model.fc2.weight.detach().cpu().numpy().astype(np.float32),
        "fc2_bias": model.fc2.bias.detach().cpu().numpy().astype(np.float32),
        "fc3_weight": model.fc3.weight.detach().cpu().numpy().astype(np.float32),
        "fc3_bias": model.fc3.bias.detach().cpu().numpy().astype(np.float32),
        "scale": np.float32(SCALE),
        "clip": np.float32(CLIP),
    }


def load_pretrained(model: STMLP, path: Path) -> None:
    if not path.is_file():
        return
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


def rows_from_npz(paths: list[Path], max_cp: float) -> tuple[np.ndarray, np.ndarray]:
    seen: set[str] = set()
    feat_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    skipped_check = 0
    skipped_cp = 0
    for path in paths:
        if not path.is_file():
            print(f"missing {path}, skip")
            continue
        blob = np.load(path, allow_pickle=True)
        key = "sf" if "sf" in blob.files else "search"
        scores = blob[key].astype(np.float32)
        packed = "packed_stm" in blob.files
        print(f"reading {path} n={len(scores)} packed={packed}", flush=True)
        if packed:
            feats = blob["features"].astype(np.int32, copy=False)
            keep = np.abs(scores) <= max_cp
            skipped_cp += int((~keep).sum())
            feat_parts.append(feats[keep])
            label_parts.append(np.clip(scores[keep], -CLIP, CLIP))
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
            labs.append(float(np.clip(cp, -CLIP, CLIP)))
        if rows:
            feat_parts.append(np.stack(rows))
            label_parts.append(np.asarray(labs, dtype=np.float32))
    n = sum(len(p) for p in label_parts)
    print(f"kept {n}  skipped_check {skipped_check}  skipped_cp {skipped_cp}")
    if n == 0:
        raise SystemExit("no training rows")
    return np.concatenate(feat_parts), np.concatenate(label_parts)


def pick_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def restore(model: nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({key: value.to(device) for key, value in state.items()})


def batched_mae(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    batch: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, features.shape[0], batch):
            xb = features[start : start + batch].to(device, non_blocking=True)
            yb = labels[start : start + batch].to(device, non_blocking=True)
            pred = model(xb)
            total += float((pred - yb).abs().sum())
            count += int(yb.shape[0])
    return total / max(count, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    model.train()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred / SCALE, yb / SCALE)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if scheduler is not None:
            scheduler.step()


def _smooth(values: list[float], beta: float = 0.98) -> np.ndarray:
    out: list[float] = []
    average = 0.0
    for index, value in enumerate(values, start=1):
        average = beta * average + (1.0 - beta) * value
        out.append(average / (1.0 - beta**index))
    return np.asarray(out, dtype=np.float64)


def lr_range_candidates(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    weight_decay: float,
    init_lr: float,
    max_lr: float,
    n_steps: int,
) -> tuple[float, float]:
    """Leslie Smith range test, scored on batch MAE in centipawns."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    gamma = (max_lr / init_lr) ** (1.0 / max(n_steps - 1, 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    records: list[tuple[float, float]] = []
    iterator = iter(loader)
    start_mae: float | None = None
    for step in range(n_steps):
        try:
            xb, yb = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            xb, yb = next(iterator)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred / SCALE, yb / SCALE)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            mae = float((pred.detach() - yb).abs().mean())
        lr = float(opt.param_groups[0]["lr"])
        records.append((lr, mae))
        if start_mae is None:
            start_mae = mae
        scheduler.step()
        if start_mae is not None and mae > 3.0 * start_mae and step > 20:
            print(f"lr range diverged at step {step} lr {lr:.2e} mae {mae:.1f}")
            break
    if len(records) < 12:
        raise SystemExit("lr range test produced too few steps")
    lrs = np.asarray([row[0] for row in records], dtype=np.float64)
    maes = _smooth([row[1] for row in records])
    skip = max(8, len(maes) // 10)
    running_min = np.minimum.accumulate(maes)
    blow = np.where(maes > running_min * 2.5)[0]
    stop = int(blow[0]) if len(blow) else len(maes)
    stop = max(skip + 4, stop)
    window = slice(skip, stop)
    log_lr = np.log(np.clip(lrs[window], 1e-12, None))
    slope = np.diff(maes[window]) / np.clip(np.diff(log_lr), 1e-12, None)
    steep_i = int(np.argmin(slope))
    steep_lr = float(lrs[window][steep_i])
    min_i = int(np.argmin(maes[window]))
    min_mae_lr = float(lrs[window][min_i])
    print(
        f"lr range: steepest {steep_lr:.2e}  min-mae {min_mae_lr:.2e} "
        f"(mae {maes[window][min_i]:.1f}cp over {stop - skip} steps)",
        flush=True,
    )
    return steep_lr, min_mae_lr


def probe_lr_by_mae(
    model: nn.Module,
    init_state: dict[str, torch.Tensor],
    loader: DataLoader,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    loss_fn: nn.Module,
    weight_decay: float,
    candidates: list[float],
    eval_batch: int,
) -> float:
    """One constant-LR epoch per candidate; keep the LR with the lowest val MAE."""
    scored: list[tuple[float, float]] = []
    for lr in candidates:
        restore(model, init_state, device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_one_epoch(model, loader, opt, loss_fn, device)
        val_mae = batched_mae(model, x_val, y_val, device, eval_batch)
        scored.append((lr, val_mae))
        print(f"lr probe {lr:.2e}  val mae {val_mae:.1f}cp", flush=True)
    restore(model, init_state, device)
    best_lr, best_mae = min(scored, key=lambda row: row[1])
    print(f"using lr {best_lr:.2e} (probe val mae {best_mae:.1f}cp)", flush=True)
    return best_lr


def make_loader(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    batch: int,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=device.type == "cuda",
        num_workers=0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pretrained", type=Path, default=V71_ROOT / "data" / "nnue.npz")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=0, help="0 = 8192 on CUDA, 512 on CPU")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0,
        help="AdamW max LR; 0 = MAE range test + 1-epoch probes",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--max-cp", type=float, default=1500.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--lr-find-steps", type=int, default=400)
    arguments = parser.parse_args()

    device = pick_device(arguments.device)
    batch = arguments.batch
    if batch <= 0:
        batch = 8192 if device.type == "cuda" else 512
    print(f"device {device}  batch {batch}", flush=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print(f"gpu {torch.cuda.get_device_name(0)}", flush=True)
    else:
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))

    torch.manual_seed(arguments.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(arguments.seed)

    features, labels = rows_from_npz(arguments.data, arguments.max_cp)
    n = features.shape[0]
    order = np.random.default_rng(arguments.seed).permutation(n)
    split = max(1, int(n * 0.9))
    train_i, val_i = order[:split], order[split:]

    x_train = torch.from_numpy(features[train_i].astype(np.int64))
    y_train = torch.from_numpy(labels[train_i])
    x_val = torch.from_numpy(features[val_i].astype(np.int64))
    y_val = torch.from_numpy(labels[val_i])
    del features, labels, order, train_i, val_i

    loader = make_loader(x_train, y_train, batch, device, shuffle=True)
    eval_batch = min(batch, 16384)
    model = STMLP().to(device)
    if not arguments.from_scratch:
        load_pretrained(model, arguments.pretrained)
    loss_fn = nn.SmoothL1Loss()

    init_mae = batched_mae(model, x_val, y_val, device, eval_batch)
    zero_mae = float(y_val.abs().mean())
    print(f"n={n} train={x_train.shape[0]} val={x_val.shape[0]}", flush=True)
    print(f"val |SF| mae {zero_mae:.1f}cp  init mae {init_mae:.1f}cp", flush=True)

    init_state = snapshot(model)
    if arguments.lr > 0:
        chosen_lr = arguments.lr
        print(f"using explicit lr {chosen_lr:.2e}", flush=True)
    else:
        restore(model, init_state, device)
        steep_lr, min_mae_lr = lr_range_candidates(
            model,
            loader,
            device,
            loss_fn,
            arguments.weight_decay,
            init_lr=1e-6,
            max_lr=3e-2,
            n_steps=min(arguments.lr_find_steps, max(80, 2 * len(loader))),
        )
        restore(model, init_state, device)
        raw = [
            steep_lr,
            steep_lr / 2.0,
            min_mae_lr / 10.0,
            min_mae_lr / 3.0,
            8e-4,
        ]
        candidates: list[float] = []
        for lr in raw:
            rounded = float(np.clip(lr, 3e-5, 2e-2))
            if all(abs(np.log10(rounded) - np.log10(other)) > 0.12 for other in candidates):
                candidates.append(rounded)
        candidates.sort()
        print("lr probe candidates " + " ".join(f"{lr:.2e}" for lr in candidates), flush=True)
        chosen_lr = probe_lr_by_mae(
            model,
            init_state,
            loader,
            x_val,
            y_val,
            device,
            loss_fn,
            arguments.weight_decay,
            candidates,
            eval_batch,
        )

    restore(model, init_state, device)
    opt = torch.optim.AdamW(model.parameters(), lr=chosen_lr, weight_decay=arguments.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=chosen_lr,
        epochs=arguments.epochs,
        steps_per_epoch=max(len(loader), 1),
        pct_start=0.12,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=50.0,
    )

    best_mae = init_mae
    best_state = arrays_from_module(model)
    stale = 0

    for epoch in range(1, arguments.epochs + 1):
        train_one_epoch(model, loader, opt, loss_fn, device, scheduler)
        val_mae = batched_mae(model, x_val, y_val, device, eval_batch)
        train_slice = min(32768, x_train.shape[0])
        train_mae = batched_mae(
            model,
            x_train[:train_slice],
            y_train[:train_slice],
            device,
            eval_batch,
        )
        current_lr = float(opt.param_groups[0]["lr"])
        print(
            f"epoch {epoch:03d} train mae {train_mae:.1f}cp  val mae {val_mae:.1f}cp  "
            f"lr {current_lr:.5f}",
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

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(arguments.out, **best_state)
    agent_root = arguments.out.resolve().parent
    if agent_root.name == "weights":
        agent_root = agent_root.parent
    for extra in (agent_root / "data" / "nnue.npz", agent_root / "nnue.npz"):
        extra.parent.mkdir(parents=True, exist_ok=True)
        np.savez(extra, **best_state)
    print(f"wrote {arguments.out} val mae {best_mae:.1f}cp")
    return 0


if __name__ == "__main__":
    sys.exit(main())

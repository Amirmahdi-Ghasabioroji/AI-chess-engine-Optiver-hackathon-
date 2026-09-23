"""Train v7.2 residual NNUE: target is Stockfish − classical on quiet positions.

    python -m training.train_v72_residual --from-scratch --device cuda
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

from training.train_v71_nnue import (
    STMLP,
    arrays_from_module,
    batched_mae,
    load_pretrained,
    lr_range_candidates,
    make_loader,
    pick_device,
    probe_lr_by_mae,
    restore,
    snapshot,
    train_one_epoch,
)
from training.v71_nnue_data import MAX_PIECES, pack_stm

ROOT = Path(__file__).resolve().parents[1]
V72 = ROOT / "local" / "engine_snapshots" / "v7.2"
if str(V72) not in sys.path:
    sys.path.insert(0, str(V72))

from engine import bb  # noqa: E402
from engine.bb import (  # noqa: E402
    BIT,
    KING,
    MAX_MOVES,
    N_BB,
    N_ST,
    NO_EP,
    OCC_B,
    OCC_W,
    ST_EP,
    attacked,
    gen_moves,
    king_square,
)
from engine.bbpos import from_board  # noqa: E402
from engine.bb_eval import PAWN_HASH_SIZE, evaluate_classical, new_pawn_cache  # noqa: E402

SCALE = 400.0
CLIP = 400.0
DEFAULT_OUT = V72 / "weights" / "nnue.npz"
CACHE = Path("data/lichess/nnue_v72_residual_quiet.npz")
DEFAULT_DATA = [
    Path("data/lichess/nnue_lichess.npz"),
    Path("data/lichess/nnue_endgame.npz"),
    Path("data/lichess/nnue_endgame.npz"),
    Path("data/nnue_train.npz"),
    Path("data/nnue_train_extra.npz"),
    Path("data/nnue_train_extra2.npz"),
    Path("data/nnue_train_play6.npz"),
    Path("data/v71_selfplay.npz"),
]


def _save_arrays(model: STMLP) -> dict[str, np.ndarray]:
    arrays = arrays_from_module(model)
    arrays["clip"] = np.float32(CLIP)
    arrays["residual"] = np.int8(1)
    return arrays


def _label_packed(features: np.ndarray, sf: np.ndarray, max_resid: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep quiet STM rows; labels are clipped (SF − classical)."""
    from numba import njit

    @njit(cache=False)
    def _filter(
        feats: np.ndarray,
        scores: np.ndarray,
        bit: np.ndarray,
        max_r: np.float32,
        require_quiet: np.int32,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = feats.shape[0]
        keep = np.zeros(n, dtype=np.uint8)
        classical = np.zeros(n, dtype=np.float32)
        moves = np.zeros(MAX_MOVES, dtype=np.int32)
        pc = np.zeros((PAWN_HASH_SIZE, 3), dtype=np.int64)
        for row in range(n):
            bbs = np.zeros(N_BB, dtype=np.int64)
            mb = np.empty(64, dtype=np.int8)
            for sq in range(64):
                mb[sq] = np.int8(-1)
            ok = True
            for i in range(feats.shape[1]):
                feat = feats[row, i]
                if feat < 0:
                    continue
                piece = feat // 64
                square = feat % 64
                if piece < 0 or piece > 11 or square < 0 or square > 63:
                    ok = False
                    break
                bbs[piece] |= bit[square]
                mb[square] = np.int8(piece)
            if not ok or bbs[KING] == 0 or bbs[6 + KING] == 0:
                continue
            occ_w = np.int64(0)
            occ_b = np.int64(0)
            for p in range(6):
                occ_w |= bbs[p]
                occ_b |= bbs[p + 6]
            bbs[OCC_W] = occ_w
            bbs[OCC_B] = occ_b
            st = np.zeros(N_ST, dtype=np.int64)
            st[ST_EP] = NO_EP
            occ = occ_w | occ_b
            if attacked(bbs, king_square(bbs, np.int64(0)), np.int64(1), occ):
                continue
            if require_quiet != 0:
                end = gen_moves(bbs, st, moves, np.int64(0), True)
                if end > 0:
                    continue
            cl = float(evaluate_classical(bbs, st, pc))
            resid = float(scores[row]) - cl
            if resid > max_r or resid < -max_r:
                continue
            keep[row] = 1
            classical[row] = np.float32(cl)
        return keep, classical

    bb.init()
    dummy = np.zeros((1, MAX_PIECES), dtype=np.int32)
    dummy[0, 0] = 5 * 64 + 4
    dummy[0, 1] = 11 * 64 + 60
    _filter(dummy, np.zeros(1, dtype=np.float32), BIT, np.float32(400.0), np.int32(0))
    print(f"labelling n={len(sf)} quiet residual (max |SF-classical| {max_resid:.0f})", flush=True)
    keep, classical = _filter(features, sf, BIT, np.float32(max_resid), np.int32(1))
    n_keep = int(keep.sum())
    print(f"  quiet kept {n_keep}", flush=True)
    if n_keep < 80_000:
        print("  too few quiet rows; also keeping not-in-check with small residual", flush=True)
        keep, classical = _filter(features, sf, BIT, np.float32(min(max_resid, 200.0)), np.int32(0))
        n_keep = int(keep.sum())
        print(f"  relaxed kept {n_keep}", flush=True)
    mask = keep.astype(bool)
    resid = np.clip(sf[mask] - classical[mask], -CLIP, CLIP).astype(np.float32)
    return features[mask], resid


def rows_from_npz(
    paths: list[Path], max_resid: float, cache: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if cache is not None and cache.is_file():
        blob = np.load(cache)
        print(f"cache hit {cache} n={len(blob['sf'])}", flush=True)
        return blob["features"].astype(np.int32, copy=False), blob["sf"].astype(np.float32)
    seen: set[str] = set()
    feat_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    skipped_check = 0
    skipped_cap = 0
    skipped_resid = 0
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
            feats, resid = _label_packed(blob["features"].astype(np.int32, copy=False), scores, max_resid)
            if len(resid):
                feat_parts.append(feats)
                label_parts.append(resid)
            continue
        fens = blob["fen"]
        rows: list[np.ndarray] = []
        labs: list[float] = []
        bb.init()
        pc = new_pawn_cache()
        moves = np.zeros(MAX_MOVES, dtype=np.int32)
        for i, raw in enumerate(fens):
            fen = str(raw)
            if fen in seen:
                continue
            seen.add(fen)
            try:
                board = chess.Board(fen)
            except ValueError:
                continue
            if board.is_check() or board.is_game_over():
                skipped_check += 1
                continue
            if any(board.is_capture(move) or move.promotion for move in board.legal_moves):
                skipped_cap += 1
                continue
            bbs, st, _mb = from_board(board)
            if attacked(bbs, king_square(bbs, st[0]), 1 - st[0], bbs[OCC_W] | bbs[OCC_B]):
                skipped_check += 1
                continue
            end = gen_moves(bbs, st, moves, np.int64(0), True)
            if end > 0:
                skipped_cap += 1
                continue
            classical = float(evaluate_classical(bbs, st, pc))
            resid = float(scores[i]) - classical
            if abs(resid) > max_resid:
                skipped_resid += 1
                continue
            rows.append(pack_stm(board))
            labs.append(float(np.clip(resid, -CLIP, CLIP)))
        if rows:
            feat_parts.append(np.stack(rows))
            label_parts.append(np.asarray(labs, dtype=np.float32))
    n = sum(len(p) for p in label_parts)
    print(
        f"kept {n}  skipped_check {skipped_check}  skipped_cap {skipped_cap}  "
        f"skipped_resid {skipped_resid}"
    )
    if n < 1000:
        raise SystemExit("no training rows")
    features = np.concatenate(feat_parts)
    labels = np.concatenate(label_parts)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, features=features, sf=labels, residual=np.int8(1))
        print(f"wrote cache {cache} n={n}", flush=True)
    return features, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pretrained", type=Path, default=V72 / "weights" / "nnue.npz")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--max-resid", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--lr-find-steps", type=int, default=300)
    arguments = parser.parse_args()

    device = pick_device(arguments.device)
    batch = arguments.batch if arguments.batch > 0 else (8192 if device.type == "cuda" else 512)
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

    features, labels = rows_from_npz(arguments.data, arguments.max_resid, arguments.cache)
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
    print(f"val |residual| {zero_mae:.1f}cp  init mae {init_mae:.1f}cp", flush=True)

    init_state = snapshot(model)
    if arguments.lr > 0:
        chosen_lr = arguments.lr
        print(f"using explicit lr {chosen_lr:.2e}", flush=True)
    else:
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
        raw = [steep_lr, steep_lr / 2.0, min_mae_lr / 10.0, min_mae_lr / 3.0, 8e-4]
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
    best_state = _save_arrays(model)
    stale = 0
    for epoch in range(1, arguments.epochs + 1):
        train_one_epoch(model, loader, opt, loss_fn, device, scheduler)
        val_mae = batched_mae(model, x_val, y_val, device, eval_batch)
        train_mae = batched_mae(
            model, x_train[:32768], y_train[:32768], device, eval_batch
        )
        print(
            f"epoch {epoch:03d} train mae {train_mae:.1f}cp  val mae {val_mae:.1f}cp  "
            f"lr {float(opt.param_groups[0]['lr']):.5f}",
            flush=True,
        )
        if val_mae < best_mae - 0.05:
            best_mae = val_mae
            best_state = _save_arrays(model)
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
    print(f"wrote {arguments.out} residual val mae {best_mae:.1f}cp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

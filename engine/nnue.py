"""Learned residual on top of PeSTO.

Search never imports torch. Training writes `weights/nnue.npz`; `load()` copies
those arrays into buffers the jitted evaluator already holds. Piece embeddings
are maintained incrementally on PVS make/unmake. Quiescence stays PeSTO.

If the npz is missing, the residual is zero and the engine is Stage 2 PeSTO
plus the linear PST fallback in `bb_eval`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit

from . import bb

N_FEAT = 12 * 64
H1 = 64
H2 = 16
CLIP = 300
SCALE = 250.0

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
WEIGHTS_PT = WEIGHTS_DIR / "nnue.pt"
WEIGHTS_NPZ = WEIGHTS_DIR / "nnue.npz"
WEIGHTS_PATH = WEIGHTS_NPZ

W0 = np.zeros((N_FEAT, H1), dtype=np.float32)
B0 = np.zeros(H1, dtype=np.float32)
W1 = np.zeros((H1, H2), dtype=np.float32)
B1 = np.zeros(H2, dtype=np.float32)
W2 = np.zeros(H2, dtype=np.float32)
B2 = np.zeros(1, dtype=np.float32)
ENABLED = np.zeros(1, dtype=np.int32)


def _install(blob: dict[str, np.ndarray]) -> None:
    w0 = np.asarray(blob["w0"], dtype=np.float32)
    if w0.shape != W0.shape:
        raise ValueError(f"weight shape {w0.shape} != {W0.shape}")
    W0[:, :] = w0
    B0[:] = np.asarray(blob["b0"], dtype=np.float32).reshape(H1)
    W1[:, :] = np.asarray(blob["w1"], dtype=np.float32).reshape(H1, H2)
    B1[:] = np.asarray(blob["b1"], dtype=np.float32).reshape(H2)
    W2[:] = np.asarray(blob["w2"], dtype=np.float32).reshape(H2)
    B2[0] = float(np.asarray(blob["b2"]).reshape(-1)[0])
    ENABLED[0] = 1


def load(path: Path | None = None) -> bool:
    """Load `weights/nnue.npz` into the numba buffers. Never raises."""
    ENABLED[0] = 0
    target = path if path is not None else WEIGHTS_NPZ
    if not target.is_file():
        return False
    try:
        blob = np.load(target)
        _install({k: blob[k] for k in ("w0", "b0", "w1", "b1", "w2", "b2")})
        return True
    except (OSError, KeyError, ValueError):
        ENABLED[0] = 0
        return False


def new_acc() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(H1, dtype=np.float32), np.zeros((bb.MAX_PLY + 8, H1), dtype=np.float32)


def disable() -> None:
    ENABLED[0] = 0
    W0[:, :] = 0
    B0[:] = 0
    W1[:, :] = 0
    B1[:] = 0
    W2[:] = 0
    B2[:] = 0


@njit(cache=False)
def _acc_add(acc: np.ndarray, feat: np.int64) -> None:
    row = W0[feat]
    for i in range(H1):
        acc[i] += row[i]


@njit(cache=False)
def _acc_sub(acc: np.ndarray, feat: np.int64) -> None:
    row = W0[feat]
    for i in range(H1):
        acc[i] -= row[i]


@njit(cache=False)
def acc_refresh(acc: np.ndarray, bbs: np.ndarray) -> None:
    """Rebuild `acc` from the current bitboards."""
    for i in range(H1):
        acc[i] = B0[i]
    if ENABLED[0] == 0:
        return
    for code in range(12):
        b = bbs[code]
        while b != 0:
            sq = bb.lsb(b)
            b &= b - 1
            _acc_add(acc, np.int64(code * 64 + sq))


@njit(cache=False)
def acc_apply_move(acc: np.ndarray, mb: np.ndarray, st: np.ndarray, mv: np.int64) -> None:
    """Update `acc` for `mv` using the mailbox *before* make_move."""
    if ENABLED[0] == 0:
        return
    frm = mv & 63
    to = (mv >> 6) & 63
    promo = (mv >> 12) & 7
    mtype = (mv >> 15) & 3
    side = st[bb.ST_SIDE]
    us = side * 6
    piece = np.int64(mb[frm])
    _acc_sub(acc, piece * 64 + frm)
    if mtype == bb.MT_EP:
        cap_sq = to - 8 if side == bb.WHITE else to + 8
        _acc_sub(acc, np.int64((1 - side) * 6 + bb.PAWN) * 64 + cap_sq)
    else:
        occupant = np.int64(mb[to])
        if occupant >= 0:
            _acc_sub(acc, occupant * 64 + to)
    if promo != 0:
        _acc_add(acc, np.int64(us + promo) * 64 + to)
    else:
        _acc_add(acc, piece * 64 + to)
    if mtype == bb.MT_CASTLE:
        if to == 6:
            _acc_sub(acc, np.int64(us + bb.ROOK) * 64 + 7)
            _acc_add(acc, np.int64(us + bb.ROOK) * 64 + 5)
        elif to == 2:
            _acc_sub(acc, np.int64(us + bb.ROOK) * 64 + 0)
            _acc_add(acc, np.int64(us + bb.ROOK) * 64 + 3)
        elif to == 62:
            _acc_sub(acc, np.int64(us + bb.ROOK) * 64 + 63)
            _acc_add(acc, np.int64(us + bb.ROOK) * 64 + 61)
        else:
            _acc_sub(acc, np.int64(us + bb.ROOK) * 64 + 56)
            _acc_add(acc, np.int64(us + bb.ROOK) * 64 + 59)


@njit(cache=False)
def acc_push(
    acc: np.ndarray, stack: np.ndarray, ply: np.int64, mb: np.ndarray, st: np.ndarray, mv: np.int64
) -> None:
    stack[ply, :] = acc
    acc_apply_move(acc, mb, st, mv)


@njit(cache=False)
def acc_pop(acc: np.ndarray, stack: np.ndarray, ply: np.int64) -> None:
    acc[:] = stack[ply]


@njit(cache=False)
def nnue_affine(acc: np.ndarray, st: np.ndarray) -> np.int64:
    """MLP + tanh on a filled accumulator. `acc` is white-oriented, pre-CReLU.

    CReLU is applied while multiplying so a node does 64×16 fused multiply-adds
    and no heap traffic. Allocating a hidden copy here was the nps tax.
    """
    if ENABLED[0] == 0:
        return np.int64(0)
    raw = B2[0]
    for j in range(H2):
        total = B1[j]
        for i in range(H1):
            x = acc[i]
            if x < 0.0:
                x = np.float32(0.0)
            elif x > 1.0:
                x = np.float32(1.0)
            total += x * W1[i, j]
        if total < 0.0:
            total = np.float32(0.0)
        elif total > 1.0:
            total = np.float32(1.0)
        raw += total * W2[j]
    value = np.int64(SCALE * np.tanh(raw))
    if value > CLIP:
        value = np.int64(CLIP)
    elif value < -CLIP:
        value = np.int64(-CLIP)
    if st[bb.ST_SIDE] == bb.BLACK:
        value = -value
    return value


@njit(cache=False)
def nnue_delta(bbs: np.ndarray, st: np.ndarray) -> np.int64:
    """Rebuild the accumulator from `bbs` and run the MLP. For tests, not search."""
    acc = np.empty(H1, dtype=np.float32)
    acc_refresh(acc, bbs)
    return nnue_affine(acc, st)

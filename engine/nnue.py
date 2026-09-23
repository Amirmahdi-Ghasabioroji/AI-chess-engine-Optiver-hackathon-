"""Sparse STM NNUE: 768 piece-square → H1 ReLU → H2 ReLU → 1.

Weights ship next to agent.py, in weights/, and in data/. The loader also
opens the submission zip so a missing data/ extract cannot fall back silently.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from numba import njit

from engine.pos import Position
from engine.bb import lsb

N_FEATURES = 768
HIDDEN1 = 160
HIDDEN2 = 32
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "data" / "nnue.npz"
WEIGHT_NAMES = ("nnue.npz", "weights/nnue.npz", "data/nnue.npz")
LOADED_FROM: str | None = None

_w1 = np.zeros((N_FEATURES, HIDDEN1), dtype=np.float32)
_b1 = np.zeros(HIDDEN1, dtype=np.float32)
_w2 = np.zeros((HIDDEN1, HIDDEN2), dtype=np.float32)
_b2 = np.zeros(HIDDEN2, dtype=np.float32)
_w3 = np.zeros((HIDDEN2, 1), dtype=np.float32)
_b3 = np.zeros(1, dtype=np.float32)
_scale = np.float32(400.0)
_clip = np.float32(2000.0)
_ready = False
_idx_buf = np.empty(32, dtype=np.int32)


@njit(cache=False)
def _forward(
    idx: np.ndarray,
    n: np.int32,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    w3: np.ndarray,
    b3: np.ndarray,
    scale: np.float32,
    clip: np.float32,
) -> np.float32:
    hidden = b1.copy()
    for i in range(n):
        hidden += w1[idx[i]]
    for i in range(hidden.shape[0]):
        if hidden[i] < 0.0:
            hidden[i] = 0.0
    h2 = b2.copy()
    for j in range(w2.shape[1]):
        acc = h2[j]
        for i in range(w2.shape[0]):
            acc += hidden[i] * w2[i, j]
        h2[j] = acc if acc > 0.0 else 0.0
    out = b3[0]
    for i in range(w3.shape[0]):
        out += h2[i] * w3[i, 0]
    score = out * scale
    if score > clip:
        score = clip
    elif score < -clip:
        score = -clip
    return np.float32(score)


def _material(pos: Position) -> int:
    vals = (100, 320, 330, 500, 900)
    score = 0
    for pt, val in enumerate(vals):
        score += val * (pos.bb[pt].bit_count() - pos.bb[pt + 6].bit_count())
    return score if pos.side == 0 else -score


@njit(cache=False)
def fill_indices_bbs(bbs: np.ndarray, side: np.int64, buf: np.ndarray) -> np.int32:
    """STM-oriented indices from the jitted `bbs` layout (same as fill_indices)."""
    flip = np.int64(56) if side != 0 else np.int64(0)
    n = np.int32(0)
    for piece in range(12):
        bits = bbs[piece]
        stm_piece = piece
        if side != 0:
            stm_piece = piece + 6 if piece < 6 else piece - 6
        plane = stm_piece * 64
        while bits != 0:
            sq = lsb(bits)
            bits &= bits - 1
            buf[n] = plane + (sq ^ flip)
            n += 1
            if n >= buf.shape[0]:
                return n
    return n


@njit(cache=False)
def evaluate_nnue_bbs(bbs: np.ndarray, side: np.int64) -> np.int64:
    """Side-to-move centipawns from the net alone."""
    buf = np.empty(32, dtype=np.int32)
    n = fill_indices_bbs(bbs, side, buf)
    if n == 0:
        return np.int64(0)
    return np.int64(_forward(buf, n, _w1, _b1, _w2, _b2, _w3, _b3, _scale, _clip))


@njit(cache=False)
def blend_residual(bbs: np.ndarray, side: np.int64, classical: np.int64) -> np.int64:
    """classical + 3/5 clip(nnue - classical). 60% NNUE, 40% classical."""
    buf = np.empty(32, dtype=np.int32)
    n = fill_indices_bbs(bbs, side, buf)
    if n == 0:
        return classical
    nn = np.int64(_forward(buf, n, _w1, _b1, _w2, _b2, _w3, _b3, _scale, _clip))
    delta = nn - classical
    if delta > 600:
        delta = np.int64(600)
    elif delta < -600:
        delta = np.int64(-600)
    return classical + (delta * 3) // 5


def fill_indices(pos: Position, buf: np.ndarray) -> int:
    stm = pos.side
    flip = 56 if stm else 0
    n = 0
    bb = pos.bb
    for piece in range(12):
        bits = bb[piece]
        stm_piece = piece + 6 if (stm and piece < 6) else piece - 6 if stm else piece
        plane = stm_piece * 64
        while bits:
            sq = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            buf[n] = plane + (sq ^ flip)
            n += 1
            if n >= buf.shape[0]:
                return n
    return n


def _unique(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _agent_root_files() -> list[Path]:
    raw = Path(__file__)
    try:
        resolved = raw.resolve()
    except OSError:
        resolved = raw
    cands: list[Path] = []
    for base in (resolved, raw):
        root = base.parent.parent
        if ".zip" in str(root).replace("\\", "/").lower():
            continue
        for name in WEIGHT_NAMES:
            cands.append(root / name)
    return cands


def _fs_fallback() -> list[Path]:
    roots: list[Path] = []
    try:
        roots.append(Path.cwd())
        roots.append(Path.cwd() / "v7")
    except OSError:
        pass
    for entry in sys.path:
        if not entry:
            continue
        path = Path(entry)
        if path.suffix.lower() == ".zip":
            continue
        roots.append(path)
        roots.append(path / "v7")
    extras = (
        "v7.1/nnue.npz",
        "v7.1/weights/nnue.npz",
        "v7.1/data/nnue.npz",
        "v7/nnue.npz",
        "v7/weights/nnue.npz",
        "v7/data/nnue.npz",
    )
    cands: list[Path] = []
    for root in _unique(roots):
        for name in WEIGHT_NAMES + extras:
            cands.append(root / name)
    return _unique(cands)


def _zip_archives() -> list[Path]:
    archives: list[Path] = []
    texts = [str(__file__)]
    try:
        texts.append(str(Path(__file__).resolve()))
    except OSError:
        pass
    for text in texts:
        lowered = text.replace("\\", "/").lower()
        idx = lowered.find(".zip/")
        if idx != -1:
            archives.append(Path(text[: idx + 4]))
        idx = text.lower().find(".zip\\")
        if idx != -1:
            archives.append(Path(text[: idx + 4]))
    for entry in sys.path:
        if entry and str(entry).lower().endswith(".zip"):
            archives.append(Path(entry))
    return _unique(archives)


def _npz_from_bytes(data: bytes) -> np.lib.npyio.NpzFile:
    return np.load(io.BytesIO(data))


def _open_from_zip(archive: Path) -> tuple[np.lib.npyio.NpzFile, str] | None:
    if not archive.is_file():
        return None
    try:
        with zipfile.ZipFile(archive) as zf:
            names = {name.replace("\\", "/") for name in zf.namelist()}
            for member in WEIGHT_NAMES:
                if member in names:
                    return _npz_from_bytes(zf.read(member)), f"{archive}::{member}"
            for name in names:
                if name.rsplit("/", 1)[-1] == "nnue.npz":
                    return _npz_from_bytes(zf.read(name)), f"{archive}::{name}"
    except (OSError, zipfile.BadZipFile, KeyError, ValueError):
        return None
    return None


def _open_from_resources() -> tuple[np.lib.npyio.NpzFile, str] | None:
    try:
        import importlib.resources as ir

        pkg = ir.files("engine")
    except (ModuleNotFoundError, TypeError, OSError, AttributeError):
        return None
    for rel in ("../nnue.npz", "../weights/nnue.npz", "../data/nnue.npz"):
        try:
            item = pkg.joinpath(rel)
            if item.is_file():
                return _npz_from_bytes(item.read_bytes()), f"resources:{rel}"
        except (OSError, FileNotFoundError, ValueError, AttributeError, NotImplementedError):
            continue
    return None


def discover_weights() -> tuple[np.lib.npyio.NpzFile, str] | None:
    for path in _agent_root_files():
        try:
            if path.is_file():
                return np.load(path), str(path)
        except OSError:
            continue
    found = _open_from_resources()
    if found is not None:
        return found
    for archive in _zip_archives():
        found = _open_from_zip(archive)
        if found is not None:
            return found
    for path in _fs_fallback():
        try:
            if path.is_file():
                return np.load(path), str(path)
        except OSError:
            continue
    return None


def load_nnue(path: Path | None = None) -> None:
    global _w1, _b1, _w2, _b2, _w3, _b3, _scale, _clip, _ready
    global HIDDEN1, HIDDEN2, LOADED_FROM
    if path is not None:
        blob = np.load(path)
        LOADED_FROM = str(path)
    else:
        found = discover_weights()
        if found is None:
            raise FileNotFoundError("nnue.npz not found next to agent.py, in weights/, or in data/")
        blob, LOADED_FROM = found
    _w1 = np.ascontiguousarray(blob["fc1_weight"].T, dtype=np.float32)
    _b1 = np.ascontiguousarray(blob["fc1_bias"], dtype=np.float32)
    _w2 = np.ascontiguousarray(blob["fc2_weight"].T, dtype=np.float32)
    _b2 = np.ascontiguousarray(blob["fc2_bias"], dtype=np.float32)
    _w3 = np.ascontiguousarray(blob["fc3_weight"].T, dtype=np.float32)
    _b3 = np.ascontiguousarray(blob["fc3_bias"], dtype=np.float32)
    _scale = np.float32(blob["scale"]) if "scale" in blob.files else np.float32(400.0)
    _clip = np.float32(blob["clip"]) if "clip" in blob.files else np.float32(2000.0)
    if _w1.shape[0] != N_FEATURES:
        raise ValueError(f"bad fc1 {_w1.shape}")
    HIDDEN1 = int(_w1.shape[1])
    HIDDEN2 = int(_w2.shape[1])
    dummy = np.array([0, 64, 700], dtype=np.int32)
    _forward(dummy, np.int32(3), _w1, _b1, _w2, _b2, _w3, _b3, _scale, _clip)
    _ready = True


def is_ready() -> bool:
    return _ready


def evaluate_nnue(pos: Position) -> int:
    n = fill_indices(pos, _idx_buf)
    if n == 0:
        return 0
    score = float(
        _forward(_idx_buf, np.int32(n), _w1, _b1, _w2, _b2, _w3, _b3, _scale, _clip)
    )
    clip = float(_clip)
    if score > clip:
        score = clip
    elif score < -clip:
        score = -clip
    return int(score)


def warmup() -> None:
    if _ready:
        return
    load_nnue()
    evaluate_nnue(Position.start())
    print(f"NNUE: loaded {LOADED_FROM}", flush=True)

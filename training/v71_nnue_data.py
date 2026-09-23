"""STM feature packing shared by v7.1 training and self-play labelling."""

from __future__ import annotations

from pathlib import Path

import chess
import numpy as np

N_FEATURES = 768
MAX_PIECES = 32
V71_ROOT = Path(__file__).resolve().parents[1] / "local" / "engine_snapshots" / "v7.1"


_FEN_PIECE = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}


def pack_stm(board: chess.Board) -> np.ndarray:
    """Indices matching `engine.nnue.fill_indices_bbs` (side-to-move, rank-flipped)."""
    feats = np.full(MAX_PIECES, -1, dtype=np.int32)
    flip = 0 if board.turn == chess.WHITE else 56
    n = 0
    for square, piece in board.piece_map().items():
        stm_piece = (0 if piece.color == board.turn else 6) + (piece.piece_type - 1)
        feats[n] = stm_piece * 64 + (square ^ flip)
        n += 1
        if n >= MAX_PIECES:
            break
    return feats


def pack_fen_stm(fen: str) -> np.ndarray | None:
    """STM indices from a FEN string. No python-chess; order need not match pack_stm."""
    parts = fen.split()
    if len(parts) < 2:
        return None
    layout, stm = parts[0], parts[1]
    if stm not in ("w", "b"):
        return None
    flip = 0 if stm == "w" else 56
    feats = np.full(MAX_PIECES, -1, dtype=np.int32)
    n = 0
    sq = 56
    for rank in layout.split("/"):
        file = 0
        for ch in rank:
            if ch.isdigit():
                file += int(ch)
                continue
            code = _FEN_PIECE.get(ch)
            if code is None or file > 7:
                return None
            stm_piece = code + 6 if (stm == "b" and code < 6) else code - 6 if stm == "b" else code
            feats[n] = stm_piece * 64 + ((sq + file) ^ flip)
            n += 1
            file += 1
            if n >= MAX_PIECES:
                return feats
        sq -= 8
    return feats if n >= 3 else None


def find_stockfish(explicit: Path | None = None) -> Path:
    root = Path(__file__).resolve().parents[1]
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit
    default = root / "tools" / "sf"
    matches = sorted(default.rglob("stockfish*.exe")) + sorted(default.rglob("stockfish*"))
    files = [
        p
        for p in matches
        if p.is_file() and p.suffix.lower() in {".exe", ""} and "wiki" not in p.parts
    ]
    if not files:
        raise FileNotFoundError(f"no Stockfish binary under {default}")
    return files[0]

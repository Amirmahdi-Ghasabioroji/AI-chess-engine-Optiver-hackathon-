"""Compact opening book: Polyglot Zobrist → weighted UCI moves.

Built at import from mainlines so curated near-equal starts still hit
when they transpose into a known line.
"""

from __future__ import annotations

import chess
import chess.polyglot

# Each line is a sequence of UCI moves from startpos.
# Weights: later we store (move, weight) per hash; repeating a move in
# several lines increases its weight.
_LINES: tuple[str, ...] = (
    # Open games
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8",
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 e8g8",
    "e2e4 e7e5 g1f3 b8c6 f1b5 g8f6 e1g1 f8c5",
    "e2e4 e7e5 g1f3 b8c6 f1b5 g8f6 e1g1 f6e4 d2d4",
    "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d3 f8c5 e1g1 d7d6",
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d4 e5d4 c3d4 c5b4",
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 d2d3 g8f6 e1g1 d7d6",
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 g8f6 d4c6 b7c6 e4e5",
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 f8c5 d4b3 c5b6",
    "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3 f6e4 d2d4",
    "e2e4 e7e5 f1c4 g8f6 d2d3 c7c6 g1f3 d7d5",
    "e2e4 e7e5 b1c3 g8f6 f1c4 b8c6 d2d3 f8c5",
    "e2e4 e7e5 f2f4 e5f4 g1f3 d7d5 e4d5 g8f6",
    # Sicilian
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 f1e2 e7e5 d4b3 f8e7",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1e3 e7e5 d4b3 c8e6",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 g7g6 c1e3 f8g7 f2f3 e8g8",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 b8c6 c1g5 e7e6 d1d2 f8e7",
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 d7d6 f1e2 e7e5 d4b3 f8e7",
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g7g6 c1e3 f8g7 b1c3 g8f6",
    "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 a7a6 f1d3 g8f6 e1g1 d8c7",
    "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 b8c6 b1c3 d8c7 f1e2 a7a6",
    "e2e4 c7c5 c2c3 g8f6 e4e5 f6d5 d2d4 c5d4 g1f3",
    "e2e4 c7c5 b1c3 b8c6 f2f4 g7g6 g1f3 f8g7",
    "e2e4 c7c5 b1c3 e7e6 g2g3 b8c6 f1g2 g7g6",
    # French / Caro / Pirc / Modern / Scandi
    "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4 e4e5 c7c5 a2a3 b4c3 b2c3",
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 c1g5 f8e7 e4e5 f6d7 g5e7 d8e7",
    "e2e4 e7e6 d2d4 d7d5 b1d2 g8f6 e4e5 f6d7 f1d3 c7c5 c2c3",
    "e2e4 e7e6 d2d4 d7d5 e4e5 c7c5 c2c3 b8c6 g1f3 d8b6",
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5 e4g3 f5g6 h2h4 h7h6",
    "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5 g1f3 e7e6 f1e2",
    "e2e4 c7c6 d2d4 d7d5 e4d5 c6d5 c2c4 g8f6 b1c3",
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6 f2f4 f8g7 g1f3 e8g8",
    "e2e4 g7g6 d2d4 f8g7 b1c3 d7d6 f2f4 g8f6 g1f3",
    "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5 d2d4 g8f6",
    "e2e4 g8f6 e4e5 f6d5 d2d4 d7d6 g1f3",
    # Queen's pawn
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c4d5 e6d5 c1g5 c7c6 e2e3",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 g1f3",
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 e7e6 e2e3 b8d7",
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 d5c4 a2a4 c8f5",
    "d2d4 d7d5 c2c4 d5c4 e2e3 g8f6 f1c4 e7e6 g1f3 c7c5",
    "d2d4 d7d5 g1f3 g8f6 c1f4 e7e6 e2e3 f8d6",
    "d2d4 d7d5 g1f3 g8f6 c2c4 e7e6 g2g3 f8e7 f1g2 e8g8",
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 d1c2 e8g8 a2a3 b4c3",
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3 e8g8 f1d3 d7d5",
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6 g2g3 c8b7 f1g2 f8e7",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8",
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5 f6d5 e2e4 d5c3 b2c3 f8g7",
    "d2d4 g8f6 c2c4 g7g6 g1f3 f8g7 g2g3 e8g8 f1g2 d7d6",
    "d2d4 g8f6 c2c4 c7c5 d4d5 e7e6 b1c3 e6d5 c4d5 d7d6",
    "d2d4 g8f6 c2c4 e7e5 d4e5 f6e4 g1f3 b8c6",
    "d2d4 f7f5 c2c4 g8f6 g2g3 e7e6 f1g2 f8e7",
    "d2d4 d7d5 c2c4 e7e6 b1c3 c7c5 c4d5 e6d5 g1f3 b8c6",
    # English / Reti / Flank
    "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6 g2g3 d7d5 c4d5 f6d5",
    "c2c4 e7e5 g2g3 g8f6 f1g2 b8c6 b1c3 f8c5",
    "c2c4 c7c5 g1f3 b8c6 b1c3 g8f6 g2g3 g7g6 f1g2 f8g7",
    "c2c4 g8f6 b1c3 e7e6 e2e4 c7c5 g1f3 b8c6",
    "c2c4 g8f6 g2g3 e7e6 f1g2 d7d5 g1f3",
    "g1f3 g8f6 c2c4 e7e6 g2g3 d7d5 f1g2 f8e7",
    "g1f3 d7d5 g2g3 g8f6 f1g2 c7c6 e1g1 c8f5",
    "g1f3 g8f6 g2g3 g7g6 f1g2 f8g7 e1g1 e8g8",
    "b2b3 e7e5 c1b2 b8c6 e2e3 g8f6",
    "g2g3 d7d5 f1g2 g8f6 g1f3 c7c6",
    # Black responses we want weighted when we sit on the black side of a book line
    "e2e4 e7e5",
    "e2e4 c7c5",
    "e2e4 e7e6",
    "e2e4 c7c6",
    "d2d4 d7d5",
    "d2d4 g8f6",
    "c2c4 e7e5",
    "g1f3 g8f6",
)

Book = dict[int, dict[int, int]]  # hash -> {packed_move: weight}

_PACK_PROMO = {None: 0, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4, chess.QUEEN: 5}


def pack_move(move: chess.Move) -> int:
    return move.from_square | (move.to_square << 6) | (_PACK_PROMO.get(move.promotion, 0) << 12)


def unpack_move(packed: int) -> chess.Move:
    promo_n = (packed >> 12) & 7
    promo = promo_n if promo_n else None
    return chess.Move(packed & 63, (packed >> 6) & 63, promo)


def _build() -> Book:
    book: Book = {}
    for line in _LINES:
        board = chess.Board()
        for token in line.split():
            move = chess.Move.from_uci(token)
            if move not in board.legal_moves:
                break
            key = chess.polyglot.zobrist_hash(board)
            packed = pack_move(move)
            slot = book.setdefault(key, {})
            slot[packed] = slot.get(packed, 0) + 1
            board.push(move)
    return book


BOOK: Book = _build()


def probe(board: chess.Board) -> chess.Move | None:
    """Return a book move, or None. Weighted among stored replies."""
    entries = BOOK.get(chess.polyglot.zobrist_hash(board))
    if not entries:
        return None
    # Deterministic: highest weight, then lowest packed value (stable).
    packed = max(entries.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    move = unpack_move(packed)
    if move in board.legal_moves:
        return move
    return None

"""Bridge between python-chess and the bitboard arrays in `bb`.

Parsing a FEN and printing a UCI move are the only places the engine touches
python-chess, so they stay in plain Python. Everything between them is jitted.
"""

from __future__ import annotations

import chess
import numpy as np

import bb

_PROMO_CHAR = {1: "n", 2: "b", 3: "r", 4: "q"}
_PROMO_CODE = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
_SQUARE_NAME = tuple(chess.SQUARE_NAMES)

_PIECE_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


def from_board(board: chess.Board) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (bbs, st, mb) from a python-chess board."""
    bbs = np.zeros(bb.N_BB, dtype=np.int64)
    st = np.zeros(bb.N_ST, dtype=np.int64)
    mb = np.full(64, -1, dtype=np.int8)

    for colour_index, colour in enumerate((chess.WHITE, chess.BLACK)):
        for type_index, piece_type in enumerate(_PIECE_ORDER):
            mask = board.pieces_mask(piece_type, colour)
            code = colour_index * 6 + type_index
            bbs[code] = bb._u64(mask)
            bbs[bb.OCC_W + colour_index] |= bbs[code]
            while mask:
                square = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                mb[square] = code

    st[bb.ST_SIDE] = bb.WHITE if board.turn == chess.WHITE else bb.BLACK
    rights = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        rights |= bb.CR_WK
    if board.has_queenside_castling_rights(chess.WHITE):
        rights |= bb.CR_WQ
    if board.has_kingside_castling_rights(chess.BLACK):
        rights |= bb.CR_BK
    if board.has_queenside_castling_rights(chess.BLACK):
        rights |= bb.CR_BQ
    st[bb.ST_CASTLE] = rights

    # Match the convention `bb.make_move` uses: an ep square only counts when an
    # enemy pawn actually attacks it, so a parsed position and a searched one hash
    # the same.
    ep = board.ep_square
    st[bb.ST_EP] = bb.NO_EP
    if ep is not None:
        side = int(st[bb.ST_SIDE])
        attackers = int(bb.PAWN_ATT[(1 - side) * 64 + ep]) & int(bbs[side * 6 + bb.PAWN])
        if attackers:
            st[bb.ST_EP] = ep

    st[bb.ST_HALF] = min(board.halfmove_clock, 100)
    st[bb.ST_HASH] = bb.compute_hash(bbs, st)
    return bbs, st, mb


def move_to_uci(mv: int) -> str:
    frm = mv & 63
    to = (mv >> 6) & 63
    promo = (mv >> 12) & 7
    text = _SQUARE_NAME[frm] + _SQUARE_NAME[to]
    return text + _PROMO_CHAR[promo] if promo else text


def move_from_chess(move: chess.Move, board: chess.Board) -> int:
    """Encode a python-chess move the way `bb.gen_moves` would."""
    frm, to = move.from_square, move.to_square
    promo = _PROMO_CODE.get(move.promotion, 0) if move.promotion else 0
    if board.is_en_passant(move):
        mtype = bb.MT_EP
    elif board.is_castling(move):
        mtype = bb.MT_CASTLE
    elif board.piece_type_at(frm) == chess.PAWN and abs(to - frm) == 16:
        mtype = bb.MT_DOUBLE
    else:
        mtype = bb.MT_NORMAL
    return frm | (to << 6) | (promo << 12) | (mtype << 15)


def position_key(board: chess.Board) -> int:
    """The engine's Zobrist key for a board, for repetition tracking."""
    _, st, _ = from_board(board)
    return int(st[bb.ST_HASH])

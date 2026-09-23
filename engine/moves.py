"""Pseudo-legal then king-safe legal move generation."""

from __future__ import annotations

from engine.attacks import KING_ATT, KNIGHT_ATT, bishop_attacks, queen_attacks, rook_attacks
from engine.const import (
    BISHOP,
    BKC,
    BLACK,
    BQC,
    FILE_A,
    FILE_H,
    FLAG_CAP,
    FLAG_CASTLE,
    FLAG_DOUBLE,
    FLAG_EP,
    FLAG_PROMO,
    KNIGHT,
    MASK64,
    PROMO_B,
    PROMO_N,
    PROMO_Q,
    PROMO_R,
    QUEEN,
    RANK_1,
    RANK_3,
    RANK_6,
    RANK_8,
    ROOK,
    WHITE,
    WKC,
    WQC,
    pack_move,
)
from engine.pos import Position

_PROMOS = (PROMO_Q, PROMO_N, PROMO_R, PROMO_B)


def _promo_or_quiet(moves: list[int], frm: int, to: int, flags: int, promo: bool) -> None:
    if promo:
        for p in _PROMOS:
            moves.append(pack_move(frm, to, p, flags | FLAG_PROMO))
    else:
        moves.append(pack_move(frm, to, 0, flags))


def gen_pseudo(pos: Position, captures_only: bool = False) -> list[int]:
    moves: list[int] = []
    us, them = pos.side, pos.side ^ 1
    occ, empty, enemy = pos.all, (~pos.all) & MASK64, pos.occ[them]
    friendly = pos.occ[us]
    bb = pos.bb
    pawn_i = us * 6
    pawns = bb[pawn_i]

    if us == WHITE:
        push = (pawns << 8) & empty
        dbl = ((push & RANK_3) << 8) & empty
        cap_a = ((pawns & ~FILE_A) << 7) & enemy
        cap_h = ((pawns & ~FILE_H) << 9) & enemy
        if pos.ep >= 0:
            ep_bb = 1 << pos.ep
            epa = ((pawns & ~FILE_A) << 7) & ep_bb
            eph = ((pawns & ~FILE_H) << 9) & ep_bb
        else:
            epa = eph = 0
        promo_mask = RANK_8
        bits = push if not captures_only else (push & promo_mask)
        while bits:
            to = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            _promo_or_quiet(moves, to - 8, to, 0, bool((1 << to) & promo_mask))
        if not captures_only:
            bits = dbl
            while bits:
                to = (bits & -bits).bit_length() - 1
                bits &= bits - 1
                moves.append(pack_move(to - 16, to, 0, FLAG_DOUBLE))
        bits = cap_a
        while bits:
            to = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            _promo_or_quiet(moves, to - 7, to, FLAG_CAP, bool((1 << to) & promo_mask))
        bits = cap_h
        while bits:
            to = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            _promo_or_quiet(moves, to - 9, to, FLAG_CAP, bool((1 << to) & promo_mask))
        bits = epa
        while bits:
            to = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            moves.append(pack_move(to - 7, to, 0, FLAG_CAP | FLAG_EP))
        bits = eph
        while bits:
            to = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            moves.append(pack_move(to - 9, to, 0, FLAG_CAP | FLAG_EP))
    else:
        push = (pawns >> 8) & empty
        dbl = ((push & RANK_6) >> 8) & empty
        cap_h = ((pawns & ~FILE_H) >> 7) & enemy
        cap_a = ((pawns & ~FILE_A) >> 9) & enemy
        if pos.ep >= 0:
            ep_bb = 1 << pos.ep
            eph = ((pawns & ~FILE_H) >> 7) & ep_bb
            epa = ((pawns & ~FILE_A) >> 9) & ep_bb
        else:
            epa = eph = 0
        promo_mask = RANK_1
        bits = push if not captures_only else (push & promo_mask)
        while bits:
            to = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            _promo_or_quiet(moves, to + 8, to, 0, bool((1 << to) & promo_mask))
        if not captures_only:
            bits = dbl
            while bits:
                to = (bits & -bits).bit_length() - 1
                bits &= bits - 1
                moves.append(pack_move(to + 16, to, 0, FLAG_DOUBLE))
        bits = cap_h
        while bits:
            to = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            _promo_or_quiet(moves, to + 7, to, FLAG_CAP, bool((1 << to) & promo_mask))
        bits = cap_a
        while bits:
            to = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            _promo_or_quiet(moves, to + 9, to, FLAG_CAP, bool((1 << to) & promo_mask))
        bits = eph
        while bits:
            to = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            moves.append(pack_move(to + 7, to, 0, FLAG_CAP | FLAG_EP))
        bits = epa
        while bits:
            to = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            moves.append(pack_move(to + 9, to, 0, FLAG_CAP | FLAG_EP))

    target = enemy if captures_only else (~friendly & MASK64)

    def _slider(piece_bb: int, attacks_fn) -> None:
        bits = piece_bb
        while bits:
            frm = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            att = attacks_fn(frm, occ) & target
            while att:
                to = (att & -att).bit_length() - 1
                att &= att - 1
                flags = FLAG_CAP if (1 << to) & enemy else 0
                moves.append(pack_move(frm, to, 0, flags))

    bits = bb[pawn_i + KNIGHT]
    while bits:
        frm = (bits & -bits).bit_length() - 1
        bits &= bits - 1
        att = KNIGHT_ATT[frm] & target
        while att:
            to = (att & -att).bit_length() - 1
            att &= att - 1
            flags = FLAG_CAP if (1 << to) & enemy else 0
            moves.append(pack_move(frm, to, 0, flags))
    _slider(bb[pawn_i + BISHOP], bishop_attacks)
    _slider(bb[pawn_i + ROOK], rook_attacks)
    _slider(bb[pawn_i + QUEEN], queen_attacks)

    frm = pos.king[us]
    att = KING_ATT[frm] & target
    while att:
        to = (att & -att).bit_length() - 1
        att &= att - 1
        flags = FLAG_CAP if (1 << to) & enemy else 0
        moves.append(pack_move(frm, to, 0, flags))
    if not captures_only:
        _castle(pos, moves, us, occ)
    return moves


def _safe(pos: Position, squares: tuple[int, ...], by: int) -> bool:
    return all(not pos.is_attacked(sq, by) for sq in squares)


def _castle(pos: Position, moves: list[int], us: int, occ: int) -> None:
    cr = pos.castle
    if us == WHITE:
        if cr & WKC and not (occ & 0x60) and _safe(pos, (4, 5, 6), BLACK):
            moves.append(pack_move(4, 6, 0, FLAG_CASTLE))
        if cr & WQC and not (occ & 0x0E) and _safe(pos, (4, 3, 2), BLACK):
            moves.append(pack_move(4, 2, 0, FLAG_CASTLE))
    else:
        if cr & BKC and not (occ & 0x6000000000000000) and _safe(pos, (60, 61, 62), WHITE):
            moves.append(pack_move(60, 62, 0, FLAG_CASTLE))
        if cr & BQC and not (occ & 0x0E00000000000000) and _safe(pos, (60, 59, 58), WHITE):
            moves.append(pack_move(60, 58, 0, FLAG_CASTLE))


def gen_legal(pos: Position, captures_only: bool = False) -> list[int]:
    out: list[int] = []
    us = pos.side
    for move in gen_pseudo(pos, captures_only=captures_only):
        pos.make(move)
        legal = not pos.in_check(us)
        pos.unmake()
        if legal:
            out.append(move)
    return out


def match_uci(pos: Position, uci: str) -> int:
    from engine.const import move_uci

    for move in gen_legal(pos):
        if move_uci(move) == uci:
            return move
    return 0

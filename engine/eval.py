"""Tapered HCE (Texel-style terms) plus NNUE dispatch."""

from __future__ import annotations

from engine.attacks import KNIGHT_ATT, PAWN_ATT, bishop_attacks, rook_attacks
from engine.const import (
    ADJ_FILES,
    BISHOP,
    BLACK,
    FILE_A,
    FILE_BB,
    FILE_H,
    ISOLATED,
    KING_ZONE,
    KNIGHT,
    MASK64,
    PASSED,
    QUEEN,
    ROOK,
    WHITE,
)
from engine.pos import Position

BISHOP_PAIR = (28, 42)
ROOK_OPEN = (18, 10)
ROOK_SEMI = (8, 6)
ROOK_7TH = (16, 22)
ISOLATED_P = (12, 16)
DOUBLED_P = (8, 14)
BACKWARD_P = (6, 10)
CONNECTED_P = (4, 6)
TEMPO = 10
PASSED_MG = [0, 2, 6, 12, 22, 38, 60, 0]
PASSED_EG = [0, 8, 16, 28, 48, 80, 140, 0]
PROT_PASS = (10, 16)
MOB_N, MOB_B, MOB_R, MOB_Q = (4, 5), (3, 4), (2, 3), (1, 1)
_SHIELD = ((1, 2), (-1, -2))

_nnue_fn = None


def evaluate(pos: Position) -> int:
    global _nnue_fn
    if _nnue_fn is not None:
        return _nnue_fn(pos)
    from engine.nnue import evaluate_nnue, is_ready

    if is_ready():
        _nnue_fn = evaluate_nnue
        return _nnue_fn(pos)
    raise RuntimeError("NNUE is required; classical eval is not a search fallback")


def evaluate_classical(pos: Position) -> int:
    mg, eg = pos.psq_mg, pos.psq_eg
    wp, bp = pos.bb[0], pos.bb[6]
    if pos.bb[2].bit_count() >= 2:
        mg += BISHOP_PAIR[0]
        eg += BISHOP_PAIR[1]
    if pos.bb[8].bit_count() >= 2:
        mg -= BISHOP_PAIR[0]
        eg -= BISHOP_PAIR[1]
    dmg, deg = _pawns(wp, bp)
    mg += dmg
    eg += deg
    dmg, deg = _rooks(pos.bb[3], wp, wp | bp, WHITE)
    mg += dmg
    eg += deg
    dmg, deg = _rooks(pos.bb[9], bp, wp | bp, BLACK)
    mg -= dmg
    eg -= deg
    dmg, deg = _king_safety(pos, WHITE)
    mg += dmg
    eg += deg
    dmg, deg = _king_safety(pos, BLACK)
    mg -= dmg
    eg -= deg
    dmg, deg = _mobility(pos)
    mg += dmg
    eg += deg
    dmg, deg = _mop_up(pos)
    mg += dmg
    eg += deg
    mg += TEMPO if pos.side == WHITE else -TEMPO
    phase = max(0, min(24, pos.phase))
    score = (mg * phase + eg * (24 - phase)) // 24
    return score if pos.side == WHITE else -score


def complexity(pos: Position, nmoves: int, ncap: int) -> float:
    cx = 1.0
    if nmoves > 35:
        cx += 0.12
    if ncap >= 4:
        cx += 0.15
    if pos.in_check():
        cx += 0.25
    if pos.phase <= 8:
        cx += 0.08
    return min(1.7, cx)


def _pawns(wp: int, bp: int) -> tuple[int, int]:
    mg = eg = 0
    w_att = (((wp & ~FILE_A) << 7) | ((wp & ~FILE_H) << 9)) & MASK64
    b_att = ((bp & ~FILE_H) >> 7) | ((bp & ~FILE_A) >> 9)
    bits = wp
    while bits:
        sq = (bits & -bits).bit_length() - 1
        bits &= bits - 1
        f, r = sq & 7, sq >> 3
        file_p = FILE_BB[f] & wp
        if file_p & (file_p - 1):
            mg -= DOUBLED_P[0]
            eg -= DOUBLED_P[1]
        if not (ISOLATED[sq] & wp):
            mg -= ISOLATED_P[0]
            eg -= ISOLATED_P[1]
        elif not (ADJ_FILES[f] & wp & ((1 << sq) - 1)):
            stop = sq + 8
            if stop < 64 and (b_att & (1 << stop)):
                mg -= BACKWARD_P[0]
                eg -= BACKWARD_P[1]
        if w_att & (1 << sq):
            mg += CONNECTED_P[0]
            eg += CONNECTED_P[1]
        if not (PASSED[WHITE][sq] & bp):
            mg += PASSED_MG[r]
            eg += PASSED_EG[r]
            if w_att & (1 << sq):
                mg += PROT_PASS[0]
                eg += PROT_PASS[1]
    bits = bp
    while bits:
        sq = (bits & -bits).bit_length() - 1
        bits &= bits - 1
        f, r = sq & 7, sq >> 3
        file_p = FILE_BB[f] & bp
        if file_p & (file_p - 1):
            mg += DOUBLED_P[0]
            eg += DOUBLED_P[1]
        if not (ISOLATED[sq] & bp):
            mg += ISOLATED_P[0]
            eg += ISOLATED_P[1]
        elif not (ADJ_FILES[f] & bp & ~((1 << (sq + 1)) - 1)):
            stop = sq - 8
            if stop >= 0 and (w_att & (1 << stop)):
                mg += BACKWARD_P[0]
                eg += BACKWARD_P[1]
        if b_att & (1 << sq):
            mg -= CONNECTED_P[0]
            eg -= CONNECTED_P[1]
        if not (PASSED[BLACK][sq] & wp):
            rr = 7 - r
            mg -= PASSED_MG[rr]
            eg -= PASSED_EG[rr]
            if b_att & (1 << sq):
                mg -= PROT_PASS[0]
                eg -= PROT_PASS[1]
    return mg, eg


def _rooks(rooks: int, own_p: int, all_p: int, color: int) -> tuple[int, int]:
    mg = eg = 0
    seventh = 0x00FF000000000000 if color == WHITE else 0x000000000000FF00
    bits = rooks
    while bits:
        sq = (bits & -bits).bit_length() - 1
        bits &= bits - 1
        f = FILE_BB[sq & 7]
        if not (f & all_p):
            mg += ROOK_OPEN[0]
            eg += ROOK_OPEN[1]
        elif not (f & own_p):
            mg += ROOK_SEMI[0]
            eg += ROOK_SEMI[1]
        if (1 << sq) & seventh:
            mg += ROOK_7TH[0]
            eg += ROOK_7TH[1]
    return mg, eg


def _king_safety(pos: Position, color: int) -> tuple[int, int]:
    ksq = pos.king[color]
    pawns = pos.bb[color * 6]
    f, r = ksq & 7, ksq >> 3
    shield = 0
    ranks = _SHIELD[color]
    for df in (-1, 0, 1):
        ff = f + df
        if not 0 <= ff < 8:
            continue
        found = 0
        for i, dr in enumerate(ranks):
            rr = r + dr
            if 0 <= rr < 8 and pawns & (1 << (rr * 8 + ff)):
                found = 12 if i == 0 else 7
                break
        shield += found if found else -16
    zone = KING_ZONE[ksq]
    opp = color ^ 1
    bb, occ = pos.bb, pos.all
    attackers = weight = 0
    if PAWN_ATT[color][ksq] & bb[opp * 6]:
        attackers += 1
        weight += 1
    kn = KNIGHT_ATT[ksq] & bb[opp * 6 + KNIGHT]
    if kn:
        n = kn.bit_count()
        attackers += n
        weight += 2 * n
    bq = bb[opp * 6 + BISHOP] | bb[opp * 6 + QUEEN]
    ba = bishop_attacks(ksq, occ) & bq
    if ba:
        n = ba.bit_count()
        attackers += n
        weight += 2 * n
    rq = bb[opp * 6 + ROOK] | bb[opp * 6 + QUEEN]
    ra = rook_attacks(ksq, occ) & rq
    if ra:
        n = ra.bit_count()
        attackers += n
        weight += 3 * n
    danger = shield - 4 * attackers * attackers - 6 * weight - 3 * (zone & pos.occ[opp]).bit_count()
    return danger, danger // 8


def _mobility(pos: Position) -> tuple[int, int]:
    occ = pos.all
    mg = eg = 0
    for color, sign in ((WHITE, 1), (BLACK, -1)):
        safe = ~pos.occ[color] & MASK64
        n = 0
        bits = pos.bb[color * 6 + KNIGHT]
        while bits:
            sq = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            n += (KNIGHT_ATT[sq] & safe).bit_count()
        mg += sign * n * MOB_N[0]
        eg += sign * n * MOB_N[1]
        n = 0
        bits = pos.bb[color * 6 + BISHOP]
        while bits:
            sq = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            n += (bishop_attacks(sq, occ) & safe).bit_count()
        mg += sign * n * MOB_B[0]
        eg += sign * n * MOB_B[1]
        n = 0
        bits = pos.bb[color * 6 + ROOK]
        while bits:
            sq = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            n += (rook_attacks(sq, occ) & safe).bit_count()
        mg += sign * n * MOB_R[0]
        eg += sign * n * MOB_R[1]
        n = 0
        bits = pos.bb[color * 6 + QUEEN]
        while bits:
            sq = (bits & -bits).bit_length() - 1
            bits &= bits - 1
            n += ((bishop_attacks(sq, occ) | rook_attacks(sq, occ)) & safe).bit_count()
        mg += sign * n * MOB_Q[0]
        eg += sign * n * MOB_Q[1]
    return mg, eg


def _mop_up(pos: Position) -> tuple[int, int]:
    if pos.phase > 4 or (pos.bb[0] | pos.bb[6]):
        return 0, 0
    wk, bk = pos.king[WHITE], pos.king[BLACK]
    wf, wr = wk & 7, wk >> 3
    bf, br = bk & 7, bk >> 3
    opp_center = abs(bf - 3.5) + abs(br - 3.5)
    md = abs(wf - bf) + abs(wr - br)
    w_mat = 100 * pos.bb[0].bit_count() + 320 * pos.bb[1].bit_count() + 330 * pos.bb[2].bit_count()
    w_mat += 500 * pos.bb[3].bit_count() + 900 * pos.bb[4].bit_count()
    b_mat = 100 * pos.bb[6].bit_count() + 320 * pos.bb[7].bit_count() + 330 * pos.bb[8].bit_count()
    b_mat += 500 * pos.bb[9].bit_count() + 900 * pos.bb[10].bit_count()
    if w_mat > b_mat + 200:
        eg = int(4 * opp_center - 2 * md)
        return 0, eg
    if b_mat > w_mat + 200:
        w_center = abs(wf - 3.5) + abs(wr - 3.5)
        return 0, -int(4 * w_center - 2 * md)
    return 0, 0

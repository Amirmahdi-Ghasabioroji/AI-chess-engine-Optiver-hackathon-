"""Tapered PeSTO evaluation and a full static exchange, on bitboards.

The terms and their weights are the ones `evaluate.py` already tuned, ported so
that the jitted search never has to cross back into python-chess. Scores are
centipawns from the side to move.
"""

from __future__ import annotations

import numpy as np
from numba import njit

import bb
from bb import (
    BISHOP,
    KING,
    KING_ATT,
    KNIGHT,
    KNIGHT_ATT,
    NOT_FILE_A,
    NOT_FILE_H,
    OCC_B,
    OCC_W,
    PAWN,
    QUEEN,
    ROOK,
    ST_CASTLE,
    ST_SIDE,
    WHITE,
    bishop_attacks,
    lsb,
    lsr,
    popcount,
    queen_attacks,
    rook_attacks,
)
from evaluate import (
    _EG_BISHOP,
    _EG_KING,
    _EG_KNIGHT,
    _EG_PAWN,
    _EG_QUEEN,
    _EG_ROOK,
    _MG_BISHOP,
    _MG_KING,
    _MG_KNIGHT,
    _MG_PAWN,
    _MG_QUEEN,
    _MG_ROOK,
    EG_VALUE,
    MG_VALUE,
)

PHASE_MAX = 24
TEMPO = 10

SEE_VALUE = np.array([100, 320, 330, 500, 900, 20000], dtype=np.int64)
PHASE_WEIGHT = np.array([0, 1, 1, 2, 4, 0], dtype=np.int64)

_MG_PST = (_MG_PAWN, _MG_KNIGHT, _MG_BISHOP, _MG_ROOK, _MG_QUEEN, _MG_KING)
_EG_PST = (_EG_PAWN, _EG_KNIGHT, _EG_BISHOP, _EG_ROOK, _EG_QUEEN, _EG_KING)

# MG_TABLE[code * 64 + square] folds material into the piece-square value and is
# already oriented for that colour, so evaluation is one lookup per piece.
MG_TABLE = np.zeros(12 * 64, dtype=np.int64)
EG_TABLE = np.zeros(12 * 64, dtype=np.int64)
for _type in range(6):
    for _square in range(64):
        MG_TABLE[_type * 64 + _square] = MG_VALUE[_type + 1] + _MG_PST[_type][_square ^ 56]
        EG_TABLE[_type * 64 + _square] = EG_VALUE[_type + 1] + _EG_PST[_type][_square ^ 56]
        MG_TABLE[(_type + 6) * 64 + _square] = MG_VALUE[_type + 1] + _MG_PST[_type][_square]
        EG_TABLE[(_type + 6) * 64 + _square] = EG_VALUE[_type + 1] + _EG_PST[_type][_square]

FILE_BB = np.array([bb._u64(0x0101010101010101 << f) for f in range(8)], dtype=np.int64)
RANK_BB = np.array([bb._u64(0xFF << (8 * r)) for r in range(8)], dtype=np.int64)

_ADJACENT = [
    (0x0101010101010101 << (f - 1) if f > 0 else 0) | (0x0101010101010101 << (f + 1) if f < 7 else 0)
    for f in range(8)
]
ISOLATED_FILE = np.array([bb._u64(m) for m in _ADJACENT], dtype=np.int64)


def _passed_mask(square: int, white: bool) -> int:
    f, r = square & 7, square >> 3
    span = _ADJACENT[f] | (0x0101010101010101 << f)
    mask = 0
    ranks = range(r + 1, 8) if white else range(r - 1, -1, -1)
    for rr in ranks:
        mask |= span & (0xFF << (8 * rr))
    return mask


PASSED_W = np.array([bb._u64(_passed_mask(s, True)) for s in range(64)], dtype=np.int64)
PASSED_B = np.array([bb._u64(_passed_mask(s, False)) for s in range(64)], dtype=np.int64)

PASSED_MG = np.array([0, 2, 6, 12, 24, 48, 80, 0], dtype=np.int64)
PASSED_EG = np.array([0, 8, 16, 32, 56, 96, 160, 0], dtype=np.int64)

# Pawn-structure cache keyed on the two pawn bitboards; a collision just recomputes.
# Numba exposes module-level arrays read-only, so the cache travels as an argument:
# columns are (key, mg, eg) and the key is forced odd so an empty slot never matches.
PAWN_HASH_BITS = 14
PAWN_HASH_SIZE = 1 << PAWN_HASH_BITS


def new_pawn_cache() -> np.ndarray:
    return np.zeros((PAWN_HASH_SIZE, 3), dtype=np.int64)


@njit(cache=False)
def pawn_terms(wp: np.int64, bp: np.int64) -> tuple:
    mg = np.int64(0)
    eg = np.int64(0)

    b = wp
    while b != 0:
        sq = lsb(b)
        b &= b - 1
        f = sq & 7
        r = sq >> 3
        ahead = FILE_BB[f] & ~((bb.BIT[sq] - 1) | bb.BIT[sq])
        if wp & ahead:
            mg -= 8
            eg -= 18
        if not (wp & ISOLATED_FILE[f]):
            mg -= 12
            eg -= 16
        if not (bp & PASSED_W[sq]):
            mg += PASSED_MG[r]
            eg += PASSED_EG[r]

    b = bp
    while b != 0:
        sq = lsb(b)
        b &= b - 1
        f = sq & 7
        r = sq >> 3
        behind = FILE_BB[f] & (bb.BIT[sq] - 1)
        if bp & behind:
            mg += 8
            eg += 18
        if not (bp & ISOLATED_FILE[f]):
            mg += 12
            eg += 16
        if not (wp & PASSED_B[sq]):
            mg -= PASSED_MG[7 - r]
            eg -= PASSED_EG[7 - r]

    return mg, eg


@njit(cache=False)
def pawn_terms_hashed(wp: np.int64, bp: np.int64, pc: np.ndarray) -> tuple:
    key = ((wp * -7046029254386353131) ^ (bp * -4417276706812531889)) | 1
    i = key & (PAWN_HASH_SIZE - 1)
    if pc[i, 0] == key:
        return pc[i, 1], pc[i, 2]
    mg, eg = pawn_terms(wp, bp)
    pc[i, 0] = key
    pc[i, 1] = mg
    pc[i, 2] = eg
    return mg, eg


@njit(cache=False)
def king_shield(king_sq: np.int64, pawns: np.int64, white: bool) -> np.int64:
    """Pawn cover on the three files in front of the king.

    Centre kings (files d/e) still get a small penalty: leaving the king on e1
    with the queen on the board is how short games end.
    """
    f = king_sq & 7
    r = king_sq >> 3
    centre = 2 < f < 5
    shield_rank = r + 1 if white else r - 1
    bonus = np.int64(-18) if centre else np.int64(0)
    if shield_rank < 0 or shield_rank > 7:
        return bonus
    for df in range(-1, 2):
        ff = f + df
        if 0 <= ff <= 7:
            if pawns & bb.BIT[shield_rank * 8 + ff]:
                bonus += 18 if not centre else 10
            else:
                bonus -= 12 if not centre else 8
            far = shield_rank + 1 if white else shield_rank - 1
            if 0 <= far <= 7 and pawns & bb.BIT[far * 8 + ff]:
                bonus += 6 if not centre else 3
    return bonus


@njit(cache=False)
def king_file_weakness(
    king_sq: np.int64, our_pawns: np.int64, enemy_pawns: np.int64
) -> np.int64:
    """Open and half-open files next to the king invite rooks and queens."""
    f = king_sq & 7
    penalty = np.int64(0)
    for df in range(-1, 2):
        ff = f + df
        if ff < 0 or ff > 7:
            continue
        file_bb = FILE_BB[ff]
        ours = our_pawns & file_bb
        theirs = enemy_pawns & file_bb
        if ours == 0:
            penalty += 18
            if theirs == 0:
                penalty += 14
        elif theirs == 0:
            penalty += 8
    return penalty


@njit(cache=False)
def king_attackers(king_sq: np.int64, bbs: np.ndarray, enemy: np.int64, occ: np.int64) -> np.int64:
    """Enemy pieces that look at the king or its adjacent squares."""
    ring = KING_ATT[king_sq] | bb.BIT[king_sq]
    base = enemy * 6
    score = np.int64(0)
    b = bbs[base + KNIGHT]
    while b != 0:
        sq = lsb(b)
        b &= b - 1
        if KNIGHT_ATT[sq] & ring:
            score += 20
    b = bbs[base + BISHOP]
    while b != 0:
        sq = lsb(b)
        b &= b - 1
        if bishop_attacks(sq, occ) & ring:
            score += 14
    b = bbs[base + ROOK]
    while b != 0:
        sq = lsb(b)
        b &= b - 1
        if rook_attacks(sq, occ) & ring:
            score += 20
    b = bbs[base + QUEEN]
    while b != 0:
        sq = lsb(b)
        b &= b - 1
        if queen_attacks(sq, occ) & ring:
            score += 36
    return score


@njit(cache=False)
def _king_flights(king_sq: np.int64, own: np.int64, enemy_pawn_att: np.int64) -> np.int64:
    """Fewer safe king steps means tactics resolve by force."""
    flights = popcount(KING_ATT[king_sq] & ~own & ~enemy_pawn_att)
    if flights >= 3:
        return np.int64(0)
    return np.int64((3 - flights) * 12)


@njit(cache=False)
def pin_pressure(king_sq: np.int64, bbs: np.ndarray, us: np.int64, occ: np.int64) -> np.int64:
    """Penalty when exactly one of our pieces sits between the king and an enemy slider."""
    them = 1 - us
    our = bbs[OCC_W + us]
    enemy_diag = bbs[them * 6 + BISHOP] | bbs[them * 6 + QUEEN]
    enemy_orth = bbs[them * 6 + ROOK] | bbs[them * 6 + QUEEN]
    penalty = np.int64(0)

    b = bishop_attacks(king_sq, occ) & our
    while b != 0:
        sq = lsb(b)
        b &= b - 1
        if bishop_attacks(king_sq, occ ^ bb.BIT[sq]) & enemy_diag:
            if bbs[us * 6 + QUEEN] & bb.BIT[sq]:
                penalty += 28
            elif bbs[us * 6 + ROOK] & bb.BIT[sq]:
                penalty += 18
            elif bbs[us * 6 + KNIGHT] & bb.BIT[sq]:
                penalty += 14
            else:
                penalty += 8

    b = rook_attacks(king_sq, occ) & our
    while b != 0:
        sq = lsb(b)
        b &= b - 1
        if rook_attacks(king_sq, occ ^ bb.BIT[sq]) & enemy_orth:
            if bbs[us * 6 + QUEEN] & bb.BIT[sq]:
                penalty += 32
            elif bbs[us * 6 + KNIGHT] & bb.BIT[sq]:
                penalty += 16
            elif bbs[us * 6 + BISHOP] & bb.BIT[sq]:
                penalty += 14
            else:
                penalty += 8
    return penalty


@njit(cache=False)
def _pawn_threats(watt: np.int64, batt: np.int64, bbs: np.ndarray) -> np.int64:
    """Pawns attacking unprotected-looking pieces: the cheap 'free take' signal."""
    w_minors = bbs[KNIGHT] | bbs[BISHOP]
    w_majors = bbs[ROOK] | bbs[QUEEN]
    b_minors = bbs[6 + KNIGHT] | bbs[6 + BISHOP]
    b_majors = bbs[6 + ROOK] | bbs[6 + QUEEN]
    mg = np.int64(0)
    mg += 22 * popcount(watt & b_minors) + 44 * popcount(watt & b_majors)
    mg -= 22 * popcount(batt & w_minors) + 44 * popcount(batt & w_majors)
    return mg


@njit(cache=False)
def pawn_storm(
    king_sq: np.int64, enemy_pawns: np.int64, white_king: bool
) -> np.int64:
    """Enemy pawns advancing on a wing king's files."""
    f = king_sq & 7
    if 2 < f < 5:
        return np.int64(0)
    penalty = np.int64(0)
    for df in range(-1, 2):
        ff = f + df
        if ff < 0 or ff > 7:
            continue
        b = enemy_pawns & FILE_BB[ff]
        while b != 0:
            sq = lsb(b)
            b &= b - 1
            r = sq >> 3
            if white_king:
                if r <= 3:
                    penalty += np.int64((4 - r) * 8)
            elif r >= 4:
                penalty += np.int64((r - 3) * 8)
    return penalty


@njit(cache=False)
def _king_pressure(
    raw: np.int64, enemy_queens: np.int64, enemy_minors: np.int64
) -> np.int64:
    """Attacks without a queen are much less dangerous; a pile-on is worse than linear."""
    if enemy_queens == 0:
        if enemy_minors < 2:
            return raw // 4
        raw = raw // 2
    if raw > 48:
        raw += raw - 48
    return raw


@njit(cache=False)
def _mobility(bbs: np.ndarray) -> tuple:
    occ = bbs[OCC_W] | bbs[OCC_B]
    wsafe = ~bbs[OCC_W]
    bsafe = ~bbs[OCC_B]
    mg = np.int64(0)
    eg = np.int64(0)

    n = np.int64(0)
    b = bbs[KNIGHT]
    while b != 0:
        sq = lsb(b)
        b &= b - 1
        n += popcount(KNIGHT_ATT[sq] & wsafe)
    mg += n * 4
    eg += n * 5
    n = np.int64(0)
    b = bbs[6 + KNIGHT]
    while b != 0:
        sq = lsb(b)
        b &= b - 1
        n += popcount(KNIGHT_ATT[sq] & bsafe)
    mg -= n * 4
    eg -= n * 5

    n = np.int64(0)
    b = bbs[BISHOP]
    while b != 0:
        sq = lsb(b)
        b &= b - 1
        n += popcount(bishop_attacks(sq, occ) & wsafe)
    mg += n * 3
    eg += n * 4
    n = np.int64(0)
    b = bbs[6 + BISHOP]
    while b != 0:
        sq = lsb(b)
        b &= b - 1
        n += popcount(bishop_attacks(sq, occ) & bsafe)
    mg -= n * 3
    eg -= n * 4

    n = np.int64(0)
    b = bbs[ROOK]
    while b != 0:
        sq = lsb(b)
        b &= b - 1
        n += popcount(rook_attacks(sq, occ) & wsafe)
    mg += n * 2
    eg += n * 3
    n = np.int64(0)
    b = bbs[6 + ROOK]
    while b != 0:
        sq = lsb(b)
        b &= b - 1
        n += popcount(rook_attacks(sq, occ) & bsafe)
    mg -= n * 2
    eg -= n * 3

    n = np.int64(0)
    b = bbs[QUEEN]
    while b != 0:
        sq = lsb(b)
        b &= b - 1
        n += popcount((bishop_attacks(sq, occ) | rook_attacks(sq, occ)) & wsafe)
    mg += n * 1
    eg += n * 1
    n = np.int64(0)
    b = bbs[6 + QUEEN]
    while b != 0:
        sq = lsb(b)
        b &= b - 1
        n += popcount((bishop_attacks(sq, occ) | rook_attacks(sq, occ)) & bsafe)
    mg -= n * 1
    eg -= n * 1
    return mg, eg


@njit(cache=False)
def _hanging_material(bbs: np.ndarray) -> np.int64:
    """White-positive: a piece that can be taken for free or by a cheaper unit."""
    occ = bbs[OCC_W] | bbs[OCC_B]
    mg = np.int64(0)
    for code in range(12):
        t = code % 6
        if t == KING:
            continue
        us = np.int64(0) if code < 6 else np.int64(1)
        them = 1 - us
        sign = np.int64(1) if code < 6 else np.int64(-1)
        our = bbs[OCC_W + us]
        their = bbs[OCC_W + them]
        b = bbs[code]
        while b != 0:
            sq = lsb(b)
            b &= b - 1
            att = attackers_to(bbs, sq, occ)
            enemy_att = att & their
            if enemy_att == 0:
                continue
            if (att & our) == 0:
                mg -= sign * SEE_VALUE[t]
                continue
            cheap = np.int64(-1)
            base = them * 6
            for pt in range(6):
                if enemy_att & bbs[base + pt]:
                    cheap = np.int64(pt)
                    break
            if cheap >= 0 and SEE_VALUE[cheap] < SEE_VALUE[t]:
                mg -= sign * (SEE_VALUE[t] - SEE_VALUE[cheap])
    return mg


@njit(cache=False)
def _passed_extras(
    bbs: np.ndarray,
    wp: np.int64,
    bp: np.int64,
    wk: np.int64,
    bk: np.int64,
    watt: np.int64,
    batt: np.int64,
) -> tuple:
    """King tropism, rook behind, protected passer, and a pawn-race term."""
    mg = np.int64(0)
    eg = np.int64(0)
    b = wp
    while b != 0:
        sq = lsb(b)
        b &= b - 1
        if bp & PASSED_W[sq]:
            continue
        r = sq >> 3
        wf = sq & 7
        our = max(abs((wk & 7) - wf), abs((wk >> 3) - r))
        their = max(abs((bk & 7) - wf), abs((bk >> 3) - r))
        eg += np.int64((their - our) * PASSED_EG[r] // 10)
        if bbs[ROOK] & FILE_BB[wf] & (bb.BIT[sq] - 1):
            mg += 10
            eg += 22
        if watt & bb.BIT[sq]:
            mg += 8
            eg += 16
        pmoves = 7 - r
        if r == 1:
            pmoves -= 1
        kdist = max(abs((bk & 7) - wf), abs((bk >> 3) - 7))
        if kdist > pmoves:
            eg += np.int64(24 + 8 * (kdist - pmoves))
        if (wk & 7) == wf and (wk >> 3) > r:
            eg += np.int64(12)
    b = bp
    while b != 0:
        sq = lsb(b)
        b &= b - 1
        if wp & PASSED_B[sq]:
            continue
        r = sq >> 3
        wf = sq & 7
        our = max(abs((bk & 7) - wf), abs((bk >> 3) - r))
        their = max(abs((wk & 7) - wf), abs((wk >> 3) - r))
        eg -= np.int64((their - our) * PASSED_EG[7 - r] // 10)
        if bbs[6 + ROOK] & FILE_BB[wf] & ~(bb.BIT[sq] - 1) & ~bb.BIT[sq]:
            mg -= 10
            eg -= 22
        if batt & bb.BIT[sq]:
            mg -= 8
            eg -= 16
        pmoves = r
        if r == 6:
            pmoves -= 1
        kdist = max(abs((wk & 7) - wf), abs((wk >> 3) - 0))
        if kdist > pmoves:
            eg -= np.int64(24 + 8 * (kdist - pmoves))
        if (bk & 7) == wf and (bk >> 3) < r:
            eg -= np.int64(12)
    return mg, eg


@njit(cache=False)
def non_pawn_material(bbs: np.ndarray, side: np.int64) -> np.int64:
    base = side * 6
    total = np.int64(0)
    for t in range(KNIGHT, KING):
        total += SEE_VALUE[t] * popcount(bbs[base + t])
    return total


@njit(cache=False)
def evaluate(bbs: np.ndarray, st: np.ndarray, pc: np.ndarray) -> np.int64:
    mg = np.int64(0)
    eg = np.int64(0)
    phase = np.int64(0)

    for code in range(12):
        b = bbs[code]
        if b == 0:
            continue
        sign = 1 if code < 6 else -1
        t = code % 6
        phase += PHASE_WEIGHT[t] * popcount(b)
        while b != 0:
            sq = lsb(b)
            b &= b - 1
            mg += sign * MG_TABLE[code * 64 + sq]
            eg += sign * EG_TABLE[code * 64 + sq]

    if popcount(bbs[BISHOP]) >= 2:
        mg += 25
        eg += 40
    if popcount(bbs[6 + BISHOP]) >= 2:
        mg -= 25
        eg -= 40

    wp = bbs[PAWN]
    bp = bbs[6 + PAWN]
    pmg, peg = pawn_terms_hashed(wp, bp, pc)
    mg += pmg
    eg += peg

    all_pawns = wp | bp
    b = bbs[ROOK]
    while b != 0:
        sq = lsb(b)
        b &= b - 1
        file_bb = FILE_BB[sq & 7]
        if not (all_pawns & file_bb):
            mg += 18
            eg += 12
        elif not (wp & file_bb):
            mg += 8
            eg += 6
        if bb.BIT[sq] & RANK_BB[6]:
            mg += 16
            eg += 24
    b = bbs[6 + ROOK]
    while b != 0:
        sq = lsb(b)
        b &= b - 1
        file_bb = FILE_BB[sq & 7]
        if not (all_pawns & file_bb):
            mg -= 18
            eg -= 12
        elif not (bp & file_bb):
            mg -= 8
            eg -= 6
        if bb.BIT[sq] & RANK_BB[1]:
            mg -= 16
            eg -= 24

    wk = lsb(bbs[KING])
    bk = lsb(bbs[6 + KING])
    mg += king_shield(wk, wp, True)
    mg -= king_shield(bk, bp, False)
    mg -= king_file_weakness(wk, wp, bp)
    mg += king_file_weakness(bk, bp, wp)
    occ = bbs[OCC_W] | bbs[OCC_B]
    watt = ((wp & NOT_FILE_A) << 7) | ((wp & NOT_FILE_H) << 9)
    batt = lsr(bp & NOT_FILE_H, 7) | lsr(bp & NOT_FILE_A, 9)
    wq = popcount(bbs[QUEEN])
    bq = popcount(bbs[6 + QUEEN])
    wmin = popcount(bbs[KNIGHT] | bbs[BISHOP])
    bmin = popcount(bbs[6 + KNIGHT] | bbs[6 + BISHOP])
    w_home = wk == 4 and (st[ST_CASTLE] & 3) != 0
    b_home = bk == 60 and (st[ST_CASTLE] & 12) != 0
    if not w_home:
        mg -= _king_pressure(king_attackers(wk, bbs, np.int64(1), occ), bq, bmin)
        mg -= pawn_storm(wk, bp, True)
        mg -= pin_pressure(wk, bbs, np.int64(0), occ)
        mg -= _king_flights(wk, bbs[OCC_W], batt)
    if not b_home:
        mg += _king_pressure(king_attackers(bk, bbs, np.int64(0), occ), wq, wmin)
        mg += pawn_storm(bk, wp, False)
        mg += pin_pressure(bk, bbs, np.int64(1), occ)
        mg += _king_flights(bk, bbs[OCC_B], watt)
    mg += _pawn_threats(watt, batt, bbs)
    mmg, meg = _mobility(bbs)
    mg += mmg
    eg += meg
    pmg2, peg2 = _passed_extras(bbs, wp, bp, wk, bk, watt, batt)
    mg += pmg2
    eg += peg2
    hang = _hanging_material(bbs)
    mg += hang // 2

    if (
        popcount(bbs[BISHOP]) == 1
        and popcount(bbs[6 + BISHOP]) == 1
        and (bbs[ROOK] | bbs[6 + ROOK] | bbs[QUEEN] | bbs[6 + QUEEN]) == 0
    ):
        ws = lsb(bbs[BISHOP])
        bs = lsb(bbs[6 + BISHOP])
        if (((ws & 7) + (ws >> 3)) & 1) != (((bs & 7) + (bs >> 3)) & 1):
            eg = eg * 5 // 8

    if phase > PHASE_MAX:
        phase = PHASE_MAX
    score = (mg * phase + eg * (PHASE_MAX - phase)) // PHASE_MAX

    side = st[ST_SIDE]
    score += TEMPO if side == WHITE else -TEMPO

    if phase <= 12:
        w_material = non_pawn_material(bbs, np.int64(0)) + 100 * popcount(wp)
        b_material = non_pawn_material(bbs, np.int64(1)) + 100 * popcount(bp)
        diff = w_material - b_material
        thresh = 250 if phase <= 8 else 450
        if diff >= thresh or diff <= -thresh:
            chebyshev = max(abs((wk & 7) - (bk & 7)), abs((wk >> 3) - (bk >> 3)))
            close = 6 if phase <= 8 else 3
            corner = 16 if phase <= 8 else 8
            if diff > 0:
                edge = min(min(bk & 7, 7 - (bk & 7)), min(bk >> 3, 7 - (bk >> 3)))
                score += close * (7 - chebyshev) + corner * (3 - edge)
            else:
                edge = min(min(wk & 7, 7 - (wk & 7)), min(wk >> 3, 7 - (wk >> 3)))
                score -= close * (7 - chebyshev) + corner * (3 - edge)

    return score if side == WHITE else -score


# ------------------------------------------------------------------------- SEE


@njit(cache=False)
def attackers_to(bbs: np.ndarray, sq: np.int64, occ: np.int64) -> np.int64:
    """Every piece of either colour that attacks `sq` given occupancy `occ`."""
    result = bb.PAWN_ATT[sq] & bbs[6 + PAWN]
    result |= bb.PAWN_ATT[64 + sq] & bbs[PAWN]
    result |= bb.KNIGHT_ATT[sq] & (bbs[KNIGHT] | bbs[6 + KNIGHT])
    result |= bb.KING_ATT[sq] & (bbs[KING] | bbs[6 + KING])
    diagonal = bishop_attacks(sq, occ)
    result |= diagonal & (bbs[BISHOP] | bbs[QUEEN] | bbs[6 + BISHOP] | bbs[6 + QUEEN])
    straight = rook_attacks(sq, occ)
    result |= straight & (bbs[ROOK] | bbs[QUEEN] | bbs[6 + ROOK] | bbs[6 + QUEEN])
    return result & occ


@njit(cache=False)
def _least_valuable(bbs: np.ndarray, attackers: np.int64, side: np.int64) -> np.int64:
    base = side * 6
    for t in range(6):
        subset = attackers & bbs[base + t]
        if subset != 0:
            return subset & -subset
    return np.int64(0)


@njit(cache=False)
def see(bbs: np.ndarray, mb: np.ndarray, mv: np.int64) -> np.int64:
    """Exact swap-off value of a capture, in centipawns, for the side to move."""
    frm = mv & 63
    to = (mv >> 6) & 63
    promo = (mv >> 12) & 7
    mtype = (mv >> 15) & 3

    moving = np.int64(mb[frm])
    side = moving // 6
    moving_type = moving % 6

    gain = np.empty(34, dtype=np.int64)
    if mtype == bb.MT_EP:
        gain[0] = SEE_VALUE[PAWN]
    else:
        victim = np.int64(mb[to])
        gain[0] = SEE_VALUE[victim % 6] if victim >= 0 else np.int64(0)

    if promo != 0:
        gain[0] += SEE_VALUE[promo] - SEE_VALUE[PAWN]
        moving_type = promo

    occ = (bbs[OCC_W] | bbs[OCC_B]) & ~bb.BIT[frm]
    if mtype == bb.MT_EP:
        occ &= ~bb.BIT[to - 8 if side == WHITE else to + 8]

    attackers = attackers_to(bbs, to, occ)
    on_move = 1 - side
    depth = 0
    captured_value = SEE_VALUE[moving_type]

    while True:
        attackers &= occ
        piece_bb = _least_valuable(bbs, attackers, on_move)
        if piece_bb == 0:
            break
        # A king may only take when nothing of the other colour still defends the
        # square, otherwise the recapture would be into check.
        if piece_bb & bbs[on_move * 6 + KING]:
            if _least_valuable(bbs, attackers & ~piece_bb, 1 - on_move) != 0:
                break
        depth += 1
        gain[depth] = captured_value - gain[depth - 1]
        # The usual `max(-gain[d-1], gain[d]) < 0` cut-off is only sign-correct.
        # The search compares SEE against non-zero thresholds, so run the swap out.
        square = lsb(piece_bb)
        captured_value = SEE_VALUE[np.int64(mb[square]) % 6]
        occ &= ~piece_bb
        attackers = attackers_to(bbs, to, occ)
        on_move = 1 - on_move
        if depth >= 31:
            break

    while depth > 0:
        gain[depth - 1] = -max(-gain[depth - 1], gain[depth])
        depth -= 1
    return gain[0]


@njit(cache=False)
def see_ge(bbs: np.ndarray, mb: np.ndarray, mv: np.int64, threshold: np.int64) -> bool:
    return see(bbs, mb, mv) >= threshold

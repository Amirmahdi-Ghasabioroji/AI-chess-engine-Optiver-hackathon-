"""Precomputed attacks and occupancy-aware slider rays."""

from __future__ import annotations

from engine.const import MASK64, lsb, msb

N, S, E, W = 8, -8, 1, -1
NE, NW, SE, SW = 9, 7, -7, -9
DIRS = (N, S, E, W, NE, NW, SE, SW)
ROOK_I = (0, 1, 2, 3)
BISHOP_I = (4, 5, 6, 7)
DIR_POS = (True, False, True, False, True, True, False, False)
_DFDR = {N: (0, 1), S: (0, -1), E: (1, 0), W: (-1, 0), NE: (1, 1), NW: (-1, 1), SE: (1, -1), SW: (-1, -1)}

RAY: list[list[int]] = [[0] * 64 for _ in range(8)]
KNIGHT_ATT = [0] * 64
KING_ATT = [0] * 64
PAWN_ATT = [[0] * 64, [0] * 64]


def _on(f: int, r: int) -> bool:
    return 0 <= f < 8 and 0 <= r < 8


def _walk(sq: int, df: int, dr: int) -> int:
    bb = 0
    f, r = sq % 8, sq // 8
    while True:
        f += df
        r += dr
        if not _on(f, r):
            break
        bb |= 1 << (r * 8 + f)
    return bb


for _sq in range(64):
    _f, _r = _sq % 8, _sq // 8
    for _i, _d in enumerate(DIRS):
        RAY[_i][_sq] = _walk(_sq, *_DFDR[_d])
    att = 0
    for df, dr in ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)):
        nf, nr = _f + df, _r + dr
        if _on(nf, nr):
            att |= 1 << (nr * 8 + nf)
    KNIGHT_ATT[_sq] = att
    att = 0
    for df, dr in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        nf, nr = _f + df, _r + dr
        if _on(nf, nr):
            att |= 1 << (nr * 8 + nf)
    KING_ATT[_sq] = att
    att = 0
    for df, dr in ((-1, 1), (1, 1)):
        nf, nr = _f + df, _r + dr
        if _on(nf, nr):
            att |= 1 << (nr * 8 + nf)
    PAWN_ATT[0][_sq] = att
    att = 0
    for df, dr in ((-1, -1), (1, -1)):
        nf, nr = _f + df, _r + dr
        if _on(nf, nr):
            att |= 1 << (nr * 8 + nf)
    PAWN_ATT[1][_sq] = att


def sliding(sq: int, occ: int, dirs: tuple[int, ...]) -> int:
    attacks = 0
    occ &= MASK64
    for i in dirs:
        ray = RAY[i][sq]
        blockers = ray & occ
        if blockers:
            first = lsb(blockers) if DIR_POS[i] else msb(blockers)
            attacks |= ray ^ RAY[i][first]
        else:
            attacks |= ray
    return attacks


def bishop_attacks(sq: int, occ: int) -> int:
    return sliding(sq, occ, BISHOP_I)


def rook_attacks(sq: int, occ: int) -> int:
    return sliding(sq, occ, ROOK_I)


def queen_attacks(sq: int, occ: int) -> int:
    return sliding(sq, occ, ROOK_I) | sliding(sq, occ, BISHOP_I)


def attackers_to(sq: int, occ: int, bb: list[int]) -> int:
    att = PAWN_ATT[1][sq] & bb[0]
    att |= PAWN_ATT[0][sq] & bb[6]
    att |= KNIGHT_ATT[sq] & (bb[1] | bb[7])
    att |= KING_ATT[sq] & (bb[5] | bb[11])
    bish = bishop_attacks(sq, occ)
    att |= bish & (bb[2] | bb[4] | bb[8] | bb[10])
    rook = rook_attacks(sq, occ)
    att |= rook & (bb[3] | bb[4] | bb[9] | bb[10])
    return att

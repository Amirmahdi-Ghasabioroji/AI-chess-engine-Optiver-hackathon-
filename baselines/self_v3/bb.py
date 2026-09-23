"""Bitboard position: magic attacks, movegen, make/unmake, Zobrist. All jitted.

Everything here works on plain numpy arrays of int64/int8 so numba can compile it
without object mode. Bitboards are int64 rather than uint64 because numba's mixed
signed/unsigned arithmetic promotes to float; the only signed hazard is that `>>`
sign-extends, so a right shift used as a whole-board shift always masks the bits
the shift should have vacated (see `lsr`).

Layout
------
`bbs`  int64[14]  twelve piece boards then the two occupancies
`st`   int64[5]   side, castling rights, ep square (64 = none), halfmove, hash
`mb`   int8[64]   mailbox, piece code per square or -1
`hist` int64[N,7] per-ply undo record (cols 5–6 used by search)

Piece codes are `colour * 6 + type` with type 0..5 = P N B R Q K.
Squares are a1=0 .. h8=63, matching python-chess.
"""

from __future__ import annotations

import numpy as np
from numba import njit

WHITE = 0
BLACK = 1

PAWN = 0
KNIGHT = 1
BISHOP = 2
ROOK = 3
QUEEN = 4
KING = 5

OCC_W = 12
OCC_B = 13
N_BB = 14

ST_SIDE = 0
ST_CASTLE = 1
ST_EP = 2
ST_HALF = 3
ST_HASH = 4
N_ST = 5

CR_WK = 1
CR_WQ = 2
CR_BK = 4
CR_BQ = 8

NO_EP = 64

MAX_PLY = 128
MAX_MOVES = 320  # comfortably above the ~218 move ceiling of a legal position

# Move layout: from | to<<6 | promo<<12 | type<<15.
# promo is a piece type (1..4 -> N B R Q offset by 1) or 0 for none.
MT_NORMAL = 0
MT_DOUBLE = 1
MT_EP = 2
MT_CASTLE = 3

def _u64(value: int) -> np.int64:
    """Reinterpret an unsigned 64-bit pattern as the int64 with the same bits."""
    return np.int64(value - (1 << 64) if value >= (1 << 63) else value)


FILE_A = _u64(0x0101010101010101)
FILE_H = _u64(0x8080808080808080)
RANK_1 = _u64(0x00000000000000FF)
RANK_3 = _u64(0x0000000000FF0000)
RANK_6 = _u64(0x0000FF0000000000)
RANK_8 = _u64(0xFF00000000000000)
NOT_FILE_A = ~FILE_A
NOT_FILE_H = ~FILE_H
PROMO_RANKS = RANK_1 | RANK_8

BIT = np.array([_u64(1 << i) for i in range(64)], dtype=np.int64)

# Masks that turn an arithmetic right shift back into a logical one.
LSR_MASK = np.array([_u64((1 << (64 - n)) - 1) for n in range(64)], dtype=np.int64)

DEBRUIJN64 = _u64(0x03F79D71B4CB0A89)
_DEB_IDX = np.zeros(64, dtype=np.int64)
for _i in range(64):
    _DEB_IDX[int((int(BIT[_i]) * int(DEBRUIJN64) % (1 << 64)) >> 58)] = _i
DEBRUIJN_IDX = _DEB_IDX

ROOK_DF = np.array([1, -1, 0, 0], dtype=np.int64)
ROOK_DR = np.array([0, 0, 1, -1], dtype=np.int64)
BISHOP_DF = np.array([1, 1, -1, -1], dtype=np.int64)
BISHOP_DR = np.array([1, -1, 1, -1], dtype=np.int64)


# --------------------------------------------------------------------------- bits


@njit(cache=False)
def lsr(b: np.int64, n: np.int64) -> np.int64:
    """Logical right shift of a bitboard."""
    return (b >> n) & LSR_MASK[n]


@njit(cache=False)
def lsb(b: np.int64) -> np.int64:
    """Index of the least significant set bit. Undefined for 0."""
    return DEBRUIJN_IDX[(((b & -b) * DEBRUIJN64) >> 58) & 63]


@njit(cache=False)
def popcount(b: np.int64) -> np.int64:
    b = b - ((b >> 1) & 0x5555555555555555)
    b = (b & 0x3333333333333333) + ((b >> 2) & 0x3333333333333333)
    b = (b + (b >> 4)) & 0x0F0F0F0F0F0F0F0F
    return ((b * 0x0101010101010101) >> 56) & 0x7F


# --------------------------------------------------------------------- attack init


def _slide_py(sq: int, occ: int, rook: bool) -> int:
    dfs = (1, -1, 0, 0) if rook else (1, 1, -1, -1)
    drs = (0, 0, 1, -1) if rook else (1, -1, 1, -1)
    att = 0
    f0, r0 = sq & 7, sq >> 3
    for d in range(4):
        f, r = f0 + dfs[d], r0 + drs[d]
        while 0 <= f <= 7 and 0 <= r <= 7:
            s = r * 8 + f
            att |= 1 << s
            if occ & (1 << s):
                break
            f += dfs[d]
            r += drs[d]
    return att


def _slide_mask_py(sq: int, rook: bool) -> int:
    """Relevant-occupancy mask: the ray squares that can block, edges excluded."""
    dfs = (1, -1, 0, 0) if rook else (1, 1, -1, -1)
    drs = (0, 0, 1, -1) if rook else (1, -1, 1, -1)
    mask = 0
    f0, r0 = sq & 7, sq >> 3
    for d in range(4):
        f, r = f0 + dfs[d], r0 + drs[d]
        while 0 <= f <= 7 and 0 <= r <= 7:
            nf, nr = f + dfs[d], r + drs[d]
            if not (0 <= nf <= 7 and 0 <= nr <= 7):
                break
            mask |= 1 << (r * 8 + f)
            f, r = nf, nr
    return mask


def _leaper_py(sq: int, deltas: tuple[tuple[int, int], ...]) -> int:
    att = 0
    f0, r0 = sq & 7, sq >> 3
    for df, dr in deltas:
        f, r = f0 + df, r0 + dr
        if 0 <= f <= 7 and 0 <= r <= 7:
            att |= 1 << (r * 8 + f)
    return att


_KNIGHT_DELTAS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
_KING_DELTAS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))

KNIGHT_ATT = np.array([_u64(_leaper_py(s, _KNIGHT_DELTAS)) for s in range(64)], dtype=np.int64)
KING_ATT = np.array([_u64(_leaper_py(s, _KING_DELTAS)) for s in range(64)], dtype=np.int64)
PAWN_ATT = np.array(
    [_u64(_leaper_py(s, ((-1, 1), (1, 1)))) for s in range(64)]
    + [_u64(_leaper_py(s, ((-1, -1), (1, -1)))) for s in range(64)],
    dtype=np.int64,
)

ROOK_MASK = np.array([_u64(_slide_mask_py(s, True)) for s in range(64)], dtype=np.int64)
BISHOP_MASK = np.array([_u64(_slide_mask_py(s, False)) for s in range(64)], dtype=np.int64)
ROOK_BITS = np.array([bin(_slide_mask_py(s, True)).count("1") for s in range(64)], dtype=np.int64)
BISHOP_BITS = np.array([bin(_slide_mask_py(s, False)).count("1") for s in range(64)], dtype=np.int64)

ROOK_OFF = np.zeros(64, dtype=np.int64)
BISHOP_OFF = np.zeros(64, dtype=np.int64)
_acc = 0
for _s in range(64):
    ROOK_OFF[_s] = _acc
    _acc += 1 << int(ROOK_BITS[_s])
ROOK_TABLE = np.zeros(_acc, dtype=np.int64)
_acc = 0
for _s in range(64):
    BISHOP_OFF[_s] = _acc
    _acc += 1 << int(BISHOP_BITS[_s])
BISHOP_TABLE = np.zeros(_acc, dtype=np.int64)

ROOK_MAGIC = np.zeros(64, dtype=np.int64)
BISHOP_MAGIC = np.zeros(64, dtype=np.int64)


@njit(cache=False)
def _rand64(state: np.ndarray) -> np.int64:
    x = state[0]
    x ^= x << 13
    x ^= lsr(x, 7)
    x ^= x << 17
    state[0] = x
    return x


@njit(cache=False)
def _index_to_occ(index: np.int64, mask: np.int64) -> np.int64:
    occ = np.int64(0)
    m = mask
    i = 0
    while m != 0:
        bit = m & -m
        m ^= bit
        if (index >> i) & 1:
            occ |= bit
        i += 1
    return occ


@njit(cache=False)
def _slide_jit(sq: np.int64, occ: np.int64, dfs: np.ndarray, drs: np.ndarray) -> np.int64:
    att = np.int64(0)
    f0 = sq & 7
    r0 = sq >> 3
    for d in range(4):
        f = f0 + dfs[d]
        r = r0 + drs[d]
        while 0 <= f <= 7 and 0 <= r <= 7:
            s = r * 8 + f
            att |= BIT[s]
            if occ & BIT[s]:
                break
            f += dfs[d]
            r += drs[d]
    return att


@njit(cache=False)
def _fill_magics(
    masks: np.ndarray,
    bits: np.ndarray,
    offsets: np.ndarray,
    magics: np.ndarray,
    table: np.ndarray,
    dfs: np.ndarray,
    drs: np.ndarray,
    seed: np.int64,
) -> np.int64:
    """Find a magic per square and populate the attack table. Returns total tries."""
    state = np.empty(1, dtype=np.int64)
    state[0] = seed
    occs = np.empty(1 << 12, dtype=np.int64)
    atts = np.empty(1 << 12, dtype=np.int64)
    used = np.empty(1 << 12, dtype=np.int64)
    stamp = np.zeros(1 << 12, dtype=np.int64)
    tries = np.int64(0)
    epoch = np.int64(0)

    for sq in range(64):
        mask = masks[sq]
        n = bits[sq]
        size = np.int64(1) << n
        for i in range(size):
            occ = _index_to_occ(np.int64(i), mask)
            occs[i] = occ
            atts[i] = _slide_jit(np.int64(sq), occ, dfs, drs)

        shift = 64 - n
        while True:
            tries += 1
            magic = _rand64(state) & _rand64(state) & _rand64(state)
            # A magic whose top byte spreads poorly almost never works; skip early.
            if popcount((mask * magic) >> 56 & 0xFF) < 6:
                continue
            epoch += 1
            ok = True
            for i in range(size):
                idx = ((occs[i] * magic) >> shift) & (size - 1)
                if stamp[idx] != epoch:
                    stamp[idx] = epoch
                    used[idx] = atts[i]
                elif used[idx] != atts[i]:
                    ok = False
                    break
            if ok:
                magics[sq] = magic
                base = offsets[sq]
                for i in range(size):
                    idx = ((occs[i] * magic) >> shift) & (size - 1)
                    table[base + idx] = atts[i]
                break
    return tries


@njit(cache=False)
def rook_attacks(sq: np.int64, occ: np.int64) -> np.int64:
    n = ROOK_BITS[sq]
    idx = (((occ & ROOK_MASK[sq]) * ROOK_MAGIC[sq]) >> (64 - n)) & ((np.int64(1) << n) - 1)
    return ROOK_TABLE[ROOK_OFF[sq] + idx]


@njit(cache=False)
def bishop_attacks(sq: np.int64, occ: np.int64) -> np.int64:
    n = BISHOP_BITS[sq]
    idx = (((occ & BISHOP_MASK[sq]) * BISHOP_MAGIC[sq]) >> (64 - n)) & ((np.int64(1) << n) - 1)
    return BISHOP_TABLE[BISHOP_OFF[sq] + idx]


@njit(cache=False)
def queen_attacks(sq: np.int64, occ: np.int64) -> np.int64:
    return rook_attacks(sq, occ) | bishop_attacks(sq, occ)


# ------------------------------------------------------------------------ zobrist

ZOB_PIECE = np.zeros(12 * 64, dtype=np.int64)
ZOB_CASTLE = np.zeros(16, dtype=np.int64)
ZOB_EP = np.zeros(9, dtype=np.int64)
ZOB_SIDE = np.int64(0)


def _init_zobrist() -> None:
    global ZOB_SIDE
    state = 0x9E3779B97F4A7C15
    values = []
    for _ in range(12 * 64 + 16 + 9 + 1):
        state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
        state ^= state >> 7
        state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
        values.append(_u64(state))
    ZOB_PIECE[:] = values[: 12 * 64]
    ZOB_CASTLE[:] = values[12 * 64 : 12 * 64 + 16]
    ZOB_EP[:] = values[12 * 64 + 16 : 12 * 64 + 25]
    ZOB_EP[8] = np.int64(0)  # "no ep file" contributes nothing
    ZOB_SIDE = values[-1]


_init_zobrist()

# Castling rights that survive a move touching a square.
CASTLE_MASK = np.full(64, 15, dtype=np.int64)
CASTLE_MASK[4] = 15 & ~(CR_WK | CR_WQ)
CASTLE_MASK[0] = 15 & ~CR_WQ
CASTLE_MASK[7] = 15 & ~CR_WK
CASTLE_MASK[60] = 15 & ~(CR_BK | CR_BQ)
CASTLE_MASK[56] = 15 & ~CR_BQ
CASTLE_MASK[63] = 15 & ~CR_BK


@njit(cache=False)
def compute_hash(bbs: np.ndarray, st: np.ndarray) -> np.int64:
    h = np.int64(0)
    for code in range(12):
        b = bbs[code]
        while b != 0:
            sq = lsb(b)
            b &= b - 1
            h ^= ZOB_PIECE[code * 64 + sq]
    h ^= ZOB_CASTLE[st[ST_CASTLE]]
    if st[ST_EP] != NO_EP:
        h ^= ZOB_EP[st[ST_EP] & 7]
    if st[ST_SIDE] == BLACK:
        h ^= ZOB_SIDE
    return h


# ------------------------------------------------------------------------ queries


@njit(cache=False)
def attacked(bbs: np.ndarray, sq: np.int64, by: np.int64, occ: np.int64) -> bool:
    base = by * 6
    # A pawn of `by` attacks sq exactly when a pawn of the other colour on sq
    # would attack that pawn's square.
    if PAWN_ATT[(1 - by) * 64 + sq] & bbs[base + PAWN]:
        return True
    if KNIGHT_ATT[sq] & bbs[base + KNIGHT]:
        return True
    if KING_ATT[sq] & bbs[base + KING]:
        return True
    if bishop_attacks(sq, occ) & (bbs[base + BISHOP] | bbs[base + QUEEN]):
        return True
    if rook_attacks(sq, occ) & (bbs[base + ROOK] | bbs[base + QUEEN]):
        return True
    return False


@njit(cache=False)
def king_square(bbs: np.ndarray, side: np.int64) -> np.int64:
    return lsb(bbs[side * 6 + KING])


@njit(cache=False)
def in_check(bbs: np.ndarray, st: np.ndarray) -> bool:
    side = st[ST_SIDE]
    return attacked(bbs, king_square(bbs, side), 1 - side, bbs[OCC_W] | bbs[OCC_B])


# ------------------------------------------------------------------------ movegen


@njit(cache=False)
def mk_move(frm: np.int64, to: np.int64, promo: np.int64, mtype: np.int64) -> np.int64:
    return frm | (to << 6) | (promo << 12) | (mtype << 15)


@njit(cache=False)
def gen_moves(
    bbs: np.ndarray, st: np.ndarray, moves: np.ndarray, base: np.int64, only_caps: bool
) -> np.int64:
    """Pseudo-legal moves into moves[base:]. Returns the new end index."""
    side = st[ST_SIDE]
    us = side * 6
    own = bbs[OCC_W + side]
    opp = bbs[OCC_W + (1 - side)]
    occ = own | opp
    empty = ~occ
    n = base
    target = opp if only_caps else ~own

    pawns = bbs[us + PAWN]
    if side == WHITE:
        push = (pawns << 8) & empty
        promo_push = push & RANK_8
        quiet_push = push & ~RANK_8
        dbl = ((push & RANK_3) << 8) & empty
        cap_l = ((pawns & NOT_FILE_A) << 7) & opp
        cap_r = ((pawns & NOT_FILE_H) << 9) & opp
        d_push, d_dbl, d_l, d_r = np.int64(8), np.int64(16), np.int64(7), np.int64(9)
    else:
        push = lsr(pawns, 8) & empty
        promo_push = push & RANK_1
        quiet_push = push & ~RANK_1
        dbl = lsr(push & RANK_6, 8) & empty
        cap_l = lsr(pawns & NOT_FILE_H, 7) & opp
        cap_r = lsr(pawns & NOT_FILE_A, 9) & opp
        d_push, d_dbl, d_l, d_r = np.int64(-8), np.int64(-16), np.int64(-7), np.int64(-9)

    b = promo_push
    while b != 0:
        to = lsb(b)
        b &= b - 1
        frm = to - d_push
        for promo in range(4, 0, -1):
            moves[n] = mk_move(frm, to, promo, MT_NORMAL)
            n += 1

    b = cap_l
    while b != 0:
        to = lsb(b)
        b &= b - 1
        frm = to - d_l
        if BIT[to] & PROMO_RANKS:
            for promo in range(4, 0, -1):
                moves[n] = mk_move(frm, to, promo, MT_NORMAL)
                n += 1
        else:
            moves[n] = mk_move(frm, to, 0, MT_NORMAL)
            n += 1

    b = cap_r
    while b != 0:
        to = lsb(b)
        b &= b - 1
        frm = to - d_r
        if BIT[to] & PROMO_RANKS:
            for promo in range(4, 0, -1):
                moves[n] = mk_move(frm, to, promo, MT_NORMAL)
                n += 1
        else:
            moves[n] = mk_move(frm, to, 0, MT_NORMAL)
            n += 1

    if st[ST_EP] != NO_EP:
        ep = st[ST_EP]
        b = PAWN_ATT[(1 - side) * 64 + ep] & pawns
        while b != 0:
            frm = lsb(b)
            b &= b - 1
            moves[n] = mk_move(frm, ep, 0, MT_EP)
            n += 1

    if not only_caps:
        b = quiet_push
        while b != 0:
            to = lsb(b)
            b &= b - 1
            moves[n] = mk_move(to - d_push, to, 0, MT_NORMAL)
            n += 1
        b = dbl
        while b != 0:
            to = lsb(b)
            b &= b - 1
            moves[n] = mk_move(to - d_dbl, to, 0, MT_DOUBLE)
            n += 1

    b = bbs[us + KNIGHT]
    while b != 0:
        frm = lsb(b)
        b &= b - 1
        att = KNIGHT_ATT[frm] & target
        while att != 0:
            to = lsb(att)
            att &= att - 1
            moves[n] = mk_move(frm, to, 0, MT_NORMAL)
            n += 1

    b = bbs[us + BISHOP]
    while b != 0:
        frm = lsb(b)
        b &= b - 1
        att = bishop_attacks(frm, occ) & target
        while att != 0:
            to = lsb(att)
            att &= att - 1
            moves[n] = mk_move(frm, to, 0, MT_NORMAL)
            n += 1

    b = bbs[us + ROOK]
    while b != 0:
        frm = lsb(b)
        b &= b - 1
        att = rook_attacks(frm, occ) & target
        while att != 0:
            to = lsb(att)
            att &= att - 1
            moves[n] = mk_move(frm, to, 0, MT_NORMAL)
            n += 1

    b = bbs[us + QUEEN]
    while b != 0:
        frm = lsb(b)
        b &= b - 1
        att = queen_attacks(frm, occ) & target
        while att != 0:
            to = lsb(att)
            att &= att - 1
            moves[n] = mk_move(frm, to, 0, MT_NORMAL)
            n += 1

    ksq = king_square(bbs, side)
    att = KING_ATT[ksq] & target
    while att != 0:
        to = lsb(att)
        att &= att - 1
        moves[n] = mk_move(ksq, to, 0, MT_NORMAL)
        n += 1

    if not only_caps:
        rights = st[ST_CASTLE]
        them = 1 - side
        if side == WHITE:
            if (
                (rights & CR_WK)
                and not (occ & (BIT[5] | BIT[6]))
                and not attacked(bbs, np.int64(4), them, occ)
                and not attacked(bbs, np.int64(5), them, occ)
                and not attacked(bbs, np.int64(6), them, occ)
            ):
                moves[n] = mk_move(np.int64(4), np.int64(6), 0, MT_CASTLE)
                n += 1
            if (
                (rights & CR_WQ)
                and not (occ & (BIT[1] | BIT[2] | BIT[3]))
                and not attacked(bbs, np.int64(4), them, occ)
                and not attacked(bbs, np.int64(3), them, occ)
                and not attacked(bbs, np.int64(2), them, occ)
            ):
                moves[n] = mk_move(np.int64(4), np.int64(2), 0, MT_CASTLE)
                n += 1
        else:
            if (
                (rights & CR_BK)
                and not (occ & (BIT[61] | BIT[62]))
                and not attacked(bbs, np.int64(60), them, occ)
                and not attacked(bbs, np.int64(61), them, occ)
                and not attacked(bbs, np.int64(62), them, occ)
            ):
                moves[n] = mk_move(np.int64(60), np.int64(62), 0, MT_CASTLE)
                n += 1
            if (
                (rights & CR_BQ)
                and not (occ & (BIT[57] | BIT[58] | BIT[59]))
                and not attacked(bbs, np.int64(60), them, occ)
                and not attacked(bbs, np.int64(59), them, occ)
                and not attacked(bbs, np.int64(58), them, occ)
            ):
                moves[n] = mk_move(np.int64(60), np.int64(58), 0, MT_CASTLE)
                n += 1

    return n


# -------------------------------------------------------------------- make/unmake


@njit(cache=False)
def _put(bbs: np.ndarray, mb: np.ndarray, code: np.int64, sq: np.int64) -> np.int64:
    bbs[code] |= BIT[sq]
    bbs[OCC_W + code // 6] |= BIT[sq]
    mb[sq] = code
    return ZOB_PIECE[code * 64 + sq]


@njit(cache=False)
def _clear(bbs: np.ndarray, mb: np.ndarray, code: np.int64, sq: np.int64) -> np.int64:
    bbs[code] &= ~BIT[sq]
    bbs[OCC_W + code // 6] &= ~BIT[sq]
    mb[sq] = -1
    return ZOB_PIECE[code * 64 + sq]


@njit(cache=False)
def make_move(
    bbs: np.ndarray, st: np.ndarray, mb: np.ndarray, hist: np.ndarray, ply: np.int64, mv: np.int64
) -> None:
    frm = mv & 63
    to = (mv >> 6) & 63
    promo = (mv >> 12) & 7
    mtype = (mv >> 15) & 3
    side = st[ST_SIDE]
    us = side * 6

    hist[ply, 0] = st[ST_CASTLE]
    hist[ply, 1] = st[ST_EP]
    hist[ply, 2] = st[ST_HALF]
    hist[ply, 3] = st[ST_HASH]

    h = st[ST_HASH]
    if st[ST_EP] != NO_EP:
        h ^= ZOB_EP[st[ST_EP] & 7]
    h ^= ZOB_CASTLE[st[ST_CASTLE]]

    piece = np.int64(mb[frm])
    captured = np.int64(-1)

    if mtype == MT_EP:
        cap_sq = to - 8 if side == WHITE else to + 8
        captured = (1 - side) * 6 + PAWN
        h ^= _clear(bbs, mb, captured, cap_sq)
        hist[ply, 4] = captured
    else:
        occupant = np.int64(mb[to])
        hist[ply, 4] = occupant
        if occupant >= 0:
            captured = occupant
            h ^= _clear(bbs, mb, occupant, to)

    h ^= _clear(bbs, mb, piece, frm)
    if promo != 0:
        h ^= _put(bbs, mb, us + promo, to)
    else:
        h ^= _put(bbs, mb, piece, to)

    if mtype == MT_CASTLE:
        if to == 6:
            h ^= _clear(bbs, mb, np.int64(us + ROOK), np.int64(7))
            h ^= _put(bbs, mb, np.int64(us + ROOK), np.int64(5))
        elif to == 2:
            h ^= _clear(bbs, mb, np.int64(us + ROOK), np.int64(0))
            h ^= _put(bbs, mb, np.int64(us + ROOK), np.int64(3))
        elif to == 62:
            h ^= _clear(bbs, mb, np.int64(us + ROOK), np.int64(63))
            h ^= _put(bbs, mb, np.int64(us + ROOK), np.int64(61))
        else:
            h ^= _clear(bbs, mb, np.int64(us + ROOK), np.int64(56))
            h ^= _put(bbs, mb, np.int64(us + ROOK), np.int64(59))

    st[ST_CASTLE] = st[ST_CASTLE] & CASTLE_MASK[frm] & CASTLE_MASK[to]
    h ^= ZOB_CASTLE[st[ST_CASTLE]]

    # Record an ep square only when an enemy pawn can actually take it, so a
    # position reached by searching hashes the same as one parsed from a FEN
    # (python-chess emits the ep field under the same rule).
    st[ST_EP] = NO_EP
    if mtype == MT_DOUBLE:
        ep_sq = (frm + to) // 2
        if PAWN_ATT[side * 64 + ep_sq] & bbs[(1 - side) * 6 + PAWN]:
            st[ST_EP] = ep_sq
            h ^= ZOB_EP[ep_sq & 7]

    if piece % 6 == PAWN or captured >= 0:
        st[ST_HALF] = 0
    else:
        st[ST_HALF] = st[ST_HALF] + 1

    st[ST_SIDE] = 1 - side
    h ^= ZOB_SIDE
    st[ST_HASH] = h


@njit(cache=False)
def unmake_move(
    bbs: np.ndarray, st: np.ndarray, mb: np.ndarray, hist: np.ndarray, ply: np.int64, mv: np.int64
) -> None:
    frm = mv & 63
    to = (mv >> 6) & 63
    promo = (mv >> 12) & 7
    mtype = (mv >> 15) & 3
    side = 1 - st[ST_SIDE]
    us = side * 6

    moved = np.int64(mb[to])
    _clear(bbs, mb, moved, to)
    if promo != 0:
        _put(bbs, mb, np.int64(us + PAWN), frm)
    else:
        _put(bbs, mb, moved, frm)

    captured = hist[ply, 4]
    if mtype == MT_EP:
        cap_sq = to - 8 if side == WHITE else to + 8
        _put(bbs, mb, (1 - side) * 6 + PAWN, cap_sq)
    elif captured >= 0:
        _put(bbs, mb, captured, to)

    if mtype == MT_CASTLE:
        if to == 6:
            _clear(bbs, mb, np.int64(us + ROOK), np.int64(5))
            _put(bbs, mb, np.int64(us + ROOK), np.int64(7))
        elif to == 2:
            _clear(bbs, mb, np.int64(us + ROOK), np.int64(3))
            _put(bbs, mb, np.int64(us + ROOK), np.int64(0))
        elif to == 62:
            _clear(bbs, mb, np.int64(us + ROOK), np.int64(61))
            _put(bbs, mb, np.int64(us + ROOK), np.int64(63))
        else:
            _clear(bbs, mb, np.int64(us + ROOK), np.int64(59))
            _put(bbs, mb, np.int64(us + ROOK), np.int64(56))

    st[ST_CASTLE] = hist[ply, 0]
    st[ST_EP] = hist[ply, 1]
    st[ST_HALF] = hist[ply, 2]
    st[ST_HASH] = hist[ply, 3]
    st[ST_SIDE] = side


@njit(cache=False)
def make_null(bbs: np.ndarray, st: np.ndarray, hist: np.ndarray, ply: np.int64) -> None:
    hist[ply, 0] = st[ST_CASTLE]
    hist[ply, 1] = st[ST_EP]
    hist[ply, 2] = st[ST_HALF]
    hist[ply, 3] = st[ST_HASH]
    hist[ply, 4] = -1
    h = st[ST_HASH]
    if st[ST_EP] != NO_EP:
        h ^= ZOB_EP[st[ST_EP] & 7]
    st[ST_EP] = NO_EP
    st[ST_HALF] = st[ST_HALF] + 1
    st[ST_SIDE] = 1 - st[ST_SIDE]
    st[ST_HASH] = h ^ ZOB_SIDE


@njit(cache=False)
def unmake_null(bbs: np.ndarray, st: np.ndarray, hist: np.ndarray, ply: np.int64) -> None:
    st[ST_CASTLE] = hist[ply, 0]
    st[ST_EP] = hist[ply, 1]
    st[ST_HALF] = hist[ply, 2]
    st[ST_HASH] = hist[ply, 3]
    st[ST_SIDE] = 1 - st[ST_SIDE]


@njit(cache=False)
def make_legal(
    bbs: np.ndarray, st: np.ndarray, mb: np.ndarray, hist: np.ndarray, ply: np.int64, mv: np.int64
) -> bool:
    """Make the move; undo and return False if it leaves our king attacked."""
    side = st[ST_SIDE]
    make_move(bbs, st, mb, hist, ply, mv)
    if attacked(bbs, king_square(bbs, side), 1 - side, bbs[OCC_W] | bbs[OCC_B]):
        unmake_move(bbs, st, mb, hist, ply, mv)
        return False
    return True


# --------------------------------------------------------------------------- misc


@njit(cache=False)
def perft(
    bbs: np.ndarray,
    st: np.ndarray,
    mb: np.ndarray,
    hist: np.ndarray,
    moves: np.ndarray,
    ply: np.int64,
    depth: np.int64,
) -> np.int64:
    if depth == 0:
        return np.int64(1)
    base = ply * MAX_MOVES
    end = gen_moves(bbs, st, moves, base, False)
    total = np.int64(0)
    for i in range(base, end):
        mv = moves[i]
        if make_legal(bbs, st, mb, hist, ply, mv):
            total += perft(bbs, st, mb, hist, moves, ply + 1, depth - 1)
            unmake_move(bbs, st, mb, hist, ply, mv)
    return total


def new_buffers() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Column 5 is the move played at this ply (countermove). Column 6 is static eval.
    hist = np.zeros((MAX_PLY + 8, 7), dtype=np.int64)
    moves = np.zeros((MAX_PLY + 8) * MAX_MOVES, dtype=np.int32)
    scores = np.zeros((MAX_PLY + 8) * MAX_MOVES, dtype=np.int32)
    return hist, moves, scores


def init() -> None:
    """Find magics and populate the sliding-attack tables. Idempotent."""
    if ROOK_MAGIC[0] != 0:
        return
    _fill_magics(
        ROOK_MASK,
        ROOK_BITS,
        ROOK_OFF,
        ROOK_MAGIC,
        ROOK_TABLE,
        ROOK_DF,
        ROOK_DR,
        _u64(0x1234567890ABCDEF),
    )
    _fill_magics(
        BISHOP_MASK,
        BISHOP_BITS,
        BISHOP_OFF,
        BISHOP_MAGIC,
        BISHOP_TABLE,
        BISHOP_DF,
        BISHOP_DR,
        _u64(0x0FEDCBA987654321),
    )

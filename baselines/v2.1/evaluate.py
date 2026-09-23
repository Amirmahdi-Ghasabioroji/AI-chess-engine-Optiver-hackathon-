"""Tapered evaluation: PeSTO PSTs plus pawn structure, king safety, mop-up.

Scores are in centipawns, from the side to move.
"""

from __future__ import annotations

import chess

# PeSTO piece values (mg, eg) — https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function
MG_VALUE = (0, 82, 337, 365, 477, 1025, 0)
EG_VALUE = (0, 94, 281, 297, 512, 936, 0)

# Phase weights: P=0 N=1 B=1 R=2 Q=4, max 24
PHASE_WT = (0, 0, 1, 1, 2, 4, 0)
PHASE_MAX = 24

# SEE / material helpers (coarse)
SEE_VAL = (0, 100, 320, 330, 500, 900, 20000)

# Tables are a8..h8, a7..h7, ..., a1..h1. White probes with sq ^ 56.
_MG_PAWN = (
    0,   0,   0,   0,   0,   0,   0,   0,
    98, 134,  61,  95,  68, 126,  34, -11,
    -6,   7,  26,  31,  65,  56,  25, -20,
   -14,  13,   6,  21,  23,  12,  17, -23,
   -27,  -2,  -5,  12,  17,   6,  10, -25,
   -26,  -4,  -4, -10,   3,   3,  33, -12,
   -35,  -1, -20, -23, -15,  24,  38, -22,
     0,   0,   0,   0,   0,   0,   0,   0,
)
_EG_PAWN = (
     0,   0,   0,   0,   0,   0,   0,   0,
   178, 173, 158, 134, 147, 132, 165, 187,
    94, 100,  85,  67,  56,  53,  82,  84,
    32,  24,  13,   5,  -2,   4,  17,  17,
    13,   9,  -3,  -7,  -7,  -8,   3,  -1,
     4,   8,  -6,   1,   0,  -5,  -1,  -8,
    13,   8,   8,  10,  13,   0,   2,  -7,
     0,   0,   0,   0,   0,   0,   0,   0,
)
_MG_KNIGHT = (
   -167, -89, -34, -49,  61, -97, -15, -107,
    -73, -41,  72,  36,  23,  62,   7,  -17,
    -47,  60,  37,  65,  84, 129,  73,   44,
     -9,  17,  19,  53,  37,  69,  18,   22,
    -13,   4,  16,  13,  28,  19,  21,   -8,
    -23,  -9,  12,  10,  19,  17,  25,  -16,
    -29, -53, -12,  -3,  -1,  18, -14,  -19,
   -105, -21, -58, -33, -17, -28, -19,  -23,
)
_EG_KNIGHT = (
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
)
_MG_BISHOP = (
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
)
_EG_BISHOP = (
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
)
_MG_ROOK = (
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
)
_EG_ROOK = (
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20,
)
_MG_QUEEN = (
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
)
_EG_QUEEN = (
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
)
_MG_KING = (
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
)
_EG_KING = (
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
)

_MG_PST = (None, _MG_PAWN, _MG_KNIGHT, _MG_BISHOP, _MG_ROOK, _MG_QUEEN, _MG_KING)
_EG_PST = (None, _EG_PAWN, _EG_KNIGHT, _EG_BISHOP, _EG_ROOK, _EG_QUEEN, _EG_KING)

# File bitboards
_FILE = chess.BB_FILES
_ADJ_FILES = tuple(
    (_FILE[f - 1] if f > 0 else 0) | _FILE[f] | (_FILE[f + 1] if f < 7 else 0)
    for f in range(8)
)


def _passed_white(sq: int) -> int:
    """Squares in front of a white pawn (same + adjacent files) that would block a passer."""
    f = sq & 7
    r = sq >> 3
    mask = 0
    for rr in range(r + 1, 8):
        mask |= _ADJ_FILES[f] & chess.BB_RANKS[rr]
    return mask


def _passed_black(sq: int) -> int:
    f = sq & 7
    r = sq >> 3
    mask = 0
    for rr in range(r - 1, -1, -1):
        mask |= _ADJ_FILES[f] & chess.BB_RANKS[rr]
    return mask


PASSED_W = tuple(_passed_white(sq) for sq in range(64))
PASSED_B = tuple(_passed_black(sq) for sq in range(64))

# Isolated: no friendly pawn on adjacent files
ISOLATED_FILE = tuple(
    (_FILE[f - 1] if f > 0 else 0) | (_FILE[f + 1] if f < 7 else 0)
    for f in range(8)
)

# Passed pawn bonuses by rank of the pawn (white's rank).
PASSED_MG = (0, 2, 6, 12, 24, 48, 80, 0)
PASSED_EG = (0, 8, 16, 32, 56, 96, 160, 0)

TEMPO_MG = 10

# Pawn-structure cache. Keyed on (white_pawns, black_pawns); collisions just recompute.
_PH_N = 16384
_ph_k: list[int | None] = [None] * _PH_N
_ph_mg = [0] * _PH_N
_ph_eg = [0] * _PH_N


def _popcount(bb: int) -> int:
    return bb.bit_count()


def evaluate(board: chess.Board) -> int:
    """Return a side-to-move centipawn score."""
    mg = 0
    eg = 0
    phase = 0

    wp = board.pieces_mask(chess.PAWN, chess.WHITE)
    bp = board.pieces_mask(chess.PAWN, chess.BLACK)

    w_np = 0  # non-pawn material, for mop-up / null-move
    b_np = 0

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        for pt in range(1, 7):
            bb = board.pieces_mask(pt, color)
            n = _popcount(bb)
            if not n:
                continue
            phase += PHASE_WT[pt] * n
            mg += sign * MG_VALUE[pt] * n
            eg += sign * EG_VALUE[pt] * n
            if pt != chess.PAWN:
                if color == chess.WHITE:
                    w_np += SEE_VAL[pt] * n
                else:
                    b_np += SEE_VAL[pt] * n
            pst_mg = _MG_PST[pt]
            pst_eg = _EG_PST[pt]
            while bb:
                lsb = bb & -bb
                sq = lsb.bit_length() - 1
                bb ^= lsb
                idx = sq ^ 56 if color == chess.WHITE else sq
                mg += sign * pst_mg[idx]
                eg += sign * pst_eg[idx]

    # Bishop pair
    if _popcount(board.pieces_mask(chess.BISHOP, chess.WHITE)) >= 2:
        mg += 25
        eg += 40
    if _popcount(board.pieces_mask(chess.BISHOP, chess.BLACK)) >= 2:
        mg -= 25
        eg -= 40

    # Pawn structure (hashed; pawn bitboards change slowly)
    mg_p, eg_p = _pawns_hashed(wp, bp)
    mg += mg_p
    eg += eg_p

    # Rooks: open/semi-open files and 7th rank
    mg_r, eg_r = _rooks(board, wp, bp)
    mg += mg_r
    eg += eg_r

    # King pawn shield (middlegame)
    mg += _king_shield(board, chess.WHITE, wp)
    mg -= _king_shield(board, chess.BLACK, bp)

    if phase > PHASE_MAX:
        phase = PHASE_MAX
    score = (mg * phase + eg * (PHASE_MAX - phase)) // PHASE_MAX

    # Tempo
    score += TEMPO_MG if board.turn == chess.WHITE else -TEMPO_MG

    # Mop-up when one side has a decisive material lead in a late phase
    if phase <= 8:
        score += _mopup(board, w_np, b_np, wp, bp)

    return score if board.turn == chess.WHITE else -score


def _pawns_hashed(wp: int, bp: int) -> tuple[int, int]:
    key = ((wp * 11400714819323198485) ^ (bp * 14029467366897019727)) & 0xFFFFFFFFFFFFFFFF
    i = key & (_PH_N - 1)
    if _ph_k[i] == key:
        return _ph_mg[i], _ph_eg[i]
    mg, eg = _pawns(wp, bp)
    _ph_k[i] = key
    _ph_mg[i] = mg
    _ph_eg[i] = eg
    return mg, eg


def _pawns(wp: int, bp: int) -> tuple[int, int]:
    mg = 0
    eg = 0

    w = wp
    while w:
        lsb = w & -w
        sq = lsb.bit_length() - 1
        w ^= lsb
        f = sq & 7
        r = sq >> 3
        file_bb = _FILE[f]
        # doubled: another white pawn in front on the same file
        ahead = file_bb & ~((1 << (sq + 1)) - 1)
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
    while b:
        lsb = b & -b
        sq = lsb.bit_length() - 1
        b ^= lsb
        f = sq & 7
        r = sq >> 3
        file_bb = _FILE[f]
        behind = file_bb & ((1 << sq) - 1)
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


def _rooks(board: chess.Board, wp: int, bp: int) -> tuple[int, int]:
    mg = 0
    eg = 0
    all_p = wp | bp
    for color, pawns, enemy_pawns, sign in (
        (chess.WHITE, wp, bp, 1),
        (chess.BLACK, bp, wp, -1),
    ):
        bb = board.pieces_mask(chess.ROOK, color)
        seventh = chess.BB_RANK_7 if color == chess.WHITE else chess.BB_RANK_2
        while bb:
            lsb = bb & -bb
            sq = lsb.bit_length() - 1
            bb ^= lsb
            file_bb = _FILE[sq & 7]
            if not (all_p & file_bb):
                mg += sign * 18
                eg += sign * 12
            elif not (pawns & file_bb):
                mg += sign * 8
                eg += sign * 6
            if (1 << sq) & seventh:
                mg += sign * 12
                eg += sign * 18
    return mg, eg


def _king_shield(board: chess.Board, color: chess.Color, pawns: int) -> int:
    king = board.king(color)
    if king is None:
        return 0
    f = king & 7
    r = king >> 3
    # Castled-ish kings: files a-c or f-h
    if 2 < f < 5:
        return 0
    bonus = 0
    rank_dir = 1 if color == chess.WHITE else -1
    shield_rank = r + rank_dir
    if 0 <= shield_rank <= 7:
        for df in (-1, 0, 1):
            ff = f + df
            if 0 <= ff <= 7:
                sq = shield_rank * 8 + ff
                if pawns & (1 << sq):
                    bonus += 12
                else:
                    bonus -= 8
    return bonus


def _mopup(board: chess.Board, w_np: int, b_np: int, wp: int, bp: int) -> int:
    """Drive the losing king to the edge when we have a big lead."""
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return 0
    w_mat = w_np + 100 * _popcount(wp)
    b_mat = b_np + 100 * _popcount(bp)
    diff = w_mat - b_mat
    if abs(diff) < 350:
        return 0
    # Chebyshev distance between kings, and centre-manhattan of the weak king
    def _cheb(a: int, b: int) -> int:
        return max(abs((a & 7) - (b & 7)), abs((a >> 3) - (b >> 3)))

    def _edge(sq: int) -> int:
        f, r = sq & 7, sq >> 3
        return min(f, 7 - f, r, 7 - r)

    if diff > 0:
        return 4 * (7 - _cheb(wk, bk)) + 12 * (3 - _edge(bk))
    return -4 * (7 - _cheb(wk, bk)) - 12 * (3 - _edge(wk))


def non_pawn_material(board: chess.Board, color: chess.Color) -> int:
    n = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        n += SEE_VAL[pt] * _popcount(board.pieces_mask(pt, color))
    return n


def see(board: chess.Board, move: chess.Move) -> int:
    """Approximate swap-off in centipawns for the side that captures.

    Positive → expected material gain. Used for move ordering; qsearch still
    uses see_ge(0) as a filter.
    """
    if move.promotion:
        promo = SEE_VAL[move.promotion] - SEE_VAL[chess.PAWN]
        victim = board.piece_type_at(move.to_square)
        return promo + (SEE_VAL[victim] if victim else 0)
    if board.is_en_passant(move):
        return SEE_VAL[chess.PAWN]
    victim = board.piece_type_at(move.to_square)
    attacker = board.piece_type_at(move.from_square)
    if victim is None or attacker is None:
        return 0
    gain = SEE_VAL[victim]
    if not board.is_attacked_by(not board.turn, move.to_square):
        return gain
    # One recapture by the opponent's least we bother to model: they take us.
    return gain - SEE_VAL[attacker]


def see_ge(board: chess.Board, move: chess.Move, threshold: int = 0) -> bool:
    """Cheap static exchange: is this capture worth at least `threshold`?"""
    return see(board, move) >= threshold

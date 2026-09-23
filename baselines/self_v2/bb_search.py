"""Iterative-deepening PVS, compiled end to end by numba.

The whole search - move ordering, transposition table, quiescence - runs inside
nopython mode, so a node costs no Python at all. numba cannot read a clock in
nopython mode, so the deadline is checked through a tiny `objmode` block every
few thousand nodes; at 0.4 us a read that is far below the noise floor.

Every mutable table travels as an argument because numba exposes module-level
arrays read-only.
"""

from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, objmode

import bb
from bb import (
    KING,
    MAX_MOVES,
    MAX_PLY,
    MT_EP,
    OCC_B,
    OCC_W,
    PAWN,
    ST_CASTLE,
    ST_HALF,
    ST_HASH,
    ST_SIDE,
    attacked,
    gen_moves,
    king_square,
    lsb,
    make_move,
    make_null,
    popcount,
    unmake_move,
    unmake_null,
)
from bb_eval import SEE_VALUE, evaluate, non_pawn_material, see

MATE = 30000
MATE_BOUND = 29000
INF = 32000
DRAW = 0

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2
TT_VALID = np.int64(1) << 50

# info[] slots
I_NODES = 0
I_SELDEPTH = 1
I_STOP = 2
I_ROOT_MOVE = 3
I_ROOT_SCORE = 4
I_REP_BASE = 5
I_GENERATION = 6
I_DEPTH_DONE = 7
N_INFO = 8

# ord32[] is one buffer holding three ordering tables.
KILLER_OFF = 0
KILLER_LEN = (MAX_PLY + 8) * 2
HISTORY_OFF = KILLER_OFF + KILLER_LEN
HISTORY_LEN = 2 * 64 * 64
COUNTER_OFF = HISTORY_OFF + HISTORY_LEN
COUNTER_LEN = 64 * 64
# Per-ply scratch listing the quiet moves already tried, so a beta cut-off can
# nudge them down. It lives here to avoid threading another array through the
# recursion.
QUIET_OFF = COUNTER_OFF + COUNTER_LEN
QUIET_PER_PLY = 64
QUIET_LEN = (MAX_PLY + 8) * QUIET_PER_PLY
ORD_LEN = QUIET_OFF + QUIET_LEN

CHECK_INTERVAL = 2047

LMR_TABLE = np.zeros((64, 64), dtype=np.int32)
for _d in range(1, 64):
    for _m in range(1, 64):
        LMR_TABLE[_d, _m] = int(0.75 + math.log(_d) * math.log(_m) / 2.25)

TT_BITS = 22
TT_SIZE = 1 << TT_BITS


def new_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tt = np.zeros((TT_SIZE, 2), dtype=np.int64)
    ord32 = np.zeros(ORD_LEN, dtype=np.int32)
    rep = np.zeros(MAX_PLY + 1024, dtype=np.int64)
    info = np.zeros(N_INFO, dtype=np.int64)
    return tt, ord32, rep, info


@njit(cache=False)
def _now() -> float:
    with objmode(t="f8"):
        t = time.perf_counter()
    return t


# ------------------------------------------------------------------ transposition


@njit(cache=False)
def tt_probe(tt: np.ndarray, key: np.int64) -> np.int64:
    i = key & (tt.shape[0] - 1)
    if tt[i, 0] == key:
        data = tt[i, 1]
        if data != 0:
            return data
    return np.int64(-1)


@njit(cache=False)
def tt_store(
    tt: np.ndarray,
    key: np.int64,
    mv: np.int64,
    depth: np.int64,
    flag: np.int64,
    score: np.int64,
    generation: np.int64,
) -> None:
    i = key & (tt.shape[0] - 1)
    old = tt[i, 1]
    if old != 0 and tt[i, 0] != key:
        old_depth = (old >> 17) & 0x7F
        old_gen = (old >> 42) & 0xFF
        if old_gen == generation and old_depth > depth + 2:
            return
    if score > 32000:
        score = 32000
    elif score < -32000:
        score = -32000
    tt[i, 0] = key
    tt[i, 1] = (
        TT_VALID
        | (mv & 0x1FFFF)
        | (depth << 17)
        | (flag << 24)
        | ((score + 32768) << 26)
        | (generation << 42)
    )


@njit(cache=False)
def _to_tt(score: np.int64, ply: np.int64) -> np.int64:
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


@njit(cache=False)
def _from_tt(score: np.int64, ply: np.int64) -> np.int64:
    if score >= MATE_BOUND:
        return score - ply
    if score <= -MATE_BOUND:
        return score + ply
    return score


# ----------------------------------------------------------------------- helpers


@njit(cache=False)
def insufficient_material(bbs: np.ndarray) -> bool:
    if bbs[PAWN] | bbs[6 + PAWN]:
        return False
    if bbs[3] | bbs[9] | bbs[4] | bbs[10]:  # rooks or queens
        return False
    return popcount(bbs[1] | bbs[2] | bbs[7] | bbs[8]) <= 1


@njit(cache=False)
def is_repetition(rep: np.ndarray, idx: np.int64, halfmove: np.int64) -> bool:
    """Has the position at rep[idx] appeared before within the halfmove window?"""
    key = rep[idx]
    limit = idx - halfmove
    if limit < 0:
        limit = 0
    i = idx - 4
    while i >= limit:
        if rep[i] == key:
            return True
        i -= 2
    return False


@njit(cache=False)
def score_moves(
    bbs: np.ndarray,
    st: np.ndarray,
    mb: np.ndarray,
    hist: np.ndarray,
    moves: np.ndarray,
    scores: np.ndarray,
    ord32: np.ndarray,
    base: np.int64,
    end: np.int64,
    tt_move: np.int64,
    ply: np.int64,
) -> None:
    side = st[ST_SIDE]
    k1 = ord32[KILLER_OFF + ply * 2]
    k2 = ord32[KILLER_OFF + ply * 2 + 1]
    counter = np.int64(0)
    if ply > 0:
        prev = hist[ply - 1, 5]
        if prev != 0:
            counter = ord32[COUNTER_OFF + ((prev & 63) * 64 + ((prev >> 6) & 63))]

    for i in range(base, end):
        mv = np.int64(moves[i])
        if mv == tt_move:
            scores[i] = 1 << 30
            continue
        to = (mv >> 6) & 63
        promo = (mv >> 12) & 7
        mtype = (mv >> 15) & 3
        victim = np.int64(mb[to])
        if victim >= 0 or mtype == MT_EP or promo != 0:
            if mtype == MT_EP:
                mvv = 10 * SEE_VALUE[PAWN] - SEE_VALUE[PAWN]
            elif victim >= 0:
                mvv = 10 * SEE_VALUE[victim % 6] - SEE_VALUE[np.int64(mb[mv & 63]) % 6]
            else:
                mvv = 0
            if promo != 0:
                mvv += 8000 * promo
            if see(bbs, mb, mv) >= 0:
                scores[i] = 1_000_000 + mvv
            else:
                scores[i] = -1_000_000 + mvv
        elif mv == k1:
            scores[i] = 900_000
        elif mv == k2:
            scores[i] = 800_000
        elif mv == counter:
            scores[i] = 700_000
        else:
            h = ord32[HISTORY_OFF + (side * 4096 + (mv & 63) * 64 + to)]
            if h > 600_000:
                h = 600_000
            scores[i] = h


@njit(cache=False)
def pick_move(moves: np.ndarray, scores: np.ndarray, i: np.int64, end: np.int64) -> np.int64:
    """Selection sort one step: swap the best remaining move into slot i."""
    best = i
    for j in range(i + 1, end):
        if scores[j] > scores[best]:
            best = j
    if best != i:
        moves[i], moves[best] = moves[best], moves[i]
        scores[i], scores[best] = scores[best], scores[i]
    return np.int64(moves[i])


@njit(cache=False)
def _update_history(ord32: np.ndarray, index: np.int64, bonus: np.int64) -> None:
    """Gravity update: large values decay towards a ceiling instead of running away."""
    if bonus > 1200:
        bonus = 1200
    elif bonus < -1200:
        bonus = -1200
    current = ord32[index]
    ord32[index] = current + bonus - (current * abs(bonus)) // 16384


# --------------------------------------------------------------------- quiescence


@njit(cache=False)
def qsearch(
    bbs: np.ndarray,
    st: np.ndarray,
    mb: np.ndarray,
    hist: np.ndarray,
    moves: np.ndarray,
    scores: np.ndarray,
    ord32: np.ndarray,
    pc: np.ndarray,
    rep: np.ndarray,
    info: np.ndarray,
    alpha: np.int64,
    beta: np.int64,
    ply: np.int64,
    deadline: float,
) -> np.int64:
    if info[I_STOP] != 0:
        return np.int64(0)
    info[I_NODES] += 1
    if (info[I_NODES] & CHECK_INTERVAL) == 0 and _now() >= deadline:
        info[I_STOP] = 1
        return np.int64(0)
    if ply > info[I_SELDEPTH]:
        info[I_SELDEPTH] = ply

    if ply >= MAX_PLY - 1:
        return evaluate(bbs, st, pc)
    if st[ST_HALF] >= 100 or insufficient_material(bbs):
        return np.int64(DRAW)

    side = st[ST_SIDE]
    checked = attacked(bbs, king_square(bbs, side), 1 - side, bbs[OCC_W] | bbs[OCC_B])

    base = ply * MAX_MOVES
    if checked:
        stand = -INF
        end = gen_moves(bbs, st, moves, base, False)
    else:
        stand = evaluate(bbs, st, pc)
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand
        end = gen_moves(bbs, st, moves, base, True)

    score_moves(bbs, st, mb, hist, moves, scores, ord32, base, end, np.int64(0), ply)

    best = stand
    legal = 0
    for i in range(base, end):
        mv = pick_move(moves, scores, i, end)
        if not checked:
            # Losing captures and hopeless ones cannot rescue the position.
            if scores[i] < 0:
                continue
            to = (mv >> 6) & 63
            victim = np.int64(mb[to])
            gain = SEE_VALUE[PAWN] if victim < 0 else SEE_VALUE[victim % 6]
            if stand + gain + 150 < alpha:
                continue
        if not _make(bbs, st, mb, hist, rep, info, ply, mv):
            continue
        legal += 1
        value = -qsearch(
            bbs, st, mb, hist, moves, scores, ord32, pc, rep, info, -beta, -alpha, ply + 1, deadline
        )
        unmake_move(bbs, st, mb, hist, ply, mv)
        if info[I_STOP] != 0:
            return np.int64(0)
        if value > best:
            best = value
            if value > alpha:
                alpha = value
                if value >= beta:
                    return value
    if checked and legal == 0:
        return np.int64(-MATE + ply)
    return best


@njit(cache=False)
def _make(
    bbs: np.ndarray,
    st: np.ndarray,
    mb: np.ndarray,
    hist: np.ndarray,
    rep: np.ndarray,
    info: np.ndarray,
    ply: np.int64,
    mv: np.int64,
) -> bool:
    """Make a pseudo-legal move, rejecting it if it leaves our king attacked."""
    side = st[ST_SIDE]
    make_move(bbs, st, mb, hist, ply, mv)
    if attacked(bbs, king_square(bbs, side), 1 - side, bbs[OCC_W] | bbs[OCC_B]):
        unmake_move(bbs, st, mb, hist, ply, mv)
        return False
    hist[ply, 5] = mv
    rep[info[I_REP_BASE] + ply + 1] = st[ST_HASH]
    return True


# -------------------------------------------------------------------------- PVS


@njit(cache=False)
def pvs(
    bbs: np.ndarray,
    st: np.ndarray,
    mb: np.ndarray,
    hist: np.ndarray,
    moves: np.ndarray,
    scores: np.ndarray,
    tt: np.ndarray,
    ord32: np.ndarray,
    pc: np.ndarray,
    rep: np.ndarray,
    info: np.ndarray,
    depth: np.int64,
    alpha: np.int64,
    beta: np.int64,
    ply: np.int64,
    deadline: float,
    can_null: bool,
) -> np.int64:
    if info[I_STOP] != 0:
        return np.int64(0)
    info[I_NODES] += 1
    if (info[I_NODES] & CHECK_INTERVAL) == 0 and _now() >= deadline:
        info[I_STOP] = 1
        return np.int64(0)

    key = st[ST_HASH]
    root = ply == 0

    if not root:
        if st[ST_HALF] >= 100 or insufficient_material(bbs):
            return np.int64(DRAW)
        if is_repetition(rep, info[I_REP_BASE] + ply, st[ST_HALF]):
            return np.int64(DRAW)
        # Mate-distance pruning keeps the tree from chasing longer mates.
        if alpha < -MATE + ply:
            alpha = -MATE + ply
        if beta > MATE - ply - 1:
            beta = MATE - ply - 1
        if alpha >= beta:
            return alpha

    side = st[ST_SIDE]
    checked = attacked(bbs, king_square(bbs, side), 1 - side, bbs[OCC_W] | bbs[OCC_B])
    if checked:
        depth += 1

    if depth <= 0 or ply >= MAX_PLY - 1:
        return qsearch(
            bbs, st, mb, hist, moves, scores, ord32, pc, rep, info, alpha, beta, ply, deadline
        )

    pv_node = beta - alpha > 1
    tt_move = np.int64(0)
    data = tt_probe(tt, key)
    if data >= 0:
        tt_move = data & 0x1FFFF
        tt_depth = (data >> 17) & 0x7F
        tt_flag = (data >> 24) & 3
        tt_score = _from_tt(((data >> 26) & 0xFFFF) - 32768, ply)
        if not pv_node and not root and tt_depth >= depth:
            if tt_flag == TT_EXACT:
                return tt_score
            if tt_flag == TT_LOWER and tt_score >= beta:
                return tt_score
            if tt_flag == TT_UPPER and tt_score <= alpha:
                return tt_score

    # Internal iterative reduction: without a hash move, search shallower first.
    if pv_node and tt_move == 0 and depth >= 4:
        depth -= 1

    static_eval = np.int64(0) if checked else evaluate(bbs, st, pc)

    if not pv_node and not checked and abs(beta) < MATE_BOUND:
        # Reverse futility
        if depth <= 6 and static_eval - 70 * depth >= beta:
            return static_eval
        # Razoring
        if depth <= 2 and static_eval + 150 * depth <= alpha:
            value = qsearch(
                bbs, st, mb, hist, moves, scores, ord32, pc, rep, info, alpha, beta, ply, deadline
            )
            if value <= alpha:
                return value
        # Null move
        if (
            can_null
            and depth >= 3
            and static_eval >= beta
            and non_pawn_material(bbs, side) > 0
        ):
            reduction = 2 + depth // 4
            make_null(bbs, st, hist, ply)
            hist[ply, 5] = 0
            rep[info[I_REP_BASE] + ply + 1] = st[ST_HASH]
            value = -pvs(
                bbs, st, mb, hist, moves, scores, tt, ord32, pc, rep, info,
                depth - 1 - reduction, -beta, -beta + 1, ply + 1, deadline, False,
            )
            unmake_null(bbs, st, hist, ply)
            if info[I_STOP] != 0:
                return np.int64(0)
            if value >= beta:
                return beta if value >= MATE_BOUND else value

    base = ply * MAX_MOVES
    end = gen_moves(bbs, st, moves, base, False)
    score_moves(bbs, st, mb, hist, moves, scores, ord32, base, end, tt_move, ply)

    original_alpha = alpha
    best_score = -INF
    best_move = np.int64(0)
    played = 0
    quiet_base = QUIET_OFF + ply * QUIET_PER_PLY
    quiet_count = 0

    for i in range(base, end):
        mv = pick_move(moves, scores, i, end)
        to = (mv >> 6) & 63
        promo = (mv >> 12) & 7
        mtype = (mv >> 15) & 3
        capture = mb[to] >= 0 or mtype == MT_EP
        quiet = not capture and promo == 0

        if (
            not pv_node
            and not checked
            and quiet
            and depth <= 3
            and played >= 3
            and abs(alpha) < MATE_BOUND
            and static_eval + 120 * depth <= alpha
        ):
            continue

        if not _make(bbs, st, mb, hist, rep, info, ply, mv):
            continue

        # Hold a legal root move from the moment one exists, so a search stopped
        # inside the very first subtree still has something to play.
        if root and played == 0:
            info[I_ROOT_MOVE] = mv

        gives_check = attacked(
            bbs, king_square(bbs, st[ST_SIDE]), 1 - st[ST_SIDE], bbs[OCC_W] | bbs[OCC_B]
        )

        reduction = np.int64(0)
        if (
            depth >= 3
            and played >= 3
            and quiet
            and not checked
            and not gives_check
            and scores[i] < 700_000
        ):
            d = depth if depth < 63 else 63
            m = played if played < 63 else 63
            reduction = np.int64(LMR_TABLE[d, m])
            if pv_node and reduction > 0:
                reduction -= 1
            if reduction > depth - 2:
                reduction = depth - 2
            if reduction < 0:
                reduction = 0

        if played == 0:
            value = -pvs(
                bbs, st, mb, hist, moves, scores, tt, ord32, pc, rep, info,
                depth - 1, -beta, -alpha, ply + 1, deadline, True,
            )
        else:
            value = -pvs(
                bbs, st, mb, hist, moves, scores, tt, ord32, pc, rep, info,
                depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, deadline, True,
            )
            if value > alpha and reduction > 0:
                value = -pvs(
                    bbs, st, mb, hist, moves, scores, tt, ord32, pc, rep, info,
                    depth - 1, -alpha - 1, -alpha, ply + 1, deadline, True,
                )
            if value > alpha and value < beta:
                value = -pvs(
                    bbs, st, mb, hist, moves, scores, tt, ord32, pc, rep, info,
                    depth - 1, -beta, -alpha, ply + 1, deadline, True,
                )

        unmake_move(bbs, st, mb, hist, ply, mv)
        if info[I_STOP] != 0:
            return np.int64(0)

        played += 1
        if quiet and quiet_count < QUIET_PER_PLY:
            ord32[quiet_base + quiet_count] = mv
            quiet_count += 1

        if value > best_score:
            best_score = value
            best_move = mv
            if root:
                info[I_ROOT_MOVE] = mv
                info[I_ROOT_SCORE] = value
            if value > alpha:
                alpha = value
                if value >= beta:
                    if quiet:
                        bonus = depth * depth
                        _store_killer(ord32, ply, mv)
                        _update_history(
                            ord32, HISTORY_OFF + (side * 4096 + (mv & 63) * 64 + to), bonus
                        )
                        if ply > 0 and hist[ply - 1, 5] != 0:
                            prev = hist[ply - 1, 5]
                            ord32[COUNTER_OFF + ((prev & 63) * 64 + ((prev >> 6) & 63))] = mv
                        # Quiets that were tried and failed get the opposite nudge.
                        for q in range(quiet_count - 1):
                            other = np.int64(ord32[quiet_base + q])
                            _update_history(
                                ord32,
                                HISTORY_OFF
                                + (side * 4096 + (other & 63) * 64 + ((other >> 6) & 63)),
                                -bonus,
                            )
                    tt_store(tt, key, mv, depth, TT_LOWER, _to_tt(value, ply), info[I_GENERATION])
                    return value

    if played == 0:
        return np.int64(-MATE + ply) if checked else np.int64(DRAW)

    flag = TT_EXACT if best_score > original_alpha else TT_UPPER
    tt_store(tt, key, best_move, depth, flag, _to_tt(best_score, ply), info[I_GENERATION])
    return best_score


@njit(cache=False)
def _store_killer(ord32: np.ndarray, ply: np.int64, mv: np.int64) -> None:
    slot = KILLER_OFF + ply * 2
    if ord32[slot] != mv:
        ord32[slot + 1] = ord32[slot]
        ord32[slot] = mv


# ------------------------------------------------------------------------- root


def search(
    bbs: np.ndarray,
    st: np.ndarray,
    mb: np.ndarray,
    hist: np.ndarray,
    moves: np.ndarray,
    scores: np.ndarray,
    tt: np.ndarray,
    ord32: np.ndarray,
    pc: np.ndarray,
    rep: np.ndarray,
    info: np.ndarray,
    max_depth: int,
    soft_deadline: float,
    hard_deadline: float,
) -> int:
    """Iterative deepening with aspiration windows. Returns the best move.

    This driver stays in Python on purpose. It runs a few dozen times per move,
    so it costs nothing measurable, and keeping it out of the jitted call graph
    cut roughly ten seconds off the one-off compile that every game pays.
    """
    info[I_NODES] = 0
    info[I_SELDEPTH] = 0
    info[I_STOP] = 0
    info[I_ROOT_MOVE] = 0
    info[I_ROOT_SCORE] = 0
    info[I_DEPTH_DONE] = 0
    info[I_GENERATION] = (int(info[I_GENERATION]) + 1) & 0xFF
    ord32[KILLER_OFF : KILLER_OFF + KILLER_LEN] = 0

    score = 0
    for depth in range(1, max_depth + 1):
        if depth <= 3:
            score = int(
                pvs(
                    bbs, st, mb, hist, moves, scores, tt, ord32, pc, rep, info,
                    np.int64(depth), np.int64(-INF), np.int64(INF), np.int64(0),
                    hard_deadline, True,
                )
            )
        else:
            window = 25
            alpha = score - window
            beta = score + window
            while True:
                value = int(
                    pvs(
                        bbs, st, mb, hist, moves, scores, tt, ord32, pc, rep, info,
                        np.int64(depth), np.int64(alpha), np.int64(beta), np.int64(0),
                        hard_deadline, True,
                    )
                )
                if info[I_STOP] != 0:
                    break
                if value <= alpha:
                    beta = (alpha + beta) // 2
                    alpha = max(-INF, value - window)
                    window *= 2
                elif value >= beta:
                    beta = min(INF, value + window)
                    window *= 2
                else:
                    score = value
                    break
                if window > 1200:
                    alpha = -INF
                    beta = INF

        if info[I_STOP] != 0:
            break
        info[I_DEPTH_DONE] = depth
        if score >= MATE_BOUND or score <= -MATE_BOUND:
            break
        if time.perf_counter() >= soft_deadline:
            break

    return int(info[I_ROOT_MOVE])

"""Python PVS reference. Not used by get_move.

Live search is engine.bb_search (numba). See ROADMAP.md.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from engine.const import FLAG_CAP, FLAG_EP, FLAG_PROMO, INF, MATE, MATE_GATE, PAWN, SEE_VAL, m_from, m_promo, m_to
from engine.endgame import insufficient_material
from engine.eval import evaluate
from engine.moves import gen_legal
from engine.params import (
    ASPIRATION,
    DELTA_MARGIN,
    FUTILITY_MARGIN,
    FUTILITY_MAX_DEPTH,
    HIST_MAX,
    LMR_C,
    LMR_K,
    LMR_MIN_DEPTH,
    LMR_MIN_MOVE,
    NMP_BASE,
    NMP_DIV,
    NMP_MIN_DEPTH,
    RAZOR_MARGIN,
    RAZOR_MAX_DEPTH,
    RFP_MARGIN,
    RFP_MAX_DEPTH,
)
from engine.pos import Position
from engine.see import see
from engine.tt import EXACT, LOWER, TT, UPPER

MAX_PLY = 96
CHECK_NODES = 64
StopFn = Callable[[], bool]


class StopSearch(Exception):
    pass


def _lmr_table() -> list[list[int]]:
    table = [[0] * 64 for _ in range(64)]
    for d in range(1, 64):
        for m in range(1, 64):
            table[d][m] = max(0, int(LMR_C + math.log(d) * math.log(m) * LMR_K))
    return table


LMR = _lmr_table()


class Searcher:
    def __init__(self) -> None:
        self.tt = TT()
        self.killers = [[0, 0] for _ in range(MAX_PLY)]
        self.history = [[[0] * 64 for _ in range(64)] for _ in range(2)]
        self.nodes = 0
        self.seldepth = 0
        self.stop: StopFn = lambda: False
        self.root_depth = 0
        self.best_move = 0
        self.root_score = 0
        self.nmp_cut = 0
        self.futility_cut = 0
        self.rfp_cut = 0

    def search(
        self,
        pos: Position,
        max_depth: int,
        stop: StopFn,
        soft_stop: StopFn | None = None,
    ) -> tuple[int, int, int]:
        self.stop = stop
        self.nodes = 0
        self.seldepth = 0
        self.root_depth = 0
        self.nmp_cut = self.futility_cut = self.rfp_cut = 0
        self.killers = [[0, 0] for _ in range(MAX_PLY)]
        moves = gen_legal(pos)
        if not moves:
            return 0, -MATE if pos.in_check() else 0, 0
        if len(moves) == 1:
            return moves[0], 0, 1
        self.best_move = moves[0]
        completed_move = moves[0]
        score = 0
        completed = 0
        for depth in range(1, max_depth + 1):
            if completed >= 1 and (self.stop() or (soft_stop is not None and soft_stop())):
                break
            self.root_depth = depth
            alpha, beta = score - ASPIRATION, score + ASPIRATION
            if depth <= 2:
                alpha, beta = -INF, INF
            try:
                while True:
                    val = self._search(pos, depth, alpha, beta, 0, True)
                    if val <= alpha:
                        alpha = -INF
                        continue
                    if val >= beta:
                        beta = INF
                        continue
                    score = val
                    break
            except StopSearch:
                break
            if self.best_move:
                completed_move = self.best_move
                completed = depth
                self.root_score = score
            if abs(score) > MATE_GATE:
                break
        return completed_move, self.root_score, completed

    def _check_time(self) -> None:
        if self.nodes & (CHECK_NODES - 1) == 0 and self.stop():
            raise StopSearch

    def _search(self, pos: Position, depth: int, alpha: int, beta: int, ply: int, pv: bool) -> int:
        self._check_time()
        self.nodes += 1
        if ply > self.seldepth:
            self.seldepth = ply
        if ply > 0:
            if pos.halfmove >= 100 or pos.is_repeat() or insufficient_material(pos):
                return 0
            if ply >= MAX_PLY - 1:
                return evaluate(pos)

        mate_alpha = -MATE + ply
        mate_beta = MATE - ply - 1
        if alpha < mate_alpha:
            alpha = mate_alpha
        if beta > mate_beta:
            beta = mate_beta
        if alpha >= beta:
            return alpha

        in_check = pos.in_check()
        if depth <= 0:
            return self._qsearch(pos, alpha, beta, ply, 0)

        tt_move = 0
        hit = self.tt.probe(pos.hash, ply)
        if hit is not None:
            tmove, tscore, tdepth, tflag = hit
            tt_move = tmove
            if not pv and tdepth >= depth:
                if tflag == EXACT:
                    return tscore
                if tflag == LOWER and tscore >= beta:
                    return tscore
                if tflag == UPPER and tscore <= alpha:
                    return tscore

        static_eval = evaluate(pos) if not in_check else 0

        if (
            not pv
            and not in_check
            and depth <= RAZOR_MAX_DEPTH
            and static_eval + RAZOR_MARGIN * depth <= alpha
            and abs(alpha) < MATE_GATE
        ):
            q = self._qsearch(pos, alpha, beta, ply, 0)
            if q <= alpha:
                return q

        if (
            not pv
            and not in_check
            and depth <= RFP_MAX_DEPTH
            and static_eval - RFP_MARGIN * depth >= beta
            and abs(beta) < MATE_GATE
        ):
            self.rfp_cut += 1
            return static_eval

        if (
            not pv
            and not in_check
            and depth >= NMP_MIN_DEPTH
            and pos.has_non_pawn(pos.side)
            and static_eval >= beta
            and abs(beta) < MATE_GATE
        ):
            r = NMP_BASE + depth // NMP_DIV
            pos.make(0)
            try:
                score = -self._search(pos, depth - 1 - r, -beta, -beta + 1, ply + 1, False)
            finally:
                pos.unmake()
            if score >= beta:
                self.nmp_cut += 1
                return beta if score > MATE_GATE else score

        moves = self._ordered(pos, ply, tt_move, False)
        if not moves:
            return -MATE + ply if in_check else 0

        best = -INF
        best_move = 0
        orig_alpha = alpha
        for legal_i, move in enumerate(moves):
            quiet = not (move & (FLAG_CAP | FLAG_PROMO | FLAG_EP))
            if (
                not pv
                and not in_check
                and quiet
                and depth <= FUTILITY_MAX_DEPTH
                and legal_i >= 1
                and static_eval + FUTILITY_MARGIN * depth <= alpha
                and abs(alpha) < MATE_GATE
            ):
                self.futility_cut += 1
                continue

            reduction = 0
            if not pv and not in_check and quiet and legal_i >= LMR_MIN_MOVE and depth >= LMR_MIN_DEPTH:
                reduction = LMR[min(depth, 63)][min(legal_i, 63)]
                hist = self.history[pos.side][m_from(move)][m_to(move)]
                if hist < 0:
                    reduction += 1
                if hist > 2000:
                    reduction = max(0, reduction - 1)

            pos.make(move)
            try:
                gives_check = pos.in_check()
                ext = 1 if gives_check and ply < 2 * self.root_depth + 4 else 0
                new_depth = depth - 1 + ext
                if legal_i == 0:
                    score = -self._search(pos, new_depth, -beta, -alpha, ply + 1, pv)
                else:
                    score = -self._search(pos, new_depth - reduction, -alpha - 1, -alpha, ply + 1, False)
                    if score > alpha and reduction:
                        score = -self._search(pos, new_depth, -alpha - 1, -alpha, ply + 1, False)
                    if score > alpha and (pv or score < beta):
                        score = -self._search(pos, new_depth, -beta, -alpha, ply + 1, True)
            finally:
                pos.unmake()

            if score > best:
                best = score
                best_move = move
                if ply == 0:
                    self.best_move = move
                if score > alpha:
                    alpha = score
                    if score >= beta:
                        if quiet:
                            self._cutoff(pos, move, ply, depth)
                        break

        flag = EXACT
        if best <= orig_alpha:
            flag = UPPER
        elif best >= beta:
            flag = LOWER
        self.tt.store(pos.hash, best_move, best, depth, flag, ply)
        return best

    def _qsearch(self, pos: Position, alpha: int, beta: int, ply: int, qs_ply: int) -> int:
        self._check_time()
        self.nodes += 1
        if ply >= MAX_PLY - 1 or qs_ply >= 16:
            return evaluate(pos)
        in_check = pos.in_check()
        if not in_check:
            stand = evaluate(pos)
            if stand >= beta:
                return stand
            if stand > alpha:
                alpha = stand
        else:
            stand = -INF
        moves = self._ordered(pos, ply, 0, captures_only=not in_check)
        if in_check and not moves:
            return -MATE + ply
        best = stand if not in_check else -INF
        for move in moves:
            if not in_check:
                promo = m_promo(move)
                gain = SEE_VAL[PAWN] if move & FLAG_EP else (
                    SEE_VAL[pos.sq[m_to(move)] % 6] if pos.sq[m_to(move)] >= 0 else 0
                )
                if promo:
                    gain += SEE_VAL[promo] - SEE_VAL[PAWN]
                if stand + gain + DELTA_MARGIN < alpha:
                    continue
                if see(pos, move) < 0:
                    continue
            pos.make(move)
            try:
                score = -self._qsearch(pos, -beta, -alpha, ply + 1, qs_ply + 1)
            finally:
                pos.unmake()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if score >= beta:
                        return score
        return best

    def _ordered(self, pos: Position, ply: int, tt_move: int, captures_only: bool) -> list[int]:
        moves = gen_legal(pos, captures_only=captures_only)
        if len(moves) <= 1:
            return moves
        scored: list[tuple[int, int]] = []
        k1, k2 = self.killers[ply]
        hist = self.history[pos.side]
        for move in moves:
            if move == tt_move:
                sc = 2_000_000
            elif move & FLAG_PROMO:
                sc = 1_500_000 + m_promo(move) * 2000
                if move & FLAG_CAP:
                    sc += 400
            elif move & (FLAG_CAP | FLAG_EP):
                victim = SEE_VAL[PAWN] if move & FLAG_EP else (
                    SEE_VAL[pos.sq[m_to(move)] % 6] if pos.sq[m_to(move)] >= 0 else 0
                )
                attacker = SEE_VAL[pos.sq[m_from(move)] % 6]
                sc = 1_000_000 + see(pos, move) * 8 + victim * 16 - attacker
            elif move == k1:
                sc = 800_000
            elif move == k2:
                sc = 750_000
            else:
                sc = hist[m_from(move)][m_to(move)]
            scored.append((sc, move))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def _cutoff(self, pos: Position, move: int, ply: int, depth: int) -> None:
        k = self.killers[ply]
        if k[0] != move:
            k[1] = k[0]
            k[0] = move
        frm, to = m_from(move), m_to(move)
        h = self.history[pos.side]
        h[frm][to] += depth * depth
        if h[frm][to] > HIST_MAX:
            self._age_history()

    def _age_history(self) -> None:
        for c in range(2):
            hc = self.history[c]
            for f in range(64):
                row = hc[f]
                for t in range(64):
                    row[t] >>= 1

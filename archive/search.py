"""Iterative-deepening PVS search with a transposition table."""

from __future__ import annotations

import time
from typing import List, Optional

import chess
import chess.polyglot

from engine import evaluate
from engine.book import pack_move, unpack_move

MATE = 30000
MATE_BOUND = 29000
DRAW = 0
MAX_PLY = 96
INF = 32000

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

# Packed-move sentinel
NO_MOVE = 0


class _TT:
    __slots__ = ("size", "mask", "key", "move", "depth", "flag", "score")

    def __init__(self, n_entries: int = 1 << 22) -> None:
        # Power of two. ~4M * ~16 B worth of parallel arrays ≈ 64–80 MB.
        self.size = n_entries
        self.mask = n_entries - 1
        self.key = [0] * n_entries
        self.move = [0] * n_entries
        self.depth = [0] * n_entries
        self.flag = [0] * n_entries
        self.score = [0] * n_entries

    def probe(self, zkey: int) -> tuple[int, int, int, int] | None:
        i = zkey & self.mask
        if self.key[i] == zkey:
            return self.move[i], self.depth[i], self.flag[i], self.score[i]
        return None

    def store(self, zkey: int, move: int, depth: int, flag: int, score: int) -> None:
        i = zkey & self.mask
        # Replace if deeper or empty / different key
        if self.key[i] != zkey or depth >= self.depth[i]:
            self.key[i] = zkey
            self.move[i] = move
            self.depth[i] = depth
            self.flag[i] = flag
            self.score[i] = score


def _to_tt(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


def _from_tt(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score - ply
    if score <= -MATE_BOUND:
        return score + ply
    return score


class SearchStopped(Exception):
    pass


class Engine:
    def __init__(self) -> None:
        self.tt = _TT()
        self.nodes = 0
        self.seldepth = 0
        self.stop = False
        self.deadline = 0.0
        self.soft_deadline = 0.0
        self.check_every = 64
        self.killers: List[List[int]] = [[0, 0] for _ in range(MAX_PLY + 4)]
        self.history = [[[0] * 64 for _ in range(64)] for _ in range(2)]
        self.counter = [[0] * 64 for _ in range(64)]
        self.root_best = NO_MOVE
        self.root_score = 0
        self.rep: dict[int, int] = {}

    def new_game(self) -> None:
        self.killers = [[0, 0] for _ in range(MAX_PLY + 4)]
        self.history = [[[0] * 64 for _ in range(64)] for _ in range(2)]
        self.counter = [[0] * 64 for _ in range(64)]
        self.root_best = NO_MOVE
        self.rep.clear()
        # Keep the TT; chess is still chess.

    def search(
        self,
        board: chess.Board,
        time_ms: int,
        rep_keys: Optional[List[int]] = None,
        max_depth: int = 64,
        soft_ms: Optional[int] = None,
    ) -> chess.Move:
        self.stop = False
        self.nodes = 0
        self.seldepth = 0
        now = time.perf_counter()
        self.deadline = now + max(0.001, time_ms / 1000.0)
        self.soft_deadline = now + max(0.001, (soft_ms if soft_ms is not None else time_ms) / 1000.0)
        self.root_best = NO_MOVE
        self.root_score = 0

        self.rep = {}
        if rep_keys:
            for k in rep_keys:
                self.rep[k] = self.rep.get(k, 0) + 1

        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("no legal moves")
        if len(legal) == 1:
            return legal[0]

        # Depth-1 fallback so we always have a move
        best = legal[0]
        score = 0
        for depth in range(1, max_depth + 1):
            try:
                if depth <= 3:
                    score = self._pvs(board, depth, -INF, INF, 0)
                else:
                    # Aspiration
                    window = 30
                    alpha = score - window
                    beta = score + window
                    while True:
                        val = self._pvs(board, depth, alpha, beta, 0)
                        if val <= alpha:
                            alpha = max(-INF, alpha - window)
                            window *= 2
                        elif val >= beta:
                            beta = min(INF, beta + window)
                            window *= 2
                        else:
                            score = val
                            break
                        if window > 2000:
                            score = self._pvs(board, depth, -INF, INF, 0)
                            break
                if self.root_best:
                    best = unpack_move(self.root_best)
                    self.root_score = score
                # Mate found: no point going deeper
                if abs(score) >= MATE_BOUND:
                    break
            except SearchStopped:
                break
            if time.perf_counter() >= self.soft_deadline:
                break

        if best not in board.legal_moves:
            best = legal[0]
        return best

    def extract_ponder(self, board: chess.Board, our_move: chess.Move) -> chess.Move | None:
        """Best predicted reply after `our_move`, from the TT PV."""
        board.push(our_move)
        try:
            hit = self.tt.probe(chess.polyglot.zobrist_hash(board))
            if not hit or not hit[0]:
                return None
            move = unpack_move(hit[0])
            if move in board.legal_moves:
                return move
            return None
        finally:
            board.pop()

    def _time_up(self) -> bool:
        if self.stop:
            return True
        if (self.nodes & (self.check_every - 1)) != 0:
            return False
        return time.perf_counter() >= self.deadline

    def _pvs(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        if self._time_up():
            raise SearchStopped()

        self.nodes += 1
        if ply > self.seldepth:
            self.seldepth = ply

        if ply > 0:
            if board.halfmove_clock >= 100:
                return DRAW
            if board.is_insufficient_material():
                return DRAW
            key = chess.polyglot.zobrist_hash(board)
            if self.rep.get(key, 0) >= 2:
                return DRAW
        else:
            key = chess.polyglot.zobrist_hash(board)

        in_check = board.is_check()
        if in_check:
            depth += 1

        if depth <= 0 or ply >= MAX_PLY - 1:
            return self._quiesce(board, alpha, beta, ply)

        pv_node = beta - alpha > 1
        tt_move = NO_MOVE
        hit = self.tt.probe(key)
        if hit:
            ttm, ttd, ttf, tts = hit
            tt_move = ttm
            tts = _from_tt(tts, ply)
            if not pv_node and ttd >= depth:
                if ttf == TT_EXACT:
                    return tts
                if ttf == TT_LOWER and tts >= beta:
                    return tts
                if ttf == TT_UPPER and tts <= alpha:
                    return tts

        # Internal iterative reduction: no hash move, shallower first.
        if pv_node and not tt_move and depth >= 4:
            depth -= 1

        static_eval = evaluate.evaluate(board) if not in_check else 0

        # Reverse futility pruning
        if (
            not pv_node
            and not in_check
            and depth <= 6
            and static_eval - 70 * depth >= beta
            and abs(beta) < MATE_BOUND
        ):
            return static_eval

        # Razoring
        if not pv_node and not in_check and depth <= 2 and static_eval + 150 * depth <= alpha:
            q = self._quiesce(board, alpha, beta, ply)
            if q <= alpha:
                return q

        # Null-move pruning
        if (
            not pv_node
            and not in_check
            and depth >= 3
            and static_eval >= beta
            and evaluate.non_pawn_material(board, board.turn) > 0
        ):
            r = 2 + depth // 4
            board.push(chess.Move.null())
            self._rep_push(board)
            try:
                score = -self._pvs(board, depth - 1 - r, -beta, -beta + 1, ply + 1)
            finally:
                self._rep_pop(board)
                board.pop()
            if score >= beta:
                if abs(score) >= MATE_BOUND:
                    return beta
                return score

        moves = self._ordered_moves(board, tt_move, ply, captures_only=False)
        if not moves:
            return -MATE + ply if in_check else DRAW

        orig_alpha = alpha
        best_score = -INF
        best_move = NO_MOVE
        move_index = 0
        k1, k2 = self.killers[ply]

        for packed, _order in moves:
            move = unpack_move(packed)
            is_cap = board.is_capture(move) or board.is_en_passant(move)
            is_promo = move.promotion is not None

            # Futility: skip late quiet moves at shallow depth
            if (
                not pv_node
                and not in_check
                and not is_cap
                and not is_promo
                and depth <= 3
                and move_index >= 3
                and static_eval + 120 * depth <= alpha
                and abs(alpha) < MATE_BOUND
            ):
                move_index += 1
                continue

            board.push(move)
            gives_check = board.is_check()
            self._rep_push(board)

            is_killer = packed == k1 or packed == k2
            reduction = 0
            if (
                depth >= 3
                and move_index >= 3
                and not is_cap
                and not is_promo
                and not in_check
                and not gives_check
                and not is_killer
            ):
                reduction = 1 + (1 if move_index >= 6 else 0) + (1 if depth >= 6 else 0)
                reduction = min(reduction, max(0, depth - 2))

            try:
                if move_index == 0:
                    score = -self._pvs(board, depth - 1, -beta, -alpha, ply + 1)
                else:
                    score = -self._pvs(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1)
                    if score > alpha and reduction:
                        score = -self._pvs(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                    if score > alpha and score < beta:
                        score = -self._pvs(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                self._rep_pop(board)
                board.pop()

            move_index += 1

            if score > best_score:
                best_score = score
                best_move = packed
                if ply == 0:
                    self.root_best = packed
                if score > alpha:
                    alpha = score
                    if score >= beta:
                        if not is_cap:
                            self._update_quiet_cut(board, packed, depth, ply)
                        self.tt.store(key, packed, depth, TT_LOWER, _to_tt(score, ply))
                        return score

        if best_move and not board.is_capture(unpack_move(best_move)):
            self._add_history(board, best_move, depth)

        flag = TT_EXACT if best_score > orig_alpha else TT_UPPER
        self.tt.store(key, best_move, depth, flag, _to_tt(best_score, ply))
        return best_score

    def _quiesce(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        if self._time_up():
            raise SearchStopped()
        self.nodes += 1
        if ply >= MAX_PLY - 1:
            return evaluate.evaluate(board)

        if board.halfmove_clock >= 100 or board.is_insufficient_material():
            return DRAW

        in_check = board.is_check()
        if in_check:
            moves = self._ordered_moves(board, NO_MOVE, ply, captures_only=False)
            if not moves:
                return -MATE + ply
            stand = -INF
        else:
            stand = evaluate.evaluate(board)
            if stand >= beta:
                return stand
            if stand > alpha:
                alpha = stand
            moves = self._ordered_moves(board, NO_MOVE, ply, captures_only=True)

        best = stand
        for packed, _order in moves:
            move = unpack_move(packed)
            if not in_check:
                if not evaluate.see_ge(board, move, 0):
                    continue
                victim = board.piece_type_at(move.to_square)
                delta = evaluate.SEE_VAL[victim] if victim else 100
                if stand + delta + 150 < alpha:
                    continue
            board.push(move)
            self._rep_push(board)
            try:
                score = -self._quiesce(board, -beta, -alpha, ply + 1)
            finally:
                self._rep_pop(board)
                board.pop()
            if score > best:
                best = score
            if score >= beta:
                return score
            if score > alpha:
                alpha = score
        return best

    def _ordered_moves(
        self, board: chess.Board, tt_move: int, ply: int, captures_only: bool
    ) -> list[tuple[int, int]]:
        scored: list[tuple[int, int]] = []
        k1, k2 = self.killers[ply]
        side = 1 if board.turn else 0
        hist = self.history[side]
        prev = board.move_stack[-1] if board.move_stack else None
        cm = self.counter[prev.from_square][prev.to_square] if prev is not None else NO_MOVE
        it = board.generate_legal_captures() if captures_only else board.legal_moves
        for move in it:
            packed = pack_move(move)
            if packed == tt_move:
                score = 1_000_000
            elif board.is_capture(move) or board.is_en_passant(move) or move.promotion:
                victim = board.piece_type_at(move.to_square) or chess.PAWN
                attacker = board.piece_type_at(move.from_square) or chess.PAWN
                mvv = 10 * evaluate.SEE_VAL[victim] - evaluate.SEE_VAL[attacker]
                if move.promotion:
                    mvv += 8000 * move.promotion
                # Winning/equal captures first; losing captures after quiets.
                if evaluate.see(board, move) >= 0:
                    score = 100_000 + mvv
                else:
                    score = -20_000 + mvv
            elif packed == k1:
                score = 90_000
            elif packed == k2:
                score = 80_000
            elif packed == cm:
                score = 75_000
            else:
                score = hist[move.from_square][move.to_square]
            scored.append((packed, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _update_quiet_cut(self, board: chess.Board, packed: int, depth: int, ply: int) -> None:
        k = self.killers[ply]
        if k[0] != packed:
            k[1] = k[0]
            k[0] = packed
        side = 1 if board.turn else 0
        m = unpack_move(packed)
        bonus = depth * depth
        h = self.history[side][m.from_square][m.to_square]
        self.history[side][m.from_square][m.to_square] = h + bonus - (h * bonus) // 16384
        if board.move_stack:
            prev = board.move_stack[-1]
            self.counter[prev.from_square][prev.to_square] = packed

    def _add_history(self, board: chess.Board, packed: int, depth: int) -> None:
        side = 1 if board.turn else 0
        m = unpack_move(packed)
        bonus = depth * depth
        h = self.history[side][m.from_square][m.to_square]
        self.history[side][m.from_square][m.to_square] = h + bonus - (h * bonus) // 16384

    def _rep_push(self, board: chess.Board) -> None:
        key = chess.polyglot.zobrist_hash(board)
        self.rep[key] = self.rep.get(key, 0) + 1

    def _rep_pop(self, board: chess.Board) -> None:
        key = chess.polyglot.zobrist_hash(board)
        n = self.rep.get(key, 1) - 1
        if n <= 0:
            self.rep.pop(key, None)
        else:
            self.rep[key] = n

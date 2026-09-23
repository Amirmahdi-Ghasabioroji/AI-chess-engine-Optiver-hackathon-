"""Competition entry. The judge imports this module and calls get_move.

The move itself comes from `bb_search`, which numba compiles to machine code at
import time. Evaluation is PeSTO plus a linear piece-square residual we trained
on Stockfish labels. python-chess is used only at the edges: parsing the FEN we
are sent and proving that the move we send back is legal.

There is deliberately no pondering. A jitted search holds the GIL for its whole
call, so a background ponder thread could not be interrupted by the thread that
needs to answer the next move request, and a missed deadline loses the game
outright. The bitboard search is worth far more than the opponent's clock.
"""

from __future__ import annotations

import os
import time
import traceback

import chess
import numpy as np

import bb
import bb_eval
import bb_search
import book
import nnue
from bbpos import from_board, move_from_chess, move_to_uci

DEBUG = os.environ.get("CHESS_DEBUG") == "1"

# Wall-clock margin left for the runner to read our reply before the flag falls.
SAFETY_MS = 90

# Deepest iteration we will start; the search stops on the clock long before this.
MAX_DEPTH = 63


def _log(message: str) -> None:
    if DEBUG:
        print(message, flush=True)


class _Agent:
    def __init__(self) -> None:
        bb.init()
        self.tt, self.ord32, self.rep, self.info = bb_search.new_tables()
        self.pc = bb_eval.new_pawn_cache()
        self.hist, self.moves, self.scores = bb.new_buffers()
        self.keys: list[int] = []
        self.after_ours: chess.Board | None = None
        self.ply = 0

    # ---------------------------------------------------------------- game state

    def start_game(self, board: chess.Board, key: int) -> None:
        self.keys = [key]
        self.after_ours = None
        self.ply = 0
        self.ord32[:] = 0
        self.tt[:, :] = 0

    def sync(self, board: chess.Board, key: int) -> None:
        """Attach this request to the running game, or start a new one.

        One process serves one game, but a reused process must not inherit a
        repetition history that never happened.
        """
        if self.after_ours is None:
            self.start_game(board, key)
            return

        probe = self.after_ours.copy(stack=False)
        for move in probe.legal_moves:
            probe.push(move)
            if _key_of(probe) == key:
                self.keys.append(key)
                self.ply += 1
                return
            probe.pop()

        self.start_game(board, key)

    def note_our_move(self, board: chess.Board, move: chess.Move) -> None:
        nxt = board.copy(stack=False)
        nxt.push(move)
        self.keys.append(_key_of(nxt))
        self.after_ours = nxt
        self.ply += 1

    # -------------------------------------------------------------------- search

    def choose(self, board: chess.Board, time_left_ms: int) -> chess.Move:
        bbs, st, mb = from_board(board)

        history = self.keys[-(len(self.rep) - bb.MAX_PLY - 8) :]
        for i, key in enumerate(history):
            self.rep[i] = key
        self.info[bb_search.I_REP_BASE] = len(history) - 1

        soft_ms, hard_ms = _allocate(time_left_ms, self.ply)
        now = time.perf_counter()
        soft_deadline = now + soft_ms / 1000.0
        hard_deadline = now + hard_ms / 1000.0

        packed = int(
            bb_search.search(
                bbs,
                st,
                mb,
                self.hist,
                self.moves,
                self.scores,
                self.tt,
                self.ord32,
                self.pc,
                self.rep,
                self.info,
                np.int64(MAX_DEPTH),
                soft_deadline,
                hard_deadline,
            )
        )
        if packed == 0:
            raise ValueError("search returned no move")

        move = chess.Move.from_uci(move_to_uci(packed))
        _log(
            f"move {move.uci()} depth {self.info[bb_search.I_DEPTH_DONE]} "
            f"score {self.info[bb_search.I_ROOT_SCORE]} nodes {self.info[bb_search.I_NODES]} "
            f"soft {soft_ms} hard {hard_ms} left {time_left_ms}"
        )
        return move


def _key_of(board: chess.Board) -> int:
    _, st, _ = from_board(board)
    return int(st[bb.ST_HASH])


def _allocate(time_left_ms: int, ply: int) -> tuple[int, int]:
    """Return (soft_ms, hard_ms).

    `time_left_ms` excludes the increment, which the API never reports, so the
    increment is treated as reserve rather than as budget. Flag falls are the
    only way this engine has ever lost a game it was winning.
    """
    usable = max(1, time_left_ms - SAFETY_MS)
    if usable <= 90:
        return 12, min(35, usable)
    if usable <= 700:
        hard = min(70, usable)
        return min(35, hard), hard

    if ply < 16:
        moves_to_go = 36
    elif ply < 40:
        moves_to_go = 28
    elif ply < 80:
        moves_to_go = 20
    else:
        moves_to_go = 14

    soft = usable // moves_to_go
    soft = min(soft, usable // 8)
    soft = max(soft, 20)
    hard = min(int(soft * 1.45), usable // 6, usable)
    hard = max(hard, soft)
    return soft, hard


_agent = _Agent()


def _get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        raise RuntimeError("asked to move in a finished position")

    key = _key_of(board)
    _agent.sync(board, key)

    if len(legal) == 1:
        _agent.note_our_move(board, legal[0])
        return legal[0].uci()

    if time_left_ms >= 80:
        opening = book.probe(board)
        if opening is not None and opening in legal:
            _log(f"book {opening.uci()}")
            _agent.note_our_move(board, opening)
            return opening.uci()

    move = _agent.choose(board, time_left_ms)
    if move not in legal:
        # A bitboard bug must never cost the game; python-chess is the authority.
        _log(f"REJECTED illegal {move.uci()} in {fen}")
        move = legal[0]
    _agent.note_our_move(board, move)
    return move.uci()


def get_move(fen: str, time_left_ms: int) -> str:
    """Required API. Always returns a legal UCI move; never raises to the runner."""
    try:
        return _get_move(fen, int(time_left_ms))
    except Exception:
        if DEBUG:
            traceback.print_exc()
        try:
            return next(iter(chess.Board(fen).legal_moves)).uci()
        except Exception:
            return "0000"


def _warmup() -> None:
    """Pay the JIT and table-build cost inside the 60s init budget.

    Warming through `bb_search.search` compiles the whole call graph with exactly
    the signatures the game will use, which is cheaper than compiling each
    function separately.
    """
    _ = len(book.BOOK)
    # Weights must be in the numpy buffers before the first `evaluate` compile;
    # numba captures those arrays at JIT time.
    nnue.load()
    bb_eval.load_residual()
    board = chess.Board()
    bbs, st, mb = from_board(board)
    acc, acc_stack = nnue.new_acc()
    nnue.acc_refresh(acc, bbs)
    packed = np.int64(move_from_chess(chess.Move.from_uci("e2e4"), board))
    nnue.acc_push(acc, acc_stack, np.int64(0), mb, st, packed)
    bb.make_move(bbs, st, mb, _agent.hist, np.int64(0), packed)
    incremental = acc.copy()
    rebuilt = np.zeros_like(acc)
    nnue.acc_refresh(rebuilt, bbs)
    if not np.allclose(incremental, rebuilt, atol=1e-4):
        print("WARN acc incremental mismatch on e2e4", flush=True)
    bb.unmake_move(bbs, st, mb, _agent.hist, np.int64(0), packed)
    nnue.acc_pop(acc, acc_stack, np.int64(0))
    _agent.rep[0] = st[bb.ST_HASH]
    _agent.info[bb_search.I_REP_BASE] = 0
    deadline = time.perf_counter() + 60.0
    bb_search.search(
        bbs,
        st,
        mb,
        _agent.hist,
        _agent.moves,
        _agent.scores,
        _agent.tt,
        _agent.ord32,
        _agent.pc,
        _agent.rep,
        _agent.info,
        np.int64(6),
        deadline,
        deadline,
    )
    _agent.tt[:, :] = 0
    _agent.ord32[:] = 0
    _agent.after_ours = None
    _agent.keys = []
    _agent.ply = 0


_warmup()

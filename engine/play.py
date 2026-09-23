"""Engine loop: wide book, numba PVS, classical + NNUE mix. Python Searcher is unused.

See ROADMAP.md. get_move runs `bb_search.search`, not engine.search.Searcher.
"""

from __future__ import annotations

import os
import time

import chess
import numpy as np

from engine import bb
from engine import bb_eval
from engine import bb_search
from engine import book
from engine.bbpos import from_board, move_to_uci
from engine.clock import allocate

DEBUG = os.environ.get("CHESS_DEBUG") == "1"
MAX_DEPTH = 63


def _log(message: str) -> None:
    if DEBUG:
        print(message, flush=True)


def _key_of(board: chess.Board) -> int:
    _, st, _ = from_board(board)
    return int(st[bb.ST_HASH])


class Engine:
    def __init__(self) -> None:
        bb.init()
        self.tt, self.ord32, self.rep, self.info = bb_search.new_tables()
        self.pc = bb_eval.new_pawn_cache()
        self.hist, self.moves, self.scores = bb.new_buffers()
        self.keys: list[int] = []
        self.after_ours: chess.Board | None = None
        self.ply = 0

    def start_game(self, board: chess.Board, key: int) -> None:
        self.keys = [key]
        self.after_ours = None
        self.ply = 0
        self.ord32[:] = 0
        self.tt[:, :] = 0

    def sync(self, board: chess.Board, key: int) -> None:
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

    def warmup(self) -> None:
        _ = len(book.BOOK)
        board = chess.Board()
        bbs, st, mb = from_board(board)
        self.rep[0] = st[bb.ST_HASH]
        self.info[bb_search.I_REP_BASE] = 0
        deadline = time.perf_counter() + 60.0
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
            np.int64(6),
            deadline,
            deadline,
        )
        self.tt[:, :] = 0
        self.ord32[:] = 0
        self.after_ours = None
        self.keys = []
        self.ply = 0

    def choose(self, board: chess.Board, time_left_ms: int) -> chess.Move:
        bbs, st, mb = from_board(board)
        history = self.keys[-(len(self.rep) - bb.MAX_PLY - 8) :]
        for i, key in enumerate(history):
            self.rep[i] = key
        self.info[bb_search.I_REP_BASE] = len(history) - 1

        soft_ms, hard_ms = allocate(time_left_ms, self.ply)
        now = time.perf_counter()
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
                now + soft_ms / 1000.0,
                now + hard_ms / 1000.0,
            )
        )
        if packed == 0:
            raise ValueError("search returned no move")
        move = chess.Move.from_uci(move_to_uci(packed))
        _log(
            f"move {move.uci()} depth {self.info[bb_search.I_DEPTH_DONE]} "
            f"score {self.info[bb_search.I_ROOT_SCORE]} nodes {self.info[bb_search.I_NODES]}"
        )
        return move


def pick_move(fen: str, time_left_ms: int, engine: Engine) -> str:
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        raise RuntimeError("asked to move in a finished position")
    key = _key_of(board)
    engine.sync(board, key)
    if len(legal) == 1:
        engine.note_our_move(board, legal[0])
        return legal[0].uci()
    if time_left_ms >= 80:
        opening = book.probe(board)
        if opening is not None and opening in legal:
            engine.note_our_move(board, opening)
            return opening.uci()
    move = engine.choose(board, time_left_ms)
    if move not in legal:
        _log(f"REJECTED illegal {move.uci()} in {fen}")
        move = legal[0]
    engine.note_our_move(board, move)
    return move.uci()

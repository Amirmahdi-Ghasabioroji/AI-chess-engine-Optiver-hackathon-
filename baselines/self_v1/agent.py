"""Competition entry. The judge imports this module and calls get_move."""

from __future__ import annotations

import os
import threading
import traceback

import chess
import chess.polyglot

import book
from search import Engine, SearchStopped

DEBUG = os.environ.get("CHESS_DEBUG") == "1"

SAFETY_MS = 90
PONDER_JOIN_S = 0.08

_engine = Engine()
_lock = threading.Lock()

# Game-lifetime state (one process = one game, but we still detect a reset).
_keys: list[int] = []
_after_ours: chess.Board | None = None
_ply = 0

_ponder_thread: threading.Thread | None = None
_ponder_fen: str | None = None


def _log(msg: str) -> None:
    if DEBUG:
        print(msg, flush=True)


def _start_game(board: chess.Board) -> None:
    global _keys, _after_ours, _ply
    _engine.new_game()
    _keys = [chess.polyglot.zobrist_hash(board)]
    _after_ours = None
    _ply = 0


def _sync(fen: str) -> chess.Board:
    """Attach this request to the ongoing game, or start a new one."""
    global _keys, _after_ours, _ply
    board = chess.Board(fen)
    key = chess.polyglot.zobrist_hash(board)

    if _after_ours is None:
        _start_game(board)
        return board

    probe = _after_ours.copy(stack=False)
    found = False
    for move in probe.legal_moves:
        probe.push(move)
        if chess.polyglot.zobrist_hash(probe) == key:
            found = True
            break
        probe.pop()

    if not found:
        _start_game(board)
        return board

    _keys.append(key)
    _ply += 1
    return board


def _note_our_move(board: chess.Board, move: chess.Move) -> None:
    global _after_ours, _keys, _ply
    nxt = board.copy(stack=False)
    nxt.push(move)
    _keys.append(chess.polyglot.zobrist_hash(nxt))
    _after_ours = nxt
    _ply += 1


def _stop_ponder() -> None:
    global _ponder_thread, _ponder_fen
    _engine.stop = True
    t = _ponder_thread
    if t is not None and t.is_alive():
        t.join(timeout=PONDER_JOIN_S)
    _ponder_thread = None
    _ponder_fen = None
    _engine.stop = False


def _start_ponder(board: chess.Board, our_move: chess.Move, rep_keys: list[int]) -> None:
    global _ponder_thread, _ponder_fen
    pred = _engine.extract_ponder(board, our_move)
    if pred is None:
        return
    ponder_board = board.copy(stack=False)
    ponder_board.push(our_move)
    ponder_board.push(pred)
    _ponder_fen = ponder_board.fen()
    keys = list(rep_keys)
    keys.append(chess.polyglot.zobrist_hash(ponder_board))

    def _run() -> None:
        try:
            _engine.search(
                ponder_board,
                time_ms=600_000,
                rep_keys=keys,
                max_depth=64,
            )
        except (SearchStopped, Exception):
            pass

    _engine.stop = False
    _ponder_thread = threading.Thread(target=_run, name="ponder", daemon=True)
    _ponder_thread.start()


def _allocate(time_left_ms: int, ply: int, n_legal: int) -> tuple[int, int]:
    """Return (soft_ms, hard_ms). Increment is not in time_left_ms.

    Do not bank the platform's 0.5 s increment: the API does not report it,
    and the starter arena uses 0.1 s. Treat increment as reserve, not spend.
    """
    usable = max(1, time_left_ms - SAFETY_MS)
    if n_legal <= 1:
        return 10, min(25, usable)
    if usable <= 90:
        return 12, min(35, usable)
    if usable <= 700:
        hard = min(70, usable)
        return min(35, hard), hard

    if ply < 16:
        mtg = 36
    elif ply < 40:
        mtg = 28
    elif ply < 80:
        mtg = 20
    else:
        mtg = 14

    soft = usable // mtg
    soft = min(soft, usable // 8)
    soft = max(soft, 20)
    hard = min(int(soft * 1.45), usable // 6, usable)
    hard = max(hard, soft)
    return soft, hard


def _first_legal(board: chess.Board) -> str:
    return next(iter(board.legal_moves)).uci()


def _get_move(fen: str, time_left_ms: int) -> str:
    _stop_ponder()
    board = _sync(fen)
    legal = list(board.legal_moves)
    if not legal:
        # Should not happen on a live game; returning anything illegal loses.
        raise RuntimeError("no legal moves")
    if len(legal) == 1:
        move = legal[0]
        _note_our_move(board, move)
        return move.uci()

    if time_left_ms >= 80:
        book_move = book.probe(board)
        if book_move is not None and book_move in legal:
            _log(f"book {book_move.uci()}")
            _note_our_move(board, book_move)
            if time_left_ms > 8000:
                _start_ponder(board, book_move, _keys)
            return book_move.uci()

    soft, hard = _allocate(time_left_ms, _ply, len(legal))
    # If ponder already filled this position, ID still runs; TT makes it cheap.
    max_depth = 64
    if time_left_ms < 400:
        max_depth = 3
    elif time_left_ms < 1500:
        max_depth = 6

    move = _engine.search(
        board,
        time_ms=hard,
        rep_keys=_keys,
        max_depth=max_depth,
        soft_ms=soft,
    )
    if move not in board.legal_moves:
        move = legal[0]

    _log(
        f"move {move.uci()} depth~{_engine.seldepth} nodes={_engine.nodes} "
        f"score={_engine.root_score} soft={soft} hard={hard}"
    )
    _note_our_move(board, move)
    if time_left_ms > 8000:
        _start_ponder(board, move, _keys)
    return move.uci()


def get_move(fen: str, time_left_ms: int) -> str:
    """Required API. Always returns a UCI string; never raises to the runner."""
    try:
        with _lock:
            return _get_move(fen, int(time_left_ms))
    except Exception:
        if DEBUG:
            traceback.print_exc()
        try:
            return _first_legal(chess.Board(fen))
        except Exception:
            return "0000"


def _warmup() -> None:
    """Land JIT / TT / book build on the 60 s init clock, not the game clock."""
    _ = len(book.BOOK)
    b = chess.Board()
    try:
        _engine.search(b, time_ms=300, max_depth=4, soft_ms=250)
    except Exception:
        pass
    _engine.new_game()


_warmup()

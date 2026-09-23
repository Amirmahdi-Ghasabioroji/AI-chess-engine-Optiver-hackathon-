"""Short local games: our agent vs a legal-random opponent."""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess

import agent


def random_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    return random.choice(list(board.legal_moves)).uci()


def play(white, black, time_ms: int = 1500, max_plies: int = 120) -> chess.Board:
    board = chess.Board()
    clocks = {chess.WHITE: time_ms, chess.BLACK: time_ms}
    while not board.is_game_over() and board.ply() < max_plies:
        side = board.turn
        fn = white if side == chess.WHITE else black
        t0 = time.perf_counter()
        uci = fn(board.fen(), clocks[side])
        used = int((time.perf_counter() - t0) * 1000)
        clocks[side] = max(0, clocks[side] - used + 100)
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise SystemExit(f"illegal {uci} at {board.fen()}")
        board.push(move)
    return board


def main() -> int:
    random.seed(0)
    wins = draws = losses = 0
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    time_ms = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    for i in range(games):
        we_white = i % 2 == 0
        white = agent.get_move if we_white else random_move
        black = random_move if we_white else agent.get_move
        # Reset agent game tracker between games
        agent._after_ours = None
        board = play(white, black, time_ms=time_ms)
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            draws += 1
            result = "draw"
        elif (outcome.winner == chess.WHITE) == we_white:
            wins += 1
            result = "win"
        else:
            losses += 1
            result = "loss"
        print(f"game {i + 1}: {result}  plies={board.ply()}  {board.result(claim_draw=True)}")
    print(f"W-D-L {wins}-{draws}-{losses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

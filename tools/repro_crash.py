"""Drive get_move in-process over whole games to hunt a process-killing bug.

The gauntlet scored several games as `crash`, which in the harness means the agent
process died rather than returned a bad move. agent.py catches Python exceptions and
falls back to a legal move, so a crash points at something below Python: an
out-of-bounds write in a jitted function, or a hard abort.

Running the game here, in one process, makes that failure land in the open with a
traceback instead of a dead pipe. A segfault still kills this process, but it kills it
at a known ply with a known position printed.

    python -m tools.repro_crash --openings 8 --move-ms 300
"""

from __future__ import annotations

import argparse
import sys
import traceback

import chess

import agent
from tools.gauntlet import opening_positions

MAX_PLIES = 300


def play_self(fen: str, budget_ms: int) -> str:
    """Agent against agent from `fen`. Returns a one word verdict."""
    board = chess.Board(fen)
    clock = {chess.WHITE: budget_ms, chess.BLACK: budget_ms}
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < MAX_PLIES:
        mover = board.turn
        uci = agent.get_move(board.fen(), clock[mover])
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            print(f"  ILLEGAL {uci} at ply {len(board.move_stack)} in {board.fen()}")
            return "illegal"
        board.push(move)
        clock[mover] = budget_ms
        print(f"  ply {len(board.move_stack)}: {uci}", flush=True)
    outcome = board.outcome(claim_draw=True)
    return outcome.termination.name.lower() if outcome else "ply_cap"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openings", type=int, default=8)
    parser.add_argument("--budget-ms", type=int, default=3000, help="time_left_ms per call")
    arguments = parser.parse_args()

    positions = opening_positions()[: arguments.openings]
    failures = 0
    for index, fen in enumerate(positions):
        print(f"\nopening {index}: {fen}", flush=True)
        try:
            verdict = play_self(fen, arguments.budget_ms)
        except Exception:
            traceback.print_exc()
            failures += 1
            continue
        print(f"opening {index} -> {verdict}", flush=True)
        if verdict == "illegal":
            failures += 1

    print(f"\n{len(positions)} openings, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

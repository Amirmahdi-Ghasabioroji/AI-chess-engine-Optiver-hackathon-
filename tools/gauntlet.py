"""Score an agent over varied openings, colour-paired, with error bars.

`harness.arena` always starts from the standard position, and both agents share the
same deterministic book, so a 40 game run there is really two games replayed twenty
times. Rated games start from curated openings instead, so this driver plays a set of
distinct opening positions, each one twice with the colours swapped, and reports the
score with a standard error taken over opening pairs rather than over games.

    python -m tools.gauntlet --opponent baselines/self_v1 --openings 30
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import chess

from engine.book import _LINES
from harness.referee import FAILED_TERMINATIONS, play_match
from harness.rules import PLY_CAP
from harness.sandbox import local

MIN_OPENING_PLIES = 8
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def opening_positions() -> list[str]:
    """End positions of the book mainlines: distinct, material-equal, out of book.

    A line's final position is the one position in it that the book never stored a
    reply for, so games from here are decided by search rather than by shared theory.
    """
    seen: dict[str, None] = {}
    for line in _LINES:
        board = chess.Board()
        for token in line.split():
            move = chess.Move.from_uci(token)
            if move not in board.legal_moves:
                break
            board.push(move)
        if len(board.move_stack) < MIN_OPENING_PLIES or board.is_game_over():
            continue
        if _material_balance(board) != 0:
            continue
        seen.setdefault(board.fen(), None)
    return list(seen)


def _material_balance(board: chess.Board) -> int:
    return sum(
        value * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
        for piece, value in PIECE_VALUES.items()
    )


@dataclass
class GameResult:
    opening: int
    agent_white: bool
    score: float
    termination: str
    fen: str = ""
    stderr: str = ""
    pgn: str = ""


def play_one(
    agent: Path, opponent: Path, fen: str, opening: int, agent_white: bool, clock: tuple[int, int]
) -> GameResult:
    ours = local(agent)
    theirs = local(opponent)
    white, black = (ours, theirs) if agent_white else (theirs, ours)
    try:
        outcome = play_match(white, black, clock[0], clock[1], PLY_CAP, fen)
    except Exception:
        # A driver-side failure is not an agent failure, but losing the whole run to one
        # is worse than scoring it as a void game and saying so.
        traceback.print_exc()
        return GameResult(opening, agent_white, 0.5, "driver_error", fen)
    if outcome.result in ("draw", "void"):
        score = 0.5
    elif (outcome.result == "white") == agent_white:
        score = 1.0
    else:
        score = 0.0
    # Only a failed game needs the evidence; a normal loss does not.
    if outcome.termination in FAILED_TERMINATIONS:
        return GameResult(
            opening, agent_white, score, outcome.termination, fen, ours.stderr_tail[-4000:],
            outcome.pgn,
        )
    return GameResult(opening, agent_white, score, outcome.termination, fen)


def elo(score: float) -> float:
    if score <= 0.0:
        return -math.inf
    if score >= 1.0:
        return math.inf
    return -400.0 * math.log10(1.0 / score - 1.0)


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _format_elo(value: float) -> str:
    if value == math.inf:
        return "+inf"
    if value == -math.inf:
        return "-inf"
    return f"{value:+.0f}"


def report(results: list[GameResult], pairs: int, label: str) -> int:
    wins = sum(1 for r in results if r.score == 1.0)
    draws = sum(1 for r in results if r.score == 0.5)
    losses = sum(1 for r in results if r.score == 0.0)
    games = len(results)
    score = (wins + draws / 2) / games

    # Each opening is played twice with colours swapped. The pair is the independent
    # unit, so the error bar comes from the spread across pairs, not across games.
    # Averaging rather than summing keeps a half-finished pair on the same scale.
    by_opening: dict[int, list[float]] = {}
    for r in results:
        by_opening.setdefault(r.opening, []).append(r.score)
    values = [sum(scores) / len(scores) for scores in by_opening.values()]
    mean = sum(values) / len(values)
    if len(values) > 1:
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        stderr = math.sqrt(variance / len(values))
    else:
        stderr = float("nan")

    terminations: dict[str, int] = {}
    for r in results:
        terminations[r.termination] = terminations.get(r.termination, 0) + 1

    print(f"\n{label} over {games} games from {pairs} openings, both colours")
    print(f"+{wins} ={draws} -{losses}, score {score:.1%}")
    if stderr > 0.0:
        low, high = mean - 1.96 * stderr, mean + 1.96 * stderr
        print(f"95% CI {low:.1%} .. {high:.1%}  (pair stderr {stderr:.1%})")
        print(f"elo {_format_elo(elo(score))} [{_format_elo(elo(low))}, {_format_elo(elo(high))}]")
        print(f"likelihood of superiority {_phi((mean - 0.5) / stderr):.1%}")
    else:
        print(f"elo {_format_elo(elo(score))}  (every opening pair scored the same)")
    print("terminations: " + ", ".join(f"{k} {v}" for k, v in sorted(terminations.items())))

    broken = {k: v for k, v in terminations.items() if k in FAILED_TERMINATIONS}
    if broken:
        print("FAIL: " + ", ".join(f"{k} {v}" for k, v in broken.items()))
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, default=Path("baselines/self_v1"))
    parser.add_argument("--openings", type=int, default=30, help="positions, each played twice")
    parser.add_argument("--base-ms", type=int, default=10_000)
    parser.add_argument("--increment-ms", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4, help="games in flight at once")
    parser.add_argument("--games", type=int, default=0, help="cap on games; 0 plays every colour pair")
    parser.add_argument("--log", type=str, default="logs/gauntlet.jsonl", help="per-game results")
    arguments = parser.parse_args()

    agent = arguments.agent.resolve()
    opponent = arguments.opponent.resolve()
    positions = opening_positions()[: arguments.openings]
    clock = (arguments.base_ms, arguments.increment_ms)
    if not positions:
        print("no opening positions available")
        return 1

    jobs = [
        (index, fen, agent_white)
        for index, fen in enumerate(positions)
        for agent_white in (True, False)
    ]
    if arguments.games > 0:
        jobs = jobs[: arguments.games]
    print(
        f"{len(jobs)} games from {len(positions)} openings, "
        f"{arguments.base_ms}ms+{arguments.increment_ms}ms, {arguments.workers} in flight"
    )

    results: list[GameResult] = []
    log = Path(arguments.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("w", encoding="utf-8") as stream, ThreadPoolExecutor(
            max_workers=arguments.workers
        ) as pool:
            futures = [
                pool.submit(play_one, agent, opponent, fen, index, agent_white, clock)
                for index, fen, agent_white in jobs
            ]
            for done, future in enumerate(futures, start=1):
                result = future.result()
                results.append(result)
                stream.write(json.dumps(asdict(result)) + "\n")
                stream.flush()
                colour = "white" if result.agent_white else "black"
                print(
                    f"game {done}/{len(jobs)}: opening {result.opening} as {colour} "
                    f"-> {result.score} by {result.termination}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\ninterrupted, reporting what finished")

    if not results:
        print("no games finished")
        return 1
    return report(results, len(positions), f"{arguments.agent} vs {arguments.opponent}")


if __name__ == "__main__":
    sys.exit(main())

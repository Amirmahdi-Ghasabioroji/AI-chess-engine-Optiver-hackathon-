"""Harness measurement: arenas vs minimax/numba plus out-of-book start FENs.

Writes JSONL so a run can be resumed or summarised later.
Does not edit harness/.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess
import chess.pgn

from engine import book
from harness.referee import FAILED_TERMINATIONS, play_match
from harness.sandbox import local
AGENT = ROOT
RESULTS = ROOT / "logs" / "measure_results.jsonl"

# Arena time control from the starter (honest, not full 120 s).
BASE_MS = 10_000
INCREMENT_MS = 100

# Balanced-ish tabiyas that rated games could resemble. Filtered at runtime
# so we only keep positions our book does not answer.
CANDIDATE_FENS: tuple[tuple[str, str], ...] = (
    (
        "london_vs_kings_indian",
        "rnbqkb1r/pppppp1p/5np1/8/3P1B2/5N2/PPP1PPPP/RN1QKB1R b KQkq - 1 3",
    ),
    (
        "queens_indian_ba6",
        "rn1qkb1r/p1pp1ppp/bp2pn2/8/2PP4/5NP1/PP2PP1P/RNBQKB1R w KQkq - 1 5",
    ),
    (
        "closed_catalan",
        "rnbq1rk1/pp2ppbp/2p2np1/3p4/2PP4/5NP1/PP2PPBP/RNBQ1RK1 w - - 4 8",
    ),
    (
        "sicilian_taimanov_qc7",
        "r1b1kb1r/ppq2ppp/2nppn2/8/3NP3/2N1B3/PPP2PPP/R2QKB1R w KQkq - 4 8",
    ),
    (
        "kings_indian_fianchetto",
        "rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP3PPP/R1BQKB1R w KQ - 2 7",
    ),
    (
        "french_advance",
        "rnbqk2r/pp2bppp/2p1pn2/3pP3/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 3 7",
    ),
    (
        "ruy_closed_h3",
        "r1bq1rk1/2ppbppp/p1n2n2/1p2p3/4P3/1BP2N1P/PP1P1PP1/RNBQR1K1 b - - 0 9",
    ),
    (
        "slav_exchange",
        "rnbqkb1r/pp3ppp/2p1pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 6",
    ),
)


def _plies(pgn: str) -> int:
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return 0
    return game.end().ply()


def _out_of_book() -> list[tuple[str, str]]:
    kept: list[tuple[str, str]] = []
    for name, fen in CANDIDATE_FENS:
        board = chess.Board(fen)
        hit = book.probe(board)
        if hit is None:
            kept.append((name, fen))
        else:
            print(f"skip {name}: book hit {hit.uci()}")
    return kept


def _play(
    opponent: Path,
    we_white: bool,
    fen: str,
    tag: str,
) -> dict:
    white, black = (AGENT, opponent) if we_white else (opponent, AGENT)
    t0 = time.perf_counter()
    outcome = play_match(
        local(white),
        local(black),
        BASE_MS,
        INCREMENT_MS,
        start_fen=fen,
    )
    elapsed = time.perf_counter() - t0
    if outcome.result in ("draw", "void"):
        ours = "draw"
    elif (outcome.result == "white") == we_white:
        ours = "win"
    else:
        ours = "loss"
    row = {
        "tag": tag,
        "opponent": str(opponent.relative_to(ROOT)).replace("\\", "/"),
        "we_white": we_white,
        "fen": fen,
        "ours": ours,
        "result": outcome.result,
        "termination": outcome.termination,
        "plies": _plies(outcome.pgn),
        "seconds": round(elapsed, 2),
        "failed": outcome.termination in FAILED_TERMINATIONS,
    }
    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def _arena(opponent: Path, games: int, tag: str) -> None:
    for i in range(games):
        we_white = i % 2 == 0
        row = _play(opponent, we_white, chess.STARTING_FEN, tag)
        colour = "white" if we_white else "black"
        print(
            f"{tag} {i + 1}/{games} as {colour}: {row['ours']} "
            f"by {row['termination']} plies={row['plies']} {row['seconds']:.1f}s",
            flush=True,
        )


def _summarise(tag: str | None = None) -> None:
    if not RESULTS.exists():
        return
    rows = [json.loads(line) for line in RESULTS.read_text(encoding="utf-8").splitlines() if line]
    if tag:
        rows = [r for r in rows if r["tag"] == tag]
    if not rows:
        return
    n = len(rows)
    wins = sum(r["ours"] == "win" for r in rows)
    draws = sum(r["ours"] == "draw" for r in rows)
    losses = sum(r["ours"] == "loss" for r in rows)
    fails = sum(r["failed"] for r in rows)
    score = (wins + draws / 2) / n
    terms: dict[str, int] = {}
    for r in rows:
        terms[r["termination"]] = terms.get(r["termination"], 0) + 1
    print(
        f"\n{tag or 'all'}: +{wins} ={draws} -{losses}  score {score:.1%}  "
        f"fail {fails}/{n}  avg_plies {sum(r['plies'] for r in rows) / n:.1f}  "
        f"terms {terms}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimax-games", type=int, default=20)
    parser.add_argument("--numba-games", type=int, default=20)
    parser.add_argument("--fen-games", type=int, default=2, help="games per out-of-book FEN")
    parser.add_argument("--skip-startpos", action="store_true")
    parser.add_argument("--skip-fens", action="store_true")
    args = parser.parse_args()

    RESULTS.write_text("", encoding="utf-8")

    if not args.skip_startpos:
        print("=== startpos vs minimax ===", flush=True)
        _arena(ROOT / "baselines" / "minimax", args.minimax_games, "minimax_start")
        _summarise("minimax_start")

        print("=== startpos vs numba ===", flush=True)
        _arena(ROOT / "baselines" / "numba", args.numba_games, "numba_start")
        _summarise("numba_start")

    if not args.skip_fens:
        fens = _out_of_book()
        print(f"=== {len(fens)} out-of-book FENs vs minimax ({args.fen_games} games each) ===", flush=True)
        for name, fen in fens:
            for i in range(args.fen_games):
                we_white = i % 2 == 0
                row = _play(ROOT / "baselines" / "minimax", we_white, fen, f"fen_{name}")
                colour = "white" if we_white else "black"
                print(
                    f"{name} as {colour}: {row['ours']} by {row['termination']} "
                    f"plies={row['plies']} {row['seconds']:.1f}s",
                    flush=True,
                )
        print("=== out-of-book FEN summary ===", flush=True)
        if RESULTS.exists():
            rows = [
                json.loads(line)
                for line in RESULTS.read_text(encoding="utf-8").splitlines()
                if line
            ]
            fen_rows = [r for r in rows if r["tag"].startswith("fen_")]
            n = len(fen_rows)
            if n:
                wins = sum(r["ours"] == "win" for r in fen_rows)
                draws = sum(r["ours"] == "draw" for r in fen_rows)
                losses = sum(r["ours"] == "loss" for r in fen_rows)
                print(
                    f"fen_all: +{wins} ={draws} -{losses}  score {(wins + draws / 2) / n:.1%}  "
                    f"n={n}",
                    flush=True,
                )
        _summarise()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Stream the Lichess puzzle dump into a tactical test suite (not NNUE labels).

Hanging-piece and endgame puzzles are for measuring search, not for training the
mix net. FEN in the dump is before the opponent's blunder; the suite position is
after the first UCI move.

    python -m training.import_lichess_puzzles --max 4000
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.request
from pathlib import Path

import chess
import zstandard as zstd

DEFAULT_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
DEFAULT_OUT = Path("data/lichess/tactics_suite.jsonl")
KEEP_THEMES = (
    "hangingPiece",
    "fork",
    "endgame",
    "attraction",
    "discoveredAttack",
    "skewer",
    "pin",
    "mate",
    "mateIn1",
    "mateIn2",
    "mateIn3",
    "kingsideAttack",
    "exposedKing",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--src", type=Path, default=None, help="local .csv.zst if already downloaded")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max", type=int, default=4000)
    parser.add_argument("--min-popularity", type=int, default=80)
    parser.add_argument("--min-plays", type=int, default=200)
    parser.add_argument("--max-rating", type=int, default=1800)
    arguments = parser.parse_args()

    if arguments.src is not None and arguments.src.is_file():
        raw = arguments.src.read_bytes()
    else:
        print(f"downloading {arguments.url}", flush=True)
        with urllib.request.urlopen(arguments.url, timeout=120) as resp:
            raw = resp.read()
        cache = Path("data/lichess/lichess_db_puzzle.csv.zst")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(raw)
        print(f"cached {cache} ({len(raw)} bytes)", flush=True)

    dctx = zstd.ZstdDecompressor()
    text = io.TextIOWrapper(dctx.stream_reader(io.BytesIO(raw)), encoding="utf-8")
    reader = csv.DictReader(text)
    kept = 0
    scanned = 0
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with arguments.out.open("w", encoding="utf-8") as out:
        for row in reader:
            scanned += 1
            themes = (row.get("Themes") or "").split()
            if not any(theme in KEEP_THEMES for theme in themes):
                continue
            try:
                popularity = int(row.get("Popularity") or 0)
                plays = int(row.get("NbPlays") or 0)
                rating = int(row.get("Rating") or 0)
            except ValueError:
                continue
            if popularity < arguments.min_popularity or plays < arguments.min_plays:
                continue
            if rating > arguments.max_rating:
                continue
            moves = (row.get("Moves") or "").split()
            fen = row.get("FEN") or ""
            if len(moves) < 2 or not fen:
                continue
            try:
                board = chess.Board(fen)
                first = chess.Move.from_uci(moves[0])
                if first not in board.legal_moves:
                    continue
                board.push(first)
            except ValueError:
                continue
            rec = {
                "id": row.get("PuzzleId"),
                "fen": board.fen(),
                "best": moves[1],
                "themes": themes,
                "rating": rating,
            }
            out.write(json.dumps(rec) + "\n")
            kept += 1
            if kept % 500 == 0:
                print(f"  kept {kept} scanned {scanned}", flush=True)
            if kept >= arguments.max:
                break
    print(f"wrote {arguments.out} kept={kept} scanned={scanned}", flush=True)
    return 0 if kept >= 50 else 1


if __name__ == "__main__":
    raise SystemExit(main())

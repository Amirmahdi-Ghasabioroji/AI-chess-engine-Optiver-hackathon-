"""Build NNUE labels from the Kaggle Lichess games dump (datasnaek/chess).

Local only: kagglehub is not on the judge. Games have SAN, not evals, so
Stockfish still labels quiet positions. Replaying PGN is cheap; we no longer
search with our engine to generate the boards.

    python -m training.import_kaggle_chess --count 40000 --out data/nnue_train.npz
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import chess
import chess.engine
import chess.polyglot
import kagglehub
import numpy as np

from engine import bb
from engine import bb_eval
from engine import nnue
from engine.bbpos import from_board
from training.gen_nnue import find_stockfish, pack_features, stockfish_cp

DATASET = "datasnaek/chess"


def pesto_of(board: chess.Board, pc: np.ndarray) -> int:
    bbs, st, _ = from_board(board)
    return int(bb_eval.evaluate_classical(bbs, st, pc))


def collect_quiet_fens(csv_path: Path, seed: int) -> list[str]:
    rng = random.Random(seed)
    seen: set[int] = set()
    fens: list[str] = []
    games = 0
    skipped = 0
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            games += 1
            try:
                turns = int(row.get("turns") or 0)
            except ValueError:
                skipped += 1
                continue
            if turns < 20:
                skipped += 1
                continue
            try:
                white_elo = int(row.get("white_rating") or 0)
                black_elo = int(row.get("black_rating") or 0)
            except ValueError:
                white_elo = black_elo = 0
            if min(white_elo, black_elo) < 1400:
                skipped += 1
                continue
            try:
                opening_ply = int(row.get("opening_ply") or 8)
            except ValueError:
                opening_ply = 8
            skip_head = max(8, opening_ply)
            board = chess.Board()
            taken = 0
            try:
                sans = row["moves"].split()
            except KeyError:
                skipped += 1
                continue
            for ply, san in enumerate(sans):
                try:
                    board.push_san(san)
                except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError):
                    break
                if ply < skip_head:
                    continue
                if ply >= len(sans) - 2:
                    break
                if taken >= 8:
                    break
                if board.is_check() or board.is_game_over(claim_draw=True):
                    continue
                if rng.random() > 0.35:
                    continue
                key = chess.polyglot.zobrist_hash(board)
                if key in seen:
                    continue
                seen.add(key)
                fens.append(board.fen())
                taken += 1
    rng.shuffle(fens)
    print(f"replayed {games} games, skipped {skipped}, unique quiet {len(fens)}", flush=True)
    return fens


def dump(
    path: Path,
    features: list[np.ndarray],
    pesto: list[int],
    sf_scores: list[int],
    stm: list[int],
    fens: list[str],
    sf_depth: int,
) -> None:
    if not features:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    sf_arr = np.array(sf_scores, dtype=np.int32)
    pesto_arr = np.array(pesto, dtype=np.int32)
    residual = sf_arr - pesto_arr
    np.savez_compressed(
        path,
        features=np.stack(features),
        pesto=pesto_arr,
        sf=sf_arr,
        residual=residual,
        stm=np.array(stm, dtype=np.int8),
        fen=np.array(fens),
        source=np.array("kaggle:datasnaek/chess"),
        sf_depth=np.int32(sf_depth),
    )
    print(
        f"wrote {path} n={len(features)} residual mean {residual.mean():.1f} std {residual.std():.1f}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=40000)
    parser.add_argument("--sf-depth", type=int, default=10)
    parser.add_argument("--sf", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("data/nnue_train.npz"))
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=[],
        help="npz files whose `fen` column should not be labelled again",
    )
    arguments = parser.parse_args()

    print("downloading datasnaek/chess via kagglehub", flush=True)
    root = Path(kagglehub.dataset_download(DATASET))
    csv_path = root / "games.csv"
    if not csv_path.is_file():
        print(f"no games.csv under {root}", file=sys.stderr)
        return 1
    print(f"Path to dataset files: {root}", flush=True)

    candidates = collect_quiet_fens(csv_path, arguments.seed)
    if not candidates:
        print("no quiet positions extracted", file=sys.stderr)
        return 1
    seen: set[str] = set()
    for extra in arguments.exclude:
        if extra.is_file():
            blob = np.load(extra)
            if "fen" in blob.files:
                seen.update(str(f) for f in blob["fen"])
    if seen:
        before = len(candidates)
        candidates = [fen for fen in candidates if fen not in seen]
        print(
            f"excluded {before - len(candidates)} already-labelled FENs, {len(candidates)} left",
            flush=True,
        )

    nnue.disable()
    bb.init()
    pc = bb_eval.new_pawn_cache()
    # Compile PeSTO once, with the residual zeroed, before any labels.
    pesto_of(chess.Board(), pc)

    sf_path = find_stockfish(arguments.sf)
    print(f"stockfish {sf_path}  sf-depth {arguments.sf_depth}  target {arguments.count}", flush=True)
    engine = chess.engine.SimpleEngine.popen_uci(str(sf_path))
    engine.configure({"Threads": arguments.threads, "Hash": 128})

    features: list[np.ndarray] = []
    pesto: list[int] = []
    sf_scores: list[int] = []
    stm: list[int] = []
    kept: list[str] = []
    started = time.perf_counter()
    last_dump = 0
    scanned = 0
    try:
        for fen in candidates:
            if len(features) >= arguments.count:
                break
            scanned += 1
            board = chess.Board(fen)
            cp = stockfish_cp(engine, board, arguments.sf_depth)
            if cp is None:
                continue
            features.append(pack_features(board))
            pesto.append(pesto_of(board, pc))
            sf_scores.append(cp)
            stm.append(1 if board.turn == chess.WHITE else -1)
            kept.append(fen)
            n = len(features)
            if n <= 3 or n % 500 == 0:
                elapsed = time.perf_counter() - started
                print(f"{n}/{arguments.count} labelled ({scanned} scanned) in {elapsed:.0f}s", flush=True)
            if n - last_dump >= 2000:
                dump(arguments.out, features, pesto, sf_scores, stm, kept, arguments.sf_depth)
                last_dump = n
    finally:
        engine.quit()

    dump(arguments.out, features, pesto, sf_scores, stm, kept, arguments.sf_depth)
    return 0 if features else 1


if __name__ == "__main__":
    sys.exit(main())

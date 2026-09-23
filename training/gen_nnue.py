"""Collect quiet positions and label them with Stockfish.

Our bitboard engine plays the games (Stockfish must not ship or play rated
moves). After each quiet position is reached, Stockfish scores it. Those
labels stay on disk under `data/` and never enter `submission.zip`.

    python -m training.gen_nnue --count 8000 --play-depth 12 --sf-depth 10 --out data/nnue_train.npz
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import chess
import chess.engine
import chess.polyglot
import numpy as np

from engine import bb
from engine import bb_eval
from engine import bb_search
from engine import nnue
from engine.bbpos import from_board, move_to_uci
from tools.gauntlet import opening_positions

MATE_BOUND = 29000
MAX_PIECES = 32
DEFAULT_SF = Path(__file__).resolve().parents[1] / "tools" / "sf"


def find_stockfish(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit
    matches = sorted(DEFAULT_SF.rglob("stockfish*.exe")) + sorted(DEFAULT_SF.rglob("stockfish*"))
    files = [p for p in matches if p.is_file() and p.suffix.lower() in {".exe", ""}]
    if not files:
        raise FileNotFoundError(
            f"no Stockfish binary under {DEFAULT_SF}; download the Windows build into tools/sf/"
        )
    return files[0]


class Teacher:
    def __init__(self) -> None:
        nnue.disable()
        bb.init()
        self.tt, self.ord32, self.rep, self.info = bb_search.new_tables()
        self.pc = bb_eval.new_pawn_cache()
        self.hist, self.moves, self.scores = bb.new_buffers()

    def search(self, board: chess.Board, depth: int, budget_s: float) -> tuple[str, int]:
        bbs, st, mb = from_board(board)
        self.rep[0] = st[bb.ST_HASH]
        self.info[bb_search.I_REP_BASE] = 0
        deadline = time.perf_counter() + budget_s
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
                np.int64(depth),
                deadline,
                deadline,
            )
        )
        if packed == 0:
            return "", 0
        return move_to_uci(packed), int(self.info[bb_search.I_ROOT_SCORE])

    def pesto(self, board: chess.Board) -> int:
        bbs, st, _ = from_board(board)
        return int(bb_eval.evaluate_classical(bbs, st, self.pc))


def pack_features(board: chess.Board) -> np.ndarray:
    feats = np.full(MAX_PIECES, -1, dtype=np.int32)
    i = 0
    for square, piece in board.piece_map().items():
        code = (0 if piece.color == chess.WHITE else 6) + (piece.piece_type - 1)
        feats[i] = code * 64 + square
        i += 1
        if i >= MAX_PIECES:
            break
    return feats


def stockfish_cp(engine: chess.engine.SimpleEngine, board: chess.Board, depth: int) -> int | None:
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    pov = info["score"].pov(board.turn)
    if pov.is_mate():
        return None
    score = pov.score()
    if score is None:
        return None
    return int(np.clip(score, -2000, 2000))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=8000)
    parser.add_argument("--play-depth", type=int, default=12)
    parser.add_argument("--sf-depth", type=int, default=10)
    parser.add_argument(
        "--play-budget",
        type=float,
        default=20.0,
        help="seconds per move so iterative deepening can actually reach --play-depth",
    )
    parser.add_argument("--sf", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-ply", type=int, default=80)
    parser.add_argument("--noise", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("data/nnue_train.npz"))
    arguments = parser.parse_args()

    sf_path = find_stockfish(arguments.sf)
    print(
        f"stockfish {sf_path}  play-depth {arguments.play_depth}  "
        f"sf-depth {arguments.sf_depth}  budget {arguments.play_budget:.0f}s",
        flush=True,
    )
    engine = chess.engine.SimpleEngine.popen_uci(str(sf_path))
    engine.configure({"Threads": arguments.threads, "Hash": 128})

    random.seed(arguments.seed)
    teacher = Teacher()
    openings = opening_positions() or [chess.STARTING_FEN]

    features: list[np.ndarray] = []
    pesto: list[int] = []
    sf_scores: list[int] = []
    stm: list[int] = []
    fens: list[str] = []
    seen: set[int] = set()
    games = 0
    started = time.perf_counter()
    last_dump = 0

    def dump(path: Path) -> None:
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
            play_depth=np.int32(arguments.play_depth),
            sf_depth=np.int32(arguments.sf_depth),
        )
        print(
            f"wrote {path} n={len(features)} "
            f"residual mean {residual.mean():.1f} std {residual.std():.1f}",
            flush=True,
        )

    try:
        while len(features) < arguments.count:
            fen = openings[games % len(openings)]
            board = chess.Board(fen)
            for _ in range(random.randint(0, 4)):
                legal = list(board.legal_moves)
                if not legal:
                    break
                board.push(random.choice(legal))
            games += 1
            for _ in range(arguments.max_ply):
                if board.is_game_over(claim_draw=True) or len(features) >= arguments.count:
                    break
                key = chess.polyglot.zobrist_hash(board)
                uci, _ours = teacher.search(board, arguments.play_depth, arguments.play_budget)
                if not uci:
                    break
                if not board.is_check() and key not in seen and board.legal_moves:
                    cp = stockfish_cp(engine, board, arguments.sf_depth)
                    if cp is not None:
                        seen.add(key)
                        features.append(pack_features(board))
                        pesto.append(teacher.pesto(board))
                        sf_scores.append(cp)
                        stm.append(1 if board.turn == chess.WHITE else -1)
                        fens.append(board.fen())
                legal = list(board.legal_moves)
                if not legal:
                    break
                if random.random() < arguments.noise:
                    board.push(random.choice(legal))
                    continue
                move = chess.Move.from_uci(uci)
                board.push(move if move in board.legal_moves else random.choice(legal))

            if games <= 2 or games % 5 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"{len(features)}/{arguments.count} labelled from {games} games "
                    f"in {elapsed:.0f}s  last_depth {int(teacher.info[bb_search.I_DEPTH_DONE])}",
                    flush=True,
                )
            if len(features) - last_dump >= 500:
                dump(arguments.out)
                last_dump = len(features)
    finally:
        engine.quit()

    dump(arguments.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""v7.1 plays; Stockfish labels quiet positions. Writes STM features.

    python -m training.gen_v71_nnue --count 4000 --out data/v71_selfplay.npz
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

ROOT = Path(__file__).resolve().parents[1]
V71 = ROOT / "local" / "engine_snapshots" / "v7.1"

from tools.gauntlet import opening_positions  # noqa: E402
from training.v71_nnue_data import find_stockfish, pack_stm  # noqa: E402

OPENINGS = opening_positions() or [chess.STARTING_FEN]

# gauntlet imports the repo `engine` package; v7.1 must replace it.
for _name in [k for k in sys.modules if k == "engine" or k.startswith("engine.")]:
    del sys.modules[_name]
if str(V71) not in sys.path:
    sys.path.insert(0, str(V71))

from engine import bb  # noqa: E402
from engine import bb_eval  # noqa: E402
from engine import bb_search  # noqa: E402
from engine.bbpos import from_board, move_to_uci  # noqa: E402
from engine.nnue import warmup as warmup_nnue  # noqa: E402


class Player:
    def __init__(self) -> None:
        warmup_nnue()
        bb.init()
        self.tt, self.ord32, self.rep, self.info = bb_search.new_tables()
        self.pc = bb_eval.new_pawn_cache()
        self.hist, self.moves, self.scores = bb.new_buffers()

    def search(self, board: chess.Board, depth: int, budget_s: float) -> str:
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
            return ""
        return move_to_uci(packed)


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
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--play-depth", type=int, default=8)
    parser.add_argument("--sf-depth", type=int, default=8)
    parser.add_argument("--play-budget", type=float, default=2.0)
    parser.add_argument("--sf", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-ply", type=int, default=64)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("data/v71_selfplay.npz"))
    arguments = parser.parse_args()

    sf_path = find_stockfish(arguments.sf)
    print(
        f"stockfish {sf_path}  play-depth {arguments.play_depth}  "
        f"sf-depth {arguments.sf_depth}  budget {arguments.play_budget:.1f}s",
        flush=True,
    )
    engine = chess.engine.SimpleEngine.popen_uci(str(sf_path))
    engine.configure({"Threads": arguments.threads, "Hash": 128})

    random.seed(arguments.seed)
    player = Player()
    openings = OPENINGS

    features: list[np.ndarray] = []
    sf_scores: list[int] = []
    fens: list[str] = []
    seen: set[int] = set()
    games = 0
    started = time.perf_counter()
    last_dump = 0

    def dump(path: Path) -> None:
        if not features:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=np.stack(features),
            sf=np.array(sf_scores, dtype=np.int32),
            fen=np.array(fens),
            play_depth=np.int32(arguments.play_depth),
            sf_depth=np.int32(arguments.sf_depth),
        )
        print(f"wrote {path} n={len(features)}", flush=True)

    try:
        while len(features) < arguments.count:
            fen = openings[games % len(openings)]
            board = chess.Board(fen)
            for _ in range(random.randint(0, 6)):
                legal = list(board.legal_moves)
                if not legal:
                    break
                board.push(random.choice(legal))
            games += 1
            for _ in range(arguments.max_ply):
                if board.is_game_over(claim_draw=True) or len(features) >= arguments.count:
                    break
                key = chess.polyglot.zobrist_hash(board)
                uci = player.search(board, arguments.play_depth, arguments.play_budget)
                if not uci:
                    break
                if not board.is_check() and key not in seen and board.legal_moves:
                    cp = stockfish_cp(engine, board, arguments.sf_depth)
                    if cp is not None:
                        seen.add(key)
                        features.append(pack_stm(board))
                        sf_scores.append(cp)
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
                    f"in {elapsed:.0f}s  last_depth {int(player.info[bb_search.I_DEPTH_DONE])}",
                    flush=True,
                )
            if len(features) - last_dump >= 250:
                dump(arguments.out)
                last_dump = len(features)
    finally:
        engine.quit()

    dump(arguments.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

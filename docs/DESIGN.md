# Agent design

## Objective

Maximise Elo against other submitted agents under a hard compute envelope: 1 core, 2 GB, 50 MB zip, 120 s + 0.5 s/move, no network, no third-party engines.

This is a time-and-node allocation problem, not a “play pretty chess” problem. Depth, move ordering, and not flagging win more games than a fancier eval that searches two ply shallower.

## Why classical search first

A model is optional. The zip cannot contain Stockfish/Lc0. A network we did not train is a disqualification risk. A well-tuned alpha-beta engine with PeSTO evaluation is a complete, auditable entry and is the right baseline:

- Correctness is inspectable (required at the final).
- Strength scales immediately with nodes.
- Opening books, tablebases, and a later ONNX eval can plug in without changing `get_move`.

A policy/value net is a *later* patch to `evaluate.py`, not the architecture.

## Process (one game)

The judge starts a fresh process per game. Import has a 60 s budget; the clock is 120 s + 500 ms increment after that.

```
import agent.py
        │
        ├─ allocate transposition table (~64 MB)
        ├─ JIT-warmup any numba kernels
        ├─ one shallow search on startpos (fills caches)
        └─ runner marks ready
                │
                ▼
        loop until game over
                │
                ├─ STOP ponder thread (if any)
                ├─ parse FEN, reconstruct opponent move, update repetition keys
                ├─ if time is critical: play TT / first legal, return
                ├─ probe opening book
                ├─ iterative deepening (1 thread) until soft time
                ├─ return UCI
                ├─ push our move into game history
                └─ START ponder on predicted reply (1 background thread)
```

One thread while we are on the clock. Extra threads share the single core and lose Elo. Pondering is the exception: it runs only while the opponent is thinking.

## Time allocation

Expected length from a balanced curated opening: ~30–40 of our moves.

```
soft = time_left / moves_to_go + 0.6 * increment
soft = min(soft, 0.20 * time_left)     # never spend a fifth of the clock
hard = min(2 * soft, time_left - 40ms) # always leave a safety margin
```

`moves_to_go` shrinks as ply grows. In panic (`time_left < 1 s`) we drop to depth 1 / TT move. Fail-low and fail-high at the root get a small extra budget so we do not play a blunder after an aspiration miss.

The increment arrives *after* the move, so it is not in `time_left_ms`. Treat it as recoverable time, not spendable this move.

## Search

Negamax PVS with:

| Device | Role |
|---|---|
| Iterative deepening | Always have a legal move; use previous PV for ordering |
| Aspiration windows | Cheap re-search when the score is stable |
| Transposition table | Zobrist via `chess.polyglot`; 2^22 entries |
| Check extensions | Tactical shots are where games end |
| Null-move pruning | Skip in check / low non-pawn material (zugzwang) |
| Reverse futility, razoring, LMR, futility | Node budget goes to the PV |
| Quiescence | Captures and promotions; stand-pat; delta prune; skip losing captures |
| Killers, history, MVV-LVA, TT move | Move ordering is Elo |
| Draw detection | 2-fold in-search, 50-move, insufficient material |

Mate scores are ply-adjusted in the TT so a mate in 3 is not stored as a mate in 1 after a transposition.

## Evaluation

Tapered PeSTO (middlegame / endgame PSTs + phase), plus cheap bitboard terms that PeSTO underweights:

- Bishop pair
- Doubled / isolated / passed pawns
- Rook on open or semi-open file, rook on 7th
- King pawn-shield in the middlegame
- Mop-up (king tropism + opponent king to the edge) when converting a winning endgame

The 300-ply adjudication is on **material**. Converting a rook-up ending still matters because of the 50-move rule, which the referee claims automatically.

## State we keep between moves

The process lives for the whole game, so we keep:

- Transposition table (warm)
- Killer / history tables (light aging)
- Position hashes we have actually seen (threefold — the FEN does not include full history)
- Ponder thread + predicted opponent move

New game detection: if the incoming FEN is not a one-move successor of the board after our last move, reset history (but keep the TT; it is still a valid cache of chess).

## Opening book

Games start from curated *near-equal* positions, not always startpos. The book is a hash map of Polyglot Zobrist → weighted UCI moves, built at import from a compact set of mainlines. A hit is instant. A miss is a no-op; search handles it.

We do not ship a 5-piece syzygy (size). 3-piece knowledge is in eval mop-up.

## Failure budget

Illegal, crash, flag, init timeout, and oversized output are all instant losses. Defensive rules:

- `get_move` is wrapped; on any exception we return a legal move.
- After iterative deepening depth 1 we always have a move.
- Hard deadline is 40 ms before flag.
- We never write to the protocol; the runner owns stdout. No debug prints in rated play.
- Import plus warmup must stay well under 60 s.

## Official harness

The [aichessathon-starter](https://github.com/advitrocks9/aichessathon-starter) tree is vendored here (`harness/`, `baselines/`). Do not edit `harness/`.

```text
python -m harness.play --white . --black baselines/greedy
python -m harness.arena --opponent baselines/minimax --games 20
python -m harness.package
```

`--white .` is this repo: `runner.py` puts the directory on `sys.path` and `import agent`, so `search.py` / `evaluate.py` / `book.py` load the same way they will from the zip.

## Local vs judge

Judge: Python 3.12, exact pins, no `pip`.  
This machine: Python 3.10, closest wheels (`requirements-local.txt`). Agent code uses only `chess`, `numpy` (optional), stdlib. No 3.12-only syntax required. Do not add imports outside the allowed set.

## Strength roadmap

Keep a frozen `submission.zip` of the previous stage as an arena opponent. A change that cannot beat that zip at 10 s + 0.1 s over ≥40 games does not ship.

1. **Now (ship)** — python-chess search + PeSTO + book + ponder + time. Clock flags patched. Classical squeeze applied (SEE ordering, pawn hash, countermove, IIR, history gravity). Frozen pre-squeeze copy: `baselines/self_v1/`. Upload: `submission.zip`. Next work: `NEXT_ENGINE.md`.
2. **Classical squeeze** — SEE in ordering, singular/check extensions already on, pawn-hash, skip qsearch checks, more accurate NMP. Cheap Elo, same representation.
3. **Bitboards + numba** — own `uint64` board, jitted make/unmake/movegen/eval. Warm every `@njit` signature at import. `python-chess` only at the root to parse FEN and emit UCI. Target: 10–50× nodes, +1–3 ply.
4. **NNUE/ONNX eval** — small net (few MB, int8/int16), trained by us on engine-labelled quiet positions. Plug into the same `evaluate()` the search already calls. Do not train until stage 3 is the baseline; a slow net on python-chess loses depth.
5. **Ladder book** — hashes from positions the platform actually serves, not startpos ECO.

Do not spend the 50 MB on a net until it beats the bitboard+PeSTO zip at the same time control.

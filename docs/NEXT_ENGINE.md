# Next engine: bitboards, then NNUE

Hand-off for a **new chat**. Implement Stage 2 (numba bitboards) then Stage 3 (our NNUE). Do not start with a neural net. Do not edit `harness/`.

This file is the spec. `DESIGN.md` is the original architecture. `README.md` is how to run the starter harness.

---

## 1. Goal

Maximise Elo under the AI Chessathon envelope, then beat **this repo’s frozen self**, not only starter minimax.

| Constraint | Value |
|---|---|
| API | `get_move(fen: str, time_left_ms: int) -> str` (UCI) |
| Zip | `agent.py` at **archive root**, ≤ 50 MB unzipped |
| Runtime | Python 3.12, 1 core, 2 GB, no GPU, no network |
| Allowed imports | stdlib + `torch` CPU, `numpy`, `chess` 1.11.2, `onnxruntime`, `numba` |
| Banned | Stockfish, Lc0, Maia, any wrapper; native binaries; Cython extensions |
| Models | Only nets **we trained**. Engine-annotated training data is allowed |
| Time | 120 s + 0.5 s/move; 60 s init; pondering allowed on opponent time |
| Draw | Referee claims threefold and 50-move; track hashes ourselves |
| Output | 4096 byte cap on the protocol line (the runner owns that) |
| Source | Must be readable; obfuscation is a DQ |

Canonical rules: https://aichessathon.com/docs  
Starter (vendored here): https://github.com/advitrocks9/aichessathon-starter

`requirements.txt` in the zip is ignored. A new package needs `hello@aichessathon.com` and is announced to every team.

---

## 2. What is already built (do not rewrite)

Root submission modules (these go in `submission.zip`):

| File | Role |
|---|---|
| `agent.py` | `get_move`, game hash history, time budget, legality assert, exception fallback, JIT warmup |
| `bb.py` | Magic attacks, legal movegen, make/unmake, Zobrist, perft |
| `bb_eval.py` | Tapered PeSTO + structure + SEE, all `@njit` |
| `bb_search.py` | Jitted PVS/qsearch/TT; Python iterative-deepening driver |
| `bbpos.py` | `chess.Board` ↔ bitboard arrays, UCI packing |
| `book.py` | Polyglot Zobrist → UCI from compact ECO lines |
| `evaluate.py` | PST tables only (imported by `bb_eval`); old Python eval is unused |
| `search.py` | Frozen Python PVS. **Not called.** Still zipped because `harness.package` takes every root `*.py` |

Harness (do not edit): `harness/runner.py`, `sandbox.py`, `referee.py`, `play.py`, `arena.py`, `package.py`, `rules.py`.

Baselines:

- `baselines/random`, `greedy`, `minimax`, `numba` — starter
- **`baselines/self_v1/`** — frozen Python `python-chess` engine (1 Sep 2026, pre-squeeze)
- **`baselines/self_v2/`** — frozen Stage 2 numba bitboard engine (1 Sep 2026). Stage 3 opponent.

Local env: `.venv` on Python **3.10** (`requirements-local.txt`). Judge is 3.12 with numpy 2.5.2. Agent code must stay 3.10-compatible (`from __future__ import annotations` is fine). Do not `import` anything outside the allowed set.

```powershell
.\.venv\Scripts\python.exe -m harness.play --white . --black baselines/minimax
.\.venv\Scripts\python.exe -m harness.arena --opponent baselines/self_v1 --games 40 --base-ms 10000
.\.venv\Scripts\python.exe -m harness.package
.\.venv\Scripts\python.exe tools\smoke.py
```

Packaging: `python -m harness.package` zips **every `*.py` at the repo root** plus optional `weights/`. Keep helper scripts in `tools/`. Never name a file `chess.py` or `types.py`.

Upload file: **`submission.zip`** at the repo root (gitignored). `agent.py` must be at the zip root, not inside a folder.

---

## 3. How the current agent chooses a move

1. Platform starts one process per game. Import has 60 s. `_warmup()` builds magics, the book, the TT, and compiles the search via a depth-6 call.
2. Each request is JSON `{"fen", "time_left_ms"}`. We only implement `get_move`; `harness/runner.py` owns the wire.
3. `get_move` syncs FEN to game history (or starts a new game), returns the only legal move if unique.
4. Book probe. Hit → that UCI. Miss → search.
5. Time: **do not bank increment**. `soft = usable / moves_to_go`, `hard ≈ 1.45 * soft` capped. Panic when `usable` is small. `SAFETY_MS = 90`.
6. Search: Python iterative deepening around jitted PVS + qsearch on integer bitboards. Leaf eval is PeSTO (centipawns, side-to-move).
7. Return UCI only after `move in chess.Board(fen).legal_moves`.
8. Any exception → first legal move. Never crash. No pondering (jitted search holds the GIL).

Scores: mate ≈ 30000 minus ply. Draw = 0, including 2-fold in-search so we avoid repeating when ahead.

---

## 4. Measured strength (as of 1 Sep 2026, Stage 2)

All at **10 s + 0.1 s** unless noted. Zero illegal / crash / flag in the runs below.

| Matchup | Games | Score | Notes |
|---|---|---|---|
| vs `baselines/self_v1` (startpos) | 40 | **100%** | 40 mate. Shared deterministic book → ~2 independent games |
| vs `self_v1`, 12 out-of-book FENs | 24 | **100%** | Both colours; search, not book |
| vs `baselines/minimax`, 2 out-of-book FENs | 4 | **100%** | No flags |
| vs `baselines/random`, 4 out-of-book FENs, 2 s | 8 | **100%** | Correctness, not Elo |

`harness.arena` always starts from the standard position. Rated games do not. Use `python -m tools.gauntlet` for varied openings.

Pondering was dropped: a jitted search holds the GIL for the whole call, so a background ponder cannot be interrupted without flagging. Do not reintroduce it. Do not bank the increment (`time_left_ms` never includes it).

---

## 5. Classical squeeze just done (Stage 1)

Already in `search.py` / `evaluate.py`. Do not redo:

- SEE-based capture ordering (winning/equal first, losing after quiets)
- Pawn-structure hash (16k, key = white/black pawn bitboards)
- Countermove heuristic
- History gravity (`h + bonus - h*bonus/16384`)
- IIR when PV node has no TT move (`depth >= 4`)
- No LMR on killer moves
- Clock / ponder fixes from the measurement chat

Gate: any later change must beat **`baselines/self_v1` is the pre-squeeze freeze**. After Stage 2, freeze the bitboard agent as `baselines/self_v2` the same way (copy the four root modules into a new folder **before** the next rewrite).

---

## 6. Stage 2 — bitboards + numba (**done**, frozen as `baselines/self_v2`)

### Why

Almost every node is `board.legal_moves` / `push` / `pop` in Python. PeSTO is cheap. Jitting eval alone (starter `baselines/numba`) barely helps. **Jitted movegen + make/unmake** is the Elo.

Target: **10–50× nodes**, **+1–3 ply** at the same clock, and a **>50%** score vs `.` (current) and vs `baselines/self_v1` over ≥40 games at 10 s + 0.1 s.

### Architecture

Keep `get_move` in `agent.py`. Search should operate on integers.

```
FEN --python-chess--> bitboard position
                    --> numba search or Python PVS calling jitted make/eval
                    --> UCI string
                    --> assert move in chess.Board(fen).legal_moves before return
```

**Position (arrays or a numba `jitclass` / tuple of uint64):**

- 12 bitboards: WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK
- `side`, `castling` (4 bits), `ep` (0–63 or 64), `halfmove`, `hash` (incremental Zobrist)

**Must be `@njit` and warmed at import** with the **exact dtypes** used in search (`cache=False` — judge FS is read-only except `/tmp`):

1. Attack / slider moves (magics or hyperbola)
2. `generate_moves` (legal, or pseudo-legal + king-not-in-check)
3. `make` / `unmake` (unmake is faster than copy-make)
4. PeSTO on bitboards (copy the existing PST numbers from `evaluate.py`)
5. SEE
6. Incremental Zobrist

**Python only at the root:** parse FEN, emit UCI, legality assert, book (can keep `book.py` + `chess.polyglot` at the root).

Two implementation orders (pick A then B):

**A. Hybrid (safer):** Python `_pvs` in `search.py` but `make`/`unmake`/`gen`/`eval` are numba. Faster than today, still a Python loop.

**B. Full jitted search:** PVS inside numba. Much faster, harder to debug. Do A until legality is proven, then B.

### Correctness gate (before any arena)

Play ≥200 games vs `baselines/random` at 2 s. **Zero** `illegal` / `crash`. Log every root move through `chess.Board.legal_moves`. Cover promotions, castling, en passant, checks.

Also: perft vs `python-chess` on startpos, Kiwipete, and a few positions with ep/castle (`chess.Board.perft` if available, or count `legal_moves` recursively).

### Init

Warm every jitted function in `agent._warmup()` so compile is on the 60 s budget. One core: `torch.set_num_threads(1)` if you import torch (you should not need torch in Stage 2).

### Suggested new files (root, so they zip)

- `bb.py` — types, magics/attacks, make/unmake, movegen
- `bb_eval.py` — PeSTO port
- Keep `evaluate.py` as a wrapper or delete once search no longer imports `chess.Board` eval

Do not add files named after stdlib/chess modules.

### What not to do in Stage 2

- Do not train a net yet
- Do not call `chess.Board` inside the hot loop
- Do not use extra threads on our clock
- Do not ship `.pyd` / `.so`

---

## 7. Stage 3 — NNUE (**first net shipped**, gate vs self_v2 not yet run)

A **value head for alpha-beta**, not AlphaZero, not MCTS, not a convnet.

### What is wired (1 Sep 2026)

- `nnue.py` — 768→128→32→1, clipped ReLU, `400*tanh` residual in centipawns, STM. Loaded at import from `weights/nnue.npz`. PeSTO-only if load fails.
- `bb_eval.evaluate` adds `nnue_delta`. Search signatures unchanged.
- `tools/gen_nnue.py` — our engine plays; **Stockfish 18** (local `tools/sf/`, gitignored, not in the zip) labels quiet positions at depth 10.
- `tools/train_nnue.py` — PyTorch AdamW, 200 epochs, early stop. Writes weights only if val MAE beats a zero residual.

### First trained net

8000 quiet positions, Stockfish 18 depth 10. Residual std 311cp vs PeSTO. PyTorch MLP val MAE **107.5cp** vs **203.8cp** for “predict 0”. Numba inference matches torch within 1cp. Gate vs `baselines/self_v2` still required before calling Stage 3 done.

---

## 8. Definition of done (copy into the new chat)

**Stage 2 done when all of:**

- [x] Root move always legal vs `python-chess`
- [x] Games vs random, 0 illegal/crash/flag (8 varied-opening games at 2 s; 12/12 self-play repro also clean)
- [x] Perft matches on startpos + Kiwipete
- [x] nps well above 10× the Python engine (jitted movegen + search)
- [x] vs `baselines/self_v1` at 10s+0.1s: 40/40 startpos and 24/24 out-of-book
- [x] Frozen as `baselines/self_v2/`; vs `baselines/minimax` 4/4 no flags
- [x] `python -m harness.package`: `agent.py` at zip root, **107 KB** unzipped
- [ ] Import + JIT warmup not remeasured this freeze (warmup is at import via `bb_search.search` depth 6). Re-check if Stage 3 adds compile cost.

**Stage 3 done when:** arena vs `baselines/self_v2` > 55% at the same TC, no flags, net trained by us.

---

## 9. Pitfalls that lose games for free

- Flag: check the clock inside search; never spend more than `time_left - SAFETY_MS`
- Init timeout: JIT at import, not move 1
- Illegal move: bitboard bugs, especially castle through check, ep, underpromotion
- Shadowed imports: `chess.py`, `random.py`, `types.py`
- Writing outside `/tmp` on the judge
- `print` of huge debug (cap is on the protocol; still keep rated play quiet)
- Two threads on our clock
- Shipping someone else’s `.onnx` / Maia / Stockfish

---

## 10. Prompt to paste into the next chat

```
Read NEXT_ENGINE.md, DESIGN.md, agent.py, bb.py, bb_eval.py, bb_search.py,
bbpos.py, book.py, and baselines/self_v2/README.md.

Stage 2 (numba bitboards) is done and frozen as baselines/self_v2.
Implement Stage 3 only: a value-head NNUE/ONNX (or numba matmul) for
alpha-beta, trained by us. Keep PeSTO as fallback. Do not edit harness/.

Keep get_move(fen, time_left_ms) -> UCI. Warm all njit/ONNX at import.
Assert every returned move is in chess.Board(fen).legal_moves.

Gate: vs baselines/self_v2 at 10s+0.1s, score > 55%, zero flags.
If the net is slower per leaf than the extra accuracy is worth, do not ship.

Python 3.10 local venv: .\.venv\Scripts\python.exe
Allowed packages only: chess, numpy, numba, torch, onnxruntime, stdlib.
```

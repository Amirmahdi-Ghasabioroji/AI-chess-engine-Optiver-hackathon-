# AI Chessathon agent

A chess engine for the [AI Chessathon](https://aichessathon.com/docs). The platform
imports `agent.py` and calls `get_move(fen, time_left_ms)`. The return value is a
UCI move such as `e2e4` or `e7e8q`. The side to move in the FEN is our side.
There is no other input.

The root engine is **v7.1.1**. `agent.py` calls `engine.play`, which runs the
numba PVS in `engine/bb_search.py`. The leaf evaluation is classical PeSTO
mixed with the trained net in `weights/nnue.npz`: classical plus three-fifths
of the clipped gap between the net and classical (60% net, 40% classical).
The net is required. `ROADMAP.md` is the short note that shipped with this
version.

python-chess parses the FEN. If `get_move` raises, it returns a legal move so
a Python exception does not forfeit the game by itself.

Two other engines are in the repo for comparison, and are not what the zip plays:

| Engine | Path | What it is |
|---|---|---|
| v2.1 | `baselines/v2.1` | Classical numba bitboard engine. No network. |
| Stage 3 | `baselines/self_v3` | The previous root: PeSTO plus a linear piece-square residual. |

## Where to start reading

A reviewer who wants the playing code, in this order:

1. `ROADMAP.md` — what v7.1.1 changed and what not to undo.
2. `agent.py` — import-time warmup, then `get_move`.
3. `engine/play.py` — clock, book, and the call into search.
4. `engine/bb_search.py` — iterative-deepening PVS, transposition table, quiescence.
5. `engine/bb_eval.py` — `evaluate()` is the classical + NNUE mix. Classical PeSTO is also used for static exchange.
6. `engine/nnue.py` — loads `weights/nnue.npz` and blends the net into the eval.
7. `engine/bb.py` — bitboards and move generation.
8. `baselines/v2.1/agent.py` — the published classical engine, with its own copies of the search files beside that `agent.py`.

Longer design notes are in `docs/DESIGN.md`, `docs/NEXT_ENGINE.md`, and `docs/IDEAS.md`.

## Repository layout

| Path | In the submission zip | Role |
|---|---|---|
| `agent.py` | yes | The only module the platform imports. |
| `engine/` | yes | v7.1.1 search, eval, book, and NNUE code. |
| `weights/nnue.npz` | yes | The net v7.1.1 evaluates. |
| `ROADMAP.md` | no | v7.1.1 notes. |
| `harness/` | no | Official game runner. Mirrors the platform clock. Leave it unchanged. |
| `baselines/` | no | Local opponents, including published `v2.1` and frozen `self_v1`–`self_v3`. |
| `tools/` | no | Pack, gauntlet, perft, smoke, benches. |
| `training/` | no | Offline labelling and training. |
| `experiments/` | no | Round-robin drivers, `RESULTS.md`, and saved game logs. |
| `docs/` | no | Design notes. |
| `archive/` | no | Stage 1 Python search, plus `stage3_weights/` from the previous root. |
| `.github/workflows/ci.yml` | no | `make gate` on Ubuntu, plus a two-game arena on macOS and Windows. |

These paths exist only on a working machine. They are gitignored:

| Path | Why it stays local |
|---|---|
| `.venv/` | Local environment. |
| `data/` | Training dumps, including a large Lichess extract. |
| `tools/sf/` | A local Stockfish binary used to label positions. Shipping any existing engine is a disqualification. |
| `local/engine_snapshots/` | Full copies of earlier engines. The round-robin scripts look here. |
| `logs/` | Gauntlet and measure output. |
| `dist/`, `*.zip` | Build artefacts. Recreate the submission with `make zip`. |

`experiments/logs/` is the exception: those small round-robin logs are committed.

## Platform contract

Confirm limits on the site before relying on a number. The two pages that change are:

- https://aichessathon.com/docs/agent-contract.md
- https://aichessathon.com/docs/rules.md

As of this tree, a game looks like this:

- One process per game. It stays alive between our moves. Module state survives inside that game and is gone for the next game.
- Import has 60 seconds before the clock starts. Numba compilation and weight loading happen there. `agent.py` warms the jitted search on the starting position during import.
- Clock is 120 seconds plus 0.5 seconds per move, per side, on wall time. One core, 2 GB RAM, no network, no GPU.
- An illegal move, a crash, a malformed reply, running out of memory, or flagging loses that game. A reply over 4 KB counts as illegal.
- The unzipped submission must stay under 50 MB. `agent.py` must sit at the root of the zip, not inside a folder.
- Rated games start from curated openings. That set is not published, so local games from the standard start position overstate the opening book.
- The filesystem is read-only except for 256 MB at `/tmp`. Weights ship inside the zip. Nothing downloads at runtime.
- The platform image already has Python 3.12, torch 2.13 (CPU), numpy 2.5, python-chess 1.11, onnxruntime 1.29, and numba 0.67. A `requirements.txt` in the zip is ignored. An import outside that set crashes on the platform even when it works locally.
- Native binaries in the zip are rejected. The code that ships is Python a judge can read.

`print` is safe. The runner points file descriptor 1 at stderr before import, so logs cannot corrupt the move protocol. Rated games discard them. The validation log shows them. Set `CHESS_DEBUG=1` to turn on the messages in `agent.py`.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/) match the judge and the GitHub workflow. From the repo root:

```powershell
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock`. Torch comes from the CPU index declared there. Dev tools are ruff and mypy.

On this checkout you can also call `.\.venv\Scripts\python.exe` directly. That environment is local and is not committed.

## Play and measure

Run everything from the repo root.

One game, official time control, against the greedy baseline. `make play` does this. Pass a start position with `make play FEN="..."`.

```powershell
uv run python -m harness.play --white . --black baselines/greedy
```

Twenty fast games from the standard start, against greedy. `make arena` does this. Both sides share a deterministic book, so a long arena from the start position repeats the same opening.

```powershell
uv run python -m harness.arena --opponent baselines/greedy --games 20
```

A more honest local score is the gauntlet. It plays distinct book tails, each twice with colours swapped, and writes a JSONL log (default `logs/gauntlet.jsonl`).

```powershell
uv run python -m tools.gauntlet --opponent baselines/self_v3 --openings 4 --workers 1
```

Frozen opponents, oldest first. Do not edit them. They are the A/B line for later stages.

| Opponent | What it is |
|---|---|
| `baselines/random`, `greedy`, `minimax`, `numba` | Starter bots from the official harness. |
| `baselines/self_v1` | Pure-Python PVS, before the bitboard port. |
| `baselines/self_v2` | Numba bitboards and PeSTO. |
| `baselines/v2.1` | Published classical engine (search and eval only, no network). |
| `baselines/self_v3` | Previous root. PeSTO plus a linear PST residual. |

Other local checks:

| Command | What it checks |
|---|---|
| `uv run python -m tools.perft` | Move generation against python-chess. |
| `uv run python -m tools.check_eval` | Written for the Stage 3 eval in `baselines/self_v3`, which had `engine/evaluate.py`. |
| `uv run python -m tools.smoke` | Legality, mate-in-one, the clock, and a book probe. |
| `uv run python -m tools.search_bench` | Node rate of the bitboard search and of `archive/search.py`. |
| `uv run python -m tools.import_cost` | How the 60 second import budget is spent. |
| `uv run python -m tools.repro_crash` | In-process games, so a search crash has a traceback. |
| `make gate` | `ruff check .`, `mypy` on `agent.py` and `harness/`, then two short games against `baselines/random`. |

`make gate` is what `.github/workflows/ci.yml` runs on Ubuntu. The same workflow also plays two short games on macOS and Windows. The platform's own validation log, after upload, is the authority for whether a zip is accepted.

## Build the submission

```powershell
uv run python -m tools.pack
```

`make zip` is the same packer: every root `*.py` (that is `agent.py`), plus `engine/` and `weights/`. The archive root must be those files, not a wrapping folder. The command prints the member list and the unzipped size, and exits non-zero if `agent.py` is missing or the unzipped total exceeds 50 MB.

Upload that zip. The latest upload that passes validation is the one that plays. Local `make gate` does not replace that check.

## Experiments

`experiments/RESULTS.md` is a finished four-engine round-robin. The clock in that file is a compressed stand-in (5 s + 50 ms, not 120 s + 0.5 s), on a local interpreter. Use it to order engines. It is not a platform Elo.

Replay it, or run the longer mix-field round-robin, from the repo root:

```powershell
.\.venv\Scripts\python.exe experiments\run_round_robin.py
.\.venv\Scripts\python.exe experiments\run_mix_field_rr.py
```

Both scripts resolve opponents under `local/engine_snapshots/`. That directory is gitignored, so a fresh clone has the drivers and the saved logs, and needs the snapshots copied back before a rerun. Per-game JSONL from those runs is in `experiments/logs/`.

`experiments/test_v*.py` and `experiments/compare_v71111_nps.py` are version checks against those same snapshots.

## Training

Training scripts live in `training/` and are not part of the zip. They read and write `data/`, which is gitignored. Labels may come from the local Stockfish in `tools/sf/`. That binary annotates positions. It is not imported by `agent.py` and must not be packed.

v7.1.1 plays the net already stored in `weights/nnue.npz`. The scripts under `training/` fit earlier nets and the Stage 3 piece-square residual. Several of them import the Stage 3 module layout, so they are not the training entry point for the root engine.

## License

MIT. See `LICENSE`.

The starter harness and the `random`, `greedy`, `minimax`, and `numba` baselines come from [aichessathon-starter](https://github.com/advitrocks9/aichessathon-starter). The agent, `engine/`, tools, and trained weights are the team's.

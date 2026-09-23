# AI Chessathon agent

A chess engine for the [AI Chessathon](https://aichessathon.com/docs). The platform
imports `agent.py` and calls `get_move(fen, time_left_ms)`. The return value is a
UCI move such as `e2e4` or `e7e8q`. The side to move in the FEN is our side.
There is no other input.

The playing engine is a numba bitboard search. Evaluation is tapered
[PeSTO](https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function) plus a
linear piece-square residual trained offline and stored in `weights/pst.npz`.
python-chess is used only to parse the FEN and to check that the move we return
is legal. If the bitboard search ever proposes an illegal move, `agent.py`
falls back to a legal one so a bug does not lose the game by itself.

Search does not call the small MLP in `engine/nnue.py`. That module, and the
`nnue*.npz` / `nnue*.pt` files next to `pst.npz`, are the training and
experiment nets. `agent.py` still loads and warms them at import so a later
switch-on does not pay compilation on the clock. The packer includes the whole
`weights/` directory, so those files are in the submission zip.

There is no pondering. A jitted search holds the GIL for the whole call, and a
missed deadline loses the game.

## Where to start reading

A reviewer who wants the playing code, in this order:

1. `agent.py` — time split, opening book, legality check, import-time warmup.
2. `engine/bb_search.py` — iterative-deepening PVS, transposition table, quiescence.
3. `engine/bb_eval.py` — PeSTO, static exchange, and the PST residual. `evaluate()` is what search calls.
4. `engine/bb.py` — bitboards and move generation.
5. `engine/book.py` — opening book built from mainlines at import.
6. `engine/bbpos.py` — FEN and UCI conversion. This is the python-chess boundary.
7. `engine/evaluate.py` — the pure-Python PeSTO reference. `tools/check_eval.py` checks the bitboard eval against it.
8. `engine/nnue.py` — MLP residual used by training and tests, not by the search leaves.

Longer design notes are in `docs/DESIGN.md`, `docs/NEXT_ENGINE.md`, and `docs/IDEAS.md`.

## Repository layout

| Path | In the submission zip | Role |
|---|---|---|
| `agent.py` | yes | The only module the platform imports. |
| `engine/` | yes | Search, eval, book, NNUE code. |
| `weights/` | yes | `pst.npz` (played) and the NNUE files (loaded, not the leaf eval). |
| `harness/` | no | Official game runner. Mirrors the platform clock. Leave it unchanged. |
| `baselines/` | no | Opponents for local games. Leave `self_v1`, `self_v2`, and `self_v3` unchanged. |
| `tools/` | no | Pack, gauntlet, perft, smoke, benches. |
| `training/` | no | Offline labelling and training. |
| `experiments/` | no | Round-robin drivers, `RESULTS.md`, and saved game logs. |
| `docs/` | no | Design notes. |
| `archive/` | no | Stage 1 Python search (`search.py`). Timed by `tools/search_bench.py`. |
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
| `baselines/self_v3` | PeSTO plus the linear PST residual. |

Other local checks:

| Command | What it checks |
|---|---|
| `uv run python -m tools.perft` | Move generation against python-chess. |
| `uv run python -m tools.check_eval` | Bitboard eval against `engine/evaluate.py`, and SEE sanity. |
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

The residual the search actually adds is produced by:

```powershell
uv run python -m training.train_pst --data data/nnue_train.npz
```

That writes `weights/pst.npz`. `training/gen_nnue.py` plays quiet positions with this engine and records Stockfish scores into `data/`. The other `train_*.py` and `import_*.py` scripts fit or fine-tune the MLP nets used in the version experiments.

## License

MIT. See `LICENSE`.

The starter harness and the `random`, `greedy`, `minimax`, and `numba` baselines come from [aichessathon-starter](https://github.com/advitrocks9/aichessathon-starter). The agent, `engine/`, tools, and trained weights are the team's.

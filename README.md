# AI Chessathon agent

Classical bitboard search (numba) plus PeSTO and a small learned PST residual.
The platform contract is `agent.py` at the zip root, exposing
`get_move(fen, time_left_ms) -> str` (UCI).

Rated games: 120 s + 0.5 s/move, 60 s init, 1 core, 2 GB, no network, no GPU.
Canonical rules: https://aichessathon.com/docs

## Layout

| Path | Role |
|---|---|
| `agent.py` | Platform entry. Time split, book, legality check, JIT warmup. |
| `engine/` | Bitboards, search, eval, book, NNUE inference. This is what ships. |
| `weights/` | Runtime nets (`pst.npz`, `nnue.npz`). |
| `harness/` | Official starter protocol. Do not edit. |
| `baselines/` | Starter bots plus frozen `self_v1`–`self_v3`. |
| `tools/` | Packer, gauntlet, perft, smoke, and benches. |
| `training/` | Offline training and data import. Not part of the submission zip. |
| `experiments/` | Round-robin drivers, `RESULTS.md`, and the small game logs. |
| `docs/` | Design notes. |
| `archive/` | Stage 1 Python search. Not on the rated path. |

Historical engine trees stay on disk under `local/engine_snapshots/` and are
gitignored. So are `.venv/`, `data/` (training dumps), `tools/sf/` (local
Stockfish, never shipped), `logs/` (gauntlet output), and `*.zip`.

## Run locally

Python 3.12 and `uv` match the judge. This checkout also has a local `.venv`:

```powershell
uv sync
uv run python -m harness.play --white . --black baselines/greedy
uv run python -m harness.arena --opponent baselines/self_v2 --games 4 --base-ms 10000
uv run python -m tools.gauntlet --opponent baselines/self_v3 --openings 4 --workers 1
uv run python -m tools.pack
```

`make play`, `make arena`, `make zip`, and `make gate` do the same jobs.

`make zip` puts `agent.py` at the archive root, plus `engine/` and `weights/`.
A zip that wraps a folder is rejected by the platform.

Round-robin and version checks:

```powershell
.\.venv\Scripts\python.exe experiments\run_round_robin.py
.\.venv\Scripts\python.exe experiments\run_mix_field_rr.py
```

Those drivers look for opponents in `local/engine_snapshots/`. Training:

```powershell
uv run python -m training.train_pst --data data/nnue_train.npz
uv run python -m training.gen_nnue --count 8000 --out data/nnue_train.npz
```

## What must never enter git or the zip

- `tools/sf/` — third-party engine. Shipping it is a disqualification.
- `.venv/`, `data/`, `local/`, `logs/`, `submission.zip`, native binaries.

Harness: vendored from [aichessathon-starter](https://github.com/advitrocks9/aichessathon-starter).

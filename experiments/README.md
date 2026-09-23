# experiments/

Round-robin drivers and the logs from those runs.

`run_round_robin.py` and `run_mix_field_rr.py` play through `harness` and
`tools.gauntlet`. Opponent trees are the local copies under
`local/engine_snapshots/`, which is gitignored. `RESULTS.md` and `logs/` are
the record of runs that already finished.

```powershell
.\.venv\Scripts\python.exe experiments\run_round_robin.py
.\.venv\Scripts\python.exe experiments\run_mix_field_rr.py
```

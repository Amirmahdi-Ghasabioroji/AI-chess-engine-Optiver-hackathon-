Frozen snapshot of the engine **before** the classical squeeze (1 Sep 2026).

This directory is a legal harness opponent:

```
python -m harness.arena --opponent baselines/self_v1 --games 40 --base-ms 10000
```

`agent.py` here imports local `search.py` / `evaluate.py` / `book.py`. Do not edit these files; they are the A/B baseline for later stages.

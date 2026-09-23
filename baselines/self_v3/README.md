Frozen snapshot of the Stage 3 engine (2 Sep 2026).

PeSTO + pawn threats + a linear PST residual (`weights/pst.npz`) on the
numba bitboard search. No PyTorch at runtime. Do not edit these files.

```
python -m harness.arena --opponent baselines/self_v3 --games 40 --base-ms 10000
python -m tools.gauntlet --opponent baselines/self_v3 --openings 2 --workers 1
```

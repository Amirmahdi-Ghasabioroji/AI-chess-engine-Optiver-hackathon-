# self_v2.1

Working classical engine. Frozen snapshot: `Test_engines/self_v2.1_base`
(and `self_v2.1_base.zip`). Do not edit `baselines/self_v2` or `self_v2.1_base`.

## vs the frozen base

- **Search** — IID at depth ≥ 8; safer NMP (depth ≥ 4, npm ≥ 500, verification
  at depth ≥ 8); contextual LMR; RFP 80/100; qsearch ignores depth-0 TT cuts
  and skips SEE-negative quiet checks after the first ply; instability time
  and depth-dependent aspiration.
- **Eval** — mobility; hanging pieces; pawn storms; two-rank king shield;
  nonlinear king pressure (queen-scaled); passed-pawn tropism / rook-behind /
  race; opposite-bishop draw factor; stronger king-hunt in winning endings.
- **Book** — deeper English/Reti lines plus extra Petrov, Scotch, Dragon,
  Taimanov, Benko, Dutch, KIA, Vienna, Philidor, QGA mainlines.

```
python -m tools.gauntlet --agent baselines/v2.1 --opponent baselines/self_v2 --openings 2 --workers 1
```

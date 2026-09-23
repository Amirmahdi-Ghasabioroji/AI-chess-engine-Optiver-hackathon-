# v7.1.1

Clone of `v7.1` with frozen v7's eval mix restored:
`classical + 3/5 clip(nnue − classical)` (60% NNUE, 40% classical).

Search is still the v7.1 numba PVS (v2.1 stack plus improving / capture history).

## Do not undo

- `get_move` runs `engine.play` → `bb_search.search`, not Python `Searcher`.
- `evaluate()` is the mix. Classical PeSTO is also used for SEE.
- Load is mandatory.

## Search deltas vs v7

- Improving (eval vs ply-2) for reverse futility and LMR.
- Capture history on losing/winning captures.
- Killers also written to ply+2.
- Qsearch TT cutoffs only from depth ≥ 1 entries (stand-pat still refined from any hit).

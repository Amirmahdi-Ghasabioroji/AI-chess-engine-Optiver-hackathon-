"""SPSA-ready search constants.

The Lichess article's SPSA workflow treats these as knobs. Values here are
hand-seeded for 120s+0.5s one-core Python; a later tune can overwrite them.
"""

from __future__ import annotations

# Reverse futility: if static eval is this far above beta, fail high.
# Kept shallow — the article's SPRT warning: speculative pruning regresses easily.
RFP_MARGIN = 180
RFP_MAX_DEPTH = 2

# Classic futility: skip late quiets when eval + margin cannot raise alpha.
FUTILITY_MARGIN = 220
FUTILITY_MAX_DEPTH = 2

# Razor into qsearch when the position looks hopeless at low depth.
RAZOR_MARGIN = 350
RAZOR_MAX_DEPTH = 1

# Null-move reduction: R = NMP_BASE + depth // NMP_DIV.
NMP_MIN_DEPTH = 3
NMP_BASE = 2
NMP_DIV = 4

# LMR: reduction = max(0, LMR_C + log(d)*log(m)*LMR_K)
LMR_C = 0.35
LMR_K = 0.52
LMR_MIN_DEPTH = 3
LMR_MIN_MOVE = 3

# Aspiration window around the previous iteration score.
ASPIRATION = 26

# Qsearch delta (stand-pat + capture + this still < alpha → skip).
DELTA_MARGIN = 80

# History gravity.
HIST_MAX = 8000

"""Deterministic Zobrist keys."""

from __future__ import annotations

import random

_rng = random.Random(0x7E11CE)

def _r() -> int:
    return _rng.getrandbits(64)

Z_PIECE = [[_r() for _ in range(64)] for _ in range(12)]
Z_SIDE = _r()
Z_CASTLE = [_r() for _ in range(16)]
Z_EP = [_r() for _ in range(8)]

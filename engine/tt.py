"""Numpy transposition table."""

from __future__ import annotations

import numpy as np

from engine.const import MATE, MATE_GATE

EXACT, LOWER, UPPER = 0, 1, 2
TT_BITS = 20
TT_SIZE = 1 << TT_BITS
TT_MASK = TT_SIZE - 1


class TT:
    __slots__ = ("depth", "flag", "key", "move", "score")

    def __init__(self, bits: int = TT_BITS) -> None:
        size = 1 << bits
        self.key = np.zeros(size, dtype=np.uint64)
        self.move = np.zeros(size, dtype=np.uint32)
        self.score = np.zeros(size, dtype=np.int32)
        self.depth = np.full(size, -1, dtype=np.int8)
        self.flag = np.zeros(size, dtype=np.int8)

    def probe(self, key: int, ply: int) -> tuple[int, int, int, int] | None:
        idx = key & TT_MASK
        if int(self.key[idx]) != (key & ((1 << 64) - 1)):
            return None
        score = int(self.score[idx])
        if score > MATE_GATE:
            score -= ply
        elif score < -MATE_GATE:
            score += ply
        return int(self.move[idx]), score, int(self.depth[idx]), int(self.flag[idx])

    def store(self, key: int, move: int, score: int, depth: int, flag: int, ply: int) -> None:
        idx = key & TT_MASK
        if int(self.depth[idx]) > depth and int(self.key[idx]) == (key & ((1 << 64) - 1)):
            return
        stored = score
        if stored > MATE_GATE:
            stored += ply
        elif stored < -MATE_GATE:
            stored -= ply
        stored = max(-MATE, min(MATE, stored))
        self.key[idx] = np.uint64(key & ((1 << 64) - 1))
        self.move[idx] = np.uint32(move)
        self.score[idx] = stored
        self.depth[idx] = np.int8(min(depth, 127))
        self.flag[idx] = np.int8(flag)

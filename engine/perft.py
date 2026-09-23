"""Perft: the article's first correctness tool."""

from __future__ import annotations

from engine.moves import gen_legal
from engine.pos import Position


def perft(pos: Position, depth: int) -> int:
    if depth == 0:
        return 1
    moves = gen_legal(pos)
    if depth == 1:
        return len(moves)
    nodes = 0
    for move in moves:
        pos.make(move)
        nodes += perft(pos, depth - 1)
        pos.unmake()
    return nodes

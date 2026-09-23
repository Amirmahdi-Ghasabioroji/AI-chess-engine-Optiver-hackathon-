"""Compact opening book: Polyglot Zobrist → weighted UCI moves.

Built at import from mainlines so curated near-equal starts still hit
when they transpose into a known line.
"""

from __future__ import annotations

import chess
import chess.polyglot

# Each line is a sequence of UCI moves from startpos.
# Weights: later we store (move, weight) per hash; repeating a move in
# several lines increases its weight.
_LINES: tuple[str, ...] = (
    # Open games
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8",
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 e8g8",
    "e2e4 e7e5 g1f3 b8c6 f1b5 g8f6 e1g1 f8c5",
    "e2e4 e7e5 g1f3 b8c6 f1b5 g8f6 e1g1 f6e4 d2d4",
    "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d3 f8c5 e1g1 d7d6",
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d4 e5d4 c3d4 c5b4",
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 d2d3 g8f6 e1g1 d7d6",
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 g8f6 d4c6 b7c6 e4e5",
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 f8c5 d4b3 c5b6",
    "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3 f6e4 d2d4",
    "e2e4 e7e5 f1c4 g8f6 d2d3 c7c6 g1f3 d7d5",
    "e2e4 e7e5 b1c3 g8f6 f1c4 b8c6 d2d3 f8c5",
    "e2e4 e7e5 f2f4 e5f4 g1f3 d7d5 e4d5 g8f6",
    # Sicilian
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 f1e2 e7e5 d4b3 f8e7",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1e3 e7e5 d4b3 c8e6",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 g7g6 c1e3 f8g7 f2f3 e8g8",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 b8c6 c1g5 e7e6 d1d2 f8e7",
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 d7d6 f1e2 e7e5 d4b3 f8e7",
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g7g6 c1e3 f8g7 b1c3 g8f6",
    "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 a7a6 f1d3 g8f6 e1g1 d8c7",
    "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 b8c6 b1c3 d8c7 f1e2 a7a6",
    "e2e4 c7c5 c2c3 g8f6 e4e5 f6d5 d2d4 c5d4 g1f3",
    "e2e4 c7c5 b1c3 b8c6 f2f4 g7g6 g1f3 f8g7",
    "e2e4 c7c5 b1c3 e7e6 g2g3 b8c6 f1g2 g7g6",
    # French / Caro / Pirc / Modern / Scandi
    "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4 e4e5 c7c5 a2a3 b4c3 b2c3",
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 c1g5 f8e7 e4e5 f6d7 g5e7 d8e7",
    "e2e4 e7e6 d2d4 d7d5 b1d2 g8f6 e4e5 f6d7 f1d3 c7c5 c2c3",
    "e2e4 e7e6 d2d4 d7d5 e4e5 c7c5 c2c3 b8c6 g1f3 d8b6",
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5 e4g3 f5g6 h2h4 h7h6",
    "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5 g1f3 e7e6 f1e2",
    "e2e4 c7c6 d2d4 d7d5 e4d5 c6d5 c2c4 g8f6 b1c3",
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6 f2f4 f8g7 g1f3 e8g8",
    "e2e4 g7g6 d2d4 f8g7 b1c3 d7d6 f2f4 g8f6 g1f3",
    "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5 d2d4 g8f6",
    "e2e4 g8f6 e4e5 f6d5 d2d4 d7d6 g1f3",
    # Queen's pawn
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c4d5 e6d5 c1g5 c7c6 e2e3",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 g1f3",
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 e7e6 e2e3 b8d7",
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 d5c4 a2a4 c8f5",
    "d2d4 d7d5 c2c4 d5c4 e2e3 g8f6 f1c4 e7e6 g1f3 c7c5",
    "d2d4 d7d5 g1f3 g8f6 c1f4 e7e6 e2e3 f8d6",
    "d2d4 d7d5 g1f3 g8f6 c2c4 e7e6 g2g3 f8e7 f1g2 e8g8",
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 d1c2 e8g8 a2a3 b4c3",
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3 e8g8 f1d3 d7d5",
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6 g2g3 c8b7 f1g2 f8e7",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8",
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5 f6d5 e2e4 d5c3 b2c3 f8g7",
    "d2d4 g8f6 c2c4 g7g6 g1f3 f8g7 g2g3 e8g8 f1g2 d7d6",
    "d2d4 g8f6 c2c4 c7c5 d4d5 e7e6 b1c3 e6d5 c4d5 d7d6",
    "d2d4 g8f6 c2c4 e7e5 d4e5 f6e4 g1f3 b8c6",
    "d2d4 f7f5 c2c4 g8f6 g2g3 e7e6 f1g2 f8e7",
    "d2d4 d7d5 c2c4 e7e6 b1c3 c7c5 c4d5 e6d5 g1f3 b8c6",
    # English / Reti / Flank — extra ply so curated starts still hit
    "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6 g2g3 d7d5 c4d5 f6d5 f1g2 d5b6 e1g1 f8e7 d2d3 e8g8",
    "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6 e2e3 f8b4 d1c2 e8g8 a2a3 b4c3 c2c3 d7d5",
    "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6 g2g3 f8b4 f1g2 e8g8 e1g1 f8e8 d2d3 h7h6",
    "c2c4 e7e5 b1c3 b8c6 g2g3 g7g6 f1g2 f8g7 e2e4 d7d6 g1e2 g8e7 d2d3 e8g8 e1g1 c8e6",
    "c2c4 e7e5 b1c3 g8f6 g2g3 f8b4 f1g2 e8g8 e2e4 b4c3 b2c3 d7d6 g1e2 b8c6 e1g1 a7a5 d2d3",
    "c2c4 e7e5 b1c3 f8b4 g2g3 b4c3 b2c3 g8f6 f1g2 e8g8 e2e4 d7d6 g1e2 c7c6 e1g1",
    "c2c4 e7e5 b1c3 b8c6 g2g3 g7g6 f1g2 f8g7 e2e3 d7d6 g1e2 g8e7 d2d4 e8g8 e1g1",
    "c2c4 e7e5 g2g3 g8f6 f1g2 b8c6 b1c3 f8c5 e2e3 a7a6 g1e2 d7d6 e1g1 e8g8",
    "c2c4 c7c5 b1c3 b8c6 g2g3 g7g6 f1g2 f8g7 g1f3 g8f6 e1g1 e8g8 d2d4 c5d4 f3d4",
    "c2c4 c7c5 g1f3 g8f6 b1c3 b8c6 g2g3 g7g6 f1g2 f8g7 e1g1 e8g8 d2d4 c5d4 f3d4 c6d4 d1d4 d7d6",
    "c2c4 c7c5 g1f3 g8f6 b1c3 e7e6 g2g3 b7b6 f1g2 c8b7 e1g1 f8e7 d2d4 c5d4 d1d4 d7d6 f1d1 a7a6 e2e4 b8d7",
    "c2c4 g8f6 b1c3 e7e6 e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 f8b4 d4c6 b7c6 f1d3 e8g8 e1g1",
    "c2c4 g8f6 g2g3 e7e6 f1g2 d7d5 g1f3 f8e7 e1g1 e8g8 d2d4 d5c4 d1c2 a7a6 a2a4",
    "c2c4 g8f6 b1c3 e7e5 g1f3 b8c6 g2g3 d7d5 c4d5 f6d5 f1g2 d5b6 e1g1 f8e7 d2d3 e8g8",
    "c2c4 e7e6 g1f3 d7d5 g2g3 g8f6 f1g2 f8e7 e1g1 e8g8 d2d4 c7c6 b1c3 b8d7",
    "c2c4 c7c6 g1f3 d7d5 g2g3 g8f6 f1g2 e7e6 e1g1 f8e7 d2d4 e8g8 d1c2 b8d7",
    "c2c4 g7g6 g2g3 f8g7 f1g2 g8f6 b1c3 e8g8 e2e4 d7d6 g1e2 e7e5 e1g1 b8c6 d2d3",
    "g1f3 d7d5 c2c4 e7e6 g2g3 g8f6 f1g2 f8e7 e1g1 e8g8 d2d4 c7c6 d1c2 b8d7 b1d2",
    "g1f3 d7d5 c2c4 d5c4 e2e3 g8f6 f1c4 e7e6 e1g1 c7c5 d2d4 a7a6 d1e2 b7b5 c4d3",
    "g1f3 d7d5 g2g3 g8f6 f1g2 c7c6 e1g1 c8f5 d2d3 e7e6 b1d2 f8e7 d1e1 e8g8 e2e4",
    "g1f3 d7d5 g2g3 c7c6 f1g2 g8f6 e1g1 c8g4 d2d3 b8d7 b1d2 e7e5 e2e4 f8d6 h2h3 g4h5",
    "g1f3 g8f6 g2g3 g7g6 f1g2 f8g7 e1g1 e8g8 d2d3 d7d5 b1d2 c7c5 e2e4 b8c6 c2c3",
    "g1f3 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 d2d4 e8g8 f1e2 e7e5 e1g1 b8c6 d4d5",
    "g1f3 g8f6 c2c4 e7e6 g2g3 d7d5 f1g2 f8e7 e1g1 e8g8 d2d4 c7c6 d1c2 b8d7 b1d2",
    "g1f3 g8f6 g2g3 d7d5 f1g2 c7c5 e1g1 b8c6 d2d4 e7e6 c2c4 d5c4 d1a4",
    "b2b3 e7e5 c1b2 b8c6 e2e3 g8f6 f1b5 d7d6 g1e2 c8d7 e1g1 f8e7 c2c4 e8g8",
    "g2g3 d7d5 f1g2 g8f6 g1f3 c7c6 e1g1 c8f5 d2d3 e7e6 b1d2 f8e7 e2e4",
    # Ruy Lopez, extra depth (Berlin, Closed, Marshall, Archangelsk, Exchange)
    "e2e4 e7e5 g1f3 b8c6 f1b5 g8f6 e1g1 f6e4 d2d4 e4d6 b5c6 d7c6 d4e5 d6f5 d1d8 e8d8",
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 e8g8 c2c3 d7d6 h2h3 c6a5 b3c2 c7c5 d2d4",
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 e8g8 c2c3 d7d5",
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 b7b5 a4b3 c8b7",
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5c6 d7c6 e1g1 f7f6 d2d4",
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8c5 c2c3 e8g8 d2d4",
    # Italian / Two Knights / Evans
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d3 d7d6 e1g1 e8g8",
    "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 f3g5 d7d5 e4d5 c6a5 c4b5 c7c6 d5c6 b7c6",
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 b2b4 c5b4 c2c3 b4a5 d2d4",
    "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d3 f8c5 c2c3 d7d6 e1g1 e8g8",
    # Sicilian, extra (Sveshnikov, Najdorf English, Accelerated, Rossolimo, Kan)
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 e7e5 d4b5 d7d6 c1g5 a7a6 b5a3 b7b5",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1e3 e7e5 d4b3 c8e6 f2f3 f8e7 d1d2 e8g8",
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g7g6 c2c4 f8g7 c1e3 g8f6 b1c3 e8g8",
    "e2e4 c7c5 g1f3 b8c6 f1b5 g7g6 e1g1 f8g7 f1e1",
    "e2e4 c7c5 g1f3 d7d6 f1b5 c8d7 b5d7 d8d7",
    "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 a7a6 c2c4 g8f6 b1c3 d8c7",
    "e2e4 c7c5 b1c3 b8c6 g2g3 g7g6 f1g2 f8g7 d2d3 d7d6",
    # French / Caro extra
    "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4 e4e5 c7c5 a2a3 b4c3 b2c3 g8e7",
    "e2e4 e7e6 d2d4 d7d5 b1d2 g8f6 e4e5 f6d7 f1d3 c7c5 c2c3 b8c6",
    "e2e4 e7e6 d2d4 d7d5 e4e5 c7c5 c2c3 b8c6 g1f3 d8b6 a2a3",
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 c8f5 e4g3 f5g6 h2h4 h7h6 g1f3 b8d7",
    "e2e4 c7c6 d2d4 d7d5 e4d5 c6d5 c2c4 g8f6 b1c3 e7e6",
    "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5 g1f3 e7e6 f1e2 c6c5",
    # QGD / Slav / Semi-Slav / Catalan
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 g1f3 b8d7 a1c1 c7c6",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 c1g5 f8e7 e2e3 e8g8 g1f3 h7h6 g5h4 b7b6",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 g1f3 f8b4",
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 e7e6 e2e3 b8d7 f1d3 d5c4 d3c4 b7b5",
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 d5c4 a2a4 c8f5 e2e3 e7e6 f1c4",
    "d2d4 g8f6 c2c4 e7e6 g2g3 d7d5 f1g2 d5c4 g1f3 f8e7 e1g1",
    "d2d4 g8f6 c2c4 e7e6 g2g3 d7d5 f1g2 f8e7 g1f3 e8g8 e1g1",
    # Nimzo / KID / Grunfeld
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3 e8g8 f1d3 d7d5 g1f3",
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 d1c2 e8g8 a2a3 b4c3 c2c3",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8 f1e2 e7e5 e1g1 b8c6 d4d5 c6e7",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 f2f3 e8g8 c1e3",
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5 f6d5 e2e4 d5c3 b2c3 f8g7 f1c4",
    "d2d4 g8f6 c2c4 g7g6 g1f3 f8g7 g2g3 e8g8 f1g2 d7d6 e1g1 b8d7",
    "d2d4 g8f6 c2c4 c7c5 d4d5 e7e6 b1c3 e6d5 c4d5 d7d6 e2e4 g7g6",
    # London / Jobava / Trompowsky / Colle
    "d2d4 d7d5 g1f3 g8f6 c1f4 e7e6 e2e3 f8d6 f4g3",
    "d2d4 g8f6 c1f4 d7d5 e2e3 e7e6 g1f3",
    "d2d4 g8f6 b1c3 d7d5 c1f4",
    "d2d4 g8f6 c1g5 e7e6 e2e4 h7h6 g5h4",
    "d2d4 d7d5 g1f3 g8f6 e2e3 e7e6 f1d3 c7c5",
    # English extra
    "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6 g2g3 d7d5 c4d5 f6d5 f1g2 d5b6",
    "c2c4 c7c5 g1f3 g8f6 b1c3 b8c6 g2g3 g7g6 f1g2 f8g7 e1g1 e8g8",
    "c2c4 e7e5 b1c3 g8f6 g2g3 d7d5 c4d5 f6d5 f1g2 d5b6 e2e3 f8e7 g1e2 e8g8 e1g1",
    # Extra mainlines for curated near-equal starts
    "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6 e5f3 f6e4 d2d4 d6d5 f1d3 b8c6 e1g1 f8e7",
    "e2e4 e7e5 g1f3 g8f6 d2d4 f6e4 f1d3 d7d5 f3e5 b8d7",
    "e2e4 e7e5 g1f3 b8c6 b1c3 g8f6 f1b5 f8b4 e1g1 e8g8 d2d3 d7d6",
    "e2e4 e7e5 g1f3 b8c6 b1c3 g8f6 d2d4 e5d4 f3d4 f8b4 d4c6 b7c6 f1d3",
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 g8f6 d4c6 b7c6 e4e5 d8e7 d1e2 f6d5",
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 f8c5 c1e3 d8f6 c2c3 g8e7",
    "e2e4 e7e5 b1c3 g8f6 f2f4 d7d5 f4e5 f6e4 d1f3",
    "e2e4 e7e5 b1c3 b8c6 f1c4 g8f6 d2d3 f8c5 g1f3 d7d6",
    "e2e4 e7e5 d2d4 e5d4 d1d4 b8c6 d4e3 g8f6 b1c3 f8b4",
    "e2e4 e7e5 g1f3 d7d6 d2d4 b8d7 f1c4 c7c6 e1g1 f8e7",
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8e7 d2d3 g8f6 e1g1 e8g8",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 g7g6 c1e3 f8g7 f2f3 e8g8 d1d2 b8c6 e1c1",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1g5 e7e6 d1d2 f8e7 e1c1",
    "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 b8c6 b1c3 a7a6 c1e3 g8f6 f1d3",
    "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 g8f6 b1c3 d7d6 f1e2 f8e7 e1g1 e8g8",
    "e2e4 c7c5 b1c3 b8c6 g2g3 g7g6 f1g2 f8g7 d2d3 d7d6 f2f4 e7e6 g1f3 g8e7",
    "e2e4 c7c5 g1f3 b8c6 f1b5 g7g6 e1g1 f8g7 f1e1 e7e5 c2c3 g8e7",
    "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4 e4e5 c7c5 a2a3 b4c3 b2c3 g8e7 d1g4",
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 c1g5 d5e4 c3e4 f8e7 g5f6 e7f6 g1f3",
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4 b8d7 g1f3 g8f6 e4f6 d7f6",
    "e2e4 c7c6 b1c3 d7d5 g1f3 c8g4 h2h3 g4f3 d1f3 e7e6",
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6 c1e3 f8g7 d1d2 e8g8 e1c1",
    "e2e4 g8f6 e4e5 f6d5 d2d4 d7d6 c2c4 d5b6 f2f4 d6e5 f4e5 b8c6",
    "e2e4 d7d5 e4d5 g8f6 d2d4 f6d5 c2c4 d5b6",
    "d2d4 d7d5 c2c4 d5c4 e2e3 e7e5 f1c4 e5d4 e3d4 g8f6 g1f3",
    "d2d4 d7d5 c2c4 e7e6 b1c3 c7c5 c4d5 e6d5 g1f3 b8c6 g2g3 g8f6 f1g2",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6 g1f3 f8b4 c1g5 h7h6 g5h4",
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 e7e6 e2e3 b8d7 d1c2 f8d6",
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 a7a6 e2e3 b7b5",
    "d2d4 g8f6 c2c4 c7c5 d4d5 b7b5 c4b5 a7a6 b5a6 e7e6",
    "d2d4 g8f6 c2c4 c7c5 d4d5 e7e6 b1c3 e6d5 c4d5 d7d6 e2e4 g7g6 g1f3 f8g7",
    "d2d4 f7f5 g2g3 g8f6 f1g2 g7g6 g1f3 f8g7 e1g1 e8g8 c2c4 d7d6",
    "d2d4 f7f5 c2c4 g8f6 g1f3 e7e6 g2g3 f8e7 f1g2 e8g8 e1g1 d7d6",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 f2f3 e8g8 c1e3 e7e5 d4d5",
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 g1f3 f8g7 d1b3 d5c4 b3c4 e8g8",
    "d2d4 g8f6 c2c4 e7e6 g2g3 d7d5 f1g2 f8e7 g1f3 e8g8 e1g1 b8d7 d1c2",
    "d2d4 d7d5 g1f3 g8f6 c1g5 e7e6 e2e3 c7c5 b1d2 b8c6 c2c3",
    "d2d4 g8f6 c1g5 e7e6 e2e4 h7h6 g5f6 d8f6",
    "d2d4 g8f6 b1c3 d7d5 c1g5 b8d7 e2e3 g7g6",
    "c2c4 e7e5 b1c3 g8f6 g2g3 f8b4 f1g2 e8g8 e2e4 b4c3 d2c3",
    "c2c4 c7c5 g1f3 g8f6 d2d4 c5d4 f3d4 e7e6 g2g3 b8c6 f1g2",
    "c2c4 g8f6 b1c3 e7e6 e2e4 d7d5 e4e5 f6d7 d2d4 c7c5",
    "g1f3 d7d5 g2g3 c7c5 f1g2 b8c6 e1g1 e7e5 d2d3 g8f6",
    "g1f3 g8f6 g2g3 g7g6 f1g2 f8g7 e1g1 e8g8 d2d3 d7d6 e2e4 e7e5",
    "g1f3 e7e6 g2g3 d7d5 f1g2 g8f6 e1g1 f8e7 d2d3 e8g8 b1d2 c7c5",
    "b2b4 e7e5 c1b2 b8c6 b4b5 c6d4 e2e3 d4e6",
    "f2f4 d7d5 g1f3 g8f6 e2e3 g7g6 f1e2 f8g7",
    # Black replies to first moves, plus common second-move branches
    "e2e4 e7e5",
    "e2e4 c7c5",
    "e2e4 e7e6",
    "e2e4 c7c6",
    "e2e4 d7d6",
    "e2e4 g7g6",
    "e2e4 d7d5",
    "e2e4 g8f6",
    "d2d4 d7d5",
    "d2d4 g8f6",
    "d2d4 e7e6",
    "d2d4 f7f5",
    "c2c4 e7e5",
    "c2c4 c7c5",
    "c2c4 g8f6",
    "g1f3 g8f6",
    "g1f3 d7d5",
    "b2b3 e7e5",
    "g2g3 d7d5",
)

Book = dict[int, dict[int, int]]  # hash -> {packed_move: weight}

_PACK_PROMO = {None: 0, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4, chess.QUEEN: 5}


def pack_move(move: chess.Move) -> int:
    return move.from_square | (move.to_square << 6) | (_PACK_PROMO.get(move.promotion, 0) << 12)


def unpack_move(packed: int) -> chess.Move:
    promo_n = (packed >> 12) & 7
    promo = promo_n if promo_n else None
    return chess.Move(packed & 63, (packed >> 6) & 63, promo)


def _build() -> Book:
    book: Book = {}
    for line in _LINES:
        board = chess.Board()
        for token in line.split():
            move = chess.Move.from_uci(token)
            if move not in board.legal_moves:
                break
            key = chess.polyglot.zobrist_hash(board)
            packed = pack_move(move)
            slot = book.setdefault(key, {})
            slot[packed] = slot.get(packed, 0) + 1
            board.push(move)
    return book


BOOK: Book = _build()


def probe(board: chess.Board) -> chess.Move | None:
    """Return a book move, or None. Weighted among stored replies."""
    entries = BOOK.get(chess.polyglot.zobrist_hash(board))
    if not entries:
        return None
    # Deterministic: highest weight, then lowest packed value (stable).
    packed = max(entries.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    move = unpack_move(packed)
    if move in board.legal_moves:
        return move
    return None

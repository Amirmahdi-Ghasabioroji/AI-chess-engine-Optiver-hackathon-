"""KXK draws and mop-up."""

from __future__ import annotations

from engine.pos import Position


def insufficient_material(pos: Position) -> bool:
    if pos.bb[0] | pos.bb[6]:
        return False
    if pos.bb[3] | pos.bb[4] | pos.bb[9] | pos.bb[10]:
        return False
    wn, wb = pos.bb[1].bit_count(), pos.bb[2].bit_count()
    bn, bb = pos.bb[7].bit_count(), pos.bb[8].bit_count()
    minors_w, minors_b = wn + wb, bn + bb
    if minors_w + minors_b <= 1:
        return True
    return minors_w == 1 and minors_b == 1 and wb + bb == 0

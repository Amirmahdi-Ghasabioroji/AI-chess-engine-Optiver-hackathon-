"""Static exchange evaluation."""

from __future__ import annotations

from engine.attacks import attackers_to
from engine.const import FLAG_EP, FLAG_PROMO, SEE_VAL, WHITE, m_from, m_promo, m_to
from engine.pos import Position


def see(pos: Position, move: int) -> int:
    frm, to, promo = m_from(move), m_to(move), m_promo(move)
    us, them = pos.side, pos.side ^ 1
    piece = pos.sq[frm] % 6
    occ = pos.all ^ (1 << frm)
    color = [pos.occ[0], pos.occ[1]]
    color[us] ^= 1 << frm
    if move & FLAG_EP:
        cap_sq = to - 8 if us == WHITE else to + 8
        occ ^= 1 << cap_sq
        color[them] ^= 1 << cap_sq
        gain0 = SEE_VAL[0]
    else:
        cap = pos.sq[to]
        if cap >= 0:
            occ ^= 1 << to
            color[them] ^= 1 << to
            gain0 = SEE_VAL[cap % 6]
        else:
            gain0 = 0
    if move & FLAG_PROMO:
        gain0 += SEE_VAL[promo] - SEE_VAL[0]
        piece = promo
    occ |= 1 << to
    color[us] |= 1 << to
    swap = [gain0]
    stm = them
    att = attackers_to(to, occ, pos.bb) & occ
    while True:
        mine = att & color[stm]
        if not mine:
            break
        lva_sq, lva_pt = _lva(pos, mine)
        if lva_pt == 5 and (att & color[stm ^ 1]):
            break
        occ ^= 1 << lva_sq
        color[stm] ^= 1 << lva_sq
        att = attackers_to(to, occ, pos.bb) & occ
        swap.append(SEE_VAL[piece] - swap[-1])
        piece = lva_pt
        stm ^= 1
        if len(swap) > 32:
            break
    for i in range(len(swap) - 2, -1, -1):
        swap[i] = min(-swap[i + 1], swap[i])
    return swap[0]


def _lva(pos: Position, mine: int) -> tuple[int, int]:
    bb = pos.bb
    for pt in range(6):
        sub = (bb[pt] | bb[pt + 6]) & mine
        if sub:
            return (sub & -sub).bit_length() - 1, pt
    return -1, 0

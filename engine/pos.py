"""Persistent bitboard position: FEN, make/unmake, incremental hash and PST."""

from __future__ import annotations

from engine.attacks import KING_ATT, KNIGHT_ATT, PAWN_ATT, bishop_attacks, rook_attacks
from engine.const import (
    BKC,
    BLACK,
    BQC,
    CASTLE_MASK,
    CHAR_FROM_PIECE,
    FLAG_CASTLE,
    FLAG_DOUBLE,
    FLAG_EP,
    KING,
    MASK64,
    NAME_TO_SQ,
    PAWN,
    PHASE_PC,
    PIECE_FROM_CHAR,
    PST_EG,
    PST_MG,
    SQ_NAMES,
    START_FEN,
    WHITE,
    WKC,
    WQC,
    m_from,
    m_promo,
    m_to,
)
from engine.zobrist import Z_CASTLE, Z_EP, Z_PIECE, Z_SIDE


class Position:
    __slots__ = (
        "all",
        "bb",
        "castle",
        "ep",
        "fullmove",
        "halfmove",
        "hash",
        "hist",
        "king",
        "occ",
        "phase",
        "psq_eg",
        "psq_mg",
        "side",
        "sq",
        "undo",
    )

    def __init__(self) -> None:
        self.bb = [0] * 12
        self.occ = [0, 0]
        self.all = 0
        self.sq = [-1] * 64
        self.side = WHITE
        self.ep = -1
        self.castle = 0
        self.halfmove = 0
        self.fullmove = 1
        self.hash = 0
        self.king = [0, 0]
        self.psq_mg = 0
        self.psq_eg = 0
        self.phase = 0
        self.undo: list[tuple] = []
        self.hist: list[int] = []

    @classmethod
    def from_fen(cls, fen: str) -> Position:
        pos = cls()
        pos.set_fen(fen)
        return pos

    @classmethod
    def start(cls) -> Position:
        return cls.from_fen(START_FEN)

    def copy(self) -> Position:
        pos = Position()
        pos.bb = self.bb[:]
        pos.occ = self.occ[:]
        pos.all = self.all
        pos.sq = self.sq[:]
        pos.side = self.side
        pos.ep = self.ep
        pos.castle = self.castle
        pos.halfmove = self.halfmove
        pos.fullmove = self.fullmove
        pos.hash = self.hash
        pos.king = self.king[:]
        pos.psq_mg = self.psq_mg
        pos.psq_eg = self.psq_eg
        pos.phase = self.phase
        pos.hist = self.hist[:]
        return pos

    def set_fen(self, fen: str) -> None:
        parts = fen.split()
        placement = parts[0]
        stm = parts[1] if len(parts) > 1 else "w"
        castle = parts[2] if len(parts) > 2 else "-"
        ep = parts[3] if len(parts) > 3 else "-"
        half = parts[4] if len(parts) > 4 else "0"
        full = parts[5] if len(parts) > 5 else "1"
        self.bb = [0] * 12
        self.occ = [0, 0]
        self.all = 0
        self.sq = [-1] * 64
        self.psq_mg = self.psq_eg = self.phase = 0
        self.hash = 0
        self.undo.clear()
        rank, file = 7, 0
        for ch in placement:
            if ch == "/":
                rank -= 1
                file = 0
            elif ch.isdigit():
                file += int(ch)
            else:
                self._place(rank * 8 + file, PIECE_FROM_CHAR[ch])
                file += 1
        self.side = WHITE if stm == "w" else BLACK
        self.castle = 0
        if "K" in castle:
            self.castle |= WKC
        if "Q" in castle:
            self.castle |= WQC
        if "k" in castle:
            self.castle |= BKC
        if "q" in castle:
            self.castle |= BQC
        self.ep = -1 if ep == "-" else NAME_TO_SQ[ep]
        self.halfmove = int(half)
        self.fullmove = int(full)
        self.king[WHITE] = _king_sq(self, WHITE)
        self.king[BLACK] = _king_sq(self, BLACK)
        self.hash = self.compute_hash()
        self.hist = [self.hash]

    def fen(self) -> str:
        rows = []
        for rank in range(7, -1, -1):
            empty = 0
            row = ""
            for file in range(8):
                p = self.sq[rank * 8 + file]
                if p < 0:
                    empty += 1
                else:
                    if empty:
                        row += str(empty)
                        empty = 0
                    row += CHAR_FROM_PIECE[p]
            if empty:
                row += str(empty)
            rows.append(row)
        stm = "w" if self.side == WHITE else "b"
        c = ""
        if self.castle & WKC:
            c += "K"
        if self.castle & WQC:
            c += "Q"
        if self.castle & BKC:
            c += "k"
        if self.castle & BQC:
            c += "q"
        if not c:
            c = "-"
        ep = "-" if self.ep < 0 else SQ_NAMES[self.ep]
        return f"{'/'.join(rows)} {stm} {c} {ep} {self.halfmove} {self.fullmove}"

    def compute_hash(self) -> int:
        h = 0
        for piece in range(12):
            bb = self.bb[piece]
            while bb:
                sq = (bb & -bb).bit_length() - 1
                h ^= Z_PIECE[piece][sq]
                bb &= bb - 1
        if self.side == BLACK:
            h ^= Z_SIDE
        h ^= Z_CASTLE[self.castle]
        if self.ep >= 0:
            h ^= Z_EP[self.ep & 7]
        return h

    def _place(self, sq: int, piece: int) -> None:
        bit = 1 << sq
        self.bb[piece] |= bit
        self.occ[piece // 6] |= bit
        self.all |= bit
        self.sq[sq] = piece
        self.hash ^= Z_PIECE[piece][sq]
        self.psq_mg += PST_MG[piece][sq]
        self.psq_eg += PST_EG[piece][sq]
        self.phase += PHASE_PC[piece]

    def _remove(self, sq: int, piece: int) -> None:
        bit = 1 << sq
        self.bb[piece] &= ~bit
        self.occ[piece // 6] &= ~bit
        self.all &= ~bit
        self.sq[sq] = -1
        self.hash ^= Z_PIECE[piece][sq]
        self.psq_mg -= PST_MG[piece][sq]
        self.psq_eg -= PST_EG[piece][sq]
        self.phase -= PHASE_PC[piece]

    def is_attacked(self, sq: int, by: int) -> bool:
        bb = self.bb
        occ = self.all
        if by == WHITE:
            return bool(
                (PAWN_ATT[BLACK][sq] & bb[0])
                or (KNIGHT_ATT[sq] & bb[1])
                or (KING_ATT[sq] & bb[5])
                or (bishop_attacks(sq, occ) & (bb[2] | bb[4]))
                or (rook_attacks(sq, occ) & (bb[3] | bb[4]))
            )
        return bool(
            (PAWN_ATT[WHITE][sq] & bb[6])
            or (KNIGHT_ATT[sq] & bb[7])
            or (KING_ATT[sq] & bb[11])
            or (bishop_attacks(sq, occ) & (bb[8] | bb[10]))
            or (rook_attacks(sq, occ) & (bb[9] | bb[10]))
        )

    def in_check(self, side: int | None = None) -> bool:
        if side is None:
            side = self.side
        return self.is_attacked(self.king[side], side ^ 1)

    def _push_undo(self, move: int, cap_piece: int, cap_sq: int) -> None:
        self.undo.append(
            (
                move,
                cap_piece,
                cap_sq,
                self.ep,
                self.castle,
                self.halfmove,
                self.hash,
                self.psq_mg,
                self.psq_eg,
                self.phase,
                self.fullmove,
                self.king[0],
                self.king[1],
            )
        )

    def make(self, move: int) -> None:
        if move == 0:
            self._make_null()
            return
        frm, to, promo = m_from(move), m_to(move), m_promo(move)
        piece = self.sq[frm]
        cap_piece = self.sq[to]
        cap_sq = to
        if move & FLAG_EP:
            cap_sq = to - 8 if self.side == WHITE else to + 8
            cap_piece = self.sq[cap_sq]
        self._push_undo(move, cap_piece, cap_sq)
        us = self.side
        if self.ep >= 0:
            self.hash ^= Z_EP[self.ep & 7]
            self.ep = -1
        self.hash ^= Z_CASTLE[self.castle]
        self._remove(frm, piece)
        if cap_piece >= 0:
            self._remove(cap_sq, cap_piece)
            self.halfmove = 0
        else:
            self.halfmove += 1
        if piece % 6 == PAWN:
            self.halfmove = 0
            if move & FLAG_DOUBLE:
                self.ep = (frm + to) // 2
                self.hash ^= Z_EP[self.ep & 7]
            if promo:
                piece = us * 6 + promo
        if move & FLAG_CASTLE:
            if to == 6:
                self._remove(7, 3)
                self._place(5, 3)
            elif to == 2:
                self._remove(0, 3)
                self._place(3, 3)
            elif to == 62:
                self._remove(63, 9)
                self._place(61, 9)
            elif to == 58:
                self._remove(56, 9)
                self._place(59, 9)
        self._place(to, piece)
        if piece % 6 == KING:
            self.king[us] = to
        self.castle &= CASTLE_MASK[frm] & CASTLE_MASK[to]
        if cap_sq != to:
            self.castle &= CASTLE_MASK[cap_sq]
        self.hash ^= Z_CASTLE[self.castle]
        self.side ^= 1
        self.hash ^= Z_SIDE
        if us == BLACK:
            self.fullmove += 1
        self.hist.append(self.hash)

    def unmake(self) -> None:
        rec = self.undo.pop()
        self.hist.pop()
        move = rec[0]
        if move == 0:
            _, _, _, ep, castle, halfmove, h, psq_mg, psq_eg, phase, fullmove, k0, k1 = rec
            self.side ^= 1
            self.ep = ep
            self.castle = castle
            self.halfmove = halfmove
            self.hash = h
            self.psq_mg = psq_mg
            self.psq_eg = psq_eg
            self.phase = phase
            self.fullmove = fullmove
            self.king[0], self.king[1] = k0, k1
            return
        move, cap_piece, cap_sq, ep, castle, halfmove, h, psq_mg, psq_eg, phase, fullmove, k0, k1 = rec
        self.side ^= 1
        us = self.side
        frm, to, promo = m_from(move), m_to(move), m_promo(move)
        piece = self.sq[to]
        if promo:
            self._remove(to, piece)
            self._place(frm, us * 6)
        else:
            self._remove(to, piece)
            self._place(frm, piece)
        if move & FLAG_CASTLE:
            if to == 6:
                self._remove(5, 3)
                self._place(7, 3)
            elif to == 2:
                self._remove(3, 3)
                self._place(0, 3)
            elif to == 62:
                self._remove(61, 9)
                self._place(63, 9)
            elif to == 58:
                self._remove(59, 9)
                self._place(56, 9)
        if cap_piece >= 0:
            self._place(cap_sq, cap_piece)
        self.ep = ep
        self.castle = castle
        self.halfmove = halfmove
        self.hash = h
        self.psq_mg = psq_mg
        self.psq_eg = psq_eg
        self.phase = phase
        self.fullmove = fullmove
        self.king[0], self.king[1] = k0, k1

    def _make_null(self) -> None:
        self._push_undo(0, -1, -1)
        if self.ep >= 0:
            self.hash ^= Z_EP[self.ep & 7]
            self.ep = -1
        self.side ^= 1
        self.hash ^= Z_SIDE
        self.halfmove += 1
        self.hist.append(self.hash)

    def is_repeat(self) -> bool:
        if self.halfmove < 4:
            return False
        key = self.hash
        hist = self.hist
        start = max(0, len(hist) - self.halfmove - 1)
        for i in range(len(hist) - 3, start - 1, -2):
            if hist[i] == key:
                return True
        return False

    def has_non_pawn(self, side: int) -> bool:
        king_pawn = self.bb[side * 6] | self.bb[side * 6 + KING]
        return (self.occ[side] & ~king_pawn & MASK64) != 0

    def snapshot(self) -> tuple:
        return (
            tuple(self.bb),
            tuple(self.occ),
            self.all,
            tuple(self.sq),
            self.side,
            self.ep,
            self.castle,
            self.halfmove,
            self.fullmove,
            self.hash,
            tuple(self.king),
            self.psq_mg,
            self.psq_eg,
            self.phase,
        )


def _king_sq(pos: Position, side: int) -> int:
    bb = pos.bb[side * 6 + KING]
    return (bb & -bb).bit_length() - 1 if bb else 0

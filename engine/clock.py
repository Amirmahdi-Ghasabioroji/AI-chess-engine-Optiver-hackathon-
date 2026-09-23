"""Time allocation. The platform never reports increment; do not spend it."""

from __future__ import annotations

SAFETY_MS = 90


def allocate(time_left_ms: int, ply: int = 0, complexity: float = 1.0) -> tuple[int, int]:
    """Return (soft_ms, hard_ms). Increment is reserve, not budget."""
    usable = max(1, time_left_ms - SAFETY_MS)
    if usable <= 90:
        return 12, min(35, usable)
    if usable <= 700:
        hard = min(70, usable)
        return min(35, hard), hard

    if ply < 16:
        moves_to_go = 36
    elif ply < 40:
        moves_to_go = 28
    elif ply < 80:
        moves_to_go = 20
    else:
        moves_to_go = 14

    soft = usable // moves_to_go
    soft = min(soft, usable // 8)
    soft = max(soft, 20)
    soft = int(soft * min(1.35, max(0.85, complexity)))
    hard = min(int(soft * 1.45), usable // 6, usable)
    hard = max(hard, soft)
    return soft, hard

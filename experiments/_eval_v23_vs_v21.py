"""Print v2.3 mix modes and v2.1 classical on the same STM positions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def main() -> int:
    print("=== v2.3 eval (cp, STM) ===", flush=True)
    first = subprocess.run([PY, str(ROOT / "_eval_v23_snapshot.py")], cwd=str(ROOT.parent))
    if first.returncode:
        return first.returncode
    print("=== v2.1 classical (cp, STM) ===", flush=True)
    second = subprocess.run([PY, str(ROOT / "_eval_v21_snapshot.py")], cwd=str(ROOT.parent))
    return second.returncode


if __name__ == "__main__":
    raise SystemExit(main())

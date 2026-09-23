"""Build submission.zip via the official harness packer (root *.py, agent.py at zip root)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.package import DEFAULT_INCLUDES, build  # noqa: E402
from harness.rules import MAX_UNZIPPED_BYTES  # noqa: E402


def main() -> int:
    out = ROOT / "submission.zip"
    written = build(ROOT, out, DEFAULT_INCLUDES + ("engine",))
    size = out.stat().st_size
    unzipped = sum((ROOT / name).stat().st_size for name in written if (ROOT / name).is_file())
    print(f"{out} ({size:,} bytes, {unzipped:,} unzipped)")
    for name in written:
        print(f"  {name}")
    if unzipped > MAX_UNZIPPED_BYTES:
        print("OVER 50 MB unzipped", file=sys.stderr)
        return 1
    if "agent.py" not in written:
        print("agent.py missing from zip", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""One-shot layout migrate. Safe to re-run. Do not keep this in the repo long-term."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"
ARCHIVE = ROOT / "archive"
DOCS = ROOT / "docs"

ENGINE.mkdir(exist_ok=True)
ARCHIVE.mkdir(exist_ok=True)
DOCS.mkdir(exist_ok=True)

MOVES = [
    (ROOT / "bb.py", ENGINE / "bb.py"),
    (ROOT / "bb_eval.py", ENGINE / "bb_eval.py"),
    (ROOT / "bb_search.py", ENGINE / "bb_search.py"),
    (ROOT / "bbpos.py", ENGINE / "bbpos.py"),
    (ROOT / "book.py", ENGINE / "book.py"),
    (ROOT / "evaluate.py", ENGINE / "evaluate.py"),
    (ROOT / "nnue.py", ENGINE / "nnue.py"),
    (ROOT / "search.py", ARCHIVE / "search.py"),
    (ROOT / "DESIGN.md", DOCS / "DESIGN.md"),
    (ROOT / "NEXT_ENGINE.md", DOCS / "NEXT_ENGINE.md"),
]


def _patch(path: Path, replacements: list[tuple[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    for old, new in replacements:
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    for src, dst in MOVES:
        if src.is_file() and src.resolve() != dst.resolve():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"moved {src.name} -> {dst.relative_to(ROOT)}")
        elif dst.is_file():
            print(f"already at {dst.relative_to(ROOT)}")
        else:
            print(f"missing {src}")

    _patch(
        ENGINE / "bbpos.py",
        [("\nimport bb\n", "\nfrom . import bb\n")],
    )
    _patch(
        ENGINE / "nnue.py",
        [
            ("\nimport bb\n", "\nfrom . import bb\n"),
            (
                'WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"',
                'WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"',
            ),
        ],
    )
    _patch(
        ENGINE / "bb_eval.py",
        [
            ("\nimport bb\n", "\nfrom . import bb\n"),
            ("from bb import (", "from .bb import ("),
            ("from evaluate import (", "from .evaluate import ("),
            (
                'RESIDUAL_PATH = Path(__file__).resolve().parent / "weights" / "pst.npz"',
                'RESIDUAL_PATH = Path(__file__).resolve().parent.parent / "weights" / "pst.npz"',
            ),
        ],
    )
    _patch(
        ENGINE / "bb_search.py",
        [
            ("\nimport bb\n", "\nfrom . import bb\n"),
            ("from bb import (", "from .bb import ("),
            ("from bb_eval import ", "from .bb_eval import "),
        ],
    )
    print("patched engine imports")


if __name__ == "__main__":
    main()

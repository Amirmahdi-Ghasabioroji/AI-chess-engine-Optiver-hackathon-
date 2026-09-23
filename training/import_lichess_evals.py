"""Stream Lichess evals (.jsonl.zst) into an STM feature npz for v7.1.

Lichess cloud scores are White-pov; labels are flipped to side-to-move.

    python -m training.import_lichess_evals --max-positions 2500000
    python -m training.import_lichess_evals --offset 60 --out data/lichess/nnue_lichess_b.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zstandard as zstd

from training.v71_nnue_data import MAX_PIECES, pack_fen_stm

DEFAULT_SRC = Path("data/lichess/lichess_db_eval.jsonl.zst")
DEFAULT_OUT = Path("data/lichess/nnue_lichess.npz")
CLIP = 2000.0


def iter_strided_lines(path: Path, stride: int, offset: int = 0):
    """Yield lines where `scanned % stride == offset` without copying the buffer per line."""
    residue = offset % stride
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as fh, dctx.stream_reader(fh) as reader:
        buf = bytearray()
        pos = 0
        scanned = 0
        while True:
            chunk = reader.read(1 << 20)
            if not chunk:
                break
            buf.extend(chunk)
            while True:
                idx = buf.find(b"\n", pos)
                if idx < 0:
                    if pos:
                        del buf[:pos]
                        pos = 0
                    break
                scanned += 1
                if scanned % stride == residue:
                    yield bytes(buf[pos:idx]), scanned
                pos = idx + 1
        rest = bytes(buf[pos:]).strip()
        if rest:
            scanned += 1
            if scanned % stride == residue:
                yield rest, scanned


def best_cp(obj: dict, min_depth: int) -> int | None:
    evals = obj.get("evals") or []
    best = None
    best_depth = -1
    for ev in evals:
        depth = int(ev.get("depth") or 0)
        if depth < min_depth or depth < best_depth:
            continue
        pvs = ev.get("pvs") or []
        if not pvs:
            continue
        pv = pvs[0]
        if pv.get("mate") is not None:
            continue
        cp = pv.get("cp")
        if cp is None:
            continue
        best = int(cp)
        best_depth = depth
    return best


def stm_label(fen: str, white_cp: int) -> int | None:
    parts = fen.split()
    if len(parts) < 2:
        return None
    if parts[1] == "w":
        return white_cp
    if parts[1] == "b":
        return -white_cp
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stride", type=int, default=120)
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Keep lines with scanned %% stride == offset. Use 60 with stride 120 for the complementary 2.5M.",
    )
    parser.add_argument("--max-positions", type=int, default=2_500_000)
    parser.add_argument("--min-depth", type=int, default=16)
    parser.add_argument("--max-cp", type=float, default=1500.0)
    arguments = parser.parse_args()

    if not arguments.src.is_file():
        print(f"missing {arguments.src}", file=sys.stderr)
        return 1

    features = np.empty((arguments.max_positions, MAX_PIECES), dtype=np.int32)
    labels = np.empty(arguments.max_positions, dtype=np.float32)
    kept = 0
    scanned = 0
    parsed = 0
    skipped = 0
    print(
        f"reading {arguments.src} stride={arguments.stride} offset={arguments.offset} "
        f"min_depth={arguments.min_depth} cap={arguments.max_positions}",
        flush=True,
    )
    for raw, scanned in iter_strided_lines(
        arguments.src, arguments.stride, arguments.offset
    ):
        parsed += 1
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            skipped += 1
            continue
        fen = obj.get("fen")
        if not isinstance(fen, str):
            skipped += 1
            continue
        white_cp = best_cp(obj, arguments.min_depth)
        if white_cp is None:
            skipped += 1
            continue
        cp = stm_label(fen, white_cp)
        if cp is None or abs(cp) > arguments.max_cp:
            skipped += 1
            continue
        row = pack_fen_stm(fen)
        if row is None:
            skipped += 1
            continue
        features[kept] = row
        labels[kept] = float(np.clip(cp, -CLIP, CLIP))
        kept += 1
        if kept % 50000 == 0:
            print(
                f"  kept {kept}  parsed {parsed}  scanned {scanned}  skipped {skipped}",
                flush=True,
            )
        if kept >= arguments.max_positions:
            break

    print(
        f"done kept={kept} parsed={parsed} scanned={scanned} skipped={skipped}",
        flush=True,
    )
    if kept < 1000:
        print("too few positions", file=sys.stderr)
        return 1
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.out,
        features=features[:kept],
        sf=labels[:kept],
        packed_stm=np.int8(1),
        min_depth=np.int32(arguments.min_depth),
        stride=np.int32(arguments.stride),
        offset=np.int32(arguments.offset),
    )
    print(f"wrote {arguments.out} n={kept}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

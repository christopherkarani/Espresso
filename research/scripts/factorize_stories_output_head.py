#!/usr/bin/env python3
"""Factorize an Espresso Llama-family output head into projection/expansion BLOBFILE weights."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.espresso_llama_weights import BLOBFILE_HEADER_BYTES, load_espresso_metadata, read_blobfile_array


def make_blobfile(path: Path, values: np.ndarray) -> None:
    payload = values.astype(np.float16, copy=False).tobytes()
    header = bytearray(BLOBFILE_HEADER_BYTES)
    header[0] = 0x01
    header[4] = 0x02
    header[64:68] = bytes([0xEF, 0xBE, 0xAD, 0xDE])
    header[68] = 0x01
    struct.pack_into("<I", header, 72, len(payload))
    struct.pack_into("<I", header, 80, BLOBFILE_HEADER_BYTES)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header) + payload)


def factorize_output_head(
    weights_dir: Path,
    rank: int,
    projection_out: Path,
    expansion_out: Path,
    stats_out: Path | None = None,
) -> dict[str, float | int]:
    metadata = load_espresso_metadata(weights_dir)
    if rank <= 0 or rank > metadata.d_model:
        raise ValueError(f"rank must be in 1..{metadata.d_model}")

    lm_head = read_blobfile_array(
        weights_dir / "lm_head.bin",
        (metadata.vocab, metadata.d_model),
    ).astype(np.float32, copy=False)
    u, s, vt = np.linalg.svd(lm_head, full_matrices=False)
    u_rank = u[:, :rank]
    s_rank = s[:rank]
    vt_rank = vt[:rank, :]

    expansion = u_rank * s_rank[np.newaxis, :]
    projection = vt_rank
    reconstruction = expansion @ projection

    residual = lm_head - reconstruction
    lm_head_norm = float(np.linalg.norm(lm_head))
    residual_norm = float(np.linalg.norm(residual))
    relative_fro_error = residual_norm / max(lm_head_norm, 1e-12)

    make_blobfile(projection_out, projection)
    make_blobfile(expansion_out, expansion)

    stats: dict[str, float | int] = {
        "rank": rank,
        "vocab": int(metadata.vocab),
        "d_model": int(metadata.d_model),
        "relative_fro_error": relative_fro_error,
        "lm_head_norm": lm_head_norm,
        "residual_norm": residual_norm,
    }
    if stats_out is not None:
        stats_out.parent.mkdir(parents=True, exist_ok=True)
        stats_out.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--projection-out", type=Path, required=True)
    parser.add_argument("--expansion-out", type=Path, required=True)
    parser.add_argument("--stats-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = factorize_output_head(
        weights_dir=args.weights_dir.expanduser().resolve(),
        rank=args.rank,
        projection_out=args.projection_out.expanduser().resolve(),
        expansion_out=args.expansion_out.expanduser().resolve(),
        stats_out=args.stats_out.expanduser().resolve() if args.stats_out else None,
    )
    print(json.dumps(stats, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

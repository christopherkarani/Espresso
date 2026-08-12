#!/usr/bin/env python3
"""Export a GPT-2 Core ML trunk for Espresso compare baselines.

Invoked by EspressoGenerate as:

    export_gpt2_coreml.py \\
        --weights <espresso-gpt2-dir> \\
        --output <gpt2_seqN.mlpackage> \\
        --seq-len N

The package accepts `input_ids` with shape `[1, seq_len]` (int32) and returns
`hidden_states` with shape `[1, seq_len, hidden]` (float16) — hidden states
*before* the final LayerNorm / LM head. Espresso's native Core ML runner applies
final norm + classifier from the Espresso weight blobs.

Requires: numpy, torch, transformers, coremltools (macOS recommended)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        required=True,
        help="Espresso GPT-2 weights directory containing metadata.json",
    )
    parser.add_argument("--output", required=True, help="Output .mlpackage path")
    parser.add_argument("--seq-len", required=True, type=int, help="Fixed sequence length")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional Hugging Face model id/path (default: inferred from metadata name)",
    )
    parser.add_argument(
        "--minimum-target",
        default="macOS15",
        choices=["macOS15"],
        help="Minimum deployment target for the converted model",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory",
    )
    return parser.parse_args(argv)


def infer_hf_model_id(metadata_name: str) -> str:
    normalized = metadata_name.lower()
    if normalized in {"gpt2_124m", "gpt2", "gpt2-small"}:
        return "gpt2"
    if "gpt2-medium" in normalized or normalized.endswith("_355m"):
        return "gpt2-medium"
    if "gpt2-large" in normalized or normalized.endswith("_774m"):
        return "gpt2-large"
    if "gpt2-xl" in normalized or normalized.endswith("_1558m"):
        return "gpt2-xl"
    return "gpt2"


def _build_trunk_class() -> Any:
    import torch
    from torch import nn

    class GPT2TrunkBeforeNorm(nn.Module):
        """GPT-2 transformer stack without final LayerNorm / LM head."""

        def __init__(self, hf_model: nn.Module, seq_len: int):
            super().__init__()
            transformer = hf_model.transformer
            self.wte = transformer.wte
            self.wpe = transformer.wpe
            self.drop = transformer.drop
            self.h = transformer.h
            self.seq_len = seq_len
            position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
            self.register_buffer("position_ids", position_ids, persistent=False)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            input_ids = input_ids.long()
            hidden_states = self.wte(input_ids) + self.wpe(self.position_ids.to(input_ids.device))
            hidden_states = self.drop(hidden_states)
            for block in self.h:
                hidden_states = block(hidden_states)[0]
            return hidden_states

    return GPT2TrunkBeforeNorm


def minimum_target(name: str):
    import coremltools as ct

    if name != "macOS15":
        raise ValueError(f"Unsupported minimum target: {name}")
    return ct.target.macOS15


def export_gpt2_coreml(
    *,
    weights_dir: pathlib.Path,
    output_path: pathlib.Path,
    seq_len: int,
    model_id: str | None = None,
    cache_dir: str | None = None,
    deployment_target: str = "macOS15",
) -> pathlib.Path:
    if seq_len <= 0:
        raise ValueError("--seq-len must be > 0")

    import coremltools as ct
    import numpy as np
    import torch
    from transformers import GPT2LMHeadModel

    GPT2TrunkBeforeNorm = _build_trunk_class()

    weights_dir = pathlib.Path(weights_dir).expanduser().resolve()
    metadata_path = weights_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata.json in {weights_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hf_model_id = model_id or infer_hf_model_id(str(metadata.get("name", "gpt2_124m")))

    print(f"Loading {hf_model_id} for Core ML export (seq_len={seq_len})", flush=True)
    hf_model = GPT2LMHeadModel.from_pretrained(
        hf_model_id,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
    )
    hf_model.eval()

    trunk = GPT2TrunkBeforeNorm(hf_model, seq_len)
    trunk.eval()
    example = torch.zeros((1, seq_len), dtype=torch.int32)

    with torch.no_grad():
        traced = torch.jit.trace(trunk, example, strict=False)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=example.shape, dtype=example.numpy().dtype)],
        outputs=[ct.TensorType(name="hidden_states", dtype=np.float16)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=minimum_target(deployment_target),
    )

    output_path = pathlib.Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if output_path.is_dir():
            import shutil

            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    mlmodel.save(str(output_path))
    print(output_path, flush=True)
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        export_gpt2_coreml(
            weights_dir=pathlib.Path(args.weights),
            output_path=pathlib.Path(args.output),
            seq_len=args.seq_len,
            model_id=args.model,
            cache_dir=args.cache_dir,
            deployment_target=args.minimum_target,
        )
    except Exception as exc:  # pragma: no cover - surfaced to espresso-generate
        print(f"export_gpt2_coreml failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

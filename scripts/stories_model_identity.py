#!/usr/bin/env python3
"""Prove or disprove Stories native-vs-HF model identity.

This script compares the canonical native Espresso Stories checkpoint
(`stories110M.bin`) against the cached Hugging Face `Xenova/llama2.c-stories110M`
snapshot and also compares tokenizer behavior on a fixed prompt suite.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_HF_MODEL = "Xenova/llama2.c-stories110M"
DEFAULT_PROMPTS = [
    "Hello",
    "Hello, world!\n",
    "The quick brown fox jumps over the lazy dog near the river",
    (
        "In a distant future where artificial intelligence has become deeply "
        "integrated into every aspect of daily life, researchers at a small "
        "university lab made an unexpected discovery"
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NATIVE_TOKENIZER_DIR = (
    Path.home() / "Library/Application Support/Espresso/demo/stories110m"
)
DEFAULT_NATIVE_CANDIDATES = [
    REPO_ROOT / "assets/models/stories110M.bin",
]


@dataclass(frozen=True)
class NativeHeader:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int


@dataclass(frozen=True)
class TensorComparison:
    name: str
    matches_exactly: bool
    max_abs_diff: float
    mean_abs_diff: float
    shape: list[int]


class EspressoSentencePieceTokenizer:
    def __init__(self, model_path: Path) -> None:
        data = model_path.read_bytes()
        if len(data) < 4:
            raise ValueError(f"truncated tokenizer model: {model_path}")

        cursor = 0
        max_token_length = struct.unpack_from("<i", data, cursor)[0]
        cursor += 4
        if max_token_length <= 0:
            raise ValueError(f"invalid max token length {max_token_length} in {model_path}")

        pieces: list[str] = []
        scores: list[float] = []
        piece_to_id: dict[str, int] = {}

        while cursor < len(data):
            if cursor + 8 > len(data):
                raise ValueError(f"truncated tokenizer entry at byte {cursor}")
            score = struct.unpack_from("<f", data, cursor)[0]
            cursor += 4
            length = struct.unpack_from("<i", data, cursor)[0]
            cursor += 4
            if length < 0 or cursor + length > len(data):
                raise ValueError(f"truncated tokenizer entry payload at byte {cursor}")
            piece = data[cursor : cursor + length].decode("utf-8")
            cursor += length
            piece_to_id[piece] = len(pieces)
            pieces.append(piece)
            scores.append(score)

        if not pieces:
            raise ValueError(f"empty tokenizer model: {model_path}")

        self.max_token_length = max_token_length
        self.pieces = pieces
        self.scores = scores
        self.piece_to_id = piece_to_id

    def encode(self, text: str) -> list[int]:
        normalized = "▁" + text.replace(" ", "▁")
        tokens: list[int] = []

        for character in normalized:
            piece = character
            token_id = self.piece_to_id.get(piece)
            if token_id is not None:
                tokens.append(token_id)
                continue

            for byte in piece.encode("utf-8"):
                fallback = f"<0x{byte:02X}>"
                fallback_id = self.piece_to_id.get(fallback)
                if fallback_id is not None:
                    tokens.append(fallback_id)

        while len(tokens) >= 2:
            best_score = float("-inf")
            best_index: int | None = None
            best_id: int | None = None

            for index in range(len(tokens) - 1):
                merged = self.pieces[tokens[index]] + self.pieces[tokens[index + 1]]
                if len(merged.encode("utf-8")) > self.max_token_length:
                    continue
                merged_id = self.piece_to_id.get(merged)
                if merged_id is None:
                    continue
                score = self.scores[merged_id]
                if score > best_score:
                    best_score = score
                    best_index = index
                    best_id = merged_id

            if best_index is None or best_id is None:
                break

            tokens[best_index] = best_id
            del tokens[best_index + 1]

        return tokens

    def decode(self, tokens: Iterable[int]) -> str:
        bytes_accumulator = bytearray()
        text_parts: list[str] = []

        for token in tokens:
            if token < 0 or token >= len(self.pieces):
                continue
            piece = self.pieces[token]
            byte_value = _byte_token_value(piece)
            if byte_value is not None:
                bytes_accumulator.append(byte_value)
                continue
            if bytes_accumulator:
                text_parts.append(bytes(bytes_accumulator).decode("utf-8"))
                bytes_accumulator.clear()
            text_parts.append(piece)

        if bytes_accumulator:
            text_parts.append(bytes(bytes_accumulator).decode("utf-8"))

        text = "".join(text_parts).replace("▁", " ")
        if text.startswith(" "):
            return text[1:]
        return text


def _byte_token_value(piece: str) -> int | None:
    if piece.startswith("<0x") and piece.endswith(">") and len(piece) == 6:
        return int(piece[3:5], 16)
    return None


def resolve_native_stories_path(explicit_path: str | None = None, env: dict[str, str] | None = None) -> Path:
    env = env or os.environ
    candidates: list[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    env_path = env.get("STORIES_MODEL_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend(DEFAULT_NATIVE_CANDIDATES)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    searched = [str(candidate) for candidate in candidates]
    raise FileNotFoundError(
        "Unable to locate native Stories checkpoint. Tried: " + ", ".join(searched)
    )


def resolve_native_tokenizer_dir(explicit_path: str | None = None) -> Path:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    candidates.append(DEFAULT_NATIVE_TOKENIZER_DIR)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    searched = [str(candidate) for candidate in candidates]
    raise FileNotFoundError(
        "Unable to locate native Stories tokenizer directory. Tried: " + ", ".join(searched)
    )


def resolve_hf_snapshot(model_ref: str) -> Path:
    path = Path(model_ref).expanduser()
    if path.exists():
        return path.resolve()

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_ref, local_files_only=True)).resolve()


def read_native_header(path: Path) -> NativeHeader:
    with path.open("rb") as handle:
        raw = handle.read(28)
    if len(raw) != 28:
        raise ValueError(f"truncated native checkpoint header: {path}")
    values = struct.unpack("<7i", raw)
    return NativeHeader(
        dim=values[0],
        hidden_dim=values[1],
        n_layers=values[2],
        n_heads=values[3],
        n_kv_heads=values[4],
        vocab_size=values[5],
        seq_len=values[6],
    )


def native_layout(header: NativeHeader) -> list[tuple[str, int]]:
    vocab = abs(header.vocab_size)
    dim = header.dim
    hidden_dim = header.hidden_dim
    n_layers = header.n_layers

    layout: list[tuple[str, int]] = [("embed", vocab * dim)]
    per_layer_segments = [
        ("rms_att", dim),
        ("wq", dim * dim),
        ("wk", dim * dim),
        ("wv", dim * dim),
        ("wo", dim * dim),
        ("rms_ffn", dim),
        ("w1", hidden_dim * dim),
        ("w2", dim * hidden_dim),
        ("w3", hidden_dim * dim),
    ]
    for name, count in per_layer_segments:
        layout.append((name, count * n_layers))
    layout.append(("rms_final", dim))
    if header.vocab_size < 0:
        layout.append(("wcls", vocab * dim))
    return layout


def load_native_checkpoint(path: Path) -> tuple[NativeHeader, dict[str, object]]:
    import numpy as np

    header = read_native_header(path)
    vocab = abs(header.vocab_size)
    dim = header.dim
    hidden_dim = header.hidden_dim

    weights: dict[str, object] = {}
    with path.open("rb") as handle:
        handle.seek(28)

        weights["model.embed_tokens.weight"] = np.fromfile(handle, dtype="<f4", count=vocab * dim).reshape(vocab, dim)

        for key, count, shape in [
            ("input_layernorm.weight", dim, (dim,)),
            ("self_attn.q_proj.weight", dim * dim, (dim, dim)),
            ("self_attn.k_proj.weight", dim * dim, (dim, dim)),
            ("self_attn.v_proj.weight", dim * dim, (dim, dim)),
            ("self_attn.o_proj.weight", dim * dim, (dim, dim)),
            ("post_attention_layernorm.weight", dim, (dim,)),
            ("mlp.gate_proj.weight", hidden_dim * dim, (hidden_dim, dim)),
            ("mlp.down_proj.weight", dim * hidden_dim, (dim, hidden_dim)),
            ("mlp.up_proj.weight", hidden_dim * dim, (hidden_dim, dim)),
        ]:
            stacked = np.fromfile(handle, dtype="<f4", count=count * header.n_layers)
            if stacked.size != count * header.n_layers:
                raise ValueError(f"truncated native segment for {key}")
            per_layer = stacked.reshape(header.n_layers, *shape)
            for layer_index in range(header.n_layers):
                weights[f"model.layers.{layer_index}.{key}"] = per_layer[layer_index]

        weights["model.norm.weight"] = np.fromfile(handle, dtype="<f4", count=dim)
        if weights["model.norm.weight"].size != dim:
            raise ValueError("truncated native final norm")

        if header.vocab_size < 0:
            classifier = np.fromfile(handle, dtype="<f4", count=vocab * dim).reshape(vocab, dim)
            if classifier.size != vocab * dim:
                raise ValueError("truncated native classifier")
            weights["lm_head.weight"] = classifier
        else:
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]

        trailing = handle.read(1)
        if trailing:
            raise ValueError(f"unexpected trailing bytes in native checkpoint: {path}")

    return header, weights


def load_hf_snapshot(snapshot_dir: Path):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot_dir),
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    model.eval()
    state_dict = {
        name: tensor.detach().cpu().to(torch.float32).numpy()
        for name, tensor in model.state_dict().items()
    }
    return model.config, state_dict


def compare_native_and_hf(
    header: NativeHeader,
    native_weights: dict[str, object],
    hf_config,
    hf_state_dict: dict[str, object],
) -> dict[str, object]:
    import numpy as np

    header_matches = {
        "dim": header.dim == getattr(hf_config, "hidden_size"),
        "hidden_dim": header.hidden_dim == getattr(hf_config, "intermediate_size"),
        "n_layers": header.n_layers == getattr(hf_config, "num_hidden_layers"),
        "n_heads": header.n_heads == getattr(hf_config, "num_attention_heads"),
        "n_kv_heads": header.n_kv_heads == getattr(hf_config, "num_key_value_heads"),
        "vocab_size": abs(header.vocab_size) == getattr(hf_config, "vocab_size"),
        "rope_theta": float(getattr(hf_config, "rope_theta", 10_000.0)) == 10_000.0,
    }

    comparisons: list[TensorComparison] = []
    for name, native in native_weights.items():
        hf = hf_state_dict.get(name)
        if hf is None:
            raise KeyError(f"HF snapshot missing tensor {name}")
        native_array = np.asarray(native, dtype=np.float32)
        hf_array = np.asarray(hf, dtype=np.float32)
        if native_array.shape != hf_array.shape:
            raise ValueError(f"shape mismatch for {name}: native {native_array.shape} vs hf {hf_array.shape}")
        diff = np.abs(native_array - hf_array)
        comparisons.append(
            TensorComparison(
                name=name,
                matches_exactly=bool(np.array_equal(native_array, hf_array)),
                max_abs_diff=float(diff.max(initial=0.0)),
                mean_abs_diff=float(diff.mean() if diff.size else 0.0),
                shape=list(native_array.shape),
            )
        )

    exact_tensors = [comparison for comparison in comparisons if comparison.matches_exactly]
    mismatched_tensors = [comparison for comparison in comparisons if not comparison.matches_exactly]
    first_mismatch = mismatched_tensors[0] if mismatched_tensors else None

    return {
        "header_matches": header_matches,
        "shared_classifier": header.vocab_size > 0,
        "tensor_count": len(comparisons),
        "exact_tensor_count": len(exact_tensors),
        "mismatch_tensor_count": len(mismatched_tensors),
        "all_tensors_exact": len(mismatched_tensors) == 0,
        "first_mismatch": _comparison_payload(first_mismatch) if first_mismatch else None,
    }


def _comparison_payload(comparison: TensorComparison | None) -> dict[str, object] | None:
    if comparison is None:
        return None
    return {
        "name": comparison.name,
        "matches_exactly": comparison.matches_exactly,
        "max_abs_diff": comparison.max_abs_diff,
        "mean_abs_diff": comparison.mean_abs_diff,
        "shape": comparison.shape,
    }


def compare_tokenizers(
    native_tokenizer_dir: Path,
    hf_snapshot_dir: Path,
    prompts: list[str],
) -> dict[str, object]:
    from transformers import AutoTokenizer

    native_model_path = native_tokenizer_dir / "tokenizer.model"
    if not native_model_path.exists():
        raise FileNotFoundError(f"missing Espresso tokenizer.model: {native_model_path}")
    native_tokenizer = EspressoSentencePieceTokenizer(native_model_path)
    hf_tokenizer = AutoTokenizer.from_pretrained(str(hf_snapshot_dir), use_fast=False, local_files_only=True)

    prompt_reports = []
    all_token_ids_match = True
    all_decoded_text_match = True

    for prompt in prompts:
        native_ids = native_tokenizer.encode(prompt)
        hf_ids = hf_tokenizer.encode(prompt, add_special_tokens=False)
        native_decoded = native_tokenizer.decode(native_ids)
        hf_decoded = hf_tokenizer.decode(
            hf_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        token_ids_match = native_ids == hf_ids
        decoded_text_match = native_decoded == hf_decoded
        all_token_ids_match &= token_ids_match
        all_decoded_text_match &= decoded_text_match
        prompt_reports.append(
            {
                "prompt": prompt,
                "native_token_ids": native_ids,
                "hf_token_ids": hf_ids,
                "token_ids_match": token_ids_match,
                "native_decoded": native_decoded,
                "hf_decoded": hf_decoded,
                "decoded_text_match": decoded_text_match,
            }
        )

    return {
        "all_token_ids_match": all_token_ids_match,
        "all_decoded_text_match": all_decoded_text_match,
        "prompts": prompt_reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-model", help="Path to native Espresso stories110M.bin")
    parser.add_argument(
        "--native-tokenizer-dir",
        help="Path to the Espresso Stories tokenizer directory (defaults to Application Support)",
    )
    parser.add_argument(
        "--hf-model",
        default=DEFAULT_HF_MODEL,
        help="Hugging Face repo id or local snapshot path for the reference model",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="Additional prompt to compare. Defaults to the fixed Stories suite when omitted.",
    )
    parser.add_argument(
        "--json-indent",
        type=int,
        default=2,
        help="Indentation for JSON output (default: 2).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    native_model_path = resolve_native_stories_path(args.native_model)
    native_tokenizer_dir = resolve_native_tokenizer_dir(args.native_tokenizer_dir)
    hf_snapshot_dir = resolve_hf_snapshot(args.hf_model)
    prompts = args.prompts or DEFAULT_PROMPTS

    header, native_weights = load_native_checkpoint(native_model_path)
    hf_config, hf_state_dict = load_hf_snapshot(hf_snapshot_dir)

    payload = {
        "native_model_path": str(native_model_path),
        "native_tokenizer_dir": str(native_tokenizer_dir),
        "hf_snapshot_dir": str(hf_snapshot_dir),
        "native_header": {
            "dim": header.dim,
            "hidden_dim": header.hidden_dim,
            "n_layers": header.n_layers,
            "n_heads": header.n_heads,
            "n_kv_heads": header.n_kv_heads,
            "vocab_size": header.vocab_size,
            "seq_len": header.seq_len,
        },
        "hf_config": {
            "hidden_size": getattr(hf_config, "hidden_size"),
            "intermediate_size": getattr(hf_config, "intermediate_size"),
            "num_hidden_layers": getattr(hf_config, "num_hidden_layers"),
            "num_attention_heads": getattr(hf_config, "num_attention_heads"),
            "num_key_value_heads": getattr(hf_config, "num_key_value_heads"),
            "vocab_size": getattr(hf_config, "vocab_size"),
            "max_position_embeddings": getattr(hf_config, "max_position_embeddings"),
            "rope_theta": float(getattr(hf_config, "rope_theta", 10_000.0)),
        },
        "weight_identity": compare_native_and_hf(header, native_weights, hf_config, hf_state_dict),
        "tokenizer_identity": compare_tokenizers(native_tokenizer_dir, hf_snapshot_dir, prompts),
    }

    print(json.dumps(payload, indent=args.json_indent, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""No-network contract tests for the Qwen2.5 converter."""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import convert_qwen25_05b_to_esp as script


def official_shape(**overrides) -> script.QwenShape:
    fields = dict(
        name="Qwen2.5-0.5B-Instruct",
        n_layer=24,
        n_head=14,
        n_kv_head=2,
        d_model=896,
        head_dim=64,
        hidden_dim=4864,
        vocab=151936,
        norm_eps=1e-6,
        rope_theta=1_000_000.0,
        eos_token=151645,
        tie_word_embeddings=True,
        max_position_embeddings=32768,
    )
    fields.update(overrides)
    return script.QwenShape(**fields)


def official_15b_shape(**overrides) -> script.QwenShape:
    fields = dict(
        name="Qwen2.5-1.5B-Instruct",
        n_layer=28,
        n_head=12,
        n_kv_head=2,
        d_model=1536,
        head_dim=128,
        hidden_dim=8960,
    )
    fields.update(overrides)
    return official_shape(**fields)


def write_empty(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def write_safetensors(path: Path, tensors: dict[str, bytes], shapes: dict[str, list[int]]) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    offset = 0
    for name, data in tensors.items():
        header[name] = {
            "dtype": "F32",
            "shape": shapes[name],
            "data_offsets": [offset, offset + len(data)],
        }
        payload.extend(data)
        offset += len(data)
    header_bytes = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + bytes(payload))


class OfficialQwen25ShapeTests(unittest.TestCase):
    def test_official_constants_match_committed_table(self) -> None:
        self.assertEqual(
            script.OFFICIAL_QWEN25_05B_INSTRUCT,
            {
                "n_layer": 24,
                "n_head": 14,
                "n_kv_head": 2,
                "d_model": 896,
                "head_dim": 64,
                "hidden_dim": 4864,
                "vocab": 151936,
            },
        )
        self.assertEqual(script.OFFICIAL_QWEN25_05B_ROPE_THETA, 1_000_000.0)
        self.assertEqual(script.OFFICIAL_QWEN25_05B_ROPE_THETA, 1e6)

    def test_official_15b_constants_match_expected_shape(self) -> None:
        self.assertEqual(
            script.OFFICIAL_QWEN25_15B_INSTRUCT,
            {
                "n_layer": 28,
                "n_head": 12,
                "n_kv_head": 2,
                "d_model": 1536,
                "head_dim": 128,
                "hidden_dim": 8960,
                "vocab": 151936,
            },
        )
        self.assertEqual(script.OFFICIAL_QWEN25_15B_ROPE_THETA, 1_000_000.0)

    def test_validate_shape_accepts_official_qwen25_05b_instruct(self) -> None:
        script.validate_shape(official_shape())
        script.validate_shape(official_shape(rope_theta=1e6))
        script.validate_shape(official_shape(rope_theta=1000000.0))

    def test_validate_shape_accepts_official_qwen25_15b_instruct(self) -> None:
        script.validate_shape(official_15b_shape())
        script.validate_shape(official_15b_shape(rope_theta=1e6))

    def test_validate_shape_rejects_other_qwen2_widths(self) -> None:
        with self.assertRaises(script.ConversionError) as caught:
            script.validate_shape(official_shape(n_layer=28, d_model=1024, hidden_dim=2816))
        message = str(caught.exception)
        self.assertIn("Qwen2.5-0.5B-Instruct", message)
        self.assertIn("n_layer=28", message)
        self.assertIn("d_model=1024", message)

    def test_validate_shape_rejects_05b_widths_on_15b_identity(self) -> None:
        with self.assertRaises(script.ConversionError) as caught:
            script.validate_shape(official_15b_shape(n_layer=24, d_model=896, hidden_dim=4864))
        message = str(caught.exception)
        self.assertIn("Qwen2.5-1.5B-Instruct", message)
        self.assertIn("n_layer=24", message)

    def test_validate_shape_rejects_llama_default_rope_theta(self) -> None:
        with self.assertRaises(script.ConversionError) as caught:
            script.validate_shape(official_shape(rope_theta=10_000.0))
        self.assertIn("rope_theta", str(caught.exception))

    def test_validate_layout_rejects_mismatched_attention_width(self) -> None:
        with self.assertRaises(script.ConversionError) as caught:
            script.validate_layout(official_shape(d_model=1024))
        self.assertIn("dModel == nHead * headDim", str(caught.exception))

    def test_validate_layout_rejects_ungrouped_kv_heads(self) -> None:
        with self.assertRaises(script.ConversionError) as caught:
            script.validate_layout(official_shape(n_kv_head=3))
        self.assertIn("nHead % nKVHead", str(caught.exception))

    def test_validate_layout_rejects_non_64_aligned_channels(self) -> None:
        for label, kwargs in (
            ("dModel", {"n_head": 7, "n_kv_head": 1, "head_dim": 60, "d_model": 420}),
            ("kvDim", {"n_kv_head": 1, "head_dim": 96, "d_model": 1344, "n_head": 14}),
            ("hiddenDim", {"hidden_dim": 4865}),
        ):
            with self.subTest(label=label):
                with self.assertRaises(script.ConversionError) as caught:
                    script.validate_layout(official_shape(**kwargs))
                message = str(caught.exception)
                self.assertIn(label, message)
                self.assertIn("multiple of 64", message)

    def test_write_metadata_stamps_hybrid_preferred_decode_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            metadata = script.write_metadata(output_dir, official_shape(), max_seq=1024)
            on_disk = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["preferredDecodePath"], "hybrid")
            self.assertEqual(on_disk["preferredDecodePath"], "hybrid")
            self.assertEqual(on_disk["nLayer"], 24)
            self.assertEqual(on_disk["nHead"], 14)
            self.assertEqual(on_disk["nKVHead"], 2)
            self.assertEqual(on_disk["dModel"], 896)
            self.assertEqual(on_disk["headDim"], 64)
            self.assertEqual(on_disk["hiddenDim"], 4864)
            self.assertEqual(on_disk["vocab"], 151936)
            self.assertEqual(on_disk["ropeTheta"], 1_000_000.0)
            self.assertEqual(on_disk["architecture"], "llama")
            self.assertEqual(on_disk["maxSeq"], 1024)

    def test_write_metadata_stamps_15b_hybrid_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            metadata = script.write_metadata(output_dir, official_15b_shape(), max_seq=1024)
            on_disk = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["preferredDecodePath"], "hybrid")
            self.assertEqual(on_disk["preferredDecodePath"], "hybrid")
            self.assertEqual(on_disk["name"], "Qwen2.5-1.5B-Instruct")
            self.assertEqual(on_disk["nLayer"], 28)
            self.assertEqual(on_disk["nHead"], 12)
            self.assertEqual(on_disk["nKVHead"], 2)
            self.assertEqual(on_disk["dModel"], 1536)
            self.assertEqual(on_disk["headDim"], 128)
            self.assertEqual(on_disk["hiddenDim"], 8960)
            self.assertEqual(on_disk["vocab"], 151936)
            self.assertEqual(on_disk["ropeTheta"], 1_000_000.0)
            self.assertEqual(on_disk["architecture"], "llama")
            self.assertEqual(on_disk["maxSeq"], 1024)


class Qwen25CLITests(unittest.TestCase):
    def test_default_model_is_05b_and_accepts_15b(self) -> None:
        default = script.parse_args([])
        self.assertEqual(default.model, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(default.max_seq, 1024)
        fifteen = script.parse_args(["--model", "Qwen/Qwen2.5-1.5B-Instruct"])
        self.assertEqual(fifteen.model, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(fifteen.max_seq, 1024)

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            script.parse_args(["--model", "Qwen/Qwen2.5-7B-Instruct"])

    def test_default_paths_stay_on_05b_slug(self) -> None:
        cache = Path("/tmp/espresso-cache")
        paths = script.default_paths(script.SUPPORTED_MODELS["Qwen/Qwen2.5-0.5B-Instruct"], cache)
        self.assertEqual(paths.bundle, cache / "qwen25-05b" / "Qwen2.5-0.5B-Instruct.esp")
        self.assertEqual(paths.native, cache / "qwen25-05b" / "Qwen2.5-0.5B-Instruct-native")
        self.assertEqual(paths.source, cache / "qwen25-05b-src")

    def test_default_15b_paths_use_qwen25_15b_slug(self) -> None:
        cache = Path("/tmp/espresso-cache")
        paths = script.default_paths(script.SUPPORTED_MODELS["Qwen/Qwen2.5-1.5B-Instruct"], cache)
        self.assertEqual(paths.bundle, cache / "qwen25-15b" / "Qwen2.5-1.5B-Instruct.esp")
        self.assertEqual(paths.native, cache / "qwen25-15b" / "Qwen2.5-1.5B-Instruct-native")
        self.assertEqual(paths.source, cache / "qwen25-15b-src")


class HuggingFaceSnapshotTests(unittest.TestCase):
    def test_resolve_source_prefers_complete_hub_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hub = root / "hub"
            cache = root / "espresso-cache"
            snap = hub / "models--Qwen--Qwen2.5-1.5B-Instruct" / "snapshots" / "abc123"
            write_empty(snap / "config.json")
            write_empty(snap / "tokenizer.json")
            write_empty(snap / "model.safetensors")
            ref = hub / "models--Qwen--Qwen2.5-1.5B-Instruct" / "refs" / "main"
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text("abc123\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}, clear=False):
                resolved = script.resolve_source_dir(
                    repo="Qwen/Qwen2.5-1.5B-Instruct",
                    explicit=None,
                    cache_root=cache,
                    source_slug="qwen25-15b-src",
                    force_download=False,
                )
            self.assertEqual(resolved, snap)

    def test_resolve_source_skips_incomplete_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hub = root / "hub"
            cache = root / "espresso-cache"
            snap = hub / "models--Qwen--Qwen2.5-1.5B-Instruct" / "snapshots" / "abc123"
            write_empty(snap / "config.json")
            ref = hub / "models--Qwen--Qwen2.5-1.5B-Instruct" / "refs" / "main"
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text("abc123\n", encoding="utf-8")

            with mock.patch.dict(os.environ, {"HF_HUB_CACHE": str(hub)}, clear=False):
                resolved = script.resolve_source_dir(
                    repo="Qwen/Qwen2.5-1.5B-Instruct",
                    explicit=None,
                    cache_root=cache,
                    source_slug="qwen25-15b-src",
                    force_download=False,
                )
            self.assertEqual(resolved, cache / "qwen25-15b-src")

    def test_snapshot_complete_when_shards_match_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            write_empty(source / "config.json")
            write_empty(source / "tokenizer.json")
            write_empty(source / "model-00001-of-00002.safetensors")
            write_empty(source / "model-00002-of-00002.safetensors")
            (source / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "a": "model-00001-of-00002.safetensors",
                            "b": "model-00002-of-00002.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(script.snapshot_is_complete(source))


class SafetensorsStoreTests(unittest.TestCase):
    def test_open_source_reads_single_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            write_safetensors(
                source / "model.safetensors",
                {"tok.weight": b"\x00\x00\x80\x3f"},
                {"tok.weight": [1]},
            )
            store = script.SafetensorsStore.open_source(source)
            self.assertEqual(store.tensor_names(), ["tok.weight"])

    def test_open_source_reads_sharded_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            write_safetensors(
                source / "model-00001-of-00002.safetensors",
                {"layer.0.weight": b"\x00\x00\x80\x3f"},
                {"layer.0.weight": [1]},
            )
            write_safetensors(
                source / "model-00002-of-00002.safetensors",
                {"layer.1.weight": b"\x00\x00\x00\x40"},
                {"layer.1.weight": [1]},
            )
            (source / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "layer.0.weight": "model-00001-of-00002.safetensors",
                            "layer.1.weight": "model-00002-of-00002.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            store = script.SafetensorsStore.open_source(source)
            self.assertEqual(store.tensor_names(), ["layer.0.weight", "layer.1.weight"])
            self.assertEqual(
                store.owners["layer.0.weight"].path.name,
                "model-00001-of-00002.safetensors",
            )
            self.assertEqual(
                store.owners["layer.1.weight"].path.name,
                "model-00002-of-00002.safetensors",
            )

    def test_open_source_requires_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(script.ConversionError) as caught:
                script.SafetensorsStore.open_source(Path(directory))
            self.assertIn("model.safetensors", str(caught.exception))


if __name__ == "__main__":
    unittest.main()

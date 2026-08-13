#!/usr/bin/env python3
"""No-network contract tests for the Qwen2.5-0.5B-Instruct converter."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


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

    def test_validate_shape_accepts_official_qwen25_05b_instruct(self) -> None:
        script.validate_shape(official_shape())
        script.validate_shape(official_shape(rope_theta=1e6))
        script.validate_shape(official_shape(rope_theta=1000000.0))

    def test_validate_shape_rejects_other_qwen2_widths(self) -> None:
        with self.assertRaises(script.ConversionError) as caught:
            script.validate_shape(official_shape(n_layer=28, d_model=1024, hidden_dim=2816))
        message = str(caught.exception)
        self.assertIn("Qwen2.5-0.5B-Instruct", message)
        self.assertIn("n_layer=28", message)
        self.assertIn("d_model=1024", message)

    def test_validate_shape_rejects_llama_default_rope_theta(self) -> None:
        with self.assertRaises(script.ConversionError) as caught:
            script.validate_shape(official_shape(rope_theta=10_000.0))
        self.assertIn("rope_theta", str(caught.exception))

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


if __name__ == "__main__":
    unittest.main()

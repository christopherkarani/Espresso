#!/usr/bin/env python3
"""No-network contract tests for the Qwen2.5 PyTorch reference runner."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import convert_qwen25_05b_to_esp as convert
from scripts import qwen25_pytorch_reference as script


class Qwen25ReferenceCLITests(unittest.TestCase):
    def test_default_model_is_05b_and_accepts_15b(self) -> None:
        default = script.parse_args(["layer-parity"])
        self.assertEqual(default.model, "Qwen/Qwen2.5-0.5B-Instruct")
        fifteen = script.parse_args(
            ["--model", "Qwen/Qwen2.5-1.5B-Instruct", "layer-parity"]
        )
        self.assertEqual(fifteen.model, "Qwen/Qwen2.5-1.5B-Instruct")

    def test_unknown_model_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            script.parse_args(["--model", "Qwen/Qwen2.5-7B-Instruct", "layer-parity"])

    def test_15b_model_defaults_source_and_native_dirs(self) -> None:
        cache = Path("/tmp/espresso-cache")
        with mock.patch.object(script, "espresso_cache_root", return_value=cache):
            source, native = script.resolved_reference_paths(
                model="Qwen/Qwen2.5-1.5B-Instruct",
                source_dir=None,
                native_dir=None,
                force_hub_lookup=False,
            )
        self.assertEqual(native, cache / "qwen25-15b" / "Qwen2.5-1.5B-Instruct-native")
        self.assertEqual(source, cache / "qwen25-15b-src")

    def test_05b_model_keeps_existing_default_paths(self) -> None:
        cache = Path("/tmp/espresso-cache")
        with mock.patch.object(script, "espresso_cache_root", return_value=cache):
            source, native = script.resolved_reference_paths(
                model="Qwen/Qwen2.5-0.5B-Instruct",
                source_dir=None,
                native_dir=None,
                force_hub_lookup=False,
            )
        self.assertEqual(native, cache / "qwen25-05b" / "Qwen2.5-0.5B-Instruct-native")
        self.assertEqual(source, cache / "qwen25-05b-src")

    def test_explicit_source_and_native_override_model_defaults(self) -> None:
        source, native = script.resolved_reference_paths(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            source_dir="/tmp/explicit-src",
            native_dir="/tmp/explicit-native",
            force_hub_lookup=False,
        )
        self.assertEqual(source, Path("/tmp/explicit-src"))
        self.assertEqual(native, Path("/tmp/explicit-native"))

    def test_resolved_source_prefers_complete_hub_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hub = root / "hub"
            cache = root / "espresso-cache"
            snap = hub / "models--Qwen--Qwen2.5-1.5B-Instruct" / "snapshots" / "abc123"
            for name in ("config.json", "tokenizer.json", "model.safetensors"):
                path = snap / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            ref = hub / "models--Qwen--Qwen2.5-1.5B-Instruct" / "refs" / "main"
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text("abc123\n", encoding="utf-8")

            with mock.patch.object(script, "espresso_cache_root", return_value=cache):
                with mock.patch.object(convert, "huggingface_hub_root", return_value=hub):
                    source, native = script.resolved_reference_paths(
                        model="Qwen/Qwen2.5-1.5B-Instruct",
                        source_dir=None,
                        native_dir=None,
                        force_hub_lookup=True,
                    )
            self.assertEqual(source, snap)
            self.assertEqual(native, cache / "qwen25-15b" / "Qwen2.5-1.5B-Instruct-native")

    def test_layer_parity_accepts_a_layer_subset(self) -> None:
        args = script.parse_args(["layer-parity", "--layers", "0"])
        self.assertEqual(script.parse_layer_list(args.layers), [0])
        args = script.parse_args(["layer-parity", "--layers", "0,3,27"])
        self.assertEqual(script.parse_layer_list(args.layers), [0, 3, 27])
        args = script.parse_args(["layer-parity"])
        self.assertIsNone(script.parse_layer_list(args.layers))

    def test_fixtures_record_model_short_name(self) -> None:
        self.assertEqual(
            script.model_short_name("Qwen/Qwen2.5-1.5B-Instruct"),
            "Qwen2.5-1.5B-Instruct",
        )
        self.assertEqual(
            script.model_short_name("Qwen/Qwen2.5-0.5B-Instruct"),
            "Qwen2.5-0.5B-Instruct",
        )

    def test_layer0_relative_gate_is_optional(self) -> None:
        args = script.parse_args(["layer-parity"])
        self.assertIsNone(args.gate_layer0_max_rel)
        args = script.parse_args(["layer-parity", "--gate-layer0-max-rel", "1e-3"])
        self.assertAlmostEqual(args.gate_layer0_max_rel, 1e-3)


if __name__ == "__main__":
    unittest.main()

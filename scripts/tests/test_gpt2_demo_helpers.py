#!/usr/bin/env python3
"""Contract tests for GPT-2 demo helper scripts required by espresso-generate."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"


def load_module(name: str, path: Path):
    if not path.is_file():
        raise unittest.SkipTest(f"missing helper script: {path.name}")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RequiredHelperScriptsExistTests(unittest.TestCase):
    def test_required_demo_helpers_are_tracked(self) -> None:
        required = [
            "bootstrap_gpt2_demo.py",
            "export_gpt2_coreml.py",
            "run_gpt2_coreml_reference.py",
            "assert_demo_helpers.py",
        ]
        missing = [name for name in required if not (SCRIPTS / name).is_file()]
        self.assertEqual(missing, [], f"missing helper scripts: {missing}")


class BootstrapGPT2DemoTests(unittest.TestCase):
    def test_parse_args_requires_outputs(self) -> None:
        bootstrap = load_module("bootstrap_gpt2_demo", SCRIPTS / "bootstrap_gpt2_demo.py")
        with self.assertRaises(SystemExit):
            bootstrap.parse_args([])

    def test_bootstrap_writes_weights_and_tokenizer(self) -> None:
        bootstrap = load_module("bootstrap_gpt2_demo", SCRIPTS / "bootstrap_gpt2_demo.py")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights_out = root / "weights"
            tokenizer_out = root / "tokenizer"
            cache_dir = root / "hf-cache"
            cache_dir.mkdir()

            vocab = {"!": 0, "a": 1}
            merges = "#version: 0.2\na a"

            class FakeTokenizer:
                def __init__(self, *args, **kwargs):
                    pass

                @classmethod
                def from_pretrained(cls, *args, **kwargs):
                    return cls()

                def save_pretrained(self, path: str) -> None:
                    dest = Path(path)
                    dest.mkdir(parents=True, exist_ok=True)
                    (dest / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
                    (dest / "merges.txt").write_text(merges, encoding="utf-8")

            def fake_convert(model_name, output_dir, cache_dir=None, metadata_name="gpt2_124m"):
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "metadata.json").write_text(
                    json.dumps({"name": metadata_name, "architecture": "gpt2"}),
                    encoding="utf-8",
                )
                payload = struct.pack("<f", 1.0)
                header = bytearray(128)
                header[0] = 0x01
                header[4] = 0x02
                header[64:68] = bytes([0xEF, 0xBE, 0xAD, 0xDE])
                header[68] = 0x01
                struct.pack_into("<I", header, 72, len(payload))
                struct.pack_into("<I", header, 80, 128)
                (output_dir / "lm_head.bin").write_bytes(bytes(header) + payload)

            bootstrap.bootstrap_gpt2_demo(
                model_name="gpt2",
                weights_out=weights_out,
                tokenizer_out=tokenizer_out,
                cache_dir=cache_dir,
                convert_pretrained_gpt2=fake_convert,
                GPT2Tokenizer=FakeTokenizer,
            )

            self.assertTrue((weights_out / "metadata.json").is_file())
            self.assertTrue((tokenizer_out / "vocab.json").is_file())
            self.assertTrue((tokenizer_out / "merges.txt").is_file())
            self.assertEqual(json.loads((tokenizer_out / "vocab.json").read_text())["a"], 1)


class ExportGPT2CoreMLTests(unittest.TestCase):
    def test_parse_args_matches_swift_invocation(self) -> None:
        export = load_module("export_gpt2_coreml", SCRIPTS / "export_gpt2_coreml.py")
        args = export.parse_args(
            [
                "--weights",
                "/tmp/gpt2_124m",
                "--output",
                "/tmp/gpt2_seq64.mlpackage",
                "--seq-len",
                "64",
            ]
        )
        self.assertEqual(args.weights, "/tmp/gpt2_124m")
        self.assertEqual(args.output, "/tmp/gpt2_seq64.mlpackage")
        self.assertEqual(args.seq_len, 64)


class RunGPT2CoreMLReferenceTests(unittest.TestCase):
    def test_module_loads_without_numpy(self) -> None:
        """CI macos-15 runners do not install numpy; parse_args/result helpers must import anyway."""
        previous = sys.modules.pop("numpy", None)

        def restore() -> None:
            if previous is None:
                sys.modules.pop("numpy", None)
            else:
                sys.modules["numpy"] = previous

        self.addCleanup(restore)
        reference = load_module("run_gpt2_coreml_reference", SCRIPTS / "run_gpt2_coreml_reference.py")
        self.assertNotIn("numpy", sys.modules)
        self.assertTrue(hasattr(reference, "parse_args"))
        self.assertTrue(hasattr(reference, "build_comparison_result"))

    def test_parse_args_matches_swift_invocation(self) -> None:
        reference = load_module("run_gpt2_coreml_reference", SCRIPTS / "run_gpt2_coreml_reference.py")
        args = reference.parse_args(
            [
                "--coreml-model",
                "/tmp/model.mlpackage",
                "--weights",
                "/tmp/weights",
                "--prompt-tokens",
                "15496,995",
                "--seq-len",
                "64",
                "--max-tokens",
                "8",
                "--temperature",
                "0",
                "--warmup",
                "1",
                "--iterations",
                "2",
                "--seed",
                "1234",
                "--compute-units",
                "cpu_and_neural_engine",
                "--emit-events",
            ]
        )
        self.assertEqual(args.prompt_tokens, [15496, 995])
        self.assertTrue(args.emit_events)

    def test_result_payload_matches_swift_decoder_keys(self) -> None:
        reference = load_module("run_gpt2_coreml_reference", SCRIPTS / "run_gpt2_coreml_reference.py")
        payload = reference.build_comparison_result(
            generated_tokens=[1, 2, 3],
            compile_time_ms=12.5,
            first_token_latency_ms=3.0,
            tokens_per_second=40.0,
            median_token_ms=25.0,
            p95_token_ms=30.0,
            token_latencies_ms=[25.0, 25.0, 30.0],
            total_time_ms=80.0,
            compute_units="cpu_only",
            seq_len=64,
        )
        required = {
            "generated_tokens",
            "compile_time_ms",
            "first_token_latency_ms",
            "tokens_per_second",
            "median_token_ms",
            "p95_token_ms",
            "token_latencies_ms",
            "total_time_ms",
            "compute_units",
            "seq_len",
        }
        self.assertTrue(required.issubset(payload.keys()))


class AssertDemoHelpersTests(unittest.TestCase):
    def test_assert_demo_helpers_passes_on_repo(self) -> None:
        assert_helpers = load_module("assert_demo_helpers", SCRIPTS / "assert_demo_helpers.py")
        self.assertEqual(assert_helpers.main(), 0)


if __name__ == "__main__":
    unittest.main()

import json
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_stories_coreml_parity as script


class DummyTokenizer:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def encode(self, text):
        return self.mapping.get(text, [0])

    def decode(self, tokens):
        return ",".join(str(token) for token in tokens)


class StoriesCoreMLParityTests(unittest.TestCase):
    def test_load_prompt_suite_parses_escape_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory) / "suite.txt"
            suite.write_text("hello:Hello\\nworld\n", encoding="utf-8")

            prompts = script.load_prompt_suite(suite)

        self.assertEqual(prompts, [("hello", "Hello\nworld")])

    def test_load_prompt_suite_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory) / "suite.txt"
            suite.write_text("hello:Hello\nhello:World\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                script.load_prompt_suite(suite)

    def test_compare_results_reports_cross_backend_matches(self) -> None:
        compare_payload = {
            "token_match": True,
            "text_match": True,
            "espresso": {
                "generated_tokens": [7, 8],
                "text": "1,2,7,8",
            },
            "coreml": {
                "generated_tokens": [7, 8],
                "text": "1,2,7,8",
            },
        }

        result = script.compare_results(
            DummyTokenizer(),
            [1, 2],
            [7, 8],
            compare_payload,
        )

        self.assertTrue(result["torch_matches_espresso_tokens"])
        self.assertTrue(result["torch_matches_coreml_tokens"])
        self.assertTrue(result["torch_matches_espresso_text"])
        self.assertTrue(result["torch_matches_coreml_text"])
        self.assertTrue(result["espresso_matches_coreml_tokens"])
        self.assertTrue(result["espresso_matches_coreml_text"])

    def test_run_espresso_compare_forces_cpu_exact_decode(self) -> None:
        completed = subprocess_completed_process(
            returncode=0,
            stdout=json.dumps({"espresso": {}, "coreml": {}, "token_match": True, "text_match": True}),
            stderr="",
        )
        with mock.patch.object(script.subprocess, "run", return_value=completed) as mocked:
            payload = script.run_espresso_compare(
                espresso_bin=Path("/tmp/espresso"),
                weights_dir=Path("/tmp/weights"),
                tokenizer_dir=Path("/tmp/tokenizer"),
                coreml_model=Path("/tmp/model.mlpackage"),
                seq_len=128,
                compute_units="cpu_only",
                max_tokens=8,
                seed=1234,
                compare_warmup=0,
                compare_iterations=1,
                prompt="Hello",
            )

        self.assertTrue(payload["token_match"])
        _, kwargs = mocked.call_args
        self.assertEqual(kwargs["env"][script.CPU_EXACT_ENV_KEY], "1")
        self.assertFalse(kwargs["check"])

    def test_run_espresso_compare_includes_subprocess_output_on_failure(self) -> None:
        completed = subprocess_completed_process(
            returncode=3,
            stdout="stdout detail",
            stderr="stderr detail",
        )
        with mock.patch.object(script.subprocess, "run", return_value=completed):
            with self.assertRaises(RuntimeError) as context:
                script.run_espresso_compare(
                    espresso_bin=Path("/tmp/espresso"),
                    weights_dir=Path("/tmp/weights"),
                    tokenizer_dir=Path("/tmp/tokenizer"),
                    coreml_model=Path("/tmp/model.mlpackage"),
                    seq_len=128,
                    compute_units="cpu_and_neural_engine",
                    max_tokens=8,
                    seed=1234,
                    compare_warmup=0,
                    compare_iterations=1,
                    prompt="Hello",
                )

        message = str(context.exception)
        self.assertIn("exit code 3", message)
        self.assertIn(script.CPU_EXACT_ENV_KEY, message)
        self.assertIn("stdout detail", message)
        self.assertIn("stderr detail", message)

    def test_validate_prompt_suite_capacity_rejects_undersized_sequence_length(self) -> None:
        tokenizer = DummyTokenizer(mapping={"long": list(range(5))})

        with self.assertRaises(ValueError) as context:
            script.validate_prompt_suite_capacity(
                tokenizer=tokenizer,
                prompt_suite=[("long_id", "long")],
                seq_len=6,
                max_tokens=2,
            )

        self.assertIn("long_id", str(context.exception))
        self.assertIn("requires sequence length 7", str(context.exception))


def subprocess_completed_process(*, returncode: int, stdout: str, stderr: str) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["espresso"], returncode=returncode, stdout=stdout, stderr=stderr)


if __name__ == "__main__":
    unittest.main()

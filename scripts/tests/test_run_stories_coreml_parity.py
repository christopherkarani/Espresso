import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_stories_coreml_parity as script


class DummyTokenizer:
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


if __name__ == "__main__":
    unittest.main()

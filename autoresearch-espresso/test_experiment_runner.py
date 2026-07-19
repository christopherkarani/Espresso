import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("experiment_runner.py")
SPEC = importlib.util.spec_from_file_location("autoresearch_experiment_runner", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ExperimentRunnerTests(unittest.TestCase):
    def test_resolve_default_baseline_summary_prefers_latest_claim_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            latest_json = root / "artifacts" / "benchmarks" / "release-serving-stories" / "latest.json"
            latest_json.parent.mkdir(parents=True)
            suite_summary = root / "results" / "release-suite-stories-20260408-000000" / "suite-summary.json"
            suite_summary.parent.mkdir(parents=True)
            suite_summary.write_text("{}")
            latest_json.write_text(json.dumps({"artifact_directory": str(suite_summary.parent.relative_to(root))}))

            resolved = MODULE.resolve_default_baseline_summary(root)

            self.assertEqual(resolved, suite_summary.resolve())

    def test_parse_suite_result_reads_summary_and_baseline_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "suite-summary.json").write_text(json.dumps({
                "aggregate": {
                    "espresso_tok_s_median": 88.1,
                    "espresso_ttft_ms_median": 1.6,
                    "espresso_median_token_ms_median": 12.8,
                    "espresso_p95_token_ms_median": 14.2,
                    "all_token_match": True,
                    "all_text_match": True,
                },
                "verdict": {
                    "all_correctness_gates_pass": True,
                    "all_performance_gates_pass": False,
                },
            }))
            (output_dir / "baseline-comparison.json").write_text(json.dumps({
                "merge_recommended": False,
            }))

            result = MODULE.parse_suite_result(output_dir)

            self.assertEqual(result.tokens_per_second, 88.1)
            self.assertEqual(result.ttft_ms, 1.6)
            self.assertEqual(result.median_token_ms, 12.8)
            self.assertEqual(result.p95_token_ms, 14.2)
            self.assertTrue(result.all_token_match)
            self.assertTrue(result.all_text_match)
            self.assertTrue(result.correctness_gates_pass)
            self.assertFalse(result.performance_gates_pass)
            self.assertFalse(result.merge_recommended)

    def test_suggest_status_requires_gates_and_improvement(self):
        result = MODULE.SuiteBenchmarkResult(
            tokens_per_second=81.0,
            ttft_ms=1.0,
            median_token_ms=12.0,
            p95_token_ms=14.0,
            all_token_match=True,
            all_text_match=True,
            correctness_gates_pass=True,
            performance_gates_pass=True,
            merge_recommended=True,
            status="ok",
        )
        self.assertEqual(MODULE.suggest_status(result, previous_best_tok_s=None), "keep")
        self.assertEqual(MODULE.suggest_status(result, previous_best_tok_s=80.0), "keep")
        self.assertEqual(MODULE.suggest_status(result, previous_best_tok_s=81.0), "discard")

        gated = MODULE.SuiteBenchmarkResult(
            tokens_per_second=90.0,
            ttft_ms=1.0,
            median_token_ms=12.0,
            p95_token_ms=14.0,
            all_token_match=True,
            all_text_match=True,
            correctness_gates_pass=False,
            performance_gates_pass=True,
            merge_recommended=False,
            status="gates_failed",
        )
        self.assertEqual(MODULE.suggest_status(gated, previous_best_tok_s=70.0), "discard")


if __name__ == "__main__":
    unittest.main()

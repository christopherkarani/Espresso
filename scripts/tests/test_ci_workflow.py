#!/usr/bin/env python3
"""Structural checks for .github/workflows/ci.yml (no PyYAML required)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CI_YML = REPO_ROOT / ".github" / "workflows" / "ci.yml"

RUN_BLOCK_START = re.compile(r"^-?\s*run:\s*\|-?\s*$")


def unindented_run_block_lines(text: str) -> list[tuple[int, str]]:
    """Return (lineno, line) for column-0 continuations inside `run: |` blocks."""
    bad: list[tuple[int, str]] = []
    in_run = False
    run_indent: int | None = None
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if RUN_BLOCK_START.match(stripped):
            in_run = True
            run_indent = indent
            continue
        if not in_run:
            continue
        if not line.strip():
            continue
        if indent == 0:
            bad.append((lineno, line))
            continue
        assert run_indent is not None
        if indent <= run_indent:
            in_run = False
    return bad


# Exact regression from the invalid workflow: python continuation at column 0.
BROKEN_RUN_CONTINUATION = """\
jobs:
  build-test:
    steps:
      - name: Resolve package (no third-party deps)
        run: |
          echo "$DEPS" | python3 -c 'import json,sys; d=json.load(sys.stdin); deps=d.get("dependencies") or [];
assert not deps, f"package graph must have zero dependencies, found: {deps}"'
  lint:
    name: Swift Package Lint
"""


class CIWorkflowTests(unittest.TestCase):
    def test_workflow_file_exists(self) -> None:
        self.assertTrue(CI_YML.is_file(), f"missing {CI_YML}")

    def test_run_block_continuations_are_indented(self) -> None:
        """A column-0 line inside `run: |` is invalid YAML (GitHub dies at the next key)."""
        text = CI_YML.read_text(encoding="utf-8")
        self.assertEqual(unindented_run_block_lines(text), [])

    def test_detector_flags_historical_column0_assert(self) -> None:
        bad = unindented_run_block_lines(BROKEN_RUN_CONTINUATION)
        self.assertEqual(len(bad), 1)
        lineno, line = bad[0]
        self.assertEqual(lineno, 7)
        self.assertTrue(line.startswith("assert not deps"), line)

    def test_jobs_and_test_filter_are_present(self) -> None:
        text = CI_YML.read_text(encoding="utf-8")
        self.assertRegex(text, r"(?m)^  build-test:")
        self.assertRegex(text, r"(?m)^  lint:")
        expected_filter = (
            "ANETypesTests|MILGeneratorTests|CPUOpsTests|ANEGraphIRTests|"
            "ANECodegenTests|ANEPassesTests|ANEBuilderTests|ModelSupportTests|"
            "DeltaCompilationTests|LoRAAdapterTests|MigrationParityTests|"
            "EspressoGenerateTests|ESPBundleTests|ESPCompilerTests|"
            "ESPRuntimeTests|ESPConvertTests|ESPBenchSupportTests|"
            "ESPCompilerCLITests|RealModelInferenceTests"
        )
        self.assertIn(f'--filter "{expected_filter}"', text)


if __name__ == "__main__":
    unittest.main()

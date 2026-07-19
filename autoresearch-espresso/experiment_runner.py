"""
Inference-first experiment runner for Espresso autoresearch.

This harness treats the hardened release-serving suite as the source-of-truth
benchmark contract for tok/s optimization. Experiments are judged on:
  - fixed-suite throughput and latency metrics
  - exact token/text parity across the suite
  - baseline regression gates from the retained shipping lane
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
AUTORESEARCH_DIR = PROJECT_ROOT / "autoresearch-espresso"
RESULTS_TSV = AUTORESEARCH_DIR / "suite-results.tsv"
RESULTS_HEADER = (
    "timestamp\tcommit\tstatus\tespresso_tokens_per_second\tespresso_ttft_ms\t"
    "espresso_median_token_ms\tespresso_p95_token_ms\tall_token_match\tall_text_match\t"
    "correctness_gates_pass\tperformance_gates_pass\tmerge_recommended\t"
    "output_dir\tbaseline_summary\tchange_summary\n"
)
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "autoresearch"
DEFAULT_BUNDLE = PROJECT_ROOT / ".build" / "release-bundles" / "stories110m-smoke.esp"
DEFAULT_PROMPTS_FILE = PROJECT_ROOT / "scripts" / "stories_release_benchmark_prompts.txt"
DEFAULT_COREML_MODEL = Path.home() / "Library" / "Application Support" / "Espresso" / "demo" / "stories110m_coreml" / "stories110m_stateful_seq128.mlpackage"
DEFAULT_COREML_SEQ_LEN = 128
DEFAULT_COMPUTE_UNITS = "cpu_only"
DEFAULT_MAX_TOKENS = 8
DEFAULT_RUNS = 1
DEFAULT_WARMUP = 1
DEFAULT_ITERATIONS = 3
DEFAULT_TIMEOUT_SECONDS = 15 * 60
DEFAULT_MIN_TOK_S_RATIO = 0.97
DEFAULT_MAX_TTFT_RATIO = 1.08
DEFAULT_MAX_MEDIAN_RATIO = 1.05
DEFAULT_MAX_P95_RATIO = 1.10
CLANG_MODULE_CACHE = PROJECT_ROOT / ".build" / "clang-module-cache"


@dataclass(frozen=True)
class HarnessConfig:
    bundle: Path
    prompts_file: Path
    coreml_model: Path
    baseline_summary: Path | None
    max_tokens: int
    runs: int
    warmup: int
    iterations: int
    coreml_seq_len: int
    compute_units: str
    min_tok_s_ratio: float
    max_ttft_ratio: float
    max_median_ratio: float
    max_p95_ratio: float


@dataclass
class SuiteBenchmarkResult:
    tokens_per_second: float
    ttft_ms: float
    median_token_ms: float
    p95_token_ms: float
    all_token_match: bool
    all_text_match: bool
    correctness_gates_pass: bool
    performance_gates_pass: bool
    merge_recommended: bool | None
    status: str
    output_dir: Path | None = None
    summary_path: Path | None = None
    baseline_comparison_path: Path | None = None
    raw_output: str = ""


def env_with_local_clang_cache() -> dict[str, str]:
    env = os.environ.copy()
    CLANG_MODULE_CACHE.mkdir(parents=True, exist_ok=True)
    env["CLANG_MODULE_CACHE_PATH"] = str(CLANG_MODULE_CACHE)
    return env


def current_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def resolve_default_baseline_summary(repo_root: Path = PROJECT_ROOT) -> Path | None:
    latest_claim = repo_root / "artifacts" / "benchmarks" / "release-serving-stories" / "latest.json"
    if latest_claim.exists():
        try:
            payload = json.loads(latest_claim.read_text())
            artifact_directory = payload.get("artifact_directory")
            if artifact_directory:
                candidate = (repo_root / artifact_directory / "suite-summary.json").resolve()
                if candidate.exists():
                    return candidate
        except json.JSONDecodeError:
            pass

    candidates = sorted(repo_root.glob("results/release-suite-stories-*/suite-summary.json"))
    if candidates:
        return candidates[-1].resolve()
    return None


def make_default_config() -> HarnessConfig:
    return HarnessConfig(
        bundle=Path(os.environ.get("ESPRESSO_AUTORESEARCH_BUNDLE", DEFAULT_BUNDLE)).expanduser().resolve(),
        prompts_file=Path(os.environ.get("ESPRESSO_AUTORESEARCH_PROMPTS_FILE", DEFAULT_PROMPTS_FILE)).expanduser().resolve(),
        coreml_model=Path(os.environ.get("ESPRESSO_AUTORESEARCH_COREML_MODEL", DEFAULT_COREML_MODEL)).expanduser().resolve(),
        baseline_summary=(
            Path(os.environ["ESPRESSO_AUTORESEARCH_BASELINE_SUMMARY"]).expanduser().resolve()
            if os.environ.get("ESPRESSO_AUTORESEARCH_BASELINE_SUMMARY")
            else resolve_default_baseline_summary()
        ),
        max_tokens=int(os.environ.get("ESPRESSO_AUTORESEARCH_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        runs=int(os.environ.get("ESPRESSO_AUTORESEARCH_RUNS", DEFAULT_RUNS)),
        warmup=int(os.environ.get("ESPRESSO_AUTORESEARCH_WARMUP", DEFAULT_WARMUP)),
        iterations=int(os.environ.get("ESPRESSO_AUTORESEARCH_ITERATIONS", DEFAULT_ITERATIONS)),
        coreml_seq_len=int(os.environ.get("ESPRESSO_AUTORESEARCH_COREML_SEQ_LEN", DEFAULT_COREML_SEQ_LEN)),
        compute_units=os.environ.get("ESPRESSO_AUTORESEARCH_COREML_COMPUTE_UNITS", DEFAULT_COMPUTE_UNITS),
        min_tok_s_ratio=float(os.environ.get("ESPRESSO_AUTORESEARCH_MIN_TOK_S_RATIO", DEFAULT_MIN_TOK_S_RATIO)),
        max_ttft_ratio=float(os.environ.get("ESPRESSO_AUTORESEARCH_MAX_TTFT_RATIO", DEFAULT_MAX_TTFT_RATIO)),
        max_median_ratio=float(os.environ.get("ESPRESSO_AUTORESEARCH_MAX_MEDIAN_RATIO", DEFAULT_MAX_MEDIAN_RATIO)),
        max_p95_ratio=float(os.environ.get("ESPRESSO_AUTORESEARCH_MAX_P95_RATIO", DEFAULT_MAX_P95_RATIO)),
    )


def ensure_paths_exist(config: HarnessConfig) -> None:
    required_paths = [
        ("bundle", config.bundle),
        ("prompts_file", config.prompts_file),
        ("coreml_model", config.coreml_model),
    ]
    for label, path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if config.baseline_summary and not config.baseline_summary.exists():
        raise FileNotFoundError(f"Missing baseline_summary: {config.baseline_summary}")


def swift_build(product: str = "espresso-generate", timeout: int = 600) -> tuple[bool, str]:
    cmd = ["swift", "build", "--product", product, "-c", "release"]
    print(f"[build] {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env_with_local_clang_cache(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "BUILD_TIMEOUT"

    output = result.stdout + result.stderr
    if result.returncode == 0:
        print("[build] succeeded")
        return True, output

    print(f"[build] FAILED (exit code {result.returncode})")
    lines = output.strip().splitlines()
    if lines:
        print("\n".join(lines[-50:]))
    return False, output


def release_binary(product: str) -> Path:
    return PROJECT_ROOT / ".build" / "release" / product


def ensure_release_binary(product: str, rebuild: bool) -> Path:
    binary = release_binary(product)
    if rebuild or not binary.exists():
        success, _ = swift_build(product=product)
        if not success:
            raise RuntimeError(f"Failed to build release product: {product}")
    if not binary.exists():
        raise FileNotFoundError(f"Missing release binary after build: {binary}")
    return binary


def default_output_dir(root: Path = DEFAULT_RESULTS_ROOT) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return root / f"suite-{timestamp}"


def parse_suite_result(
    output_dir: Path,
    raw_output: str = "",
    status: str | None = None,
) -> SuiteBenchmarkResult:
    summary_path = output_dir / "suite-summary.json"
    if not summary_path.exists():
        return SuiteBenchmarkResult(
            tokens_per_second=0.0,
            ttft_ms=0.0,
            median_token_ms=0.0,
            p95_token_ms=0.0,
            all_token_match=False,
            all_text_match=False,
            correctness_gates_pass=False,
            performance_gates_pass=False,
            merge_recommended=None,
            status=status or "bench_failed",
            output_dir=output_dir,
            raw_output=raw_output[-4000:],
        )

    summary = json.loads(summary_path.read_text())
    aggregate = summary["aggregate"]
    verdict = summary["verdict"]

    baseline_comparison_path = output_dir / "baseline-comparison.json"
    merge_recommended: bool | None = None
    if baseline_comparison_path.exists():
        baseline_payload = json.loads(baseline_comparison_path.read_text())
        merge_recommended = baseline_payload.get("merge_recommended", baseline_payload.get("all_pass"))

    return SuiteBenchmarkResult(
        tokens_per_second=float(aggregate["espresso_tok_s_median"]),
        ttft_ms=float(aggregate["espresso_ttft_ms_median"]),
        median_token_ms=float(aggregate["espresso_median_token_ms_median"]),
        p95_token_ms=float(aggregate["espresso_p95_token_ms_median"]),
        all_token_match=bool(aggregate["all_token_match"]),
        all_text_match=bool(aggregate["all_text_match"]),
        correctness_gates_pass=bool(verdict["all_correctness_gates_pass"]),
        performance_gates_pass=bool(verdict.get("all_performance_gates_pass", True)),
        merge_recommended=merge_recommended,
        status=status or ("ok" if bool(verdict.get("all_correctness_gates_pass", False)) and bool(verdict.get("all_performance_gates_pass", True)) else "gates_failed"),
        output_dir=output_dir,
        summary_path=summary_path,
        baseline_comparison_path=baseline_comparison_path if baseline_comparison_path.exists() else None,
        raw_output=raw_output[-4000:],
    )


def run_suite_benchmark(
    config: HarnessConfig,
    *,
    rebuild: bool,
    output_dir: Path | None,
    timeout: int,
) -> SuiteBenchmarkResult:
    ensure_paths_exist(config)
    binary = ensure_release_binary("espresso-generate", rebuild=rebuild)
    run_output_dir = (output_dir or default_output_dir()).resolve()
    run_output_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(binary),
        "suite",
        "--bundle", str(config.bundle),
        "--prompts", str(config.prompts_file),
        "--max-tokens", str(config.max_tokens),
        "--runs", str(config.runs),
        "--compare-warmup", str(config.warmup),
        "--compare-iterations", str(config.iterations),
        "--coreml-model", str(config.coreml_model),
        "--coreml-seq-len", str(config.coreml_seq_len),
        "--coreml-compute-units", config.compute_units,
        "--output-dir", str(run_output_dir),
    ]

    if config.baseline_summary:
        cmd.extend([
            "--baseline-summary", str(config.baseline_summary),
            "--min-espresso-tok-s-ratio", str(config.min_tok_s_ratio),
            "--max-espresso-ttft-ratio", str(config.max_ttft_ratio),
            "--max-espresso-median-token-ratio", str(config.max_median_ratio),
            "--max-espresso-p95-token-ratio", str(config.max_p95_ratio),
        ])

    print(f"[suite] {' '.join(cmd)}")
    try:
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env_with_local_clang_cache(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return SuiteBenchmarkResult(
            tokens_per_second=0.0,
            ttft_ms=0.0,
            median_token_ms=0.0,
            p95_token_ms=0.0,
            all_token_match=False,
            all_text_match=False,
            correctness_gates_pass=False,
            performance_gates_pass=False,
            merge_recommended=None,
            status="timeout",
            output_dir=run_output_dir,
        )

    output = completed.stdout + completed.stderr
    if completed.returncode == 0:
        return parse_suite_result(run_output_dir, raw_output=output, status="ok")

    if (run_output_dir / "suite-summary.json").exists():
        return parse_suite_result(run_output_dir, raw_output=output, status="gates_failed")

    return SuiteBenchmarkResult(
        tokens_per_second=0.0,
        ttft_ms=0.0,
        median_token_ms=0.0,
        p95_token_ms=0.0,
        all_token_match=False,
        all_text_match=False,
        correctness_gates_pass=False,
        performance_gates_pass=False,
        merge_recommended=None,
        status="bench_failed",
        output_dir=run_output_dir,
        raw_output=output[-4000:],
    )


def init_results() -> None:
    AUTORESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(RESULTS_HEADER)


def best_kept_tok_s(results_path: Path = RESULTS_TSV) -> float | None:
    if not results_path.exists():
        return None
    lines = results_path.read_text().splitlines()
    if not lines:
        return None
    header = lines[0].split("\t")
    try:
        status_idx = header.index("status")
        tok_idx = header.index("espresso_tokens_per_second")
    except ValueError:
        return None

    best: float | None = None
    for line in lines[1:]:
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) <= max(status_idx, tok_idx):
            continue
        if fields[status_idx] != "keep":
            continue
        try:
            value = float(fields[tok_idx])
        except ValueError:
            continue
        best = value if best is None else max(best, value)
    return best


def suggest_status(result: SuiteBenchmarkResult, previous_best_tok_s: float | None) -> str:
    if result.status in {"build_failed", "bench_failed", "timeout", "crash"}:
        return "crash"
    if not result.correctness_gates_pass or not result.performance_gates_pass:
        return "discard"
    if previous_best_tok_s is None:
        return "keep"
    return "keep" if result.tokens_per_second > previous_best_tok_s else "discard"


def append_result(
    commit: str,
    result: SuiteBenchmarkResult,
    status: str,
    baseline_summary: Path | None,
    description: str,
) -> None:
    init_results()
    row = [
        current_timestamp(),
        commit,
        status,
        f"{result.tokens_per_second:.6f}",
        f"{result.ttft_ms:.6f}",
        f"{result.median_token_ms:.6f}",
        f"{result.p95_token_ms:.6f}",
        "true" if result.all_token_match else "false",
        "true" if result.all_text_match else "false",
        "true" if result.correctness_gates_pass else "false",
        "true" if result.performance_gates_pass else "false",
        "" if result.merge_recommended is None else ("true" if result.merge_recommended else "false"),
        str(result.output_dir) if result.output_dir else "",
        str(baseline_summary) if baseline_summary else "",
        description,
    ]
    with RESULTS_TSV.open("a") as handle:
        handle.write("\t".join(row) + "\n")


def print_suite_result(result: SuiteBenchmarkResult) -> None:
    print(
        "[suite] tok/s={:.2f} ttft_ms={:.2f} median_token_ms={:.2f} p95_token_ms={:.2f}".format(
            result.tokens_per_second,
            result.ttft_ms,
            result.median_token_ms,
            result.p95_token_ms,
        )
    )
    print(
        "[quality] token_match={} text_match={} correctness_gates={} performance_gates={}".format(
            "PASS" if result.all_token_match else "FAIL",
            "PASS" if result.all_text_match else "FAIL",
            "PASS" if result.correctness_gates_pass else "FAIL",
            "PASS" if result.performance_gates_pass else "FAIL",
        )
    )
    if result.merge_recommended is not None:
        print(f"[baseline] merge_recommended={'YES' if result.merge_recommended else 'NO'}")
    if result.output_dir:
        print(f"[artifacts] suite_dir={result.output_dir}")


def print_json_result(result: SuiteBenchmarkResult) -> None:
    payload = {
        "tokens_per_second": result.tokens_per_second,
        "ttft_ms": result.ttft_ms,
        "median_token_ms": result.median_token_ms,
        "p95_token_ms": result.p95_token_ms,
        "all_token_match": result.all_token_match,
        "all_text_match": result.all_text_match,
        "correctness_gates_pass": result.correctness_gates_pass,
        "performance_gates_pass": result.performance_gates_pass,
        "merge_recommended": result.merge_recommended,
        "status": result.status,
        "output_dir": str(result.output_dir) if result.output_dir else None,
        "summary_path": str(result.summary_path) if result.summary_path else None,
        "baseline_comparison_path": str(result.baseline_comparison_path) if result.baseline_comparison_path else None,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def add_common_options(parser: argparse.ArgumentParser) -> None:
    defaults = make_default_config()
    parser.add_argument("--bundle", type=Path, default=defaults.bundle)
    parser.add_argument("--prompts-file", type=Path, default=defaults.prompts_file)
    parser.add_argument("--coreml-model", type=Path, default=defaults.coreml_model)
    parser.add_argument("--baseline-summary", type=Path, default=defaults.baseline_summary)
    parser.add_argument("--max-tokens", type=int, default=defaults.max_tokens)
    parser.add_argument("--runs", type=int, default=defaults.runs)
    parser.add_argument("--warmup", type=int, default=defaults.warmup)
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--coreml-seq-len", type=int, default=defaults.coreml_seq_len)
    parser.add_argument("--compute-units", default=defaults.compute_units)
    parser.add_argument("--min-tok-s-ratio", type=float, default=defaults.min_tok_s_ratio)
    parser.add_argument("--max-ttft-ratio", type=float, default=defaults.max_ttft_ratio)
    parser.add_argument("--max-median-ratio", type=float, default=defaults.max_median_ratio)
    parser.add_argument("--max-p95-ratio", type=float, default=defaults.max_p95_ratio)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--json", action="store_true")


def config_from_args(args: argparse.Namespace) -> HarnessConfig:
    return HarnessConfig(
        bundle=args.bundle.expanduser().resolve(),
        prompts_file=args.prompts_file.expanduser().resolve(),
        coreml_model=args.coreml_model.expanduser().resolve(),
        baseline_summary=args.baseline_summary.expanduser().resolve() if args.baseline_summary else None,
        max_tokens=args.max_tokens,
        runs=args.runs,
        warmup=args.warmup,
        iterations=args.iterations,
        coreml_seq_len=args.coreml_seq_len,
        compute_units=args.compute_units,
        min_tok_s_ratio=args.min_tok_s_ratio,
        max_ttft_ratio=args.max_ttft_ratio,
        max_median_ratio=args.max_median_ratio,
        max_p95_ratio=args.max_p95_ratio,
    )


def cmd_benchmark(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    result = run_suite_benchmark(
        config,
        rebuild=args.rebuild,
        output_dir=args.output_dir,
        timeout=args.timeout,
    )
    if args.json:
        print_json_result(result)
    else:
        print_suite_result(result)
    return 0 if result.status == "ok" else 1


def cmd_quality(args: argparse.Namespace) -> int:
    return cmd_benchmark(args)


def cmd_full(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    result = run_suite_benchmark(
        config,
        rebuild=True,
        output_dir=args.output_dir,
        timeout=args.timeout,
    )
    previous_best = best_kept_tok_s()
    commit = git_current_commit()
    status = suggest_status(result, previous_best)
    append_result(
        commit=commit,
        result=result,
        status=status,
        baseline_summary=config.baseline_summary,
        description=args.description,
    )
    if args.json:
        payload = {
            "commit": commit,
            "previous_best_tok_s": previous_best,
            "suggested_status": status,
        }
        payload.update(json.loads(json.dumps({
            "tokens_per_second": result.tokens_per_second,
            "ttft_ms": result.ttft_ms,
            "median_token_ms": result.median_token_ms,
            "p95_token_ms": result.p95_token_ms,
            "all_token_match": result.all_token_match,
            "all_text_match": result.all_text_match,
            "correctness_gates_pass": result.correctness_gates_pass,
            "performance_gates_pass": result.performance_gates_pass,
            "merge_recommended": result.merge_recommended,
            "status": result.status,
            "output_dir": str(result.output_dir) if result.output_dir else None,
        })))
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_suite_result(result)
        print(f"[results] commit={commit} previous_best_tok_s={previous_best} suggested_status={status}")
        print(f"[results] tsv={RESULTS_TSV}")
    return 0 if status == "keep" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference-first autoresearch harness for Espresso.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="Run the hardened suite benchmark.")
    add_common_options(benchmark)
    benchmark.add_argument("--rebuild", action="store_true", help="Rebuild the release binary before benchmarking.")
    benchmark.set_defaults(func=cmd_benchmark)

    quality = subparsers.add_parser("quality-check", help="Run the suite and report quality/parity gates.")
    add_common_options(quality)
    quality.add_argument("--rebuild", action="store_true", help="Rebuild the release binary before benchmarking.")
    quality.set_defaults(func=cmd_quality)

    full = subparsers.add_parser("full", help="Build, benchmark, score against the retained baseline, and log results.")
    add_common_options(full)
    full.add_argument("--description", default="manual run", help="Short experiment description for TSV logging.")
    full.set_defaults(func=cmd_full)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1
    except RuntimeError as error:
        print(f"[error] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

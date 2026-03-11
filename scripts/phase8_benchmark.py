#!/usr/bin/env python3
"""Phase 8 benchmark + grading runner.

Generates:
  - artifacts/benchmarks/phase8/latest.json
  - artifacts/benchmarks/phase8/latest.csv
  - artifacts/benchmarks/phase8/latest.md
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import random
import re
import shlex
import statistics
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TIER_BANDS: list[tuple[str, int, int]] = [
    ("S+", 97, 100),
    ("S", 93, 96),
    ("A", 85, 92),
    ("B", 75, 84),
    ("C", 65, 74),
    ("D", 50, 64),
    ("F", 0, 49),
]

TEST_CASE_RE = re.compile(
    r"Test Case '-\[(?P<suite>[^ ]+) (?P<test>[^\]]+)\]' (?P<status>passed|failed|skipped) \((?P<seconds>[0-9.]+) seconds\)\."
)
SKIP_REASON_RE = re.compile(r"Test skipped - (?P<reason>.+)$")


@dataclass
class CommandSpec:
    name: str
    command: list[str]
    env: dict[str, str]
    gate: str | None
    required: bool = True


@dataclass
class CommandRun:
    name: str
    gate: str | None
    display_command: str
    returncode: int
    attempts: int
    duration_s: float
    success: bool
    output_path: str
    output_tail: str
    test_cases: list[dict[str, Any]]
    phase8_metrics: list[dict[str, Any]]
    skip_reasons: list[str]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except Exception:
        return default
    if not math.isfinite(f):
        return default
    return f


def tier_for_score(score: float | None) -> str:
    if score is None:
        return "N/A"
    rounded = int(round(clamp(score, 0.0, 100.0)))
    for tier, lo, hi in TIER_BANDS:
        if lo <= rounded <= hi:
            return tier
    return "F"


def score_numerical_parity(max_abs_diff: float, mean_abs_diff: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 0.0
    max_ratio = max_abs_diff / tolerance
    mean_ratio = mean_abs_diff / tolerance
    worst_ratio = max(max_ratio, mean_ratio)
    if worst_ratio <= 1.0:
        # Passing tolerance gets top band; closer to 0 gets 100.
        return round(100.0 - (worst_ratio * 5.0), 2)
    # Above tolerance degrades quickly.
    return round(clamp(95.0 - ((worst_ratio - 1.0) * 95.0), 0.0, 95.0), 2)


def score_performance_ratio(ratio: float) -> float:
    if ratio <= 0:
        return 0.0
    if ratio <= 1.0:
        return 100.0
    # 1.5x slower maps to 50, 2.0x maps to 0.
    return round(clamp(100.0 - ((ratio - 1.0) * 100.0), 0.0, 100.0), 2)


def format_metric(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def run_capture(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


def run_capture_text(cmd: list[str], cwd: Path) -> str:
    code, out = run_capture(cmd, cwd)
    if code != 0:
        return ""
    return out.strip()


def parse_test_cases(output: str, retries: int, command_name: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for m in TEST_CASE_RE.finditer(output):
        seconds = safe_float(m.group("seconds"), 0.0)
        throughput = (1.0 / seconds) if seconds > 0 else None
        cases.append(
            {
                "command": command_name,
                "suite": m.group("suite"),
                "test": m.group("test"),
                "name": f"{m.group('suite')}.{m.group('test')}",
                "status": m.group("status"),
                "latency_s": seconds,
                "throughput_tests_per_s": throughput,
                "retries": retries,
            }
        )
    return cases


def parse_skip_reasons(output: str) -> list[str]:
    reasons: list[str] = []
    for line in output.splitlines():
        m = SKIP_REASON_RE.search(line)
        if m:
            reasons.append(m.group("reason").strip())
    # de-dupe preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


def parse_phase8_metrics(output: str) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for line in output.splitlines():
        text = line.strip()
        if not text.startswith("{"):
            continue
        if '"phase8_metric"' not in text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "phase8_metric":
            metrics.append(obj)
    return metrics


def display_command(command: list[str], env: dict[str, str]) -> str:
    env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    cmd = " ".join(shlex.quote(part) for part in command)
    if env_prefix:
        return f"{env_prefix} {cmd}"
    return cmd


def phase8_subprocess_env(cwd: Path) -> dict[str, str]:
    phase8_home = cwd / ".build" / "phase8-home"
    phase8_cache = cwd / ".build" / "phase8-cache"
    clang_cache = cwd / ".build" / "phase8-clang-cache"
    phase8_home.mkdir(parents=True, exist_ok=True)
    phase8_cache.mkdir(parents=True, exist_ok=True)
    clang_cache.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HOME"] = str(phase8_home)
    env["XDG_CACHE_HOME"] = str(phase8_cache)
    env["CLANG_MODULE_CACHE_PATH"] = str(clang_cache)
    env["SWIFTPM_MODULECACHE_OVERRIDE"] = str(clang_cache)
    return env


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def run_command(spec: CommandSpec, cwd: Path, log_dir: Path, max_retries: int, timeout_s: int | None = None) -> CommandRun:
    attempts = 0
    final_output = ""
    final_rc = 1
    total_duration = 0.0
    output_path = ""

    env = phase8_subprocess_env(cwd)
    env.update(spec.env)

    for attempt in range(1, max_retries + 2):
        attempts = attempt
        start = time.monotonic()
        timed_out = False
        timeout_note = ""
        try:
            proc = subprocess.run(
                spec.command,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
            )
            final_output = proc.stdout
            final_rc = proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            partial = _as_text(exc.stdout)
            final_output = partial + f"\n[phase8_benchmark] command timed out after {timeout_s}s"
            final_rc = 124
            timeout_note = f"\n[phase8_benchmark] timeout after {timeout_s}s"

        elapsed = time.monotonic() - start
        total_duration += elapsed

        log_path = log_dir / f"{spec.name}-attempt{attempt}.log"
        log_path.write_text(final_output + timeout_note, encoding="utf-8")
        output_path = str(log_path.relative_to(cwd))

        if (not timed_out) and final_rc == 0:
            break

    tail_lines = final_output.splitlines()[-20:]
    retries = max(0, attempts - 1)
    return CommandRun(
        name=spec.name,
        gate=spec.gate,
        display_command=display_command(spec.command, spec.env),
        returncode=final_rc,
        attempts=attempts,
        duration_s=round(total_duration, 3),
        success=(final_rc == 0),
        output_path=output_path,
        output_tail="\n".join(tail_lines),
        test_cases=parse_test_cases(final_output, retries=retries, command_name=spec.name),
        phase8_metrics=parse_phase8_metrics(final_output),
        skip_reasons=parse_skip_reasons(final_output),
    )


def build_metadata(project_root: Path, timestamp: str) -> dict[str, Any]:
    git_sha = run_capture_text(["git", "rev-parse", "HEAD"], project_root)
    if not git_sha:
        git_sha = "no git repo"

    hw_model = run_capture_text(["sysctl", "-n", "hw.model"], project_root)
    cpu_brand = run_capture_text(["sysctl", "-n", "machdep.cpu.brand_string"], project_root)
    mem_bytes = run_capture_text(["sysctl", "-n", "hw.memsize"], project_root)
    os_version = run_capture_text(["sw_vers", "-productVersion"], project_root)
    os_build = run_capture_text(["sw_vers", "-buildVersion"], project_root)
    swift_version = run_capture_text(["swift", "--version"], project_root)

    if not hw_model:
        hw_model = run_capture_text(["uname", "-m"], project_root) or platform.machine()
    if not cpu_brand:
        cpu_brand = platform.processor() or platform.machine()

    if mem_bytes.isdigit():
        mem_value: int | None = int(mem_bytes)
    else:
        try:
            mem_value = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except Exception:
            mem_value = None

    return {
        "timestamp": timestamp,
        "git_sha": git_sha,
        "hardware_model": hw_model or "unknown",
        "cpu": cpu_brand or "unknown",
        "memory_bytes": mem_value,
        "os_version": os_version or "unknown",
        "os_build": os_build or "unknown",
        "toolchain": swift_version or "unknown",
    }


def evaluate_performance(
    project_root: Path,
    logs_dir: Path,
    warmup_runs: int,
    measure_runs: int,
    steps_per_run: int,
) -> dict[str, Any]:
    objc_src = project_root / "archive" / "training" / "train_large.m"
    timing_keys = ["t_ane", "t_io", "t_cls", "t_elem", "t_rms", "t_cblas_wait"]

    def empty_perf(reason: str, objc_ms: float | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "available": False,
            "reason": reason,
            "swift_ms": None,
            "objc_ms": objc_ms,
            "ratio": None,
            "score": None,
            "tier": "N/A",
            "swift_breakdown": None,
            "objc_breakdown": None,
            "breakdown_available": False,
            "measurement": {
                "method": "median",
                "warmup_runs": warmup_runs,
                "measure_runs": measure_runs,
                "steps_per_run": steps_per_run,
            },
        }
        if extra:
            payload.update(extra)
        return payload

    if not objc_src.exists():
        return empty_perf(reason=f"missing ObjC source: {objc_src}")

    def parse_ms_per_step(output: str) -> float | None:
        values: list[float] = []
        for line in output.splitlines():
            text = line.strip()
            if not text.startswith("{"):
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            kind = payload.get("type")
            if kind not in {"batch", "step"}:
                continue
            value = safe_float(payload.get("ms_per_step"), default=-1.0)
            if value <= 0 and kind == "step":
                value = safe_float(payload.get("ms"), default=-1.0)
            if value > 0:
                values.append(value)
        return values[-1] if values else None

    def parse_step_breakdown(output: str) -> dict[str, float] | None:
        rows: list[dict[str, float]] = []
        ms_per_step_values: list[float] = []
        for line in output.splitlines():
            text = line.strip()
            if not text.startswith("{"):
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            ms_per_step = safe_float(payload.get("ms_per_step"), default=-1.0)
            if ms_per_step <= 0 and payload.get("type") == "step":
                ms_per_step = safe_float(payload.get("ms"), default=-1.0)
            if ms_per_step > 0:
                ms_per_step_values.append(ms_per_step)

            if payload.get("type") != "step":
                continue

            row: dict[str, float] = {}
            valid = True
            for key in timing_keys:
                value = safe_float(payload.get(key), default=-1.0)
                if value < 0:
                    valid = False
                    break
                row[key] = value
            if valid:
                rows.append(row)

        if not rows:
            return None
        row = rows[-1]
        if not ms_per_step_values:
            return None
        row["ms_per_step"] = ms_per_step_values[-1]
        return row

    def write_tokens(path: Path, vocab: int = 32000, count: int = 8192) -> None:
        rng = random.Random(42)
        with path.open("wb") as f:
            for _ in range(count):
                f.write(struct.pack("<H", rng.randrange(vocab)))

    env = phase8_subprocess_env(project_root)
    perf_dir = project_root / ".build" / "phase8-perf"
    perf_dir.mkdir(parents=True, exist_ok=True)

    token_path = perf_dir / "tinystories_data00.bin"
    write_tokens(token_path)

    objc_bin = perf_dir / "objc-train-large"
    swift_bin = project_root / ".build" / "release" / "espresso-train"
    swift_ckpt = perf_dir / "swift_phase8_perf_ckpt.bin"
    missing_model = perf_dir / "missing_model.bin"

    objc_compile_cmd = [
        "xcrun",
        "clang",
        "-O2",
        "-Wall",
        "-Wno-deprecated-declarations",
        "-fobjc-arc",
        "-o",
        str(objc_bin),
        str(objc_src),
        "-framework",
        "Foundation",
        "-framework",
        "CoreML",
        "-framework",
        "IOSurface",
        "-framework",
        "Accelerate",
        "-ldl",
        "-lobjc",
    ]
    objc_compile = subprocess.run(
        objc_compile_cmd,
        cwd=str(project_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    (logs_dir / "perf_objc_compile.log").write_text(objc_compile.stdout, encoding="utf-8")
    if objc_compile.returncode != 0:
        return empty_perf(
            reason="ObjC benchmark compile failed",
            extra={"compile_log": "artifacts/benchmarks/phase8/logs/perf_objc_compile.log"},
        )

    swift_build = subprocess.run(
        ["swift", "build", "-c", "release", "--product", "espresso-train"],
        cwd=str(project_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    (logs_dir / "perf_swift_build.log").write_text(swift_build.stdout, encoding="utf-8")
    if swift_build.returncode != 0:
        return empty_perf(
            reason="Swift benchmark build failed",
            extra={"build_log": "artifacts/benchmarks/phase8/logs/perf_swift_build.log"},
        )

    def aggregate_breakdowns(values: list[dict[str, float]]) -> dict[str, float] | None:
        if not values:
            return None
        keys = timing_keys + ["ms_per_step"]
        medians: dict[str, float] = {}
        for key in keys:
            key_values = [safe_float(row.get(key), default=-1.0) for row in values]
            valid = [v for v in key_values if v >= 0]
            if not valid:
                return None
            medians[key] = float(statistics.median(valid))
        return medians

    def run_perf_series(
        name: str,
        cmd: list[str],
        cwd: Path,
        warmups: int,
        measured: int,
    ) -> dict[str, Any]:
        samples: list[float] = []
        breakdown_samples: list[dict[str, float]] = []
        measured_logs: list[str] = []
        last_output = ""

        for i in range(warmups):
            run = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            warmup_path = logs_dir / f"perf_{name}_warmup_{i + 1}.log"
            warmup_path.write_text(run.stdout, encoding="utf-8")
            if run.returncode != 0:
                return {
                    "ok": False,
                    "reason": f"{name} warmup run {i + 1} failed",
                    "run_log": str(warmup_path.relative_to(project_root)),
                }

        for i in range(measured):
            run = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            run_path = logs_dir / f"perf_{name}_run_{i + 1}.log"
            run_path.write_text(run.stdout, encoding="utf-8")
            measured_logs.append(str(run_path.relative_to(project_root)))
            last_output = run.stdout
            if run.returncode != 0:
                return {
                    "ok": False,
                    "reason": f"{name} benchmark run {i + 1} failed",
                    "run_log": str(run_path.relative_to(project_root)),
                }

            ms = parse_ms_per_step(run.stdout)
            if ms is None:
                return {
                    "ok": False,
                    "reason": f"{name} benchmark run {i + 1} missing ms_per_step",
                    "run_log": str(run_path.relative_to(project_root)),
                }
            samples.append(ms)

            breakdown = parse_step_breakdown(run.stdout)
            if breakdown is not None:
                breakdown_samples.append(breakdown)

        canonical_log = logs_dir / f"perf_{name}_run.log"
        canonical_log.write_text(last_output, encoding="utf-8")
        median_ms = float(statistics.median(samples))
        return {
            "ok": True,
            "median_ms": median_ms,
            "samples": samples,
            "breakdown": aggregate_breakdowns(breakdown_samples),
            "run_log": str(canonical_log.relative_to(project_root)),
            "run_logs": measured_logs,
        }

    objc_result = run_perf_series(
        name="objc",
        cmd=[str(objc_bin), "--steps", str(steps_per_run)],
        cwd=perf_dir,
        warmups=warmup_runs,
        measured=measure_runs,
    )
    if not objc_result.get("ok", False):
        return empty_perf(
            reason=str(objc_result.get("reason", "ObjC benchmark failed")),
            extra={"run_log": objc_result.get("run_log")},
        )
    objc_ms = safe_float(objc_result.get("median_ms"), default=-1.0)
    objc_breakdown = objc_result.get("breakdown")
    if objc_ms <= 0:
        return empty_perf(
            reason="ObjC benchmark run failed or ms_per_step missing",
            extra={"run_log": objc_result.get("run_log")},
        )

    swift_result = run_perf_series(
        name="swift",
        cmd=[
            str(swift_bin),
            "--steps",
            str(steps_per_run),
            "--data",
            str(token_path),
            "--model",
            str(missing_model),
            "--ckpt",
            str(swift_ckpt),
        ],
        cwd=project_root,
        warmups=warmup_runs,
        measured=measure_runs,
    )
    if not swift_result.get("ok", False):
        return empty_perf(
            reason=str(swift_result.get("reason", "Swift benchmark failed")),
            objc_ms=objc_ms,
            extra={"run_log": swift_result.get("run_log")},
        )
    swift_ms = safe_float(swift_result.get("median_ms"), default=-1.0)
    swift_breakdown = swift_result.get("breakdown")
    if swift_ms <= 0:
        return empty_perf(
            reason="Swift benchmark run failed or ms_per_step missing",
            objc_ms=objc_ms,
            extra={"run_log": swift_result.get("run_log")},
        )

    ratio = swift_ms / objc_ms if objc_ms > 0 else None
    score = score_performance_ratio(ratio) if ratio is not None else None
    return {
        "available": ratio is not None,
        "reason": "measured (median)",
        "swift_ms": round(swift_ms, 3),
        "objc_ms": round(objc_ms, 3),
        "ratio": None if ratio is None else round(ratio, 6),
        "score": score,
        "tier": tier_for_score(score),
        "swift_breakdown": swift_breakdown,
        "objc_breakdown": objc_breakdown,
        "breakdown_available": bool(swift_breakdown is not None and objc_breakdown is not None),
        "measurement": {
            "method": "median",
            "warmup_runs": warmup_runs,
            "measure_runs": measure_runs,
            "steps_per_run": steps_per_run,
            "swift_samples_ms": [round(safe_float(v), 6) for v in swift_result.get("samples", [])],
            "objc_samples_ms": [round(safe_float(v), 6) for v in objc_result.get("samples", [])],
        },
        "objc_source": str(objc_src),
        "swift_log": str(swift_result.get("run_log", "artifacts/benchmarks/phase8/logs/perf_swift_run.log")),
        "objc_log": str(objc_result.get("run_log", "artifacts/benchmarks/phase8/logs/perf_objc_run.log")),
        "swift_logs": swift_result.get("run_logs", []),
        "objc_logs": objc_result.get("run_logs", []),
    }


def to_area_rows(
    metric_rows: list[dict[str, Any]],
    parity_score: float | None,
    stability_score: float | None,
    performance: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def perf_breakdown_text(prefix: str) -> str:
        breakdown = performance.get(f"{prefix}_breakdown")
        if not isinstance(breakdown, dict):
            return ""
        return (
            f"; t_ane={format_metric(breakdown.get('t_ane'), 3)}"
            f"; t_io={format_metric(breakdown.get('t_io'), 3)}"
            f"; t_cls={format_metric(breakdown.get('t_cls'), 3)}"
            f"; t_elem={format_metric(breakdown.get('t_elem'), 3)}"
            f"; t_rms={format_metric(breakdown.get('t_rms'), 3)}"
            f"; t_cblas_wait={format_metric(breakdown.get('t_cblas_wait'), 3)}"
        )

    for m in metric_rows:
        rows.append(
            {
                "area": m["area"],
                "swift_metric": f"max_abs_diff={m['max_abs_diff']:.6f}; mean_abs_diff={m['mean_abs_diff']:.6f}",
                "objc_metric": "max_abs_diff=0.000000; mean_abs_diff=0.000000",
                "delta": f"max={m['max_abs_diff']:.6f}; mean={m['mean_abs_diff']:.6f}",
                "tier": m["tier"],
                "score": m["score"],
            }
        )

    rows.append(
        {
            "area": "stability_repeatability",
            "swift_metric": "command retries + lane pass rate",
            "objc_metric": "N/A",
            "delta": "N/A",
            "tier": tier_for_score(stability_score),
            "score": stability_score,
        }
    )

    rows.append(
        {
            "area": "performance_vs_objc",
            "swift_metric": f"swift_ms={format_metric(performance.get('swift_ms'))}{perf_breakdown_text('swift')}",
            "objc_metric": f"objc_ms={format_metric(performance.get('objc_ms'))}{perf_breakdown_text('objc')}",
            "delta": f"ratio={format_metric(performance.get('ratio'))}",
            "tier": performance.get("tier", "N/A"),
            "score": performance.get("score"),
        }
    )

    rows.append(
        {
            "area": "numerical_parity_overall",
            "swift_metric": "aggregated parity score",
            "objc_metric": "reference fixtures",
            "delta": "N/A",
            "tier": tier_for_score(parity_score),
            "score": parity_score,
        }
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "git_sha", "area", "swift_metric", "objc_metric", "delta", "tier", "score"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "timestamp": metadata["timestamp"],
                    "git_sha": metadata["git_sha"],
                    "area": row["area"],
                    "swift_metric": row["swift_metric"],
                    "objc_metric": row["objc_metric"],
                    "delta": row["delta"],
                    "tier": row["tier"],
                    "score": "N/A" if row["score"] is None else f"{float(row['score']):.2f}",
                }
            )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep] + body)


def gate_status_text(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def summarize_regression(previous: dict[str, Any] | None, overall_score: float | None, threshold_pct: float) -> dict[str, Any]:
    if previous is None:
        return {
            "available": False,
            "status": "PASS",
            "reason": "No previous benchmark snapshot found",
            "old_score": None,
            "new_score": overall_score,
            "abs_change": None,
            "pct_change": None,
            "threshold_pct": threshold_pct,
            "exceeds_threshold": False,
        }

    old_score = previous.get("grades", {}).get("overall", {}).get("score")
    if old_score is None or overall_score is None:
        return {
            "available": True,
            "status": "PASS",
            "reason": "Insufficient score data for regression comparison",
            "old_score": old_score,
            "new_score": overall_score,
            "abs_change": None,
            "pct_change": None,
            "threshold_pct": threshold_pct,
            "exceeds_threshold": False,
        }

    old = safe_float(old_score)
    new = safe_float(overall_score)
    abs_change = round(new - old, 2)
    pct_change = round(((new - old) / old) * 100.0, 2) if old != 0 else None

    exceeds = (abs_change < 0) and (pct_change is not None) and (abs(pct_change) > threshold_pct)
    return {
        "available": True,
        "status": "FAIL" if exceeds else "PASS",
        "reason": "Regression exceeds threshold" if exceeds else "Within threshold",
        "old_score": old,
        "new_score": new,
        "abs_change": abs_change,
        "pct_change": pct_change,
        "threshold_pct": threshold_pct,
        "exceeds_threshold": exceeds,
    }


def build_md_report(
    metadata: dict[str, Any],
    command_runs: list[CommandRun],
    per_test_rows: list[dict[str, Any]],
    parity: dict[str, Any],
    performance: dict[str, Any],
    grades: dict[str, Any],
    gates: dict[str, Any],
    regression: dict[str, Any],
    blockers: list[dict[str, str]],
) -> str:
    lines: list[str] = []
    lines.append("# Phase 8 Benchmark Report")
    lines.append("")

    lines.append("## Metadata")
    meta_rows = [
        ["timestamp", metadata["timestamp"]],
        ["git_sha", metadata["git_sha"]],
        ["hardware_model", str(metadata.get("hardware_model", "unknown"))],
        ["cpu", str(metadata.get("cpu", "unknown"))],
        ["memory_bytes", str(metadata.get("memory_bytes", "unknown"))],
        ["os_version", str(metadata.get("os_version", "unknown"))],
        ["os_build", str(metadata.get("os_build", "unknown"))],
        ["toolchain", str(metadata.get("toolchain", "unknown")).replace("\n", " ")],
    ]
    lines.append(markdown_table(["field", "value"], meta_rows))
    lines.append("")

    lines.append("## Commands Run")
    lines.append("```bash")
    for run in command_runs:
        lines.append(run.display_command)
    lines.append("```")
    lines.append("")

    lines.append("## Command Results")
    cmd_rows = []
    for run in command_runs:
        cmd_rows.append(
            [
                run.name,
                run.gate or "-",
                gate_status_text(run.success),
                str(run.attempts - 1),
                f"{run.duration_s:.3f}",
                run.output_path,
            ]
        )
    lines.append(markdown_table(["command", "gate", "status", "retries", "duration_s", "log"], cmd_rows))
    lines.append("")

    lines.append("## Per-Test Latency/Throughput")
    if per_test_rows:
        test_rows: list[list[str]] = []
        for row in per_test_rows:
            test_rows.append(
                [
                    row["command"],
                    row["name"],
                    row["status"],
                    f"{safe_float(row['latency_s']) * 1000.0:.3f}",
                    "N/A" if row["throughput_tests_per_s"] is None else f"{safe_float(row['throughput_tests_per_s']):.3f}",
                    str(row["retries"]),
                ]
            )
        lines.append(markdown_table(["command", "test", "status", "latency_ms", "throughput_tests_per_s", "retries"], test_rows))
    else:
        lines.append("No parsed XCTest per-test rows.")
    lines.append("")

    lines.append("## Numerical Parity vs ObjC")
    metric_rows = []
    for m in parity.get("metrics", []):
        metric_rows.append(
            [
                m["area"],
                f"{safe_float(m['max_abs_diff']):.8f}",
                f"{safe_float(m['mean_abs_diff']):.8f}",
                f"{safe_float(m['tolerance']):.8f}",
                gate_status_text(bool(m["tolerance_pass"])),
                f"{safe_float(m['swift_eval_ms']):.6f}",
                m["tier"],
                f"{safe_float(m['score']):.2f}",
            ]
        )
    if metric_rows:
        lines.append(
            markdown_table(
                [
                    "area",
                    "max_abs_diff",
                    "mean_abs_diff",
                    "tolerance",
                    "tolerance_pass",
                    "swift_eval_ms",
                    "tier",
                    "score",
                ],
                metric_rows,
            )
        )
    else:
        lines.append("No numerical parity metrics were captured.")

    lines.append("")
    lines.append(
        f"Aggregated parity: max_abs_diff={format_metric(parity.get('max_abs_diff'))}, "
        f"mean_abs_diff={format_metric(parity.get('mean_abs_diff'))}, "
        f"status={gate_status_text(bool(parity.get('pass', False)))}"
    )
    lines.append("")

    lines.append("## Performance vs ObjC")
    perf_rows = [
        [
            format_metric(performance.get("swift_ms")),
            format_metric(performance.get("objc_ms")),
            format_metric(performance.get("ratio")),
            "N/A" if not performance.get("available", False) else gate_status_text(True),
            performance.get("reason", ""),
            performance.get("tier", "N/A"),
            "N/A" if performance.get("score") is None else f"{safe_float(performance['score']):.2f}",
        ]
    ]
    lines.append(markdown_table(["swift_time_ms", "objc_time_ms", "ratio_swift_over_objc", "status", "note", "tier", "score"], perf_rows))
    measurement = performance.get("measurement")
    if isinstance(measurement, dict):
        method = measurement.get("method", "unknown")
        warmup = measurement.get("warmup_runs", "N/A")
        measured = measurement.get("measure_runs", "N/A")
        steps = measurement.get("steps_per_run", "N/A")
        lines.append(
            f"Aggregation: `{method}` over `{measured}` measured run(s) after `{warmup}` warmup run(s), "
            f"with `{steps}` step(s)/run."
        )
    swift_breakdown = performance.get("swift_breakdown")
    objc_breakdown = performance.get("objc_breakdown")
    if isinstance(swift_breakdown, dict) and isinstance(objc_breakdown, dict):
        lines.append("")
        lines.append("### Step Timing Breakdown (ms/step)")
        breakdown_rows: list[list[str]] = []
        for key in ["t_ane", "t_io", "t_cls", "t_elem", "t_rms", "t_cblas_wait"]:
            swift_value = safe_float(swift_breakdown.get(key))
            objc_value = safe_float(objc_breakdown.get(key))
            delta = swift_value - objc_value
            breakdown_rows.append([key, f"{swift_value:.3f}", f"{objc_value:.3f}", f"{delta:.3f}"])
        lines.append(markdown_table(["component", "swift", "objc", "delta_swift_minus_objc"], breakdown_rows))
    lines.append("")

    lines.append("## Grade Table")
    grade_rows = []
    for g in grades.get("components", []):
        grade_rows.append(
            [
                g["area"],
                "N/A" if g["score"] is None else f"{safe_float(g['score']):.2f}",
                g["tier"],
                "N/A" if g["weight"] is None else f"{safe_float(g['weight']):.2f}",
            ]
        )
    lines.append(markdown_table(["area", "score", "tier", "weight"], grade_rows))
    lines.append(
        f"Overall grade: {grades['overall']['tier']} + "
        f"{('N/A' if grades['overall']['score'] is None else f"{safe_float(grades['overall']['score']):.2f}")}"
    )
    if grades.get("weight_note"):
        lines.append(grades["weight_note"])
    lines.append("")

    lines.append("## Regression vs Previous Snapshot")
    if regression.get("available"):
        lines.append(
            markdown_table(
                ["old_score", "new_score", "abs_change", "pct_change", "threshold_pct", "status", "reason"],
                [
                    [
                        format_metric(regression.get("old_score"), 2),
                        format_metric(regression.get("new_score"), 2),
                        format_metric(regression.get("abs_change"), 2),
                        format_metric(regression.get("pct_change"), 2),
                        format_metric(regression.get("threshold_pct"), 2),
                        regression.get("status", "PASS"),
                        regression.get("reason", ""),
                    ]
                ],
            )
        )
    else:
        lines.append(regression.get("reason", "No previous snapshot"))
    lines.append("")

    lines.append("## GATE PASS/FAIL")
    gate_rows = []
    for gate_name in ["G0", "G1", "G2", "G3", "G4", "G5"]:
        gate = gates[gate_name]
        gate_rows.append([gate_name, gate["command"], gate_status_text(gate["pass"]), gate["evidence"]])
    lines.append(markdown_table(["gate", "command", "pass_fail", "evidence"], gate_rows))
    lines.append("")

    lines.append("## Blockers")
    if blockers:
        for b in blockers:
            lines.append(f"- Command: `{b['command']}`")
            lines.append(f"  Reason: {b['reason']}")
            lines.append("  Output snippet:")
            lines.append("  ```text")
            lines.append(b["snippet"])
            lines.append("  ```")
    else:
        lines.append("- None")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 8 benchmark/gating suite")
    parser.add_argument("--regression-threshold-pct", type=float, default=5.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--perf-warmup-runs", type=int, default=1)
    parser.add_argument("--perf-measure-runs", type=int, default=5)
    parser.add_argument("--perf-steps", type=int, default=1)
    parser.add_argument("--command-timeout-s", type=int, default=420)
    args = parser.parse_args()
    if args.perf_warmup_runs < 0:
        parser.error("--perf-warmup-runs must be >= 0")
    if args.perf_measure_runs < 1:
        parser.error("--perf-measure-runs must be >= 1")
    if args.perf_steps < 1:
        parser.error("--perf-steps must be >= 1")
    if args.command_timeout_s < 1:
        parser.error("--command-timeout-s must be >= 1")

    project_root = Path(__file__).resolve().parent.parent
    artifacts_dir = project_root / "artifacts" / "benchmarks" / "phase8"
    logs_dir = artifacts_dir / "logs"
    history_dir = artifacts_dir / "history"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    latest_json = artifacts_dir / "latest.json"
    latest_csv = artifacts_dir / "latest.csv"
    latest_md = artifacts_dir / "latest.md"

    previous_snapshot: dict[str, Any] | None = None
    if latest_json.exists():
        try:
            previous_snapshot = json.loads(latest_json.read_text(encoding="utf-8"))
        except Exception:
            previous_snapshot = None

    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    metadata = build_metadata(project_root=project_root, timestamp=timestamp)

    specs = [
        CommandSpec(name="g0_swift_build", command=["swift", "build"], env={}, gate="G0", required=True),
        CommandSpec(name="g0_swift_test", command=["swift", "test"], env={}, gate="G0", required=True),
        CommandSpec(
            name="g1_hw_correctness",
            command=["swift", "test", "--filter", "ANERuntimeTests|EspressoTests|CrossValidationTests"],
            env={"ANE_HARDWARE_TESTS": "1"},
            gate="G1",
            required=True,
        ),
        CommandSpec(
            name="g2_objc_cross_validation",
            command=["swift", "test", "--filter", "CrossValidationTests"],
            env={
                "OBJC_CROSS_VALIDATION": "1",
                "ANE_HARDWARE_TESTS": "1",
                "PHASE8_BENCHMARKS": "1",
            },
            gate="G2",
            required=True,
        ),
        CommandSpec(
            name="probe_loss_decreases",
            command=["swift", "test", "--filter", "EspressoTests.test_10_steps_loss_decreases"],
            env={"ANE_HARDWARE_TESTS": "1", "ESPRESSO_INTEGRATION_TESTS": "1"},
            gate=None,
            required=False,
        ),
        CommandSpec(
            name="probe_100_steps_benchmark",
            command=["swift", "test", "--filter", "EspressoTests.test_100_steps_benchmark"],
            env={"ANE_HARDWARE_TESTS": "1", "ESPRESSO_PERF_TESTS": "1"},
            gate=None,
            required=False,
        ),
        CommandSpec(
            name="probe_cross_validate_refresh",
            command=["./scripts/cross_validate.sh"],
            env={},
            gate=None,
            required=False,
        ),
    ]

    command_runs = [
        run_command(
            spec,
            cwd=project_root,
            log_dir=logs_dir,
            max_retries=args.max_retries,
            timeout_s=args.command_timeout_s,
        )
        for spec in specs
    ]
    run_by_name = {run.name: run for run in command_runs}

    per_test_rows: list[dict[str, Any]] = []
    for run in command_runs:
        per_test_rows.extend(run.test_cases)

    cv_metrics_raw = run_by_name["g2_objc_cross_validation"].phase8_metrics
    metric_rows: list[dict[str, Any]] = []
    parity_max_abs: float | None = None
    parity_mean_abs: float | None = None
    parity_score: float | None = None
    parity_pass = False

    if cv_metrics_raw:
        for m in cv_metrics_raw:
            area = str(m.get("area", "unknown"))
            max_abs_diff = safe_float(m.get("max_abs_diff"))
            mean_abs_diff = safe_float(m.get("mean_abs_diff"))
            tolerance = safe_float(m.get("tolerance"))
            swift_eval_ms = safe_float(m.get("swift_eval_ms"))
            tolerance_pass = (max_abs_diff <= tolerance) and (mean_abs_diff <= tolerance)
            score = score_numerical_parity(max_abs_diff=max_abs_diff, mean_abs_diff=mean_abs_diff, tolerance=tolerance)
            metric_rows.append(
                {
                    "area": area,
                    "max_abs_diff": max_abs_diff,
                    "mean_abs_diff": mean_abs_diff,
                    "tolerance": tolerance,
                    "swift_eval_ms": swift_eval_ms,
                    "tolerance_pass": tolerance_pass,
                    "score": score,
                    "tier": tier_for_score(score),
                }
            )

        parity_max_abs = max(row["max_abs_diff"] for row in metric_rows)
        parity_mean_abs = statistics.fmean(row["mean_abs_diff"] for row in metric_rows)
        parity_pass = all(row["tolerance_pass"] for row in metric_rows)
        parity_score = round(statistics.fmean(row["score"] for row in metric_rows), 2)

    mandatory_runs = [run_by_name["g0_swift_build"], run_by_name["g0_swift_test"], run_by_name["g1_hw_correctness"], run_by_name["g2_objc_cross_validation"]]
    all_mandatory_ok = all(r.success for r in mandatory_runs)
    total_retries = sum(max(0, r.attempts - 1) for r in mandatory_runs)
    stability_score = 0.0 if not all_mandatory_ok else round(clamp(100.0 - (total_retries * 10.0), 0.0, 100.0), 2)

    performance = evaluate_performance(
        project_root=project_root,
        logs_dir=logs_dir,
        warmup_runs=args.perf_warmup_runs,
        measure_runs=args.perf_measure_runs,
        steps_per_run=args.perf_steps,
    )
    if performance.get("available") and performance.get("ratio") is not None:
        perf_score = score_performance_ratio(safe_float(performance["ratio"]))
        performance["score"] = perf_score
        performance["tier"] = tier_for_score(perf_score)

    base_weights = {"numerical": 60.0, "stability": 20.0, "performance": 20.0}
    available_scores: dict[str, float] = {}
    if parity_score is not None:
        available_scores["numerical"] = parity_score
    if stability_score is not None:
        available_scores["stability"] = stability_score
    if performance.get("score") is not None:
        available_scores["performance"] = safe_float(performance["score"])

    redistributed_weights: dict[str, float] = {}
    overall_score: float | None = None
    weight_note = ""
    if available_scores:
        weight_total = sum(base_weights[k] for k in available_scores)
        for k in available_scores:
            redistributed_weights[k] = base_weights[k] / weight_total
        overall_score = round(sum(available_scores[k] * redistributed_weights[k] for k in available_scores), 2)

        if set(available_scores.keys()) != {"numerical", "stability", "performance"}:
            weight_note = (
                "Performance component is N/A; weights redistributed proportionally across measurable components "
                f"({', '.join(f'{k}={redistributed_weights[k]:.4f}' for k in sorted(redistributed_weights))})."
            )

    overall_tier = tier_for_score(overall_score)

    blockers: list[dict[str, str]] = []
    for run in command_runs:
        if not run.success:
            blockers.append(
                {
                    "command": run.display_command,
                    "reason": f"exit code {run.returncode}",
                    "snippet": run.output_tail or "(no output)",
                }
            )

    for run in command_runs:
        if run.skip_reasons:
            snippet = "\n".join(run.skip_reasons[:5])
            blockers.append(
                {
                    "command": run.display_command,
                    "reason": "host-limited skip(s)",
                    "snippet": snippet,
                }
            )

    if not performance.get("available", False):
        blockers.append(
            {
                "command": "performance_vs_objc",
                "reason": "performance component unavailable",
                "snippet": performance.get("reason", "N/A"),
            }
        )

    regression = summarize_regression(previous=previous_snapshot, overall_score=overall_score, threshold_pct=args.regression_threshold_pct)

    g0_pass = run_by_name["g0_swift_build"].success and run_by_name["g0_swift_test"].success
    g1_pass = run_by_name["g1_hw_correctness"].success
    g2_pass = run_by_name["g2_objc_cross_validation"].success

    missing_parity_metrics = (len(metric_rows) == 0)
    parity_host_block_evidence = bool(run_by_name["g2_objc_cross_validation"].skip_reasons)

    g4_pass = True
    g4_reasons: list[str] = []
    if overall_score is None:
        g4_pass = False
        g4_reasons.append("overall score missing")
    if missing_parity_metrics and not parity_host_block_evidence:
        g4_pass = False
        g4_reasons.append("required numerical parity metrics missing without host-block evidence")
    if metric_rows and parity_max_abs is None:
        g4_pass = False
        g4_reasons.append("parity aggregates missing")

    area_rows = to_area_rows(metric_rows=metric_rows, parity_score=parity_score, stability_score=stability_score, performance=performance)

    grades = {
        "components": [
            {
                "area": "numerical_parity_vs_objc",
                "score": parity_score,
                "tier": tier_for_score(parity_score),
                "weight": redistributed_weights.get("numerical") if overall_score is not None else None,
            },
            {
                "area": "stability_repeatability",
                "score": stability_score,
                "tier": tier_for_score(stability_score),
                "weight": redistributed_weights.get("stability") if overall_score is not None else None,
            },
            {
                "area": "performance_vs_objc",
                "score": performance.get("score"),
                "tier": performance.get("tier", "N/A"),
                "weight": redistributed_weights.get("performance") if overall_score is not None and "performance" in redistributed_weights else None,
            },
        ],
        "overall": {"score": overall_score, "tier": overall_tier},
        "weight_note": weight_note,
    }

    payload = {
        "metadata": metadata,
        "commands": [
            {
                "name": run.name,
                "gate": run.gate,
                "command": run.display_command,
                "attempts": run.attempts,
                "retries": max(0, run.attempts - 1),
                "duration_s": run.duration_s,
                "returncode": run.returncode,
                "success": run.success,
                "log": run.output_path,
                "skip_reasons": run.skip_reasons,
            }
            for run in command_runs
        ],
        "per_test": per_test_rows,
        "numerical_parity": {
            "metrics": metric_rows,
            "max_abs_diff": parity_max_abs,
            "mean_abs_diff": parity_mean_abs,
            "pass": parity_pass,
            "missing_metrics": missing_parity_metrics,
            "host_block_evidence": run_by_name["g2_objc_cross_validation"].skip_reasons,
            "score": parity_score,
            "tier": tier_for_score(parity_score),
        },
        "stability": {
            "score": stability_score,
            "tier": tier_for_score(stability_score),
            "mandatory_pass": all_mandatory_ok,
            "total_retries": total_retries,
        },
        "performance": performance,
        "grades": grades,
        "regression": regression,
        "area_rows": area_rows,
        "blockers": blockers,
    }

    # G3 depends on artifacts carrying current timestamp + git sha/no-git.
    payload_has_metadata = bool(payload["metadata"].get("timestamp")) and bool(payload["metadata"].get("git_sha"))

    gates = {
        "G0": {
            "pass": g0_pass,
            "command": "swift build && swift test",
            "evidence": (
                f"swift build rc={run_by_name['g0_swift_build'].returncode}, "
                f"swift test rc={run_by_name['g0_swift_test'].returncode}"
            ),
        },
        "G1": {
            "pass": g1_pass,
            "command": 'ANE_HARDWARE_TESTS=1 swift test --filter "ANERuntimeTests|EspressoTests|CrossValidationTests"',
            "evidence": f"rc={run_by_name['g1_hw_correctness'].returncode}, log={run_by_name['g1_hw_correctness'].output_path}",
        },
        "G2": {
            "pass": g2_pass,
            "command": "OBJC_CROSS_VALIDATION=1 ANE_HARDWARE_TESTS=1 PHASE8_BENCHMARKS=1 swift test --filter CrossValidationTests",
            "evidence": (
                f"rc={run_by_name['g2_objc_cross_validation'].returncode}, "
                f"phase8_metrics={len(metric_rows)}, log={run_by_name['g2_objc_cross_validation'].output_path}"
            ),
        },
        "G3": {
            "pass": False,
            "command": "artifact completeness",
            "evidence": "pending artifact write",
        },
        "G4": {
            "pass": g4_pass,
            "command": "grading completeness",
            "evidence": "; ".join(g4_reasons) if g4_reasons else "overall grade + parity checks present",
        },
        "G5": {
            "pass": regression["status"] == "PASS",
            "command": "regression threshold comparison",
            "evidence": regression.get("reason", ""),
        },
    }

    payload["gates"] = gates

    md_report = build_md_report(
        metadata=metadata,
        command_runs=command_runs,
        per_test_rows=per_test_rows,
        parity=payload["numerical_parity"],
        performance=performance,
        grades=grades,
        gates=gates,
        regression=regression,
        blockers=blockers,
    )

    latest_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(path=latest_csv, rows=area_rows, metadata=metadata)
    latest_md.write_text(md_report, encoding="utf-8")

    g3_files_exist = latest_json.exists() and latest_csv.exists() and latest_md.exists()
    g3_has_metadata = False
    if g3_files_exist:
        md_text = latest_md.read_text(encoding="utf-8")
        g3_has_metadata = (metadata["timestamp"] in md_text) and (metadata["git_sha"] in md_text)

    gates["G3"] = {
        "pass": bool(g3_files_exist and payload_has_metadata and g3_has_metadata),
        "command": "artifact completeness",
        "evidence": (
            f"files_exist={g3_files_exist}, payload_metadata={payload_has_metadata}, "
            f"md_contains_timestamp_gitsha={g3_has_metadata}"
        ),
    }

    payload["gates"] = gates
    md_report = build_md_report(
        metadata=metadata,
        command_runs=command_runs,
        per_test_rows=per_test_rows,
        parity=payload["numerical_parity"],
        performance=performance,
        grades=grades,
        gates=gates,
        regression=regression,
        blockers=blockers,
    )

    latest_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    latest_md.write_text(md_report, encoding="utf-8")

    snapshot_name = f"phase8-{timestamp.replace(':', '').replace('-', '').replace('.', '')}.json"
    (history_dir / snapshot_name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote: {latest_json}")
    print(f"Wrote: {latest_csv}")
    print(f"Wrote: {latest_md}")
    print(f"Overall grade: {overall_tier} + {('N/A' if overall_score is None else f'{overall_score:.2f}')}")

    return 0 if all(gates[g]["pass"] for g in ["G0", "G1", "G2", "G3", "G4", "G5"]) else 1


if __name__ == "__main__":
    sys.exit(main())

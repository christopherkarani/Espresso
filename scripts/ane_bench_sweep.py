#!/usr/bin/env python3
"""Reproducible ANE decode/prefill option sweeps for espresso-bench.

Produces:
  - <output-root>/summary.csv
  - <output-root>/best_of.json
  - <output-root>/run_plan.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODE_DECODE = "decode"
MODE_PREFILL = "prefill"


@dataclass(frozen=True)
class RunSpec:
    name: str
    env: dict[str, str]
    note: str


def parse_key_value(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in assignment: {item}")
        out[key] = value
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic ANE option sweeps.")
    parser.add_argument("--mode", choices=[MODE_DECODE, MODE_PREFILL], required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--decode-max-seq", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--matrix", choices=["quick", "full"], default="full")
    parser.add_argument("--ane-only", action="store_true", default=True)
    parser.add_argument("--include-coreml", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--strict-options", action="store_true", default=True)
    parser.add_argument("--top-confirm-count", type=int, default=2)
    parser.add_argument("--top-confirm-repeats", type=int, default=3)
    parser.add_argument("--extra-env", action="append", default=[])
    return parser.parse_args()


def build_benchmark(binary: Path, cwd: Path) -> None:
    cmd = ["swift", "build", "-c", "release", "--product", "espresso-bench"]
    print(f"[build] {' '.join(shlex.quote(p) for p in cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError("swift build failed")
    if not binary.exists():
        raise RuntimeError(f"espresso-bench binary missing after build: {binary}")


def base_command(args: argparse.Namespace, binary: Path) -> list[str]:
    cmd = [str(binary)]
    if args.include_coreml:
        pass
    else:
        cmd.append("--ane-only")

    cmd += ["--profile-kernels", "--warmup", str(args.warmup), "--iterations", str(args.iterations)]
    cmd += ["--layers", str(args.layers)]

    if args.mode == MODE_DECODE:
        cmd += ["--decode", "--decode-steps", str(args.decode_steps), "--decode-max-seq", str(args.decode_max_seq)]
    else:
        cmd += ["--perf-stats", "--inference-only"]
    return cmd


def matrix_specs(mode: str, matrix: str) -> list[RunSpec]:
    specs: list[RunSpec] = [
        RunSpec("baseline_auto", {"ANE_COMPILE_CACHE_POLICY": "auto"}, "baseline (auto cache policy)"),
        RunSpec("baseline_prefer_cached", {"ANE_COMPILE_CACHE_POLICY": "preferCached"}, "baseline (preferCached)"),
        RunSpec("baseline_force_cold", {"ANE_COMPILE_CACHE_POLICY": "forceCold"}, "baseline (forceCold)"),
        RunSpec("eval_path_client", {"ANE_COMPILE_CACHE_POLICY": "preferCached", "ANE_EVAL_PATH": "client"}, "eval path client"),
        RunSpec(
            "eval_path_client_direct",
            {"ANE_COMPILE_CACHE_POLICY": "preferCached", "ANE_EVAL_PATH": "clientDirect"},
            "eval path clientDirect",
        ),
        RunSpec(
            "eval_path_realtime",
            {"ANE_COMPILE_CACHE_POLICY": "preferCached", "ANE_EVAL_PATH": "realtime"},
            "eval path realtime",
        ),
        RunSpec(
            "opt_disable_power_saving",
            {"ANE_COMPILE_CACHE_POLICY": "preferCached", "ANE_DISABLE_POWER_SAVING": "1"},
            "disable power saving",
        ),
        RunSpec(
            "opt_keep_model_wired",
            {"ANE_COMPILE_CACHE_POLICY": "preferCached", "ANE_KEEP_MODEL_WIRED": "1"},
            "keep model wired",
        ),
        RunSpec(
            "opt_enable_late_latch",
            {"ANE_COMPILE_CACHE_POLICY": "preferCached", "ANE_ENABLE_LATE_LATCH": "1"},
            "enable late latch",
        ),
        RunSpec(
            "opt_skip_prepare",
            {"ANE_COMPILE_CACHE_POLICY": "preferCached", "ANE_SKIP_PREPARE": "1"},
            "skip prepare phase",
        ),
        RunSpec(
            "opt_disable_io_fences",
            {"ANE_COMPILE_CACHE_POLICY": "preferCached", "ANE_DISABLE_IO_FENCES": "1"},
            "disable io fences",
        ),
        RunSpec(
            "opt_enable_fw_to_fw_signal",
            {"ANE_COMPILE_CACHE_POLICY": "preferCached", "ANE_ENABLE_FW_TO_FW_SIGNAL": "1"},
            "enable fw-to-fw signal",
        ),
        RunSpec(
            "opt_use_compiler_options",
            {"ANE_COMPILE_CACHE_POLICY": "preferCached", "ANE_USE_COMPILER_OPTIONS": "1"},
            "use compiler option synthesis",
        ),
    ]

    for depth in [1, 2, 4, 8, 16, 32, 64, 127]:
        specs.append(
            RunSpec(
                f"queue_depth_{depth}",
                {"ANE_COMPILE_CACHE_POLICY": "preferCached", "ANE_QUEUE_DEPTH": str(depth)},
                f"queue depth {depth}",
            )
        )

    for pool_id in [0, 1, 2]:
        specs.append(
            RunSpec(
                f"memory_pool_{pool_id}",
                {"ANE_COMPILE_CACHE_POLICY": "preferCached", "ANE_MEMORY_POOL_ID": str(pool_id)},
                f"memory pool {pool_id}",
            )
        )

    if mode == MODE_PREFILL:
        for mask in ["0x1", "0x3", "0xF"]:
            specs.append(
                RunSpec(
                    f"perf_stats_mask_{mask.lower().replace('0x', '')}",
                    {"ANE_COMPILE_CACHE_POLICY": "preferCached", "ANE_PERF_STATS_MASK": mask},
                    f"perf-stats mask {mask}",
                )
            )

    if matrix == "quick":
        keep_names = {
            "baseline_prefer_cached",
            "eval_path_client_direct",
            "opt_keep_model_wired",
            "opt_disable_power_saving",
            "queue_depth_8",
            "queue_depth_16",
        }
        specs = [spec for spec in specs if spec.name in keep_names]
    return specs


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("._-")


def validate_artifacts(run_dir: Path, mode: str, include_coreml: bool) -> list[str]:
    if mode == MODE_DECODE:
        required = [
            "summary.txt",
            "summary.json",
            "ane_decode_token_latencies.csv",
            "ane_decode_kernel_profile.csv",
        ]
        if include_coreml:
            required += [
                "coreml_decode_all_token_latencies.csv",
                "coreml_decode_cpuandneuralengine_token_latencies.csv",
            ]
    else:
        required = [
            "summary.txt",
            "summary.json",
            "ane_inference_latencies.csv",
            "ane_inference_kernel_profile.csv",
        ]
        if include_coreml:
            required += [
                "coreml_all_latencies.csv",
                "coreml_cpuandneuralengine_latencies.csv",
                "coreml_cpuandgpu_latencies.csv",
            ]
    return [filename for filename in required if not (run_dir / filename).exists()]


def extract_metrics(summary: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == MODE_DECODE:
        metrics = summary.get("ane_decode", {}).get("metrics", {})
        return {
            "mean_ms": metrics.get("mean_ms"),
            "median_ms": metrics.get("median_ms"),
            "p95_ms": metrics.get("p95_ms"),
            "p99_ms": metrics.get("p99_ms"),
            "tokens_per_second": metrics.get("tokens_per_second"),
            "speedup_vs_fastest_coreml_decode": summary.get("speedup_vs_fastest_coreml_decode"),
        }
    metrics = summary.get("ane_inference", {}).get("metrics", {})
    return {
        "mean_ms": metrics.get("mean_ms"),
        "median_ms": metrics.get("median_ms"),
        "p95_ms": metrics.get("p95_ms"),
        "p99_ms": metrics.get("p99_ms"),
        "tokens_per_second": metrics.get("tokens_per_second"),
        "speedup_vs_fastest_coreml_decode": None,
    }


def run_one(
    run_index: int,
    spec: RunSpec,
    cmd_base: list[str],
    output_root: Path,
    cwd: Path,
    env_base: dict[str, str],
    mode: str,
    include_coreml: bool,
) -> dict[str, Any]:
    run_name = f"{run_index:03d}_{sanitize_name(spec.name)}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = cmd_base + ["--output", str(run_dir)]

    env = env_base.copy()
    env.update(spec.env)
    env["ESPRESSO_BENCH_SEED"] = env.get("ESPRESSO_BENCH_SEED", "1")

    started = time.time()
    print(f"[run:{run_name}] {spec.note}")
    print(f"[cmd] {' '.join(shlex.quote(p) for p in cmd)}")
    print(f"[env] {json.dumps(spec.env, sort_keys=True)}")
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed_s = time.time() - started

    log_path = run_dir / "run.log"
    log_path.write_text(proc.stdout, encoding="utf-8")

    status = "success" if proc.returncode == 0 else "failed"
    missing_artifacts: list[str] = []
    summary: dict[str, Any] = {}
    metrics: dict[str, Any] = {
        "mean_ms": None,
        "median_ms": None,
        "p95_ms": None,
        "p99_ms": None,
        "tokens_per_second": None,
        "speedup_vs_fastest_coreml_decode": None,
    }

    if status == "success":
        missing_artifacts = validate_artifacts(run_dir, mode=mode, include_coreml=include_coreml)
        if missing_artifacts:
            status = "invalid_artifacts"
        else:
            with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
                summary = json.load(f)
            metrics = extract_metrics(summary, mode=mode)

    return {
        "run_index": run_index,
        "run_name": run_name,
        "spec_name": spec.name,
        "status": status,
        "returncode": proc.returncode,
        "elapsed_s": elapsed_s,
        "output_dir": str(run_dir),
        "log_path": str(log_path),
        "env_overrides": spec.env,
        "note": spec.note,
        "missing_artifacts": missing_artifacts,
        "git_sha": summary.get("git_sha"),
        "mean_ms": metrics.get("mean_ms"),
        "median_ms": metrics.get("median_ms"),
        "p95_ms": metrics.get("p95_ms"),
        "p99_ms": metrics.get("p99_ms"),
        "tokens_per_second": metrics.get("tokens_per_second"),
        "speedup_vs_fastest_coreml_decode": metrics.get("speedup_vs_fastest_coreml_decode"),
    }


def to_float_or_inf(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float("inf")
    return parsed


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_index",
        "run_name",
        "spec_name",
        "status",
        "returncode",
        "elapsed_s",
        "mean_ms",
        "median_ms",
        "p95_ms",
        "p99_ms",
        "tokens_per_second",
        "speedup_vs_fastest_coreml_decode",
        "git_sha",
        "output_dir",
        "log_path",
        "note",
        "env_overrides",
        "missing_artifacts",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = row.copy()
            out["env_overrides"] = json.dumps(row.get("env_overrides", {}), sort_keys=True)
            out["missing_artifacts"] = json.dumps(row.get("missing_artifacts", []))
            writer.writerow(out)


def main() -> int:
    args = parse_args()
    if args.iterations is None:
        args.iterations = 500 if args.mode == MODE_DECODE else 1000
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be >= 0 and iterations must be > 0")
    if args.mode == MODE_DECODE and (args.decode_steps <= 0 or args.decode_max_seq <= 1):
        raise ValueError("decode-steps must be >0 and decode-max-seq must be >1")

    cwd = Path(__file__).resolve().parents[1]
    binary = cwd / ".build" / "release" / "espresso-bench"
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.build:
        build_benchmark(binary=binary, cwd=cwd)
    elif not binary.exists():
        raise FileNotFoundError(f"espresso-bench not found at {binary}; run with --build first")

    include_coreml = bool(args.include_coreml)
    cmd_base = base_command(args, binary=binary)
    base_env = os.environ.copy()
    if args.strict_options:
        base_env["ANE_STRICT_OPTIONS"] = "1"
    extra_env = parse_key_value(args.extra_env)
    base_env.update(extra_env)

    specs = matrix_specs(mode=args.mode, matrix=args.matrix)
    run_plan = {
        "mode": args.mode,
        "output_root": str(output_root),
        "command_base": cmd_base,
        "strict_options": args.strict_options,
        "include_coreml": include_coreml,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "decode_steps": args.decode_steps,
        "decode_max_seq": args.decode_max_seq,
        "layers": args.layers,
        "matrix": args.matrix,
        "spec_count": len(specs),
        "specs": [{"name": s.name, "env": s.env, "note": s.note} for s in specs],
    }
    (output_root / "run_plan.json").write_text(json.dumps(run_plan, indent=2, sort_keys=True), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    run_index = 1
    for spec in specs:
        rows.append(
            run_one(
                run_index=run_index,
                spec=spec,
                cmd_base=cmd_base,
                output_root=output_root,
                cwd=cwd,
                env_base=base_env,
                mode=args.mode,
                include_coreml=include_coreml,
            )
        )
        run_index += 1

    successful = [row for row in rows if row["status"] == "success" and row.get("median_ms") is not None]
    successful.sort(key=lambda row: to_float_or_inf(row["median_ms"]))
    top = successful[: max(0, args.top_confirm_count)]

    for winner in top:
        for repeat_idx in range(args.top_confirm_repeats):
            spec = RunSpec(
                name=f"confirm_{winner['spec_name']}_r{repeat_idx + 1}",
                env=winner["env_overrides"],
                note=f"confirmation repeat {repeat_idx + 1} for {winner['spec_name']}",
            )
            rows.append(
                run_one(
                    run_index=run_index,
                    spec=spec,
                    cmd_base=cmd_base,
                    output_root=output_root,
                    cwd=cwd,
                    env_base=base_env,
                    mode=args.mode,
                    include_coreml=include_coreml,
                )
            )
            run_index += 1

    write_summary_csv(output_root / "summary.csv", rows)

    best = None
    valid_rows = [row for row in rows if row["status"] == "success" and row.get("median_ms") is not None]
    if valid_rows:
        best = min(valid_rows, key=lambda row: to_float_or_inf(row["median_ms"]))
    best_of = {
        "mode": args.mode,
        "best_run": best,
        "successful_runs": len(valid_rows),
        "total_runs": len(rows),
        "generated_at_epoch_s": time.time(),
    }
    (output_root / "best_of.json").write_text(json.dumps(best_of, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[done] wrote {(output_root / 'summary.csv')}")
    print(f"[done] wrote {(output_root / 'best_of.json')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

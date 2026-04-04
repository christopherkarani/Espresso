"""
Experiment runner for Espresso autoresearch — ANE kernel tok/s optimization.

Usage:
    python experiment_runner.py benchmark           # Run baseline benchmark
    python experiment_runner.py bench-decode         # Run pure ANE decode benchmark
    python experiment_runner.py generate            # Run .esp bundle generate benchmark
    python experiment_runner.py quality [prompt]    # Check generation quality
    python experiment_runner.py full                # Build + benchmark + quality
"""

import os
import sys
import re
import json
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ESP_BUNDLE = "/tmp/stories110m.esp"

# 4 fixed prompts for quality checking
BENCHMARK_PROMPTS = [
    "Once upon a time, there was a little girl named Lily. She had a red",
    "The sun was shining brightly on a beautiful day. Birds were singing in",
    "Deep in the forest, there lived a wise old owl. Every night, the",
    "Tom was a brave astronaut. He flew his spaceship to the moon and",
]

QUALITY_MAX_TOKENS = 64
BENCH_DECODE_STEPS = 32
BENCH_DECODE_WARMUP = 2
BENCH_DECODE_ITERATIONS = 50
BENCH_DECODE_LAYERS = 12

@dataclass
class BenchmarkResult:
    tokens_per_second: float
    ms_per_token: float
    compile_time_ms: float
    status: str  # "ok", "build_failed", "bench_failed", "timeout", "crash"
    raw_output: str = ""

@dataclass
class QualityResult:
    passed: bool
    generated_text: str = ""
    error: str = ""

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def swift_build(product="espresso-bench", timeout=300):
    """Run swift build and return (success, output)."""
    cmd = ["swift", "build", "--product", product, "-c", "release"]
    print(f"[build] {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
        if success:
            print(f"[build] succeeded")
        else:
            print(f"[build] FAILED (exit code {result.returncode})")
            lines = output.strip().split('\n')
            print("\n".join(lines[-50:]))
        return success, output
    except subprocess.TimeoutExpired:
        print(f"[build] TIMED OUT after {timeout}s")
        return False, "BUILD_TIMEOUT"

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def run_bench_decode(timeout=180):
    """Run espresso-generate benchmark (.esp bundle) for REAL tok/s with actual attention."""
    gen_path = PROJECT_ROOT / ".build" / "release" / "espresso-generate"
    if not gen_path.exists():
        success, _ = swift_build("espresso-generate")
        if not success:
            return BenchmarkResult(0, 0, 0, "build_failed")

    cmd = [
        str(gen_path), "bench",
        "--bundle", ESP_BUNDLE,
        "--max-tokens", str(32),
        "--no-power", "--no-stats",
        "--compare-warmup", "1",
        "--compare-iterations", "10",
        BENCHMARK_PROMPTS[0],
    ]

    print(f"[bench-generate] {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout + result.stderr
        return parse_generate_benchmark(output)
    except subprocess.TimeoutExpired:
        print(f"[bench-generate] TIMED OUT after {timeout}s")
        return BenchmarkResult(0, 0, 0, "timeout")

def run_bench_generate(timeout=180):
    """Run espresso-generate bench (real .esp bundle path)."""
    gen_path = PROJECT_ROOT / ".build" / "release" / "espresso-generate"
    if not gen_path.exists():
        success, _ = swift_build("espresso-generate")
        if not success:
            return BenchmarkResult(0, 0, 0, "build_failed")

    cmd = [
        str(gen_path), "bench",
        "--bundle", ESP_BUNDLE,
        "--max-tokens", str(32),
        "--no-power", "--no-stats",
        "--compare-warmup", "1",
        "--compare-iterations", "5",
        BENCHMARK_PROMPTS[0],
    ]

    print(f"[bench-generate] {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout + result.stderr
        return parse_generate_benchmark(output)
    except subprocess.TimeoutExpired:
        print(f"[bench-generate] TIMED OUT after {timeout}s")
        return BenchmarkResult(0, 0, 0, "timeout")

def parse_decode_benchmark(output):
    """Parse espresso-bench --decode output.
    
    Example: "Throughput: 161.2 tok/s" and "Mean: 6.203 ms/token"
    """
    result = BenchmarkResult(0, 0, 0, "bench_failed", raw_output=output[-3000:])
    
    # Throughput
    match = re.search(r'Throughput:\s*([\d.]+)\s*tok/s', output)
    if match:
        result.tokens_per_second = float(match.group(1))
    
    # ANE tokens/sec
    match = re.search(r'ANE tokens/sec:\s*([\d.]+)', output)
    if match:
        result.tokens_per_second = float(match.group(1))
    
    # Mean ms/token
    match = re.search(r'Mean:\s*([\d.]+)\s*ms/token', output)
    if match:
        result.ms_per_token = float(match.group(1))
    
    # Compile time
    match = re.search(r'Decode compile time:\s*([\d.]+)\s*ms', output)
    if match:
        result.compile_time_ms = float(match.group(1))
    
    # Status
    if result.tokens_per_second > 0:
        result.status = "ok"
    
    return result

def parse_generate_benchmark(output):
    """Parse espresso-generate bench output.
    
    Example: "tok_per_s=75.25 median_token_ms=13.64"
    """
    result = BenchmarkResult(0, 0, 0, "bench_failed", raw_output=output[-3000:])
    
    # tok_per_s
    match = re.search(r'tok_per_s=([\d.]+)', output)
    if match:
        result.tokens_per_second = float(match.group(1))
    
    # median_token_ms
    match = re.search(r'median_token_ms=([\d.]+)', output)
    if match:
        result.ms_per_token = float(match.group(1))
    
    # compile_ms
    match = re.search(r'compile_ms=([\d.]+)', output)
    if match:
        result.compile_time_ms = float(match.group(1))
    
    # Status
    if result.tokens_per_second > 0:
        result.status = "ok"
    
    return result

# ---------------------------------------------------------------------------
# Quality check
# ---------------------------------------------------------------------------

def run_quality_check(prompt=None, timeout=120):
    """Run generation on a prompt and check that output is coherent."""
    gen_path = PROJECT_ROOT / ".build" / "release" / "espresso-generate"
    if not gen_path.exists():
        success, _ = swift_build("espresso-generate")
        if not success:
            return QualityResult(passed=False, error="build_failed")

    test_prompt = prompt or BENCHMARK_PROMPTS[0]
    cmd = [
        str(gen_path), "generate",
        "--bundle", ESP_BUNDLE,
        "--max-tokens", str(QUALITY_MAX_TOKENS),
        "--no-power", "--no-stats",
        test_prompt,
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout.strip()
        
        # Check for coherence: output should be non-empty and not contain error markers
        if not output:
            return QualityResult(passed=False, error="empty_output")
        
        # Check for common error patterns
        error_patterns = ["error:", "Error:", "failed:", "panic", "trap", "fatal"]
        for pattern in error_patterns:
            if pattern in output.lower():
                return QualityResult(passed=False, error=f"error_pattern: {pattern}")
        
        # Check for repetition / degeneration
        words = output.split()
        if len(words) < 10:
            return QualityResult(passed=False, error=f"too_short: {len(words)} words")
        
        # Check for extreme repetition (simple check)
        if len(words) > 20:
            bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
            from collections import Counter
            bigram_counts = Counter(bigrams)
            most_common_count = bigram_counts.most_common(1)[0][1] if bigram_counts else 0
            if most_common_count > len(bigrams) * 0.5:
                return QualityResult(passed=False, error="repetition_degeneration")
        
        return QualityResult(passed=True, generated_text=output)
        
    except subprocess.TimeoutExpired:
        return QualityResult(passed=False, error="timeout")
    except Exception as e:
        return QualityResult(passed=False, error=str(e))

# ---------------------------------------------------------------------------
# Swift source modification helpers
# ---------------------------------------------------------------------------

def read_swift(relative_path):
    full_path = PROJECT_ROOT / relative_path
    if not full_path.exists():
        raise FileNotFoundError(f"Swift file not found: {full_path}")
    return full_path.read_text()

def write_swift(relative_path, content):
    full_path = PROJECT_ROOT / relative_path
    full_path.write_text(content)

def patch_file(relative_path, old_text, new_text):
    """Simple text replacement in a Swift file. Returns True if patch applied."""
    content = read_swift(relative_path)
    if old_text not in content:
        print(f"[patch] Pattern not found in {relative_path}")
        return False
    new_content = content.replace(old_text, new_text, 1)
    write_swift(relative_path, new_content)
    print(f"[patch] Applied to {relative_path}")
    return True

def patch_file_all(relative_path, old_text, new_text):
    """Replace ALL occurrences of old_text in a Swift file."""
    content = read_swift(relative_path)
    if old_text not in content:
        print(f"[patch] Pattern not found in {relative_path}")
        return False
    new_content = content.replace(old_text, new_text)
    write_swift(relative_path, new_content)
    count = content.count(old_text)
    print(f"[patch] Applied {count} replacements in {relative_path}")
    return True

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git_commit(message):
    subprocess.run(["git", "add", "-A"], cwd=PROJECT_ROOT, check=True,
                  capture_output=True)
    result = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True
    )
    if result.returncode == 0:
        h = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"],
                          cwd=PROJECT_ROOT, capture_output=True, text=True)
        commit = h.stdout.strip()
        print(f"[git] Committed: {commit} — {message}")
        return commit
    else:
        print(f"[git] Commit failed: {result.stderr}")
        return None

def git_reset_to(commit_hash):
    result = subprocess.run(
        ["git", "reset", "--hard", commit_hash],
        cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"[git] Reset to {commit_hash}")
        return True
    print(f"[git] Reset failed: {result.stderr}")
    return False

def git_current_commit():
    r = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"],
                      cwd=PROJECT_ROOT, capture_output=True, text=True)
    return r.stdout.strip()

def git_current_branch():
    r = subprocess.run(["git", "branch", "--show-current"],
                      cwd=PROJECT_ROOT, capture_output=True, text=True)
    return r.stdout.strip()

# ---------------------------------------------------------------------------
# Results logging
# ---------------------------------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "autoresearch-espresso"
RESULTS_TSV = RESULTS_DIR / "results.tsv"
RESULTS_HEADER = "commit\ttok_s\tms_per_tok\tcompile_ms\tquality\tstatus\tdescription\n"

def init_results():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text(RESULTS_HEADER)
        print(f"[results] Created {RESULTS_TSV}")

def log_result(commit, bench, quality, description):
    init_results()
    status = "keep" if bench.tokens_per_second > 0 and quality.passed else "discard"
    if bench.status in ("build_failed", "bench_failed", "timeout", "crash"):
        status = "crash"
    
    row = f"{commit}\t{bench.tokens_per_second:.1f}\t{bench.ms_per_token:.1f}\t{bench.compile_time_ms:.0f}\t{1.0 if quality.passed else 0.0:.2f}\t{status}\t{description}\n"
    with open(RESULTS_TSV, "a") as f:
        f.write(row)
    print(f"[results] {commit} | {bench.tokens_per_second:.1f} tok/s | quality={'PASS' if quality.passed else 'FAIL'} | {status} | {description}")

# ---------------------------------------------------------------------------
# Reference generations
# ---------------------------------------------------------------------------

REF_GEN_PATH = RESULTS_DIR / "reference_generations.json"

def save_reference_generations():
    """Generate reference text for all prompts."""
    gens = {}
    for prompt in BENCHMARK_PROMPTS:
        qr = run_quality_check(prompt)
        if qr.passed:
            gens[prompt] = qr.generated_text
            print(f"[ref] Saved reference for prompt: {prompt[:50]}...")
        else:
            print(f"[ref] FAILED to generate reference: {qr.error}")
    REF_GEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    REF_GEN_PATH.write_text(json.dumps(gens, indent=2))
    print(f"[ref] Saved {len(gens)} references to {REF_GEN_PATH}")
    return gens

def load_reference_generations():
    if REF_GEN_PATH.exists():
        return json.loads(REF_GEN_PATH.read_text())
    return {}

def check_quality_against_reference(prompt=None):
    """Check that generation matches reference exactly."""
    ref = load_reference_generations()
    if not ref:
        # No reference, use simple coherence check
        return run_quality_check(prompt)
    
    test_prompt = prompt or BENCHMARK_PROMPTS[0]
    if test_prompt not in ref:
        return run_quality_check(prompt)
    
    qr = run_quality_check(prompt)
    if qr.passed:
        # Check exact match
        if qr.generated_text.strip() == ref[test_prompt].strip():
            return qr
        else:
            return QualityResult(passed=False, error="output_differs_from_reference")
    
    return qr

# ---------------------------------------------------------------------------
# Main commands
# ---------------------------------------------------------------------------

def cmd_benchmark():
    print("=" * 60)
    print("BENCHMARK: Pure ANE decode (espresso-bench --decode)")
    print("=" * 60)
    r = run_bench_decode()
    print(f"\n[RESULT] tok/s={r.tokens_per_second:.1f} ms/tok={r.ms_per_token:.1f} compile={r.compile_time_ms:.0f}ms status={r.status}")
    return r

def cmd_generate_bench():
    print("=" * 60)
    print("BENCHMARK: .esp bundle generate (espresso-generate bench)")
    print("=" * 60)
    r = run_bench_generate()
    print(f"\n[RESULT] tok/s={r.tokens_per_second:.1f} ms/tok={r.ms_per_token:.1f} compile={r.compile_time_ms:.0f}ms status={r.status}")
    return r

def cmd_quality():
    print("=" * 60)
    print("QUALITY CHECK")
    print("=" * 60)
    r = run_quality_check()
    text_preview = r.generated_text[:200] if r.generated_text else ""
    print(f"\n[RESULT] quality={'PASS' if r.passed else 'FAIL'} error={r.error}")
    if text_preview:
        print(f"[TEXT] {text_preview}...")
    return r

def cmd_save_references():
    print("=" * 60)
    print("SAVE REFERENCE GENERATIONS")
    print("=" * 60)
    save_reference_generations()

def cmd_full():
    print("=" * 60)
    print("FULL: Build → Bench-decode → Bench-generate → Quality")
    print("=" * 60)
    
    # Bench decode
    print("\n--- BENCH DECODE (pure ANE) ---")
    decode_r = run_bench_decode()
    print(f"  tok/s={decode_r.tokens_per_second:.1f}")
    
    # Bench generate
    print("\n--- BENCH GENERATE (.esp bundle) ---")
    gen_r = run_bench_generate()
    print(f"  tok/s={gen_r.tokens_per_second:.1f}")
    
    # Quality
    print("\n--- QUALITY ---")
    qr = run_quality_check()
    print(f"  {'PASS' if qr.passed else 'FAIL'}: {qr.error}")
    
    return decode_r, gen_r, qr

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} [bench-decode|bench-generate|quality|full|references]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    cmds = {
        "benchmark": cmd_benchmark,
        "bench-decode": cmd_benchmark,
        "bench-generate": cmd_generate_bench,
        "quality": cmd_quality,
        "full": cmd_full,
        "references": cmd_save_references,
    }
    
    if cmd not in cmds:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(cmds.keys())}")
        sys.exit(1)
    
    cmds[cmd]()

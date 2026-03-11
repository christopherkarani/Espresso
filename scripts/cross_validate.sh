#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ACTIVE_SRC="$PROJECT_ROOT/training"
ARCHIVE_SRC="$PROJECT_ROOT/archive/training"

if [[ -f "$ACTIVE_SRC/Makefile" ]]; then
    SRC_DIR="$ACTIVE_SRC"
elif [[ -f "$ARCHIVE_SRC/Makefile" ]]; then
    SRC_DIR="$ARCHIVE_SRC"
else
    echo "ERROR: no training source directory with Makefile found" >&2
    exit 1
fi

GOLDEN_DIR="$PROJECT_ROOT/training/golden_outputs"
BUILD_ROOT="$PROJECT_ROOT/.build/phase8-cross-validate"
BIN_DIR="$BUILD_ROOT/bin"
FAILED_DIR="$BUILD_ROOT/failed"
mkdir -p "$GOLDEN_DIR" "$BIN_DIR" "$FAILED_DIR"

REQUIRED_PROBES=(
    test_full_fused
    test_fused_bwd
    test_ane_sdpa5
)

OPTIONAL_PROBES=(
    test_weight_reload
    test_perf_stats
    test_qos_sweep
    test_ane_advanced
    test_fused_qkv
    test_ane_causal_attn
    test_conv_attn3
)

REQUIRED_STDOUT_FIXTURES=(
    test_full_fused.txt
    test_fused_bwd.txt
    test_ane_sdpa5.txt
)

REQUIRED_ORACLES=(
    weight_blob_4x4.bin
    causal_mask_seq8.bin
    full_fused.mil
    full_fused_wq.bin
    full_fused_wk.bin
    full_fused_wv.bin
    full_fused_wo.bin
    full_fused_mask.bin
    full_fused_input_seq64_f32le.bin
    full_fused_out_seq64_f32le.bin
    fused_bwd.mil
    fused_bwd_w1t.bin
    fused_bwd_w3t.bin
    fused_bwd_input_seq64_f32le.bin
    fused_bwd_dx_seq64_f32le.bin
)

ALL_PROBES=("${REQUIRED_PROBES[@]}" "${OPTIONAL_PROBES[@]}")

declare -a REQUIRED_FAILURES=()
declare -a OPTIONAL_WARNINGS=()

is_required_probe() {
    local needle="$1"
    local probe
    for probe in "${REQUIRED_PROBES[@]}"; do
        if [[ "$probe" == "$needle" ]]; then
            return 0
        fi
    done
    return 1
}

record_failure() {
    local name="$1"
    local reason="$2"
    REQUIRED_FAILURES+=("$name:$reason")
    echo "  ERROR: $name ($reason)" >&2
}

record_warning() {
    local name="$1"
    local reason="$2"
    OPTIONAL_WARNINGS+=("$name:$reason")
    echo "  WARN: $name ($reason)" >&2
}

build_probe() {
    local name="$1"
    local src="$SRC_DIR/$name.m"
    local out="$BIN_DIR/$name"
    local build_log="$FAILED_DIR/$name-build-$(date +%Y%m%d-%H%M%S)-$$.log"

    if [[ ! -f "$src" ]]; then
        if is_required_probe "$name"; then
            record_failure "$name" "missing source: $src"
        else
            record_warning "$name" "missing source: $src"
        fi
        return
    fi

    if xcrun clang -O2 -Wall -Wno-deprecated-declarations -fobjc-arc \
        -o "$out" "$src" \
        -framework Foundation -framework CoreML -framework IOSurface -ldl >"$build_log" 2>&1; then
        rm -f "$build_log"
        echo "  Built: $name"
    else
        if is_required_probe "$name"; then
            record_failure "$name" "compile failed (log: $build_log)"
        else
            record_warning "$name" "compile failed (log: $build_log)"
        fi
    fi
}

run_probe() {
    local name="$1"
    local out="$GOLDEN_DIR/$name.txt"
    local tmp="$BUILD_ROOT/$name.$$.tmp"
    local fail_out="$FAILED_DIR/$name-run-$(date +%Y%m%d-%H%M%S)-$$.txt"
    local bin="$BIN_DIR/$name"

    if [[ ! -x "$bin" ]]; then
        if is_required_probe "$name"; then
            record_failure "$name" "binary missing: $bin"
        else
            record_warning "$name" "binary missing: $bin"
        fi
        return
    fi

    echo "  Running: $name"
    if "$bin" >"$tmp" 2>&1; then
        if [[ -s "$tmp" ]]; then
            mv "$tmp" "$out"
            chmod 644 "$out"
            echo "  Captured: $out ($(wc -l < "$out") lines)"
        else
            mv "$tmp" "$fail_out"
            if is_required_probe "$name"; then
                record_failure "$name" "empty stdout (log: $fail_out)"
            else
                record_warning "$name" "empty stdout (log: $fail_out)"
            fi
        fi
    else
        mv "$tmp" "$fail_out"
        if is_required_probe "$name"; then
            record_failure "$name" "non-zero exit (log: $fail_out)"
        else
            record_warning "$name" "non-zero exit (log: $fail_out)"
        fi
    fi
}

echo "=== Source dir: $SRC_DIR ==="
echo "=== Golden dir: $GOLDEN_DIR ==="
echo "=== Build dir: $BUILD_ROOT ==="

echo "=== Building ObjC probes into .build ==="
for name in "${ALL_PROBES[@]}"; do
    build_probe "$name"
done

echo "=== Capturing probe stdout fixtures ==="
for name in "${REQUIRED_PROBES[@]}"; do
    run_probe "$name"
done
for name in "${OPTIONAL_PROBES[@]}"; do
    run_probe "$name"
done

if ((${#REQUIRED_FAILURES[@]} > 0)); then
    echo "=== Summary ==="
    echo "REQUIRED: FAIL"
    echo "OPTIONAL: WARN"
    echo "Required failures:"
    printf '  - %s\n' "${REQUIRED_FAILURES[@]}"
    if ((${#OPTIONAL_WARNINGS[@]} > 0)); then
        echo "Optional warnings:"
        printf '  - %s\n' "${OPTIONAL_WARNINGS[@]}"
    fi
    echo "Failed logs: $FAILED_DIR"
    echo "Refusing to report success with missing required artifacts." >&2
    exit 1
fi

echo "=== Building binary oracle generator into .build ==="
GEN_SRC="$SRC_DIR/gen_cross_validation_goldens.m"
GEN_BIN="$BIN_DIR/gen_cross_validation_goldens"
if [[ ! -f "$GEN_SRC" ]]; then
    echo "ERROR: missing source $GEN_SRC" >&2
    exit 1
fi
xcrun clang -O2 -Wall -Wno-deprecated-declarations -fobjc-arc \
    -o "$GEN_BIN" "$GEN_SRC" \
    -framework Foundation -framework CoreML -framework IOSurface -ldl

echo "=== Generating binary oracles (strict, atomic promote) ==="
ORACLE_TMP="$BUILD_ROOT/oracles-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$ORACLE_TMP"
"$GEN_BIN" "$ORACLE_TMP"

for name in "${REQUIRED_ORACLES[@]}"; do
    src="$ORACLE_TMP/$name"
    if [[ ! -s "$src" ]]; then
        echo "ERROR: missing or empty oracle artifact: $name" >&2
        exit 1
    fi
done

for name in "${REQUIRED_ORACLES[@]}"; do
    cp "$ORACLE_TMP/$name" "$GOLDEN_DIR/$name"
    chmod 644 "$GOLDEN_DIR/$name"
done

for name in "${REQUIRED_STDOUT_FIXTURES[@]}"; do
    out="$GOLDEN_DIR/$name"
    if [[ ! -s "$out" ]]; then
        echo "ERROR: missing or empty required probe stdout fixture: $out" >&2
        exit 1
    fi
done

echo "=== Summary ==="
echo "REQUIRED: PASS"
if ((${#OPTIONAL_WARNINGS[@]} > 0)); then
    echo "OPTIONAL: WARN"
    printf '  - %s\n' "${OPTIONAL_WARNINGS[@]}"
else
    echo "OPTIONAL: PASS"
fi
echo "Failed logs: $FAILED_DIR"
echo "=== Done ==="

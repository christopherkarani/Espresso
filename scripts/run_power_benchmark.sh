#!/bin/bash
# Run power monitoring alongside the benchmark.
# Requires: sudo access for powermetrics.
#
# Usage:
#   ./scripts/run_power_benchmark.sh [ane|coreml|both]

set -euo pipefail

MODE="${1:-both}"
RESULTS_DIR="benchmarks/results/power-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Power benchmark — results: $RESULTS_DIR"
echo "Mode: $MODE"
echo ""

if [[ "$MODE" == "ane" || "$MODE" == "both" ]]; then
    echo "=== ANE Direct (60s sustained) ==="
    echo "Starting powermetrics in background (requires sudo)..."
    sudo powermetrics \
        --samplers cpu_power,gpu_power,ane_power \
        --sample-interval 1000 \
        -n 60 \
        > "$RESULTS_DIR/power_ane_direct.log" 2>&1 &
    POWER_PID=$!

    .build/release/EspressoBench --ane-only --sustained \
        --output "$RESULTS_DIR/ane_direct"

    wait $POWER_PID 2>/dev/null || true
    echo "  Power log: $RESULTS_DIR/power_ane_direct.log"
fi

if [[ "$MODE" == "coreml" || "$MODE" == "both" ]]; then
    echo ""
    echo "=== Core ML (60s sustained) ==="
    echo "Starting powermetrics in background (requires sudo)..."
    sudo powermetrics \
        --samplers cpu_power,gpu_power,ane_power \
        --sample-interval 1000 \
        -n 60 \
        > "$RESULTS_DIR/power_coreml.log" 2>&1 &
    POWER_PID=$!

    .build/release/EspressoBench --sustained \
        --output "$RESULTS_DIR/coreml"

    wait $POWER_PID 2>/dev/null || true
    echo "  Power log: $RESULTS_DIR/power_coreml.log"
fi

echo ""
echo "Done. Results in $RESULTS_DIR/"

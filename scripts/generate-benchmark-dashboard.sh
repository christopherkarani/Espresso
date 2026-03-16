#!/usr/bin/env bash
# generate-benchmark-dashboard.sh
# Reads benchmarks/results/latest.json and writes docs/benchmarks.md
# Run locally after updating benchmark results, then commit both files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
JSON="$REPO_ROOT/benchmarks/results/latest.json"
OUT="$REPO_ROOT/docs/benchmarks.md"

command -v jq >/dev/null 2>&1 || { echo "jq required but not found"; exit 1; }

updated=$(jq -r '.updated' "$JSON")
hardware=$(jq -r '.hardware' "$JSON")
os=$(jq -r '.os' "$JSON")
version=$(jq -r '.espresso_version' "$JSON")
speedup=$(jq -r '.speedup_vs_coreml' "$JSON")

cat > "$OUT" <<HEADER
---
layout: default
title: Benchmark Dashboard
---

# Espresso Benchmark Dashboard

**Updated:** ${updated} &nbsp;·&nbsp; **Hardware:** ${hardware} &nbsp;·&nbsp; **OS:** ${os} &nbsp;·&nbsp; **Version:** v${version}

## Token Generation Performance

| Framework | Backend | ms / token | tok / s | Notes |
|-----------|---------|-----------|---------|-------|
HEADER

jq -r '.results[] | "| \(.name) | \(.backend) | \(.ms_per_token) | \(.tokens_per_sec) | \(.notes) |"' "$JSON" >> "$OUT"

cat >> "$OUT" <<FOOTER

> **${speedup}x faster than CoreML** on Apple M3 Max.
> Measured on stories110m (Llama architecture, 6 layers).
> See [benchmarks/results/latest.json](../benchmarks/results/latest.json) for raw data.

## Reproducing

\`\`\`bash
# Build and run the ANE benchmark (requires Apple Silicon, macOS 15+)
swift run espresso-bench --ane-only --inference --layers 6 --warmup 20 --iterations 100
\`\`\`

Results are committed to \`benchmarks/results/latest.json\` and this page is regenerated
by running \`scripts/generate-benchmark-dashboard.sh\`.
FOOTER

echo "Dashboard written to $OUT"

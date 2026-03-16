---
layout: default
title: Benchmark Dashboard
---

# Espresso Benchmark Dashboard

**Updated:** 2026-03-17 &nbsp;·&nbsp; **Hardware:** Apple M3 Max (16-core CPU, 40-core GPU, 16-core ANE) &nbsp;·&nbsp; **OS:** macOS 15.0 &nbsp;·&nbsp; **Version:** v1.1.0

## Token Generation Performance

| Framework | Backend | ms / token | tok / s | Notes |
|-----------|---------|-----------|---------|-------|
| Espresso ANE (recurrent fused, 6-layer) | ANE | 1.929 | 519 | Fused 3-layer recurrent decode + ANE classifier head + direct-select argmax |
| Espresso ANE (direct transformer, 6-layer) | ANE | 6.559 | 153 | Transformer decode without recurrent fusion |
| CoreML (cpuAndNeuralEngine baseline) | CPU+ANE | 6.582 | 152 | Apple CoreML default neural engine path |

> **3.41x faster than CoreML** on Apple M3 Max.
> Measured on stories110m (Llama architecture, 6 layers).
> See [benchmarks/results/latest.json](../benchmarks/results/latest.json) for raw data.

## Reproducing

```bash
# Build and run the ANE benchmark (requires Apple Silicon, macOS 15+)
swift run espresso-bench --ane-only --inference --layers 6 --warmup 20 --iterations 100
```

Results are committed to `benchmarks/results/latest.json` and this page is regenerated
by running `scripts/generate-benchmark-dashboard.sh`.

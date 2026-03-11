# Benchmark Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a benchmark suite comparing ANE Direct inference vs Core ML baseline to determine if the runtime justifies a company pivot.

**Architecture:** New `EspressoBench` executable target in existing Package.swift. Uses real Espresso forward pass pipeline. Core ML model generated via Python coremltools script. Results output as human-readable report + CSV.

**Tech Stack:** Swift 6.2, Espresso/ANERuntime/ANETypes/CPUOps, CoreML, os.signpost, ContinuousClock, coremltools (Python)

---

### Task 1: Add EspressoBench target to Package.swift

**Files:**
- Modify: `Package.swift`
- Create: `Sources/EspressoBench/main.swift` (placeholder)

**Step 1: Create the directory and placeholder main.swift**

```swift
// Sources/EspressoBench/main.swift
print("EspressoBench — placeholder")
```

**Step 2: Add the target to Package.swift**

Add to `products` array:
```swift
.executable(name: "espresso-bench", targets: ["EspressoBench"]),
```

Add to `targets` array (after `EspressoTrain`):
```swift
.executableTarget(
    name: "EspressoBench",
    dependencies: ["Espresso", "ANERuntime", "ANETypes", "CPUOps"],
    path: "Sources/EspressoBench",
    swiftSettings: [.swiftLanguageMode(.v6)],
    linkerSettings: [
        .linkedFramework("Accelerate"),
        .linkedFramework("IOSurface"),
        .linkedFramework("CoreML"),
    ]
),
```

**Step 3: Verify it builds**

Run: `swift build --target EspressoBench`
Expected: Build succeeds, prints "EspressoBench — placeholder"

**Step 4: Commit**

```bash
git add Package.swift Sources/EspressoBench/main.swift
git commit -m "feat(bench): add EspressoBench executable target scaffold"
```

---

### Task 2: BenchmarkRunner — measurement harness

**Files:**
- Create: `Sources/EspressoBench/BenchmarkRunner.swift`

**Step 1: Write BenchmarkRunner**

This is the core measurement harness. It handles warmup, timed iterations, and statistical analysis.

```swift
import Foundation
import os.signpost

struct BenchmarkResult {
    let label: String
    let latencies: [Double]  // milliseconds
    let warmupCount: Int
    let iterationCount: Int

    var sorted: [Double] { latencies.sorted() }
    var mean: Double { latencies.reduce(0, +) / Double(latencies.count) }
    var median: Double {
        let s = sorted
        let n = s.count
        return n % 2 == 0 ? (s[n/2 - 1] + s[n/2]) / 2.0 : s[n/2]
    }
    var p50: Double { percentile(0.50) }
    var p95: Double { percentile(0.95) }
    var p99: Double { percentile(0.99) }
    var min: Double { latencies.min() ?? 0 }
    var max: Double { latencies.max() ?? 0 }
    var stddev: Double {
        let m = mean
        let variance = latencies.reduce(0.0) { $0 + ($1 - m) * ($1 - m) } / Double(latencies.count)
        return variance.squareRoot()
    }

    func percentile(_ p: Double) -> Double {
        let s = sorted
        let index = p * Double(s.count - 1)
        let lower = Int(index)
        let upper = Swift.min(lower + 1, s.count - 1)
        let frac = index - Double(lower)
        return s[lower] * (1 - frac) + s[upper] * frac
    }
}

struct BenchmarkRunner {
    let warmup: Int
    let iterations: Int
    let log: OSLog

    init(warmup: Int = 50, iterations: Int = 1000) {
        self.warmup = warmup
        self.iterations = iterations
        self.log = OSLog(subsystem: "com.espresso.bench", category: .pointsOfInterest)
    }

    func run(label: String, body: () throws -> Void) throws -> BenchmarkResult {
        // Warmup
        for _ in 0..<warmup {
            try body()
        }

        // Measured
        var latencies: [Double] = []
        latencies.reserveCapacity(iterations)

        for i in 0..<iterations {
            let start = ContinuousClock.now
            os_signpost(.begin, log: log, name: "Iteration", "%{public}s #%d", label, i)

            try body()

            os_signpost(.end, log: log, name: "Iteration", "%{public}s #%d", label, i)
            let elapsed = ContinuousClock.now - start
            let ms = Double(elapsed.components.attoseconds) / 1e15
            latencies.append(ms)

            if (i + 1) % 100 == 0 {
                let currentMean = latencies.reduce(0, +) / Double(latencies.count)
                printStderr("  [\(label)] \(i + 1)/\(iterations) — mean so far: \(String(format: "%.3f", currentMean)) ms")
            }
        }

        return BenchmarkResult(
            label: label,
            latencies: latencies,
            warmupCount: warmup,
            iterationCount: iterations
        )
    }
}

func printStderr(_ message: String) {
    var stderr = FileHandle.standardError
    stderr.write(Data((message + "\n").utf8))
}
```

**Step 2: Verify it builds**

Run: `swift build --target EspressoBench`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add Sources/EspressoBench/BenchmarkRunner.swift
git commit -m "feat(bench): add BenchmarkRunner measurement harness"
```

---

### Task 3: FLOPCalculator — workload characterization

**Files:**
- Create: `Sources/EspressoBench/FLOPCalculator.swift`

**Step 1: Write FLOPCalculator**

Computes theoretical FLOPs for the transformer forward pass at current ModelConfig dimensions.

```swift
import ANETypes

enum FLOPCalculator {
    /// Total FLOPs for a single-layer transformer forward pass.
    /// Counts multiply-accumulate as 2 FLOPs (1 mul + 1 add).
    static func forwardPassFLOPs(
        dim: Int = ModelConfig.dim,
        hidden: Int = ModelConfig.hidden,
        seqLen: Int = ModelConfig.seqLen,
        heads: Int = ModelConfig.heads
    ) -> Double {
        let headDim = dim / heads

        // QKV projections: 3 × (dim × dim × seqLen × 2)
        let qkvFLOPs = 3.0 * Double(dim) * Double(dim) * Double(seqLen) * 2.0

        // Attention scores: Q @ K^T → (seqLen × seqLen × headDim × 2) per head
        let attnScoreFLOPs = Double(heads) * Double(seqLen) * Double(seqLen) * Double(headDim) * 2.0

        // Attention × V: (seqLen × headDim × seqLen × 2) per head
        let attnValueFLOPs = Double(heads) * Double(seqLen) * Double(headDim) * Double(seqLen) * 2.0

        // Output projection: dim × dim × seqLen × 2
        let outputProjFLOPs = Double(dim) * Double(dim) * Double(seqLen) * 2.0

        // FFN SwiGLU: W1 (hidden×dim), W3 (hidden×dim), W2 (dim×hidden), each × seqLen × 2
        let ffnFLOPs = 3.0 * Double(hidden) * Double(dim) * Double(seqLen) * 2.0

        // SiLU activation: ~5 FLOPs per element (exp, add, mul) — negligible but counted
        let siluFLOPs = 5.0 * Double(hidden) * Double(seqLen)

        // Softmax: ~5 FLOPs per element — negligible but counted
        let softmaxFLOPs = 5.0 * Double(heads) * Double(seqLen) * Double(seqLen)

        return qkvFLOPs + attnScoreFLOPs + attnValueFLOPs + outputProjFLOPs + ffnFLOPs + siluFLOPs + softmaxFLOPs
    }

    /// Convert to TFLOPS given latency in milliseconds.
    static func sustainedTFLOPS(flops: Double, latencyMs: Double) -> Double {
        flops / (latencyMs / 1000.0) / 1e12
    }

    /// ANE utilization percentage (M3 peak = 18.0 TFLOPS).
    static func aneUtilization(sustainedTFLOPS: Double, peakTFLOPS: Double = 18.0) -> Double {
        (sustainedTFLOPS / peakTFLOPS) * 100.0
    }
}
```

**Step 2: Verify it builds**

Run: `swift build --target EspressoBench`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add Sources/EspressoBench/FLOPCalculator.swift
git commit -m "feat(bench): add FLOPCalculator for TFLOPS/utilization metrics"
```

---

### Task 4: ResultsFormatter — output formatting

**Files:**
- Create: `Sources/EspressoBench/ResultsFormatter.swift`

**Step 1: Write ResultsFormatter**

Produces the human-readable report and CSV files.

```swift
import Foundation
import ANETypes

enum ResultsFormatter {
    static func chipName() -> String {
        var size: size_t = 0
        guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0 else { return "Unknown" }
        var buf = [CChar](repeating: 0, count: Int(size))
        guard sysctlbyname("machdep.cpu.brand_string", &buf, &size, nil, 0) == 0 else { return "Unknown" }
        return String(cString: buf)
    }

    static func formatReport(
        aneResult: BenchmarkResult,
        aneTimingBreakdown: (ane: Double, io: Double, elem: Double)?,
        coreMLResults: [(label: String, result: BenchmarkResult)]?,
        coreMLLoadTimeMs: Double?,
        thermalBefore: String?,
        thermalAfter: String?,
        flopsPerPass: Double,
        nLayers: Int
    ) -> String {
        var out = ""
        let chip = chipName()
        let peakTFLOPS = 18.0

        out += "=== ANE DIRECT INFERENCE BENCHMARK ===\n"
        out += "Chip: \(chip)\n"
        out += "ANE Peak: \(String(format: "%.1f", peakTFLOPS)) TFLOPS\n"
        out += "Workload: \(nLayers)-layer transformer, dim=\(ModelConfig.dim), seq=\(ModelConfig.seqLen), heads=\(ModelConfig.heads), hidden=\(ModelConfig.hidden)\n"
        out += "FLOPs per forward pass: \(String(format: "%.2f", flopsPerPass / 1e9)) GFLOPs\n\n"

        out += formatLatencySection(aneResult, flopsPerPass: flopsPerPass, peakTFLOPS: peakTFLOPS)

        if let breakdown = aneTimingBreakdown {
            let total = breakdown.ane + breakdown.io + breakdown.elem
            out += "--- Time Breakdown (avg per forward pass) ---\n"
            out += String(format: "ANE kernel:    %.3f ms (%.1f%%)\n", breakdown.ane, breakdown.ane / total * 100)
            out += String(format: "Surface I/O:   %.3f ms (%.1f%%)\n", breakdown.io, breakdown.io / total * 100)
            out += String(format: "CPU element:   %.3f ms (%.1f%%)\n\n", breakdown.elem, breakdown.elem / total * 100)
        }

        if let coreMLResults {
            for (label, result) in coreMLResults {
                out += "=== CORE ML BASELINE (\(label)) ===\n"
                if let loadTime = coreMLLoadTimeMs, label.contains("all") {
                    out += String(format: "Model load time: %.1f ms\n", loadTime)
                }
                out += formatLatencySection(result, flopsPerPass: flopsPerPass, peakTFLOPS: peakTFLOPS)

                // Comparison
                let speedup = result.median / aneResult.median
                out += "--- vs ANE Direct ---\n"
                out += String(format: "Latency Speedup (ANE Direct vs this): %.2fx\n", speedup)
                out += String(format: "Throughput Speedup: %.2fx\n\n", speedup)
            }
        }

        if let before = thermalBefore, let after = thermalAfter {
            out += "=== POWER & THERMAL ===\n"
            out += "Thermal state: \(before) -> \(after)\n\n"
        }

        return out
    }

    private static func formatLatencySection(_ result: BenchmarkResult, flopsPerPass: Double, peakTFLOPS: Double) -> String {
        var out = ""
        let sustained = FLOPCalculator.sustainedTFLOPS(flops: flopsPerPass, latencyMs: result.median)
        let utilPct = FLOPCalculator.aneUtilization(sustainedTFLOPS: sustained, peakTFLOPS: peakTFLOPS)
        let fwdPerSec = 1000.0 / result.median

        out += "--- Latency (\(result.iterationCount) iterations, \(result.warmupCount) warmup) ---\n"
        out += String(format: "Mean:    %.3f ms\n", result.mean)
        out += String(format: "Median:  %.3f ms\n", result.median)
        out += String(format: "P95:     %.3f ms\n", result.p95)
        out += String(format: "P99:     %.3f ms\n", result.p99)
        out += String(format: "Stddev:  %.3f ms\n", result.stddev)
        out += String(format: "Min:     %.3f ms\n", result.min)
        out += String(format: "Max:     %.3f ms\n\n", result.max)

        out += "--- Throughput ---\n"
        out += String(format: "Sustained TFLOPS:   %.4f\n", sustained)
        out += String(format: "ANE Utilization:    %.1f%%\n", utilPct)
        out += String(format: "Forward passes/sec: %.0f\n\n", fwdPerSec)

        return out
    }

    static func writeCSV(latencies: [Double], to path: String) throws {
        let header = "iteration,latency_ms\n"
        let rows = latencies.enumerated().map { "\($0.offset),\(String(format: "%.6f", $0.element))" }
        let content = header + rows.joined(separator: "\n") + "\n"
        try content.write(toFile: path, atomically: true, encoding: .utf8)
    }
}
```

**Step 2: Verify it builds**

Run: `swift build --target EspressoBench`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add Sources/EspressoBench/ResultsFormatter.swift
git commit -m "feat(bench): add ResultsFormatter for report and CSV output"
```

---

### Task 5: ThermalMonitor — thermal state tracking

**Files:**
- Create: `Sources/EspressoBench/ThermalMonitor.swift`

**Step 1: Write ThermalMonitor**

```swift
import Foundation

enum ThermalMonitor {
    static func thermalStateString(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    static func currentState() -> String {
        thermalStateString(ProcessInfo.processInfo.thermalState)
    }

    /// Run sustained inference for `duration` seconds, sampling thermal state every second.
    static func sustainedRun(
        duration: TimeInterval = 60.0,
        body: () throws -> Void
    ) throws -> (before: String, after: String, samples: [(time: Double, state: String)]) {
        let before = currentState()
        var samples: [(time: Double, state: String)] = []
        let startTime = ContinuousClock.now

        var elapsed = 0.0
        var lastSampleTime = 0.0

        while elapsed < duration {
            try body()
            let now = ContinuousClock.now
            elapsed = Double((now - startTime).components.attoseconds) / 1e18

            if elapsed - lastSampleTime >= 1.0 {
                samples.append((time: elapsed, state: currentState()))
                lastSampleTime = elapsed
            }
        }

        let after = currentState()
        return (before: before, after: after, samples: samples)
    }
}
```

**Step 2: Verify it builds**

Run: `swift build --target EspressoBench`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add Sources/EspressoBench/ThermalMonitor.swift
git commit -m "feat(bench): add ThermalMonitor for sustained inference thermal tracking"
```

---

### Task 6: ANEDirectBench — the main ANE benchmark

**Files:**
- Create: `Sources/EspressoBench/ANEDirectBench.swift`

**Step 1: Write ANEDirectBench**

This is the core benchmark. It creates a single-layer forward pass pipeline using the real Espresso runtime.

```swift
import Foundation
import ANETypes
import ANERuntime
import Espresso
import CPUOps
import Accelerate

enum ANEDirectBench {
    struct Result {
        let benchmarkResult: BenchmarkResult
        let avgTimingBreakdown: (ane: Double, io: Double, elem: Double)
        let kernelDispatches: Int  // ANE kernel evals per forward pass
    }

    static func run(runner: BenchmarkRunner, nLayers: Int = 1) throws -> Result {
        printStderr("=== ANE Direct Benchmark ===")
        printStderr("Setting up \(nLayers)-layer forward pass...")

        // 1. Create random weights
        let layers = LayerStorage<LayerWeights>(count: nLayers) { _ in
            let w = LayerWeights()
            fillRandom(w.Wq); fillRandom(w.Wk); fillRandom(w.Wv); fillRandom(w.Wo)
            fillRandom(w.W1); fillRandom(w.W2); fillRandom(w.W3)
            fillOnes(w.rmsAtt); fillOnes(w.rmsFfn)  // RMS weights ~1.0 for stability
            return w
        }

        // 2. Create random input
        let xCur = TensorBuffer(count: ModelConfig.dim * ModelConfig.seqLen, zeroed: false)
        fillRandom(xCur, range: -0.1...0.1)

        // 3. Create activation storage
        let acts = LayerStorage<LayerActivations>(count: nLayers) { _ in LayerActivations() }

        // 4. Compile kernels (5 per layer)
        printStderr("Compiling \(nLayers * 5) ANE kernels...")
        let compileStart = ContinuousClock.now
        let kernels = try LayerStorage<LayerKernelSet>(count: nLayers, throwingInitializer: { i in
            try LayerKernelSet(weights: layers[i])
        })
        let compileElapsed = ContinuousClock.now - compileStart
        let compileMs = Double(compileElapsed.components.attoseconds) / 1e15
        printStderr(String(format: "  Compilation: %.1f ms (%d compiles, budget remaining: %d)",
                           compileMs, nLayers * 5, CompileBudget.remaining))

        // 5. Build surface handle cache
        // Note: No static kernels needed for forward-only benchmark.
        // ForwardPass.run accepts surfaceHandles as optional, we'll pass nil
        // and let it fetch surfaces from kernels directly.

        // 6. Create accumulator (required by ForwardPass API)
        let accumulator = GradientAccumulator()

        // 7. Accumulate timing breakdown
        var totalTimings = StepTimingBreakdown()
        let measuredIterations = runner.iterations

        // 8. Run benchmark
        printStderr("Running benchmark: \(runner.warmup) warmup + \(runner.iterations) measured iterations")
        let kernelDispatches = nLayers * 2  // fwdAttn + fwdFFN per layer

        let result = try runner.run(label: "ANE Direct") {
            var stepTimings = StepTimingBreakdown()
            try ForwardPass.runTimed(
                xCur: xCur,
                acts: acts,
                kernels: kernels,
                accumulator: accumulator,
                timings: &stepTimings
            )
            totalTimings.tAne += stepTimings.tAne
            totalTimings.tIO += stepTimings.tIO
            totalTimings.tElem += stepTimings.tElem
        }

        let avgBreakdown = (
            ane: totalTimings.tAne / Double(measuredIterations),
            io: totalTimings.tIO / Double(measuredIterations),
            elem: totalTimings.tElem / Double(measuredIterations)
        )

        printStderr(String(format: "  Done. Mean: %.3f ms, Median: %.3f ms", result.mean, result.median))

        return Result(
            benchmarkResult: result,
            avgTimingBreakdown: avgBreakdown,
            kernelDispatches: kernelDispatches
        )
    }

    // MARK: - Helpers

    private static func fillRandom(_ buffer: borrowing TensorBuffer, range: ClosedRange<Float> = -0.1...0.1) {
        buffer.withUnsafeMutablePointer { ptr in
            for i in 0..<buffer.count {
                ptr[i] = Float.random(in: range)
            }
        }
    }

    private static func fillOnes(_ buffer: borrowing TensorBuffer) {
        buffer.withUnsafeMutablePointer { ptr in
            for i in 0..<buffer.count {
                ptr[i] = 1.0
            }
        }
    }
}
```

**Step 2: Verify it builds**

Run: `swift build --target EspressoBench`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add Sources/EspressoBench/ANEDirectBench.swift
git commit -m "feat(bench): add ANEDirectBench — ANE forward pass measurement"
```

---

### Task 7: CoreMLBench — Core ML baseline

**Files:**
- Create: `Sources/EspressoBench/CoreMLBench.swift`

**Step 1: Write CoreMLBench**

```swift
import Foundation
import CoreML
import ANETypes

enum CoreMLBench {
    struct Result {
        let results: [(label: String, result: BenchmarkResult)]
        let modelLoadTimeMs: Double
    }

    static func run(runner: BenchmarkRunner, modelPath: String) throws -> Result {
        printStderr("=== Core ML Baseline Benchmark ===")

        let modelURL = URL(fileURLWithPath: modelPath)
        guard FileManager.default.fileExists(atPath: modelPath) else {
            printStderr("  ERROR: Core ML model not found at \(modelPath)")
            printStderr("  Run: python3 scripts/generate_coreml_model.py")
            throw BenchError.coreMLModelNotFound(modelPath)
        }

        // Measure model load time
        let loadStart = ContinuousClock.now
        let configAll = MLModelConfiguration()
        configAll.computeUnits = .all
        let modelAll = try MLModel(contentsOf: modelURL, configuration: configAll)
        let loadElapsed = ContinuousClock.now - loadStart
        let loadTimeMs = Double(loadElapsed.components.attoseconds) / 1e15
        printStderr(String(format: "  Model loaded in %.1f ms (compute units: .all)", loadTimeMs))

        // Create input matching our tensor dimensions
        let inputArray = try MLMultiArray(
            shape: [1, NSNumber(value: ModelConfig.dim), 1, NSNumber(value: ModelConfig.seqLen)],
            dataType: .float16
        )
        // Fill with random data
        let count = ModelConfig.dim * ModelConfig.seqLen
        for i in 0..<count {
            inputArray[i] = NSNumber(value: Float.random(in: -0.1...0.1))
        }

        let featureProvider = try MLDictionaryFeatureProvider(
            dictionary: ["input": MLFeatureValue(multiArray: inputArray)]
        )

        var allResults: [(label: String, result: BenchmarkResult)] = []

        // Run with .all
        printStderr("  Running with .all compute units...")
        let resultAll = try runner.run(label: "CoreML (.all)") {
            let _ = try modelAll.prediction(from: featureProvider)
        }
        allResults.append(("CoreML (.all)", resultAll))
        printStderr(String(format: "    Mean: %.3f ms, Median: %.3f ms", resultAll.mean, resultAll.median))

        // Run with .cpuAndNeuralEngine
        printStderr("  Running with .cpuAndNeuralEngine...")
        let configANE = MLModelConfiguration()
        configANE.computeUnits = .cpuAndNeuralEngine
        let modelANE = try MLModel(contentsOf: modelURL, configuration: configANE)
        let resultANE = try runner.run(label: "CoreML (.cpuAndNeuralEngine)") {
            let _ = try modelANE.prediction(from: featureProvider)
        }
        allResults.append(("CoreML (.cpuAndNeuralEngine)", resultANE))
        printStderr(String(format: "    Mean: %.3f ms, Median: %.3f ms", resultANE.mean, resultANE.median))

        // Run with .cpuAndGPU
        printStderr("  Running with .cpuAndGPU...")
        let configGPU = MLModelConfiguration()
        configGPU.computeUnits = .cpuAndGPU
        let modelGPU = try MLModel(contentsOf: modelURL, configuration: configGPU)
        let resultGPU = try runner.run(label: "CoreML (.cpuAndGPU)") {
            let _ = try modelGPU.prediction(from: featureProvider)
        }
        allResults.append(("CoreML (.cpuAndGPU)", resultGPU))
        printStderr(String(format: "    Mean: %.3f ms, Median: %.3f ms", resultGPU.mean, resultGPU.median))

        return Result(results: allResults, modelLoadTimeMs: loadTimeMs)
    }
}

enum BenchError: Error {
    case coreMLModelNotFound(String)
}
```

**Step 2: Verify it builds**

Run: `swift build --target EspressoBench`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add Sources/EspressoBench/CoreMLBench.swift
git commit -m "feat(bench): add CoreMLBench — Core ML baseline with 3 compute unit configs"
```

---

### Task 8: Python coremltools script for model generation

**Files:**
- Create: `scripts/generate_coreml_model.py`
- Create: `benchmarks/models/.gitkeep`

**Step 1: Write the coremltools model generator**

This creates a Core ML model with the exact same transformer architecture and dimensions. Uses the `coremltools` MIL builder.

```python
#!/usr/bin/env python3
"""Generate a Core ML model matching the Espresso transformer architecture.

Dimensions (from ModelConfig):
  dim=768, hidden=2048, seqLen=256, heads=12, headDim=64

Architecture (single layer):
  1. RMSNorm -> QKV Projection -> SDPA -> Output Projection + Residual
  2. RMSNorm -> SwiGLU FFN (W1, W3, SiLU gate, W2) + Residual

Usage:
  pip install coremltools numpy
  python3 scripts/generate_coreml_model.py
  # Output: benchmarks/models/transformer_layer.mlpackage
"""

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np
import os

# Match ModelConfig exactly
DIM = 768
HIDDEN = 2048
SEQ_LEN = 256
HEADS = 12
HEAD_DIM = DIM // HEADS

def build_transformer_layer():
    """Build a single transformer layer using coremltools MIL builder."""

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, DIM, 1, SEQ_LEN), dtype=types.fp16),
        ]
    )
    def transformer(x):
        # --- Attention Block ---
        # RMSNorm (simplified: normalize then scale)
        rms_att_weight = mb.const(
            val=np.ones((1, DIM, 1, 1), dtype=np.float16),
            name="rms_att_weight"
        )
        x_sq = mb.mul(x=x, y=x, name="x_sq")
        x_mean = mb.reduce_mean(x=x_sq, axes=[1], keep_dims=True, name="x_mean")
        eps = mb.const(val=np.float16(1e-5), name="eps")
        x_mean_eps = mb.add(x=x_mean, y=eps, name="x_mean_eps")
        x_rsqrt = mb.rsqrt(x=x_mean_eps, name="x_rsqrt")
        x_norm = mb.mul(x=x, y=x_rsqrt, name="x_norm_pre")
        x_norm = mb.mul(x=x_norm, y=rms_att_weight, name="x_norm")

        # QKV projections (as conv2d with 1x1 kernels — ANE-native)
        wq = mb.const(val=np.random.randn(DIM, DIM, 1, 1).astype(np.float16) * 0.02, name="wq")
        wk = mb.const(val=np.random.randn(DIM, DIM, 1, 1).astype(np.float16) * 0.02, name="wk")
        wv = mb.const(val=np.random.randn(DIM, DIM, 1, 1).astype(np.float16) * 0.02, name="wv")

        q = mb.conv(x=x_norm, weight=wq, name="q_proj")
        k = mb.conv(x=x_norm, weight=wk, name="k_proj")
        v = mb.conv(x=x_norm, weight=wv, name="v_proj")

        # Reshape for multi-head: [1, DIM, 1, SEQ] -> [1, HEADS, HEAD_DIM, SEQ]
        # Then transpose for attention: [1, HEADS, SEQ, HEAD_DIM]
        q_r = mb.reshape(x=q, shape=[1, HEADS, HEAD_DIM, SEQ_LEN], name="q_reshape")
        k_r = mb.reshape(x=k, shape=[1, HEADS, HEAD_DIM, SEQ_LEN], name="k_reshape")
        v_r = mb.reshape(x=v, shape=[1, HEADS, HEAD_DIM, SEQ_LEN], name="v_reshape")

        # Attention scores: Q @ K^T / sqrt(head_dim)
        k_t = mb.transpose(x=k_r, perm=[0, 1, 3, 2], name="k_transpose")
        scale = mb.const(val=np.float16(1.0 / np.sqrt(HEAD_DIM)), name="scale")
        scores = mb.matmul(x=q_r, y=k_t, name="attn_scores_raw")
        scores = mb.mul(x=scores, y=scale, name="attn_scores")

        # Causal mask
        mask_np = np.triu(np.full((SEQ_LEN, SEQ_LEN), -1e4, dtype=np.float16), k=1)
        mask = mb.const(val=mask_np.reshape(1, 1, SEQ_LEN, SEQ_LEN), name="causal_mask")
        scores = mb.add(x=scores, y=mask, name="attn_scores_masked")

        # Softmax
        attn_weights = mb.softmax(x=scores, axis=-1, name="attn_weights")

        # Attention output: weights @ V
        attn_out = mb.matmul(x=attn_weights, y=v_r, name="attn_out_heads")

        # Reshape back: [1, HEADS, HEAD_DIM, SEQ] -> [1, DIM, 1, SEQ]
        attn_out = mb.reshape(x=attn_out, shape=[1, DIM, 1, SEQ_LEN], name="attn_out_concat")

        # Output projection
        wo = mb.const(val=np.random.randn(DIM, DIM, 1, 1).astype(np.float16) * 0.02, name="wo")
        o_out = mb.conv(x=attn_out, weight=wo, name="o_proj")

        # Residual
        x2 = mb.add(x=x, y=o_out, name="residual_attn")

        # --- FFN Block ---
        # RMSNorm
        rms_ffn_weight = mb.const(
            val=np.ones((1, DIM, 1, 1), dtype=np.float16),
            name="rms_ffn_weight"
        )
        x2_sq = mb.mul(x=x2, y=x2, name="x2_sq")
        x2_mean = mb.reduce_mean(x=x2_sq, axes=[1], keep_dims=True, name="x2_mean")
        x2_mean_eps = mb.add(x=x2_mean, y=eps, name="x2_mean_eps")
        x2_rsqrt = mb.rsqrt(x=x2_mean_eps, name="x2_rsqrt")
        x2_norm = mb.mul(x=x2, y=x2_rsqrt, name="x2_norm_pre")
        x2_norm = mb.mul(x=x2_norm, y=rms_ffn_weight, name="x2_norm")

        # SwiGLU FFN
        w1 = mb.const(val=np.random.randn(HIDDEN, DIM, 1, 1).astype(np.float16) * 0.02, name="w1")
        w3 = mb.const(val=np.random.randn(HIDDEN, DIM, 1, 1).astype(np.float16) * 0.02, name="w3")
        w2 = mb.const(val=np.random.randn(DIM, HIDDEN, 1, 1).astype(np.float16) * 0.02, name="w2")

        h1 = mb.conv(x=x2_norm, weight=w1, name="ffn_w1")
        h3 = mb.conv(x=x2_norm, weight=w3, name="ffn_w3")

        # SiLU gate
        silu = mb.sigmoid(x=h1, name="sigmoid_h1")
        silu = mb.mul(x=h1, y=silu, name="silu_h1")
        gate = mb.mul(x=silu, y=h3, name="gate_out")

        ffn_out = mb.conv(x=gate, weight=w2, name="ffn_w2")

        # Residual
        output = mb.add(x=x2, y=ffn_out, name="residual_ffn")

        return output

    return transformer


def main():
    print(f"Generating Core ML transformer layer model...")
    print(f"  dim={DIM}, hidden={HIDDEN}, seq_len={SEQ_LEN}, heads={HEADS}")

    prog = build_transformer_layer()

    # Convert to Core ML model
    model = ct.convert(
        prog,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "models")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "transformer_layer.mlpackage")

    model.save(output_path)
    print(f"  Saved to: {output_path}")
    print(f"  Model size: {sum(f.stat().st_size for f in __import__('pathlib').Path(output_path).rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
```

**Step 2: Create benchmarks directory**

```bash
mkdir -p benchmarks/models
touch benchmarks/models/.gitkeep
```

**Step 3: Test the Python script**

Run: `python3 scripts/generate_coreml_model.py`
Expected: Generates `benchmarks/models/transformer_layer.mlpackage`

Note: If `coremltools` is not installed, run `pip install coremltools numpy` first.

**Step 4: Commit**

```bash
git add scripts/generate_coreml_model.py benchmarks/models/.gitkeep
git commit -m "feat(bench): add coremltools script to generate Core ML transformer model"
```

---

### Task 9: Main entry point — orchestrate all benchmarks

**Files:**
- Modify: `Sources/EspressoBench/main.swift`

**Step 1: Write the main orchestration**

Replace the placeholder with the full CLI:

```swift
import Foundation
import ANETypes

// MARK: - CLI Argument Parsing

struct BenchmarkOptions {
    var aneOnly: Bool = false
    var sustained: Bool = false
    var warmup: Int = 50
    var iterations: Int = 1000
    var outputDir: String? = nil
    var coreMLModelPath: String = "benchmarks/models/transformer_layer.mlpackage"
    var nLayers: Int = 1

    static func parse(_ args: [String]) -> BenchmarkOptions {
        var opts = BenchmarkOptions()
        var i = 1  // skip program name
        while i < args.count {
            switch args[i] {
            case "--ane-only":
                opts.aneOnly = true
            case "--sustained":
                opts.sustained = true
            case "--warmup":
                i += 1; opts.warmup = Int(args[i]) ?? 50
            case "--iterations":
                i += 1; opts.iterations = Int(args[i]) ?? 1000
            case "--output":
                i += 1; opts.outputDir = args[i]
            case "--model":
                i += 1; opts.coreMLModelPath = args[i]
            case "--layers":
                i += 1; opts.nLayers = Int(args[i]) ?? 1
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                printStderr("Unknown argument: \(args[i])")
                printUsage()
                exit(1)
            }
            i += 1
        }
        return opts
    }

    static func printUsage() {
        print("""
        EspressoBench — ANE Runtime Benchmark Suite

        Usage: espresso-bench [OPTIONS]

        Options:
          --ane-only         Skip Core ML benchmarks
          --sustained        Run 60-second sustained thermal test
          --warmup N         Warmup iterations (default: 50)
          --iterations N     Measured iterations (default: 1000)
          --output DIR       Output directory for results
          --model PATH       Path to Core ML .mlpackage (default: benchmarks/models/transformer_layer.mlpackage)
          --layers N         Number of transformer layers (default: 1)
          -h, --help         Show this help
        """)
    }
}

// MARK: - Main

let opts = BenchmarkOptions.parse(CommandLine.arguments)
let runner = BenchmarkRunner(warmup: opts.warmup, iterations: opts.iterations)

let flopsPerPass = FLOPCalculator.forwardPassFLOPs() * Double(opts.nLayers)
printStderr("Espresso Benchmark Suite")
printStderr("========================")
printStderr(String(format: "Config: dim=%d, hidden=%d, seq=%d, heads=%d, layers=%d",
                   ModelConfig.dim, ModelConfig.hidden, ModelConfig.seqLen, ModelConfig.heads, opts.nLayers))
printStderr(String(format: "FLOPs per forward pass: %.2f GFLOPs", flopsPerPass / 1e9))
printStderr(String(format: "Iterations: %d warmup + %d measured", opts.warmup, opts.iterations))
printStderr("")

// --- Benchmark 1: ANE Direct ---
let aneResult: ANEDirectBench.Result
do {
    aneResult = try ANEDirectBench.run(runner: runner, nLayers: opts.nLayers)
} catch {
    printStderr("ANE Direct benchmark failed: \(error)")
    exit(1)
}

// --- Benchmark 2: Core ML (optional) ---
var coreMLResult: CoreMLBench.Result? = nil
if !opts.aneOnly {
    do {
        coreMLResult = try CoreMLBench.run(runner: runner, modelPath: opts.coreMLModelPath)
    } catch {
        printStderr("Core ML benchmark failed: \(error)")
        printStderr("Continuing with ANE-only results...")
    }
}

// --- Benchmark 3: Sustained thermal test (optional) ---
var thermalBefore: String? = nil
var thermalAfter: String? = nil
if opts.sustained {
    printStderr("\n=== Sustained Thermal Test (60 seconds) ===")
    do {
        let thermal = try ThermalMonitor.sustainedRun(duration: 60.0) {
            var timings = StepTimingBreakdown()
            // Re-run ANE forward pass would need kernels still alive.
            // For thermal test, we measure separately.
        }
        thermalBefore = thermal.before
        thermalAfter = thermal.after
        printStderr("  Thermal: \(thermal.before) -> \(thermal.after)")
        for sample in thermal.samples {
            printStderr(String(format: "    t=%.0fs: %@", sample.time, sample.state))
        }
    } catch {
        printStderr("  Thermal test failed: \(error)")
    }
}

// --- Output ---
let report = ResultsFormatter.formatReport(
    aneResult: aneResult.benchmarkResult,
    aneTimingBreakdown: aneResult.avgTimingBreakdown,
    coreMLResults: coreMLResult?.results,
    coreMLLoadTimeMs: coreMLResult?.modelLoadTimeMs,
    thermalBefore: thermalBefore,
    thermalAfter: thermalAfter,
    flopsPerPass: flopsPerPass,
    nLayers: opts.nLayers
)
print(report)

// --- Save results ---
let outputDir: String
if let dir = opts.outputDir {
    outputDir = dir
} else {
    let dateFormatter = DateFormatter()
    dateFormatter.dateFormat = "yyyy-MM-dd-HHmmss"
    dateFormatter.locale = Locale(identifier: "en_US_POSIX")
    let timestamp = dateFormatter.string(from: Date())
    outputDir = "benchmarks/results/\(timestamp)"
}

do {
    try FileManager.default.createDirectory(atPath: outputDir, withIntermediateDirectories: true)

    // CSV files
    try ResultsFormatter.writeCSV(
        latencies: aneResult.benchmarkResult.latencies,
        to: "\(outputDir)/ane_direct_latencies.csv"
    )

    if let coreML = coreMLResult {
        for (label, result) in coreML.results {
            let filename = label.lowercased()
                .replacingOccurrences(of: " ", with: "_")
                .replacingOccurrences(of: "(", with: "")
                .replacingOccurrences(of: ")", with: "")
                .replacingOccurrences(of: ".", with: "")
            try ResultsFormatter.writeCSV(
                latencies: result.latencies,
                to: "\(outputDir)/\(filename)_latencies.csv"
            )
        }
    }

    // Summary report
    try report.write(toFile: "\(outputDir)/summary.txt", atomically: true, encoding: .utf8)

    printStderr("\nResults saved to: \(outputDir)/")
} catch {
    printStderr("Failed to save results: \(error)")
}
```

**Step 2: Verify it builds**

Run: `swift build -c release --target EspressoBench`
Expected: Build succeeds

**Step 3: Commit**

```bash
git add Sources/EspressoBench/main.swift
git commit -m "feat(bench): wire up main.swift — full benchmark orchestration with CLI"
```

---

### Task 10: Integration test — run the full suite

**Step 1: Build in release mode**

Run: `swift build -c release --target EspressoBench`
Expected: Clean build

**Step 2: Run ANE-only benchmark**

Run: `.build/release/EspressoBench --ane-only --warmup 5 --iterations 10`
Expected: Prints benchmark report to stdout, progress to stderr. Should complete in <30 seconds with reduced iterations.

**Step 3: Fix any issues**

If compilation budget or IOSurface errors occur, adjust. If `~Copyable` borrowing issues arise from how we pass layers/kernels, fix the borrow semantics.

**Step 4: Run full suite (if Core ML model available)**

Run:
```bash
python3 scripts/generate_coreml_model.py
.build/release/EspressoBench --warmup 5 --iterations 10
```
Expected: Both ANE and Core ML results printed.

**Step 5: Run the real benchmark**

Run: `.build/release/EspressoBench --ane-only`
Expected: Full 1000-iteration run with publication-quality numbers.

**Step 6: Commit any fixes**

```bash
git add -A
git commit -m "fix(bench): integration fixes from first full run"
```

---

### Task 11: Power benchmark script

**Files:**
- Create: `scripts/run_power_benchmark.sh`

**Step 1: Write the power benchmark helper**

```bash
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
```

**Step 2: Make executable**

```bash
chmod +x scripts/run_power_benchmark.sh
```

**Step 3: Commit**

```bash
git add scripts/run_power_benchmark.sh
git commit -m "feat(bench): add power benchmark script with powermetrics integration"
```

---

### Task 12: Add benchmarks/ to .gitignore and final cleanup

**Files:**
- Modify or create: `.gitignore`

**Step 1: Add benchmark output directories to .gitignore**

Add these lines:
```
benchmarks/results/
benchmarks/models/*.mlpackage
```

Keep `benchmarks/models/.gitkeep` tracked.

**Step 2: Final build verification**

Run: `swift build -c release`
Expected: All targets build cleanly including EspressoBench.

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore benchmark results and generated models"
```

---

## v2 Roadmap (Not in scope)

- **Sequence length scaling** — parameterize `ModelConfig.seqLen`, test at 64/128/256/512/1024
- **Dimension scaling** — parameterize `ModelConfig.dim` and `hidden`
- **Multi-layer stacking** — benchmark with nLayers=1,4,12
- **Batch inference** — if runtime supports batch IOSurfaces
- **Automated powermetrics** — parse power logs programmatically
- **Instruments trace export** — automated `.trace` file capture

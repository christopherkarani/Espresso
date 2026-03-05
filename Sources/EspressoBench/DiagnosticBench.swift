import Foundation
import IOSurface
import ANETypes
import ANERuntime
import Espresso
import os

/// Diagnostic benchmark harness for Phase 2 experiments.
///
/// Three experiments to isolate where inference overhead lives:
/// 1. **Convs-Only**: Pure convolution throughput (compute floor)
/// 2. **Batched-Head**: Batched reshape→transpose→matmul vs split-head
/// 3. **Whole-Model**: All layers in one kernel (dispatch overhead)
enum DiagnosticBench {

    // MARK: - Experiment 1: Convs-Only Baseline

    static func runConvsOnly(warmup: Int, iterations: Int, nLayers: Int) throws -> ANEDirectBench.Result {
        printStderr("\n=== Diagnostic Exp 1: Convs-Only Baseline ===")
        printStderr("Setting up \(nLayers)-layer convs-only forward pass...")

        let signposter = OSSignposter(subsystem: "com.espresso.bench", category: .pointsOfInterest)

        // 1. Create random weights
        let layers = LayerStorage<LayerWeights>(count: nLayers) { _ in
            let w = LayerWeights()
            ANEDirectBench.randomFill(w.Wq); ANEDirectBench.randomFill(w.Wk); ANEDirectBench.randomFill(w.Wv); ANEDirectBench.randomFill(w.Wo)
            ANEDirectBench.randomFill(w.W1); ANEDirectBench.randomFill(w.W2); ANEDirectBench.randomFill(w.W3)
            ANEDirectBench.onesFill(w.rmsAtt); ANEDirectBench.onesFill(w.rmsFfn)
            return w
        }

        // 2. Create random input
        let xCur = TensorBuffer(count: ModelConfig.dim * ModelConfig.seqLen, zeroed: false)
        ANEDirectBench.randomFill(xCur, range: -0.1...0.1)

        // 3. Compile convs-only kernels (1 per layer)
        printStderr("Compiling \(nLayers) convs-only ANE kernels...")
        let compileStart = ContinuousClock.now
        let kernels = try LayerStorage<ConvsOnlyKernel>(count: nLayers, throwingInitializer: { i in
            try ConvsOnlyKernel(weights: layers[i])
        })
        let compileMs = durationMs(ContinuousClock.now - compileStart)
        printStderr(String(format: "  Compilation: %.1f ms (budget remaining: %d)", compileMs, CompileBudget.remaining))

        // 4. Pre-resolve IOSurface handles
        var inputSurfaces: [IOSurfaceRef] = []
        var outputSurfaces: [IOSurfaceRef] = []
        inputSurfaces.reserveCapacity(nLayers)
        outputSurfaces.reserveCapacity(nLayers)
        for i in 0..<nLayers {
            inputSurfaces.append(try kernels[i].kernel.inputSurface(at: 0))
            outputSurfaces.append(try kernels[i].kernel.outputSurface(at: 0))
        }

        let dim = ModelConfig.dim
        let seqLen = ModelConfig.seqLen

        // 5. Warmup
        printStderr("Warmup: \(warmup) iterations...")
        for _ in 0..<warmup {
            for L in 0..<nLayers {
                xCur.withUnsafeBufferPointer { xBuf in
                    SurfaceIO.writeFP16(to: inputSurfaces[L], data: xBuf, channels: dim, spatial: seqLen)
                }
                try kernels[L].kernel.eval()
                xCur.withUnsafeMutableBufferPointer { xBuf in
                    SurfaceIO.readFP16(from: outputSurfaces[L], into: xBuf, channelOffset: 0, channels: dim, spatial: seqLen)
                }
            }
        }

        // 6. Measured iterations
        printStderr("Measuring: \(iterations) iterations...")
        var latencies: [Double] = []
        latencies.reserveCapacity(iterations)
        var totalTimings = StepTimingBreakdown()

        for iter in 0..<iterations {
            let state = signposter.beginInterval("ConvsOnlyForwardPass")
            let start = ContinuousClock.now

            for L in 0..<nLayers {
                var t0 = ContinuousClock.now
                xCur.withUnsafeBufferPointer { xBuf in
                    SurfaceIO.writeFP16(to: inputSurfaces[L], data: xBuf, channels: dim, spatial: seqLen)
                }
                totalTimings.tIO += durationMs(ContinuousClock.now - t0)

                t0 = ContinuousClock.now
                try kernels[L].kernel.eval()
                totalTimings.tAne += durationMs(ContinuousClock.now - t0)

                t0 = ContinuousClock.now
                xCur.withUnsafeMutableBufferPointer { xBuf in
                    SurfaceIO.readFP16(from: outputSurfaces[L], into: xBuf, channelOffset: 0, channels: dim, spatial: seqLen)
                }
                totalTimings.tIO += durationMs(ContinuousClock.now - t0)
            }

            let ms = durationMs(ContinuousClock.now - start)
            signposter.endInterval("ConvsOnlyForwardPass", state)
            latencies.append(ms)

            if (iter + 1) % 100 == 0 {
                let currentMean = latencies.reduce(0, +) / Double(latencies.count)
                printStderr(String(format: "  [Convs-Only] %d/%d — mean: %.3f ms", iter + 1, iterations, currentMean))
            }
        }

        let result = BenchmarkResult(
            label: "Convs-Only Baseline",
            latencies: latencies,
            warmupCount: warmup,
            iterationCount: iterations
        )

        let n = Double(iterations)
        let avgBreakdown = (
            ane: totalTimings.tAne / n,
            io: totalTimings.tIO / n,
            elem: 0.0
        )

        printStderr(String(format: "  Done. Mean: %.3f ms, Median: %.3f ms", result.mean, result.median))

        return ANEDirectBench.Result(
            benchmarkResult: result,
            avgTimingBreakdown: avgBreakdown,
            compileTimeMs: compileMs
        )
    }

    // MARK: - Experiment 2: Batched-Head Fused

    static func runBatchedHead(warmup: Int, iterations: Int, nLayers: Int) throws -> ANEDirectBench.Result {
        printStderr("\n=== Diagnostic Exp 2: Batched-Head Fused ===")
        printStderr("Setting up \(nLayers)-layer batched-head fused inference...")

        let signposter = OSSignposter(subsystem: "com.espresso.bench", category: .pointsOfInterest)

        // 1. Create random weights
        let layers = LayerStorage<LayerWeights>(count: nLayers) { _ in
            let w = LayerWeights()
            ANEDirectBench.randomFill(w.Wq); ANEDirectBench.randomFill(w.Wk); ANEDirectBench.randomFill(w.Wv); ANEDirectBench.randomFill(w.Wo)
            ANEDirectBench.randomFill(w.W1); ANEDirectBench.randomFill(w.W2); ANEDirectBench.randomFill(w.W3)
            ANEDirectBench.onesFill(w.rmsAtt); ANEDirectBench.onesFill(w.rmsFfn)
            return w
        }

        // 2. Create random input
        let xCur = TensorBuffer(count: ModelConfig.dim * ModelConfig.seqLen, zeroed: false)
        ANEDirectBench.randomFill(xCur, range: -0.1...0.1)

        // 3. Compile batched-head kernels
        printStderr("Compiling \(nLayers) batched-head fused ANE kernels...")
        let compileStart = ContinuousClock.now
        let kernels = try LayerStorage<BatchedHeadFusedKernel>(count: nLayers, throwingInitializer: { i in
            try BatchedHeadFusedKernel(weights: layers[i])
        })
        let compileMs = durationMs(ContinuousClock.now - compileStart)
        printStderr(String(format: "  Compilation: %.1f ms (budget remaining: %d)", compileMs, CompileBudget.remaining))

        // 4. Pre-resolve IOSurface handles
        var inputSurfaces: [IOSurfaceRef] = []
        var outputSurfaces: [IOSurfaceRef] = []
        inputSurfaces.reserveCapacity(nLayers)
        outputSurfaces.reserveCapacity(nLayers)
        for i in 0..<nLayers {
            inputSurfaces.append(try kernels[i].kernel.inputSurface(at: 0))
            outputSurfaces.append(try kernels[i].kernel.outputSurface(at: 0))
        }

        let dim = ModelConfig.dim
        let seqLen = ModelConfig.seqLen

        // 5. Warmup
        printStderr("Warmup: \(warmup) iterations...")
        for _ in 0..<warmup {
            for L in 0..<nLayers {
                xCur.withUnsafeBufferPointer { xBuf in
                    SurfaceIO.writeFP16(to: inputSurfaces[L], data: xBuf, channels: dim, spatial: seqLen)
                }
                try kernels[L].kernel.eval()
                xCur.withUnsafeMutableBufferPointer { xBuf in
                    SurfaceIO.readFP16(from: outputSurfaces[L], into: xBuf, channelOffset: 0, channels: dim, spatial: seqLen)
                }
            }
        }

        // 6. Measured iterations
        printStderr("Measuring: \(iterations) iterations...")
        var latencies: [Double] = []
        latencies.reserveCapacity(iterations)
        var totalTimings = StepTimingBreakdown()

        for iter in 0..<iterations {
            let state = signposter.beginInterval("BatchedHeadForwardPass")
            let start = ContinuousClock.now

            for L in 0..<nLayers {
                var t0 = ContinuousClock.now
                xCur.withUnsafeBufferPointer { xBuf in
                    SurfaceIO.writeFP16(to: inputSurfaces[L], data: xBuf, channels: dim, spatial: seqLen)
                }
                totalTimings.tIO += durationMs(ContinuousClock.now - t0)

                t0 = ContinuousClock.now
                try kernels[L].kernel.eval()
                totalTimings.tAne += durationMs(ContinuousClock.now - t0)

                t0 = ContinuousClock.now
                xCur.withUnsafeMutableBufferPointer { xBuf in
                    SurfaceIO.readFP16(from: outputSurfaces[L], into: xBuf, channelOffset: 0, channels: dim, spatial: seqLen)
                }
                totalTimings.tIO += durationMs(ContinuousClock.now - t0)
            }

            let ms = durationMs(ContinuousClock.now - start)
            signposter.endInterval("BatchedHeadForwardPass", state)
            latencies.append(ms)

            if (iter + 1) % 100 == 0 {
                let currentMean = latencies.reduce(0, +) / Double(latencies.count)
                printStderr(String(format: "  [Batched-Head] %d/%d — mean: %.3f ms", iter + 1, iterations, currentMean))
            }
        }

        let result = BenchmarkResult(
            label: "Batched-Head Fused",
            latencies: latencies,
            warmupCount: warmup,
            iterationCount: iterations
        )

        let n = Double(iterations)
        let avgBreakdown = (
            ane: totalTimings.tAne / n,
            io: totalTimings.tIO / n,
            elem: 0.0
        )

        printStderr(String(format: "  Done. Mean: %.3f ms, Median: %.3f ms", result.mean, result.median))

        return ANEDirectBench.Result(
            benchmarkResult: result,
            avgTimingBreakdown: avgBreakdown,
            compileTimeMs: compileMs
        )
    }

    // MARK: - Experiment 3: Whole-Model Single Kernel

    static func runWholeModel(warmup: Int, iterations: Int, nLayers: Int) throws -> ANEDirectBench.Result {
        printStderr("\n=== Diagnostic Exp 3: Whole-Model Single Kernel ===")
        printStderr("Setting up \(nLayers)-layer whole-model kernel...")

        let signposter = OSSignposter(subsystem: "com.espresso.bench", category: .pointsOfInterest)

        // 1. Create random weights for all layers
        let layers = LayerStorage<LayerWeights>(count: nLayers) { _ in
            let w = LayerWeights()
            ANEDirectBench.randomFill(w.Wq); ANEDirectBench.randomFill(w.Wk); ANEDirectBench.randomFill(w.Wv); ANEDirectBench.randomFill(w.Wo)
            ANEDirectBench.randomFill(w.W1); ANEDirectBench.randomFill(w.W2); ANEDirectBench.randomFill(w.W3)
            ANEDirectBench.onesFill(w.rmsAtt); ANEDirectBench.onesFill(w.rmsFfn)
            return w
        }

        // 2. Create random input
        let xCur = TensorBuffer(count: ModelConfig.dim * ModelConfig.seqLen, zeroed: false)
        ANEDirectBench.randomFill(xCur, range: -0.1...0.1)

        // 3. Compile ONE whole-model kernel
        printStderr("Compiling 1 whole-model ANE kernel (\(nLayers) layers fused)...")
        let compileStart = ContinuousClock.now
        let kernel = try WholeModelKernel(layers: layers, nLayers: nLayers)
        let compileMs = durationMs(ContinuousClock.now - compileStart)
        printStderr(String(format: "  Compilation: %.1f ms (budget remaining: %d)", compileMs, CompileBudget.remaining))

        // 4. Pre-resolve single input + output surface
        let inputSurface = try kernel.kernel.inputSurface(at: 0)
        let outputSurface = try kernel.kernel.outputSurface(at: 0)

        let dim = ModelConfig.dim
        let seqLen = ModelConfig.seqLen

        // 5. Warmup
        printStderr("Warmup: \(warmup) iterations...")
        for _ in 0..<warmup {
            xCur.withUnsafeBufferPointer { xBuf in
                SurfaceIO.writeFP16(to: inputSurface, data: xBuf, channels: dim, spatial: seqLen)
            }
            try kernel.kernel.eval()
            xCur.withUnsafeMutableBufferPointer { xBuf in
                SurfaceIO.readFP16(from: outputSurface, into: xBuf, channelOffset: 0, channels: dim, spatial: seqLen)
            }
        }

        // 6. Measured iterations
        printStderr("Measuring: \(iterations) iterations...")
        var latencies: [Double] = []
        latencies.reserveCapacity(iterations)
        var totalTimings = StepTimingBreakdown()

        for iter in 0..<iterations {
            let state = signposter.beginInterval("WholeModelForwardPass")
            let start = ContinuousClock.now

            var t0 = ContinuousClock.now
            xCur.withUnsafeBufferPointer { xBuf in
                SurfaceIO.writeFP16(to: inputSurface, data: xBuf, channels: dim, spatial: seqLen)
            }
            totalTimings.tIO += durationMs(ContinuousClock.now - t0)

            t0 = ContinuousClock.now
            try kernel.kernel.eval()
            totalTimings.tAne += durationMs(ContinuousClock.now - t0)

            t0 = ContinuousClock.now
            xCur.withUnsafeMutableBufferPointer { xBuf in
                SurfaceIO.readFP16(from: outputSurface, into: xBuf, channelOffset: 0, channels: dim, spatial: seqLen)
            }
            totalTimings.tIO += durationMs(ContinuousClock.now - t0)

            let ms = durationMs(ContinuousClock.now - start)
            signposter.endInterval("WholeModelForwardPass", state)
            latencies.append(ms)

            if (iter + 1) % 100 == 0 {
                let currentMean = latencies.reduce(0, +) / Double(latencies.count)
                printStderr(String(format: "  [Whole-Model] %d/%d — mean: %.3f ms", iter + 1, iterations, currentMean))
            }
        }

        let result = BenchmarkResult(
            label: "Whole-Model (\(nLayers)L)",
            latencies: latencies,
            warmupCount: warmup,
            iterationCount: iterations
        )

        let n = Double(iterations)
        let avgBreakdown = (
            ane: totalTimings.tAne / n,
            io: totalTimings.tIO / n,
            elem: 0.0
        )

        printStderr(String(format: "  Done. Mean: %.3f ms, Median: %.3f ms", result.mean, result.median))
        printStderr(String(format: "  Compile time: %.1f ms (single kernel for %d layers)", compileMs, nLayers))

        return ANEDirectBench.Result(
            benchmarkResult: result,
            avgTimingBreakdown: avgBreakdown,
            compileTimeMs: compileMs
        )
    }
}

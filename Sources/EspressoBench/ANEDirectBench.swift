import Foundation
import ANETypes
import ANERuntime
import Espresso
import os

enum ANEDirectBench {
    struct Result {
        let benchmarkResult: BenchmarkResult
        let avgTimingBreakdown: (ane: Double, io: Double, elem: Double)
        let compileTimeMs: Double
        let kernelProfile: InferenceKernelProfile?
    }

    /// Main ANE benchmark. Inlines measurement loop to avoid ~Copyable closure captures.
    static func run(warmup: Int, iterations: Int, nLayers: Int = 1) throws -> Result {
        printStderr("=== ANE Direct Benchmark ===")
        printStderr("Setting up \(nLayers)-layer forward pass...")

        let signposter = OSSignposter(subsystem: "com.espresso.bench", category: .pointsOfInterest)

        // 1. Create random weights
        let layers = LayerStorage<LayerWeights>(count: nLayers) { _ in
            let w = LayerWeights()
            randomFill(w.Wq); randomFill(w.Wk); randomFill(w.Wv); randomFill(w.Wo)
            randomFill(w.W1); randomFill(w.W2); randomFill(w.W3)
            onesFill(w.rmsAtt); onesFill(w.rmsFfn)
            return w
        }

        // 2. Create random input
        let xCur = TensorBuffer(count: ModelConfig.dim * ModelConfig.seqLen, zeroed: false)
        randomFill(xCur, range: -0.1...0.1)

        // 3. Create activation storage
        let acts = LayerStorage<LayerActivations>(count: nLayers) { _ in LayerActivations() }

        // 4. Compile kernels
        printStderr("Compiling \(nLayers * ModelConfig.kernelsPerLayer) ANE kernels...")
        let compileStart = ContinuousClock.now
        let kernels = try LayerStorage<LayerKernelSet>(count: nLayers, throwingInitializer: { i in
            try LayerKernelSet(weights: layers[i])
        })
        let compileMs = durationMs(ContinuousClock.now - compileStart)
        printStderr(String(format: "  Compilation: %.1f ms (budget remaining: %d)", compileMs, CompileBudget.remaining))

        // 5. Gradient accumulator (required by ForwardPass API, unused in fwd-only)
        let accumulator = GradientAccumulator()

        // 6. Warmup — absorbs JIT, cache warming, first-run overhead
        printStderr("Warmup: \(warmup) iterations...")
        for _ in 0..<warmup {
            var timings = StepTimingBreakdown()
            try ForwardPass.runTimed(
                xCur: xCur,
                acts: acts,
                kernels: kernels,
                accumulator: accumulator,
                timings: &timings
            )
        }

        // 7. Measured iterations
        printStderr("Measuring: \(iterations) iterations...")
        var latencies: [Double] = []
        latencies.reserveCapacity(iterations)
        var totalTimings = StepTimingBreakdown()

        for i in 0..<iterations {
            var stepTimings = StepTimingBreakdown()
            let state = signposter.beginInterval("ForwardPass")
            let start = ContinuousClock.now

            try ForwardPass.runTimed(
                xCur: xCur,
                acts: acts,
                kernels: kernels,
                accumulator: accumulator,
                timings: &stepTimings
            )

            let ms = durationMs(ContinuousClock.now - start)
            signposter.endInterval("ForwardPass", state)

            latencies.append(ms)
            totalTimings.tAne += stepTimings.tAne
            totalTimings.tIO += stepTimings.tIO
            totalTimings.tElem += stepTimings.tElem

            if (i + 1) % 100 == 0 {
                let currentMean = latencies.reduce(0, +) / Double(latencies.count)
                printStderr(String(format: "  [ANE Direct] %d/%d — mean: %.3f ms", i + 1, iterations, currentMean))
            }
        }

        let result = BenchmarkResult(
            label: "ANE Direct",
            latencies: latencies,
            warmupCount: warmup,
            iterationCount: iterations
        )

        let n = Double(iterations)
        let avgBreakdown = (
            ane: totalTimings.tAne / n,
            io: totalTimings.tIO / n,
            elem: totalTimings.tElem / n
        )

        printStderr(String(format: "  Done. Mean: %.3f ms, Median: %.3f ms", result.mean, result.median))

        return Result(
            benchmarkResult: result,
            avgTimingBreakdown: avgBreakdown,
            compileTimeMs: compileMs,
            kernelProfile: nil
        )
    }

    /// Inference-optimized benchmark using fused-residual kernels.
    /// Only 2 kernels per layer (vs 5 for training), smaller output surfaces.
    static func runInference(
        warmup: Int,
        iterations: Int,
        nLayers: Int = 1,
        handoff: ForwardPass.InferenceInterKernelHandoff = .cpuRoundTrip,
        profileKernels: Bool = false
    ) throws -> Result {
        printStderr("\n=== ANE Inference Benchmark (Fused Residuals) ===")
        printStderr("Setting up \(nLayers)-layer inference forward pass...")

        let signposter = OSSignposter(subsystem: "com.espresso.bench", category: .pointsOfInterest)

        // 1. Create random weights
        let layers = LayerStorage<LayerWeights>(count: nLayers) { _ in
            let w = LayerWeights()
            randomFill(w.Wq); randomFill(w.Wk); randomFill(w.Wv); randomFill(w.Wo)
            randomFill(w.W1); randomFill(w.W2); randomFill(w.W3)
            onesFill(w.rmsAtt); onesFill(w.rmsFfn)
            return w
        }

        // 2. Create random input
        let xCur = TensorBuffer(count: ModelConfig.dim * ModelConfig.seqLen, zeroed: false)
        randomFill(xCur, range: -0.1...0.1)

        // 3. Compile inference kernels (2 per layer, not 5)
        printStderr("Compiling \(nLayers * 2) inference ANE kernels...")
        let compileStart = ContinuousClock.now
        let kernels = try LayerStorage<InferenceKernelSet>(count: nLayers, throwingInitializer: { i in
            try InferenceKernelSet(weights: layers[i])
        })
        let compileMs = durationMs(ContinuousClock.now - compileStart)
        printStderr(String(format: "  Compilation: %.1f ms (budget remaining: %d)", compileMs, CompileBudget.remaining))

        // 4. Pre-resolve IOSurface handles (removes CFRetain noise from hot loop)
        var surfaceHandles: [InferenceSurfaceHandles] = []
        surfaceHandles.reserveCapacity(nLayers)
        for i in 0..<nLayers {
            surfaceHandles.append(try InferenceSurfaceHandles(kernels: kernels[i]))
        }

        // 5. Warmup
        printStderr("Warmup: \(warmup) iterations...")
        for _ in 0..<warmup {
            var timings = StepTimingBreakdown()
            try ForwardPass.runInferenceTimed(
                xCur: xCur,
                kernels: kernels,
                surfaceHandles: surfaceHandles,
                handoff: handoff,
                profiler: nil,
                timings: &timings
            )
        }

        // 6. Measured iterations
        printStderr("Measuring: \(iterations) iterations...")
        var latencies: [Double] = []
        latencies.reserveCapacity(iterations)
        var totalTimings = StepTimingBreakdown()

        let profiler = profileKernels ? InferenceKernelProfiler(layerCount: nLayers, reservedSamplesPerLayer: iterations) : nil

        for i in 0..<iterations {
            var stepTimings = StepTimingBreakdown()
            let state = signposter.beginInterval("InferenceForwardPass")
            let start = ContinuousClock.now

            try ForwardPass.runInferenceTimed(
                xCur: xCur,
                kernels: kernels,
                surfaceHandles: surfaceHandles,
                handoff: handoff,
                profiler: profiler,
                timings: &stepTimings
            )

            let ms = durationMs(ContinuousClock.now - start)
            signposter.endInterval("InferenceForwardPass", state)

            latencies.append(ms)
            totalTimings.tAne += stepTimings.tAne
            totalTimings.tIO += stepTimings.tIO
            totalTimings.tElem += stepTimings.tElem

            if (i + 1) % 100 == 0 {
                let currentMean = latencies.reduce(0, +) / Double(latencies.count)
                printStderr(String(format: "  [ANE Inference] %d/%d — mean: %.3f ms", i + 1, iterations, currentMean))
            }
        }

        let result = BenchmarkResult(
            label: "ANE Inference",
            latencies: latencies,
            warmupCount: warmup,
            iterationCount: iterations
        )

        let n = Double(iterations)
        let avgBreakdown = (
            ane: totalTimings.tAne / n,
            io: totalTimings.tIO / n,
            elem: totalTimings.tElem / n
        )

        printStderr(String(format: "  Done. Mean: %.3f ms, Median: %.3f ms", result.mean, result.median))

        return Result(
            benchmarkResult: result,
            avgTimingBreakdown: avgBreakdown,
            compileTimeMs: compileMs,
            kernelProfile: profiler?.profile
        )
    }

    /// Sustained inference for thermal monitoring. Creates its own kernel set.
    static func runSustained(
        duration: TimeInterval,
        nLayers: Int = 1
    ) throws -> (before: String, after: String, samples: [(time: Double, state: String)], iterations: Int) {
        printStderr("Setting up sustained thermal test...")

        let layers = LayerStorage<LayerWeights>(count: nLayers) { _ in
            let w = LayerWeights()
            randomFill(w.Wq); randomFill(w.Wk); randomFill(w.Wv); randomFill(w.Wo)
            randomFill(w.W1); randomFill(w.W2); randomFill(w.W3)
            onesFill(w.rmsAtt); onesFill(w.rmsFfn)
            return w
        }

        let xCur = TensorBuffer(count: ModelConfig.dim * ModelConfig.seqLen, zeroed: false)
        randomFill(xCur, range: -0.1...0.1)

        let acts = LayerStorage<LayerActivations>(count: nLayers) { _ in LayerActivations() }
        let kernels = try LayerStorage<LayerKernelSet>(count: nLayers, throwingInitializer: { i in
            try LayerKernelSet(weights: layers[i])
        })
        let accumulator = GradientAccumulator()

        printStderr("Running sustained inference for \(Int(duration))s...")

        let before = ThermalMonitor.currentState()
        var samples: [(time: Double, state: String)] = []
        let startTime = ContinuousClock.now
        var lastSampleTime = 0.0
        var elapsed = 0.0
        var iterCount = 0

        while elapsed < duration {
            var timings = StepTimingBreakdown()
            try ForwardPass.runTimed(
                xCur: xCur,
                acts: acts,
                kernels: kernels,
                accumulator: accumulator,
                timings: &timings
            )
            iterCount += 1

            elapsed = durationMs(ContinuousClock.now - startTime) / 1000.0

            if elapsed - lastSampleTime >= 1.0 {
                let state = ThermalMonitor.currentState()
                samples.append((time: elapsed, state: state))
                lastSampleTime = elapsed
            }
        }

        let after = ThermalMonitor.currentState()
        printStderr("  Completed \(iterCount) forward passes in \(Int(elapsed))s")
        printStderr("  Thermal: \(before) -> \(after)")

        return (before: before, after: after, samples: samples, iterations: iterCount)
    }

    // MARK: - Helpers

    private static func randomFill(_ buffer: borrowing TensorBuffer, range: ClosedRange<Float> = -0.1...0.1) {
        buffer.withUnsafeMutablePointer { ptr in
            for i in 0..<buffer.count {
                ptr[i] = Float.random(in: range)
            }
        }
    }

    private static func onesFill(_ buffer: borrowing TensorBuffer) {
        buffer.withUnsafeMutablePointer { ptr in
            for i in 0..<buffer.count {
                ptr[i] = 1.0
            }
        }
    }
}

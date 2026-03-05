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
        let decodeKernelProfile: DecodeKernelProfile?
        let tokensPerSecond: Double?
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
            kernelProfile: nil,
            decodeKernelProfile: nil,
            tokensPerSecond: nil
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

        var rng: SplitMix64? = nil
        if let seed = benchSeed() {
            rng = SplitMix64(seed: seed)
            printStderr("  RNG seed: \(seed)")
        }

        // 1. Create random weights
        let layers = LayerStorage<LayerWeights>(count: nLayers) { _ in
            let w = LayerWeights()
            randomFill(w.Wq, rng: &rng); randomFill(w.Wk, rng: &rng); randomFill(w.Wv, rng: &rng); randomFill(w.Wo, rng: &rng)
            randomFill(w.W1, rng: &rng); randomFill(w.W2, rng: &rng); randomFill(w.W3, rng: &rng)
            onesFill(w.rmsAtt); onesFill(w.rmsFfn)
            return w
        }

        // 2. Create random input
        let xCur = TensorBuffer(count: ModelConfig.dim * ModelConfig.seqLen, zeroed: false)
        randomFill(xCur, range: -0.1...0.1, rng: &rng)

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
            kernelProfile: profiler?.profile,
            decodeKernelProfile: nil,
            tokensPerSecond: nil
        )
    }

    /// Decode benchmark: autoregressive token-by-token eval with persistent KV cache surfaces.
    static func runDecode(
        warmup: Int,
        iterations: Int,
        decodeSteps: Int,
        decodeMaxSeq: Int,
        nLayers: Int = 1,
        profileKernels: Bool = false
    ) throws -> Result {
        guard decodeSteps > 0 else {
            throw ANEError.invalidArguments("decodeSteps must be > 0")
        }
        guard decodeMaxSeq > 1 else {
            throw ANEError.invalidArguments("decodeMaxSeq must be > 1")
        }
        let decodeLaneSpatial = DecodeKernelSet.resolvedLaneSpatialForCurrentProcess()
        guard decodeMaxSeq == decodeLaneSpatial else {
            throw ANEError.invalidArguments(
                "decodeMaxSeq (\(decodeMaxSeq)) must equal decode lane spatial (\(decodeLaneSpatial)) on the current ANE decode path; set --decode-max-seq \(decodeLaneSpatial)"
            )
        }
        guard decodeSteps <= decodeMaxSeq else {
            throw ANEError.invalidArguments("decodeSteps (\(decodeSteps)) must be <= decodeMaxSeq (\(decodeMaxSeq))")
        }

        printStderr("\n=== ANE Decode Benchmark (KV Cache) ===")
        printStderr("Setting up \(nLayers)-layer decode path (steps=\(decodeSteps), maxSeq=\(decodeMaxSeq))...")

        var rng: SplitMix64? = nil
        if let seed = benchSeed() {
            rng = SplitMix64(seed: seed)
            printStderr("  RNG seed: \(seed)")
        }

        // 1. Random weights
        let layers = LayerStorage<LayerWeights>(count: nLayers) { _ in
            let w = LayerWeights()
            randomFill(w.Wq, rng: &rng); randomFill(w.Wk, rng: &rng); randomFill(w.Wv, rng: &rng); randomFill(w.Wo, rng: &rng)
            randomFill(w.W1, rng: &rng); randomFill(w.W2, rng: &rng); randomFill(w.W3, rng: &rng)
            onesFill(w.rmsAtt); onesFill(w.rmsFfn)
            return w
        }

        // 2. Compile decode kernels
        printStderr("Compiling \(nLayers * 2) decode ANE kernels...")
        let compileStart = ContinuousClock.now
        let kernels = try LayerStorage<DecodeKernelSet>(count: nLayers, throwingInitializer: { i in
            try DecodeKernelSet(weights: layers[i], maxSeq: decodeMaxSeq)
        })
        let compileMs = durationMs(ContinuousClock.now - compileStart)
        printStderr(String(format: "  Compilation: %.1f ms (budget remaining: %d)", compileMs, CompileBudget.remaining))

        // 3. Pre-resolve surface handles
        var handles: [DecodeSurfaceHandles] = []
        handles.reserveCapacity(nLayers)
        for i in 0..<nLayers {
            handles.append(try DecodeSurfaceHandles(kernels: kernels[i]))
        }

        // 4. Pre-generate token embeddings for one sequence
        var tokenInputs = [Float](repeating: 0, count: decodeSteps * ModelConfig.dim)
        for i in 0..<tokenInputs.count {
            if rng != nil {
                tokenInputs[i] = rng!.nextFloat(in: -0.1...0.1)
            } else {
                tokenInputs[i] = Float.random(in: -0.1...0.1)
            }
        }

        let xCur = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        let measuredTokens = iterations * decodeSteps
        let profiler = profileKernels
            ? DecodeKernelProfiler(layerCount: nLayers, reservedSamplesPerLayer: measuredTokens)
            : nil

        // 5. Warmup sequences
        printStderr("Warmup: \(warmup) sequences × \(decodeSteps) steps...")
        for _ in 0..<warmup {
            ForwardPass.initializeDecodeCachesAndMask(surfaceHandles: handles)
            var decodeState = try DecodeState(maxSeq: decodeMaxSeq)
            for step in 0..<decodeSteps {
                loadDecodeToken(step: step, tokenInputs: tokenInputs, into: xCur)
                var timings = StepTimingBreakdown()
                try ForwardPass.runDecodeTimed(
                    xCur: xCur,
                    kernels: kernels,
                    surfaceHandles: handles,
                    decodeState: &decodeState,
                    timings: &timings,
                    profiler: nil
                )
            }
        }

        // 6. Measured token latencies
        printStderr("Measuring: \(iterations) sequences × \(decodeSteps) steps...")
        var latencies: [Double] = []
        latencies.reserveCapacity(measuredTokens)
        var totalTimings = StepTimingBreakdown()

        var completedTokens = 0
        for _ in 0..<iterations {
            ForwardPass.initializeDecodeCachesAndMask(surfaceHandles: handles)
            var decodeState = try DecodeState(maxSeq: decodeMaxSeq)
            for step in 0..<decodeSteps {
                loadDecodeToken(step: step, tokenInputs: tokenInputs, into: xCur)
                var stepTimings = StepTimingBreakdown()
                let start = ContinuousClock.now
                try ForwardPass.runDecodeTimed(
                    xCur: xCur,
                    kernels: kernels,
                    surfaceHandles: handles,
                    decodeState: &decodeState,
                    timings: &stepTimings,
                    profiler: profiler
                )
                latencies.append(durationMs(ContinuousClock.now - start))
                totalTimings.tAne += stepTimings.tAne
                totalTimings.tIO += stepTimings.tIO
                totalTimings.tElem += stepTimings.tElem
                completedTokens += 1

                if completedTokens % 100 == 0 {
                    let currentMean = latencies.reduce(0, +) / Double(latencies.count)
                    printStderr(String(format: "  [ANE Decode] %d/%d tokens — mean: %.3f ms", completedTokens, measuredTokens, currentMean))
                }
            }
        }

        let result = BenchmarkResult(
            label: "ANE Decode",
            latencies: latencies,
            warmupCount: warmup * decodeSteps,
            iterationCount: measuredTokens
        )
        let n = Double(max(1, measuredTokens))
        let avgBreakdown = (
            ane: totalTimings.tAne / n,
            io: totalTimings.tIO / n,
            elem: totalTimings.tElem / n
        )
        let tokensPerSecond = result.mean > 0 ? 1000.0 / result.mean : 0
        printStderr(String(format: "  Done. Mean: %.3f ms/token, Median: %.3f ms/token, Throughput: %.1f tok/s", result.mean, result.median, tokensPerSecond))

        return Result(
            benchmarkResult: result,
            avgTimingBreakdown: avgBreakdown,
            compileTimeMs: compileMs,
            kernelProfile: nil,
            decodeKernelProfile: profiler?.profile,
            tokensPerSecond: tokensPerSecond
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
        var none: SplitMix64? = nil
        randomFill(buffer, range: range, rng: &none)
    }

    private static func randomFill(_ buffer: borrowing TensorBuffer, range: ClosedRange<Float> = -0.1...0.1, rng: inout SplitMix64?) {
        buffer.withUnsafeMutablePointer { ptr in
            for i in 0..<buffer.count {
                if rng != nil {
                    ptr[i] = rng!.nextFloat(in: range)
                } else {
                    ptr[i] = Float.random(in: range)
                }
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

    private static func loadDecodeToken(
        step: Int,
        tokenInputs: [Float],
        into buffer: borrowing TensorBuffer
    ) {
        let dim = ModelConfig.dim
        let base = step * dim
        precondition(base + dim <= tokenInputs.count)
        precondition(buffer.count == dim)
        buffer.withUnsafeMutablePointer { dst in
            for i in 0..<dim {
                dst[i] = tokenInputs[base + i]
            }
        }
    }

    private static func benchSeed() -> UInt64? {
        guard let s = ProcessInfo.processInfo.environment["ESPRESSO_BENCH_SEED"], !s.isEmpty else {
            return nil
        }
        if s.hasPrefix("0x") || s.hasPrefix("0X") {
            return UInt64(s.dropFirst(2), radix: 16)
        }
        return UInt64(s)
    }

    private struct SplitMix64 {
        private var state: UInt64

        init(seed: UInt64) {
            self.state = seed
        }

        mutating func next() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }

        mutating func nextFloat(in range: ClosedRange<Float>) -> Float {
            // 53-bit precision fraction in [0,1).
            let u = Double(next() >> 11) * (1.0 / 9007199254740992.0)  // 2^53
            let lo = Double(range.lowerBound)
            let hi = Double(range.upperBound)
            return Float(lo + (hi - lo) * u)
        }
    }
}

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

        // Compile .mlpackage -> .mlmodelc (required before loading)
        printStderr("  Compiling .mlpackage...")
        let compileStart = ContinuousClock.now
        let compiledURL = try MLModel.compileModel(at: modelURL)
        let compileTimeMs = durationMs(ContinuousClock.now - compileStart)
        printStderr(String(format: "  Compiled in %.1f ms", compileTimeMs))

        // Measure model load time
        let loadStart = ContinuousClock.now
        let configAll = MLModelConfiguration()
        configAll.computeUnits = .all
        let modelAll = try MLModel(contentsOf: compiledURL, configuration: configAll)
        let loadTimeMs = durationMs(ContinuousClock.now - loadStart)
        printStderr(String(format: "  Model loaded in %.1f ms (compute units: .all)", loadTimeMs))

        // Create input matching tensor dimensions: [1, dim, 1, seqLen]
        let inputArray = try MLMultiArray(
            shape: [1, NSNumber(value: ModelConfig.dim), 1, NSNumber(value: ModelConfig.seqLen)],
            dataType: .float16
        )
        let count = ModelConfig.dim * ModelConfig.seqLen
        for i in 0..<count {
            inputArray[i] = NSNumber(value: Float.random(in: -0.1...0.1))
        }

        // Input key "x" matches coremltools function parameter name
        let featureProvider = try MLDictionaryFeatureProvider(
            dictionary: ["x": MLFeatureValue(multiArray: inputArray)]
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
        let modelANE = try MLModel(contentsOf: compiledURL, configuration: configANE)
        let resultANE = try runner.run(label: "CoreML (.cpuAndNeuralEngine)") {
            let _ = try modelANE.prediction(from: featureProvider)
        }
        allResults.append(("CoreML (.cpuAndNeuralEngine)", resultANE))
        printStderr(String(format: "    Mean: %.3f ms, Median: %.3f ms", resultANE.mean, resultANE.median))

        // Run with .cpuAndGPU
        printStderr("  Running with .cpuAndGPU...")
        let configGPU = MLModelConfiguration()
        configGPU.computeUnits = .cpuAndGPU
        let modelGPU = try MLModel(contentsOf: compiledURL, configuration: configGPU)
        let resultGPU = try runner.run(label: "CoreML (.cpuAndGPU)") {
            let _ = try modelGPU.prediction(from: featureProvider)
        }
        allResults.append(("CoreML (.cpuAndGPU)", resultGPU))
        printStderr(String(format: "    Mean: %.3f ms, Median: %.3f ms", resultGPU.mean, resultGPU.median))

        return Result(results: allResults, modelLoadTimeMs: loadTimeMs)
    }

    static func runNaiveDecode(
        warmup: Int,
        iterations: Int,
        decodeSteps: Int,
        decodeMaxSeq: Int,
        modelPath: String
    ) throws -> Result {
        guard decodeSteps > 0 else {
            throw BenchError.invalidDecodeArguments("decodeSteps must be > 0")
        }
        guard decodeMaxSeq > 0 else {
            throw BenchError.invalidDecodeArguments("decodeMaxSeq must be > 0")
        }
        guard decodeSteps <= decodeMaxSeq else {
            throw BenchError.invalidDecodeArguments("decodeSteps (\(decodeSteps)) must be <= decodeMaxSeq (\(decodeMaxSeq))")
        }
        guard decodeMaxSeq <= ModelConfig.seqLen else {
            throw BenchError.invalidDecodeArguments("decodeMaxSeq (\(decodeMaxSeq)) must be <= model seqLen (\(ModelConfig.seqLen))")
        }

        printStderr("\n=== Core ML Naive Decode Baseline ===")

        let modelURL = URL(fileURLWithPath: modelPath)
        guard FileManager.default.fileExists(atPath: modelPath) else {
            printStderr("  ERROR: Core ML model not found at \(modelPath)")
            printStderr("  Run: python3 scripts/generate_coreml_model.py")
            throw BenchError.coreMLModelNotFound(modelPath)
        }

        printStderr("  Compiling .mlpackage...")
        let compileStart = ContinuousClock.now
        let compiledURL = try MLModel.compileModel(at: modelURL)
        let compileMs = durationMs(ContinuousClock.now - compileStart)
        printStderr(String(format: "  Compiled in %.1f ms", compileMs))

        let loadStart = ContinuousClock.now
        let configAll = MLModelConfiguration()
        configAll.computeUnits = .all
        let modelAll = try MLModel(contentsOf: compiledURL, configuration: configAll)
        let loadTimeMs = durationMs(ContinuousClock.now - loadStart)
        printStderr(String(format: "  Model loaded in %.1f ms (compute units: .all)", loadTimeMs))

        let configANE = MLModelConfiguration()
        configANE.computeUnits = .cpuAndNeuralEngine
        let modelANE = try MLModel(contentsOf: compiledURL, configuration: configANE)

        var rng: SplitMix64? = nil
        if let seed = benchSeed() {
            rng = SplitMix64(seed: seed)
        }

        var tokenInputs = [Float](repeating: 0, count: decodeSteps * ModelConfig.dim)
        for i in 0..<tokenInputs.count {
            if rng != nil {
                tokenInputs[i] = rng!.nextFloat(in: -0.1...0.1)
            } else {
                tokenInputs[i] = Float.random(in: -0.1...0.1)
            }
        }

        let resultAll = try runNaiveDecodeSingleModel(
            model: modelAll,
            label: "CoreML Decode (.all)",
            warmup: warmup,
            iterations: iterations,
            decodeSteps: decodeSteps,
            tokenInputs: tokenInputs
        )
        let resultANE = try runNaiveDecodeSingleModel(
            model: modelANE,
            label: "CoreML Decode (.cpuAndNeuralEngine)",
            warmup: warmup,
            iterations: iterations,
            decodeSteps: decodeSteps,
            tokenInputs: tokenInputs
        )

        return Result(
            results: [
                ("CoreML Decode (.all)", resultAll),
                ("CoreML Decode (.cpuAndNeuralEngine)", resultANE),
            ],
            modelLoadTimeMs: loadTimeMs
        )
    }

    private static func runNaiveDecodeSingleModel(
        model: MLModel,
        label: String,
        warmup: Int,
        iterations: Int,
        decodeSteps: Int,
        tokenInputs: [Float]
    ) throws -> BenchmarkResult {
        printStderr("  Running \(label)...")

        let inputArray = try MLMultiArray(
            shape: [1, NSNumber(value: ModelConfig.dim), 1, NSNumber(value: ModelConfig.seqLen)],
            dataType: .float16
        )
        let featureProvider = try MLDictionaryFeatureProvider(
            dictionary: ["x": MLFeatureValue(multiArray: inputArray)]
        )

        let totalElements = inputArray.count
        let ptr = inputArray.dataPointer.bindMemory(to: Float16.self, capacity: totalElements)
        let chStride = inputArray.strides[1].intValue
        let spStride = inputArray.strides[3].intValue

        let measuredTokens = iterations * decodeSteps
        var latencies: [Double] = []
        latencies.reserveCapacity(measuredTokens)
        var measuredCount = 0

        for seq in 0..<(warmup + iterations) {
            for i in 0..<totalElements {
                ptr[i] = 0
            }

            for step in 0..<decodeSteps {
                let base = step * ModelConfig.dim
                for d in 0..<ModelConfig.dim {
                    let idx = d * chStride + step * spStride
                    ptr[idx] = Float16(tokenInputs[base + d])
                }

                let start = ContinuousClock.now
                let _ = try model.prediction(from: featureProvider)
                let ms = durationMs(ContinuousClock.now - start)
                if seq >= warmup {
                    latencies.append(ms)
                    measuredCount += 1
                    if measuredCount % 100 == 0 {
                        let currentMean = latencies.reduce(0, +) / Double(latencies.count)
                        printStderr(String(format: "    [%@] %d/%d tokens — mean: %.3f ms", label, measuredCount, measuredTokens, currentMean))
                    }
                }
            }
        }

        let result = BenchmarkResult(
            label: label,
            latencies: latencies,
            warmupCount: warmup * decodeSteps,
            iterationCount: measuredTokens
        )
        printStderr(String(format: "    Mean: %.3f ms/token, Median: %.3f ms/token", result.mean, result.median))
        return result
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
            let u = Double(next() >> 11) * (1.0 / 9007199254740992.0)
            let lo = Double(range.lowerBound)
            let hi = Double(range.upperBound)
            return Float(lo + (hi - lo) * u)
        }
    }
}

enum BenchError: Error, CustomStringConvertible {
    case coreMLModelNotFound(String)
    case invalidDecodeArguments(String)

    var description: String {
        switch self {
        case .coreMLModelNotFound(let path):
            return "Core ML model not found at \(path). Run: python3 scripts/generate_coreml_model.py"
        case .invalidDecodeArguments(let message):
            return "Invalid decode benchmark arguments: \(message)"
        }
    }
}

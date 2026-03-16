import Accelerate
import CoreML
import Foundation
import ModelSupport

struct DemoDefaults {
    let repoRoot: URL?
    let workingDirectory: URL
    let stateRoot: URL
    let cacheRoot: URL
    let reportsRoot: URL
    let hfCacheRoot: URL
    let weightsDir: URL
    let tokenizerDir: URL
    let coreMLDir: URL
    let toolsVenvDir: URL
    let scriptsDir: URL?
    let legacyArtifactsRoot: URL?

    private func helperScriptURL(named name: String) -> URL? {
        guard let scriptsDir else { return nil }
        let scriptURL = scriptsDir.appendingPathComponent(name)
        return FileManager().fileExists(atPath: scriptURL.path) ? scriptURL : nil
    }

    var bootstrapScriptURL: URL? {
        helperScriptURL(named: "bootstrap_gpt2_demo.py")
    }

    var exportScriptURL: URL? {
        helperScriptURL(named: "export_gpt2_coreml.py")
    }

    var coreMLReferenceScriptURL: URL? {
        helperScriptURL(named: "run_gpt2_coreml_reference.py")
    }

    var bootstrapScriptAvailable: Bool {
        bootstrapScriptURL != nil
    }

    var exportScriptAvailable: Bool {
        exportScriptURL != nil
    }

    var referenceScriptAvailable: Bool {
        coreMLReferenceScriptURL != nil
    }

    var scriptsAvailable: Bool {
        bootstrapScriptAvailable && exportScriptAvailable
    }

    func coreMLModelURL(sequenceLength: Int, weightsDirURL: URL? = nil) -> URL {
        if let weightsDirURL {
            let parent = weightsDirURL.deletingLastPathComponent()
            let basename = weightsDirURL.lastPathComponent
            let directoryName = basename == "gpt2_124m" ? "gpt2_coreml" : "\(basename)_coreml"
            return parent
                .appendingPathComponent(directoryName, isDirectory: true)
                .appendingPathComponent("gpt2_seq\(sequenceLength).mlpackage")
        }
        return coreMLDir.appendingPathComponent("gpt2_seq\(sequenceLength).mlpackage")
    }
}

struct CoreMLComparisonResult: Decodable, Sendable {
    let generatedTokens: [Int]
    let compileTimeMs: Double
    let firstTokenLatencyMs: Double
    let tokensPerSecond: Double
    let medianTokenMs: Double
    let p95TokenMs: Double
    let tokenLatenciesMs: [Double]
    let totalTimeMs: Double
    let computeUnits: String
    let seqLen: Int

    private enum CodingKeys: String, CodingKey {
        case generatedTokens = "generated_tokens"
        case compileTimeMs = "compile_time_ms"
        case firstTokenLatencyMs = "first_token_latency_ms"
        case tokensPerSecond = "tokens_per_second"
        case medianTokenMs = "median_token_ms"
        case p95TokenMs = "p95_token_ms"
        case tokenLatenciesMs = "token_latencies_ms"
        case totalTimeMs = "total_time_ms"
        case computeUnits = "compute_units"
        case seqLen = "seq_len"
    }
}

enum CoreMLStreamEvent: Sendable {
    case compile(compileTimeMs: Double, computeUnits: String, seqLen: Int)
    case token(token: UInt16, tokenIndex: Int, elapsedMs: Double, tokenLatencyMs: Double, tokensPerSecond: Double)
    case completed(CoreMLComparisonResult)
}

private struct ProcessOutput {
    let status: Int32
    let stdout: String
    let stderr: String
}

private struct RawCoreMLStreamEvent: Decodable {
    let type: String
    let token: Int?
    let tokenIndex: Int?
    let elapsedMs: Double?
    let tokenLatencyMs: Double?
    let tokensPerSecond: Double?
    let compileTimeMs: Double?
    let computeUnits: String?
    let seqLen: Int?
    let generatedTokens: [Int]?
    let firstTokenLatencyMs: Double?
    let medianTokenMs: Double?
    let p95TokenMs: Double?
    let tokenLatenciesMs: [Double]?
    let totalTimeMs: Double?

    private enum CodingKeys: String, CodingKey {
        case type
        case token
        case tokenIndex = "token_index"
        case elapsedMs = "elapsed_ms"
        case tokenLatencyMs = "token_latency_ms"
        case tokensPerSecond = "tokens_per_second"
        case compileTimeMs = "compile_time_ms"
        case computeUnits = "compute_units"
        case seqLen = "seq_len"
        case generatedTokens = "generated_tokens"
        case firstTokenLatencyMs = "first_token_latency_ms"
        case medianTokenMs = "median_token_ms"
        case p95TokenMs = "p95_token_ms"
        case tokenLatenciesMs = "token_latencies_ms"
        case totalTimeMs = "total_time_ms"
    }
}

private struct NativeGPT2TopLevelWeights {
    let hiddenSize: Int
    let vocabSize: Int
    let finalNormGamma: [Float]
    let finalNormBeta: [Float]
    let lmHead: [Float]
}

private struct NativeGPT2CoreMLRunResult {
    let generatedTokens: [UInt16]
    let firstTokenLatencyMs: Double
    let tokensPerSecond: Double
    let tokenLatenciesMs: [Double]
    let totalTimeMs: Double
}

private struct NativeSplitMix64: RandomNumberGenerator {
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
}

private struct NativeGPT2CoreMLReferenceRunner {
    let vocabSize: Int
    let maxSequenceTokens: Int
    let compileTimeMs: Double
    let computeUnits: String

    private let hiddenSize: Int
    private let model: MLModel
    private let inputFeatureName: String
    private let outputFeatureName: String
    private let inputArray: MLMultiArray
    private let inputDataType: MLMultiArrayDataType
    private let epsilon: Float
    private let finalNormGamma: [Float]
    private let finalNormBeta: [Float]
    private let lmHead: [Float]

    private var currentTokens: [UInt16]
    private var stepHidden: [Float]
    private var stepNorm: [Float]
    private var stepLogits: [Float]

    init(
        modelPath: String,
        weightsDir: String,
        sequenceLength: Int,
        computeUnits: String,
        epsilon: Float = 1e-5
    ) throws {
        let expandedModelPath = NSString(string: modelPath).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: expandedModelPath) else {
            throw CLIError.usage("Core ML model path does not exist: \(modelPath)")
        }

        let weights = try loadNativeGPT2TopLevelWeights(weightsDir: weightsDir)
        let compileStart = monotonicNow()
        let compiledURL: URL
        do {
            compiledURL = try MLModel.compileModel(at: URL(fileURLWithPath: expandedModelPath))
        } catch {
            throw CLIError.runtime("Core ML compile failed: \(error)")
        }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = try parseCoreMLComputeUnits(computeUnits)

        let model: MLModel
        do {
            model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        } catch {
            throw CLIError.runtime("Core ML load failed: \(error)")
        }

        let inputNames = Array(model.modelDescription.inputDescriptionsByName.keys)
        let outputNames = Array(model.modelDescription.outputDescriptionsByName.keys)
        guard inputNames.count == 1, let inputFeatureName = inputNames.first else {
            throw CLIError.runtime("expected exactly one Core ML input, found \(inputNames.count)")
        }
        guard outputNames.count == 1, let outputFeatureName = outputNames.first else {
            throw CLIError.runtime("expected exactly one Core ML output, found \(outputNames.count)")
        }

        guard let inputConstraint = model.modelDescription.inputDescriptionsByName[inputFeatureName]?.multiArrayConstraint else {
            throw CLIError.runtime("Core ML input '\(inputFeatureName)' is not an MLMultiArray")
        }
        let inputShape = inputConstraint.shape.map(\.intValue)
        guard inputShape.count == 2 else {
            throw CLIError.runtime("Expected a rank-2 Core ML input, found rank \(inputShape.count)")
        }
        guard inputShape[0] == 1 else {
            throw CLIError.runtime("Expected Core ML input shape [1, seq], found \(inputShape)")
        }
        guard sequenceLength <= inputShape[1] else {
            throw CLIError.runtime(
                "Requested Core ML sequence length \(sequenceLength) exceeds model capacity \(inputShape[1])"
            )
        }

        let inputArray: MLMultiArray
        do {
            inputArray = try MLMultiArray(
                shape: inputConstraint.shape,
                dataType: inputConstraint.dataType
            )
        } catch {
            throw CLIError.runtime("failed to allocate Core ML input array: \(error)")
        }

        self.vocabSize = weights.vocabSize
        self.maxSequenceTokens = inputShape[1]
        self.compileTimeMs = monotonicMilliseconds(since: compileStart)
        self.computeUnits = computeUnits
        self.hiddenSize = weights.hiddenSize
        self.model = model
        self.inputFeatureName = inputFeatureName
        self.outputFeatureName = outputFeatureName
        self.inputArray = inputArray
        self.inputDataType = inputConstraint.dataType
        self.epsilon = epsilon
        self.finalNormGamma = weights.finalNormGamma
        self.finalNormBeta = weights.finalNormBeta
        self.lmHead = weights.lmHead
        self.currentTokens = []
        self.stepHidden = [Float](repeating: 0, count: weights.hiddenSize)
        self.stepNorm = [Float](repeating: 0, count: weights.hiddenSize)
        self.stepLogits = [Float](repeating: 0, count: weights.vocabSize)

        try zeroInputArray()
    }

    mutating func benchmark(
        promptTokens: [UInt16],
        maxTokens: Int,
        temperature: Float,
        warmup: Int,
        iterations: Int,
        seed: Int,
        onEvent: ((CoreMLStreamEvent) -> Void)? = nil
    ) throws -> CoreMLComparisonResult {
        guard !promptTokens.isEmpty else {
            throw CLIError.runtime("Cannot compare Core ML without prompt tokens.")
        }
        guard maxTokens > 0 else {
            throw CLIError.usage("max-tokens must be > 0 for Core ML compare.")
        }
        guard promptTokens.count + maxTokens <= maxSequenceTokens else {
            throw CLIError.runtime(
                "Prompt length \(promptTokens.count) + max tokens \(maxTokens) exceeds Core ML sequence length \(maxSequenceTokens)"
            )
        }

        onEvent?(.compile(compileTimeMs: compileTimeMs, computeUnits: computeUnits, seqLen: maxSequenceTokens))

        var aggregatedLatencySamples: [Double] = []
        var lastMeasured: NativeGPT2CoreMLRunResult?

        for iteration in 0..<(warmup + iterations) {
            let emitEvents = iteration == (warmup + iterations - 1) ? onEvent : nil
            let measured = try runOnce(
                promptTokens: promptTokens,
                maxTokens: maxTokens,
                temperature: temperature,
                seed: seed,
                onEvent: emitEvents
            )
            if iteration >= warmup {
                lastMeasured = measured
                aggregatedLatencySamples.append(contentsOf: measured.tokenLatenciesMs)
            }
        }

        guard let lastMeasured else {
            throw CLIError.runtime("Core ML benchmark did not produce a measured run.")
        }

        let result = CoreMLComparisonResult(
            generatedTokens: lastMeasured.generatedTokens.map(Int.init),
            compileTimeMs: compileTimeMs,
            firstTokenLatencyMs: lastMeasured.firstTokenLatencyMs,
            tokensPerSecond: lastMeasured.tokensPerSecond,
            medianTokenMs: nativePercentile(aggregatedLatencySamples, percentile: 0.5),
            p95TokenMs: nativePercentile(aggregatedLatencySamples, percentile: 0.95),
            tokenLatenciesMs: lastMeasured.tokenLatenciesMs,
            totalTimeMs: lastMeasured.totalTimeMs,
            computeUnits: computeUnits,
            seqLen: maxSequenceTokens
        )
        onEvent?(.completed(result))
        return result
    }

    private mutating func runOnce(
        promptTokens: [UInt16],
        maxTokens: Int,
        temperature: Float,
        seed: Int,
        onEvent: ((CoreMLStreamEvent) -> Void)?
    ) throws -> NativeGPT2CoreMLRunResult {
        try reset()

        var generatedTokens: [UInt16] = []
        generatedTokens.reserveCapacity(maxTokens)
        var tokenLatenciesMs: [Double] = []
        tokenLatenciesMs.reserveCapacity(maxTokens)
        var rng = NativeSplitMix64(seed: UInt64(bitPattern: Int64(seed)))

        let generationStart = monotonicNow()
        let prefillStart = monotonicNow()
        let firstLogits = try prefill(promptTokens: promptTokens)
        let firstTokenLatencyMs = monotonicMilliseconds(since: prefillStart)
        let firstToken = sampleNativeToken(from: firstLogits, temperature: temperature, using: &rng)
        generatedTokens.append(firstToken)
        tokenLatenciesMs.append(firstTokenLatencyMs)
        onEvent?(
            .token(
                token: firstToken,
                tokenIndex: 1,
                elapsedMs: monotonicMilliseconds(since: generationStart),
                tokenLatencyMs: firstTokenLatencyMs,
                tokensPerSecond: throughput(tokensGenerated: 1, totalTimeMs: monotonicMilliseconds(since: generationStart))
            )
        )

        var currentToken = firstToken
        if maxTokens > 1 {
            for tokenIndex in 2...maxTokens {
                let tokenStart = monotonicNow()
                let logits = try decode(nextToken: currentToken)
                let tokenLatencyMs = monotonicMilliseconds(since: tokenStart)
                currentToken = sampleNativeToken(from: logits, temperature: temperature, using: &rng)
                generatedTokens.append(currentToken)
                tokenLatenciesMs.append(tokenLatencyMs)
                let elapsedMs = monotonicMilliseconds(since: generationStart)
                onEvent?(
                    .token(
                        token: currentToken,
                        tokenIndex: tokenIndex,
                        elapsedMs: elapsedMs,
                        tokenLatencyMs: tokenLatencyMs,
                        tokensPerSecond: throughput(tokensGenerated: tokenIndex, totalTimeMs: elapsedMs)
                    )
                )
            }
        }

        let totalTimeMs = monotonicMilliseconds(since: generationStart)
        return NativeGPT2CoreMLRunResult(
            generatedTokens: generatedTokens,
            firstTokenLatencyMs: firstTokenLatencyMs,
            tokensPerSecond: throughput(tokensGenerated: generatedTokens.count, totalTimeMs: totalTimeMs),
            tokenLatenciesMs: tokenLatenciesMs,
            totalTimeMs: totalTimeMs
        )
    }

    private mutating func reset() throws {
        currentTokens.removeAll(keepingCapacity: true)
        stepHidden.withUnsafeMutableBufferPointer { buffer in
            for index in buffer.indices {
                buffer[index] = 0
            }
        }
        stepNorm.withUnsafeMutableBufferPointer { buffer in
            for index in buffer.indices {
                buffer[index] = 0
            }
        }
        stepLogits.withUnsafeMutableBufferPointer { buffer in
            for index in buffer.indices {
                buffer[index] = 0
            }
        }
        try zeroInputArray()
    }

    private mutating func prefill(promptTokens: [UInt16]) throws -> [Float] {
        guard !promptTokens.isEmpty else {
            throw CLIError.runtime("Cannot compare Core ML without prompt tokens.")
        }
        currentTokens = promptTokens
        for (position, token) in promptTokens.enumerated() {
            try writeToken(token, at: position)
        }
        return try runPrediction(sequenceLength: promptTokens.count)
    }

    private mutating func decode(nextToken: UInt16) throws -> [Float] {
        guard currentTokens.count < maxSequenceTokens else {
            throw CLIError.runtime("Core ML decode overflow at sequence length \(maxSequenceTokens)")
        }
        currentTokens.append(nextToken)
        try writeToken(nextToken, at: currentTokens.count - 1)
        return try runPrediction(sequenceLength: currentTokens.count)
    }

    private mutating func runPrediction(sequenceLength: Int) throws -> [Float] {
        let provider: MLDictionaryFeatureProvider
        do {
            provider = try MLDictionaryFeatureProvider(
                dictionary: [inputFeatureName: MLFeatureValue(multiArray: inputArray)]
            )
        } catch {
            throw CLIError.runtime("Core ML feature provider creation failed: \(error)")
        }

        let prediction: MLFeatureProvider
        do {
            prediction = try model.prediction(from: provider)
        } catch {
            throw CLIError.runtime("Core ML prediction failed: \(error)")
        }

        guard let outputArray = prediction.featureValue(for: outputFeatureName)?.multiArrayValue else {
            throw CLIError.runtime("Core ML output '\(outputFeatureName)' missing MLMultiArray value")
        }

        try extractLastHidden(from: outputArray, sequenceLength: sequenceLength)
        projectCurrentLogits()
        return stepLogits
    }

    private mutating func extractLastHidden(
        from outputArray: MLMultiArray,
        sequenceLength: Int
    ) throws {
        let outputShape = outputArray.shape.map(\.intValue)
        guard outputShape.count == 3 else {
            throw CLIError.runtime("Expected rank-3 Core ML output, found rank \(outputShape.count)")
        }
        guard outputShape[0] == 1 else {
            throw CLIError.runtime("Expected Core ML output shape [1, seq, hidden], found \(outputShape)")
        }
        guard outputShape[2] == hiddenSize else {
            throw CLIError.runtime(
                "Core ML output hidden size \(outputShape[2]) does not match GPT-2 weights hidden size \(hiddenSize)"
            )
        }
        guard sequenceLength > 0, sequenceLength <= outputShape[1] else {
            throw CLIError.runtime(
                "Requested output slice \(sequenceLength) exceeds Core ML output sequence capacity \(outputShape[1])"
            )
        }

        let sequenceStride = outputArray.strides[1].intValue
        let hiddenStride = outputArray.strides[2].intValue
        let sequenceIndex = sequenceLength - 1

        switch outputArray.dataType {
        case .float16:
            let source = outputArray.dataPointer.bindMemory(to: Float16.self, capacity: outputArray.count)
            for dimIndex in 0..<hiddenSize {
                stepHidden[dimIndex] = Float(source[sequenceIndex * sequenceStride + dimIndex * hiddenStride])
            }
        case .float32:
            let source = outputArray.dataPointer.bindMemory(to: Float.self, capacity: outputArray.count)
            for dimIndex in 0..<hiddenSize {
                stepHidden[dimIndex] = source[sequenceIndex * sequenceStride + dimIndex * hiddenStride]
            }
        default:
            throw CLIError.runtime("Unsupported Core ML output dtype: \(outputArray.dataType.rawValue)")
        }
    }

    private mutating func projectCurrentLogits() {
        let mean = stepHidden.reduce(0, +) / Float(hiddenSize)
        var variance: Float = 0
        for value in stepHidden {
            let delta = value - mean
            variance += delta * delta
        }
        variance /= Float(hiddenSize)
        let inverseStd = 1.0 / sqrtf(variance + epsilon)

        for index in 0..<hiddenSize {
            stepNorm[index] = ((stepHidden[index] - mean) * inverseStd * finalNormGamma[index]) + finalNormBeta[index]
        }

        stepLogits.withUnsafeMutableBufferPointer { logitsBuffer in
            lmHead.withUnsafeBufferPointer { lmHeadBuffer in
                stepNorm.withUnsafeBufferPointer { normBuffer in
                    guard let logitsBase = logitsBuffer.baseAddress,
                          let lmHeadBase = lmHeadBuffer.baseAddress,
                          let normBase = normBuffer.baseAddress else {
                        return
                    }
                    cblas_sgemv(
                        CblasRowMajor,
                        CblasNoTrans,
                        Int32(vocabSize),
                        Int32(hiddenSize),
                        1.0,
                        lmHeadBase,
                        Int32(hiddenSize),
                        normBase,
                        1,
                        0.0,
                        logitsBase,
                        1
                    )
                }
            }
        }
    }

    private func zeroInputArray() throws {
        switch inputDataType {
        case .int32:
            let pointer = inputArray.dataPointer.bindMemory(to: Int32.self, capacity: inputArray.count)
            for index in 0..<inputArray.count {
                pointer[index] = 0
            }
        default:
            throw CLIError.runtime("Unsupported Core ML input dtype: \(inputDataType.rawValue)")
        }
    }

    private func writeToken(_ token: UInt16, at position: Int) throws {
        guard Int(token) < vocabSize else {
            throw CLIError.runtime("Token \(token) exceeds Core ML vocab size \(vocabSize)")
        }
        guard position >= 0, position < maxSequenceTokens else {
            throw CLIError.runtime("Position \(position) exceeds Core ML sequence length \(maxSequenceTokens)")
        }

        let sequenceStride = inputArray.strides[1].intValue

        switch inputDataType {
        case .int32:
            let pointer = inputArray.dataPointer.bindMemory(to: Int32.self, capacity: inputArray.count)
            pointer[position * sequenceStride] = Int32(token)
        default:
            throw CLIError.runtime("Unsupported Core ML input dtype: \(inputDataType.rawValue)")
        }
    }
}

private func loadNativeGPT2TopLevelWeights(weightsDir: String) throws -> NativeGPT2TopLevelWeights {
    let root = URL(fileURLWithPath: NSString(string: weightsDir).expandingTildeInPath, isDirectory: true).standardizedFileURL
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory), isDirectory.boolValue else {
        throw CLIError.usage("Weights directory does not exist: \(weightsDir)")
    }

    let gammaPath = try requiredHelperFile(
        root: root,
        candidates: ["final_norm_gamma.bin", "ln_f_gamma.bin", "rms_final.bin"],
        label: "final norm gamma"
    )
    let finalNormGamma = try loadBlobWeightTable(at: gammaPath.path)
    guard !finalNormGamma.isEmpty else {
        throw CLIError.runtime("Final norm gamma is empty at \(gammaPath.path)")
    }
    let hiddenSize = finalNormGamma.count

    let betaPath = try requiredHelperFile(
        root: root,
        candidates: ["final_norm_beta.bin", "ln_f_beta.bin", "rms_final_beta.bin"],
        label: "final norm beta"
    )
    let finalNormBeta = try loadBlobWeightTable(at: betaPath.path, expectedCount: hiddenSize)

    let lmHeadPath = try requiredHelperFile(
        root: root,
        candidates: ["lm_head.bin", "classifier.bin"],
        label: "lm head"
    )
    let lmHead = try loadBlobWeightTable(at: lmHeadPath.path)
    guard lmHead.count.isMultiple(of: hiddenSize) else {
        throw CLIError.runtime("lm head count \(lmHead.count) is not divisible by hidden size \(hiddenSize)")
    }
    let vocabSize = lmHead.count / hiddenSize

    return NativeGPT2TopLevelWeights(
        hiddenSize: hiddenSize,
        vocabSize: vocabSize,
        finalNormGamma: finalNormGamma,
        finalNormBeta: finalNormBeta,
        lmHead: lmHead
    )
}

private func requiredHelperFile(root: URL, candidates: [String], label: String) throws -> URL {
    let fileManager = FileManager.default
    for candidate in candidates {
        let url = root.appendingPathComponent(candidate)
        if fileManager.fileExists(atPath: url.path) {
            return url
        }
    }
    throw CLIError.runtime("Missing \(label) in \(root.path)")
}

private func loadBlobWeightTable(at path: String, expectedCount: Int? = nil) throws -> [Float] {
    let values: [Float]
    do {
        values = try BlobWeightLoader.load(from: path)
    } catch {
        throw CLIError.runtime("Failed to load weight blob \(path): \(error)")
    }
    if let expectedCount, values.count != expectedCount {
        throw CLIError.runtime("Invalid weight count for \(path): expected \(expectedCount), got \(values.count)")
    }
    return values
}

private func parseCoreMLComputeUnits(_ rawValue: String) throws -> MLComputeUnits {
    switch rawValue {
    case "all":
        return .all
    case "cpu_and_neural_engine":
        return .cpuAndNeuralEngine
    case "cpu_and_gpu":
        return .cpuAndGPU
    case "cpu_only":
        return .cpuOnly
    default:
        throw CLIError.usage("Expected --coreml-compute-units all|cpu_and_neural_engine|cpu_and_gpu|cpu_only")
    }
}

private func sampleNativeToken<R: RandomNumberGenerator>(
    from logits: [Float],
    temperature: Float,
    using rng: inout R
) -> UInt16 {
    if temperature <= 0 {
        return UInt16(logits.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0)
    }

    let maxLogit = logits.max() ?? 0
    var scaled = [Double](repeating: 0, count: logits.count)
    var total = 0.0
    for index in logits.indices {
        let value = exp(Double((logits[index] - maxLogit) / temperature))
        scaled[index] = value
        total += value
    }
    if !total.isFinite || total <= 0 {
        return UInt16(logits.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0)
    }

    var threshold = Double.random(in: 0..<total, using: &rng)
    for index in scaled.indices {
        threshold -= scaled[index]
        if threshold <= 0 {
            return UInt16(index)
        }
    }
    return UInt16(max(0, scaled.count - 1))
}

private func nativePercentile(_ values: [Double], percentile: Double) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let clamped = min(max(percentile, 0), 1)
    let index = clamped * Double(sorted.count - 1)
    let lower = Int(index.rounded(.down))
    let upper = min(lower + 1, sorted.count - 1)
    if lower == upper {
        return sorted[lower]
    }
    let fraction = index - Double(lower)
    return sorted[lower] + ((sorted[upper] - sorted[lower]) * fraction)
}

@inline(__always)
private func throughput(tokensGenerated: Int, totalTimeMs: Double) -> Double {
    guard totalTimeMs > 0 else { return 0 }
    return Double(tokensGenerated) * 1000.0 / totalTimeMs
}

@inline(__always)
private func monotonicNow() -> UInt64 {
    DispatchTime.now().uptimeNanoseconds
}

@inline(__always)
private func monotonicMilliseconds(since start: UInt64) -> Double {
    Double(DispatchTime.now().uptimeNanoseconds &- start) / 1_000_000.0
}

func detectDemoDefaults() -> DemoDefaults {
    let fileManager = FileManager()
    let repoRoot = locateRepoRoot(fileManager: fileManager)
    let stateRoot = preferredStateRoot(fileManager: fileManager)
    let cacheRoot = preferredCacheRoot(fileManager: fileManager)
    let scriptsDir = resolvedScriptsDirectory(repoRoot: repoRoot)
    let legacyArtifactsRoot = repoRoot?.appendingPathComponent(".artifacts", isDirectory: true)
    let workingDirectory = repoRoot ?? stateRoot

    return DemoDefaults(
        repoRoot: repoRoot,
        workingDirectory: workingDirectory,
        stateRoot: stateRoot,
        cacheRoot: cacheRoot,
        reportsRoot: stateRoot.appendingPathComponent("reports", isDirectory: true),
        hfCacheRoot: cacheRoot.appendingPathComponent("huggingface", isDirectory: true),
        weightsDir: stateRoot.appendingPathComponent("demo/gpt2_124m", isDirectory: true),
        tokenizerDir: stateRoot.appendingPathComponent("demo/gpt2_tokenizer", isDirectory: true),
        coreMLDir: stateRoot.appendingPathComponent("coreml/gpt2_124m", isDirectory: true),
        toolsVenvDir: stateRoot.appendingPathComponent("tools/python/gpt2-tools-venv", isDirectory: true),
        scriptsDir: scriptsDir,
        legacyArtifactsRoot: legacyArtifactsRoot
    )
}

func shouldUseDefaultGPT2Demo(_ options: Options) -> Bool {
    if options.prepareDemo || options.command == .demo {
        return true
    }
    guard options.weightsDir == nil else {
        return false
    }
    guard let modelName = options.modelName else {
        return true
    }
    return canonicalModelName(for: modelName) == ModelRegistry.gpt2_124m.name
}

func nextPowerOfTwo(_ value: Int) -> Int {
    guard value > 1 else { return 1 }
    var result = 1
    while result < value {
        result <<= 1
    }
    return result
}

func ensureGPT2DemoWeightsAndTokenizer(
    defaults: DemoDefaults,
    allowBootstrap: Bool
) throws {
    try ensureStateDirectories(defaults)
    try migrateLegacyArtifactsIfNeeded(defaults)

    if hasGPT2TokenizerAssets(in: defaults.tokenizerDir),
       FileManager().fileExists(atPath: defaults.weightsDir.appendingPathComponent("metadata.json").path)
    {
        return
    }

    guard allowBootstrap else {
        throw CLIError.usage(
            "Missing default GPT-2 demo artifacts. Re-run without --no-bootstrap, run `espresso-generate doctor`, or pass explicit --weights/--tokenizer paths."
        )
    }

    let bootstrapScript = try requireHelperScript(
        defaults,
        named: "bootstrap_gpt2_demo.py",
        purpose: "Bootstrapping default GPT-2 demo artifacts"
    )
    stderrLine("Preparing default GPT-2 demo artifacts in \(defaults.stateRoot.path)")
    let python = try ensurePythonEnvironment(
        defaults: defaults,
        requiredModules: ["numpy", "torch", "transformers"]
    )
    try runProcessStreaming(
        executable: python,
        arguments: [
            bootstrapScript.path,
            "--weights-out", defaults.weightsDir.path,
            "--tokenizer-out", defaults.tokenizerDir.path,
            "--cache-dir", defaults.hfCacheRoot.path,
        ],
        workingDirectory: defaults.workingDirectory,
        environment: pythonEnvironment(defaults: defaults)
    )
}

func ensureGPT2CoreMLModel(
    defaults: DemoDefaults,
    weightsDir: String,
    sequenceLength: Int,
    explicitModelPath: String?,
    allowBootstrap: Bool
) throws -> String {
    try ensureStateDirectories(defaults)
    try migrateLegacyArtifactsIfNeeded(defaults)

    if let explicitModelPath, !explicitModelPath.isEmpty {
        let expanded = NSString(string: explicitModelPath).expandingTildeInPath
        guard FileManager().fileExists(atPath: expanded) else {
            throw CLIError.usage("Core ML model path does not exist: \(explicitModelPath)")
        }
        return URL(fileURLWithPath: expanded).standardizedFileURL.path
    }

    let weightsURL = URL(fileURLWithPath: weightsDir, isDirectory: true).standardizedFileURL
    let modelURL = defaults.coreMLModelURL(sequenceLength: sequenceLength, weightsDirURL: weightsURL)
    if FileManager().fileExists(atPath: modelURL.path) {
        return modelURL.path
    }

    if let fallbackModel = locateCompatibleCoreMLModel(
        in: modelURL.deletingLastPathComponent(),
        minimumSequenceLength: sequenceLength
    ) {
        return fallbackModel.path
    }

    if let legacyArtifactsRoot = defaults.legacyArtifactsRoot {
        let legacyModel = legacyArtifactsRoot
            .appendingPathComponent("gpt2_coreml", isDirectory: true)
            .appendingPathComponent("gpt2_seq\(sequenceLength).mlpackage")
        if FileManager().fileExists(atPath: legacyModel.path) {
            try copyItemIfMissing(from: legacyModel, to: modelURL)
            return modelURL.path
        }
        if let fallbackModel = locateCompatibleCoreMLModel(
            in: legacyModel.deletingLastPathComponent(),
            minimumSequenceLength: sequenceLength
        ) {
            return fallbackModel.path
        }
    }

    guard allowBootstrap else {
        throw CLIError.usage(
            "Missing GPT-2 Core ML baseline for sequence length \(sequenceLength). Re-run without --no-bootstrap, run `espresso-generate doctor`, or pass --coreml-model."
        )
    }

    let exportScript = try requireHelperScript(
        defaults,
        named: "export_gpt2_coreml.py",
        purpose: "Exporting the GPT-2 Core ML baseline"
    )
    stderrLine("Exporting GPT-2 Core ML baseline for seq_len=\(sequenceLength)")
    let python = try ensurePythonEnvironment(
        defaults: defaults,
        requiredModules: ["numpy", "torch", "coremltools"]
    )
    try runProcessStreaming(
        executable: python,
        arguments: [
            exportScript.path,
            "--weights", weightsURL.path,
            "--output", modelURL.path,
            "--seq-len", String(sequenceLength),
        ],
        workingDirectory: defaults.workingDirectory,
        environment: pythonEnvironment(defaults: defaults)
    )
    return modelURL.path
}

private func locateCompatibleCoreMLModel(in directory: URL, minimumSequenceLength: Int) -> URL? {
    let fileManager = FileManager.default
    guard let children = try? fileManager.contentsOfDirectory(
        at: directory,
        includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles]
    ) else {
        return nil
    }

    let candidates = children.compactMap { url -> (sequenceLength: Int, url: URL)? in
        guard url.pathExtension == "mlpackage" else { return nil }
        let basename = url.deletingPathExtension().lastPathComponent
        guard basename.hasPrefix("gpt2_seq"),
              let sequenceLength = Int(basename.dropFirst("gpt2_seq".count))
        else {
            return nil
        }
        return (sequenceLength, url)
    }

    return candidates
        .filter { $0.sequenceLength >= minimumSequenceLength }
        .sorted {
            if $0.sequenceLength == $1.sequenceLength {
                return $0.url.path < $1.url.path
            }
            return $0.sequenceLength < $1.sequenceLength
        }
        .first?
        .url
}

func runGPT2CoreMLReference(
    defaults: DemoDefaults,
    coreMLModelPath: String,
    weightsDir: String,
    promptTokens: [UInt16],
    sequenceLength: Int,
    maxTokens: Int,
    temperature: Float,
    warmup: Int,
    iterations: Int,
    computeUnits: String,
    seed: Int,
    allowBootstrap: Bool
) throws -> CoreMLComparisonResult {
    guard !promptTokens.isEmpty else {
        throw CLIError.runtime("Cannot compare Core ML without prompt tokens.")
    }

    if let referenceScript = defaults.coreMLReferenceScriptURL {
        let python = try ensurePythonEnvironment(
            defaults: defaults,
            requiredModules: ["numpy", "coremltools"]
        )
        let promptTokenString = promptTokens.map(String.init).joined(separator: ",")
        let output = try runProcessCaptured(
            executable: python,
            arguments: [
                referenceScript.path,
                "--coreml-model", coreMLModelPath,
                "--weights", weightsDir,
                "--prompt-tokens", promptTokenString,
                "--seq-len", String(sequenceLength),
                "--max-tokens", String(maxTokens),
                "--temperature", String(temperature),
                "--warmup", String(warmup),
                "--iterations", String(iterations),
                "--seed", String(seed),
                "--compute-units", computeUnits,
            ],
            workingDirectory: defaults.workingDirectory,
            environment: pythonEnvironment(defaults: defaults),
            allowBootstrap: allowBootstrap
        )

        guard output.status == 0 else {
            throw CLIError.runtime(
                output.stderr.isEmpty
                    ? "Core ML reference runner failed with status \(output.status)"
                    : output.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            )
        }

        let decoder = JSONDecoder()
        do {
            return try decoder.decode(CoreMLComparisonResult.self, from: Data(output.stdout.utf8))
        } catch {
            throw CLIError.runtime("Failed to decode Core ML comparison output: \(error)")
        }
    }

    var runner = try NativeGPT2CoreMLReferenceRunner(
        modelPath: coreMLModelPath,
        weightsDir: weightsDir,
        sequenceLength: sequenceLength,
        computeUnits: computeUnits
    )
    return try runner.benchmark(
        promptTokens: promptTokens,
        maxTokens: maxTokens,
        temperature: temperature,
        warmup: warmup,
        iterations: iterations,
        seed: seed
    )
}

func runGPT2CoreMLReferenceStreaming(
    defaults: DemoDefaults,
    coreMLModelPath: String,
    weightsDir: String,
    promptTokens: [UInt16],
    sequenceLength: Int,
    maxTokens: Int,
    temperature: Float,
    computeUnits: String,
    seed: Int,
    allowBootstrap: Bool,
    onEvent: @escaping (CoreMLStreamEvent) -> Void
) throws -> CoreMLComparisonResult {
    guard !promptTokens.isEmpty else {
        throw CLIError.runtime("Cannot compare Core ML without prompt tokens.")
    }
    if let referenceScript = defaults.coreMLReferenceScriptURL {
        let python = try ensurePythonEnvironment(
            defaults: defaults,
            requiredModules: ["numpy", "coremltools"]
        )
        let promptTokenString = promptTokens.map(String.init).joined(separator: ",")

        var completed: CoreMLComparisonResult?
        let output = try runProcessStreamingLinesCapture(
            executable: python,
            arguments: [
                referenceScript.path,
                "--coreml-model", coreMLModelPath,
                "--weights", weightsDir,
                "--prompt-tokens", promptTokenString,
                "--seq-len", String(sequenceLength),
                "--max-tokens", String(maxTokens),
                "--temperature", String(temperature),
                "--warmup", "0",
                "--iterations", "1",
                "--seed", String(seed),
                "--compute-units", computeUnits,
                "--emit-events",
            ],
            workingDirectory: defaults.workingDirectory,
            environment: pythonEnvironment(defaults: defaults),
            allowBootstrap: allowBootstrap,
            onStdoutLine: { line in
                guard !line.isEmpty else { return }
                do {
                    let raw = try JSONDecoder().decode(RawCoreMLStreamEvent.self, from: Data(line.utf8))
                    switch raw.type {
                    case "compile":
                        if let compileTimeMs = raw.compileTimeMs,
                           let computeUnits = raw.computeUnits,
                           let seqLen = raw.seqLen
                        {
                            onEvent(.compile(compileTimeMs: compileTimeMs, computeUnits: computeUnits, seqLen: seqLen))
                        }
                    case "token":
                        guard let token = raw.token,
                              let tokenIndex = raw.tokenIndex,
                              let elapsedMs = raw.elapsedMs,
                              let tokenLatencyMs = raw.tokenLatencyMs,
                              let tokensPerSecond = raw.tokensPerSecond,
                              token >= 0,
                              token <= Int(UInt16.max)
                        else {
                            return
                        }
                        onEvent(
                            .token(
                                token: UInt16(token),
                                tokenIndex: tokenIndex,
                                elapsedMs: elapsedMs,
                                tokenLatencyMs: tokenLatencyMs,
                                tokensPerSecond: tokensPerSecond
                            )
                        )
                    case "completed":
                        let result = try JSONDecoder().decode(CoreMLComparisonResult.self, from: Data(line.utf8))
                        completed = result
                        onEvent(.completed(result))
                    default:
                        break
                    }
                } catch {
                    stderrLine("espresso-generate warning: failed to parse Core ML stream event: \(error)")
                }
            }
        )

        guard output.status == 0 else {
            throw CLIError.runtime(
                output.stderr.isEmpty
                    ? "Core ML reference runner failed with status \(output.status)"
                    : output.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            )
        }
        guard let completed else {
            throw CLIError.runtime("Core ML reference runner did not emit a completion event.")
        }
        return completed
    }

    var runner = try NativeGPT2CoreMLReferenceRunner(
        modelPath: coreMLModelPath,
        weightsDir: weightsDir,
        sequenceLength: sequenceLength,
        computeUnits: computeUnits
    )
    return try runner.benchmark(
        promptTokens: promptTokens,
        maxTokens: maxTokens,
        temperature: temperature,
        warmup: 0,
        iterations: 1,
        seed: seed,
        onEvent: onEvent
    )
}

private func locateRepoRoot(fileManager: FileManager) -> URL? {
    if let override = ProcessInfo.processInfo.environment["ESPRESSO_REPO_ROOT"], !override.isEmpty {
        let candidatePath = NSString(string: override).expandingTildeInPath
        if isRepoRoot(candidatePath, fileManager: fileManager) {
            return URL(fileURLWithPath: candidatePath, isDirectory: true).standardizedFileURL
        }
    }

    let currentDirectoryPath = NSString(string: fileManager.currentDirectoryPath).standardizingPath
    if let resolved = ascendToRepoRoot(startPath: currentDirectoryPath, fileManager: fileManager) {
        return URL(fileURLWithPath: resolved, isDirectory: true).standardizedFileURL
    }

    if let executablePath = Bundle.main.executableURL?.path,
       let resolved = ascendToRepoRoot(
           startPath: (NSString(string: executablePath).deletingLastPathComponent as NSString).standardizingPath,
           fileManager: fileManager
       )
    {
        return URL(fileURLWithPath: resolved, isDirectory: true).standardizedFileURL
    }

    let argv0 = CommandLine.arguments.first ?? ""
    if !argv0.isEmpty {
        let expanded = NSString(string: argv0).expandingTildeInPath
        let absoluteExecutablePath: String
        if expanded.hasPrefix("/") {
            absoluteExecutablePath = expanded
        } else {
            absoluteExecutablePath = (fileManager.currentDirectoryPath as NSString).appendingPathComponent(expanded)
        }
        let executableDirectory = (absoluteExecutablePath as NSString).deletingLastPathComponent
        if let resolved = ascendToRepoRoot(
            startPath: (NSString(string: executableDirectory).standardizingPath),
            fileManager: fileManager
        ) {
            return URL(fileURLWithPath: resolved, isDirectory: true).standardizedFileURL
        }
    }

    return nil
}

private func ascendToRepoRoot(startPath: String, fileManager: FileManager) -> String? {
    var currentPath = NSString(string: startPath).standardizingPath
    while true {
        if isRepoRoot(currentPath, fileManager: fileManager) {
            return currentPath
        }
        let parentPath = (currentPath as NSString).deletingLastPathComponent
        if parentPath.isEmpty || parentPath == currentPath {
            return nil
        }
        currentPath = parentPath
    }
}

private func isRepoRoot(_ candidatePath: String, fileManager: FileManager) -> Bool {
    let scriptsPath = (candidatePath as NSString).appendingPathComponent("scripts")
    let packagePath = (candidatePath as NSString).appendingPathComponent("Package.swift")
    var isDirectory: ObjCBool = false
    return fileManager.fileExists(atPath: packagePath) &&
        fileManager.fileExists(atPath: scriptsPath, isDirectory: &isDirectory) &&
        isDirectory.boolValue
}

private func preferredStateRoot(fileManager: FileManager) -> URL {
    if let override = ProcessInfo.processInfo.environment["ESPRESSO_HOME"], !override.isEmpty {
        return URL(fileURLWithPath: NSString(string: override).expandingTildeInPath, isDirectory: true).standardizedFileURL
    }
    let base = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first ??
        URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support", isDirectory: true)
    return base.appendingPathComponent("Espresso", isDirectory: true)
}

private func preferredCacheRoot(fileManager: FileManager) -> URL {
    if let override = ProcessInfo.processInfo.environment["ESPRESSO_CACHE_HOME"], !override.isEmpty {
        return URL(fileURLWithPath: NSString(string: override).expandingTildeInPath, isDirectory: true).standardizedFileURL
    }
    let base = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first ??
        URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Caches", isDirectory: true)
    return base.appendingPathComponent("Espresso", isDirectory: true)
}

private func resolvedScriptsDirectory(repoRoot: URL?) -> URL? {
    if let override = ProcessInfo.processInfo.environment["ESPRESSO_SCRIPTS_DIR"], !override.isEmpty {
        return URL(fileURLWithPath: NSString(string: override).expandingTildeInPath, isDirectory: true).standardizedFileURL
    }
    return repoRoot?.appendingPathComponent("scripts", isDirectory: true)
}

private func requireScriptsDirectory(_ defaults: DemoDefaults) throws -> URL {
    guard let scriptsDir = defaults.scriptsDir else {
        throw CLIError.runtime(
            "Espresso helper scripts are unavailable. Run from a repository checkout or set ESPRESSO_SCRIPTS_DIR to the scripts directory."
        )
    }
    var isDirectory: ObjCBool = false
    guard FileManager().fileExists(atPath: scriptsDir.path, isDirectory: &isDirectory), isDirectory.boolValue else {
        throw CLIError.runtime(
            "Espresso helper scripts directory is unavailable. Run from a repository checkout or set ESPRESSO_SCRIPTS_DIR to the scripts directory."
        )
    }
    return scriptsDir
}

private func requireHelperScript(
    _ defaults: DemoDefaults,
    named name: String,
    purpose: String
) throws -> URL {
    let scriptsDir = try requireScriptsDirectory(defaults)
    let scriptURL = scriptsDir.appendingPathComponent(name)
    guard FileManager().fileExists(atPath: scriptURL.path) else {
        throw CLIError.runtime(
            "\(purpose) requires the missing helper script \(name). Run from a repository checkout that includes helper scripts or set ESPRESSO_SCRIPTS_DIR."
        )
    }
    return scriptURL
}

private func ensureStateDirectories(_ defaults: DemoDefaults) throws {
    let fileManager = FileManager()
    for directory in [
        defaults.stateRoot,
        defaults.cacheRoot,
        defaults.reportsRoot,
        defaults.hfCacheRoot,
        defaults.weightsDir.deletingLastPathComponent(),
        defaults.tokenizerDir.deletingLastPathComponent(),
        defaults.coreMLDir,
        defaults.toolsVenvDir.deletingLastPathComponent(),
    ] {
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true, attributes: nil)
    }
}

private func migrateLegacyArtifactsIfNeeded(_ defaults: DemoDefaults) throws {
    guard let legacyArtifactsRoot = defaults.legacyArtifactsRoot else {
        return
    }
    let legacyWeights = legacyArtifactsRoot.appendingPathComponent("gpt2_124m", isDirectory: true)
    let legacyTokenizer = legacyArtifactsRoot.appendingPathComponent("gpt2_tokenizer", isDirectory: true)
    let legacyCoreML = legacyArtifactsRoot.appendingPathComponent("gpt2_coreml", isDirectory: true)

    if FileManager().fileExists(atPath: legacyWeights.path),
       !FileManager().fileExists(atPath: defaults.weightsDir.path)
    {
        try copyItemIfMissing(from: legacyWeights, to: defaults.weightsDir)
    }
    if FileManager().fileExists(atPath: legacyTokenizer.path),
       !FileManager().fileExists(atPath: defaults.tokenizerDir.path)
    {
        try copyItemIfMissing(from: legacyTokenizer, to: defaults.tokenizerDir)
    }
    if FileManager().fileExists(atPath: legacyCoreML.path),
       !FileManager().fileExists(atPath: defaults.coreMLDir.path)
    {
        try copyItemIfMissing(from: legacyCoreML, to: defaults.coreMLDir)
    }
}

private func copyItemIfMissing(from source: URL, to destination: URL) throws {
    let fileManager = FileManager()
    if fileManager.fileExists(atPath: destination.path) {
        return
    }
    try fileManager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true, attributes: nil)
    try fileManager.copyItem(at: source, to: destination)
}

private func ensurePythonEnvironment(
    defaults: DemoDefaults,
    requiredModules: [String]
) throws -> String {
    if let override = ProcessInfo.processInfo.environment["ESPRESSO_TOOLS_PYTHON"], !override.isEmpty {
        if try pythonSupportsModules(override, requiredModules: requiredModules, defaults: defaults) {
            return override
        }
        throw CLIError.runtime("ESPRESSO_TOOLS_PYTHON is set but missing required modules: \(requiredModules.joined(separator: ", "))")
    }

    let bootstrapPython = try preferredBootstrapPython(defaults: defaults)
    let versionTag = try pythonVersionTag(bootstrapPython, defaults: defaults)
    let managedVenvDir = defaults.toolsVenvDir.deletingLastPathComponent()
        .appendingPathComponent("gpt2-tools-\(versionTag)", isDirectory: true)
    let managedPython = managedVenvDir.appendingPathComponent("bin/python3").path
    if try pythonSupportsModules(managedPython, requiredModules: requiredModules, defaults: defaults) {
        return managedPython
    }

    let fileManager = FileManager()
    try fileManager.createDirectory(
        at: managedVenvDir.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    if !fileManager.fileExists(atPath: managedPython) {
        stderrLine("Creating managed Python environment at \(managedVenvDir.path)")
        try runProcessStreaming(
            executable: bootstrapPython,
            arguments: ["-m", "venv", managedVenvDir.path],
            workingDirectory: defaults.workingDirectory,
            environment: pythonEnvironment(defaults: defaults)
        )
    }

    try runProcessStreaming(
        executable: managedPython,
        arguments: ["-m", "pip", "install", "--upgrade", "pip"],
        workingDirectory: defaults.workingDirectory,
        environment: pythonEnvironment(defaults: defaults)
    )

    let packages = packagesForModules(requiredModules)
    stderrLine("Installing Python packages: \(packages.joined(separator: ", "))")
    try runProcessStreaming(
        executable: managedPython,
        arguments: ["-m", "pip", "install"] + packages,
        workingDirectory: defaults.workingDirectory,
        environment: pythonEnvironment(defaults: defaults)
    )

    guard try pythonSupportsModules(managedPython, requiredModules: requiredModules, defaults: defaults) else {
        throw CLIError.runtime("Managed Python environment is still missing required modules after installation.")
    }
    return managedPython
}

func preferredBootstrapPython(defaults: DemoDefaults) throws -> String {
    if let override = ProcessInfo.processInfo.environment["ESPRESSO_BOOTSTRAP_PYTHON"], !override.isEmpty {
        return override
    }

    for candidate in ["python3.13", "python3.12", "python3"] {
        let output = try runProcessCaptured(
            executable: candidate,
            arguments: ["--version"],
            workingDirectory: defaults.workingDirectory,
            environment: pythonEnvironment(defaults: defaults),
            allowBootstrap: true
        )
        if output.status == 0 {
            return candidate
        }
    }

    throw CLIError.runtime("Unable to find python3.13, python3.12, or python3 for GPT-2 demo setup.")
}

private func pythonVersionTag(_ executable: String, defaults: DemoDefaults) throws -> String {
    let output = try runProcessCaptured(
        executable: executable,
        arguments: ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        workingDirectory: defaults.workingDirectory,
        environment: pythonEnvironment(defaults: defaults),
        allowBootstrap: true
    )
    guard output.status == 0 else {
        throw CLIError.runtime("Unable to determine Python version for \(executable)")
    }
    let trimmed = output.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
    return trimmed.replacingOccurrences(of: ".", with: "_")
}

private func packagesForModules(_ modules: [String]) -> [String] {
    var packages: Set<String> = ["numpy"]
    for module in modules {
        switch module {
        case "numpy":
            packages.insert("numpy")
        case "torch":
            packages.insert("torch")
        case "transformers":
            packages.insert("transformers")
        case "coremltools":
            packages.insert("coremltools")
        default:
            packages.insert(module)
        }
    }
    return packages.sorted()
}

private func pythonSupportsModules(
    _ executable: String,
    requiredModules: [String],
    defaults: DemoDefaults
) throws -> Bool {
    let fileManager = FileManager()
    if executable.contains("/"), !fileManager.fileExists(atPath: executable) {
        return false
    }
    let command = "import " + requiredModules.joined(separator: ", ")
    let output = try runProcessCaptured(
        executable: executable,
        arguments: ["-c", command],
        workingDirectory: defaults.workingDirectory,
        environment: pythonEnvironment(defaults: defaults),
        allowBootstrap: true
    )
    return output.status == 0
}

private func pythonEnvironment(defaults: DemoDefaults) -> [String: String] {
    var environment = ProcessInfo.processInfo.environment
    environment["HF_HOME"] = defaults.hfCacheRoot.path
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment
}

private func runProcessStreaming(
    executable: String,
    arguments: [String],
    workingDirectory: URL,
    environment: [String: String]
) throws {
    let process = configuredProcess(
        executable: executable,
        arguments: arguments,
        workingDirectory: workingDirectory,
        environment: environment
    )
    process.standardOutput = FileHandle.standardError
    process.standardError = FileHandle.standardError
    try process.run()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
        throw CLIError.runtime("Process failed (\(process.terminationStatus)): \(executable) \(arguments.joined(separator: " "))")
    }
}

private func runProcessCaptured(
    executable: String,
    arguments: [String],
    workingDirectory: URL,
    environment: [String: String],
    allowBootstrap: Bool
) throws -> ProcessOutput {
    let process = configuredProcess(
        executable: executable,
        arguments: arguments,
        workingDirectory: workingDirectory,
        environment: environment
    )
    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe
    do {
        try process.run()
    } catch {
        if allowBootstrap {
            throw error
        }
        throw CLIError.runtime("\(error)")
    }
    process.waitUntilExit()
    let stdout = String(decoding: stdoutPipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
    let stderr = String(decoding: stderrPipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
    return ProcessOutput(status: process.terminationStatus, stdout: stdout, stderr: stderr)
}

private func runProcessStreamingLinesCapture(
    executable: String,
    arguments: [String],
    workingDirectory: URL,
    environment: [String: String],
    allowBootstrap: Bool,
    onStdoutLine: @escaping (String) -> Void
) throws -> ProcessOutput {
    let process = configuredProcess(
        executable: executable,
        arguments: arguments,
        workingDirectory: workingDirectory,
        environment: environment
    )
    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe

    let stdoutAccumulator = LockedStringBuffer()
    let stderrAccumulator = LockedStringBuffer()
    let stdoutParser = LineStreamParser { line in
        stdoutAccumulator.append(line + "\n")
        onStdoutLine(line)
    }
    let stderrParser = LineStreamParser { line in
        stderrAccumulator.append(line + "\n")
        stderrLine(line)
    }

    stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
        let data = handle.availableData
        if data.isEmpty {
            return
        }
        stdoutParser.append(data)
    }
    stderrPipe.fileHandleForReading.readabilityHandler = { handle in
        let data = handle.availableData
        if data.isEmpty {
            return
        }
        stderrParser.append(data)
    }

    do {
        try process.run()
    } catch {
        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        stderrPipe.fileHandleForReading.readabilityHandler = nil
        if allowBootstrap {
            throw error
        }
        throw CLIError.runtime("\(error)")
    }

    process.waitUntilExit()
    stdoutPipe.fileHandleForReading.readabilityHandler = nil
    stderrPipe.fileHandleForReading.readabilityHandler = nil
    stdoutParser.finish()
    stderrParser.finish()

    return ProcessOutput(
        status: process.terminationStatus,
        stdout: stdoutAccumulator.value,
        stderr: stderrAccumulator.value
    )
}

private func configuredProcess(
    executable: String,
    arguments: [String],
    workingDirectory: URL,
    environment: [String: String]
) -> Process {
    let process = Process()
    if executable.contains("/") {
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
    } else {
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [executable] + arguments
    }
    process.currentDirectoryURL = workingDirectory
    process.environment = environment
    return process
}

private final class LockedStringBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var storage = ""

    func append(_ text: String) {
        lock.lock()
        storage.append(text)
        lock.unlock()
    }

    var value: String {
        lock.lock()
        let snapshot = storage
        lock.unlock()
        return snapshot
    }
}

private final class LineStreamParser: @unchecked Sendable {
    private let lock = NSLock()
    private let onLine: (String) -> Void
    private var buffer = Data()

    init(onLine: @escaping (String) -> Void) {
        self.onLine = onLine
    }

    func append(_ data: Data) {
        lock.lock()
        buffer.append(data)
        emitLockedLines()
        lock.unlock()
    }

    func finish() {
        lock.lock()
        if !buffer.isEmpty {
            let line = String(decoding: buffer, as: UTF8.self).trimmingCharacters(in: .newlines)
            if !line.isEmpty {
                onLine(line)
            }
            buffer.removeAll(keepingCapacity: false)
        }
        lock.unlock()
    }

    private func emitLockedLines() {
        while let newlineIndex = buffer.firstIndex(of: 0x0A) {
            let lineData = buffer[..<newlineIndex]
            buffer.removeSubrange(...newlineIndex)
            let line = String(decoding: lineData, as: UTF8.self).trimmingCharacters(in: .newlines)
            onLine(line)
        }
    }
}

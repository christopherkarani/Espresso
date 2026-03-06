import Accelerate
import XCTest
import ANETypes
import CoreML
import CPUOps
@testable import Espresso

private func requireGenerationHardware(file: StaticString = #filePath, line: UInt = #line) throws {
    guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
        throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run generation hardware tests", file: file, line: line)
    }
}

private struct GenerationBenchmarkSample {
    let medianTokenMs: Double
    let medianTokensPerSecond: Double
    let acceptanceRate: Double?
    let compileTimeMs: Double
    let medianTrunkMsPerToken: Double
    let medianLogitsMsPerToken: Double
}

private func fill(_ buffer: borrowing TensorBuffer, value: Float) {
    buffer.withUnsafeMutableBufferPointer { ptr in
        for idx in ptr.indices {
            ptr[idx] = value
        }
    }
}

private func makeEchoGenerationWeights(layerCount: Int) -> GenerationWeights {
    let layers = LayerStorage<LayerWeights>(count: layerCount) { _ in
        let weights = LayerWeights()
        fill(weights.Wq, value: 0)
        fill(weights.Wk, value: 0)
        fill(weights.Wv, value: 0)
        fill(weights.Wo, value: 0)
        fill(weights.W1, value: 0)
        fill(weights.W2, value: 0)
        fill(weights.W3, value: 0)
        fill(weights.rmsAtt, value: 1)
        fill(weights.rmsFfn, value: 1)
        return weights
    }

    let rmsFinal = TensorBuffer(count: ModelConfig.dim, zeroed: false)
    fill(rmsFinal, value: 1)

    let embedding = TensorBuffer(count: ModelConfig.vocab * ModelConfig.dim, zeroed: true)
    embedding.withUnsafeMutablePointer { ptr in
        for dimIdx in 0..<ModelConfig.dim {
            ptr[dimIdx] = 1
        }
    }

    return GenerationWeights(
        layers: layers,
        rmsFinal: rmsFinal,
        embedding: embedding,
        classifier: TensorBuffer(count: 0, zeroed: true),
        sharedClassifier: true
    )
}

private func makeEchoRecurrentGenerationWeights(layerCount: Int) -> RecurrentGenerationWeights {
    let layers = LayerStorage<RWKVStyleRecurrentWeights>(count: layerCount) { _ in
        let weights = RWKVStyleRecurrentWeights()
        fill(weights.rms, value: 1)
        fill(weights.Wx, value: 0)
        fill(weights.Ws, value: 0)
        fill(weights.Wd, value: 0)
        fill(weights.Wo, value: 0)
        return weights
    }

    let rmsFinal = TensorBuffer(count: ModelConfig.dim, zeroed: false)
    fill(rmsFinal, value: 1)

    let embedding = TensorBuffer(count: ModelConfig.vocab * ModelConfig.dim, zeroed: true)
    embedding.withUnsafeMutablePointer { ptr in
        for dimIdx in 0..<ModelConfig.dim {
            ptr[dimIdx] = 1
        }
    }

    return RecurrentGenerationWeights(
        layers: layers,
        rmsFinal: rmsFinal,
        embedding: embedding,
        classifier: TensorBuffer(count: 0, zeroed: true),
        sharedClassifier: true
    )
}

private struct CoreMLGenerationBenchmarkModel: ~Copyable, AutoregressiveLanguageModel, GenerationPerformanceTrackable {
    let vocabSize: Int
    let maxSequenceTokens: Int

    private let model: MLModel
    private let inputFeatureName: String
    private let outputFeatureName: String
    private let inputArray: MLMultiArray
    private let rmsFinal: TensorBuffer
    private let embedding: TensorBuffer
    private let classifier: TensorBuffer
    private let sharedClassifier: Bool
    private let hiddenStates: TensorBuffer
    private let stepHidden: TensorBuffer
    private let stepNorm: TensorBuffer
    private let stepLogits: TensorBuffer
    private let verifyNorm: TensorBuffer
    private let verifyLogits: TensorBuffer
    private let stepRMSWorkspace: RMSNorm.Workspace
    private let verifyRMSWorkspace: RMSNorm.Workspace
    private var currentTokens: [UInt16]
    private(set) var compileTimeMs: Double
    private var trunkLatencyMs: Double
    private var logitsLatencyMs: Double

    var performanceSnapshot: GenerationPerformanceSnapshot {
        GenerationPerformanceSnapshot(
            compileTimeMs: compileTimeMs,
            trunkLatencyMs: trunkLatencyMs,
            logitsLatencyMs: logitsLatencyMs
        )
    }

    init(
        modelPath: String,
        headWeights: borrowing GenerationWeights,
        maxSequenceTokens: Int = ModelConfig.seqLen
    ) throws(GenerationError) {
        guard maxSequenceTokens > 0, maxSequenceTokens <= ModelConfig.seqLen else {
            throw .invalidArguments("maxSequenceTokens must be in 1...\(ModelConfig.seqLen)")
        }
        guard FileManager.default.fileExists(atPath: modelPath) else {
            throw .modelLoadFailed("CoreML model not found at \(modelPath)")
        }

        let compileStart = GenerationClock.now()
        let modelURL = URL(fileURLWithPath: modelPath)
        let compiledURL: URL
        do {
            compiledURL = try MLModel.compileModel(at: modelURL)
        } catch {
            throw .modelLoadFailed("CoreML compile failed: \(error)")
        }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine

        let model: MLModel
        do {
            model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        } catch {
            throw .modelLoadFailed("CoreML load failed: \(error)")
        }

        let inputNames = Array(model.modelDescription.inputDescriptionsByName.keys)
        let outputNames = Array(model.modelDescription.outputDescriptionsByName.keys)
        guard inputNames.count == 1, let inputFeatureName = inputNames.first else {
            throw .modelLoadFailed("expected exactly one CoreML input, found \(inputNames.count)")
        }
        guard outputNames.count == 1, let outputFeatureName = outputNames.first else {
            throw .modelLoadFailed("expected exactly one CoreML output, found \(outputNames.count)")
        }

        let inputArray: MLMultiArray
        do {
            inputArray = try MLMultiArray(
                shape: [1, NSNumber(value: ModelConfig.dim), 1, NSNumber(value: ModelConfig.seqLen)],
                dataType: .float16
            )
        } catch {
            throw .modelLoadFailed("failed to allocate CoreML input array: \(error)")
        }

        self.vocabSize = headWeights.vocabSize
        self.maxSequenceTokens = maxSequenceTokens
        self.model = model
        self.inputFeatureName = inputFeatureName
        self.outputFeatureName = outputFeatureName
        self.inputArray = inputArray
        self.rmsFinal = GenerationWeightCloner.cloneTensor(headWeights.rmsFinal)
        self.embedding = GenerationWeightCloner.cloneTensor(headWeights.embedding)
        self.classifier = headWeights.sharedClassifier
            ? TensorBuffer(count: 0, zeroed: true)
            : GenerationWeightCloner.cloneTensor(headWeights.classifier)
        self.sharedClassifier = headWeights.sharedClassifier
        self.hiddenStates = TensorBuffer(count: ModelConfig.dim * ModelConfig.seqLen, zeroed: true)
        self.stepHidden = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        self.stepNorm = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        self.stepLogits = TensorBuffer(count: headWeights.vocabSize, zeroed: true)
        self.verifyNorm = TensorBuffer(count: ModelConfig.dim * ModelConfig.seqLen, zeroed: true)
        self.verifyLogits = TensorBuffer(count: headWeights.vocabSize * ModelConfig.seqLen, zeroed: true)
        self.stepRMSWorkspace = RMSNorm.Workspace(seqLen: 1)
        self.verifyRMSWorkspace = RMSNorm.Workspace(seqLen: ModelConfig.seqLen)
        self.currentTokens = []
        self.compileTimeMs = GenerationClock.milliseconds(start: compileStart, end: GenerationClock.now())
        self.trunkLatencyMs = 0
        self.logitsLatencyMs = 0

        zeroInputArray()
    }

    mutating func reset() throws(GenerationError) {
        currentTokens.removeAll(keepingCapacity: true)
        hiddenStates.zero()
        stepHidden.zero()
        stepNorm.zero()
        stepLogits.zero()
        verifyNorm.zero()
        verifyLogits.zero()
        trunkLatencyMs = 0
        logitsLatencyMs = 0
        zeroInputArray()
    }

    mutating func prefill(promptTokens: [UInt16]) throws(GenerationError) -> [Float] {
        guard !promptTokens.isEmpty else {
            throw .invalidArguments("promptTokens must not be empty")
        }
        guard promptTokens.count <= maxSequenceTokens else {
            throw .invalidArguments("prompt length \(promptTokens.count) exceeds maxSequenceTokens \(maxSequenceTokens)")
        }

        currentTokens = promptTokens
        for (position, token) in promptTokens.enumerated() {
            try writeToken(token, at: position)
        }
        try runPrediction(sequenceLength: promptTokens.count)
        return try projectCurrentLogits(sequenceIndex: promptTokens.count - 1)
    }

    mutating func decode(nextToken: UInt16) throws(GenerationError) -> [Float] {
        guard currentTokens.count < maxSequenceTokens else {
            throw .invalidArguments("decode overflow at maxSequenceTokens \(maxSequenceTokens)")
        }

        currentTokens.append(nextToken)
        try writeToken(nextToken, at: currentTokens.count - 1)
        try runPrediction(sequenceLength: currentTokens.count)
        return try projectCurrentLogits(sequenceIndex: currentTokens.count - 1)
    }

    mutating func verify(
        sequenceTokens: [UInt16],
        startIndex: Int
    ) throws(GenerationError) -> [[Float]] {
        guard !sequenceTokens.isEmpty else {
            throw .invalidArguments("sequenceTokens must not be empty")
        }
        guard sequenceTokens.count <= maxSequenceTokens else {
            throw .invalidArguments("sequence length \(sequenceTokens.count) exceeds maxSequenceTokens \(maxSequenceTokens)")
        }
        guard startIndex >= 0, startIndex < sequenceTokens.count else {
            throw .invalidArguments("startIndex \(startIndex) must be within sequence length \(sequenceTokens.count)")
        }

        let savedTokens = currentTokens
        defer {
            currentTokens = savedTokens
        }

        zeroInputArray()
        for (position, token) in sequenceTokens.enumerated() {
            try writeToken(token, at: position)
        }
        try runPrediction(sequenceLength: sequenceTokens.count)
        try projectSequenceLogits()

        var outputs: [[Float]] = []
        outputs.reserveCapacity(sequenceTokens.count - startIndex)
        verifyLogits.withUnsafeBufferPointer { logitsPtr in
            for seqPos in startIndex..<sequenceTokens.count {
                var column = [Float](repeating: 0, count: vocabSize)
                for vocabIdx in 0..<vocabSize {
                    column[vocabIdx] = logitsPtr[vocabIdx * ModelConfig.seqLen + seqPos]
                }
                outputs.append(column)
            }
        }

        if savedTokens.isEmpty {
            hiddenStates.zero()
            zeroInputArray()
        } else {
            zeroInputArray()
            for (position, token) in savedTokens.enumerated() {
                try writeToken(token, at: position)
            }
            try runPrediction(sequenceLength: savedTokens.count)
        }

        return outputs
    }

    @inline(__always)
    private func classifierPointer<R>(_ body: (UnsafePointer<Float>) throws -> R) rethrows -> R {
        if sharedClassifier {
            return try embedding.withUnsafePointer(body)
        }
        return try classifier.withUnsafePointer(body)
    }

    private func zeroInputArray() {
        let ptr = inputArray.dataPointer.bindMemory(to: Float16.self, capacity: inputArray.count)
        for idx in 0..<inputArray.count {
            ptr[idx] = 0
        }
    }

    private func writeToken(_ token: UInt16, at position: Int) throws(GenerationError) {
        guard Int(token) < vocabSize else {
            throw .invalidArguments("token \(token) exceeds vocab size \(vocabSize)")
        }
        guard position >= 0, position < maxSequenceTokens else {
            throw .invalidArguments("position \(position) exceeds maxSequenceTokens \(maxSequenceTokens)")
        }

        let ptr = inputArray.dataPointer.bindMemory(to: Float16.self, capacity: inputArray.count)
        let channelStride = inputArray.strides[1].intValue
        let spatialStride = inputArray.strides[3].intValue

        embedding.withUnsafePointer { embeddingPtr in
            let base = Int(token) * ModelConfig.dim
            for dimIdx in 0..<ModelConfig.dim {
                ptr[dimIdx * channelStride + position * spatialStride] = Float16(embeddingPtr[base + dimIdx])
            }
        }
    }

    private mutating func runPrediction(sequenceLength: Int) throws(GenerationError) {
        let trunkStart = GenerationClock.now()
        let provider: MLDictionaryFeatureProvider
        do {
            provider = try MLDictionaryFeatureProvider(
                dictionary: [inputFeatureName: MLFeatureValue(multiArray: inputArray)]
            )
        } catch {
            throw .runtimeFailure("CoreML feature provider creation failed: \(error)")
        }

        let prediction: MLFeatureProvider
        do {
            prediction = try model.prediction(from: provider)
        } catch {
            throw .runtimeFailure("CoreML prediction failed: \(error)")
        }

        guard let outputArray = prediction.featureValue(for: outputFeatureName)?.multiArrayValue else {
            throw .runtimeFailure("CoreML output '\(outputFeatureName)' missing MLMultiArray value")
        }

        do {
            try copyHiddenStates(from: outputArray, sequenceLength: sequenceLength)
        } catch {
            throw .runtimeFailure("CoreML output copy failed: \(error)")
        }

        trunkLatencyMs += GenerationClock.milliseconds(start: trunkStart, end: GenerationClock.now())
    }

    private func copyHiddenStates(
        from outputArray: MLMultiArray,
        sequenceLength: Int
    ) throws(GenerationError) {
        guard sequenceLength > 0, sequenceLength <= maxSequenceTokens else {
            throw .invalidArguments("sequenceLength \(sequenceLength) exceeds maxSequenceTokens \(maxSequenceTokens)")
        }

        let channelStride = outputArray.strides[1].intValue
        let spatialStride = outputArray.strides[3].intValue

        hiddenStates.withUnsafeMutablePointer { dst in
            switch outputArray.dataType {
            case .float16:
                let src = outputArray.dataPointer.bindMemory(to: Float16.self, capacity: outputArray.count)
                for dimIdx in 0..<ModelConfig.dim {
                    let dstBase = dimIdx * ModelConfig.seqLen
                    let srcBase = dimIdx * channelStride
                    for seqIdx in 0..<sequenceLength {
                        dst[dstBase + seqIdx] = Float(src[srcBase + seqIdx * spatialStride])
                    }
                }
            case .float32:
                let src = outputArray.dataPointer.bindMemory(to: Float.self, capacity: outputArray.count)
                for dimIdx in 0..<ModelConfig.dim {
                    let dstBase = dimIdx * ModelConfig.seqLen
                    let srcBase = dimIdx * channelStride
                    for seqIdx in 0..<sequenceLength {
                        dst[dstBase + seqIdx] = src[srcBase + seqIdx * spatialStride]
                    }
                }
            default:
                break
            }
        }

        guard outputArray.dataType == .float16 || outputArray.dataType == .float32 else {
            throw .runtimeFailure("unsupported CoreML output dtype \(outputArray.dataType.rawValue)")
        }
    }

    private mutating func projectCurrentLogits(sequenceIndex: Int) throws(GenerationError) -> [Float] {
        let logitsStart = GenerationClock.now()
        stepHidden.withUnsafeMutablePointer { hiddenPtr in
            hiddenStates.withUnsafePointer { src in
                for dimIdx in 0..<ModelConfig.dim {
                    hiddenPtr[dimIdx] = src[dimIdx * ModelConfig.seqLen + sequenceIndex]
                }
            }
        }

        stepHidden.withUnsafePointer { hiddenPtr in
            stepNorm.withUnsafeMutablePointer { normPtr in
                rmsFinal.withUnsafePointer { rmsPtr in
                    RMSNorm.forward(
                        output: normPtr,
                        input: hiddenPtr,
                        weights: rmsPtr,
                        dim: ModelConfig.dim,
                        seqLen: 1,
                        workspace: stepRMSWorkspace
                    )
                }
            }
        }

        stepLogits.zero()
        stepLogits.withUnsafeMutablePointer { logitsPtr in
            classifierPointer { clsPtr in
                stepNorm.withUnsafePointer { normPtr in
                    BLAS.sgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        m: Int32(vocabSize),
                        n: 1,
                        k: Int32(ModelConfig.dim),
                        alpha: 1.0,
                        a: clsPtr,
                        lda: Int32(ModelConfig.dim),
                        b: normPtr,
                        ldb: 1,
                        beta: 0.0,
                        c: logitsPtr,
                        ldc: 1
                    )
                }
            }
        }

        logitsLatencyMs += GenerationClock.milliseconds(start: logitsStart, end: GenerationClock.now())
        return stepLogits.withUnsafeBufferPointer { Array($0) }
    }

    private mutating func projectSequenceLogits() throws(GenerationError) {
        let logitsStart = GenerationClock.now()
        verifyNorm.zero()
        verifyLogits.zero()

        hiddenStates.withUnsafePointer { inPtr in
            verifyNorm.withUnsafeMutablePointer { outPtr in
                rmsFinal.withUnsafePointer { rmsPtr in
                    RMSNorm.forward(
                        output: outPtr,
                        input: inPtr,
                        weights: rmsPtr,
                        dim: ModelConfig.dim,
                        seqLen: ModelConfig.seqLen,
                        workspace: verifyRMSWorkspace
                    )
                }
            }
        }

        verifyLogits.withUnsafeMutablePointer { logitsPtr in
            classifierPointer { clsPtr in
                verifyNorm.withUnsafePointer { normPtr in
                    BLAS.sgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        m: Int32(vocabSize),
                        n: Int32(ModelConfig.seqLen),
                        k: Int32(ModelConfig.dim),
                        alpha: 1.0,
                        a: clsPtr,
                        lda: Int32(ModelConfig.dim),
                        b: normPtr,
                        ldb: Int32(ModelConfig.seqLen),
                        beta: 0.0,
                        c: logitsPtr,
                        ldc: Int32(ModelConfig.seqLen)
                    )
                }
            }
        }

        logitsLatencyMs += GenerationClock.milliseconds(start: logitsStart, end: GenerationClock.now())
    }
}

private func median(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let mid = sorted.count / 2
    if sorted.count.isMultiple(of: 2) {
        return (sorted[mid - 1] + sorted[mid]) * 0.5
    }
    return sorted[mid]
}

final class GenerationHarnessHardwareTests: XCTestCase {
    func test_ane_direct_generation_model_generates_echo_tokens_on_hardware() throws {
        try requireGenerationHardware()

        let weights = makeEchoGenerationWeights(layerCount: 2)
        let model = try ANEDirectGenerationModel(weights: weights, layerCount: 2, decodeMaxSeq: 32)
        var harness = AutoregressiveGenerationHarness(model: model, strategy: .argmax)

        let trace = try harness.generate(promptTokens: [0], maxNewTokens: 4)

        XCTAssertEqual(trace.generatedTokens, [0, 0, 0, 0])
        XCTAssertGreaterThan(trace.tokensPerSecond, 0)
        XCTAssertGreaterThan(trace.totalLatencyMs, 0)
    }

    func test_speculative_upper_bound_reports_metrics_on_hardware() throws {
        try requireGenerationHardware()

        let prompt: [UInt16] = [0]
        let warmup = 3
        let iterations = 20
        let maxNewTokens = 8

        let direct = try benchmarkDirectEchoGeneration(
            layerCount: 6,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations
        )
        let speculativeK2 = try benchmarkSpeculativeEchoGeneration(
            fullLayers: 6,
            draftLayers: 2,
            candidateCount: 2,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations
        )
        let speculativeK4 = try benchmarkSpeculativeEchoGeneration(
            fullLayers: 6,
            draftLayers: 2,
            candidateCount: 4,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations
        )

        print(
            """
            direct echo median=\(direct.medianTokenMs) ms/token tps=\(direct.medianTokensPerSecond)
            speculative echo k=2 median=\(speculativeK2.medianTokenMs) ms/token tps=\(speculativeK2.medianTokensPerSecond) acceptance=\(speculativeK2.acceptanceRate ?? -1)
            speculative echo k=4 median=\(speculativeK4.medianTokenMs) ms/token tps=\(speculativeK4.medianTokensPerSecond) acceptance=\(speculativeK4.acceptanceRate ?? -1)
            """
        )

        XCTAssertGreaterThan(direct.medianTokenMs, 0)
        XCTAssertEqual(speculativeK2.acceptanceRate ?? -1, 1.0, accuracy: 1e-6)
        XCTAssertEqual(speculativeK4.acceptanceRate ?? -1, 1.0, accuracy: 1e-6)
    }

    func test_recurrent_generation_reports_compile_and_runtime_breakdown_on_hardware() throws {
        try requireGenerationHardware()

        let prompt: [UInt16] = [0]
        let warmup = 3
        let iterations = 20
        let maxNewTokens = 8

        let direct = try benchmarkDirectEchoGeneration(
            layerCount: 6,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations
        )
        let recurrent2 = try benchmarkRecurrentEchoGeneration(
            layerCount: 2,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations
        )

        print(
            """
            recurrent generation direct median=\(direct.medianTokenMs) ms/token tps=\(direct.medianTokensPerSecond) compile=\(direct.compileTimeMs) trunk=\(direct.medianTrunkMsPerToken) logits=\(direct.medianLogitsMsPerToken)
            recurrent generation 2-layer median=\(recurrent2.medianTokenMs) ms/token tps=\(recurrent2.medianTokensPerSecond) compile=\(recurrent2.compileTimeMs) trunk=\(recurrent2.medianTrunkMsPerToken) logits=\(recurrent2.medianLogitsMsPerToken)
            """
        )

        XCTAssertGreaterThan(direct.compileTimeMs, 0)
        XCTAssertGreaterThan(direct.medianTrunkMsPerToken, 0)
        XCTAssertGreaterThan(direct.medianLogitsMsPerToken, 0)
        XCTAssertGreaterThan(recurrent2.compileTimeMs, 0)
        XCTAssertGreaterThan(recurrent2.medianTrunkMsPerToken, 0)
        XCTAssertGreaterThan(recurrent2.medianLogitsMsPerToken, 0)
    }

    func test_recurrent_generation_6layer_and_coreml_generation_baseline_if_gate_passes() throws {
        try requireGenerationHardware()

        let prompt: [UInt16] = [0]
        let warmup = 3
        let iterations = 20
        let maxNewTokens = 8

        let direct = try benchmarkDirectEchoGeneration(
            layerCount: 6,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations
        )
        let recurrent2 = try benchmarkRecurrentEchoGeneration(
            layerCount: 2,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations
        )

        guard recurrent2.medianTokenMs * 2.0 <= direct.medianTokenMs else {
            throw XCTSkip(
                "2-layer recurrent generation did not clear the >=2x speed gate; skipping 6-layer/coreml follow-up"
            )
        }

        let recurrent6 = try benchmarkRecurrentEchoGeneration(
            layerCount: 6,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations
        )
        let coreML = try benchmarkCoreMLGeneration(
            modelPath: "benchmarks/models/transformer_6layer.mlpackage",
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations
        )

        print(
            """
            recurrent generation 6-layer median=\(recurrent6.medianTokenMs) ms/token tps=\(recurrent6.medianTokensPerSecond) compile=\(recurrent6.compileTimeMs) trunk=\(recurrent6.medianTrunkMsPerToken) logits=\(recurrent6.medianLogitsMsPerToken)
            coreml generation median=\(coreML.medianTokenMs) ms/token tps=\(coreML.medianTokensPerSecond) compile=\(coreML.compileTimeMs) trunk=\(coreML.medianTrunkMsPerToken) logits=\(coreML.medianLogitsMsPerToken)
            """
        )

        XCTAssertGreaterThan(recurrent6.medianTokenMs, 0)
        XCTAssertGreaterThan(coreML.medianTokenMs, 0)
        XCTAssertGreaterThan(coreML.compileTimeMs, 0)
        XCTAssertGreaterThan(coreML.medianTrunkMsPerToken, 0)
        XCTAssertGreaterThan(coreML.medianLogitsMsPerToken, 0)
    }

    func test_recurrent_generation_ane_classifier_head_reports_comparison_on_hardware() throws {
        try requireGenerationHardware()

        let prompt: [UInt16] = [0]
        let warmup = 3
        let iterations = 20
        let maxNewTokens = 8

        let cpuHead = try benchmarkRecurrentEchoGeneration(
            layerCount: 6,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations,
            outputHeadBackend: .cpu
        )
        let aneHead = try benchmarkRecurrentEchoGeneration(
            layerCount: 6,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations,
            outputHeadBackend: .aneClassifier
        )

        print(
            """
            recurrent generation cpu-head median=\(cpuHead.medianTokenMs) ms/token tps=\(cpuHead.medianTokensPerSecond) compile=\(cpuHead.compileTimeMs) trunk=\(cpuHead.medianTrunkMsPerToken) logits=\(cpuHead.medianLogitsMsPerToken)
            recurrent generation ane-head median=\(aneHead.medianTokenMs) ms/token tps=\(aneHead.medianTokensPerSecond) compile=\(aneHead.compileTimeMs) trunk=\(aneHead.medianTrunkMsPerToken) logits=\(aneHead.medianLogitsMsPerToken)
            """
        )

        XCTAssertGreaterThan(cpuHead.medianTokenMs, 0)
        XCTAssertGreaterThan(aneHead.medianTokenMs, 0)
        XCTAssertGreaterThan(aneHead.compileTimeMs, 0)
        XCTAssertGreaterThan(aneHead.medianLogitsMsPerToken, 0)
    }

    private func benchmarkDirectEchoGeneration(
        layerCount: Int,
        promptTokens: [UInt16],
        maxNewTokens: Int,
        warmup: Int,
        iterations: Int,
        outputHeadBackend: GenerationOutputHeadBackend = .cpu
    ) throws -> GenerationBenchmarkSample {
        let weights = makeEchoGenerationWeights(layerCount: layerCount)
        let model = try ANEDirectGenerationModel(
            weights: weights,
            layerCount: layerCount,
            decodeMaxSeq: 32,
            outputHeadBackend: outputHeadBackend
        )
        var harness = AutoregressiveGenerationHarness(model: model, strategy: .argmax)
        return try benchmarkAutoregressiveHarness(
            harness: &harness,
            promptTokens: promptTokens,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations
        )
    }

    private func benchmarkSpeculativeEchoGeneration(
        fullLayers: Int,
        draftLayers: Int,
        candidateCount: Int,
        promptTokens: [UInt16],
        maxNewTokens: Int,
        warmup: Int,
        iterations: Int
    ) throws -> GenerationBenchmarkSample {
        let weights = makeEchoGenerationWeights(layerCount: fullLayers)
        let draftModel = try ANEDirectGenerationModel(weights: weights, layerCount: draftLayers, decodeMaxSeq: 32)
        let fullModel = try ANEDirectGenerationModel(weights: weights, layerCount: fullLayers, decodeMaxSeq: 32)
        var harness = SpeculativeGenerationHarness(
            draftModel: draftModel,
            fullModel: fullModel,
            strategy: .argmax,
            candidateCount: candidateCount
        )

        var tokenLatencies: [Double] = []
        var throughput: [Double] = []
        var acceptanceRates: [Double] = []
        tokenLatencies.reserveCapacity(iterations)
        throughput.reserveCapacity(iterations)
        acceptanceRates.reserveCapacity(iterations)

        for iter in 0..<(warmup + iterations) {
            let trace = try harness.generate(promptTokens: promptTokens, maxNewTokens: maxNewTokens)
            if iter >= warmup {
                tokenLatencies.append(trace.totalLatencyMs / Double(maxNewTokens))
                throughput.append(trace.effectiveTokensPerSecond)
                acceptanceRates.append(trace.acceptanceRate)
            }
        }

        return GenerationBenchmarkSample(
            medianTokenMs: median(tokenLatencies),
            medianTokensPerSecond: median(throughput),
            acceptanceRate: median(acceptanceRates),
            compileTimeMs: 0,
            medianTrunkMsPerToken: 0,
            medianLogitsMsPerToken: 0
        )
    }

    private func benchmarkRecurrentEchoGeneration(
        layerCount: Int,
        promptTokens: [UInt16],
        maxNewTokens: Int,
        warmup: Int,
        iterations: Int,
        outputHeadBackend: GenerationOutputHeadBackend = .cpu
    ) throws -> GenerationBenchmarkSample {
        let weights = makeEchoRecurrentGenerationWeights(layerCount: layerCount)
        let model = try ANERecurrentGenerationModel(
            weights: weights,
            layerCount: layerCount,
            maxSequenceTokens: 32,
            outputHeadBackend: outputHeadBackend
        )
        var harness = AutoregressiveGenerationHarness(model: model, strategy: .argmax)
        return try benchmarkAutoregressiveHarness(
            harness: &harness,
            promptTokens: promptTokens,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations
        )
    }

    private func benchmarkCoreMLGeneration(
        modelPath: String,
        promptTokens: [UInt16],
        maxNewTokens: Int,
        warmup: Int,
        iterations: Int
    ) throws -> GenerationBenchmarkSample {
        let headWeights = makeEchoGenerationWeights(layerCount: 1)
        let model = try CoreMLGenerationBenchmarkModel(
            modelPath: modelPath,
            headWeights: headWeights,
            maxSequenceTokens: 32
        )
        var harness = AutoregressiveGenerationHarness(model: model, strategy: .argmax)
        return try benchmarkAutoregressiveHarness(
            harness: &harness,
            promptTokens: promptTokens,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations
        )
    }

    private func benchmarkAutoregressiveHarness<Model>(
        harness: inout AutoregressiveGenerationHarness<Model>,
        promptTokens: [UInt16],
        maxNewTokens: Int,
        warmup: Int,
        iterations: Int
    ) throws -> GenerationBenchmarkSample
    where Model: AutoregressiveLanguageModel & GenerationPerformanceTrackable, Model: ~Copyable {
        var tokenLatencies: [Double] = []
        var throughput: [Double] = []
        var trunkLatencies: [Double] = []
        var logitsLatencies: [Double] = []
        tokenLatencies.reserveCapacity(iterations)
        throughput.reserveCapacity(iterations)
        trunkLatencies.reserveCapacity(iterations)
        logitsLatencies.reserveCapacity(iterations)

        let compileTimeMs = harness.model.performanceSnapshot.compileTimeMs

        for iter in 0..<(warmup + iterations) {
            let trace = try harness.generate(promptTokens: promptTokens, maxNewTokens: maxNewTokens)
            if iter >= warmup {
                let snapshot = harness.model.performanceSnapshot
                tokenLatencies.append(trace.totalLatencyMs / Double(maxNewTokens))
                throughput.append(trace.tokensPerSecond)
                trunkLatencies.append(snapshot.trunkLatencyMs / Double(maxNewTokens))
                logitsLatencies.append(snapshot.logitsLatencyMs / Double(maxNewTokens))
            }
        }

        return GenerationBenchmarkSample(
            medianTokenMs: median(tokenLatencies),
            medianTokensPerSecond: median(throughput),
            acceptanceRate: nil,
            compileTimeMs: compileTimeMs,
            medianTrunkMsPerToken: median(trunkLatencies),
            medianLogitsMsPerToken: median(logitsLatencies)
        )
    }
}

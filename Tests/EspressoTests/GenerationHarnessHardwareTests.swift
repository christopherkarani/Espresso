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

private struct ExactTwoTokenBenchmarkSample {
    let medianTokenMs: Double
    let medianTokensPerSecond: Double
    let compileTimeMs: Double
    let medianCommittedExactTokensPerPass: Double
    let medianAcceptedFutureTokensPerPass: Double
    let medianProposerMsPerPass: Double
    let medianVerifierTrunkMsPerPass: Double
    let medianVerifierLogitsMsPerPass: Double
    let medianStateAdvanceMsPerPass: Double
}

private struct CompileInitBenchmarkSample {
    let wallInitMs: Double
    let reportedCompileTimeMs: Double
}

private struct SurfacePeak: Equatable {
    let channel: Int
    let spatial: Int
    let value: Float
}

private func peakValue(of values: [Float], spatial: Int) -> SurfacePeak {
    precondition(spatial > 0)
    precondition(values.count.isMultiple(of: spatial))
    let winner = values.enumerated().max { lhs, rhs in lhs.element < rhs.element }!
    return SurfacePeak(
        channel: winner.offset / spatial,
        spatial: winner.offset % spatial,
        value: winner.element
    )
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

    func test_recurrent_exact_two_token_upper_bound_reports_pass_breakdown_on_hardware() throws {
        try requireGenerationHardware()

        let prompt: [UInt16] = [0]
        let warmup = 3
        let iterations = 20
        let maxNewTokens = 8

        let control = try benchmarkRecurrentEchoGeneration(
            layerCount: 6,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations,
            outputHeadBackend: .aneRMSNormClassifier,
            useDirectTokenSelection: true,
            trunkBackend: .fusedThreeLayerTriplets
        )
        let upperBound = try benchmarkRecurrentExactTwoTokenUpperBoundGeneration(
            layerCount: 6,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations,
            outputHeadBackend: .aneRMSNormClassifier,
            trunkBackend: .fusedThreeLayerTriplets
        )

        print(
            """
            recurrent exact control median=\(control.medianTokenMs) ms/token tps=\(control.medianTokensPerSecond) compile=\(control.compileTimeMs) trunk=\(control.medianTrunkMsPerToken) logits=\(control.medianLogitsMsPerToken)
            recurrent exact two-token upper-bound median=\(upperBound.medianTokenMs) ms/token tps=\(upperBound.medianTokensPerSecond) compile=\(upperBound.compileTimeMs) committed_exact_tokens_per_pass=\(upperBound.medianCommittedExactTokensPerPass) accepted_future_tokens_per_pass=\(upperBound.medianAcceptedFutureTokensPerPass) proposer=\(upperBound.medianProposerMsPerPass) verifier_trunk=\(upperBound.medianVerifierTrunkMsPerPass) verifier_logits=\(upperBound.medianVerifierLogitsMsPerPass) state_advance=\(upperBound.medianStateAdvanceMsPerPass)
            """
        )

        XCTAssertGreaterThan(control.medianTokenMs, 0)
        XCTAssertGreaterThan(upperBound.medianTokenMs, 0)
        XCTAssertGreaterThan(upperBound.compileTimeMs, 0)
        XCTAssertGreaterThan(upperBound.medianCommittedExactTokensPerPass, 1.0)
        XCTAssertGreaterThanOrEqual(upperBound.medianAcceptedFutureTokensPerPass, 0.0)
    }

    func test_recurrent_single_layer_control_reports_compile_init_only_on_hardware() throws {
        try requireGenerationHardware()

        let control = try measureRecurrentSingleLayerControlCompileInitOnly(
            layerCount: 1,
            outputHeadBackend: .aneRMSNormClassifier
        )

        print(
            """
            recurrent single-layer exact control compile-init-only init_wall=\(control.wallInitMs) ms compile=\(control.reportedCompileTimeMs) ms
            """
        )

        XCTAssertGreaterThan(control.wallInitMs, 0)
        XCTAssertGreaterThan(control.reportedCompileTimeMs, 0)
    }

    func test_recurrent_exact_two_token_branch_state_promotion_reports_compile_init_only_on_hardware() throws {
        try requireGenerationHardware()

        let promoted = try measureRecurrentExactTwoTokenBranchStatePromotionCompileInitOnly(
            layerCount: 1,
            outputHeadBackend: .aneRMSNormClassifier
        )

        print(
            """
            recurrent single-layer exact two-token branch-state-promotion compile-init-only init_wall=\(promoted.wallInitMs) ms compile=\(promoted.reportedCompileTimeMs) ms
            """
        )

        XCTAssertGreaterThan(promoted.wallInitMs, 0)
        XCTAssertGreaterThan(promoted.reportedCompileTimeMs, 0)
    }

    func test_ane_rmsnorm_classifier_head_selects_nonzero_token_on_hardware() throws {
        try requireGenerationHardware()

        let rmsFinal = TensorBuffer(count: ModelConfig.dim, zeroed: false)
        rmsFinal.withUnsafeMutablePointer { ptr in
            for idx in 0..<ModelConfig.dim {
                ptr[idx] = 1
            }
        }

        let classifier = TensorBuffer(count: ModelConfig.vocab * ModelConfig.dim, zeroed: true)
        classifier.withUnsafeMutablePointer { ptr in
            ptr[105 * ModelConfig.dim + 35] = 10
        }

        let head = try ANEGenerationRMSNormClassifierHead(
            rmsFinal: rmsFinal,
            classifierWeights: classifier,
            vocabSize: ModelConfig.vocab,
            laneSpatial: 32
        )

        let rawInput = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        rawInput.withUnsafeMutablePointer { ptr in
            ptr[35] = 1
        }

        let selected = try head.selectArgmax(rawInput: rawInput)
        XCTAssertEqual(selected, 105)
    }

    func test_recurrent_single_layer_zero_trunk_raw_surface_probe_on_hardware() throws {
        try requireGenerationHardware()
        guard ProcessInfo.processInfo.environment["ANE_NEGATIVE_PROBES"] == "1" else {
            throw XCTSkip("Set ANE_NEGATIVE_PROBES=1 to run the documented negative recurrent raw-surface probe")
        }

        let weights = makeEchoRecurrentGenerationWeights(layerCount: 1)
        var session = try RWKVStyleRecurrentSession(weights: weights.layers[0], laneSpatial: 32)
        try session.reset()

        let input = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        input.withUnsafeMutablePointer { ptr in
            ptr[35] = 1
        }
        try SurfaceIO.copyFP16(
            dst: session.handles.xIn,
            dstChannelOffset: 0,
            src: session.handles.zeroLane,
            srcChannelOffset: 0,
            channels: ModelConfig.dim,
            spatial: session.handles.laneSpatial
        )
        try input.withUnsafeBufferPointer { tokenBuf in
            try SurfaceIO.writeFP16SpatialSlice(
                to: session.handles.xIn,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: session.handles.laneSpatial,
                data: tokenBuf,
                channels: ModelConfig.dim
            )
        }

        let surfaceCount = ModelConfig.dim * session.handles.laneSpatial
        var xInValues = [Float](repeating: 0, count: surfaceCount)
        var xOutValues = [Float](repeating: 0, count: surfaceCount)
        var stateOutValues = [Float](repeating: 0, count: surfaceCount)
        xInValues.withUnsafeMutableBufferPointer { buffer in
            SurfaceIO.readFP16(
                from: session.handles.xIn,
                into: buffer,
                channelOffset: 0,
                channels: ModelConfig.dim,
                spatial: session.handles.laneSpatial
            )
        }

        try session.kernels.step.eval()

        xOutValues.withUnsafeMutableBufferPointer { buffer in
            SurfaceIO.readFP16(
                from: session.handles.xOut,
                into: buffer,
                channelOffset: 0,
                channels: ModelConfig.dim,
                spatial: session.handles.laneSpatial
            )
        }
        stateOutValues.withUnsafeMutableBufferPointer { buffer in
            SurfaceIO.readFP16(
                from: session.handles.stateOut,
                into: buffer,
                channelOffset: 0,
                channels: ModelConfig.dim,
                spatial: session.handles.laneSpatial
            )
        }

        let xInPeak = peakValue(of: xInValues, spatial: session.handles.laneSpatial)
        let xOutPeak = peakValue(of: xOutValues, spatial: session.handles.laneSpatial)
        let stateOutPeak = peakValue(of: stateOutValues, spatial: session.handles.laneSpatial)

        print(
            """
            recurrent raw probe xInPeak=\(xInPeak) xOutPeak=\(xOutPeak) stateOutPeak=\(stateOutPeak)
            """
        )

        XCTAssertEqual(xInPeak.channel, 35)
        XCTAssertEqual(xInPeak.spatial, 0)
        XCTAssertGreaterThan(xInPeak.value, 0.5)
        XCTAssertTrue(
            xOutPeak.value > 0.5 || stateOutPeak.value > 0.5,
            "expected recurrent eval to preserve nonzero signal on at least one output surface; xOutPeak=\(xOutPeak) stateOutPeak=\(stateOutPeak)"
        )
    }

    func test_identity_zero_trunk_local_bigram_recurrent_generation_matches_cpu_teacher_on_hardware() throws {
        try requireGenerationHardware()

        let recurrentWeights = try LocalBigramArtifactBuilder.buildRecurrentWeights(
            tokens: [35, 105, 110, 116, 32, 105, 110, 116, 32],
            layerCount: 1,
            vocabSize: ModelConfig.vocab
        )
        let expectedTokens: [UInt16] = [105, 110, 116, 32]

        let cpuTeacher = try CPURecurrentGenerationModel(
            weights: recurrentWeights,
            layerCount: 1
        )
        var cpuHarness = DirectTokenSelectionGenerationHarness(model: cpuTeacher, strategy: .argmax)
        let cpuTrace = try cpuHarness.generate(promptTokens: [35], maxNewTokens: 4)

        let aneModel = try ANERecurrentGenerationModel(
            weights: recurrentWeights,
            layerCount: 1,
            maxSequenceTokens: 32,
            outputHeadBackend: .aneRMSNormClassifier,
            trunkBackend: .identityZeroTrunk
        )
        var aneHarness = DirectTokenSelectionGenerationHarness(model: aneModel, strategy: .argmax)
        let aneTrace = try aneHarness.generate(promptTokens: [35], maxNewTokens: 4)

        XCTAssertEqual(cpuTrace.generatedTokens, expectedTokens)
        XCTAssertEqual(aneTrace.generatedTokens, expectedTokens)
        XCTAssertEqual(aneTrace.generatedTokens, cpuTrace.generatedTokens)
    }

    func test_identity_zero_trunk_local_bigram_exact_two_token_generation_matches_cpu_teacher_on_hardware() throws {
        try requireGenerationHardware()

        let recurrentWeights = try LocalBigramArtifactBuilder.buildRecurrentWeights(
            tokens: [35, 105, 110, 116, 32, 105, 110, 116, 32],
            layerCount: 1,
            vocabSize: ModelConfig.vocab
        )
        let futureSidecar = try LocalBigramArtifactBuilder.buildFutureSidecar(
            tokens: [35, 105, 110, 116, 32, 105, 110, 116, 32],
            layerCount: 1,
            vocabSize: ModelConfig.vocab
        )

        let cpuTeacher = try CPURecurrentGenerationModel(
            weights: recurrentWeights,
            layerCount: 1
        )
        var cpuHarness = DirectTokenSelectionGenerationHarness(model: cpuTeacher, strategy: .argmax)
        let cpuTrace = try cpuHarness.generate(promptTokens: [35], maxNewTokens: 4)

        let aneModel = try ANEExactTwoTokenBranchStatePromotionModel(
            weights: recurrentWeights,
            futureSidecar: futureSidecar,
            layerCount: 1,
            maxSequenceTokens: 32,
            outputHeadBackend: .aneRMSNormClassifier,
            trunkBackend: .identityZeroTrunk
        )
        var aneHarness = ExactTwoTokenGenerationHarness(model: aneModel, strategy: .argmax)
        let aneTrace = try aneHarness.generate(promptTokens: [35], maxNewTokens: 4)

        XCTAssertEqual(aneTrace.generatedTokens, cpuTrace.generatedTokens)
        XCTAssertEqual(aneTrace.generatedTokens, [105, 110, 116, 32])
        XCTAssertEqual(aneTrace.committedExactTokenCounts, [2, 2])
        XCTAssertEqual(aneTrace.acceptedFutureTokenCounts, [1, 1])
        XCTAssertEqual(aneTrace.committedExactTokensPerPass, 2, accuracy: 0.0001)
        XCTAssertEqual(aneTrace.acceptedFutureTokensPerPass, 1, accuracy: 0.0001)
    }

    func test_recurrent_exact_two_token_branch_state_promotion_reports_pass_breakdown_on_hardware() throws {
        try requireGenerationHardware()

        let prompt: [UInt16] = [0]
        let warmup = 3
        let iterations = 20
        let maxNewTokens = 8

        let weights = makeEchoRecurrentGenerationWeights(layerCount: 1)

        let controlInitStart = mach_absolute_time()
        let controlModel = try ANERecurrentGenerationModel(
            weights: weights,
            layerCount: 1,
            maxSequenceTokens: 32,
            outputHeadBackend: .aneRMSNormClassifier,
            trunkBackend: .singleLayer,
            trunkLaneSpatial: 32,
            outputHeadLaneSpatial: 32
        )
        let controlInitMs = machMilliseconds(mach_absolute_time() - controlInitStart)
        var controlHarness = DirectTokenSelectionGenerationHarness(model: controlModel, strategy: .argmax)

        let promotedInitStart = mach_absolute_time()
        let promotedModel = try ANEExactTwoTokenBranchStatePromotionModel(
            weights: weights,
            layerCount: 1,
            maxSequenceTokens: 32,
            outputHeadBackend: .aneRMSNormClassifier,
            trunkLaneSpatial: 32,
            outputHeadLaneSpatial: 32
        )
        let promotedInitMs = machMilliseconds(mach_absolute_time() - promotedInitStart)
        var promotedHarness = ExactTwoTokenGenerationHarness(model: promotedModel, strategy: .argmax)

        let controlParityTrace = try controlHarness.generate(promptTokens: prompt, maxNewTokens: maxNewTokens)
        let promotedParityTrace = try promotedHarness.generate(promptTokens: prompt, maxNewTokens: maxNewTokens)
        let exactParity = controlParityTrace.generatedTokens == promotedParityTrace.generatedTokens

        let control = try benchmarkDirectSelectionHarness(
            harness: &controlHarness,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations
        )
        let promoted = try benchmarkExactTwoTokenHarness(
            harness: &promotedHarness,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations
        )

        print(
            """
            recurrent single-layer exact control init_wall=\(controlInitMs) ms compile=\(control.compileTimeMs) ms committed_exact_tokens_per_pass=1.0 median=\(control.medianTokenMs) ms/token tps=\(control.medianTokensPerSecond) trunk=\(control.medianTrunkMsPerToken) logits=\(control.medianLogitsMsPerToken)
            recurrent single-layer exact two-token branch-state-promotion init_wall=\(promotedInitMs) ms compile=\(promoted.compileTimeMs) ms committed_exact_tokens_per_pass=\(promoted.medianCommittedExactTokensPerPass) accepted_future_tokens_per_pass=\(promoted.medianAcceptedFutureTokensPerPass) median=\(promoted.medianTokenMs) ms/token tps=\(promoted.medianTokensPerSecond) proposer=\(promoted.medianProposerMsPerPass) verifier_trunk=\(promoted.medianVerifierTrunkMsPerPass) verifier_logits=\(promoted.medianVerifierLogitsMsPerPass) state_advance=\(promoted.medianStateAdvanceMsPerPass)
            recurrent single-layer exact parity status=\(exactParity ? "match" : "mismatch")
            """
        )

        XCTAssertGreaterThan(controlInitMs, 0)
        XCTAssertGreaterThan(promotedInitMs, 0)
        XCTAssertGreaterThan(control.medianTokenMs, 0)
        XCTAssertGreaterThan(promoted.medianTokenMs, 0)
        XCTAssertGreaterThan(promoted.compileTimeMs, 0)
        XCTAssertGreaterThan(promoted.medianCommittedExactTokensPerPass, 1.0)
        XCTAssertGreaterThanOrEqual(promoted.medianAcceptedFutureTokensPerPass, 0.0)
        XCTAssertTrue(exactParity)
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
        let fusedHead = try benchmarkRecurrentEchoGeneration(
            layerCount: 6,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations,
            outputHeadBackend: .aneRMSNormClassifier
        )

        print(
            """
            recurrent generation cpu-head median=\(cpuHead.medianTokenMs) ms/token tps=\(cpuHead.medianTokensPerSecond) compile=\(cpuHead.compileTimeMs) trunk=\(cpuHead.medianTrunkMsPerToken) logits=\(cpuHead.medianLogitsMsPerToken)
            recurrent generation ane-head median=\(aneHead.medianTokenMs) ms/token tps=\(aneHead.medianTokensPerSecond) compile=\(aneHead.compileTimeMs) trunk=\(aneHead.medianTrunkMsPerToken) logits=\(aneHead.medianLogitsMsPerToken)
            recurrent generation fused-head median=\(fusedHead.medianTokenMs) ms/token tps=\(fusedHead.medianTokensPerSecond) compile=\(fusedHead.compileTimeMs) trunk=\(fusedHead.medianTrunkMsPerToken) logits=\(fusedHead.medianLogitsMsPerToken)
            """
        )

        XCTAssertGreaterThan(cpuHead.medianTokenMs, 0)
        XCTAssertGreaterThan(aneHead.medianTokenMs, 0)
        XCTAssertGreaterThan(aneHead.compileTimeMs, 0)
        XCTAssertGreaterThan(aneHead.medianLogitsMsPerToken, 0)
        XCTAssertGreaterThan(fusedHead.medianTokenMs, 0)
        XCTAssertGreaterThan(fusedHead.compileTimeMs, 0)
        XCTAssertGreaterThan(fusedHead.medianLogitsMsPerToken, 0)
    }

    func test_recurrent_generation_fused_head_direct_surface_argmax_reports_reduced_readback_on_hardware() throws {
        try requireGenerationHardware()

        let prompt: [UInt16] = [0]
        let warmup = 3
        let iterations = 20
        let maxNewTokens = 8

        let parityWeights = makeEchoRecurrentGenerationWeights(layerCount: 6)
        let materializedModel = try ANERecurrentGenerationModel(
            weights: parityWeights,
            layerCount: 6,
            maxSequenceTokens: 32,
            outputHeadBackend: .aneRMSNormClassifier
        )
        let directSelectModel = try ANERecurrentGenerationModel(
            weights: parityWeights,
            layerCount: 6,
            maxSequenceTokens: 32,
            outputHeadBackend: .aneRMSNormClassifier
        )
        var materializedHarness = AutoregressiveGenerationHarness(
            model: materializedModel,
            strategy: .argmax
        )
        var directSelectHarness = DirectTokenSelectionGenerationHarness(
            model: directSelectModel,
            strategy: .argmax
        )
        let materializedTrace = try materializedHarness.generate(
            promptTokens: prompt,
            maxNewTokens: maxNewTokens
        )
        let directSelectTrace = try directSelectHarness.generate(
            promptTokens: prompt,
            maxNewTokens: maxNewTokens
        )
        XCTAssertEqual(directSelectTrace.generatedTokens, materializedTrace.generatedTokens)

        let materialized = try benchmarkRecurrentEchoGeneration(
            layerCount: 6,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations,
            outputHeadBackend: .aneRMSNormClassifier,
            useDirectTokenSelection: false
        )
        let directSelect = try benchmarkRecurrentEchoGeneration(
            layerCount: 6,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations,
            outputHeadBackend: .aneRMSNormClassifier,
            useDirectTokenSelection: true
        )

        print(
            """
            recurrent generation fused-head materialized median=\(materialized.medianTokenMs) ms/token tps=\(materialized.medianTokensPerSecond) compile=\(materialized.compileTimeMs) trunk=\(materialized.medianTrunkMsPerToken) logits=\(materialized.medianLogitsMsPerToken)
            recurrent generation fused-head direct-select median=\(directSelect.medianTokenMs) ms/token tps=\(directSelect.medianTokensPerSecond) compile=\(directSelect.compileTimeMs) trunk=\(directSelect.medianTrunkMsPerToken) logits=\(directSelect.medianLogitsMsPerToken)
            """
        )

        XCTAssertGreaterThan(materialized.medianTokenMs, 0)
        XCTAssertGreaterThan(directSelect.medianTokenMs, 0)
        XCTAssertGreaterThan(materialized.compileTimeMs, 0)
        XCTAssertGreaterThan(directSelect.compileTimeMs, 0)
    }

    func test_recurrent_generation_fused_pair_direct_select_reports_comparison_on_hardware() throws {
        try requireGenerationHardware()

        let prompt: [UInt16] = [0]
        let warmup = 3
        let iterations = 20
        let maxNewTokens = 8

        let parityWeights = makeEchoRecurrentGenerationWeights(layerCount: 6)
        let singleModel = try ANERecurrentGenerationModel(
            weights: parityWeights,
            layerCount: 6,
            maxSequenceTokens: 32,
            outputHeadBackend: .aneRMSNormClassifier,
            trunkBackend: .singleLayer
        )
        let fusedModel = try ANERecurrentGenerationModel(
            weights: parityWeights,
            layerCount: 6,
            maxSequenceTokens: 32,
            outputHeadBackend: .aneRMSNormClassifier,
            trunkBackend: .fusedTwoLayerPairs
        )
        var singleHarness = DirectTokenSelectionGenerationHarness(
            model: singleModel,
            strategy: .argmax
        )
        var fusedHarness = DirectTokenSelectionGenerationHarness(
            model: fusedModel,
            strategy: .argmax
        )
        let singleTrace = try singleHarness.generate(
            promptTokens: prompt,
            maxNewTokens: maxNewTokens
        )
        let fusedTrace = try fusedHarness.generate(
            promptTokens: prompt,
            maxNewTokens: maxNewTokens
        )
        XCTAssertEqual(fusedTrace.generatedTokens, singleTrace.generatedTokens)

        let single = try benchmarkRecurrentEchoGeneration(
            layerCount: 6,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations,
            outputHeadBackend: .aneRMSNormClassifier,
            useDirectTokenSelection: true,
            trunkBackend: .singleLayer
        )
        let fused = try benchmarkRecurrentEchoGeneration(
            layerCount: 6,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations,
            outputHeadBackend: .aneRMSNormClassifier,
            useDirectTokenSelection: true,
            trunkBackend: .fusedTwoLayerPairs
        )

        print(
            """
            recurrent generation single-layer direct-select median=\(single.medianTokenMs) ms/token tps=\(single.medianTokensPerSecond) compile=\(single.compileTimeMs) trunk=\(single.medianTrunkMsPerToken) logits=\(single.medianLogitsMsPerToken)
            recurrent generation fused-pair direct-select median=\(fused.medianTokenMs) ms/token tps=\(fused.medianTokensPerSecond) compile=\(fused.compileTimeMs) trunk=\(fused.medianTrunkMsPerToken) logits=\(fused.medianLogitsMsPerToken)
            """
        )

        XCTAssertGreaterThan(single.medianTokenMs, 0)
        XCTAssertGreaterThan(fused.medianTokenMs, 0)
        XCTAssertGreaterThan(single.compileTimeMs, 0)
        XCTAssertGreaterThan(fused.compileTimeMs, 0)
    }

    func test_recurrent_generation_fused_pair_direct_select_trunk_lane_spatial_sweep_on_hardware() throws {
        try requireGenerationHardware()

        let prompt: [UInt16] = [0]
        let warmup = 3
        let iterations = 20
        let maxNewTokens = 8
        let laneCandidates = [32, 16, 8, 1]

        let referenceWeights = makeEchoRecurrentGenerationWeights(layerCount: 6)
        let referenceModel = try ANERecurrentGenerationModel(
            weights: referenceWeights,
            layerCount: 6,
            maxSequenceTokens: 32,
            outputHeadBackend: .aneRMSNormClassifier,
            trunkBackend: .fusedTwoLayerPairs,
            trunkLaneSpatial: 32
        )
        var referenceHarness = DirectTokenSelectionGenerationHarness(
            model: referenceModel,
            strategy: .argmax
        )
        let referenceTrace = try referenceHarness.generate(
            promptTokens: prompt,
            maxNewTokens: maxNewTokens
        )

        for laneSpatial in laneCandidates {
            do {
                let parityWeights = makeEchoRecurrentGenerationWeights(layerCount: 6)
                let candidateModel = try ANERecurrentGenerationModel(
                    weights: parityWeights,
                    layerCount: 6,
                    maxSequenceTokens: 32,
                    outputHeadBackend: .aneRMSNormClassifier,
                    trunkBackend: .fusedTwoLayerPairs,
                    trunkLaneSpatial: laneSpatial
                )
                var candidateHarness = DirectTokenSelectionGenerationHarness(
                    model: candidateModel,
                    strategy: .argmax
                )
                let candidateTrace = try candidateHarness.generate(
                    promptTokens: prompt,
                    maxNewTokens: maxNewTokens
                )
                XCTAssertEqual(candidateTrace.generatedTokens, referenceTrace.generatedTokens)

                let sample = try benchmarkRecurrentEchoGeneration(
                    layerCount: 6,
                    promptTokens: prompt,
                    maxNewTokens: maxNewTokens,
                    warmup: warmup,
                    iterations: iterations,
                    outputHeadBackend: .aneRMSNormClassifier,
                    useDirectTokenSelection: true,
                    trunkBackend: .fusedTwoLayerPairs,
                    trunkLaneSpatial: laneSpatial
                )
                print(
                    "recurrent generation fused-pair direct-select lane=\(laneSpatial) median=\(sample.medianTokenMs) ms/token tps=\(sample.medianTokensPerSecond) compile=\(sample.compileTimeMs) trunk=\(sample.medianTrunkMsPerToken) logits=\(sample.medianLogitsMsPerToken)"
                )
                XCTAssertGreaterThan(sample.medianTokenMs, 0)
                XCTAssertGreaterThan(sample.compileTimeMs, 0)
            } catch {
                if laneSpatial == 32 {
                    throw error
                }
                print("recurrent generation fused-pair direct-select lane=\(laneSpatial) unsupported: \(error)")
            }
        }
    }

    func test_recurrent_generation_fused_triplet_direct_select_reports_comparison_on_hardware() throws {
        try requireGenerationHardware()

        let prompt: [UInt16] = [0]
        let warmup = 3
        let iterations = 20
        let maxNewTokens = 8

        let parityWeights = makeEchoRecurrentGenerationWeights(layerCount: 6)
        let pairModel = try ANERecurrentGenerationModel(
            weights: parityWeights,
            layerCount: 6,
            maxSequenceTokens: 32,
            outputHeadBackend: .aneRMSNormClassifier,
            trunkBackend: .fusedTwoLayerPairs
        )
        let tripletModel = try ANERecurrentGenerationModel(
            weights: parityWeights,
            layerCount: 6,
            maxSequenceTokens: 32,
            outputHeadBackend: .aneRMSNormClassifier,
            trunkBackend: .fusedThreeLayerTriplets
        )
        var pairHarness = DirectTokenSelectionGenerationHarness(
            model: pairModel,
            strategy: .argmax
        )
        var tripletHarness = DirectTokenSelectionGenerationHarness(
            model: tripletModel,
            strategy: .argmax
        )
        let pairTrace = try pairHarness.generate(
            promptTokens: prompt,
            maxNewTokens: maxNewTokens
        )
        let tripletTrace = try tripletHarness.generate(
            promptTokens: prompt,
            maxNewTokens: maxNewTokens
        )
        XCTAssertEqual(tripletTrace.generatedTokens, pairTrace.generatedTokens)

        let pair = try benchmarkRecurrentEchoGeneration(
            layerCount: 6,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations,
            outputHeadBackend: .aneRMSNormClassifier,
            useDirectTokenSelection: true,
            trunkBackend: .fusedTwoLayerPairs
        )
        let triplet = try benchmarkRecurrentEchoGeneration(
            layerCount: 6,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations,
            outputHeadBackend: .aneRMSNormClassifier,
            useDirectTokenSelection: true,
            trunkBackend: .fusedThreeLayerTriplets
        )

        print(
            """
            recurrent generation fused-pair direct-select median=\(pair.medianTokenMs) ms/token tps=\(pair.medianTokensPerSecond) compile=\(pair.compileTimeMs) trunk=\(pair.medianTrunkMsPerToken) logits=\(pair.medianLogitsMsPerToken)
            recurrent generation fused-triplet direct-select median=\(triplet.medianTokenMs) ms/token tps=\(triplet.medianTokensPerSecond) compile=\(triplet.compileTimeMs) trunk=\(triplet.medianTrunkMsPerToken) logits=\(triplet.medianLogitsMsPerToken)
            """
        )

        XCTAssertGreaterThan(pair.medianTokenMs, 0)
        XCTAssertGreaterThan(triplet.medianTokenMs, 0)
        XCTAssertGreaterThan(pair.compileTimeMs, 0)
        XCTAssertGreaterThan(triplet.compileTimeMs, 0)
    }

    func test_recurrent_generation_fused_triplet_direct_select_vs_autoregressive_materialized_on_hardware() throws {
        try requireGenerationHardware()

        let prompt: [UInt16] = [0]
        let maxNewTokens = 8
        let weights = makeEchoRecurrentGenerationWeights(layerCount: 6)

        let materializedModel = try ANERecurrentGenerationModel(
            weights: weights,
            layerCount: 6,
            maxSequenceTokens: 32,
            outputHeadBackend: .aneRMSNormClassifier,
            trunkBackend: .fusedThreeLayerTriplets,
            trunkLaneSpatial: 32,
            outputHeadLaneSpatial: 32
        )
        let directModel = try ANERecurrentGenerationModel(
            weights: weights,
            layerCount: 6,
            maxSequenceTokens: 32,
            outputHeadBackend: .aneRMSNormClassifier,
            trunkBackend: .fusedThreeLayerTriplets,
            trunkLaneSpatial: 32,
            outputHeadLaneSpatial: 32
        )

        var materializedHarness = AutoregressiveGenerationHarness(
            model: materializedModel,
            strategy: .argmax
        )
        var directHarness = DirectTokenSelectionGenerationHarness(
            model: directModel,
            strategy: .argmax
        )

        let materializedTrace = try materializedHarness.generate(
            promptTokens: prompt,
            maxNewTokens: maxNewTokens
        )
        let directTrace = try directHarness.generate(
            promptTokens: prompt,
            maxNewTokens: maxNewTokens
        )

        XCTAssertEqual(directTrace.generatedTokens, materializedTrace.generatedTokens)
    }

    func test_recurrent_generation_fused_triplet_direct_select_output_head_lane_sweep_on_hardware() throws {
        try requireGenerationHardware()

        let prompt: [UInt16] = [0]
        let warmup = 3
        let iterations = 20
        let maxNewTokens = 8
        let headLaneCandidates = [32, 16, 8, 1]

        let referenceWeights = makeEchoRecurrentGenerationWeights(layerCount: 6)
        let referenceModel = try ANERecurrentGenerationModel(
            weights: referenceWeights,
            layerCount: 6,
            maxSequenceTokens: 32,
            outputHeadBackend: .aneRMSNormClassifier,
            trunkBackend: .fusedThreeLayerTriplets,
            outputHeadLaneSpatial: 32
        )
        var referenceHarness = DirectTokenSelectionGenerationHarness(
            model: referenceModel,
            strategy: .argmax
        )
        let referenceTrace = try referenceHarness.generate(
            promptTokens: prompt,
            maxNewTokens: maxNewTokens
        )

        for outputHeadLaneSpatial in headLaneCandidates {
            do {
                let parityWeights = makeEchoRecurrentGenerationWeights(layerCount: 6)
                let candidateModel = try ANERecurrentGenerationModel(
                    weights: parityWeights,
                    layerCount: 6,
                    maxSequenceTokens: 32,
                    outputHeadBackend: .aneRMSNormClassifier,
                    trunkBackend: .fusedThreeLayerTriplets,
                    outputHeadLaneSpatial: outputHeadLaneSpatial
                )
                var candidateHarness = DirectTokenSelectionGenerationHarness(
                    model: candidateModel,
                    strategy: .argmax
                )
                let candidateTrace = try candidateHarness.generate(
                    promptTokens: prompt,
                    maxNewTokens: maxNewTokens
                )
                XCTAssertEqual(candidateTrace.generatedTokens, referenceTrace.generatedTokens)

                let sample = try benchmarkRecurrentEchoGeneration(
                    layerCount: 6,
                    promptTokens: prompt,
                    maxNewTokens: maxNewTokens,
                    warmup: warmup,
                    iterations: iterations,
                    outputHeadBackend: .aneRMSNormClassifier,
                    useDirectTokenSelection: true,
                    trunkBackend: .fusedThreeLayerTriplets,
                    outputHeadLaneSpatial: outputHeadLaneSpatial
                )
                print(
                    "recurrent generation fused-triplet direct-select output-head-lane=\(outputHeadLaneSpatial) median=\(sample.medianTokenMs) ms/token tps=\(sample.medianTokensPerSecond) compile=\(sample.compileTimeMs) trunk=\(sample.medianTrunkMsPerToken) logits=\(sample.medianLogitsMsPerToken)"
                )
                XCTAssertGreaterThan(sample.medianTokenMs, 0)
                XCTAssertGreaterThan(sample.compileTimeMs, 0)
            } catch {
                if outputHeadLaneSpatial == 32 {
                    throw error
                }
                print("recurrent generation fused-triplet direct-select output-head-lane=\(outputHeadLaneSpatial) unsupported: \(error)")
            }
        }
    }

    func test_concurrent_generation_scaling_sample_normalizes_round_time_by_stream_and_token_count() {
        let sample = ConcurrentGenerationScalingSample(
            streamCount: 4,
            tokensPerStream: 8,
            compileTimeMs: 123.0,
            roundLatenciesMs: [96.0, 80.0, 88.0]
        )

        XCTAssertEqual(sample.medianRoundLatencyMs, 88.0, accuracy: 0.0001)
        XCTAssertEqual(sample.medianMsPerToken, 2.75, accuracy: 0.0001)
        XCTAssertEqual(sample.aggregateTokensPerSecond, 363.6363636, accuracy: 0.0001)
        XCTAssertEqual(sample.perStreamTokensPerSecond, 90.9090909, accuracy: 0.0001)
        XCTAssertEqual(sample.compileTimeMs, 123.0, accuracy: 0.0001)
    }

    func test_concurrent_generation_synthetic_runner_builds_isolated_workers_and_synchronizes_rounds() throws {
        let sample = try runSyntheticConcurrentRoundBenchmark(
            streamCount: 3,
            warmup: 1,
            iterations: 2,
            tokensPerStream: 4
        )

        XCTAssertEqual(sample.streamCount, 3)
        XCTAssertEqual(sample.builderStreamIndices, [0, 1, 2])
        XCTAssertTrue(sample.roundOrderIsSynchronized)
        XCTAssertEqual(sample.completedRoundsByStream, [2, 2, 2])
        XCTAssertGreaterThan(sample.compileTimeMs, 0)
        XCTAssertEqual(sample.roundLatenciesMs.count, 2)
    }

    func test_recurrent_generation_concurrent_multistream_scaling_reports_matched_ane_and_coreml_baselines_on_hardware() throws {
        try requireGenerationHardware()

        let prompt: [UInt16] = [0]
        let warmup = 3
        let iterations = 20
        let maxNewTokens = 8
        let streamCounts = [1, 2, 3, 4, 5, 6]
        let modelPath = "benchmarks/models/transformer_6layer.mlpackage"

        let ane = try benchmarkConcurrentRecurrentEchoGeneration(
            layerCount: 6,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations,
            streamCounts: streamCounts,
            outputHeadBackend: .aneRMSNormClassifier,
            trunkBackend: .fusedThreeLayerTriplets,
            useDirectTokenSelection: true
        )
        let coreml = try benchmarkConcurrentCoreMLGeneration(
            modelPath: modelPath,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations,
            streamCounts: streamCounts
        )

        XCTAssertEqual(ane.promptLength, prompt.count)
        XCTAssertEqual(coreml.promptLength, prompt.count)
        XCTAssertEqual(ane.maxNewTokens, maxNewTokens)
        XCTAssertEqual(coreml.maxNewTokens, maxNewTokens)
        XCTAssertEqual(ane.warmupCount, warmup)
        XCTAssertEqual(coreml.warmupCount, warmup)
        XCTAssertEqual(ane.iterationCount, iterations)
        XCTAssertEqual(coreml.iterationCount, iterations)
        XCTAssertEqual(ane.samples.map(\.streamCount), streamCounts)
        XCTAssertEqual(coreml.samples.map(\.streamCount), streamCounts)
        XCTAssertTrue(ane.samples.allSatisfy { $0.medianMsPerToken > 0 })
        XCTAssertTrue(coreml.samples.allSatisfy { $0.medianMsPerToken > 0 })

        for idx in 0..<streamCounts.count {
            let aneSample = ane.samples[idx]
            let coremlSample = coreml.samples[idx]
            print(
                """
                concurrent ane streams=\(aneSample.streamCount) median_ms_token=\(aneSample.medianMsPerToken) aggregate_tps=\(aneSample.aggregateTokensPerSecond) per_stream_tps=\(aneSample.perStreamTokensPerSecond) compile=\(aneSample.compileTimeMs) round_ms=\(aneSample.medianRoundLatencyMs)
                concurrent coreml streams=\(coremlSample.streamCount) median_ms_token=\(coremlSample.medianMsPerToken) aggregate_tps=\(coremlSample.aggregateTokensPerSecond) per_stream_tps=\(coremlSample.perStreamTokensPerSecond) compile=\(coremlSample.compileTimeMs) round_ms=\(coremlSample.medianRoundLatencyMs)
                """
            )
        }
    }

    private func machMilliseconds(_ deltaTicks: UInt64) -> Double {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        let nanos = Double(deltaTicks) * Double(info.numer) / Double(info.denom)
        return nanos / 1_000_000.0
    }

    private final class LockedBox<Value>: @unchecked Sendable {
        private let lock = NSLock()
        private var value: Value

        init(_ value: Value) {
            self.value = value
        }

        func get() -> Value {
            lock.lock()
            defer { lock.unlock() }
            return value
        }

        func set(_ newValue: Value) {
            lock.lock()
            value = newValue
            lock.unlock()
        }

        func withValue<R>(_ body: (inout Value) -> R) -> R {
            lock.lock()
            defer { lock.unlock() }
            return body(&value)
        }
    }

    private struct ConcurrentGenerationScalingSample {
        let streamCount: Int
        let tokensPerStream: Int
        let compileTimeMs: Double
        let roundLatenciesMs: [Double]

        var medianRoundLatencyMs: Double {
            median(roundLatenciesMs)
        }

        var medianMsPerToken: Double {
            let totalTokens = streamCount * tokensPerStream
            guard totalTokens > 0 else { return 0 }
            return medianRoundLatencyMs / Double(totalTokens)
        }

        var aggregateTokensPerSecond: Double {
            guard medianRoundLatencyMs > 0 else { return 0 }
            return Double(streamCount * tokensPerStream) * 1000.0 / medianRoundLatencyMs
        }

        var perStreamTokensPerSecond: Double {
            guard streamCount > 0 else { return 0 }
            return aggregateTokensPerSecond / Double(streamCount)
        }
    }

    private struct ConcurrentGenerationScalingReport {
        let label: String
        let promptLength: Int
        let maxNewTokens: Int
        let warmupCount: Int
        let iterationCount: Int
        let samples: [ConcurrentGenerationScalingSample]
    }

    private struct SyntheticConcurrentRoundBenchmarkSample {
        let streamCount: Int
        let tokensPerStream: Int
        let compileTimeMs: Double
        let roundLatenciesMs: [Double]
        let builderStreamIndices: [Int]
        let completedRoundsByStream: [Int]
        let roundOrderIsSynchronized: Bool
    }

    private func runSyntheticConcurrentRoundBenchmark(
        streamCount: Int,
        warmup: Int,
        iterations: Int,
        tokensPerStream: Int
    ) throws -> SyntheticConcurrentRoundBenchmarkSample {
        let builderStreamIndices = Array(0..<streamCount)
        let compileStart = GenerationClock.now()
        let startSignals = (0..<streamCount).map { _ in DispatchSemaphore(value: 0) }
        let doneSignals = (0..<streamCount).map { _ in DispatchSemaphore(value: 0) }
        let readySignals = (0..<streamCount).map { _ in DispatchSemaphore(value: 0) }
        let shutdown = LockedBox(false)
        let activeRound = LockedBox(0)
        let completedRounds = LockedBox([Int](repeating: 0, count: streamCount))
        let roundOrderIsSynchronized = LockedBox(true)
        let exitGroup = DispatchGroup()

        for streamIndex in builderStreamIndices {
            let startSignal = startSignals[streamIndex]
            let doneSignal = doneSignals[streamIndex]
            let readySignal = readySignals[streamIndex]
            let queue = DispatchQueue(label: "com.espresso.synthetic.stream.\(streamIndex)")
            exitGroup.enter()
            queue.async {
                defer { exitGroup.leave() }
                readySignal.signal()

                while true {
                    startSignal.wait()
                    if shutdown.get() {
                        return
                    }

                    let round = activeRound.get()
                    let minimumCompleted = completedRounds.get().min() ?? 0
                    if minimumCompleted < round {
                        roundOrderIsSynchronized.set(false)
                    }

                    usleep(useconds_t(500 * (streamIndex + 1)))
                    completedRounds.withValue { values in
                        values[streamIndex] += 1
                    }
                    doneSignal.signal()
                }
            }
        }

        for readySignal in readySignals {
            readySignal.wait()
        }
        let compileTimeMs = machMilliseconds(GenerationClock.now() - compileStart)

        for round in 0..<warmup {
            activeRound.set(round)
            for startSignal in startSignals {
                startSignal.signal()
            }
            for doneSignal in doneSignals {
                doneSignal.wait()
            }
        }
        completedRounds.set([Int](repeating: 0, count: streamCount))

        var roundLatenciesMs: [Double] = []
        roundLatenciesMs.reserveCapacity(iterations)
        for round in 0..<iterations {
            activeRound.set(round)
            let start = GenerationClock.now()
            for startSignal in startSignals {
                startSignal.signal()
            }
            for doneSignal in doneSignals {
                doneSignal.wait()
            }
            roundLatenciesMs.append(machMilliseconds(GenerationClock.now() - start))
        }

        shutdown.set(true)
        for startSignal in startSignals {
            startSignal.signal()
        }
        exitGroup.wait()

        return SyntheticConcurrentRoundBenchmarkSample(
            streamCount: streamCount,
            tokensPerStream: tokensPerStream,
            compileTimeMs: compileTimeMs,
            roundLatenciesMs: roundLatenciesMs,
            builderStreamIndices: builderStreamIndices,
            completedRoundsByStream: completedRounds.get(),
            roundOrderIsSynchronized: roundOrderIsSynchronized.get()
        )
    }

    private func benchmarkConcurrentRecurrentEchoGeneration(
        layerCount: Int,
        promptTokens: [UInt16],
        maxNewTokens: Int,
        warmup: Int,
        iterations: Int,
        streamCounts: [Int],
        outputHeadBackend: GenerationOutputHeadBackend = .cpu,
        trunkBackend: RecurrentGenerationTrunkBackend = .singleLayer,
        useDirectTokenSelection: Bool = false,
        trunkLaneSpatial: Int = 32,
        outputHeadLaneSpatial: Int = 32
    ) throws -> ConcurrentGenerationScalingReport {
        var samples: [ConcurrentGenerationScalingSample] = []
        samples.reserveCapacity(streamCounts.count)

        for streamCount in streamCounts {
            let compileStart = GenerationClock.now()
            let startSignals = (0..<streamCount).map { _ in DispatchSemaphore(value: 0) }
            let doneSignals = (0..<streamCount).map { _ in DispatchSemaphore(value: 0) }
            let readySignals = (0..<streamCount).map { _ in DispatchSemaphore(value: 0) }
            let shutdown = LockedBox(false)
            let firstError = LockedBox<Error?>(nil)
            let exitGroup = DispatchGroup()

            for streamIndex in 0..<streamCount {
                let startSignal = startSignals[streamIndex]
                let doneSignal = doneSignals[streamIndex]
                let readySignal = readySignals[streamIndex]
                let queue = DispatchQueue(label: "com.espresso.concurrent.ane.stream.\(streamIndex)")
                exitGroup.enter()
                queue.async {
                    defer { exitGroup.leave() }
                    do {
                        let weights = makeEchoRecurrentGenerationWeights(layerCount: layerCount)
                        let model = try ANERecurrentGenerationModel(
                            weights: weights,
                            layerCount: layerCount,
                            maxSequenceTokens: 32,
                            outputHeadBackend: outputHeadBackend,
                            trunkBackend: trunkBackend,
                            trunkLaneSpatial: trunkLaneSpatial,
                            outputHeadLaneSpatial: outputHeadLaneSpatial
                        )

                        if useDirectTokenSelection {
                            var harness = DirectTokenSelectionGenerationHarness(
                                model: model,
                                strategy: .argmax
                            )
                            readySignal.signal()
                            while true {
                                startSignal.wait()
                                if shutdown.get() {
                                    return
                                }
                                do {
                                    _ = try harness.generate(
                                        promptTokens: promptTokens,
                                        maxNewTokens: maxNewTokens
                                    )
                                } catch {
                                    firstError.withValue { current in
                                        if current == nil { current = error }
                                    }
                                }
                                doneSignal.signal()
                            }
                        } else {
                            var harness = AutoregressiveGenerationHarness(
                                model: model,
                                strategy: .argmax
                            )
                            readySignal.signal()
                            while true {
                                startSignal.wait()
                                if shutdown.get() {
                                    return
                                }
                                do {
                                    _ = try harness.generate(
                                        promptTokens: promptTokens,
                                        maxNewTokens: maxNewTokens
                                    )
                                } catch {
                                    firstError.withValue { current in
                                        if current == nil { current = error }
                                    }
                                }
                                doneSignal.signal()
                            }
                        }
                    } catch {
                        firstError.withValue { current in
                            if current == nil { current = error }
                        }
                        readySignal.signal()
                        while true {
                            startSignal.wait()
                            if shutdown.get() {
                                return
                            }
                            doneSignal.signal()
                        }
                    }
                }
            }

            for readySignal in readySignals {
                readySignal.wait()
            }
            let compileTimeMs = machMilliseconds(GenerationClock.now() - compileStart)

            if let error = firstError.get() {
                shutdown.set(true)
                for startSignal in startSignals {
                    startSignal.signal()
                }
                exitGroup.wait()
                throw error
            }

            for _ in 0..<warmup {
                for startSignal in startSignals {
                    startSignal.signal()
                }
                for doneSignal in doneSignals {
                    doneSignal.wait()
                }
                if let error = firstError.get() {
                    shutdown.set(true)
                    for startSignal in startSignals {
                        startSignal.signal()
                    }
                    exitGroup.wait()
                    throw error
                }
            }

            var roundLatenciesMs: [Double] = []
            roundLatenciesMs.reserveCapacity(iterations)
            for _ in 0..<iterations {
                let start = GenerationClock.now()
                for startSignal in startSignals {
                    startSignal.signal()
                }
                for doneSignal in doneSignals {
                    doneSignal.wait()
                }
                if let error = firstError.get() {
                    shutdown.set(true)
                    for startSignal in startSignals {
                        startSignal.signal()
                    }
                    exitGroup.wait()
                    throw error
                }
                roundLatenciesMs.append(machMilliseconds(GenerationClock.now() - start))
            }

            shutdown.set(true)
            for startSignal in startSignals {
                startSignal.signal()
            }
            exitGroup.wait()

            samples.append(
                ConcurrentGenerationScalingSample(
                    streamCount: streamCount,
                    tokensPerStream: maxNewTokens,
                    compileTimeMs: compileTimeMs,
                    roundLatenciesMs: roundLatenciesMs
                )
            )
        }

        return ConcurrentGenerationScalingReport(
            label: "ANE Recurrent Concurrent",
            promptLength: promptTokens.count,
            maxNewTokens: maxNewTokens,
            warmupCount: warmup,
            iterationCount: iterations,
            samples: samples
        )
    }

    private func benchmarkConcurrentCoreMLGeneration(
        modelPath: String,
        promptTokens: [UInt16],
        maxNewTokens: Int,
        warmup: Int,
        iterations: Int,
        streamCounts: [Int]
    ) throws -> ConcurrentGenerationScalingReport {
        var samples: [ConcurrentGenerationScalingSample] = []
        samples.reserveCapacity(streamCounts.count)

        for streamCount in streamCounts {
            let compileStart = GenerationClock.now()
            let startSignals = (0..<streamCount).map { _ in DispatchSemaphore(value: 0) }
            let doneSignals = (0..<streamCount).map { _ in DispatchSemaphore(value: 0) }
            let readySignals = (0..<streamCount).map { _ in DispatchSemaphore(value: 0) }
            let shutdown = LockedBox(false)
            let firstError = LockedBox<Error?>(nil)
            let exitGroup = DispatchGroup()

            for streamIndex in 0..<streamCount {
                let startSignal = startSignals[streamIndex]
                let doneSignal = doneSignals[streamIndex]
                let readySignal = readySignals[streamIndex]
                let queue = DispatchQueue(label: "com.espresso.concurrent.coreml.stream.\(streamIndex)")
                exitGroup.enter()
                queue.async {
                    defer { exitGroup.leave() }
                    do {
                        let headWeights = makeEchoGenerationWeights(layerCount: 1)
                        let model = try CoreMLGenerationBenchmarkModel(
                            modelPath: modelPath,
                            headWeights: headWeights,
                            maxSequenceTokens: 32
                        )
                        var harness = AutoregressiveGenerationHarness(
                            model: model,
                            strategy: .argmax
                        )
                        readySignal.signal()
                        while true {
                            startSignal.wait()
                            if shutdown.get() {
                                return
                            }
                            do {
                                _ = try harness.generate(
                                    promptTokens: promptTokens,
                                    maxNewTokens: maxNewTokens
                                )
                            } catch {
                                firstError.withValue { current in
                                    if current == nil { current = error }
                                }
                            }
                            doneSignal.signal()
                        }
                    } catch {
                        firstError.withValue { current in
                            if current == nil { current = error }
                        }
                        readySignal.signal()
                        while true {
                            startSignal.wait()
                            if shutdown.get() {
                                return
                            }
                            doneSignal.signal()
                        }
                    }
                }
            }

            for readySignal in readySignals {
                readySignal.wait()
            }
            let compileTimeMs = machMilliseconds(GenerationClock.now() - compileStart)

            if let error = firstError.get() {
                shutdown.set(true)
                for startSignal in startSignals {
                    startSignal.signal()
                }
                exitGroup.wait()
                throw error
            }

            for _ in 0..<warmup {
                for startSignal in startSignals {
                    startSignal.signal()
                }
                for doneSignal in doneSignals {
                    doneSignal.wait()
                }
                if let error = firstError.get() {
                    shutdown.set(true)
                    for startSignal in startSignals {
                        startSignal.signal()
                    }
                    exitGroup.wait()
                    throw error
                }
            }

            var roundLatenciesMs: [Double] = []
            roundLatenciesMs.reserveCapacity(iterations)
            for _ in 0..<iterations {
                let start = GenerationClock.now()
                for startSignal in startSignals {
                    startSignal.signal()
                }
                for doneSignal in doneSignals {
                    doneSignal.wait()
                }
                if let error = firstError.get() {
                    shutdown.set(true)
                    for startSignal in startSignals {
                        startSignal.signal()
                    }
                    exitGroup.wait()
                    throw error
                }
                roundLatenciesMs.append(machMilliseconds(GenerationClock.now() - start))
            }

            shutdown.set(true)
            for startSignal in startSignals {
                startSignal.signal()
            }
            exitGroup.wait()

            samples.append(
                ConcurrentGenerationScalingSample(
                    streamCount: streamCount,
                    tokensPerStream: maxNewTokens,
                    compileTimeMs: compileTimeMs,
                    roundLatenciesMs: roundLatenciesMs
                )
            )
        }

        return ConcurrentGenerationScalingReport(
            label: "CoreML Concurrent",
            promptLength: promptTokens.count,
            maxNewTokens: maxNewTokens,
            warmupCount: warmup,
            iterationCount: iterations,
            samples: samples
        )
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
        outputHeadBackend: GenerationOutputHeadBackend = .cpu,
        useDirectTokenSelection: Bool = false,
        trunkBackend: RecurrentGenerationTrunkBackend = .singleLayer,
        trunkLaneSpatial: Int = 32,
        outputHeadLaneSpatial: Int = 32
    ) throws -> GenerationBenchmarkSample {
        let weights = makeEchoRecurrentGenerationWeights(layerCount: layerCount)
        let model = try ANERecurrentGenerationModel(
            weights: weights,
            layerCount: layerCount,
            maxSequenceTokens: 32,
            outputHeadBackend: outputHeadBackend,
            trunkBackend: trunkBackend,
            trunkLaneSpatial: trunkLaneSpatial,
            outputHeadLaneSpatial: outputHeadLaneSpatial
        )
        if useDirectTokenSelection {
            var harness = DirectTokenSelectionGenerationHarness(model: model, strategy: .argmax)
            return try benchmarkDirectSelectionHarness(
                harness: &harness,
                promptTokens: promptTokens,
                maxNewTokens: maxNewTokens,
                warmup: warmup,
                iterations: iterations
            )
        } else {
            var harness = AutoregressiveGenerationHarness(model: model, strategy: .argmax)
            return try benchmarkAutoregressiveHarness(
                harness: &harness,
                promptTokens: promptTokens,
                maxNewTokens: maxNewTokens,
                warmup: warmup,
                iterations: iterations
            )
        }
    }

    private func benchmarkRecurrentExactTwoTokenUpperBoundGeneration(
        layerCount: Int,
        promptTokens: [UInt16],
        maxNewTokens: Int,
        warmup: Int,
        iterations: Int,
        outputHeadBackend: GenerationOutputHeadBackend = .aneRMSNormClassifier,
        trunkBackend: RecurrentGenerationTrunkBackend = .fusedThreeLayerTriplets,
        trunkLaneSpatial: Int = 32,
        outputHeadLaneSpatial: Int = 32
    ) throws -> ExactTwoTokenBenchmarkSample {
        let weights = makeEchoRecurrentGenerationWeights(layerCount: layerCount)
        let model = try ANEExactTwoTokenUpperBoundGenerationModel(
            weights: weights,
            layerCount: layerCount,
            maxSequenceTokens: 32,
            outputHeadBackend: outputHeadBackend,
            trunkBackend: trunkBackend,
            trunkLaneSpatial: trunkLaneSpatial,
            outputHeadLaneSpatial: outputHeadLaneSpatial
        )
        var harness = ExactTwoTokenGenerationHarness(model: model, strategy: .argmax)
        return try benchmarkExactTwoTokenHarness(
            harness: &harness,
            promptTokens: promptTokens,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations
        )
    }

    private func benchmarkRecurrentExactTwoTokenBranchStatePromotionGeneration(
        layerCount: Int,
        promptTokens: [UInt16],
        maxNewTokens: Int,
        warmup: Int,
        iterations: Int,
        outputHeadBackend: GenerationOutputHeadBackend = .aneRMSNormClassifier,
        trunkLaneSpatial: Int = 32,
        outputHeadLaneSpatial: Int = 32
    ) throws -> ExactTwoTokenBenchmarkSample {
        let weights = makeEchoRecurrentGenerationWeights(layerCount: layerCount)
        let model = try ANEExactTwoTokenBranchStatePromotionModel(
            weights: weights,
            layerCount: layerCount,
            maxSequenceTokens: 32,
            outputHeadBackend: outputHeadBackend,
            trunkLaneSpatial: trunkLaneSpatial,
            outputHeadLaneSpatial: outputHeadLaneSpatial
        )
        var harness = ExactTwoTokenGenerationHarness(model: model, strategy: .argmax)
        return try benchmarkExactTwoTokenHarness(
            harness: &harness,
            promptTokens: promptTokens,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations
        )
    }

    private func measureRecurrentSingleLayerControlCompileInitOnly(
        layerCount: Int,
        outputHeadBackend: GenerationOutputHeadBackend = .aneRMSNormClassifier,
        trunkLaneSpatial: Int = 32,
        outputHeadLaneSpatial: Int = 32
    ) throws -> CompileInitBenchmarkSample {
        let weights = makeEchoRecurrentGenerationWeights(layerCount: layerCount)
        let start = mach_absolute_time()
        let model = try ANERecurrentGenerationModel(
            weights: weights,
            layerCount: layerCount,
            maxSequenceTokens: 32,
            outputHeadBackend: outputHeadBackend,
            trunkBackend: .singleLayer,
            trunkLaneSpatial: trunkLaneSpatial,
            outputHeadLaneSpatial: outputHeadLaneSpatial
        )
        let wallInitMs = machMilliseconds(mach_absolute_time() - start)
        return CompileInitBenchmarkSample(
            wallInitMs: wallInitMs,
            reportedCompileTimeMs: model.performanceSnapshot.compileTimeMs
        )
    }

    private func measureRecurrentExactTwoTokenBranchStatePromotionCompileInitOnly(
        layerCount: Int,
        outputHeadBackend: GenerationOutputHeadBackend = .aneRMSNormClassifier,
        trunkLaneSpatial: Int = 32,
        outputHeadLaneSpatial: Int = 32
    ) throws -> CompileInitBenchmarkSample {
        let weights = makeEchoRecurrentGenerationWeights(layerCount: layerCount)
        let start = mach_absolute_time()
        let model = try ANEExactTwoTokenBranchStatePromotionModel(
            weights: weights,
            layerCount: layerCount,
            maxSequenceTokens: 32,
            outputHeadBackend: outputHeadBackend,
            trunkLaneSpatial: trunkLaneSpatial,
            outputHeadLaneSpatial: outputHeadLaneSpatial
        )
        let wallInitMs = machMilliseconds(mach_absolute_time() - start)
        return CompileInitBenchmarkSample(
            wallInitMs: wallInitMs,
            reportedCompileTimeMs: model.performanceSnapshot.compileTimeMs
        )
    }

    private func benchmarkExactTwoTokenHarness<Model>(
        harness: inout ExactTwoTokenGenerationHarness<Model>,
        promptTokens: [UInt16],
        maxNewTokens: Int,
        warmup: Int,
        iterations: Int
    ) throws -> ExactTwoTokenBenchmarkSample
    where Model: ExactTwoTokenGeneratingLanguageModel & GenerationPerformanceTrackable, Model: ~Copyable {
        var tokenLatencies: [Double] = []
        var throughput: [Double] = []
        var committedExactTokensPerPass: [Double] = []
        var acceptedFutureTokensPerPass: [Double] = []
        var proposerMsPerPass: [Double] = []
        var verifierTrunkMsPerPass: [Double] = []
        var verifierLogitsMsPerPass: [Double] = []
        var stateAdvanceMsPerPass: [Double] = []

        tokenLatencies.reserveCapacity(iterations)
        throughput.reserveCapacity(iterations)
        committedExactTokensPerPass.reserveCapacity(iterations)
        acceptedFutureTokensPerPass.reserveCapacity(iterations)
        proposerMsPerPass.reserveCapacity(iterations)
        verifierTrunkMsPerPass.reserveCapacity(iterations)
        verifierLogitsMsPerPass.reserveCapacity(iterations)
        stateAdvanceMsPerPass.reserveCapacity(iterations)

        let compileTimeMs = harness.model.performanceSnapshot.compileTimeMs

        for iter in 0..<(warmup + iterations) {
            let trace = try harness.generate(promptTokens: promptTokens, maxNewTokens: maxNewTokens)
            if iter >= warmup {
                tokenLatencies.append(trace.totalLatencyMs / Double(maxNewTokens))
                throughput.append(trace.effectiveTokensPerSecond)
                committedExactTokensPerPass.append(trace.committedExactTokensPerPass)
                acceptedFutureTokensPerPass.append(trace.acceptedFutureTokensPerPass)
                proposerMsPerPass.append(trace.proposerLatencyMsPerPass)
                verifierTrunkMsPerPass.append(trace.verifierTrunkLatencyMsPerPass)
                verifierLogitsMsPerPass.append(trace.verifierLogitsLatencyMsPerPass)
                stateAdvanceMsPerPass.append(trace.stateAdvanceLatencyMsPerPass)
            }
        }

        return ExactTwoTokenBenchmarkSample(
            medianTokenMs: median(tokenLatencies),
            medianTokensPerSecond: median(throughput),
            compileTimeMs: compileTimeMs,
            medianCommittedExactTokensPerPass: median(committedExactTokensPerPass),
            medianAcceptedFutureTokensPerPass: median(acceptedFutureTokensPerPass),
            medianProposerMsPerPass: median(proposerMsPerPass),
            medianVerifierTrunkMsPerPass: median(verifierTrunkMsPerPass),
            medianVerifierLogitsMsPerPass: median(verifierLogitsMsPerPass),
            medianStateAdvanceMsPerPass: median(stateAdvanceMsPerPass)
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

    private func benchmarkDirectSelectionHarness<Model>(
        harness: inout DirectTokenSelectionGenerationHarness<Model>,
        promptTokens: [UInt16],
        maxNewTokens: Int,
        warmup: Int,
        iterations: Int
    ) throws -> GenerationBenchmarkSample
    where Model: DirectTokenSelectingLanguageModel & GenerationPerformanceTrackable, Model: ~Copyable {
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

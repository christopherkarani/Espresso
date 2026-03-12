import Accelerate
import XCTest
import ANETypes
import ANERuntime
import ANEInterop
import CoreML
import CPUOps
import IOSurface
import Metal
@testable import MILGenerator
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

    func test_batched_multistream_scaling_reports_aggregate_throughput_on_hardware() throws {
        try requireGenerationHardware()

        let prompt: [UInt16] = [0]
        let warmup = 3
        let iterations = 20
        let maxNewTokens = 8
        let streamCounts = [1, 2, 3, 4, 5, 6]
        let modelPath = "benchmarks/models/transformer_6layer.mlpackage"

        let batched = try benchmarkBatchedRecurrentGeneration(
            layerCount: 6,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations,
            streamCounts: streamCounts
        )

        let concurrent = try benchmarkConcurrentRecurrentEchoGeneration(
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

        XCTAssertEqual(batched.samples.map(\.streamCount), streamCounts)
        XCTAssertTrue(batched.samples.allSatisfy { $0.medianMsPerToken > 0 })

        for idx in 0..<streamCounts.count {
            let bSample = batched.samples[idx]
            let aSample = concurrent.samples[idx]
            let cSample = coreml.samples[idx]
            print(
                """
                batched ane streams=\(bSample.streamCount) median_ms_token=\(bSample.medianMsPerToken) aggregate_tps=\(bSample.aggregateTokensPerSecond) per_stream_tps=\(bSample.perStreamTokensPerSecond) compile=\(bSample.compileTimeMs) round_ms=\(bSample.medianRoundLatencyMs)
                concurrent ane streams=\(aSample.streamCount) median_ms_token=\(aSample.medianMsPerToken) aggregate_tps=\(aSample.aggregateTokensPerSecond) per_stream_tps=\(aSample.perStreamTokensPerSecond) compile=\(aSample.compileTimeMs) round_ms=\(aSample.medianRoundLatencyMs)
                concurrent coreml streams=\(cSample.streamCount) median_ms_token=\(cSample.medianMsPerToken) aggregate_tps=\(cSample.aggregateTokensPerSecond) per_stream_tps=\(cSample.perStreamTokensPerSecond) compile=\(cSample.compileTimeMs) round_ms=\(cSample.medianRoundLatencyMs)
                """
            )
        }
    }

    private func benchmarkBatchedRecurrentGeneration(
        layerCount: Int,
        promptTokens: [UInt16],
        maxNewTokens: Int,
        warmup: Int,
        iterations: Int,
        streamCounts: [Int],
        groups: Int = 1
    ) throws -> ConcurrentGenerationScalingReport {
        let dim = ModelConfig.dim
        var samples: [ConcurrentGenerationScalingSample] = []
        samples.reserveCapacity(streamCounts.count)

        for streamCount in streamCounts {
            // laneSpatial must be a power of 2 >= 32 (ANE constraint)
            let laneSpatial: Int
            if streamCount <= 32 { laneSpatial = 32 }
            else if streamCount <= 64 { laneSpatial = 64 }
            else if streamCount <= 128 { laneSpatial = 128 }
            else if streamCount <= 256 { laneSpatial = 256 }
            else { laneSpatial = 512 }

            let compileStart = GenerationClock.now()

            let weights = makeEchoRecurrentGenerationWeights(layerCount: layerCount)
            let tripletCount = layerCount / 3

            var tripletSessions = try LayerStorage<RWKVStyleFusedThreeLayerSession>(
                count: tripletCount,
                throwingInitializer: { tripletIdx in
                    let base = tripletIdx * 3
                    return try RWKVStyleFusedThreeLayerSession(
                        weights0: weights.layers[base],
                        weights1: weights.layers[base + 1],
                        weights2: weights.layers[base + 2],
                        laneSpatial: laneSpatial,
                        groups: groups
                    )
                }
            )

            // Head kernel — use factored classifier for lower weight-loading latency
            let factoredHead = try FactoredGenerationRMSNormClassifierKernelSet(
                rmsFinal: weights.rmsFinal,
                classifierProjection: TensorBuffer(count: 128 * ModelConfig.dim, zeroed: true),
                classifierExpansion: TensorBuffer(count: weights.vocabSize * 128, zeroed: true),
                vocabSize: weights.vocabSize,
                bottleneck: 128,
                laneSpatial: laneSpatial
            )
            let headOutputSurface = try factoredHead.rmsNormClassifier.outputSurface(at: 0)

            let vocabSize = weights.vocabSize

            // === ZERO-COPY REBINDING ===
            // Rebind T0 state inputs → T0 state outputs (in-place state)
            let t0sOut0 = tripletSessions[0].handles.stateOut0
            let t0sOut1 = tripletSessions[0].handles.stateOut1
            let t0sOut2 = tripletSessions[0].handles.stateOut2
            try tripletSessions[0].kernels.step.rebindInput(at: 1, to: t0sOut0)
            try tripletSessions[0].kernels.step.rebindInput(at: 2, to: t0sOut1)
            try tripletSessions[0].kernels.step.rebindInput(at: 3, to: t0sOut2)

            if tripletCount > 1 {
                // Rebind T1 xIn → T0 xOut (zero-copy transfer)
                let t0xOut = tripletSessions[0].handles.xOut
                try tripletSessions[1].kernels.step.rebindInput(at: 0, to: t0xOut)

                // Rebind T1 state inputs → T1 state outputs (in-place state)
                let t1sOut0 = tripletSessions[1].handles.stateOut0
                let t1sOut1 = tripletSessions[1].handles.stateOut1
                let t1sOut2 = tripletSessions[1].handles.stateOut2
                try tripletSessions[1].kernels.step.rebindInput(at: 1, to: t1sOut0)
                try tripletSessions[1].kernels.step.rebindInput(at: 2, to: t1sOut1)
                try tripletSessions[1].kernels.step.rebindInput(at: 3, to: t1sOut2)

                // Rebind head input → T1 xOut (zero-copy transfer)
                let t1xOut = tripletSessions[1].handles.xOut
                try factoredHead.rmsNormClassifier.rebindInput(at: 0, to: t1xOut)
            } else {
                let t0xOut = tripletSessions[0].handles.xOut
                try factoredHead.rmsNormClassifier.rebindInput(at: 0, to: t0xOut)
            }

            let t0xIn = tripletSessions[0].handles.xIn
            let headOut = headOutputSurface
            let compileTimeMs = machMilliseconds(GenerationClock.now() - compileStart)

            // Zero-copy reset: zero the shared state surfaces directly
            func resetZeroCopy() throws {
                let zeroLane = tripletSessions[0].handles.zeroLane
                try SurfaceIO.copyFP16(dst: t0sOut0, dstChannelOffset: 0, src: zeroLane, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
                try SurfaceIO.copyFP16(dst: t0sOut1, dstChannelOffset: 0, src: zeroLane, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
                try SurfaceIO.copyFP16(dst: t0sOut2, dstChannelOffset: 0, src: zeroLane, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
                if tripletCount > 1 {
                    let zl1 = tripletSessions[1].handles.zeroLane
                    let s10 = tripletSessions[1].handles.stateOut0
                    let s11 = tripletSessions[1].handles.stateOut1
                    let s12 = tripletSessions[1].handles.stateOut2
                    try SurfaceIO.copyFP16(dst: s10, dstChannelOffset: 0, src: zl1, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
                    try SurfaceIO.copyFP16(dst: s11, dstChannelOffset: 0, src: zl1, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
                    try SurfaceIO.copyFP16(dst: s12, dstChannelOffset: 0, src: zl1, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
                }
            }

            // Warmup
            for _ in 0..<warmup {
                try resetZeroCopy()
                var tokens = Array(repeating: promptTokens[0], count: streamCount)
                for _ in 0..<maxNewTokens {
                    try batchedTokenStepZeroCopy(
                        tokens: &tokens,
                        streamCount: streamCount,
                        embedding: weights.embedding,
                        dim: dim,
                        laneSpatial: laneSpatial,
                        tripletCount: tripletCount,
                        tripletSessions: &tripletSessions,
                        t0xIn: t0xIn,
                        headOut: headOut,
                        headEval: { try factoredHead.rmsNormClassifier.eval() },
                        vocabSize: vocabSize
                    )
                }
            }

            // Timed iterations
            var roundLatenciesMs: [Double] = []
            roundLatenciesMs.reserveCapacity(iterations)
            for _ in 0..<iterations {
                try resetZeroCopy()
                var tokens = Array(repeating: promptTokens[0], count: streamCount)

                let start = GenerationClock.now()
                for _ in 0..<maxNewTokens {
                    try batchedTokenStepZeroCopy(
                        tokens: &tokens,
                        streamCount: streamCount,
                        embedding: weights.embedding,
                        dim: dim,
                        laneSpatial: laneSpatial,
                        tripletCount: tripletCount,
                        tripletSessions: &tripletSessions,
                        t0xIn: t0xIn,
                        headOut: headOut,
                        headEval: { try factoredHead.rmsNormClassifier.eval() },
                        vocabSize: vocabSize
                    )
                }
                roundLatenciesMs.append(machMilliseconds(GenerationClock.now() - start))
            }

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
            label: "ANE Batched Recurrent",
            promptLength: promptTokens.count,
            maxNewTokens: maxNewTokens,
            warmupCount: warmup,
            iterationCount: iterations,
            samples: samples
        )
    }

    private func batchedTokenStep(
        tokens: inout [UInt16],
        streamCount: Int,
        embedding: borrowing TensorBuffer,
        dim: Int,
        laneSpatial: Int,
        tripletCount: Int,
        tripletSessions: inout LayerStorage<RWKVStyleFusedThreeLayerSession>,
        t0xIn: IOSurfaceRef, t0xOut: IOSurfaceRef,
        t0sIn0: IOSurfaceRef, t0sIn1: IOSurfaceRef, t0sIn2: IOSurfaceRef,
        t0sOut0: IOSurfaceRef, t0sOut1: IOSurfaceRef, t0sOut2: IOSurfaceRef,
        t1xIn: IOSurfaceRef, t1xOut: IOSurfaceRef,
        t1sIn0: IOSurfaceRef, t1sIn1: IOSurfaceRef, t1sIn2: IOSurfaceRef,
        t1sOut0: IOSurfaceRef, t1sOut1: IOSurfaceRef, t1sOut2: IOSurfaceRef,
        headIn: IOSurfaceRef, headOut: IOSurfaceRef,
        head: ANEGenerationRMSNormClassifierHead,
        vocabSize: Int
    ) throws {
        try batchedTokenStepWithEval(
            tokens: &tokens, streamCount: streamCount,
            embedding: embedding, dim: dim, laneSpatial: laneSpatial,
            tripletCount: tripletCount, tripletSessions: &tripletSessions,
            t0xIn: t0xIn, t0xOut: t0xOut,
            t0sIn0: t0sIn0, t0sIn1: t0sIn1, t0sIn2: t0sIn2,
            t0sOut0: t0sOut0, t0sOut1: t0sOut1, t0sOut2: t0sOut2,
            t1xIn: t1xIn, t1xOut: t1xOut,
            t1sIn0: t1sIn0, t1sIn1: t1sIn1, t1sIn2: t1sIn2,
            t1sOut0: t1sOut0, t1sOut1: t1sOut1, t1sOut2: t1sOut2,
            headIn: headIn, headOut: headOut,
            headEval: { try head.kernelSet.rmsNormClassifier.eval() },
            vocabSize: vocabSize
        )
    }

    private func batchedTokenStepWithEval(
        tokens: inout [UInt16],
        streamCount: Int,
        embedding: borrowing TensorBuffer,
        dim: Int,
        laneSpatial: Int,
        tripletCount: Int,
        tripletSessions: inout LayerStorage<RWKVStyleFusedThreeLayerSession>,
        t0xIn: IOSurfaceRef, t0xOut: IOSurfaceRef,
        t0sIn0: IOSurfaceRef, t0sIn1: IOSurfaceRef, t0sIn2: IOSurfaceRef,
        t0sOut0: IOSurfaceRef, t0sOut1: IOSurfaceRef, t0sOut2: IOSurfaceRef,
        t1xIn: IOSurfaceRef, t1xOut: IOSurfaceRef,
        t1sIn0: IOSurfaceRef, t1sIn1: IOSurfaceRef, t1sIn2: IOSurfaceRef,
        t1sOut0: IOSurfaceRef, t1sOut1: IOSurfaceRef, t1sOut2: IOSurfaceRef,
        headIn: IOSurfaceRef, headOut: IOSurfaceRef,
        headEval: () throws -> Void,
        vocabSize: Int
    ) throws {
        try embedding.withUnsafePointer { embPtr in
            try tokens.withUnsafeBufferPointer { tokenBuf in
                try SurfaceIO.writeEmbeddingBatchFP16(
                    to: t0xIn, channelOffset: 0, spatial: laneSpatial,
                    embeddingTable: embPtr, dim: dim,
                    tokenIDs: tokenBuf.baseAddress!, streamCount: streamCount
                )
            }
        }

        try tripletSessions[0].kernels.step.eval()

        try SurfaceIO.copyFP16(dst: t0sIn0, dstChannelOffset: 0, src: t0sOut0, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
        try SurfaceIO.copyFP16(dst: t0sIn1, dstChannelOffset: 0, src: t0sOut1, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
        try SurfaceIO.copyFP16(dst: t0sIn2, dstChannelOffset: 0, src: t0sOut2, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)

        if tripletCount > 1 {
            try SurfaceIO.copyFP16(dst: t1xIn, dstChannelOffset: 0, src: t0xOut, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            try tripletSessions[1].kernels.step.eval()
            try SurfaceIO.copyFP16(dst: t1sIn0, dstChannelOffset: 0, src: t1sOut0, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            try SurfaceIO.copyFP16(dst: t1sIn1, dstChannelOffset: 0, src: t1sOut1, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            try SurfaceIO.copyFP16(dst: t1sIn2, dstChannelOffset: 0, src: t1sOut2, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
        }

        let lastXOut = tripletCount > 1 ? t1xOut : t0xOut
        try SurfaceIO.copyFP16(dst: headIn, dstChannelOffset: 0, src: lastXOut, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)

        try headEval()

        let argmaxResults = try SurfaceIO.argmaxBatchFP16SpatialParallel(
            from: headOut, channelOffset: 0, spatial: laneSpatial,
            channels: vocabSize, streamCount: streamCount, nBlocks: 8
        )
        for streamIdx in 0..<streamCount {
            tokens[streamIdx] = UInt16(argmaxResults[streamIdx].index)
        }
    }

    func test_fused_trunk_head_compile_probe_on_hardware() throws {
        try requireGenerationHardware()

        let laneSpatial = 256
        let dim = ModelConfig.dim
        let weights = makeEchoRecurrentGenerationWeights(layerCount: 6)
        let k = 128
        let vocabSize = weights.vocabSize

        // Build MIL for N-layer trunk + factored head, and try compile+eval
        func probeNLayerFusedHead(layerCount: Int) {
            let invd: Float = 1.0 / Float(dim)
            var b = MILBuilder(reserveCapacity: 32_768)
            b.append(MILText.header)

            var inputs = "tensor<fp16, [1, \(dim), 1, \(laneSpatial)]> x"
            for i in 0..<layerCount {
                inputs += ", tensor<fp16, [1, \(dim), 1, \(laneSpatial)]> stateIn\(i)"
            }
            b.appendLine("    func main<ios18>(\(inputs)) {")

            // Shared constants
            b.appendLine("        tensor<int32, [1]> raxCh = const()[name=string(\"rax_ch\"), val=tensor<int32, [1]>([1])];")
            b.appendLine("        bool kd = const()[name=string(\"kd\"), val=bool(true)];")
            b.append("        fp16 invd = const()[name=string(\"invd\"), val=fp16(")
            b.appendFP16(invd)
            b.appendLine(")];")
            b.appendLine("        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];")
            b.appendLine("        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];")
            b.append(MILText.convConst)

            // Recurrent layers
            for i in 0..<layerCount {
                let p = "l\(i)_"
                let xIn = i == 0 ? "x" : "l\(i-1)_xNext"
                let xOut = i == layerCount - 1 ? "xNext" : "l\(i)_xNext"
                let sIn = "stateIn\(i)"
                let sOut = "stateOut\(i)"

                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(p)sq = mul(x=\(xIn),y=\(xIn))[name=string(\"\(p)sq\")];")
                b.appendLine("        tensor<fp16, [1,1,1,\(laneSpatial)]> \(p)ss = reduce_sum(x=\(p)sq,axes=raxCh,keep_dims=kd)[name=string(\"\(p)ss\")];")
                b.appendLine("        tensor<fp16, [1,1,1,\(laneSpatial)]> \(p)ss2 = mul(x=\(p)ss,y=invd)[name=string(\"\(p)ss2\")];")
                b.appendLine("        tensor<fp16, [1,1,1,\(laneSpatial)]> \(p)ss3 = add(x=\(p)ss2,y=eps)[name=string(\"\(p)ss3\")];")
                b.appendLine("        tensor<fp16, [1,1,1,\(laneSpatial)]> \(p)rrms = pow(x=\(p)ss3,y=nhalf)[name=string(\"\(p)rrms\")];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(p)xr = mul(x=\(xIn),y=\(p)rrms)[name=string(\"\(p)xr\")];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,1]> \(p)rw = const()[name=string(\"\(p)rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rwkv_rms\(i).bin\"), offset=uint64(64)))];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(p)xn = mul(x=\(p)xr,y=\(p)rw)[name=string(\"\(p)xn\")];")
                b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> \(p)Wx = const()[name=string(\"\(p)Wx\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wx\(i).bin\"), offset=uint64(64)))];")
                b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> \(p)Ws = const()[name=string(\"\(p)Ws\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/ws\(i).bin\"), offset=uint64(64)))];")
                b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> \(p)Wd = const()[name=string(\"\(p)Wd\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wd\(i).bin\"), offset=uint64(64)))];")
                b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> \(p)Wo = const()[name=string(\"\(p)Wo\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo\(i).bin\"), offset=uint64(64)))];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(p)xMix = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(p)Wx,x=\(p)xn)[name=string(\"\(p)x_mix\")];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(p)sMix = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(p)Ws,x=\(sIn))[name=string(\"\(p)s_mix\")];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(p)carry = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(p)Wd,x=\(sIn))[name=string(\"\(p)carry\")];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(p)mixPre = add(x=\(p)xMix,y=\(p)sMix)[name=string(\"\(p)mix_pre\")];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(p)gate = sigmoid(x=\(p)mixPre)[name=string(\"\(p)gate\")];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(p)gatedCarry = mul(x=\(p)carry,y=\(p)gate)[name=string(\"\(p)gated_carry\")];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(sOut) = add(x=\(p)xMix,y=\(p)gatedCarry)[name=string(\"\(p)state_out\")];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(p)proj = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(p)Wo,x=\(sOut))[name=string(\"\(p)proj\")];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(xOut) = add(x=\(xIn),y=\(p)proj)[name=string(\"\(p)x_next\")];")
            }

            // Factored head on xNext
            b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> h_sq = mul(x=xNext,y=xNext)[name=string(\"h_sq\")];")
            b.appendLine("        tensor<fp16, [1,1,1,\(laneSpatial)]> h_ss = reduce_sum(x=h_sq,axes=raxCh,keep_dims=kd)[name=string(\"h_ss\")];")
            b.appendLine("        tensor<fp16, [1,1,1,\(laneSpatial)]> h_ss2 = mul(x=h_ss,y=invd)[name=string(\"h_ss2\")];")
            b.appendLine("        tensor<fp16, [1,1,1,\(laneSpatial)]> h_ss3 = add(x=h_ss2,y=eps)[name=string(\"h_ss3\")];")
            b.appendLine("        tensor<fp16, [1,1,1,\(laneSpatial)]> h_rrms = pow(x=h_ss3,y=nhalf)[name=string(\"h_rrms\")];")
            b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> h_xr = mul(x=xNext,y=h_rrms)[name=string(\"h_xr\")];")
            b.appendLine("        tensor<fp16, [1,\(dim),1,1]> h_rw = const()[name=string(\"h_rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_final.bin\"), offset=uint64(64)))];")
            b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> h_xn = mul(x=h_xr,y=h_rw)[name=string(\"h_xn\")];")
            b.appendLine("        tensor<fp16, [\(k), \(dim), 1, 1]> Wproj = const()[name=string(\"Wproj\"), val=tensor<fp16, [\(k), \(dim), 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/cls_proj.bin\"), offset=uint64(64)))];")
            b.appendLine("        tensor<fp16, [1, \(k), 1, \(laneSpatial)]> h_proj = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wproj,x=h_xn)[name=string(\"h_proj\")];")
            b.appendLine("        tensor<fp16, [\(vocabSize), \(k), 1, 1]> Wexp = const()[name=string(\"Wexp\"), val=tensor<fp16, [\(vocabSize), \(k), 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/cls_expand.bin\"), offset=uint64(64)))];")
            b.appendLine("        tensor<fp16, [1, \(vocabSize), 1, \(laneSpatial)]> logits = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wexp,x=h_proj)[name=string(\"h_expand\")];")

            var outputs = "xNext"
            for i in 0..<layerCount {
                outputs += ",stateOut\(i)"
            }
            outputs += ",logits"
            b.appendLine("    } -> (\(outputs));")
            b.appendLine("}")

            // Build weight blobs
            var weightBlobs: [(path: String, data: Data)] = []
            for i in 0..<layerCount {
                let idx = i >= weights.layers.count ? 0 : i
                weightBlobs.append(("@model_path/weights/rwkv_rms\(i).bin", weights.layers[idx].rms.withUnsafeBufferPointer { WeightBlob.build(from: $0, rows: 1, cols: dim) }))
                weightBlobs.append(("@model_path/weights/wx\(i).bin", weights.layers[idx].Wx.withUnsafeBufferPointer { WeightBlob.build(from: $0, rows: dim, cols: dim) }))
                weightBlobs.append(("@model_path/weights/ws\(i).bin", weights.layers[idx].Ws.withUnsafeBufferPointer { WeightBlob.build(from: $0, rows: dim, cols: dim) }))
                weightBlobs.append(("@model_path/weights/wd\(i).bin", weights.layers[idx].Wd.withUnsafeBufferPointer { WeightBlob.build(from: $0, rows: dim, cols: dim) }))
                weightBlobs.append(("@model_path/weights/wo\(i).bin", weights.layers[idx].Wo.withUnsafeBufferPointer { WeightBlob.build(from: $0, rows: dim, cols: dim) }))
            }
            // Head weights
            weightBlobs.append(("@model_path/weights/rms_final.bin", weights.rmsFinal.withUnsafeBufferPointer { WeightBlob.build(from: $0, rows: 1, cols: dim) }))
            let projBuf = TensorBuffer(count: k * dim, zeroed: true)
            weightBlobs.append(("@model_path/weights/cls_proj.bin", projBuf.withUnsafeBufferPointer { WeightBlob.build(from: $0, rows: k, cols: dim) }))
            let expBuf = TensorBuffer(count: vocabSize * k, zeroed: true)
            weightBlobs.append(("@model_path/weights/cls_expand.bin", expBuf.withUnsafeBufferPointer { WeightBlob.build(from: $0, rows: vocabSize, cols: k) }))

            let stateBytes = dim * laneSpatial * 2
            let inputSizes = [stateBytes] + Array(repeating: stateBytes, count: layerCount)
            let outputSizes = [stateBytes] + Array(repeating: stateBytes, count: layerCount) + [vocabSize * laneSpatial * 2]

            let compileStart = GenerationClock.now()
            do {
                let kernel = try ANEKernel(
                    milText: b.text,
                    weights: weightBlobs,
                    inputSizes: inputSizes,
                    outputSizes: outputSizes
                )
                let compileMs = machMilliseconds(GenerationClock.now() - compileStart)

                // Try eval
                try kernel.eval()

                // Benchmark 20 evals
                var times: [Double] = []
                for _ in 0..<20 {
                    let t = GenerationClock.now()
                    try kernel.eval()
                    times.append(machMilliseconds(GenerationClock.now() - t))
                }
                times.sort()
                let median = times[times.count / 2]
                print("\(layerCount)L+head: COMPILE OK (\(String(format: "%.0f", compileMs))ms), EVAL OK, median=\(String(format: "%.3f", median))ms")
            } catch {
                let compileMs = machMilliseconds(GenerationClock.now() - compileStart)
                print("\(layerCount)L+head: FAILED after \(String(format: "%.0f", compileMs))ms — \(error)")
            }
        }

        // Also probe standalone N-layer trunks (no head)
        func probeNLayerTrunk(layerCount: Int) {
            let invd: Float = 1.0 / Float(dim)
            var b = MILBuilder(reserveCapacity: 32_768)
            b.append(MILText.header)

            var inputs = "tensor<fp16, [1, \(dim), 1, \(laneSpatial)]> x"
            for i in 0..<layerCount {
                inputs += ", tensor<fp16, [1, \(dim), 1, \(laneSpatial)]> stateIn\(i)"
            }
            b.appendLine("    func main<ios18>(\(inputs)) {")
            b.appendLine("        tensor<int32, [1]> raxCh = const()[name=string(\"rax_ch\"), val=tensor<int32, [1]>([1])];")
            b.appendLine("        bool kd = const()[name=string(\"kd\"), val=bool(true)];")
            b.append("        fp16 invd = const()[name=string(\"invd\"), val=fp16(")
            b.appendFP16(invd)
            b.appendLine(")];")
            b.appendLine("        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];")
            b.appendLine("        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];")
            b.append(MILText.convConst)

            for i in 0..<layerCount {
                let p = "l\(i)_"
                let xIn = i == 0 ? "x" : "l\(i-1)_xNext"
                let xOut = i == layerCount - 1 ? "xNext" : "l\(i)_xNext"
                let sIn = "stateIn\(i)"
                let sOut = "stateOut\(i)"

                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(p)sq = mul(x=\(xIn),y=\(xIn))[name=string(\"\(p)sq\")];")
                b.appendLine("        tensor<fp16, [1,1,1,\(laneSpatial)]> \(p)ss = reduce_sum(x=\(p)sq,axes=raxCh,keep_dims=kd)[name=string(\"\(p)ss\")];")
                b.appendLine("        tensor<fp16, [1,1,1,\(laneSpatial)]> \(p)ss2 = mul(x=\(p)ss,y=invd)[name=string(\"\(p)ss2\")];")
                b.appendLine("        tensor<fp16, [1,1,1,\(laneSpatial)]> \(p)ss3 = add(x=\(p)ss2,y=eps)[name=string(\"\(p)ss3\")];")
                b.appendLine("        tensor<fp16, [1,1,1,\(laneSpatial)]> \(p)rrms = pow(x=\(p)ss3,y=nhalf)[name=string(\"\(p)rrms\")];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(p)xr = mul(x=\(xIn),y=\(p)rrms)[name=string(\"\(p)xr\")];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,1]> \(p)rw = const()[name=string(\"\(p)rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rwkv_rms\(i).bin\"), offset=uint64(64)))];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(p)xn = mul(x=\(p)xr,y=\(p)rw)[name=string(\"\(p)xn\")];")
                b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> \(p)Wx = const()[name=string(\"\(p)Wx\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wx\(i).bin\"), offset=uint64(64)))];")
                b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> \(p)Ws = const()[name=string(\"\(p)Ws\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/ws\(i).bin\"), offset=uint64(64)))];")
                b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> \(p)Wd = const()[name=string(\"\(p)Wd\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wd\(i).bin\"), offset=uint64(64)))];")
                b.appendLine("        tensor<fp16, [\(dim),\(dim),1,1]> \(p)Wo = const()[name=string(\"\(p)Wo\"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo\(i).bin\"), offset=uint64(64)))];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(p)xMix = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(p)Wx,x=\(p)xn)[name=string(\"\(p)x_mix\")];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(p)sMix = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(p)Ws,x=\(sIn))[name=string(\"\(p)s_mix\")];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(p)carry = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(p)Wd,x=\(sIn))[name=string(\"\(p)carry\")];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(p)mixPre = add(x=\(p)xMix,y=\(p)sMix)[name=string(\"\(p)mix_pre\")];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(p)gate = sigmoid(x=\(p)mixPre)[name=string(\"\(p)gate\")];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(p)gatedCarry = mul(x=\(p)carry,y=\(p)gate)[name=string(\"\(p)gated_carry\")];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(sOut) = add(x=\(p)xMix,y=\(p)gatedCarry)[name=string(\"\(p)state_out\")];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(p)proj = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(p)Wo,x=\(sOut))[name=string(\"\(p)proj\")];")
                b.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> \(xOut) = add(x=\(xIn),y=\(p)proj)[name=string(\"\(p)x_next\")];")
            }

            var outputs = "xNext"
            for i in 0..<layerCount { outputs += ",stateOut\(i)" }
            b.appendLine("    } -> (\(outputs));")
            b.appendLine("}")

            var weightBlobs: [(path: String, data: Data)] = []
            for i in 0..<layerCount {
                let idx = i >= weights.layers.count ? 0 : i
                weightBlobs.append(("@model_path/weights/rwkv_rms\(i).bin", weights.layers[idx].rms.withUnsafeBufferPointer { WeightBlob.build(from: $0, rows: 1, cols: dim) }))
                weightBlobs.append(("@model_path/weights/wx\(i).bin", weights.layers[idx].Wx.withUnsafeBufferPointer { WeightBlob.build(from: $0, rows: dim, cols: dim) }))
                weightBlobs.append(("@model_path/weights/ws\(i).bin", weights.layers[idx].Ws.withUnsafeBufferPointer { WeightBlob.build(from: $0, rows: dim, cols: dim) }))
                weightBlobs.append(("@model_path/weights/wd\(i).bin", weights.layers[idx].Wd.withUnsafeBufferPointer { WeightBlob.build(from: $0, rows: dim, cols: dim) }))
                weightBlobs.append(("@model_path/weights/wo\(i).bin", weights.layers[idx].Wo.withUnsafeBufferPointer { WeightBlob.build(from: $0, rows: dim, cols: dim) }))
            }

            let stateBytes = dim * laneSpatial * 2
            let inputSizes = [stateBytes] + Array(repeating: stateBytes, count: layerCount)
            let outputSizes = [stateBytes] + Array(repeating: stateBytes, count: layerCount)

            let compileStart = GenerationClock.now()
            do {
                let kernel = try ANEKernel(
                    milText: b.text,
                    weights: weightBlobs,
                    inputSizes: inputSizes,
                    outputSizes: outputSizes
                )
                let compileMs = machMilliseconds(GenerationClock.now() - compileStart)

                try kernel.eval()

                var times: [Double] = []
                for _ in 0..<20 {
                    let t = GenerationClock.now()
                    try kernel.eval()
                    times.append(machMilliseconds(GenerationClock.now() - t))
                }
                times.sort()
                let median = times[times.count / 2]
                print("\(layerCount)L trunk: COMPILE OK (\(String(format: "%.0f", compileMs))ms), EVAL OK, median=\(String(format: "%.3f", median))ms")
            } catch {
                let compileMs = machMilliseconds(GenerationClock.now() - compileStart)
                print("\(layerCount)L trunk: FAILED after \(String(format: "%.0f", compileMs))ms — \(error)")
            }
        }

        // Probe matrix: trunk-only and fused trunk+head at different layer counts
        print("=== Trunk-only probes ===")
        probeNLayerTrunk(layerCount: 4)
        probeNLayerTrunk(layerCount: 5)

        print("=== Fused trunk+head probes ===")
        probeNLayerFusedHead(layerCount: 1)
        probeNLayerFusedHead(layerCount: 2)
        probeNLayerFusedHead(layerCount: 3)

        // Probe: factored head variants
        print("=== Factored head output variants ===")
        let blockSize = 125 // 32000 / 256 = 125
        let numBlocks = 256

        // Helper to build factored head MIL with configurable outputs
        func buildFactoredHeadMILProbe(outputLogits: Bool, outputBmax: Bool, outputProj: Bool) -> String {
            let invdVal: Float = 1.0 / Float(dim)
            var hb = MILBuilder(reserveCapacity: 4_096)
            hb.append(MILText.header)
            hb.appendLine("    func main<ios18>(tensor<fp16, [1, \(dim), 1, \(laneSpatial)]> x) {")
            hb.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> sq = mul(x=x,y=x)[name=string(\"sq\")];")
            hb.appendLine("        tensor<int32, [1]> raxCh = const()[name=string(\"rax_ch\"), val=tensor<int32, [1]>([1])];")
            hb.appendLine("        bool kd = const()[name=string(\"kd\"), val=bool(true)];")
            hb.append("        fp16 invd = const()[name=string(\"invd\"), val=fp16(")
            hb.appendFP16(invdVal)
            hb.appendLine(")];")
            hb.appendLine("        tensor<fp16, [1,1,1,\(laneSpatial)]> ss = reduce_sum(x=sq,axes=raxCh,keep_dims=kd)[name=string(\"ss\")];")
            hb.appendLine("        tensor<fp16, [1,1,1,\(laneSpatial)]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];")
            hb.appendLine("        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];")
            hb.appendLine("        tensor<fp16, [1,1,1,\(laneSpatial)]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];")
            hb.appendLine("        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];")
            hb.appendLine("        tensor<fp16, [1,1,1,\(laneSpatial)]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];")
            hb.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];")
            hb.appendLine("        tensor<fp16, [1,\(dim),1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_final.bin\"), offset=uint64(64)))];")
            hb.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];")
            hb.append(MILText.convConst)
            hb.appendLine("        tensor<fp16, [\(k), \(dim), 1, 1]> Wproj = const()[name=string(\"Wproj\"), val=tensor<fp16, [\(k), \(dim), 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/cls_proj.bin\"), offset=uint64(64)))];")
            hb.appendLine("        tensor<fp16, [1, \(k), 1, \(laneSpatial)]> proj = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wproj,x=xn)[name=string(\"proj\")];")
            hb.appendLine("        tensor<fp16, [\(vocabSize), \(k), 1, 1]> Wexp = const()[name=string(\"Wexp\"), val=tensor<fp16, [\(vocabSize), \(k), 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/cls_expand.bin\"), offset=uint64(64)))];")
            hb.appendLine("        tensor<fp16, [1, \(vocabSize), 1, \(laneSpatial)]> logits = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wexp,x=proj)[name=string(\"expand\")];")
            if outputBmax {
                hb.appendLine("        tensor<int32, [4]> rs = const()[name=string(\"rs\"), val=tensor<int32, [4]>([1, \(numBlocks), \(blockSize), \(laneSpatial)])];")
                hb.appendLine("        tensor<fp16, [1, \(numBlocks), \(blockSize), \(laneSpatial)]> lr = reshape(x=logits, shape=rs)[name=string(\"lr\")];")
                hb.appendLine("        tensor<int32, [1]> raxH = const()[name=string(\"rax_h\"), val=tensor<int32, [1]>([2])];")
                hb.appendLine("        tensor<fp16, [1, \(numBlocks), 1, \(laneSpatial)]> bmax = reduce_max(x=lr,axes=raxH,keep_dims=kd)[name=string(\"bmax\")];")
            }
            var outs: [String] = []
            if outputLogits { outs.append("logits") }
            if outputBmax { outs.append("bmax") }
            if outputProj { outs.append("proj") }
            hb.appendLine("    } -> (\(outs.joined(separator: ",")));")
            hb.appendLine("}")
            return hb.text
        }

        func probeHeadVariant(label: String, milText: String, outputSizes: [Int]) {
            let rmsBlob = weights.rmsFinal.withUnsafeBufferPointer { WeightBlob.build(from: $0, rows: 1, cols: dim) }
            let projBuf2 = TensorBuffer(count: k * dim, zeroed: true)
            let projBlob2 = projBuf2.withUnsafeBufferPointer { WeightBlob.build(from: $0, rows: k, cols: dim) }
            let expBuf2 = TensorBuffer(count: vocabSize * k, zeroed: true)
            let expBlob2 = expBuf2.withUnsafeBufferPointer { WeightBlob.build(from: $0, rows: vocabSize, cols: k) }

            let t = GenerationClock.now()
            do {
                let kernel = try ANEKernel(
                    milText: milText,
                    weights: [
                        (path: "@model_path/weights/rms_final.bin", data: rmsBlob),
                        (path: "@model_path/weights/cls_proj.bin", data: projBlob2),
                        (path: "@model_path/weights/cls_expand.bin", data: expBlob2),
                    ],
                    inputSizes: [dim * laneSpatial * 2],
                    outputSizes: outputSizes
                )
                let cMs = machMilliseconds(GenerationClock.now() - t)
                try kernel.eval()
                var times: [Double] = []
                for _ in 0..<20 {
                    let t0 = GenerationClock.now()
                    try kernel.eval()
                    times.append(machMilliseconds(GenerationClock.now() - t0))
                }
                times.sort()
                print("\(label): OK compile=\(String(format: "%.0f", cMs))ms eval_median=\(String(format: "%.3f", times[10]))ms")
            } catch {
                let cMs = machMilliseconds(GenerationClock.now() - t)
                print("\(label): FAILED after \(String(format: "%.0f", cMs))ms")
            }
        }

        // Variant A: logits only (baseline — should work)
        probeHeadVariant(
            label: "logits_only",
            milText: buildFactoredHeadMILProbe(outputLogits: true, outputBmax: false, outputProj: false),
            outputSizes: [vocabSize * laneSpatial * 2]
        )
        // Variant B: logits + bmax (two outputs)
        probeHeadVariant(
            label: "logits+bmax",
            milText: buildFactoredHeadMILProbe(outputLogits: true, outputBmax: true, outputProj: false),
            outputSizes: [vocabSize * laneSpatial * 2, numBlocks * laneSpatial * 2]
        )
        // Variant C: bmax only (no full logits)
        probeHeadVariant(
            label: "bmax_only",
            milText: buildFactoredHeadMILProbe(outputLogits: false, outputBmax: true, outputProj: false),
            outputSizes: [numBlocks * laneSpatial * 2]
        )
        // Variant D: proj + bmax (bottleneck + blocked max, no full logits)
        probeHeadVariant(
            label: "proj+bmax",
            milText: buildFactoredHeadMILProbe(outputLogits: false, outputBmax: true, outputProj: true),
            outputSizes: [numBlocks * laneSpatial * 2, k * laneSpatial * 2]
        )
        // Variant E: proj only (bottleneck only)
        probeHeadVariant(
            label: "proj_only",
            milText: buildFactoredHeadMILProbe(outputLogits: false, outputBmax: false, outputProj: true),
            outputSizes: [k * laneSpatial * 2]
        )

        // K-sweep at 256 lanes: measure head eval with different bottleneck sizes
        print("=== K-sweep at lane=\(laneSpatial) ===")
        for kVal in [64, 80, 96, 112, 128, 160, 192] {
            let kInvd: Float = 1.0 / Float(dim)
            var kb = MILBuilder(reserveCapacity: 4_096)
            kb.append(MILText.header)
            kb.appendLine("    func main<ios18>(tensor<fp16, [1, \(dim), 1, \(laneSpatial)]> x) {")
            kb.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> sq = mul(x=x,y=x)[name=string(\"sq\")];")
            kb.appendLine("        tensor<int32, [1]> raxCh = const()[name=string(\"rax_ch\"), val=tensor<int32, [1]>([1])];")
            kb.appendLine("        bool kd = const()[name=string(\"kd\"), val=bool(true)];")
            kb.append("        fp16 invd = const()[name=string(\"invd\"), val=fp16(")
            kb.appendFP16(kInvd)
            kb.appendLine(")];")
            kb.appendLine("        tensor<fp16, [1,1,1,\(laneSpatial)]> ss = reduce_sum(x=sq,axes=raxCh,keep_dims=kd)[name=string(\"ss\")];")
            kb.appendLine("        tensor<fp16, [1,1,1,\(laneSpatial)]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];")
            kb.appendLine("        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];")
            kb.appendLine("        tensor<fp16, [1,1,1,\(laneSpatial)]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];")
            kb.appendLine("        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];")
            kb.appendLine("        tensor<fp16, [1,1,1,\(laneSpatial)]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];")
            kb.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];")
            kb.appendLine("        tensor<fp16, [1,\(dim),1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_final.bin\"), offset=uint64(64)))];")
            kb.appendLine("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];")
            kb.append(MILText.convConst)
            kb.appendLine("        tensor<fp16, [\(kVal), \(dim), 1, 1]> Wproj = const()[name=string(\"Wproj\"), val=tensor<fp16, [\(kVal), \(dim), 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/cls_proj.bin\"), offset=uint64(64)))];")
            kb.appendLine("        tensor<fp16, [1, \(kVal), 1, \(laneSpatial)]> proj = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wproj,x=xn)[name=string(\"proj\")];")
            kb.appendLine("        tensor<fp16, [\(vocabSize), \(kVal), 1, 1]> Wexp = const()[name=string(\"Wexp\"), val=tensor<fp16, [\(vocabSize), \(kVal), 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/cls_expand.bin\"), offset=uint64(64)))];")
            kb.appendLine("        tensor<fp16, [1, \(vocabSize), 1, \(laneSpatial)]> logits = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wexp,x=proj)[name=string(\"expand\")];")
            kb.appendLine("    } -> (logits);")
            kb.appendLine("}")

            let rmsBlob2 = weights.rmsFinal.withUnsafeBufferPointer { WeightBlob.build(from: $0, rows: 1, cols: dim) }
            let projBuf3 = TensorBuffer(count: kVal * dim, zeroed: true)
            let projBlob3 = projBuf3.withUnsafeBufferPointer { WeightBlob.build(from: $0, rows: kVal, cols: dim) }
            let expBuf3 = TensorBuffer(count: vocabSize * kVal, zeroed: true)
            let expBlob3 = expBuf3.withUnsafeBufferPointer { WeightBlob.build(from: $0, rows: vocabSize, cols: kVal) }

            let ct = GenerationClock.now()
            do {
                let kernel = try ANEKernel(
                    milText: kb.text,
                    weights: [
                        (path: "@model_path/weights/rms_final.bin", data: rmsBlob2),
                        (path: "@model_path/weights/cls_proj.bin", data: projBlob3),
                        (path: "@model_path/weights/cls_expand.bin", data: expBlob3),
                    ],
                    inputSizes: [dim * laneSpatial * 2],
                    outputSizes: [vocabSize * laneSpatial * 2]
                )
                let cMs = machMilliseconds(GenerationClock.now() - ct)
                try kernel.eval()
                var times: [Double] = []
                for _ in 0..<20 {
                    let t0 = GenerationClock.now()
                    try kernel.eval()
                    times.append(machMilliseconds(GenerationClock.now() - t0))
                }
                times.sort()
                let wtMB = Double(kVal * dim + vocabSize * kVal) * 2.0 / 1_000_000.0
                print("k=\(kVal): eval_median=\(String(format: "%.3f", times[10]))ms compile=\(String(format: "%.0f", cMs))ms weights=\(String(format: "%.1f", wtMB))MB")
            } catch {
                print("k=\(kVal): FAILED — \(error)")
            }
        }
    }

    func test_batched_multistream_high_lane_sweep_on_hardware() throws {
        try requireGenerationHardware()

        let prompt: [UInt16] = [0]
        let warmup = 3
        let iterations = 20
        let maxNewTokens = 8
        let streamCounts = [32, 64, 128, 256, 512]

        let batched = try benchmarkBatchedRecurrentGeneration(
            layerCount: 6,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations,
            streamCounts: streamCounts,
            groups: 16
        )

        XCTAssertEqual(batched.samples.map(\.streamCount), streamCounts)
        XCTAssertTrue(batched.samples.allSatisfy { $0.medianMsPerToken > 0 })

        for sample in batched.samples {
            print(
                """
                batched ane streams=\(sample.streamCount) median_ms_token=\(sample.medianMsPerToken) aggregate_tps=\(sample.aggregateTokensPerSecond) per_stream_tps=\(sample.perStreamTokensPerSecond) compile=\(sample.compileTimeMs) round_ms=\(sample.medianRoundLatencyMs)
                """
            )
        }
    }

    /// Zero-copy step: no state copies, no xfer copy, no head copy.
    /// Requires surfaces to be pre-rebound so inputs alias outputs.
    private func batchedTokenStepZeroCopy(
        tokens: inout [UInt16],
        streamCount: Int,
        embedding: borrowing TensorBuffer,
        dim: Int,
        laneSpatial: Int,
        tripletCount: Int,
        tripletSessions: inout LayerStorage<RWKVStyleFusedThreeLayerSession>,
        t0xIn: IOSurfaceRef,
        headOut: IOSurfaceRef,
        headEval: () throws -> Void,
        vocabSize: Int
    ) throws {
        try embedding.withUnsafePointer { embPtr in
            try tokens.withUnsafeBufferPointer { tokenBuf in
                try SurfaceIO.writeEmbeddingBatchFP16(
                    to: t0xIn, channelOffset: 0, spatial: laneSpatial,
                    embeddingTable: embPtr, dim: dim,
                    tokenIDs: tokenBuf.baseAddress!, streamCount: streamCount
                )
            }
        }

        try tripletSessions[0].kernels.step.eval()

        if tripletCount > 1 {
            // T1.xIn is already rebound to T0.xOut — no copy needed
            try tripletSessions[1].kernels.step.eval()
        }

        // headIn is already rebound to lastXOut — no copy needed
        try headEval()

        let argmaxResults = try SurfaceIO.argmaxBatchFP16SpatialParallel(
            from: headOut, channelOffset: 0, spatial: laneSpatial,
            channels: vocabSize, streamCount: streamCount, nBlocks: 8
        )
        for streamIdx in 0..<streamCount {
            tokens[streamIdx] = UInt16(argmaxResults[streamIdx].index)
        }
    }

    // MARK: - Zero-copy correctness check

    func test_zerocopy_correctness_on_hardware() throws {
        try requireGenerationHardware()

        let laneSpatial = 512
        let dim = ModelConfig.dim
        let layerCount = 6
        let streamCount = 512
        let maxNewTokens = 8

        let weights = makeEchoRecurrentGenerationWeights(layerCount: layerCount)
        let tripletCount = layerCount / 3

        // === BASELINE: normal copies ===
        var baselineSessions = try LayerStorage<RWKVStyleFusedThreeLayerSession>(
            count: tripletCount,
            throwingInitializer: { tripletIdx in
                let base = tripletIdx * 3
                return try RWKVStyleFusedThreeLayerSession(
                    weights0: weights.layers[base],
                    weights1: weights.layers[base + 1],
                    weights2: weights.layers[base + 2],
                    laneSpatial: laneSpatial
                )
            }
        )
        let baselineHead = try FactoredGenerationRMSNormClassifierKernelSet(
            rmsFinal: weights.rmsFinal,
            classifierProjection: TensorBuffer(count: 128 * dim, zeroed: true),
            classifierExpansion: TensorBuffer(count: weights.vocabSize * 128, zeroed: true),
            vocabSize: weights.vocabSize, bottleneck: 128, laneSpatial: laneSpatial
        )
        var baselineTokens = Array(repeating: UInt16(0), count: streamCount)
        for tIdx in 0..<tripletCount { try baselineSessions[tIdx].reset() }
        for _ in 0..<maxNewTokens {
            try batchedTokenStepWithEval(
                tokens: &baselineTokens, streamCount: streamCount,
                embedding: weights.embedding, dim: dim, laneSpatial: laneSpatial,
                tripletCount: tripletCount, tripletSessions: &baselineSessions,
                t0xIn: baselineSessions[0].handles.xIn, t0xOut: baselineSessions[0].handles.xOut,
                t0sIn0: baselineSessions[0].handles.stateIn0, t0sIn1: baselineSessions[0].handles.stateIn1, t0sIn2: baselineSessions[0].handles.stateIn2,
                t0sOut0: baselineSessions[0].handles.stateOut0, t0sOut1: baselineSessions[0].handles.stateOut1, t0sOut2: baselineSessions[0].handles.stateOut2,
                t1xIn: baselineSessions[1].handles.xIn, t1xOut: baselineSessions[1].handles.xOut,
                t1sIn0: baselineSessions[1].handles.stateIn0, t1sIn1: baselineSessions[1].handles.stateIn1, t1sIn2: baselineSessions[1].handles.stateIn2,
                t1sOut0: baselineSessions[1].handles.stateOut0, t1sOut1: baselineSessions[1].handles.stateOut1, t1sOut2: baselineSessions[1].handles.stateOut2,
                headIn: try baselineHead.rmsNormClassifier.inputSurface(at: 0),
                headOut: try baselineHead.rmsNormClassifier.outputSurface(at: 0),
                headEval: { try baselineHead.rmsNormClassifier.eval() },
                vocabSize: weights.vocabSize
            )
        }

        // === ZERO-COPY: in-place rebind ===
        var zcSessions = try LayerStorage<RWKVStyleFusedThreeLayerSession>(
            count: tripletCount,
            throwingInitializer: { tripletIdx in
                let base = tripletIdx * 3
                return try RWKVStyleFusedThreeLayerSession(
                    weights0: weights.layers[base],
                    weights1: weights.layers[base + 1],
                    weights2: weights.layers[base + 2],
                    laneSpatial: laneSpatial
                )
            }
        )
        let zcHead = try FactoredGenerationRMSNormClassifierKernelSet(
            rmsFinal: weights.rmsFinal,
            classifierProjection: TensorBuffer(count: 128 * dim, zeroed: true),
            classifierExpansion: TensorBuffer(count: weights.vocabSize * 128, zeroed: true),
            vocabSize: weights.vocabSize, bottleneck: 128, laneSpatial: laneSpatial
        )

        // Rebind for zero-copy
        let t0sOut0 = zcSessions[0].handles.stateOut0
        let t0sOut1 = zcSessions[0].handles.stateOut1
        let t0sOut2 = zcSessions[0].handles.stateOut2
        try zcSessions[0].kernels.step.rebindInput(at: 1, to: t0sOut0)
        try zcSessions[0].kernels.step.rebindInput(at: 2, to: t0sOut1)
        try zcSessions[0].kernels.step.rebindInput(at: 3, to: t0sOut2)
        let t0xOut = zcSessions[0].handles.xOut
        try zcSessions[1].kernels.step.rebindInput(at: 0, to: t0xOut)
        let t1sOut0 = zcSessions[1].handles.stateOut0
        let t1sOut1 = zcSessions[1].handles.stateOut1
        let t1sOut2 = zcSessions[1].handles.stateOut2
        try zcSessions[1].kernels.step.rebindInput(at: 1, to: t1sOut0)
        try zcSessions[1].kernels.step.rebindInput(at: 2, to: t1sOut1)
        try zcSessions[1].kernels.step.rebindInput(at: 3, to: t1sOut2)
        let t1xOut = zcSessions[1].handles.xOut
        try zcHead.rmsNormClassifier.rebindInput(at: 0, to: t1xOut)

        // Zero the shared state surfaces
        let zl = zcSessions[0].handles.zeroLane
        try SurfaceIO.copyFP16(dst: t0sOut0, dstChannelOffset: 0, src: zl, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
        try SurfaceIO.copyFP16(dst: t0sOut1, dstChannelOffset: 0, src: zl, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
        try SurfaceIO.copyFP16(dst: t0sOut2, dstChannelOffset: 0, src: zl, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
        let zl1 = zcSessions[1].handles.zeroLane
        try SurfaceIO.copyFP16(dst: t1sOut0, dstChannelOffset: 0, src: zl1, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
        try SurfaceIO.copyFP16(dst: t1sOut1, dstChannelOffset: 0, src: zl1, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
        try SurfaceIO.copyFP16(dst: t1sOut2, dstChannelOffset: 0, src: zl1, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)

        var zcTokens = Array(repeating: UInt16(0), count: streamCount)
        let zcXIn = zcSessions[0].handles.xIn
        let zcHeadOut = try zcHead.rmsNormClassifier.outputSurface(at: 0)
        for _ in 0..<maxNewTokens {
            try batchedTokenStepZeroCopy(
                tokens: &zcTokens, streamCount: streamCount,
                embedding: weights.embedding, dim: dim, laneSpatial: laneSpatial,
                tripletCount: tripletCount, tripletSessions: &zcSessions,
                t0xIn: zcXIn, headOut: zcHeadOut,
                headEval: { try zcHead.rmsNormClassifier.eval() },
                vocabSize: weights.vocabSize
            )
        }

        // Compare tokens
        var mismatches = 0
        for i in 0..<streamCount {
            if baselineTokens[i] != zcTokens[i] { mismatches += 1 }
        }
        print("zerocopy_correctness: \(mismatches)/\(streamCount) mismatches")
        XCTAssertEqual(mismatches, 0, "Zero-copy tokens must match baseline tokens")
    }

    // MARK: - Zero-copy surface rebind benchmark

    func test_zerocopy_rebind_benchmark_on_hardware() throws {
        try requireGenerationHardware()

        let dim = ModelConfig.dim
        let layerCount = 6
        let warmup = 3
        let iterations = 20
        let maxNewTokens = 8
        let streamCounts = [32, 64, 128, 256, 512]
        let prompt: [UInt16] = [0]

        for streamCount in streamCounts {
            let laneSpatial: Int
            if streamCount <= 32 { laneSpatial = 32 }
            else if streamCount <= 64 { laneSpatial = 64 }
            else if streamCount <= 128 { laneSpatial = 128 }
            else if streamCount <= 256 { laneSpatial = 256 }
            else { laneSpatial = 512 }

            let compileStart = GenerationClock.now()
            let weights = makeEchoRecurrentGenerationWeights(layerCount: layerCount)
            let tripletCount = layerCount / 3

            var tripletSessions = try LayerStorage<RWKVStyleFusedThreeLayerSession>(
                count: tripletCount,
                throwingInitializer: { tripletIdx in
                    let base = tripletIdx * 3
                    return try RWKVStyleFusedThreeLayerSession(
                        weights0: weights.layers[base],
                        weights1: weights.layers[base + 1],
                        weights2: weights.layers[base + 2],
                        laneSpatial: laneSpatial
                    )
                }
            )

            let factoredHead = try FactoredGenerationRMSNormClassifierKernelSet(
                rmsFinal: weights.rmsFinal,
                classifierProjection: TensorBuffer(count: 128 * ModelConfig.dim, zeroed: true),
                classifierExpansion: TensorBuffer(count: weights.vocabSize * 128, zeroed: true),
                vocabSize: weights.vocabSize,
                bottleneck: 128,
                laneSpatial: laneSpatial
            )
            let headOut = try factoredHead.rmsNormClassifier.outputSurface(at: 0)
            let vocabSize = weights.vocabSize

            // === ZERO-COPY REBINDING ===
            // 1. Rebind T0 state inputs → T0 state outputs (in-place state)
            let t0sOut0 = tripletSessions[0].handles.stateOut0
            let t0sOut1 = tripletSessions[0].handles.stateOut1
            let t0sOut2 = tripletSessions[0].handles.stateOut2
            try tripletSessions[0].kernels.step.rebindInput(at: 1, to: t0sOut0)
            try tripletSessions[0].kernels.step.rebindInput(at: 2, to: t0sOut1)
            try tripletSessions[0].kernels.step.rebindInput(at: 3, to: t0sOut2)

            if tripletCount > 1 {
                // 2. Rebind T1 xIn → T0 xOut (zero-copy transfer)
                let t0xOut = tripletSessions[0].handles.xOut
                try tripletSessions[1].kernels.step.rebindInput(at: 0, to: t0xOut)

                // 3. Rebind T1 state inputs → T1 state outputs (in-place state)
                let t1sOut0 = tripletSessions[1].handles.stateOut0
                let t1sOut1 = tripletSessions[1].handles.stateOut1
                let t1sOut2 = tripletSessions[1].handles.stateOut2
                try tripletSessions[1].kernels.step.rebindInput(at: 1, to: t1sOut0)
                try tripletSessions[1].kernels.step.rebindInput(at: 2, to: t1sOut1)
                try tripletSessions[1].kernels.step.rebindInput(at: 3, to: t1sOut2)

                // 4. Rebind head input → T1 xOut (zero-copy transfer)
                let t1xOut = tripletSessions[1].handles.xOut
                try factoredHead.rmsNormClassifier.rebindInput(at: 0, to: t1xOut)
            } else {
                let t0xOut = tripletSessions[0].handles.xOut
                try factoredHead.rmsNormClassifier.rebindInput(at: 0, to: t0xOut)
            }

            let t0xIn = tripletSessions[0].handles.xIn
            let compileTimeMs = machMilliseconds(GenerationClock.now() - compileStart)

            // Zero surfaces for reset (state surfaces are now the output surfaces)
            func resetZeroCopy() throws {
                let zeroLane = tripletSessions[0].handles.zeroLane
                try SurfaceIO.copyFP16(dst: t0sOut0, dstChannelOffset: 0, src: zeroLane, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
                try SurfaceIO.copyFP16(dst: t0sOut1, dstChannelOffset: 0, src: zeroLane, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
                try SurfaceIO.copyFP16(dst: t0sOut2, dstChannelOffset: 0, src: zeroLane, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
                if tripletCount > 1 {
                    let zl1 = tripletSessions[1].handles.zeroLane
                    let s10 = tripletSessions[1].handles.stateOut0
                    let s11 = tripletSessions[1].handles.stateOut1
                    let s12 = tripletSessions[1].handles.stateOut2
                    try SurfaceIO.copyFP16(dst: s10, dstChannelOffset: 0, src: zl1, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
                    try SurfaceIO.copyFP16(dst: s11, dstChannelOffset: 0, src: zl1, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
                    try SurfaceIO.copyFP16(dst: s12, dstChannelOffset: 0, src: zl1, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
                }
            }

            // Warmup
            for _ in 0..<warmup {
                try resetZeroCopy()
                var tokens = Array(repeating: prompt[0], count: streamCount)
                for _ in 0..<maxNewTokens {
                    try batchedTokenStepZeroCopy(
                        tokens: &tokens, streamCount: streamCount,
                        embedding: weights.embedding, dim: dim, laneSpatial: laneSpatial,
                        tripletCount: tripletCount, tripletSessions: &tripletSessions,
                        t0xIn: t0xIn, headOut: headOut,
                        headEval: { try factoredHead.rmsNormClassifier.eval() },
                        vocabSize: vocabSize
                    )
                }
            }

            // Timed iterations
            var roundLatenciesMs: [Double] = []
            roundLatenciesMs.reserveCapacity(iterations)
            for _ in 0..<iterations {
                try resetZeroCopy()
                var tokens = Array(repeating: prompt[0], count: streamCount)
                let start = GenerationClock.now()
                for _ in 0..<maxNewTokens {
                    try batchedTokenStepZeroCopy(
                        tokens: &tokens, streamCount: streamCount,
                        embedding: weights.embedding, dim: dim, laneSpatial: laneSpatial,
                        tripletCount: tripletCount, tripletSessions: &tripletSessions,
                        t0xIn: t0xIn, headOut: headOut,
                        headEval: { try factoredHead.rmsNormClassifier.eval() },
                        vocabSize: vocabSize
                    )
                }
                let elapsed = machMilliseconds(GenerationClock.now() - start)
                roundLatenciesMs.append(elapsed / Double(maxNewTokens))
            }
            roundLatenciesMs.sort()
            let medianRound = roundLatenciesMs[roundLatenciesMs.count / 2]
            let medianPerToken = medianRound / Double(streamCount)
            let aggTps = Double(streamCount) / (medianRound / 1000.0)
            print("zerocopy ane streams=\(streamCount) median_ms_token=\(medianPerToken) aggregate_tps=\(aggTps) compile=\(compileTimeMs) round_ms=\(medianRound)")
        }
    }

    // MARK: - Parallel argmax probe

    func test_parallel_argmax_vs_serial_on_hardware() throws {
        try requireGenerationHardware()

        let laneSpatial = 256
        let streamCount = 256
        let vocabSize = ModelConfig.vocab  // 32000
        let iterations = 50
        let warmup = 10

        // Build a factored head to get a realistic logits surface
        let weights = makeEchoRecurrentGenerationWeights(layerCount: 6)
        let factoredHead = try FactoredGenerationRMSNormClassifierKernelSet(
            rmsFinal: weights.rmsFinal,
            classifierProjection: TensorBuffer(count: 128 * ModelConfig.dim, zeroed: true),
            classifierExpansion: TensorBuffer(count: vocabSize * 128, zeroed: true),
            vocabSize: vocabSize,
            bottleneck: 128,
            laneSpatial: laneSpatial
        )
        let headOut = try factoredHead.rmsNormClassifier.outputSurface(at: 0)
        // Eval head once to populate the surface
        try factoredHead.rmsNormClassifier.eval()

        // Warmup both paths
        for _ in 0..<warmup {
            let _ = try SurfaceIO.argmaxBatchFP16Spatial(
                from: headOut, channelOffset: 0, spatial: laneSpatial,
                channels: vocabSize, streamCount: streamCount
            )
            let _ = try SurfaceIO.argmaxBatchFP16SpatialParallel(
                from: headOut, channelOffset: 0, spatial: laneSpatial,
                channels: vocabSize, streamCount: streamCount, nBlocks: 2
            )
            let _ = try SurfaceIO.argmaxBatchFP16SpatialParallel(
                from: headOut, channelOffset: 0, spatial: laneSpatial,
                channels: vocabSize, streamCount: streamCount, nBlocks: 8
            )
        }

        // Serial argmax
        var serialUs: [Double] = []
        for _ in 0..<iterations {
            let t = GenerationClock.now()
            let _ = try SurfaceIO.argmaxBatchFP16Spatial(
                from: headOut, channelOffset: 0, spatial: laneSpatial,
                channels: vocabSize, streamCount: streamCount
            )
            serialUs.append(machMicroseconds(GenerationClock.now() - t))
        }

        // Parallel argmax n=2
        var par2Us: [Double] = []
        for _ in 0..<iterations {
            let t = GenerationClock.now()
            let _ = try SurfaceIO.argmaxBatchFP16SpatialParallel(
                from: headOut, channelOffset: 0, spatial: laneSpatial,
                channels: vocabSize, streamCount: streamCount, nBlocks: 2
            )
            par2Us.append(machMicroseconds(GenerationClock.now() - t))
        }

        // Parallel argmax n=4
        var par4Us: [Double] = []
        for _ in 0..<iterations {
            let t = GenerationClock.now()
            let _ = try SurfaceIO.argmaxBatchFP16SpatialParallel(
                from: headOut, channelOffset: 0, spatial: laneSpatial,
                channels: vocabSize, streamCount: streamCount, nBlocks: 8
            )
            par4Us.append(machMicroseconds(GenerationClock.now() - t))
        }

        // Parallel argmax n=8
        var par8Us: [Double] = []
        for _ in 0..<iterations {
            let t = GenerationClock.now()
            let _ = try SurfaceIO.argmaxBatchFP16SpatialParallel(
                from: headOut, channelOffset: 0, spatial: laneSpatial,
                channels: vocabSize, streamCount: streamCount, nBlocks: 8
            )
            par8Us.append(machMicroseconds(GenerationClock.now() - t))
        }

        // Parallel argmax n=16
        var par16Us: [Double] = []
        for _ in 0..<iterations {
            let t = GenerationClock.now()
            let _ = try SurfaceIO.argmaxBatchFP16SpatialParallel(
                from: headOut, channelOffset: 0, spatial: laneSpatial,
                channels: vocabSize, streamCount: streamCount, nBlocks: 16
            )
            par16Us.append(machMicroseconds(GenerationClock.now() - t))
        }

        // Correctness check
        let serialResult = try SurfaceIO.argmaxBatchFP16Spatial(
            from: headOut, channelOffset: 0, spatial: laneSpatial,
            channels: vocabSize, streamCount: streamCount
        )
        let par8Result = try SurfaceIO.argmaxBatchFP16SpatialParallel(
            from: headOut, channelOffset: 0, spatial: laneSpatial,
            channels: vocabSize, streamCount: streamCount, nBlocks: 8
        )
        let par16Result = try SurfaceIO.argmaxBatchFP16SpatialParallel(
            from: headOut, channelOffset: 0, spatial: laneSpatial,
            channels: vocabSize, streamCount: streamCount, nBlocks: 16
        )
        for i in 0..<streamCount {
            XCTAssertEqual(serialResult[i].index, par8Result[i].index,
                           "Parallel_8 argmax mismatch at lane \(i)")
            XCTAssertEqual(serialResult[i].index, par16Result[i].index,
                           "Parallel_16 argmax mismatch at lane \(i)")
        }

        func median(_ arr: [Double]) -> Double {
            let sorted = arr.sorted()
            return sorted[sorted.count / 2]
        }

        let sMedian = median(serialUs)
        let p2Median = median(par2Us)
        let p4Median = median(par4Us)
        let p8Median = median(par8Us)
        let p16Median = median(par16Us)
        print("argmax @\(streamCount) streams, \(vocabSize) vocab, \(iterations) iterations:")
        print("  serial:      \(String(format: "%.1f", sMedian)) µs")
        print("  parallel_2:  \(String(format: "%.1f", p2Median)) µs (\(String(format: "%+.1f", p2Median - sMedian)) µs, \(String(format: "%.1f", (sMedian - p2Median) / sMedian * 100))% faster)")
        print("  parallel_4:  \(String(format: "%.1f", p4Median)) µs (\(String(format: "%+.1f", p4Median - sMedian)) µs, \(String(format: "%.1f", (sMedian - p4Median) / sMedian * 100))% faster)")
        print("  parallel_8:  \(String(format: "%.1f", p8Median)) µs (\(String(format: "%+.1f", p8Median - sMedian)) µs, \(String(format: "%.1f", (sMedian - p8Median) / sMedian * 100))% faster)")
        print("  parallel_16: \(String(format: "%.1f", p16Median)) µs (\(String(format: "%+.1f", p16Median - sMedian)) µs, \(String(format: "%.1f", (sMedian - p16Median) / sMedian * 100))% faster)")
    }

    // MARK: - MIL op ANE compile probes (argmax alternatives)

    func test_ane_argmax_alternatives_compile_probe_on_hardware() throws {
        try requireGenerationHardware()

        let inCh = 256
        let spatial = 32

        // --- Probe 1: reduce_argmax (scalar axis variant) ---
        let milArgmax = """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
            func main<ios18>(tensor<fp16, [1, \(inCh), 1, \(spatial)]> x) {
                int32 ax = const()[name=string("ax"), val=int32(1)];
                bool kd = const()[name=string("kd"), val=bool(true)];
                tensor<int32, [1, 1, 1, \(spatial)]> idx = reduce_argmax(x=x, axis=ax, keep_dims=kd)[name=string("argmax")];
                string to_fp32 = const()[name=string("to_fp32"), val=string("fp32")];
                tensor<fp32, [1, 1, 1, \(spatial)]> idx_f = cast(dtype=to_fp32, x=idx)[name=string("cast_out")];
            } -> (idx_f);
        }
        """
        let argmaxResult = tryCompileAndEval(
            label: "reduce_argmax_scalar_axis",
            milText: milArgmax,
            inputBytes: inCh * spatial * 2,
            outputBytes: 1 * spatial * 4
        )
        print("probe reduce_argmax_scalar_axis: \(argmaxResult)")

        // --- Probe 2: topk(k=1) returning values only ---
        let milTopk = """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
            func main<ios18>(tensor<fp16, [1, \(inCh), 1, \(spatial)]> x) {
                int32 k = const()[name=string("k"), val=int32(1)];
                int32 ax = const()[name=string("ax"), val=int32(1)];
                bool asc = const()[name=string("asc"), val=bool(false)];
                tensor<fp16, [1, 1, 1, \(spatial)]> vals, tensor<int32, [1, 1, 1, \(spatial)]> indices = topk(x=x, k=k, axis=ax, ascending=asc)[name=string("topk")];
                string to_fp32 = const()[name=string("to_fp32"), val=string("fp32")];
                tensor<fp32, [1, 1, 1, \(spatial)]> idx_f = cast(dtype=to_fp32, x=indices)[name=string("cast_idx")];
            } -> (idx_f);
        }
        """
        let topkResult = tryCompileAndEval(
            label: "topk_k1",
            milText: milTopk,
            inputBytes: inCh * spatial * 2,
            outputBytes: 1 * spatial * 4
        )
        print("probe topk_k1: \(topkResult)")

        // --- Probe 3: reduce_max (confirmed in memory, verify with vocab-scale) ---
        let milReduceMax = """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
            func main<ios18>(tensor<fp16, [1, \(inCh), 1, \(spatial)]> x) {
                tensor<int32, [1]> ax = const()[name=string("ax"), val=tensor<int32, [1]>([1])];
                bool kd = const()[name=string("kd"), val=bool(true)];
                tensor<fp16, [1, 1, 1, \(spatial)]> mx = reduce_max(x=x, axes=ax, keep_dims=kd)[name=string("rmax")];
            } -> (mx);
        }
        """
        let reduceMaxResult = tryCompileAndEval(
            label: "reduce_max",
            milText: milReduceMax,
            inputBytes: inCh * spatial * 2,
            outputBytes: 1 * spatial * 2
        )
        print("probe reduce_max: \(reduceMaxResult)")

        // --- Probe 4: reduce_argmin (test if any reduce_arg* works) ---
        let milArgmin = """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
            func main<ios18>(tensor<fp16, [1, \(inCh), 1, \(spatial)]> x) {
                tensor<fp16, [1, \(inCh), 1, \(spatial)]> neg = mul(x=x, y=fp16(-1.0))[name=string("neg")];
                tensor<int32, [1]> ax = const()[name=string("ax"), val=tensor<int32, [1]>([1])];
                bool kd = const()[name=string("kd"), val=bool(true)];
                tensor<int32, [1, 1, 1, \(spatial)]> idx = reduce_argmin(x=neg, axis=ax, keep_dims=kd)[name=string("argmin")];
                string to_fp32 = const()[name=string("to_fp32"), val=string("fp32")];
                tensor<fp32, [1, 1, 1, \(spatial)]> idx_f = cast(dtype=to_fp32, x=idx)[name=string("cast_out")];
            } -> (idx_f);
        }
        """
        let argminResult = tryCompileAndEval(
            label: "reduce_argmin_via_neg",
            milText: milArgmin,
            inputBytes: inCh * spatial * 2,
            outputBytes: 1 * spatial * 4
        )
        print("probe reduce_argmin_via_neg: \(argminResult)")

        // --- Probe 5: Constructed argmax via reduce_max + equal + mul + reduce_sum ---
        let milConstructed = """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
            func main<ios18>(tensor<fp16, [1, \(inCh), 1, \(spatial)]> x) {
                tensor<int32, [1]> ax = const()[name=string("ax"), val=tensor<int32, [1]>([1])];
                bool kd = const()[name=string("kd"), val=bool(true)];
                tensor<fp16, [1, 1, 1, \(spatial)]> mx = reduce_max(x=x, axes=ax, keep_dims=kd)[name=string("rmax")];
                tensor<bool, [1, \(inCh), 1, \(spatial)]> eq = equal(x=x, y=mx)[name=string("eq")];
                string to_fp16 = const()[name=string("to_fp16"), val=string("fp16")];
                tensor<fp16, [1, \(inCh), 1, \(spatial)]> mask = cast(dtype=to_fp16, x=eq)[name=string("mask")];
                tensor<fp16, [1, \(inCh), 1, 1]> indices = const()[name=string("indices"), val=tensor<fp16, [1, \(inCh), 1, 1]>(BLOBFILE(path=string("@model_path/weights/indices.bin"), offset=uint64(64)))];
                tensor<fp16, [1, \(inCh), 1, \(spatial)]> masked = mul(x=mask, y=indices)[name=string("masked")];
                tensor<fp16, [1, 1, 1, \(spatial)]> result = reduce_max(x=masked, axes=ax, keep_dims=kd)[name=string("result")];
            } -> (result);
        }
        """
        // Build indices weight blob: [0, 1, 2, ..., inCh-1] as fp16
        var indexData = Data(count: 128 + inCh * 2)  // 128-byte header + fp16 values
        for i in 0..<inCh {
            let val = Float16(Float(i))
            var bits = val.bitPattern
            indexData.replaceSubrange((128 + i * 2)..<(128 + i * 2 + 2), with: Data(bytes: &bits, count: 2))
        }
        let constructedResult = tryCompileAndEvalWithWeights(
            label: "constructed_argmax",
            milText: milConstructed,
            weights: [("@model_path/weights/indices.bin", indexData)],
            inputBytes: inCh * spatial * 2,
            outputBytes: 1 * spatial * 2
        )
        print("probe constructed_argmax: \(constructedResult)")

        // --- Probe 6: Blocked reduce_max (reshape + reduce_max along H axis) ---
        // Reshape [1, 256, 1, 32] → [1, 4, 64, 32] → reduce_max(axis=2) → [1, 4, 1, 32]
        let blocks = 4
        let blockSize = inCh / blocks  // 64
        let milBlocked = """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
            func main<ios18>(tensor<fp16, [1, \(inCh), 1, \(spatial)]> x) {
                tensor<int32, [4]> shape = const()[name=string("shape"), val=tensor<int32, [4]>([1, \(blocks), \(blockSize), \(spatial)])];
                tensor<fp16, [1, \(blocks), \(blockSize), \(spatial)]> xr = reshape(x=x, shape=shape)[name=string("reshape")];
                tensor<int32, [1]> ax = const()[name=string("ax"), val=tensor<int32, [1]>([2])];
                bool kd = const()[name=string("kd"), val=bool(true)];
                tensor<fp16, [1, \(blocks), 1, \(spatial)]> bmax = reduce_max(x=xr, axes=ax, keep_dims=kd)[name=string("bmax")];
            } -> (bmax);
        }
        """
        let blockedResult = tryCompileAndEval(
            label: "blocked_reduce_max",
            milText: milBlocked,
            inputBytes: inCh * spatial * 2,
            outputBytes: blocks * spatial * 2
        )
        print("probe blocked_reduce_max: \(blockedResult)")

        // --- Probe 7: Full-vocab blocked reduce_max at realistic scale ---
        // Reshape [1, 32000, 1, 32] → [1, 250, 128, 32] → reduce_max(axis=2) → [1, 250, 1, 32]
        let vocabBlocks = 250
        let vocabBlockSize = 128
        let fullVocab = vocabBlocks * vocabBlockSize  // 32000
        let milFullBlocked = """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
            func main<ios18>(tensor<fp16, [1, \(fullVocab), 1, \(spatial)]> x) {
                tensor<int32, [4]> shape = const()[name=string("shape"), val=tensor<int32, [4]>([1, \(vocabBlocks), \(vocabBlockSize), \(spatial)])];
                tensor<fp16, [1, \(vocabBlocks), \(vocabBlockSize), \(spatial)]> xr = reshape(x=x, shape=shape)[name=string("reshape")];
                tensor<int32, [1]> ax = const()[name=string("ax"), val=tensor<int32, [1]>([2])];
                bool kd = const()[name=string("kd"), val=bool(true)];
                tensor<fp16, [1, \(vocabBlocks), 1, \(spatial)]> bmax = reduce_max(x=xr, axes=ax, keep_dims=kd)[name=string("bmax")];
            } -> (bmax);
        }
        """
        let fullBlockedResult = tryCompileAndEval(
            label: "full_vocab_blocked_reduce_max",
            milText: milFullBlocked,
            inputBytes: fullVocab * spatial * 2,
            outputBytes: vocabBlocks * spatial * 2
        )
        print("probe full_vocab_blocked_reduce_max: \(fullBlockedResult)")

        // --- Probe 8: Fused head (RMSNorm + classifier + reduce_max) with two outputs ---
        // This combines the head kernel with reduce_max to output both logits and max value
        let dim = ModelConfig.dim
        let vocab = 32000
        let invd: Float = 1.0 / Float(dim)
        let milFusedHead = """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
            func main<ios18>(tensor<fp16, [1, \(dim), 1, \(spatial)]> x) {
                tensor<fp16, [1,\(dim),1,\(spatial)]> sq = mul(x=x,y=x)[name=string("sq")];
                tensor<int32, [1]> raxCh = const()[name=string("rax_ch"), val=tensor<int32, [1]>([1])];
                bool kd = const()[name=string("kd"), val=bool(true)];
                tensor<fp16, [1,1,1,\(spatial)]> ss = reduce_sum(x=sq,axes=raxCh,keep_dims=kd)[name=string("ss")];
                fp16 invd = const()[name=string("invd"), val=fp16(\(String(format: "%.6f", invd)))];
                tensor<fp16, [1,1,1,\(spatial)]> ss2 = mul(x=ss,y=invd)[name=string("ss2")];
                fp16 eps = const()[name=string("eps"), val=fp16(0.00001)];
                tensor<fp16, [1,1,1,\(spatial)]> ss3 = add(x=ss2,y=eps)[name=string("ss3")];
                fp16 nhalf = const()[name=string("nhalf"), val=fp16(-0.5)];
                tensor<fp16, [1,1,1,\(spatial)]> rrms = pow(x=ss3,y=nhalf)[name=string("rrms")];
                tensor<fp16, [1,\(dim),1,\(spatial)]> xr = mul(x=x,y=rrms)[name=string("xr")];
                tensor<fp16, [1,\(dim),1,1]> rw = const()[name=string("rw"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string("@model_path/weights/rms_final.bin"), offset=uint64(64)))];
                tensor<fp16, [1,\(dim),1,\(spatial)]> xn = mul(x=xr,y=rw)[name=string("xn")];
                string pt = const()[name=string("pt"), val=string("valid")];
                tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
                tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
                tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
                int32 gr = const()[name=string("gr"), val=int32(1)];
                tensor<fp16, [\(vocab), \(dim), 1, 1]> Wcls = const()[name=string("Wcls"), val=tensor<fp16, [\(vocab), \(dim), 1, 1]>(BLOBFILE(path=string("@model_path/weights/classifier.bin"), offset=uint64(64)))];
                tensor<fp16, [1, \(vocab), 1, \(spatial)]> logits = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wcls,x=xn)[name=string("cls")];
                tensor<fp16, [1, 1, 1, \(spatial)]> maxval = reduce_max(x=logits,axes=raxCh,keep_dims=kd)[name=string("maxval")];
            } -> (logits, maxval);
        }
        """
        let rmsWeightSize = dim * 2 + 128  // fp16 + header
        let clsWeightSize = vocab * dim * 2 + 128
        var rmsData = Data(repeating: 0, count: rmsWeightSize)
        // Fill RMS weights with 1.0
        for i in 0..<dim {
            let val = Float16(1.0)
            var bits = val.bitPattern
            rmsData.replaceSubrange((128 + i * 2)..<(128 + i * 2 + 2), with: Data(bytes: &bits, count: 2))
        }
        var clsData = Data(repeating: 0, count: clsWeightSize)
        // Fill classifier with small random-ish values (just zeros is fine for compile test)
        let logitsOutBytes = vocab * spatial * 2
        let maxvalOutBytes = 1 * spatial * 2
        let fusedHeadResult = tryCompileAndEvalWithWeights(
            label: "fused_head_with_reduce_max",
            milText: milFusedHead,
            weights: [
                ("@model_path/weights/rms_final.bin", rmsData),
                ("@model_path/weights/classifier.bin", clsData)
            ],
            inputBytes: dim * spatial * 2,
            outputSizes: [logitsOutBytes, maxvalOutBytes]
        )
        print("probe fused_head_with_reduce_max: \(fusedHeadResult)")
    }

    // MARK: - Int8 weight quantization ANE compile probe

    func test_int8_quantized_weight_compile_probe_on_hardware() throws {
        try requireGenerationHardware()

        let inCh = 64
        let outCh = 64
        let spatial = 32

        // Helper: build an int8 weight blob (same header, 1 byte per element)
        func buildInt8Blob(count: Int) -> Data {
            let payloadBytes = count
            let total = 128 + payloadBytes
            var data = Data(count: total)
            data.withUnsafeMutableBytes { raw in
                let base = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
                base[0] = 1
                base[4] = 2
                base[64] = 0xEF
                base[65] = 0xBE
                base[66] = 0xAD
                base[67] = 0xDE
                base[68] = 1
                raw.storeBytes(of: UInt32(payloadBytes).littleEndian, toByteOffset: 72, as: UInt32.self)
                raw.storeBytes(of: UInt32(128).littleEndian, toByteOffset: 80, as: UInt32.self)
                // Fill with small int8 values
                for i in 0..<payloadBytes {
                    base[128 + i] = UInt8(bitPattern: Int8(i % 127))
                }
            }
            return data
        }

        // Helper: build fp16 scalar blob
        func buildFP16ScalarBlob(value: Float) -> Data {
            var data = Data(count: 128 + 2)
            data.withUnsafeMutableBytes { raw in
                let base = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
                base[0] = 1; base[4] = 2
                base[64] = 0xEF; base[65] = 0xBE; base[66] = 0xAD; base[67] = 0xDE; base[68] = 1
                raw.storeBytes(of: UInt32(2).littleEndian, toByteOffset: 72, as: UInt32.self)
                raw.storeBytes(of: UInt32(128).littleEndian, toByteOffset: 80, as: UInt32.self)
                let payload = raw.baseAddress!.advanced(by: 128).assumingMemoryBound(to: UInt16.self)
                payload[0] = Float16(value).bitPattern
            }
            return data
        }

        // Helper: build per-channel fp16 blob
        func buildFP16PerChannelBlob(count: Int, value: Float) -> Data {
            let payloadBytes = count * 2
            var data = Data(count: 128 + payloadBytes)
            data.withUnsafeMutableBytes { raw in
                let base = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
                base[0] = 1; base[4] = 2
                base[64] = 0xEF; base[65] = 0xBE; base[66] = 0xAD; base[67] = 0xDE; base[68] = 1
                raw.storeBytes(of: UInt32(payloadBytes).littleEndian, toByteOffset: 72, as: UInt32.self)
                raw.storeBytes(of: UInt32(128).littleEndian, toByteOffset: 80, as: UInt32.self)
                let payload = raw.baseAddress!.advanced(by: 128).assumingMemoryBound(to: UInt16.self)
                for i in 0..<count { payload[i] = Float16(value).bitPattern }
            }
            return data
        }

        // Helper: build per-channel int8 blob (zero points)
        func buildInt8PerChannelBlob(count: Int, value: Int8) -> Data {
            let payloadBytes = count
            var data = Data(count: 128 + payloadBytes)
            data.withUnsafeMutableBytes { raw in
                let base = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
                base[0] = 1; base[4] = 2
                base[64] = 0xEF; base[65] = 0xBE; base[66] = 0xAD; base[67] = 0xDE; base[68] = 1
                raw.storeBytes(of: UInt32(payloadBytes).littleEndian, toByteOffset: 72, as: UInt32.self)
                raw.storeBytes(of: UInt32(128).littleEndian, toByteOffset: 80, as: UInt32.self)
                for i in 0..<payloadBytes { base[128 + i] = UInt8(bitPattern: value) }
            }
            return data
        }

        let int8WeightBlob = buildInt8Blob(count: outCh * inCh)
        let scaleBlob = buildFP16PerChannelBlob(count: outCh, value: 0.01)
        let zpBlob = buildInt8PerChannelBlob(count: outCh, value: 0)

        // --- Probe A: constexpr_affine_dequantize with per-channel scale ---
        let milQuantA = """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
            func main<ios18>(tensor<fp16, [1, \(inCh), 1, \(spatial)]> x) {
                tensor<int8, [\(outCh), \(inCh), 1, 1]> Wq = const()[name=string("Wq"), val=tensor<int8, [\(outCh), \(inCh), 1, 1]>(BLOBFILE(path=string("@model_path/weights/wq.bin"), offset=uint64(128)))];
                tensor<fp16, [\(outCh), 1, 1, 1]> scale = const()[name=string("scale"), val=tensor<fp16, [\(outCh), 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/scale.bin"), offset=uint64(128)))];
                tensor<int8, [\(outCh), 1, 1, 1]> zp = const()[name=string("zp"), val=tensor<int8, [\(outCh), 1, 1, 1]>(BLOBFILE(path=string("@model_path/weights/zp.bin"), offset=uint64(128)))];
                tensor<fp16, [\(outCh), \(inCh), 1, 1]> W = constexpr_affine_dequantize(quantized_data=Wq, zero_point=zp, scale=scale, axis=int32(0))[name=string("W")];
                string pt = const()[name=string("pt"), val=string("valid")];
                tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
                tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
                tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
                int32 gr = const()[name=string("gr"), val=int32(1)];
                tensor<fp16, [1, \(outCh), 1, \(spatial)]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string("out")];
            } -> (out);
        }
        """
        let resultA = tryCompileAndEvalWithWeights(
            label: "constexpr_affine_dequantize_per_channel",
            milText: milQuantA,
            weights: [
                ("@model_path/weights/wq.bin", int8WeightBlob),
                ("@model_path/weights/scale.bin", scaleBlob),
                ("@model_path/weights/zp.bin", zpBlob)
            ],
            inputBytes: inCh * spatial * 2,
            outputBytes: outCh * spatial * 2
        )
        print("probe constexpr_affine_dequantize (per-channel): \(resultA)")

        // --- Probe B: constexpr_affine_dequantize with scalar scale ---
        let scalarScaleBlob = buildFP16ScalarBlob(value: 0.01)
        let scalarZpData = Data([0])  // single int8 zero
        var scalarZpBlob = Data(count: 128 + 1)
        scalarZpBlob.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            base[0] = 1; base[4] = 2
            base[64] = 0xEF; base[65] = 0xBE; base[66] = 0xAD; base[67] = 0xDE; base[68] = 1
            raw.storeBytes(of: UInt32(1).littleEndian, toByteOffset: 72, as: UInt32.self)
            raw.storeBytes(of: UInt32(128).littleEndian, toByteOffset: 80, as: UInt32.self)
            base[128] = 0
        }
        let milQuantB = """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
            func main<ios18>(tensor<fp16, [1, \(inCh), 1, \(spatial)]> x) {
                tensor<int8, [\(outCh), \(inCh), 1, 1]> Wq = const()[name=string("Wq"), val=tensor<int8, [\(outCh), \(inCh), 1, 1]>(BLOBFILE(path=string("@model_path/weights/wq.bin"), offset=uint64(128)))];
                fp16 scale = const()[name=string("scale"), val=fp16(0.01)];
                int8 zp = const()[name=string("zp"), val=int8(0)];
                tensor<fp16, [\(outCh), \(inCh), 1, 1]> W = constexpr_affine_dequantize(quantized_data=Wq, zero_point=zp, scale=scale, axis=int32(0))[name=string("W")];
                string pt = const()[name=string("pt"), val=string("valid")];
                tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
                tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
                tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
                int32 gr = const()[name=string("gr"), val=int32(1)];
                tensor<fp16, [1, \(outCh), 1, \(spatial)]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string("out")];
            } -> (out);
        }
        """
        let resultB = tryCompileAndEvalWithWeights(
            label: "constexpr_affine_dequantize_scalar",
            milText: milQuantB,
            weights: [
                ("@model_path/weights/wq.bin", int8WeightBlob)
            ],
            inputBytes: inCh * spatial * 2,
            outputBytes: outCh * spatial * 2
        )
        print("probe constexpr_affine_dequantize (scalar): \(resultB)")

        // --- Probe C: direct cast int8→fp16 then conv ---
        let milQuantC = """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
            func main<ios18>(tensor<fp16, [1, \(inCh), 1, \(spatial)]> x) {
                tensor<int8, [\(outCh), \(inCh), 1, 1]> Wq = const()[name=string("Wq"), val=tensor<int8, [\(outCh), \(inCh), 1, 1]>(BLOBFILE(path=string("@model_path/weights/wq.bin"), offset=uint64(128)))];
                string to_fp16 = const()[name=string("to_fp16"), val=string("fp16")];
                tensor<fp16, [\(outCh), \(inCh), 1, 1]> W = cast(dtype=to_fp16, x=Wq)[name=string("W")];
                string pt = const()[name=string("pt"), val=string("valid")];
                tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
                tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
                tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
                int32 gr = const()[name=string("gr"), val=int32(1)];
                tensor<fp16, [1, \(outCh), 1, \(spatial)]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string("out")];
            } -> (out);
        }
        """
        let resultC = tryCompileAndEvalWithWeights(
            label: "int8_cast_to_fp16_conv",
            milText: milQuantC,
            weights: [
                ("@model_path/weights/wq.bin", int8WeightBlob)
            ],
            inputBytes: inCh * spatial * 2,
            outputBytes: outCh * spatial * 2
        )
        print("probe int8_cast_to_fp16_conv: \(resultC)")

        // --- Probe D: constexpr_lut_to_dense (palette quantization) ---
        // This is another quantization approach used in CoreML
        // Skip for now if constexpr_affine_dequantize works

        // --- Probe E: uint8 instead of int8 ---
        let milQuantE = """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
            func main<ios18>(tensor<fp16, [1, \(inCh), 1, \(spatial)]> x) {
                tensor<uint8, [\(outCh), \(inCh), 1, 1]> Wq = const()[name=string("Wq"), val=tensor<uint8, [\(outCh), \(inCh), 1, 1]>(BLOBFILE(path=string("@model_path/weights/wq.bin"), offset=uint64(128)))];
                fp16 scale = const()[name=string("scale"), val=fp16(0.01)];
                uint8 zp = const()[name=string("zp"), val=uint8(128)];
                tensor<fp16, [\(outCh), \(inCh), 1, 1]> W = constexpr_affine_dequantize(quantized_data=Wq, zero_point=zp, scale=scale, axis=int32(0))[name=string("W")];
                string pt = const()[name=string("pt"), val=string("valid")];
                tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
                tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
                tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
                int32 gr = const()[name=string("gr"), val=int32(1)];
                tensor<fp16, [1, \(outCh), 1, \(spatial)]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string("out")];
            } -> (out);
        }
        """
        let resultE = tryCompileAndEvalWithWeights(
            label: "constexpr_affine_dequantize_uint8",
            milText: milQuantE,
            weights: [
                ("@model_path/weights/wq.bin", int8WeightBlob)
            ],
            inputBytes: inCh * spatial * 2,
            outputBytes: outCh * spatial * 2
        )
        print("probe constexpr_affine_dequantize (uint8): \(resultE)")
    }

    private enum ProbeResult: CustomStringConvertible {
        case compileFailed(String)
        case evalFailed(String)
        case evalSuccess
        var description: String {
            switch self {
            case .compileFailed(let e): return "COMPILE_FAILED: \(e)"
            case .evalFailed(let e): return "EVAL_FAILED: \(e)"
            case .evalSuccess: return "SUCCESS"
            }
        }
    }

    private func tryCompileAndEval(label: String, milText: String, inputBytes: Int, outputBytes: Int) -> ProbeResult {
        tryCompileAndEvalWithWeights(label: label, milText: milText, weights: [], inputBytes: inputBytes, outputBytes: outputBytes)
    }

    private func tryCompileAndEvalWithWeights(label: String, milText: String, weights: [(path: String, data: Data)], inputBytes: Int, outputBytes: Int) -> ProbeResult {
        tryCompileAndEvalWithWeights(label: label, milText: milText, weights: weights, inputBytes: inputBytes, outputSizes: [outputBytes])
    }

    private func tryCompileAndEvalWithWeights(label: String, milText: String, weights: [(path: String, data: Data)], inputBytes: Int, outputSizes: [Int]) -> ProbeResult {
        do {
            let kernel = try ANEKernel(
                milText: milText,
                weights: weights.map { (path: $0.0, data: $0.1) },
                inputSizes: [inputBytes],
                outputSizes: outputSizes,
                checkBudget: false
            )
            do {
                try kernel.eval()
                return .evalSuccess
            } catch {
                return .evalFailed("\(error)")
            }
        } catch {
            return .compileFailed("\(error)")
        }
    }

    func test_batched_step_component_timing_on_hardware() throws {
        try requireGenerationHardware()

        let layerCount = 6
        let streamCount = 32
        let dim = ModelConfig.dim
        let laneSpatial = 32
        let iterations = 100

        let weights = makeEchoRecurrentGenerationWeights(layerCount: layerCount)
        let tripletCount = layerCount / 3

        var tripletSessions = try LayerStorage<RWKVStyleFusedThreeLayerSession>(
            count: tripletCount,
            throwingInitializer: { tripletIdx in
                let base = tripletIdx * 3
                return try RWKVStyleFusedThreeLayerSession(
                    weights0: weights.layers[base],
                    weights1: weights.layers[base + 1],
                    weights2: weights.layers[base + 2],
                    laneSpatial: laneSpatial
                )
            }
        )

        let head = try ANEGenerationRMSNormClassifierHead(
            rmsFinal: weights.rmsFinal,
            classifierWeights: weights.embedding,
            vocabSize: weights.vocabSize,
            laneSpatial: laneSpatial
        )

        let t0xIn = tripletSessions[0].handles.xIn
        let t0xOut = tripletSessions[0].handles.xOut
        let t0sIn0 = tripletSessions[0].handles.stateIn0
        let t0sIn1 = tripletSessions[0].handles.stateIn1
        let t0sIn2 = tripletSessions[0].handles.stateIn2
        let t0sOut0 = tripletSessions[0].handles.stateOut0
        let t0sOut1 = tripletSessions[0].handles.stateOut1
        let t0sOut2 = tripletSessions[0].handles.stateOut2
        let t1xIn = tripletSessions[1].handles.xIn
        let t1xOut = tripletSessions[1].handles.xOut
        let t1sIn0 = tripletSessions[1].handles.stateIn0
        let t1sIn1 = tripletSessions[1].handles.stateIn1
        let t1sIn2 = tripletSessions[1].handles.stateIn2
        let t1sOut0 = tripletSessions[1].handles.stateOut0
        let t1sOut1 = tripletSessions[1].handles.stateOut1
        let t1sOut2 = tripletSessions[1].handles.stateOut2
        let headIn = head.inputSurface
        let headOut = head.outputSurface
        let vocabSize = weights.vocabSize

        var tokens = Array(repeating: UInt16(0), count: streamCount)

        // Warmup
        for _ in 0..<5 {
            try batchedTokenStep(
                tokens: &tokens, streamCount: streamCount,
                embedding: weights.embedding, dim: dim, laneSpatial: laneSpatial,
                tripletCount: tripletCount, tripletSessions: &tripletSessions,
                t0xIn: t0xIn, t0xOut: t0xOut,
                t0sIn0: t0sIn0, t0sIn1: t0sIn1, t0sIn2: t0sIn2,
                t0sOut0: t0sOut0, t0sOut1: t0sOut1, t0sOut2: t0sOut2,
                t1xIn: t1xIn, t1xOut: t1xOut,
                t1sIn0: t1sIn0, t1sIn1: t1sIn1, t1sIn2: t1sIn2,
                t1sOut0: t1sOut0, t1sOut1: t1sOut1, t1sOut2: t1sOut2,
                headIn: headIn, headOut: headOut,
                head: head, vocabSize: vocabSize
            )
        }

        // Per-component timing
        var embedUs: [Double] = []
        var evalT0Us: [Double] = []
        var stateT0Us: [Double] = []
        var xferUs: [Double] = []
        var evalT1Us: [Double] = []
        var stateT1Us: [Double] = []
        var headCopyUs: [Double] = []
        var evalHeadUs: [Double] = []
        var argmaxUs: [Double] = []

        for _ in 0..<iterations {
            var t = GenerationClock.now()

            // Embed write
            try weights.embedding.withUnsafePointer { embPtr in
                try tokens.withUnsafeBufferPointer { tokenBuf in
                    try SurfaceIO.writeEmbeddingBatchFP16(
                        to: t0xIn, channelOffset: 0, spatial: laneSpatial,
                        embeddingTable: embPtr, dim: dim,
                        tokenIDs: tokenBuf.baseAddress!, streamCount: streamCount
                    )
                }
            }
            var t2 = GenerationClock.now(); embedUs.append(machMicroseconds(t2 - t)); t = t2

            // Eval T0
            try tripletSessions[0].kernels.step.eval()
            t2 = GenerationClock.now(); evalT0Us.append(machMicroseconds(t2 - t)); t = t2

            // State T0 copies
            try SurfaceIO.copyFP16(dst: t0sIn0, dstChannelOffset: 0, src: t0sOut0, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            try SurfaceIO.copyFP16(dst: t0sIn1, dstChannelOffset: 0, src: t0sOut1, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            try SurfaceIO.copyFP16(dst: t0sIn2, dstChannelOffset: 0, src: t0sOut2, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            t2 = GenerationClock.now(); stateT0Us.append(machMicroseconds(t2 - t)); t = t2

            // Inter-triplet transfer
            try SurfaceIO.copyFP16(dst: t1xIn, dstChannelOffset: 0, src: t0xOut, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            t2 = GenerationClock.now(); xferUs.append(machMicroseconds(t2 - t)); t = t2

            // Eval T1
            try tripletSessions[1].kernels.step.eval()
            t2 = GenerationClock.now(); evalT1Us.append(machMicroseconds(t2 - t)); t = t2

            // State T1 copies
            try SurfaceIO.copyFP16(dst: t1sIn0, dstChannelOffset: 0, src: t1sOut0, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            try SurfaceIO.copyFP16(dst: t1sIn1, dstChannelOffset: 0, src: t1sOut1, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            try SurfaceIO.copyFP16(dst: t1sIn2, dstChannelOffset: 0, src: t1sOut2, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            t2 = GenerationClock.now(); stateT1Us.append(machMicroseconds(t2 - t)); t = t2

            // Head copy
            try SurfaceIO.copyFP16(dst: headIn, dstChannelOffset: 0, src: t1xOut, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            t2 = GenerationClock.now(); headCopyUs.append(machMicroseconds(t2 - t)); t = t2

            // Eval head
            try head.kernelSet.rmsNormClassifier.eval()
            t2 = GenerationClock.now(); evalHeadUs.append(machMicroseconds(t2 - t)); t = t2

            // Argmax
            let argmaxResults = try SurfaceIO.argmaxBatchFP16Spatial(
                from: headOut, channelOffset: 0, spatial: laneSpatial,
                channels: vocabSize, streamCount: streamCount
            )
            for i in 0..<streamCount { tokens[i] = UInt16(argmaxResults[i].index) }
            t2 = GenerationClock.now(); argmaxUs.append(machMicroseconds(t2 - t))
        }

        func median(_ arr: [Double]) -> Double {
            let sorted = arr.sorted()
            let n = sorted.count
            return n % 2 == 0 ? (sorted[n/2 - 1] + sorted[n/2]) / 2.0 : sorted[n/2]
        }

        let totalUs = median(embedUs) + median(evalT0Us) + median(stateT0Us) + median(xferUs) +
            median(evalT1Us) + median(stateT1Us) + median(headCopyUs) + median(evalHeadUs) + median(argmaxUs)

        print("Component timing (median µs, \(streamCount) streams, \(iterations) iterations):")
        print("  embed_write: \(String(format: "%.1f", median(embedUs))) µs (\(String(format: "%.1f", median(embedUs) / totalUs * 100))%)")
        print("  eval_T0:     \(String(format: "%.1f", median(evalT0Us))) µs (\(String(format: "%.1f", median(evalT0Us) / totalUs * 100))%)")
        print("  state_T0:    \(String(format: "%.1f", median(stateT0Us))) µs (\(String(format: "%.1f", median(stateT0Us) / totalUs * 100))%)")
        print("  xfer_T0T1:   \(String(format: "%.1f", median(xferUs))) µs (\(String(format: "%.1f", median(xferUs) / totalUs * 100))%)")
        print("  eval_T1:     \(String(format: "%.1f", median(evalT1Us))) µs (\(String(format: "%.1f", median(evalT1Us) / totalUs * 100))%)")
        print("  state_T1:    \(String(format: "%.1f", median(stateT1Us))) µs (\(String(format: "%.1f", median(stateT1Us) / totalUs * 100))%)")
        print("  head_copy:   \(String(format: "%.1f", median(headCopyUs))) µs (\(String(format: "%.1f", median(headCopyUs) / totalUs * 100))%)")
        print("  eval_head:   \(String(format: "%.1f", median(evalHeadUs))) µs (\(String(format: "%.1f", median(evalHeadUs) / totalUs * 100))%)")
        print("  argmax:      \(String(format: "%.1f", median(argmaxUs))) µs (\(String(format: "%.1f", median(argmaxUs) / totalUs * 100))%)")
        print("  TOTAL:       \(String(format: "%.1f", totalUs)) µs")
    }

    func test_factored_head_eval_benchmark_on_hardware() throws {
        try requireGenerationHardware()

        let dim = ModelConfig.dim  // 768
        let vocab = 32000
        let laneSpatial = 256
        let warmup = 5
        let iterations = 30

        // Build proper weight blobs using WeightBlob
        let rmsWeights = [Float](repeating: 1.0, count: dim)
        let rmsBlob = WeightBlob.build(from: rmsWeights, rows: 1, cols: dim)

        let clsWeights = [Float](repeating: 0.0, count: vocab * dim)
        let clsBlob = WeightBlob.build(from: clsWeights, rows: vocab, cols: dim)

        // --- Build direct head kernel (existing) ---
        let directGen = GenerationRMSNormClassifierGenerator(vocabSize: vocab, laneSpatial: laneSpatial)
        let directKernel = try ANEKernel(
            milText: directGen.milText,
            weights: [
                (path: "@model_path/weights/rms_final.bin", data: rmsBlob),
                (path: "@model_path/weights/classifier.bin", data: clsBlob)
            ],
            inputBytes: directGen.inputBytes,
            outputBytes: vocab * laneSpatial * 2,
            checkBudget: false
        )

        // --- Build factored head kernel: [768→k→32000] ---
        let k = 128
        let factoredMIL = buildFactoredHeadMIL(dim: dim, vocab: vocab, lane: laneSpatial, bottleneck: k)

        let w1Weights = [Float](repeating: 0.0, count: k * dim)
        let w1Blob = WeightBlob.build(from: w1Weights, rows: k, cols: dim)

        let w2Weights = [Float](repeating: 0.0, count: vocab * k)
        let w2Blob = WeightBlob.build(from: w2Weights, rows: vocab, cols: k)

        let factoredKernel = try ANEKernel(
            milText: factoredMIL,
            weights: [
                (path: "@model_path/weights/rms_final.bin", data: rmsBlob),
                (path: "@model_path/weights/cls_proj.bin", data: w1Blob),
                (path: "@model_path/weights/cls_expand.bin", data: w2Blob)
            ],
            inputBytes: dim * laneSpatial * 2,
            outputBytes: vocab * laneSpatial * 2,
            checkBudget: false
        )

        // --- Benchmark direct ---
        for _ in 0..<warmup { try directKernel.eval() }
        var directUs: [Double] = []
        for _ in 0..<iterations {
            let t = GenerationClock.now()
            try directKernel.eval()
            let t2 = GenerationClock.now()
            directUs.append(machMicroseconds(t2 - t))
        }

        // --- Benchmark factored k=128 ---
        for _ in 0..<warmup { try factoredKernel.eval() }
        var factoredUs: [Double] = []
        for _ in 0..<iterations {
            let t = GenerationClock.now()
            try factoredKernel.eval()
            let t2 = GenerationClock.now()
            factoredUs.append(machMicroseconds(t2 - t))
        }

        // --- Sweep additional k values ---
        let kValues = [32, 64, 256]
        var kResults: [(k: Int, median: Double)] = []

        for kv in kValues {
            let mil = buildFactoredHeadMIL(dim: dim, vocab: vocab, lane: laneSpatial, bottleneck: kv)
            let w1 = WeightBlob.build(from: [Float](repeating: 0.0, count: kv * dim), rows: kv, cols: dim)
            let w2 = WeightBlob.build(from: [Float](repeating: 0.0, count: vocab * kv), rows: vocab, cols: kv)
            do {
                let kernel = try ANEKernel(
                    milText: mil,
                    weights: [
                        (path: "@model_path/weights/rms_final.bin", data: rmsBlob),
                        (path: "@model_path/weights/cls_proj.bin", data: w1),
                        (path: "@model_path/weights/cls_expand.bin", data: w2)
                    ],
                    inputBytes: dim * laneSpatial * 2,
                    outputBytes: vocab * laneSpatial * 2,
                    checkBudget: false
                )
                for _ in 0..<warmup { try kernel.eval() }
                var us: [Double] = []
                for _ in 0..<iterations {
                    let t = GenerationClock.now()
                    try kernel.eval()
                    let t2 = GenerationClock.now()
                    us.append(machMicroseconds(t2 - t))
                }
                kResults.append((k: kv, median: median(us)))
            } catch {
                print("  k=\(kv): COMPILE FAILED")
            }
        }

        func median(_ arr: [Double]) -> Double {
            let sorted = arr.sorted()
            let n = sorted.count
            return n % 2 == 0 ? (sorted[n/2 - 1] + sorted[n/2]) / 2.0 : sorted[n/2]
        }

        let directMedian = median(directUs)
        let factoredMedian = median(factoredUs)

        print("Head eval sweep (laneSpatial=\(laneSpatial), dim=\(dim), vocab=\(vocab)):")
        print("  direct [768→32000]:  \(String(format: "%.1f", directMedian)) µs (weights \(String(format: "%.1f", Double(vocab * dim * 2) / 1_048_576.0)) MB)")
        print("  k=\(k): \(String(format: "%.1f", factoredMedian)) µs (\(String(format: "%.2f", directMedian / factoredMedian))x, weights \(String(format: "%.1f", Double(k * dim * 2 + vocab * k * 2) / 1_048_576.0)) MB)")
        for r in kResults {
            let kWt = Double(r.k * dim * 2 + vocab * r.k * 2) / 1_048_576.0
            print("  k=\(r.k): \(String(format: "%.1f", r.median)) µs (\(String(format: "%.2f", directMedian / r.median))x, weights \(String(format: "%.1f", kWt)) MB)")
        }
    }

    private func buildFactoredHeadMIL(dim: Int, vocab: Int, lane: Int, bottleneck k: Int) -> String {
        let invd: Float = 1.0 / Float(dim)
        var b = MILBuilder(reserveCapacity: 4_096)
        b.append(MILText.header)
        b.appendLine("    func main<ios18>(tensor<fp16, [1, \(dim), 1, \(lane)]> x) {")
        // RMSNorm
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> sq = mul(x=x,y=x)[name=string(\"sq\")];")
        b.appendLine("        tensor<int32, [1]> raxCh = const()[name=string(\"rax_ch\"), val=tensor<int32, [1]>([1])];")
        b.appendLine("        bool kd = const()[name=string(\"kd\"), val=bool(true)];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> ss = reduce_sum(x=sq,axes=raxCh,keep_dims=kd)[name=string(\"ss\")];")
        b.append("        fp16 invd = const()[name=string(\"invd\"), val=fp16(")
        b.appendFP16(invd)
        b.appendLine(")];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];")
        b.appendLine("        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];")
        b.appendLine("        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];")
        b.appendLine("        tensor<fp16, [1,1,1,\(lane)]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];")
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];")
        b.appendLine(
            "        tensor<fp16, [1,\(dim),1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms_final.bin\"), offset=uint64(64)))];"
        )
        b.appendLine("        tensor<fp16, [1,\(dim),1,\(lane)]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];")
        b.append(MILText.convConst)
        // Factored conv: [dim→k]
        b.appendLine(
            "        tensor<fp16, [\(k), \(dim), 1, 1]> Wproj = const()[name=string(\"Wproj\"), val=tensor<fp16, [\(k), \(dim), 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/cls_proj.bin\"), offset=uint64(64)))];"
        )
        b.appendLine(
            "        tensor<fp16, [1, \(k), 1, \(lane)]> proj = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wproj,x=xn)[name=string(\"proj\")];"
        )
        // Factored conv: [k→vocab]
        b.appendLine(
            "        tensor<fp16, [\(vocab), \(k), 1, 1]> Wexp = const()[name=string(\"Wexp\"), val=tensor<fp16, [\(vocab), \(k), 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/cls_expand.bin\"), offset=uint64(64)))];"
        )
        b.appendLine(
            "        tensor<fp16, [1, \(vocab), 1, \(lane)]> logits = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wexp,x=proj)[name=string(\"expand\")];"
        )
        b.appendLine("    } -> (logits);")
        b.appendLine("}")
        return b.text
    }

    // MARK: - Grouped conv ANE probe

    func test_grouped_conv_probe_on_hardware() throws {
        try requireGenerationHardware()

        let dim = ModelConfig.dim  // 768
        let lane = 512
        let warmup = 5
        let iterations = 30

        // Build a fp16 blob with 128-byte header (matches WeightBlob format)
        func buildFP16Blob(elementCount: Int) -> Data {
            let payloadBytes = elementCount * 2
            var data = Data(count: 128 + payloadBytes)
            data.withUnsafeMutableBytes { raw in
                let base = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
                base[0] = 1
                base[4] = 2
                base[64] = 0xEF
                base[65] = 0xBE
                base[66] = 0xAD
                base[67] = 0xDE
                base[68] = 1
                raw.storeBytes(of: UInt32(payloadBytes).littleEndian, toByteOffset: 72, as: UInt32.self)
                raw.storeBytes(of: UInt32(128).littleEndian, toByteOffset: 80, as: UInt32.self)
                // Fill payload with small fp16 values
                let payload = raw.baseAddress!.advanced(by: 128).assumingMemoryBound(to: UInt16.self)
                for i in 0..<elementCount {
                    payload[i] = Float16(Float(i % 256) * 0.001).bitPattern
                }
            }
            return data
        }

        // First: minimal compile check — single conv with weight blob
        let minimalMil = """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
            func main<ios18>(tensor<fp16, [1, \(dim), 1, \(lane)]> x) {
                string pt = const()[name=string("pt"), val=string("valid")];
                tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
                tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
                tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
                int32 gr = const()[name=string("gr"), val=int32(1)];
                tensor<fp16, [\(dim),\(dim),1,1]> W = const()[name=string("W"), val=tensor<fp16, [\(dim),\(dim),1,1]>(BLOBFILE(path=string("@model_path/weights/w.bin"), offset=uint64(64)))];
                tensor<fp16, [1,\(dim),1,\(lane)]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string("out")];
            } -> (out);
        }
        """
        let wBlob = buildFP16Blob(elementCount: dim * dim)
        do {
            let kernel = try ANEKernel(
                milText: minimalMil,
                weights: [(path: "@model_path/weights/w.bin", data: wBlob)],
                inputSizes: [dim * lane * 2],
                outputSizes: [dim * lane * 2],
                checkBudget: false
            )
            try kernel.eval()
            print("minimal_conv_probe: SUCCESS")
        } catch {
            print("minimal_conv_probe: FAILED \(error)")
        }

        // Test each group count
        let groupCounts = [1, 4, 8, 16, 64, 768]

        for groups in groupCounts {
            guard dim % groups == 0 else { continue }
            let chPerGroup = dim / groups

            // Build single-layer RWKV MIL with grouped convs
            let mil = """
            program(1.3)
            [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
            {
                func main<ios18>(tensor<fp16, [1, \(dim), 1, \(lane)]> x, tensor<fp16, [1, \(dim), 1, \(lane)]> stateIn) {
                    tensor<int32, [1]> raxCh = const()[name=string("rax_ch"), val=tensor<int32, [1]>([1])];
                    bool kd = const()[name=string("kd"), val=bool(true)];
                    fp16 invd = const()[name=string("invd"), val=fp16(0.0013)];
                    fp16 eps = const()[name=string("eps"), val=fp16(0.00001)];
                    fp16 nhalf = const()[name=string("nhalf"), val=fp16(-0.5)];
                    string pt = const()[name=string("pt"), val=string("valid")];
                    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
                    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
                    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
                    int32 gr = const()[name=string("gr"), val=int32(\(groups))];

                    tensor<fp16, [1,\(dim),1,\(lane)]> sq = mul(x=x,y=x)[name=string("sq")];
                    tensor<fp16, [1,1,1,\(lane)]> ss = reduce_sum(x=sq,axes=raxCh,keep_dims=kd)[name=string("ss")];
                    tensor<fp16, [1,1,1,\(lane)]> ss2 = mul(x=ss,y=invd)[name=string("ss2")];
                    tensor<fp16, [1,1,1,\(lane)]> ss3 = add(x=ss2,y=eps)[name=string("ss3")];
                    tensor<fp16, [1,1,1,\(lane)]> rrms = pow(x=ss3,y=nhalf)[name=string("rrms")];
                    tensor<fp16, [1,\(dim),1,\(lane)]> xr = mul(x=x,y=rrms)[name=string("xr")];
                    tensor<fp16, [1,\(dim),1,1]> rw = const()[name=string("rw"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string("@model_path/weights/rms.bin"), offset=uint64(64)))];
                    tensor<fp16, [1,\(dim),1,\(lane)]> xn = mul(x=xr,y=rw)[name=string("xn")];

                    tensor<fp16, [\(dim),\(chPerGroup),1,1]> Wx = const()[name=string("Wx"), val=tensor<fp16, [\(dim),\(chPerGroup),1,1]>(BLOBFILE(path=string("@model_path/weights/wx.bin"), offset=uint64(64)))];
                    tensor<fp16, [\(dim),\(chPerGroup),1,1]> Ws = const()[name=string("Ws"), val=tensor<fp16, [\(dim),\(chPerGroup),1,1]>(BLOBFILE(path=string("@model_path/weights/ws.bin"), offset=uint64(64)))];
                    tensor<fp16, [\(dim),\(chPerGroup),1,1]> Wd = const()[name=string("Wd"), val=tensor<fp16, [\(dim),\(chPerGroup),1,1]>(BLOBFILE(path=string("@model_path/weights/wd.bin"), offset=uint64(64)))];
                    tensor<fp16, [\(dim),\(chPerGroup),1,1]> Wo = const()[name=string("Wo"), val=tensor<fp16, [\(dim),\(chPerGroup),1,1]>(BLOBFILE(path=string("@model_path/weights/wo.bin"), offset=uint64(64)))];

                    tensor<fp16, [1,\(dim),1,\(lane)]> xMix = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wx,x=xn)[name=string("x_mix")];
                    tensor<fp16, [1,\(dim),1,\(lane)]> sMix = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Ws,x=stateIn)[name=string("s_mix")];
                    tensor<fp16, [1,\(dim),1,\(lane)]> carry = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wd,x=stateIn)[name=string("carry")];
                    tensor<fp16, [1,\(dim),1,\(lane)]> mixPre = add(x=xMix,y=sMix)[name=string("mix_pre")];
                    tensor<fp16, [1,\(dim),1,\(lane)]> gate = sigmoid(x=mixPre)[name=string("gate")];
                    tensor<fp16, [1,\(dim),1,\(lane)]> gatedCarry = mul(x=carry,y=gate)[name=string("gated_carry")];
                    tensor<fp16, [1,\(dim),1,\(lane)]> stateOut = add(x=xMix,y=gatedCarry)[name=string("state_out")];
                    tensor<fp16, [1,\(dim),1,\(lane)]> proj = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wo,x=stateOut)[name=string("proj")];
                    tensor<fp16, [1,\(dim),1,\(lane)]> xNext = add(x=x,y=proj)[name=string("x_next")];
                } -> (xNext, stateOut);
            }
            """

            let convWeightSize = dim * chPerGroup
            let rmsBlob = buildFP16Blob(elementCount: dim)
            let convBlob = buildFP16Blob(elementCount: convWeightSize)
            let inputBytes = dim * lane * 2

            do {
                let kernel = try ANEKernel(
                    milText: mil,
                    weights: [
                        (path: "@model_path/weights/rms.bin", data: rmsBlob),
                        (path: "@model_path/weights/wx.bin", data: convBlob),
                        (path: "@model_path/weights/ws.bin", data: convBlob),
                        (path: "@model_path/weights/wd.bin", data: convBlob),
                        (path: "@model_path/weights/wo.bin", data: convBlob)
                    ],
                    inputSizes: [inputBytes, inputBytes],
                    outputSizes: [inputBytes, inputBytes],
                    checkBudget: false
                )

                // Warmup
                for _ in 0..<warmup { try kernel.eval() }

                // Timed
                var evalUs: [Double] = []
                evalUs.reserveCapacity(iterations)
                for _ in 0..<iterations {
                    let start = GenerationClock.now()
                    try kernel.eval()
                    let elapsed = machMicroseconds(GenerationClock.now() - start)
                    evalUs.append(elapsed)
                }
                evalUs.sort()
                let median = evalUs[evalUs.count / 2]
                let weightMB = Double(dim * chPerGroup * 4 * 2 + dim * 2) / (1024.0 * 1024.0)
                print("grouped_conv_probe groups=\(groups) chPerGroup=\(chPerGroup) eval_median_us=\(median) weight_MB=\(String(format: "%.2f", weightMB))")
            } catch {
                print("grouped_conv_probe groups=\(groups) chPerGroup=\(chPerGroup) FAILED: \(error)")
            }
        }
    }

    // MARK: - 3-layer fused grouped conv probe

    func test_grouped_conv_3layer_fused_probe_on_hardware() throws {
        try requireGenerationHardware()

        let dim = ModelConfig.dim  // 768
        let lane = 512
        let warmup = 5
        let iterations = 30

        func buildFP16Blob3L(elementCount: Int) -> Data {
            let payloadBytes = elementCount * 2
            var data = Data(count: 128 + payloadBytes)
            data.withUnsafeMutableBytes { raw in
                let base = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
                base[0] = 1; base[4] = 2
                base[64] = 0xEF; base[65] = 0xBE; base[66] = 0xAD; base[67] = 0xDE; base[68] = 1
                raw.storeBytes(of: UInt32(payloadBytes).littleEndian, toByteOffset: 72, as: UInt32.self)
                raw.storeBytes(of: UInt32(128).littleEndian, toByteOffset: 80, as: UInt32.self)
                let payload = raw.baseAddress!.advanced(by: 128).assumingMemoryBound(to: UInt16.self)
                for i in 0..<elementCount { payload[i] = Float16(Float(i % 256) * 0.001).bitPattern }
            }
            return data
        }

        func buildFused3LayerMIL(groups: Int) -> String {
            let chPerGroup = dim / groups
            var mil = """
            program(1.3)
            [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
            {
                func main<ios18>(tensor<fp16, [1, \(dim), 1, \(lane)]> x, tensor<fp16, [1, \(dim), 1, \(lane)]> stateIn0, tensor<fp16, [1, \(dim), 1, \(lane)]> stateIn1, tensor<fp16, [1, \(dim), 1, \(lane)]> stateIn2) {
                    tensor<int32, [1]> raxCh = const()[name=string("rax_ch"), val=tensor<int32, [1]>([1])];
                    bool kd = const()[name=string("kd"), val=bool(true)];
                    fp16 invd = const()[name=string("invd"), val=fp16(0.0013)];
                    fp16 eps = const()[name=string("eps"), val=fp16(0.00001)];
                    fp16 nhalf = const()[name=string("nhalf"), val=fp16(-0.5)];
                    string pt = const()[name=string("pt"), val=string("valid")];
                    tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
                    tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
                    tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
                    int32 gr = const()[name=string("gr"), val=int32(\(groups))];

            """
            for i in 0..<3 {
                let p = "l\(i)_"
                let xIn = i == 0 ? "x" : "l\(i-1)_xNext"
                let xOut = i == 2 ? "xNext" : "l\(i)_xNext"
                let sIn = "stateIn\(i)"
                let sOut = "stateOut\(i)"
                mil += """
                        tensor<fp16, [1,\(dim),1,\(lane)]> \(p)sq = mul(x=\(xIn),y=\(xIn))[name=string("\(p)sq")];
                        tensor<fp16, [1,1,1,\(lane)]> \(p)ss = reduce_sum(x=\(p)sq,axes=raxCh,keep_dims=kd)[name=string("\(p)ss")];
                        tensor<fp16, [1,1,1,\(lane)]> \(p)ss2 = mul(x=\(p)ss,y=invd)[name=string("\(p)ss2")];
                        tensor<fp16, [1,1,1,\(lane)]> \(p)ss3 = add(x=\(p)ss2,y=eps)[name=string("\(p)ss3")];
                        tensor<fp16, [1,1,1,\(lane)]> \(p)rrms = pow(x=\(p)ss3,y=nhalf)[name=string("\(p)rrms")];
                        tensor<fp16, [1,\(dim),1,\(lane)]> \(p)xr = mul(x=\(xIn),y=\(p)rrms)[name=string("\(p)xr")];
                        tensor<fp16, [1,\(dim),1,1]> \(p)rw = const()[name=string("\(p)rw"), val=tensor<fp16, [1,\(dim),1,1]>(BLOBFILE(path=string("@model_path/weights/rms\(i).bin"), offset=uint64(64)))];
                        tensor<fp16, [1,\(dim),1,\(lane)]> \(p)xn = mul(x=\(p)xr,y=\(p)rw)[name=string("\(p)xn")];
                        tensor<fp16, [\(dim),\(chPerGroup),1,1]> \(p)Wx = const()[name=string("\(p)Wx"), val=tensor<fp16, [\(dim),\(chPerGroup),1,1]>(BLOBFILE(path=string("@model_path/weights/wx\(i).bin"), offset=uint64(64)))];
                        tensor<fp16, [\(dim),\(chPerGroup),1,1]> \(p)Ws = const()[name=string("\(p)Ws"), val=tensor<fp16, [\(dim),\(chPerGroup),1,1]>(BLOBFILE(path=string("@model_path/weights/ws\(i).bin"), offset=uint64(64)))];
                        tensor<fp16, [\(dim),\(chPerGroup),1,1]> \(p)Wd = const()[name=string("\(p)Wd"), val=tensor<fp16, [\(dim),\(chPerGroup),1,1]>(BLOBFILE(path=string("@model_path/weights/wd\(i).bin"), offset=uint64(64)))];
                        tensor<fp16, [\(dim),\(chPerGroup),1,1]> \(p)Wo = const()[name=string("\(p)Wo"), val=tensor<fp16, [\(dim),\(chPerGroup),1,1]>(BLOBFILE(path=string("@model_path/weights/wo\(i).bin"), offset=uint64(64)))];
                        tensor<fp16, [1,\(dim),1,\(lane)]> \(p)xMix = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(p)Wx,x=\(p)xn)[name=string("\(p)x_mix")];
                        tensor<fp16, [1,\(dim),1,\(lane)]> \(p)sMix = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(p)Ws,x=\(sIn))[name=string("\(p)s_mix")];
                        tensor<fp16, [1,\(dim),1,\(lane)]> \(p)carry = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(p)Wd,x=\(sIn))[name=string("\(p)carry")];
                        tensor<fp16, [1,\(dim),1,\(lane)]> \(p)mixPre = add(x=\(p)xMix,y=\(p)sMix)[name=string("\(p)mix_pre")];
                        tensor<fp16, [1,\(dim),1,\(lane)]> \(p)gate = sigmoid(x=\(p)mixPre)[name=string("\(p)gate")];
                        tensor<fp16, [1,\(dim),1,\(lane)]> \(p)gatedCarry = mul(x=\(p)carry,y=\(p)gate)[name=string("\(p)gated_carry")];
                        tensor<fp16, [1,\(dim),1,\(lane)]> \(sOut) = add(x=\(p)xMix,y=\(p)gatedCarry)[name=string("\(p)state_out")];
                        tensor<fp16, [1,\(dim),1,\(lane)]> \(p)proj = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=\(p)Wo,x=\(sOut))[name=string("\(p)proj")];
                        tensor<fp16, [1,\(dim),1,\(lane)]> \(xOut) = add(x=\(xIn),y=\(p)proj)[name=string("\(p)x_next")];

                """
            }
            mil += """
                } -> (xNext,stateOut0,stateOut1,stateOut2);
            }
            """
            return mil
        }

        let groupCounts = [1, 16, 64]
        let inputBytes = dim * lane * 2

        for groups in groupCounts {
            let chPerGroup = dim / groups
            let convWeightSize = dim * chPerGroup
            let rmsBlob = buildFP16Blob3L(elementCount: dim)
            let convBlob = buildFP16Blob3L(elementCount: convWeightSize)

            var weightPairs: [(path: String, data: Data)] = []
            for i in 0..<3 {
                weightPairs.append(("@model_path/weights/rms\(i).bin", rmsBlob))
                weightPairs.append(("@model_path/weights/wx\(i).bin", convBlob))
                weightPairs.append(("@model_path/weights/ws\(i).bin", convBlob))
                weightPairs.append(("@model_path/weights/wd\(i).bin", convBlob))
                weightPairs.append(("@model_path/weights/wo\(i).bin", convBlob))
            }

            let mil = buildFused3LayerMIL(groups: groups)

            do {
                let kernel = try ANEKernel(
                    milText: mil,
                    weights: weightPairs,
                    inputSizes: [inputBytes, inputBytes, inputBytes, inputBytes],
                    outputSizes: [inputBytes, inputBytes, inputBytes, inputBytes],
                    checkBudget: false
                )

                for _ in 0..<warmup { try kernel.eval() }

                var evalUs: [Double] = []
                evalUs.reserveCapacity(iterations)
                for _ in 0..<iterations {
                    let start = GenerationClock.now()
                    try kernel.eval()
                    let elapsed = machMicroseconds(GenerationClock.now() - start)
                    evalUs.append(elapsed)
                }
                evalUs.sort()
                let median = evalUs[evalUs.count / 2]
                let weightMB = Double(3 * (dim * chPerGroup * 4 * 2 + dim * 2)) / (1024.0 * 1024.0)
                print("fused_3l_grouped groups=\(groups) chPerGroup=\(chPerGroup) eval_median_us=\(median) weight_MB=\(String(format: "%.2f", weightMB))")
            } catch {
                print("fused_3l_grouped groups=\(groups) chPerGroup=\(chPerGroup) FAILED: \(error)")
            }
        }
    }

    func test_metal_argmax_probe_on_hardware() throws {
        try requireGenerationHardware()

        guard let device = MTLCreateSystemDefaultDevice() else {
            print("METAL: No GPU device available")
            return
        }
        guard let queue = device.makeCommandQueue() else {
            print("METAL: No command queue")
            return
        }

        // Metal compute shader for per-lane argmax over channels
        // Each thread handles one spatial lane. Iterates over all channels.
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void argmax_channels(
            device const half *data [[buffer(0)]],
            device int *outIndices [[buffer(1)]],
            device half *outValues [[buffer(2)]],
            constant uint &channels [[buffer(3)]],
            constant uint &spatial [[buffer(4)]],
            uint lane [[thread_position_in_grid]]
        ) {
            if (lane >= spatial) return;
            half bestVal = data[lane]; // channel 0
            int bestIdx = 0;
            for (uint c = 1; c < channels; c++) {
                half val = data[c * spatial + lane];
                if (val > bestVal) {
                    bestVal = val;
                    bestIdx = int(c);
                }
            }
            outIndices[lane] = bestIdx;
            outValues[lane] = bestVal;
        }
        """

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: shaderSource, options: nil)
        } catch {
            print("METAL: Shader compile failed: \(error)")
            return
        }
        guard let function = library.makeFunction(name: "argmax_channels"),
              let pso = try? device.makeComputePipelineState(function: function) else {
            print("METAL: Pipeline state failed")
            return
        }

        let laneSpatial = 256
        let channels = 32000  // ModelConfig.vocab
        let dim = ModelConfig.dim

        // Create a factored head kernel to get a real output surface
        let weights = makeEchoRecurrentGenerationWeights(layerCount: 6)
        let factoredHead = try FactoredGenerationRMSNormClassifierKernelSet(
            rmsFinal: weights.rmsFinal,
            classifierProjection: TensorBuffer(count: 128 * dim, zeroed: true),
            classifierExpansion: TensorBuffer(count: channels * 128, zeroed: true),
            vocabSize: channels,
            bottleneck: 128,
            laneSpatial: laneSpatial
        )
        let headOut = try factoredHead.rmsNormClassifier.outputSurface(at: 0)

        // Eval once to populate the output surface
        try factoredHead.rmsNormClassifier.eval()

        // Create Metal buffer from IOSurface base address (unified memory, zero-copy)
        let surfaceSize = IOSurfaceGetAllocSize(headOut)
        IOSurfaceLock(headOut, .readOnly, nil)
        let baseAddr = IOSurfaceGetBaseAddress(headOut)
        let metalBuf = device.makeBuffer(bytesNoCopy: baseAddr, length: surfaceSize, options: .storageModeShared, deallocator: nil)!
        IOSurfaceUnlock(headOut, .readOnly, nil)

        // Output buffers
        let indexBuf = device.makeBuffer(length: laneSpatial * MemoryLayout<Int32>.size, options: .storageModeShared)!
        let valueBuf = device.makeBuffer(length: laneSpatial * MemoryLayout<UInt16>.size, options: .storageModeShared)!
        var channelsVal = UInt32(channels)
        var spatialVal = UInt32(laneSpatial)

        // Warmup
        for _ in 0..<5 {
            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pso)
            enc.setBuffer(metalBuf, offset: 0, index: 0)
            enc.setBuffer(indexBuf, offset: 0, index: 1)
            enc.setBuffer(valueBuf, offset: 0, index: 2)
            enc.setBytes(&channelsVal, length: 4, index: 3)
            enc.setBytes(&spatialVal, length: 4, index: 4)
            enc.dispatchThreads(MTLSize(width: laneSpatial, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: min(laneSpatial, pso.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }

        // Benchmark Metal argmax
        var metalTimesUs: [Double] = []
        for _ in 0..<50 {
            IOSurfaceLock(headOut, .readOnly, nil)
            let t = GenerationClock.now()

            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(pso)
            enc.setBuffer(metalBuf, offset: 0, index: 0)
            enc.setBuffer(indexBuf, offset: 0, index: 1)
            enc.setBuffer(valueBuf, offset: 0, index: 2)
            enc.setBytes(&channelsVal, length: 4, index: 3)
            enc.setBytes(&spatialVal, length: 4, index: 4)
            enc.dispatchThreads(MTLSize(width: laneSpatial, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: min(laneSpatial, pso.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()

            metalTimesUs.append(machMicroseconds(GenerationClock.now() - t))
            IOSurfaceUnlock(headOut, .readOnly, nil)
        }

        // Benchmark NEON argmax for comparison
        var neonTimesUs: [Double] = []
        for _ in 0..<50 {
            let t = GenerationClock.now()
            let _ = try SurfaceIO.argmaxBatchFP16Spatial(
                from: headOut, channelOffset: 0, spatial: laneSpatial,
                channels: channels, streamCount: laneSpatial
            )
            neonTimesUs.append(machMicroseconds(GenerationClock.now() - t))
        }

        metalTimesUs.sort()
        neonTimesUs.sort()
        let metalMedian = metalTimesUs[25]
        let neonMedian = neonTimesUs[25]

        print("Metal argmax: median=\(String(format: "%.1f", metalMedian))µs (32K ch × 256 lanes)")
        print("NEON argmax:  median=\(String(format: "%.1f", neonMedian))µs")
        print("Speedup: \(String(format: "%.2f", neonMedian / metalMedian))x")

        // Verify correctness
        let metalIndices = indexBuf.contents().bindMemory(to: Int32.self, capacity: laneSpatial)
        let neonResults = try SurfaceIO.argmaxBatchFP16Spatial(
            from: headOut, channelOffset: 0, spatial: laneSpatial,
            channels: channels, streamCount: laneSpatial
        )
        var correct = true
        for i in 0..<laneSpatial {
            if metalIndices[i] != Int32(neonResults[i].index) {
                print("MISMATCH at lane \(i): metal=\(metalIndices[i]) neon=\(neonResults[i].index)")
                correct = false
                break
            }
        }
        print("Correctness: \(correct ? "MATCH" : "MISMATCH")")
    }

    func test_batched_step_component_timing_256_on_hardware() throws {
        try requireGenerationHardware()

        let layerCount = 6
        let streamCount = 256
        let dim = ModelConfig.dim
        let laneSpatial = 256
        let iterations = 50

        let weights = makeEchoRecurrentGenerationWeights(layerCount: layerCount)
        let tripletCount = layerCount / 3

        var tripletSessions = try LayerStorage<RWKVStyleFusedThreeLayerSession>(
            count: tripletCount,
            throwingInitializer: { tripletIdx in
                let base = tripletIdx * 3
                return try RWKVStyleFusedThreeLayerSession(
                    weights0: weights.layers[base],
                    weights1: weights.layers[base + 1],
                    weights2: weights.layers[base + 2],
                    laneSpatial: laneSpatial
                )
            }
        )

        let factoredHead = try FactoredGenerationRMSNormClassifierKernelSet(
            rmsFinal: weights.rmsFinal,
            classifierProjection: TensorBuffer(count: 128 * ModelConfig.dim, zeroed: true),
            classifierExpansion: TensorBuffer(count: weights.vocabSize * 128, zeroed: true),
            vocabSize: weights.vocabSize,
            bottleneck: 128,
            laneSpatial: laneSpatial
        )

        let t0xIn = tripletSessions[0].handles.xIn
        let t0xOut = tripletSessions[0].handles.xOut
        let t0sIn0 = tripletSessions[0].handles.stateIn0
        let t0sIn1 = tripletSessions[0].handles.stateIn1
        let t0sIn2 = tripletSessions[0].handles.stateIn2
        let t0sOut0 = tripletSessions[0].handles.stateOut0
        let t0sOut1 = tripletSessions[0].handles.stateOut1
        let t0sOut2 = tripletSessions[0].handles.stateOut2
        let t1xIn = tripletSessions[1].handles.xIn
        let t1xOut = tripletSessions[1].handles.xOut
        let t1sIn0 = tripletSessions[1].handles.stateIn0
        let t1sIn1 = tripletSessions[1].handles.stateIn1
        let t1sIn2 = tripletSessions[1].handles.stateIn2
        let t1sOut0 = tripletSessions[1].handles.stateOut0
        let t1sOut1 = tripletSessions[1].handles.stateOut1
        let t1sOut2 = tripletSessions[1].handles.stateOut2
        let headIn = try factoredHead.rmsNormClassifier.inputSurface(at: 0)
        let headOut = try factoredHead.rmsNormClassifier.outputSurface(at: 0)
        let vocabSize = weights.vocabSize

        var tokens = Array(repeating: UInt16(0), count: streamCount)

        // Warmup
        for _ in 0..<3 {
            try batchedTokenStepWithEval(
                tokens: &tokens, streamCount: streamCount,
                embedding: weights.embedding, dim: dim, laneSpatial: laneSpatial,
                tripletCount: tripletCount, tripletSessions: &tripletSessions,
                t0xIn: t0xIn, t0xOut: t0xOut,
                t0sIn0: t0sIn0, t0sIn1: t0sIn1, t0sIn2: t0sIn2,
                t0sOut0: t0sOut0, t0sOut1: t0sOut1, t0sOut2: t0sOut2,
                t1xIn: t1xIn, t1xOut: t1xOut,
                t1sIn0: t1sIn0, t1sIn1: t1sIn1, t1sIn2: t1sIn2,
                t1sOut0: t1sOut0, t1sOut1: t1sOut1, t1sOut2: t1sOut2,
                headIn: headIn, headOut: headOut,
                headEval: { try factoredHead.rmsNormClassifier.eval() },
                vocabSize: vocabSize
            )
        }

        var embedUs: [Double] = []
        var evalT0Us: [Double] = []
        var stateT0Us: [Double] = []
        var xferUs: [Double] = []
        var evalT1Us: [Double] = []
        var stateT1Us: [Double] = []
        var headCopyUs: [Double] = []
        var evalHeadUs: [Double] = []
        var argmaxUs: [Double] = []

        for _ in 0..<iterations {
            var t = GenerationClock.now()

            try weights.embedding.withUnsafePointer { embPtr in
                try tokens.withUnsafeBufferPointer { tokenBuf in
                    try SurfaceIO.writeEmbeddingBatchFP16(
                        to: t0xIn, channelOffset: 0, spatial: laneSpatial,
                        embeddingTable: embPtr, dim: dim,
                        tokenIDs: tokenBuf.baseAddress!, streamCount: streamCount
                    )
                }
            }
            var t2 = GenerationClock.now(); embedUs.append(machMicroseconds(t2 - t)); t = t2

            try tripletSessions[0].kernels.step.eval()
            t2 = GenerationClock.now(); evalT0Us.append(machMicroseconds(t2 - t)); t = t2

            try SurfaceIO.copyFP16(dst: t0sIn0, dstChannelOffset: 0, src: t0sOut0, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            try SurfaceIO.copyFP16(dst: t0sIn1, dstChannelOffset: 0, src: t0sOut1, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            try SurfaceIO.copyFP16(dst: t0sIn2, dstChannelOffset: 0, src: t0sOut2, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            t2 = GenerationClock.now(); stateT0Us.append(machMicroseconds(t2 - t)); t = t2

            try SurfaceIO.copyFP16(dst: t1xIn, dstChannelOffset: 0, src: t0xOut, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            t2 = GenerationClock.now(); xferUs.append(machMicroseconds(t2 - t)); t = t2

            try tripletSessions[1].kernels.step.eval()
            t2 = GenerationClock.now(); evalT1Us.append(machMicroseconds(t2 - t)); t = t2

            try SurfaceIO.copyFP16(dst: t1sIn0, dstChannelOffset: 0, src: t1sOut0, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            try SurfaceIO.copyFP16(dst: t1sIn1, dstChannelOffset: 0, src: t1sOut1, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            try SurfaceIO.copyFP16(dst: t1sIn2, dstChannelOffset: 0, src: t1sOut2, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            t2 = GenerationClock.now(); stateT1Us.append(machMicroseconds(t2 - t)); t = t2

            try SurfaceIO.copyFP16(dst: headIn, dstChannelOffset: 0, src: t1xOut, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            t2 = GenerationClock.now(); headCopyUs.append(machMicroseconds(t2 - t)); t = t2

            try factoredHead.rmsNormClassifier.eval()
            t2 = GenerationClock.now(); evalHeadUs.append(machMicroseconds(t2 - t)); t = t2

            let argmaxResults = try SurfaceIO.argmaxBatchFP16SpatialParallel(
                from: headOut, channelOffset: 0, spatial: laneSpatial,
                channels: vocabSize, streamCount: streamCount, nBlocks: 8
            )
            for i in 0..<streamCount { tokens[i] = UInt16(argmaxResults[i].index) }
            t2 = GenerationClock.now(); argmaxUs.append(machMicroseconds(t2 - t))
        }

        func median(_ arr: [Double]) -> Double {
            let sorted = arr.sorted()
            let n = sorted.count
            return n % 2 == 0 ? (sorted[n/2 - 1] + sorted[n/2]) / 2.0 : sorted[n/2]
        }

        let totalUs = median(embedUs) + median(evalT0Us) + median(stateT0Us) + median(xferUs) +
            median(evalT1Us) + median(stateT1Us) + median(headCopyUs) + median(evalHeadUs) + median(argmaxUs)

        print("Component timing (median µs, \(streamCount) streams @ lane=\(laneSpatial), \(iterations) iterations):")
        print("  embed_write: \(String(format: "%.1f", median(embedUs))) µs (\(String(format: "%.1f", median(embedUs) / totalUs * 100))%)")
        print("  eval_T0:     \(String(format: "%.1f", median(evalT0Us))) µs (\(String(format: "%.1f", median(evalT0Us) / totalUs * 100))%)")
        print("  state_T0:    \(String(format: "%.1f", median(stateT0Us))) µs (\(String(format: "%.1f", median(stateT0Us) / totalUs * 100))%)")
        print("  xfer_T0T1:   \(String(format: "%.1f", median(xferUs))) µs (\(String(format: "%.1f", median(xferUs) / totalUs * 100))%)")
        print("  eval_T1:     \(String(format: "%.1f", median(evalT1Us))) µs (\(String(format: "%.1f", median(evalT1Us) / totalUs * 100))%)")
        print("  state_T1:    \(String(format: "%.1f", median(stateT1Us))) µs (\(String(format: "%.1f", median(stateT1Us) / totalUs * 100))%)")
        print("  head_copy:   \(String(format: "%.1f", median(headCopyUs))) µs (\(String(format: "%.1f", median(headCopyUs) / totalUs * 100))%)")
        print("  eval_head:   \(String(format: "%.1f", median(evalHeadUs))) µs (\(String(format: "%.1f", median(evalHeadUs) / totalUs * 100))%)")
        print("  argmax:      \(String(format: "%.1f", median(argmaxUs))) µs (\(String(format: "%.1f", median(argmaxUs) / totalUs * 100))%)")
        print("  TOTAL:       \(String(format: "%.1f", totalUs)) µs")
        print("  implied_tps: \(String(format: "%.1f", Double(streamCount) / (totalUs / 1_000_000.0)))")
    }

    func test_batched_step_component_timing_512_on_hardware() throws {
        try requireGenerationHardware()

        let layerCount = 6
        let streamCount = 512
        let dim = ModelConfig.dim
        let laneSpatial = 512
        let iterations = 50

        let weights = makeEchoRecurrentGenerationWeights(layerCount: layerCount)
        let tripletCount = layerCount / 3

        var tripletSessions = try LayerStorage<RWKVStyleFusedThreeLayerSession>(
            count: tripletCount,
            throwingInitializer: { tripletIdx in
                let base = tripletIdx * 3
                return try RWKVStyleFusedThreeLayerSession(
                    weights0: weights.layers[base],
                    weights1: weights.layers[base + 1],
                    weights2: weights.layers[base + 2],
                    laneSpatial: laneSpatial
                )
            }
        )

        let factoredHead = try FactoredGenerationRMSNormClassifierKernelSet(
            rmsFinal: weights.rmsFinal,
            classifierProjection: TensorBuffer(count: 128 * ModelConfig.dim, zeroed: true),
            classifierExpansion: TensorBuffer(count: weights.vocabSize * 128, zeroed: true),
            vocabSize: weights.vocabSize,
            bottleneck: 128,
            laneSpatial: laneSpatial
        )

        let t0xIn = tripletSessions[0].handles.xIn
        let t0xOut = tripletSessions[0].handles.xOut
        let t0sIn0 = tripletSessions[0].handles.stateIn0
        let t0sIn1 = tripletSessions[0].handles.stateIn1
        let t0sIn2 = tripletSessions[0].handles.stateIn2
        let t0sOut0 = tripletSessions[0].handles.stateOut0
        let t0sOut1 = tripletSessions[0].handles.stateOut1
        let t0sOut2 = tripletSessions[0].handles.stateOut2
        let t1xIn = tripletSessions[1].handles.xIn
        let t1xOut = tripletSessions[1].handles.xOut
        let t1sIn0 = tripletSessions[1].handles.stateIn0
        let t1sIn1 = tripletSessions[1].handles.stateIn1
        let t1sIn2 = tripletSessions[1].handles.stateIn2
        let t1sOut0 = tripletSessions[1].handles.stateOut0
        let t1sOut1 = tripletSessions[1].handles.stateOut1
        let t1sOut2 = tripletSessions[1].handles.stateOut2
        let headIn = try factoredHead.rmsNormClassifier.inputSurface(at: 0)
        let headOut = try factoredHead.rmsNormClassifier.outputSurface(at: 0)
        let vocabSize = weights.vocabSize

        var tokens = Array(repeating: UInt16(0), count: streamCount)

        // Warmup
        for _ in 0..<3 {
            try batchedTokenStepWithEval(
                tokens: &tokens, streamCount: streamCount,
                embedding: weights.embedding, dim: dim, laneSpatial: laneSpatial,
                tripletCount: tripletCount, tripletSessions: &tripletSessions,
                t0xIn: t0xIn, t0xOut: t0xOut,
                t0sIn0: t0sIn0, t0sIn1: t0sIn1, t0sIn2: t0sIn2,
                t0sOut0: t0sOut0, t0sOut1: t0sOut1, t0sOut2: t0sOut2,
                t1xIn: t1xIn, t1xOut: t1xOut,
                t1sIn0: t1sIn0, t1sIn1: t1sIn1, t1sIn2: t1sIn2,
                t1sOut0: t1sOut0, t1sOut1: t1sOut1, t1sOut2: t1sOut2,
                headIn: headIn, headOut: headOut,
                headEval: { try factoredHead.rmsNormClassifier.eval() },
                vocabSize: vocabSize
            )
        }

        var embedUs: [Double] = []
        var evalT0Us: [Double] = []
        var stateT0Us: [Double] = []
        var xferUs: [Double] = []
        var evalT1Us: [Double] = []
        var stateT1Us: [Double] = []
        var headCopyUs: [Double] = []
        var evalHeadUs: [Double] = []
        var argmaxUs: [Double] = []

        for _ in 0..<iterations {
            var t = GenerationClock.now()

            try weights.embedding.withUnsafePointer { embPtr in
                try tokens.withUnsafeBufferPointer { tokenBuf in
                    try SurfaceIO.writeEmbeddingBatchFP16(
                        to: t0xIn, channelOffset: 0, spatial: laneSpatial,
                        embeddingTable: embPtr, dim: dim,
                        tokenIDs: tokenBuf.baseAddress!, streamCount: streamCount
                    )
                }
            }
            var t2 = GenerationClock.now(); embedUs.append(machMicroseconds(t2 - t)); t = t2

            try tripletSessions[0].kernels.step.eval()
            t2 = GenerationClock.now(); evalT0Us.append(machMicroseconds(t2 - t)); t = t2

            try SurfaceIO.copyFP16(dst: t0sIn0, dstChannelOffset: 0, src: t0sOut0, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            try SurfaceIO.copyFP16(dst: t0sIn1, dstChannelOffset: 0, src: t0sOut1, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            try SurfaceIO.copyFP16(dst: t0sIn2, dstChannelOffset: 0, src: t0sOut2, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            t2 = GenerationClock.now(); stateT0Us.append(machMicroseconds(t2 - t)); t = t2

            try SurfaceIO.copyFP16(dst: t1xIn, dstChannelOffset: 0, src: t0xOut, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            t2 = GenerationClock.now(); xferUs.append(machMicroseconds(t2 - t)); t = t2

            try tripletSessions[1].kernels.step.eval()
            t2 = GenerationClock.now(); evalT1Us.append(machMicroseconds(t2 - t)); t = t2

            try SurfaceIO.copyFP16(dst: t1sIn0, dstChannelOffset: 0, src: t1sOut0, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            try SurfaceIO.copyFP16(dst: t1sIn1, dstChannelOffset: 0, src: t1sOut1, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            try SurfaceIO.copyFP16(dst: t1sIn2, dstChannelOffset: 0, src: t1sOut2, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            t2 = GenerationClock.now(); stateT1Us.append(machMicroseconds(t2 - t)); t = t2

            try SurfaceIO.copyFP16(dst: headIn, dstChannelOffset: 0, src: t1xOut, srcChannelOffset: 0, channels: dim, spatial: laneSpatial)
            t2 = GenerationClock.now(); headCopyUs.append(machMicroseconds(t2 - t)); t = t2

            try factoredHead.rmsNormClassifier.eval()
            t2 = GenerationClock.now(); evalHeadUs.append(machMicroseconds(t2 - t)); t = t2

            let argmaxResults = try SurfaceIO.argmaxBatchFP16SpatialParallel(
                from: headOut, channelOffset: 0, spatial: laneSpatial,
                channels: vocabSize, streamCount: streamCount, nBlocks: 8
            )
            for i in 0..<streamCount { tokens[i] = UInt16(argmaxResults[i].index) }
            t2 = GenerationClock.now(); argmaxUs.append(machMicroseconds(t2 - t))
        }

        func median(_ arr: [Double]) -> Double {
            let sorted = arr.sorted()
            let n = sorted.count
            return n % 2 == 0 ? (sorted[n/2 - 1] + sorted[n/2]) / 2.0 : sorted[n/2]
        }

        let totalUs = median(embedUs) + median(evalT0Us) + median(stateT0Us) + median(xferUs) +
            median(evalT1Us) + median(stateT1Us) + median(headCopyUs) + median(evalHeadUs) + median(argmaxUs)

        print("Component timing (median µs, \(streamCount) streams @ lane=\(laneSpatial), \(iterations) iterations):")
        print("  embed_write: \(String(format: "%.1f", median(embedUs))) µs (\(String(format: "%.1f", median(embedUs) / totalUs * 100))%)")
        print("  eval_T0:     \(String(format: "%.1f", median(evalT0Us))) µs (\(String(format: "%.1f", median(evalT0Us) / totalUs * 100))%)")
        print("  state_T0:    \(String(format: "%.1f", median(stateT0Us))) µs (\(String(format: "%.1f", median(stateT0Us) / totalUs * 100))%)")
        print("  xfer_T0T1:   \(String(format: "%.1f", median(xferUs))) µs (\(String(format: "%.1f", median(xferUs) / totalUs * 100))%)")
        print("  eval_T1:     \(String(format: "%.1f", median(evalT1Us))) µs (\(String(format: "%.1f", median(evalT1Us) / totalUs * 100))%)")
        print("  state_T1:    \(String(format: "%.1f", median(stateT1Us))) µs (\(String(format: "%.1f", median(stateT1Us) / totalUs * 100))%)")
        print("  head_copy:   \(String(format: "%.1f", median(headCopyUs))) µs (\(String(format: "%.1f", median(headCopyUs) / totalUs * 100))%)")
        print("  eval_head:   \(String(format: "%.1f", median(evalHeadUs))) µs (\(String(format: "%.1f", median(evalHeadUs) / totalUs * 100))%)")
        print("  argmax:      \(String(format: "%.1f", median(argmaxUs))) µs (\(String(format: "%.1f", median(argmaxUs) / totalUs * 100))%)")
        print("  TOTAL:       \(String(format: "%.1f", totalUs)) µs")
        print("  implied_tps: \(String(format: "%.1f", Double(streamCount) / (totalUs / 1_000_000.0)))")
    }

    func test_head_k_sweep_512_on_hardware() throws {
        try requireGenerationHardware()

        let layerCount = 6
        let streamCount = 512
        let dim = ModelConfig.dim
        let laneSpatial = 512
        let iterations = 30

        let weights = makeEchoRecurrentGenerationWeights(layerCount: layerCount)
        let vocabSize = weights.vocabSize

        for k in [32, 64, 96, 128, 192, 256] {
            let factoredHead = try FactoredGenerationRMSNormClassifierKernelSet(
                rmsFinal: weights.rmsFinal,
                classifierProjection: TensorBuffer(count: k * dim, zeroed: true),
                classifierExpansion: TensorBuffer(count: vocabSize * k, zeroed: true),
                vocabSize: vocabSize,
                bottleneck: k,
                laneSpatial: laneSpatial
            )
            let headOut = try factoredHead.rmsNormClassifier.outputSurface(at: 0)

            // Warmup
            for _ in 0..<5 { try factoredHead.rmsNormClassifier.eval() }

            var evalUs: [Double] = []
            var argmaxUs: [Double] = []
            for _ in 0..<iterations {
                var t = GenerationClock.now()
                try factoredHead.rmsNormClassifier.eval()
                var t2 = GenerationClock.now()
                evalUs.append(machMicroseconds(t2 - t))

                t = t2
                let _ = try SurfaceIO.argmaxBatchFP16SpatialParallel(
                    from: headOut, channelOffset: 0, spatial: laneSpatial,
                    channels: vocabSize, streamCount: streamCount, nBlocks: 8
                )
                t2 = GenerationClock.now()
                argmaxUs.append(machMicroseconds(t2 - t))
            }

            func median(_ arr: [Double]) -> Double {
                let sorted = arr.sorted()
                let n = sorted.count
                return n % 2 == 0 ? (sorted[n/2-1] + sorted[n/2]) / 2.0 : sorted[n/2]
            }

            let headMs = median(evalUs) / 1000.0
            let argMs = median(argmaxUs) / 1000.0
            let projWeightKB = k * dim * 2 / 1024
            let expandWeightKB = vocabSize * k * 2 / 1024
            print("k=\(k): head_eval=\(String(format: "%.3f", headMs))ms argmax=\(String(format: "%.3f", argMs))ms total=\(String(format: "%.3f", headMs + argMs))ms proj_w=\(projWeightKB)KB expand_w=\(expandWeightKB)KB")
        }
    }

    private func machMicroseconds(_ deltaTicks: UInt64) -> Double {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        let nanos = Double(deltaTicks) * Double(info.numer) / Double(info.denom)
        return nanos / 1_000.0
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

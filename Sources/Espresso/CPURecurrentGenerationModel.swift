import Accelerate
import CPUOps
import Foundation
import ANETypes

public struct CPURecurrentGenerationModel: ~Copyable, FutureTokenProposingLanguageModel, GenerationPerformanceTrackable {
    public let vocabSize: Int
    public let layerCount: Int

    private let layerWeights: [CPULayerWeights]
    private let rmsFinal: [Float]
    private let embedding: [Float]
    private let classifier: [Float]
    private let sharedClassifier: Bool
    private let futureRMS: [Float]
    private let futureClassifier: [Float]
    private let hasFutureHead: Bool
    private let stepRMSWorkspace: RMSNorm.Workspace
    private let futureRMSWorkspace: RMSNorm.Workspace
    private var states: [[Float]]
    private var currentActivation: [Float]
    private var hasCurrentActivation: Bool

    public var performanceSnapshot: GenerationPerformanceSnapshot {
        GenerationPerformanceSnapshot()
    }

    public init(
        weights: borrowing RecurrentGenerationWeights,
        layerCount: Int
    ) throws(GenerationError) {
        let futureHead = CPUFutureHead.none(vocabSize: weights.vocabSize)
        try self.init(weights: weights, layerCount: layerCount, futureHead: futureHead)
    }

    public init(
        weights: borrowing RecurrentGenerationWeights,
        layerCount: Int,
        futureSidecar: borrowing TwoStepStudentSidecar
    ) throws(GenerationError) {
        guard futureSidecar.contract.dim == ModelConfig.dim else {
            throw .invalidArguments("futureSidecar dim \(futureSidecar.contract.dim) does not match ModelConfig.dim \(ModelConfig.dim)")
        }
        guard futureSidecar.contract.vocabSize == weights.vocabSize else {
            throw .invalidArguments("futureSidecar vocab \(futureSidecar.contract.vocabSize) does not match weights vocab \(weights.vocabSize)")
        }
        let futureHead = CPUFutureHead(
            rms: copyArray(from: futureSidecar.futureRMS),
            classifier: copyArray(from: futureSidecar.futureClassifier),
            hasHead: true
        )
        try self.init(weights: weights, layerCount: layerCount, futureHead: futureHead)
    }

    private init(
        weights: borrowing RecurrentGenerationWeights,
        layerCount: Int,
        futureHead: CPUFutureHead
    ) throws(GenerationError) {
        guard layerCount > 0 else {
            throw .invalidArguments("layerCount must be > 0")
        }
        guard layerCount <= weights.layers.count else {
            throw .invalidArguments("layerCount \(layerCount) exceeds available recurrent layers \(weights.layers.count)")
        }
        guard weights.vocabSize > 0 else {
            throw .invalidArguments("vocabSize must be > 0")
        }

        var cpuLayers: [CPULayerWeights] = []
        cpuLayers.reserveCapacity(layerCount)
        for idx in 0..<layerCount {
            cpuLayers.append(
                CPULayerWeights(
                    rms: copyArray(from: weights.layers[idx].rms),
                    wx: copyArray(from: weights.layers[idx].Wx),
                    ws: copyArray(from: weights.layers[idx].Ws),
                    wd: copyArray(from: weights.layers[idx].Wd),
                    wo: copyArray(from: weights.layers[idx].Wo)
                )
            )
        }

        self.vocabSize = weights.vocabSize
        self.layerCount = layerCount
        self.layerWeights = cpuLayers
        self.rmsFinal = copyArray(from: weights.rmsFinal)
        self.embedding = copyArray(from: weights.embedding)
        self.classifier = weights.sharedClassifier ? [] : copyArray(from: weights.classifier)
        self.sharedClassifier = weights.sharedClassifier
        self.futureRMS = futureHead.rms
        self.futureClassifier = futureHead.classifier
        self.hasFutureHead = futureHead.hasHead
        self.stepRMSWorkspace = RMSNorm.Workspace(seqLen: 1)
        self.futureRMSWorkspace = RMSNorm.Workspace(seqLen: 1)
        self.states = Array(repeating: Array(repeating: 0, count: ModelConfig.dim), count: layerCount)
        self.currentActivation = Array(repeating: 0, count: ModelConfig.dim)
        self.hasCurrentActivation = false
    }

    public mutating func reset() throws(GenerationError) {
        for idx in states.indices {
            states[idx].withUnsafeMutableBufferPointer { ptr in
                ptr.initialize(repeating: 0)
            }
        }
        currentActivation.withUnsafeMutableBufferPointer { $0.initialize(repeating: 0) }
        hasCurrentActivation = false
    }

    public mutating func prefill(promptTokens: [UInt16]) throws(GenerationError) -> [Float] {
        guard !promptTokens.isEmpty else {
            throw .invalidArguments("promptTokens must not be empty")
        }

        var logits: [Float] = []
        for token in promptTokens {
            logits = try runStep(token: token)
        }
        return logits
    }

    public mutating func decode(nextToken: UInt16) throws(GenerationError) -> [Float] {
        try runStep(token: nextToken)
    }

    public mutating func verify(
        sequenceTokens: [UInt16],
        startIndex: Int
    ) throws(GenerationError) -> [[Float]] {
        guard !sequenceTokens.isEmpty else {
            throw .invalidArguments("sequenceTokens must not be empty")
        }
        guard startIndex >= 0, startIndex < sequenceTokens.count else {
            throw .invalidArguments("startIndex \(startIndex) must be within sequence length \(sequenceTokens.count)")
        }

        try reset()
        var outputs: [[Float]] = []
        outputs.reserveCapacity(sequenceTokens.count - startIndex)
        for (index, token) in sequenceTokens.enumerated() {
            let logits = try runStep(token: token)
            if index >= startIndex {
                outputs.append(logits)
            }
        }
        return outputs
    }

    public mutating func prefillSelectedToken(
        promptTokens: [UInt16],
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> UInt16 {
        let logits = try prefill(promptTokens: promptTokens)
        return try selectToken(from: logits, strategy: strategy)
    }

    public mutating func decodeSelectedToken(
        nextToken: UInt16,
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> UInt16 {
        let logits = try decode(nextToken: nextToken)
        return try selectToken(from: logits, strategy: strategy)
    }

    public mutating func proposeFutureToken(
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> UInt16 {
        guard hasFutureHead else {
            throw .runtimeFailure("future proposer requested without a loaded two-step student sidecar")
        }
        guard hasCurrentActivation else {
            throw .runtimeFailure("future proposer requested before any committed activation exists")
        }
        let logits = scoreHead(
            activation: currentActivation,
            rmsWeights: futureRMS,
            classifierWeights: futureClassifier,
            vocabSize: vocabSize,
            workspace: futureRMSWorkspace
        )
        return try selectToken(from: logits, strategy: strategy)
    }

    private mutating func runStep(token: UInt16) throws(GenerationError) -> [Float] {
        guard Int(token) < vocabSize else {
            throw .invalidArguments("token \(token) exceeds vocab size \(vocabSize)")
        }

        var activation = embeddingSlice(token: Int(token))
        for idx in 0..<layerCount {
            activation = Self.runLayer(
                weights: layerWeights[idx],
                state: &states[idx],
                input: activation,
                stepRMSWorkspace: stepRMSWorkspace
            )
        }
        currentActivation = activation
        hasCurrentActivation = true

        return scoreHead(
            activation: activation,
            rmsWeights: rmsFinal,
            classifierWeights: sharedClassifier ? embedding : classifier,
            vocabSize: vocabSize,
            workspace: stepRMSWorkspace
        )
    }

    private static func runLayer(
        weights: CPULayerWeights,
        state: inout [Float],
        input: [Float],
        stepRMSWorkspace: borrowing RMSNorm.Workspace
    ) -> [Float] {
        var normBuffer = Array(repeating: Float(0), count: ModelConfig.dim)
        var xMix = Array(repeating: Float(0), count: ModelConfig.dim)
        var sMix = Array(repeating: Float(0), count: ModelConfig.dim)
        var carry = Array(repeating: Float(0), count: ModelConfig.dim)
        var gate = Array(repeating: Float(0), count: ModelConfig.dim)
        var stateOut = Array(repeating: Float(0), count: ModelConfig.dim)
        var projection = Array(repeating: Float(0), count: ModelConfig.dim)
        input.withUnsafeBufferPointer { inPtr in
            normBuffer.withUnsafeMutableBufferPointer { normPtr in
                weights.rms.withUnsafeBufferPointer { rmsPtr in
                    RMSNorm.forward(
                        output: normPtr.baseAddress!,
                        input: inPtr.baseAddress!,
                        weights: rmsPtr.baseAddress!,
                        dim: ModelConfig.dim,
                        seqLen: 1,
                        workspace: stepRMSWorkspace
                    )
                }
            }
        }

        matrixVector(weights.wx, vector: normBuffer, into: &xMix)
        matrixVector(weights.ws, vector: state, into: &sMix)
        matrixVector(weights.wd, vector: state, into: &carry)

        for idx in 0..<ModelConfig.dim {
            let gatePre = xMix[idx] + sMix[idx]
            let gateValue = 1.0 / (1.0 + expf(-gatePre))
            gate[idx] = gateValue
            stateOut[idx] = xMix[idx] + carry[idx] * gateValue
        }

        matrixVector(weights.wo, vector: stateOut, into: &projection)

        var output = input
        for idx in 0..<ModelConfig.dim {
            output[idx] += projection[idx]
            state[idx] = stateOut[idx]
        }
        return output
    }

    private func embeddingSlice(token: Int) -> [Float] {
        let base = token * ModelConfig.dim
        return Array(embedding[base..<(base + ModelConfig.dim)])
    }

    private static func matrixVector(_ matrix: [Float], vector: [Float], into output: inout [Float]) {
        matrix.withUnsafeBufferPointer { matrixPtr in
            vector.withUnsafeBufferPointer { vectorPtr in
                output.withUnsafeMutableBufferPointer { outPtr in
                    BLAS.sgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        m: Int32(ModelConfig.dim),
                        n: 1,
                        k: Int32(ModelConfig.dim),
                        alpha: 1.0,
                        a: matrixPtr.baseAddress!,
                        lda: Int32(ModelConfig.dim),
                        b: vectorPtr.baseAddress!,
                        ldb: 1,
                        beta: 0.0,
                        c: outPtr.baseAddress!,
                        ldc: 1
                    )
                }
            }
        }
    }

    private func scoreHead(
        activation: [Float],
        rmsWeights: [Float],
        classifierWeights: [Float],
        vocabSize: Int,
        workspace: borrowing RMSNorm.Workspace
    ) -> [Float] {
        var normBuffer = Array(repeating: Float(0), count: ModelConfig.dim)
        var logitsBuffer = Array(repeating: Float(0), count: vocabSize)
        activation.withUnsafeBufferPointer { inPtr in
            normBuffer.withUnsafeMutableBufferPointer { normPtr in
                rmsWeights.withUnsafeBufferPointer { rmsPtr in
                    RMSNorm.forward(
                        output: normPtr.baseAddress!,
                        input: inPtr.baseAddress!,
                        weights: rmsPtr.baseAddress!,
                        dim: ModelConfig.dim,
                        seqLen: 1,
                        workspace: workspace
                    )
                }
            }
        }

        logitsBuffer.withUnsafeMutableBufferPointer { $0.initialize(repeating: 0) }
        classifierWeights.withUnsafeBufferPointer { classifierPtr in
            normBuffer.withUnsafeBufferPointer { normPtr in
                logitsBuffer.withUnsafeMutableBufferPointer { logitsPtr in
                    BLAS.sgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        m: Int32(vocabSize),
                        n: 1,
                        k: Int32(ModelConfig.dim),
                        alpha: 1.0,
                        a: classifierPtr.baseAddress!,
                        lda: Int32(ModelConfig.dim),
                        b: normPtr.baseAddress!,
                        ldb: 1,
                        beta: 0.0,
                        c: logitsPtr.baseAddress!,
                        ldc: 1
                    )
                }
            }
        }

        return logitsBuffer
    }
}

private struct CPULayerWeights {
    let rms: [Float]
    let wx: [Float]
    let ws: [Float]
    let wd: [Float]
    let wo: [Float]
}

private struct CPUFutureHead {
    let rms: [Float]
    let classifier: [Float]
    let hasHead: Bool

    static func none(vocabSize: Int) -> CPUFutureHead {
        CPUFutureHead(rms: [], classifier: Array(repeating: 0, count: vocabSize * ModelConfig.dim), hasHead: false)
    }
}

private func copyArray(from buffer: borrowing TensorBuffer) -> [Float] {
    buffer.withUnsafeBufferPointer { Array($0) }
}

@inline(__always)
private func selectToken(from logits: [Float], strategy: TokenSelectionStrategy) throws(GenerationError) -> UInt16 {
    guard !logits.isEmpty else {
        throw .runtimeFailure("logits must not be empty")
    }

    switch strategy {
    case .argmax:
        var bestIndex = 0
        var bestValue = logits[0]
        for idx in 1..<logits.count where logits[idx] > bestValue {
            bestValue = logits[idx]
            bestIndex = idx
        }
        guard bestIndex <= Int(UInt16.max) else {
            throw .runtimeFailure("best token index \(bestIndex) exceeds UInt16 range")
        }
        return UInt16(bestIndex)
    }
}

import XCTest
@testable import Espresso
import ANETypes

final class RealArtifactPipelineTests: XCTestCase {
    func test_generation_model_weight_store_tiny_roundtrip_preserves_header_and_weights() throws {
        let path = Self.makeTempBinaryPath(prefix: "tiny-generation-model")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let dim = 4
        let hidden = 8
        let nLayers = 2
        let nHeads = 1
        let seqLen = 3
        let vocab = 5

        let layers = LayerStorage<GenerationModelWeightStore.TinyLayerWeights>(count: nLayers) { _ in
            GenerationModelWeightStore.TinyLayerWeights(dim: dim, hidden: hidden)
        }
        for layer in 0..<nLayers {
            Self.fillRamp(layers[layer].Wq, base: Float(10 * layer + 1))
            Self.fillRamp(layers[layer].Wk, base: Float(10 * layer + 2))
            Self.fillRamp(layers[layer].Wv, base: Float(10 * layer + 3))
            Self.fillRamp(layers[layer].Wo, base: Float(10 * layer + 4))
            Self.fillRamp(layers[layer].W1, base: Float(10 * layer + 5))
            Self.fillRamp(layers[layer].W2, base: Float(10 * layer + 6))
            Self.fillRamp(layers[layer].W3, base: Float(10 * layer + 7))
            Self.fillRamp(layers[layer].rmsAtt, base: Float(10 * layer + 8))
            Self.fillRamp(layers[layer].rmsFfn, base: Float(10 * layer + 9))
        }

        let rmsFinal = TensorBuffer(count: dim, zeroed: false)
        let embed = TensorBuffer(count: vocab * dim, zeroed: false)
        let classifier = TensorBuffer(count: vocab * dim, zeroed: false)
        Self.fillRamp(rmsFinal, base: 100)
        Self.fillRamp(embed, base: 200)
        Self.fillRamp(classifier, base: 300)

        try GenerationModelWeightStore._saveTiny(
            path: path,
            dim: dim,
            hidden: hidden,
            nLayers: nLayers,
            nHeads: nHeads,
            seqLen: seqLen,
            vocab: vocab,
            sharedClassifier: false,
            layers: layers,
            rmsFinal: rmsFinal,
            embed: embed,
            classifier: classifier
        )

        let loaded = try GenerationModelWeightStore._loadTiny(
            path: path,
            dim: dim,
            hidden: hidden,
            nLayers: nLayers,
            nHeads: nHeads,
            seqLen: seqLen,
            vocab: vocab,
            sharedClassifier: false
        )

        XCTAssertEqual(Self.floats(loaded.layers[1].W2), Self.floats(layers[1].W2))
        XCTAssertEqual(Self.floats(loaded.layers[0].rmsFfn), Self.floats(layers[0].rmsFfn))
        XCTAssertEqual(Self.floats(loaded.rmsFinal), Self.floats(rmsFinal))
        XCTAssertEqual(Self.floats(loaded.embed), Self.floats(embed))
        XCTAssertEqual(Self.floats(loaded.classifier), Self.floats(classifier))
    }

    func test_offline_exact_acceptance_evaluator_reports_two_exact_tokens_per_pass_when_student_matches_teacher() throws {
        var teacher = StubExactModel(sequence: [10, 11, 12, 13, 14, 15])
        var student = StubFutureProposingModel(
            sequence: [10, 11, 12, 13, 14, 15],
            futureProposals: [11, 13, 15]
        )

        let trace = try OfflineExactAcceptanceEvaluator.evaluate(
            teacher: &teacher,
            student: &student,
            promptTokens: [0],
            maxNewTokens: 4,
            strategy: .argmax
        )

        XCTAssertEqual(trace.generatedTokens, [10, 11, 12, 13])
        XCTAssertEqual(trace.committedExactTokenCounts, [2, 2])
        XCTAssertEqual(trace.acceptedFutureTokenCounts, [1, 1])
        XCTAssertTrue(trace.parityMatchedAllCommittedTokens)
        XCTAssertEqual(trace.committedExactTokensPerPass, 2, accuracy: 0.0001)
        XCTAssertEqual(trace.acceptedFutureTokensPerPass, 1, accuracy: 0.0001)
    }

    func test_offline_exact_acceptance_evaluator_rejects_future_token_when_proposal_misses_exact_prefix() throws {
        var teacher = StubExactModel(sequence: [20, 21, 22, 23, 24])
        var student = StubFutureProposingModel(
            sequence: [20, 21, 22, 23, 24],
            futureProposals: [99, 99, 99]
        )

        let trace = try OfflineExactAcceptanceEvaluator.evaluate(
            teacher: &teacher,
            student: &student,
            promptTokens: [0],
            maxNewTokens: 3,
            strategy: .argmax
        )

        XCTAssertEqual(trace.generatedTokens, [20, 21, 22])
        XCTAssertEqual(trace.committedExactTokenCounts, [1, 1, 1])
        XCTAssertEqual(trace.acceptedFutureTokenCounts, [0, 0, 0])
        XCTAssertTrue(trace.parityMatchedAllCommittedTokens)
        XCTAssertEqual(trace.committedExactTokensPerPass, 1, accuracy: 0.0001)
        XCTAssertEqual(trace.acceptedFutureTokensPerPass, 0, accuracy: 0.0001)
    }

    func test_local_text_token_dataset_builder_encodes_utf8_bytes_and_separators() throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("local-text-corpus-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let a = directory.appendingPathComponent("a.txt")
        let b = directory.appendingPathComponent("b.md")
        try Data("A".utf8).write(to: a)
        try Data("BC".utf8).write(to: b)

        let tokens = try LocalTextTokenDatasetBuilder.collectTokens(
            roots: [directory.path],
            allowedExtensions: ["txt", "md"]
        )

        XCTAssertEqual(tokens, [65, LocalTextTokenDatasetBuilder.fileSeparatorToken, 66, 67])
    }

    func test_cpu_recurrent_generation_model_with_zero_trunk_uses_classifier_and_future_head() throws {
        let recurrent = Self.makeZeroTrunkRecurrentWeights(vocabSize: 4)
        let futureSidecar = try Self.makeFutureSidecar(vocabSize: 4)
        var model = try CPURecurrentGenerationModel(
            weights: recurrent,
            layerCount: 1,
            futureSidecar: futureSidecar
        )

        let current = try model.prefillSelectedToken(promptTokens: [0], strategy: .argmax)
        XCTAssertEqual(current, 1)
        let proposedFuture = try model.proposeFutureToken(strategy: .argmax)
        XCTAssertEqual(proposedFuture, 1)

        let next = try model.decodeSelectedToken(nextToken: current, strategy: .argmax)
        XCTAssertEqual(next, 2)
    }

    func test_local_bigram_artifact_builder_turns_real_token_transitions_into_matching_teacher_and_student() throws {
        let artifacts = try LocalBigramArtifactBuilder.build(tokens: [0, 1, 0, 1, 0, 2], layerCount: 1, vocabSize: 4)
        var student = try CPURecurrentGenerationModel(
            weights: artifacts.recurrentWeights,
            layerCount: 1,
            futureSidecar: artifacts.futureSidecar
        )

        let current = try student.prefillSelectedToken(promptTokens: [0], strategy: .argmax)
        XCTAssertEqual(current, 1, "Expected bigram teacher/student to prefer token 1 after token 0")

        let proposedFuture = try student.proposeFutureToken(strategy: .argmax)
        XCTAssertEqual(
            proposedFuture,
            0,
            "Future proposer should predict the teacher's exact t+2 token from the current context activation"
        )
    }

    func test_local_bigram_artifact_builder_supports_exact_two_token_offline_acceptance_on_real_token_transitions() throws {
        let artifacts = try LocalBigramArtifactBuilder.build(tokens: [0, 1, 0, 1, 0, 1, 0, 2], layerCount: 1, vocabSize: 4)
        var teacher = try CPURecurrentGenerationModel(
            weights: artifacts.recurrentWeights,
            layerCount: 1
        )
        var student = try CPURecurrentGenerationModel(
            weights: artifacts.recurrentWeights,
            layerCount: 1,
            futureSidecar: artifacts.futureSidecar
        )

        let trace = try OfflineExactAcceptanceEvaluator.evaluate(
            teacher: &teacher,
            student: &student,
            promptTokens: [0],
            maxNewTokens: 4,
            strategy: .argmax
        )

        XCTAssertEqual(trace.generatedTokens, [1, 0, 1, 0])
        XCTAssertTrue(trace.parityMatchedAllCommittedTokens)
        XCTAssertEqual(trace.committedExactTokenCounts, [2, 2])
        XCTAssertEqual(trace.acceptedFutureTokenCounts, [1, 1])
        XCTAssertEqual(trace.committedExactTokensPerPass, 2, accuracy: 0.0001)
        XCTAssertEqual(trace.acceptedFutureTokensPerPass, 1, accuracy: 0.0001)
    }

    func test_local_real_artifact_pipeline_exports_manifest_and_runs_offline_gate_from_saved_artifacts() throws {
        let datasetPath = Self.makeTempBinaryPath(prefix: "local-real-artifact-dataset")
        let prefix = NSTemporaryDirectory() + "/local-real-artifact-\(UUID().uuidString)"
        defer {
            try? FileManager.default.removeItem(atPath: datasetPath)
            try? FileManager.default.removeItem(atPath: "\(prefix).generation.bin")
            try? FileManager.default.removeItem(atPath: "\(prefix).recurrent.bin")
            try? FileManager.default.removeItem(atPath: "\(prefix).future-sidecar.bin")
            try? FileManager.default.removeItem(atPath: "\(prefix).manifest.json")
        }

        try LocalTextTokenDatasetBuilder.writeUInt16Dataset(
            tokens: [0, 1, 0, 1, 0, 1, 0, 2],
            to: datasetPath
        )

        let manifest = try LocalRealArtifactPipeline.exportLocalBigramArtifacts(
            datasetPath: datasetPath,
            prefix: prefix,
            layerCount: 1,
            vocabSize: ModelConfig.vocab
        )

        XCTAssertEqual(manifest.promptToken, 0)
        XCTAssertEqual(manifest.tokenCount, 8)
        XCTAssertEqual(manifest.layerCount, 1)
        XCTAssertTrue(FileManager.default.fileExists(atPath: manifest.generationModelPath))
        XCTAssertTrue(FileManager.default.fileExists(atPath: manifest.recurrentCheckpointPath))
        XCTAssertTrue(FileManager.default.fileExists(atPath: manifest.futureSidecarPath))
        XCTAssertTrue(FileManager.default.fileExists(atPath: manifest.manifestPath))

        let trace = try LocalRealArtifactPipeline.offlineAcceptanceGate(
            recurrentCheckpointPath: manifest.recurrentCheckpointPath,
            futureSidecarPath: manifest.futureSidecarPath,
            promptTokens: [manifest.promptToken],
            maxNewTokens: 4
        )

        XCTAssertTrue(trace.parityMatchedAllCommittedTokens)
        XCTAssertEqual(trace.committedExactTokenCounts, [2, 2])
        XCTAssertEqual(trace.acceptedFutureTokenCounts, [1, 1])
    }

    private struct StubExactModel: DirectTokenSelectingLanguageModel {
        let vocabSize: Int = 32_000
        private let sequence: [UInt16]
        private var prefillWasCalled = false
        private var nextIndex = 0

        init(sequence: [UInt16]) {
            self.sequence = sequence
        }

        mutating func reset() throws(GenerationError) {
            prefillWasCalled = false
            nextIndex = 0
        }

        mutating func prefill(promptTokens: [UInt16]) throws(GenerationError) -> [Float] {
            throw .runtimeFailure("unused in stub")
        }

        mutating func decode(nextToken: UInt16) throws(GenerationError) -> [Float] {
            throw .runtimeFailure("unused in stub")
        }

        mutating func verify(sequenceTokens: [UInt16], startIndex: Int) throws(GenerationError) -> [[Float]] {
            throw .runtimeFailure("unused in stub")
        }

        mutating func prefillSelectedToken(
            promptTokens: [UInt16],
            strategy: TokenSelectionStrategy
        ) throws(GenerationError) -> UInt16 {
            precondition(!promptTokens.isEmpty)
            prefillWasCalled = true
            nextIndex = 1
            return sequence[0]
        }

        mutating func decodeSelectedToken(
            nextToken: UInt16,
            strategy: TokenSelectionStrategy
        ) throws(GenerationError) -> UInt16 {
            precondition(prefillWasCalled)
            precondition(nextIndex < sequence.count)
            let expectedCommitted = sequence[nextIndex - 1]
            XCTAssertEqual(nextToken, expectedCommitted)
            let predicted = sequence[nextIndex]
            nextIndex += 1
            return predicted
        }
    }

    private struct StubFutureProposingModel: FutureTokenProposingLanguageModel {
        let vocabSize: Int = 32_000
        private let sequence: [UInt16]
        private let futureProposals: [UInt16]
        private var nextIndex = 0
        private var proposalIndex = 0

        init(sequence: [UInt16], futureProposals: [UInt16]) {
            self.sequence = sequence
            self.futureProposals = futureProposals
        }

        mutating func reset() throws(GenerationError) {
            nextIndex = 0
            proposalIndex = 0
        }

        mutating func prefill(promptTokens: [UInt16]) throws(GenerationError) -> [Float] {
            throw .runtimeFailure("unused in stub")
        }

        mutating func decode(nextToken: UInt16) throws(GenerationError) -> [Float] {
            throw .runtimeFailure("unused in stub")
        }

        mutating func verify(sequenceTokens: [UInt16], startIndex: Int) throws(GenerationError) -> [[Float]] {
            throw .runtimeFailure("unused in stub")
        }

        mutating func prefillSelectedToken(
            promptTokens: [UInt16],
            strategy: TokenSelectionStrategy
        ) throws(GenerationError) -> UInt16 {
            precondition(!promptTokens.isEmpty)
            nextIndex = 1
            proposalIndex = 0
            return sequence[0]
        }

        mutating func decodeSelectedToken(
            nextToken: UInt16,
            strategy: TokenSelectionStrategy
        ) throws(GenerationError) -> UInt16 {
            precondition(nextIndex < sequence.count)
            let expectedCommitted = sequence[nextIndex - 1]
            XCTAssertEqual(nextToken, expectedCommitted)
            let predicted = sequence[nextIndex]
            nextIndex += 1
            return predicted
        }

        mutating func proposeFutureToken(
            strategy: TokenSelectionStrategy
        ) throws(GenerationError) -> UInt16 {
            precondition(proposalIndex < futureProposals.count)
            let proposal = futureProposals[proposalIndex]
            proposalIndex += 1
            return proposal
        }
    }

    private static func makeTempBinaryPath(prefix: String) -> String {
        let filename = "\(prefix)-\(UUID().uuidString).bin"
        return URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(filename).path
    }

    private static func fillRamp(_ buffer: borrowing TensorBuffer, base: Float) {
        buffer.withUnsafeMutablePointer { ptr in
            for idx in 0..<buffer.count {
                ptr[idx] = base + Float(idx) * 0.25
            }
        }
    }

    private static func floats(_ buffer: borrowing TensorBuffer) -> [Float] {
        buffer.withUnsafeBufferPointer { Array($0) }
    }

    private static func makeZeroTrunkRecurrentWeights(vocabSize: Int) -> RecurrentGenerationWeights {
        let layers = LayerStorage<RWKVStyleRecurrentWeights>(count: 1) { _ in
            let weights = RWKVStyleRecurrentWeights()
            fill(weights.rms, with: 1)
            fill(weights.Wx, with: 0)
            fill(weights.Ws, with: 0)
            fill(weights.Wd, with: 0)
            fill(weights.Wo, with: 0)
            return weights
        }

        let rmsFinal = TensorBuffer(count: ModelConfig.dim, zeroed: false)
        fill(rmsFinal, with: 1)

        let embedding = TensorBuffer(count: vocabSize * ModelConfig.dim, zeroed: true)
        embedding.withUnsafeMutablePointer { ptr in
            ptr[0 * ModelConfig.dim + 0] = 1
            ptr[1 * ModelConfig.dim + 1] = 1
            ptr[2 * ModelConfig.dim + 2] = 1
            ptr[3 * ModelConfig.dim + 3] = 1
        }

        let classifier = TensorBuffer(count: vocabSize * ModelConfig.dim, zeroed: true)
        classifier.withUnsafeMutablePointer { ptr in
            ptr[1 * ModelConfig.dim + 0] = 10
            ptr[2 * ModelConfig.dim + 1] = 10
            ptr[3 * ModelConfig.dim + 2] = 10
            ptr[0 * ModelConfig.dim + 3] = 10
        }

        return RecurrentGenerationWeights(
            layers: layers,
            rmsFinal: rmsFinal,
            embedding: embedding,
            classifier: classifier,
            sharedClassifier: false,
            vocabSize: vocabSize
        )
    }

    private static func makeFutureSidecar(vocabSize: Int) throws -> TwoStepStudentSidecar {
        let contract = try TwoStepStudentContract(
            dim: ModelConfig.dim,
            vocabSize: vocabSize,
            layerCount: 1,
            teacherClassifierWasShared: false
        )
        let futureRMS = TensorBuffer(count: ModelConfig.dim, zeroed: false)
        fill(futureRMS, with: 1)
        let futureClassifier = TensorBuffer(count: vocabSize * ModelConfig.dim, zeroed: true)
        futureClassifier.withUnsafeMutablePointer { ptr in
            ptr[1 * ModelConfig.dim + 0] = 10
            ptr[2 * ModelConfig.dim + 1] = 10
            ptr[3 * ModelConfig.dim + 2] = 10
            ptr[0 * ModelConfig.dim + 3] = 10
        }
        return try TwoStepStudentSidecar(
            contract: contract,
            futureRMS: futureRMS,
            futureClassifier: futureClassifier
        )
    }

    private static func fill(_ buffer: borrowing TensorBuffer, with value: Float) {
        buffer.withUnsafeMutablePointer { ptr in
            for idx in 0..<buffer.count {
                ptr[idx] = value
            }
        }
    }
}

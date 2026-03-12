import XCTest
import CoreML
import ANETypes
@testable import Espresso

final class MultitokenProbeSupportTests: XCTestCase {
    func test_recurrent_generation_weight_store_round_trips_weights() throws {
        let weights = makeTestRecurrentWeights(layerCount: 2, sharedClassifier: false)
        let path = temporaryFilePath(named: "recurrent-weights.bin")

        try RecurrentGenerationWeightStore.save(weights, to: path)
        let loaded = try RecurrentGenerationWeightStore.load(from: path)

        XCTAssertEqual(loaded.layers.count, 2)
        XCTAssertFalse(loaded.sharedClassifier)
        XCTAssertEqual(loaded.vocabSize, ModelConfig.vocab)
        XCTAssertEqual(floats(loaded.rmsFinal, prefix: 8), floats(weights.rmsFinal, prefix: 8))
        XCTAssertEqual(floats(loaded.embedding, prefix: 16), floats(weights.embedding, prefix: 16))
        XCTAssertEqual(floats(loaded.classifier, prefix: 16), floats(weights.classifier, prefix: 16))
        XCTAssertEqual(floats(loaded.layers[0].rms, prefix: 8), floats(weights.layers[0].rms, prefix: 8))
        XCTAssertEqual(floats(loaded.layers[1].Wo, prefix: 16), floats(weights.layers[1].Wo, prefix: 16))
    }

    func test_generation_model_weight_store_loads_benchmark_head_weights_with_non_default_layer_count() throws {
        let weights = makeTestGenerationWeights(layerCount: 6, sharedClassifier: false)
        let path = temporaryFilePath(named: "generation-weights.bin")

        try GenerationModelWeightStore.save(weights, to: path)
        let loaded = try GenerationModelWeightStore.load(path: path)

        XCTAssertEqual(loaded.layers.count, 6)
        XCTAssertFalse(loaded.sharedClassifier)
        XCTAssertEqual(loaded.vocabSize, ModelConfig.vocab)
        XCTAssertEqual(floats(loaded.rmsFinal, prefix: 8), floats(weights.rmsFinal, prefix: 8))
        XCTAssertEqual(floats(loaded.embedding, prefix: 16), floats(weights.embedding, prefix: 16))
        XCTAssertEqual(floats(loaded.classifier, prefix: 16), floats(weights.classifier, prefix: 16))
        XCTAssertEqual(floats(loaded.layers[5].W2, prefix: 16), floats(weights.layers[5].W2, prefix: 16))
    }

    func test_probe_plan_requires_generation_model_for_coreml_when_recurrent_checkpoint_is_selected() throws {
        let configuration = MultitokenProbeConfiguration(
            input: .recurrentCheckpoint(path: "/tmp/recurrent.bin"),
            compareCoreML: true,
            coreMLModelPath: "benchmarks/models/transformer_6layer.mlpackage",
            generationModelPath: nil
        )

        XCTAssertThrowsError(try configuration.validated()) { error in
            XCTAssertEqual(
                error as? MultitokenProbeConfigurationError,
                .missingGenerationModelPathForCoreML
            )
        }
    }

    func test_probe_plan_uses_cpu_and_neural_engine_for_matched_coreml_runs() throws {
        let configuration = MultitokenProbeConfiguration(
            input: .echo,
            compareCoreML: true,
            coreMLModelPath: "benchmarks/models/transformer_6layer.mlpackage",
            generationModelPath: nil
        )

        let plan = try configuration.validated()

        XCTAssertEqual(plan.coreMLRequest?.computeUnits, .cpuAndNeuralEngine)
        XCTAssertEqual(plan.coreMLRequest?.headWeightsSource, .echo)
    }

    func test_probe_plan_requires_explicit_input_mode() throws {
        let configuration = MultitokenProbeConfiguration(
            input: nil,
            compareCoreML: false,
            coreMLModelPath: nil,
            generationModelPath: nil
        )

        XCTAssertThrowsError(try configuration.validated()) { error in
            XCTAssertEqual(error as? MultitokenProbeConfigurationError, .missingInput)
        }
    }

    // MARK: - GenerationMetrics.percentile tests

    func test_percentile_returns_zero_for_empty_array() {
        XCTAssertEqual(GenerationMetrics.percentile([], at: 50), 0)
        XCTAssertEqual(GenerationMetrics.percentile([], at: 95), 0)
        XCTAssertEqual(GenerationMetrics.percentile([], at: 99), 0)
    }

    func test_percentile_returns_single_value_for_singleton() {
        XCTAssertEqual(GenerationMetrics.percentile([42.0], at: 0), 42.0)
        XCTAssertEqual(GenerationMetrics.percentile([42.0], at: 50), 42.0)
        XCTAssertEqual(GenerationMetrics.percentile([42.0], at: 100), 42.0)
    }

    func test_percentile_50_matches_median_for_odd_count() {
        let values = [1.0, 3.0, 5.0, 7.0, 9.0]
        let p50 = GenerationMetrics.percentile(values, at: 50)
        let med = GenerationMetrics.median(values)
        XCTAssertEqual(p50, med, accuracy: 1e-12)
    }

    func test_percentile_p95_and_p99_on_known_distribution() {
        // 20 values: 1.0, 2.0, ..., 20.0
        let values = (1...20).map(Double.init)
        // NumPy linear percentile at 95: rank = 0.95 * 19 = 18.05
        // lower=18 (value 19.0), upper=19 (value 20.0), frac=0.05
        // expected = 19.0 + 0.05 * 1.0 = 19.05
        let p95 = GenerationMetrics.percentile(values, at: 95)
        XCTAssertEqual(p95, 19.05, accuracy: 1e-12)

        // p99: rank = 0.99 * 19 = 18.81
        // lower=18 (value 19.0), upper=19 (value 20.0), frac=0.81
        // expected = 19.0 + 0.81 * 1.0 = 19.81
        let p99 = GenerationMetrics.percentile(values, at: 99)
        XCTAssertEqual(p99, 19.81, accuracy: 1e-12)
    }

    func test_percentile_boundaries() {
        let values = [10.0, 20.0, 30.0]
        XCTAssertEqual(GenerationMetrics.percentile(values, at: 0), 10.0, accuracy: 1e-12)
        XCTAssertEqual(GenerationMetrics.percentile(values, at: 100), 30.0, accuracy: 1e-12)
    }

    func test_percentile_handles_unsorted_input() {
        let values = [9.0, 1.0, 5.0, 3.0, 7.0]
        let p50 = GenerationMetrics.percentile(values, at: 50)
        XCTAssertEqual(p50, 5.0, accuracy: 1e-12)
    }

    // MARK: - Helpers

    private func makeTestRecurrentWeights(
        layerCount: Int,
        sharedClassifier: Bool
    ) -> RecurrentGenerationWeights {
        let layers = LayerStorage<RWKVStyleRecurrentWeights>(count: layerCount) { layerIndex in
            let weights = RWKVStyleRecurrentWeights()
            fill(weights.rms, seed: 10_000 + layerIndex * 100)
            fill(weights.Wx, seed: 20_000 + layerIndex * 100)
            fill(weights.Ws, seed: 30_000 + layerIndex * 100)
            fill(weights.Wd, seed: 40_000 + layerIndex * 100)
            fill(weights.Wo, seed: 50_000 + layerIndex * 100)
            return weights
        }

        let rmsFinal = TensorBuffer(count: ModelConfig.dim, zeroed: false)
        fill(rmsFinal, seed: 60_000)

        let embedding = TensorBuffer(count: ModelConfig.vocab * ModelConfig.dim, zeroed: false)
        fill(embedding, seed: 70_000)

        let classifier = TensorBuffer(
            count: sharedClassifier ? 0 : ModelConfig.vocab * ModelConfig.dim,
            zeroed: false
        )
        if !sharedClassifier {
            fill(classifier, seed: 80_000)
        }

        return RecurrentGenerationWeights(
            layers: layers,
            rmsFinal: rmsFinal,
            embedding: embedding,
            classifier: classifier,
            sharedClassifier: sharedClassifier
        )
    }

    private func makeTestGenerationWeights(
        layerCount: Int,
        sharedClassifier: Bool
    ) -> GenerationWeights {
        let layers = LayerStorage<LayerWeights>(count: layerCount) { layerIndex in
            let weights = LayerWeights()
            fill(weights.Wq, seed: 1_000 + layerIndex * 100)
            fill(weights.Wk, seed: 2_000 + layerIndex * 100)
            fill(weights.Wv, seed: 3_000 + layerIndex * 100)
            fill(weights.Wo, seed: 4_000 + layerIndex * 100)
            fill(weights.W1, seed: 5_000 + layerIndex * 100)
            fill(weights.W2, seed: 6_000 + layerIndex * 100)
            fill(weights.W3, seed: 7_000 + layerIndex * 100)
            fill(weights.rmsAtt, seed: 8_000 + layerIndex * 100)
            fill(weights.rmsFfn, seed: 9_000 + layerIndex * 100)
            return weights
        }

        let rmsFinal = TensorBuffer(count: ModelConfig.dim, zeroed: false)
        fill(rmsFinal, seed: 10_000)

        let embedding = TensorBuffer(count: ModelConfig.vocab * ModelConfig.dim, zeroed: false)
        fill(embedding, seed: 11_000)

        let classifier = TensorBuffer(
            count: sharedClassifier ? 0 : ModelConfig.vocab * ModelConfig.dim,
            zeroed: false
        )
        if !sharedClassifier {
            fill(classifier, seed: 12_000)
        }

        return GenerationWeights(
            layers: layers,
            rmsFinal: rmsFinal,
            embedding: embedding,
            classifier: classifier,
            sharedClassifier: sharedClassifier
        )
    }

    private func fill(_ buffer: borrowing TensorBuffer, seed: Int) {
        buffer.withUnsafeMutableBufferPointer { ptr in
            for idx in ptr.indices {
                ptr[idx] = Float(seed + idx) * 0.001
            }
        }
    }

    private func floats(_ buffer: borrowing TensorBuffer, prefix: Int) -> [Float] {
        buffer.withUnsafeBufferPointer { ptr in
            Array(ptr.prefix(prefix))
        }
    }

    private func temporaryFilePath(named name: String) -> String {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("espresso-multitoken-tests", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appendingPathComponent(UUID().uuidString + "-" + name).path
    }
}

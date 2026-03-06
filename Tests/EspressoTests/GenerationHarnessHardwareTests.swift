import XCTest
import ANETypes
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

    private func benchmarkDirectEchoGeneration(
        layerCount: Int,
        promptTokens: [UInt16],
        maxNewTokens: Int,
        warmup: Int,
        iterations: Int
    ) throws -> GenerationBenchmarkSample {
        let weights = makeEchoGenerationWeights(layerCount: layerCount)
        let model = try ANEDirectGenerationModel(weights: weights, layerCount: layerCount, decodeMaxSeq: 32)
        var harness = AutoregressiveGenerationHarness(model: model, strategy: .argmax)

        var tokenLatencies: [Double] = []
        var throughput: [Double] = []
        tokenLatencies.reserveCapacity(iterations)
        throughput.reserveCapacity(iterations)

        for iter in 0..<(warmup + iterations) {
            let trace = try harness.generate(promptTokens: promptTokens, maxNewTokens: maxNewTokens)
            if iter >= warmup {
                tokenLatencies.append(trace.totalLatencyMs / Double(maxNewTokens))
                throughput.append(trace.tokensPerSecond)
            }
        }

        return GenerationBenchmarkSample(
            medianTokenMs: median(tokenLatencies),
            medianTokensPerSecond: median(throughput),
            acceptanceRate: nil
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
            acceptanceRate: median(acceptanceRates)
        )
    }
}

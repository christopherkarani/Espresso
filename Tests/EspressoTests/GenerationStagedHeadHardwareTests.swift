import XCTest
import ANETypes
@testable import Espresso

private func requireStagedHeadHardware(file: StaticString = #filePath, line: UInt = #line) throws {
    guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
        throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run staged-head hardware tests", file: file, line: line)
    }
}

private struct StagedHeadBenchmarkSample {
    let medianTokenMs: Double
    let medianTokensPerSecond: Double
    let compileTimeMs: Double
    let medianTrunkMsPerToken: Double
    let medianLogitsMsPerToken: Double
    let generatedTokens: [UInt16]
}

private func fillStagedHeadBuffer(_ buffer: borrowing TensorBuffer, value: Float) {
    buffer.withUnsafeMutableBufferPointer { ptr in
        for index in ptr.indices {
            ptr[index] = value
        }
    }
}

private func makeStagedHeadEchoRecurrentWeights(layerCount: Int) -> RecurrentGenerationWeights {
    let layers = LayerStorage<RWKVStyleRecurrentWeights>(count: layerCount) { _ in
        let weights = RWKVStyleRecurrentWeights()
        fillStagedHeadBuffer(weights.rms, value: 1)
        fillStagedHeadBuffer(weights.Wx, value: 0)
        fillStagedHeadBuffer(weights.Ws, value: 0)
        fillStagedHeadBuffer(weights.Wd, value: 0)
        fillStagedHeadBuffer(weights.Wo, value: 0)
        return weights
    }

    let rmsFinal = TensorBuffer(count: ModelConfig.dim, zeroed: false)
    fillStagedHeadBuffer(rmsFinal, value: 1)

    let embedding = TensorBuffer(count: ModelConfig.vocab * ModelConfig.dim, zeroed: true)
    embedding.withUnsafeMutablePointer { ptr in
        for dimIndex in 0..<ModelConfig.dim {
            ptr[dimIndex] = 1
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

private func stagedHeadMedian(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let middle = sorted.count / 2
    if sorted.count.isMultiple(of: 2) {
        return (sorted[middle - 1] + sorted[middle]) * 0.5
    }
    return sorted[middle]
}

final class GenerationStagedHeadHardwareTests: XCTestCase {
    func test_recurrent_generation_cpu_exact_staged_head_reports_hardware_comparison() throws {
        try requireStagedHeadHardware()

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
            outputHeadBackend: .aneRMSNormClassifier
        )
        let staged = try benchmarkRecurrentEchoGeneration(
            layerCount: 6,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations,
            outputHeadBackend: .cpuExactStaged
        )

        print(
            """
            recurrent generation fused-triplet control median=\(control.medianTokenMs) ms/token tps=\(control.medianTokensPerSecond) compile=\(control.compileTimeMs) trunk=\(control.medianTrunkMsPerToken) logits=\(control.medianLogitsMsPerToken)
            recurrent generation fused-triplet cpu-staged-exact median=\(staged.medianTokenMs) ms/token tps=\(staged.medianTokensPerSecond) compile=\(staged.compileTimeMs) trunk=\(staged.medianTrunkMsPerToken) logits=\(staged.medianLogitsMsPerToken)
            """
        )

        XCTAssertEqual(control.generatedTokens, staged.generatedTokens)
        XCTAssertEqual(control.generatedTokens, Array(repeating: 0, count: maxNewTokens))
        XCTAssertGreaterThan(control.medianTokenMs, 0)
        XCTAssertGreaterThan(staged.medianTokenMs, 0)
        XCTAssertGreaterThan(control.medianLogitsMsPerToken, 0)
        XCTAssertGreaterThan(staged.medianLogitsMsPerToken, 0)
    }

    func test_recurrent_generation_cpu_exact_clustered_head_reports_hardware_comparison() throws {
        try requireStagedHeadHardware()

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
            outputHeadBackend: .aneRMSNormClassifier
        )
        let clustered = try benchmarkRecurrentEchoGeneration(
            layerCount: 6,
            promptTokens: prompt,
            maxNewTokens: maxNewTokens,
            warmup: warmup,
            iterations: iterations,
            outputHeadBackend: .cpuExactClustered
        )

        print(
            """
            recurrent generation fused-triplet control median=\(control.medianTokenMs) ms/token tps=\(control.medianTokensPerSecond) compile=\(control.compileTimeMs) trunk=\(control.medianTrunkMsPerToken) logits=\(control.medianLogitsMsPerToken)
            recurrent generation fused-triplet cpu-clustered-exact median=\(clustered.medianTokenMs) ms/token tps=\(clustered.medianTokensPerSecond) compile=\(clustered.compileTimeMs) trunk=\(clustered.medianTrunkMsPerToken) logits=\(clustered.medianLogitsMsPerToken)
            """
        )

        XCTAssertEqual(control.generatedTokens, clustered.generatedTokens)
        XCTAssertEqual(control.generatedTokens, Array(repeating: 0, count: maxNewTokens))
        XCTAssertGreaterThan(control.medianTokenMs, 0)
        XCTAssertGreaterThan(clustered.medianTokenMs, 0)
    }

    private func benchmarkRecurrentEchoGeneration(
        layerCount: Int,
        promptTokens: [UInt16],
        maxNewTokens: Int,
        warmup: Int,
        iterations: Int,
        outputHeadBackend: GenerationOutputHeadBackend
    ) throws -> StagedHeadBenchmarkSample {
        let weights = makeStagedHeadEchoRecurrentWeights(layerCount: layerCount)
        let model = try ANERecurrentGenerationModel(
            weights: weights,
            layerCount: layerCount,
            maxSequenceTokens: 32,
            outputHeadBackend: outputHeadBackend,
            trunkBackend: .fusedThreeLayerTriplets
        )
        var harness = DirectTokenSelectionGenerationHarness(model: model, strategy: .argmax)

        var tokenLatencies: [Double] = []
        var throughput: [Double] = []
        var trunkLatencies: [Double] = []
        var logitsLatencies: [Double] = []
        var generatedTokens: [UInt16] = []
        tokenLatencies.reserveCapacity(iterations)
        throughput.reserveCapacity(iterations)
        trunkLatencies.reserveCapacity(iterations)
        logitsLatencies.reserveCapacity(iterations)

        let compileTimeMs = harness.model.performanceSnapshot.compileTimeMs

        for iteration in 0..<(warmup + iterations) {
            let trace = try harness.generate(
                promptTokens: promptTokens,
                maxNewTokens: maxNewTokens
            )
            if iteration == warmup {
                generatedTokens = trace.generatedTokens
            }
            if iteration >= warmup {
                let snapshot = harness.model.performanceSnapshot
                tokenLatencies.append(trace.totalLatencyMs / Double(maxNewTokens))
                throughput.append(trace.tokensPerSecond)
                trunkLatencies.append(snapshot.trunkLatencyMs / Double(maxNewTokens))
                logitsLatencies.append(snapshot.logitsLatencyMs / Double(maxNewTokens))
            }
        }

        return StagedHeadBenchmarkSample(
            medianTokenMs: stagedHeadMedian(tokenLatencies),
            medianTokensPerSecond: stagedHeadMedian(throughput),
            compileTimeMs: compileTimeMs,
            medianTrunkMsPerToken: stagedHeadMedian(trunkLatencies),
            medianLogitsMsPerToken: stagedHeadMedian(logitsLatencies),
            generatedTokens: generatedTokens
        )
    }
}

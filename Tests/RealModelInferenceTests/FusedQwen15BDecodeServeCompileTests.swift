import XCTest
import Foundation
import ANERuntime
import ESPRuntime
import MILGenerator
import ModelSupport
@testable import RealModelInference

/// Phase 12 serving compile: one N=1 layer with attention in-graph.
/// Do not combine with the Phase 11 dummy-block probe or the 28-layer greedy fixture.
final class FusedQwen15BDecodeServeCompileTests: XCTestCase {
    func test_n1_serving_layer0_compiles_on_real_15b_weights() throws {
        guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
            throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run ANE hardware tests")
        }
        guard let artifact = QwenParityArtifact.resolve(profile: .qwen25_15b) else {
            throw XCTSkip("Converted Qwen2.5-1.5B-Instruct.esp not found")
        }

        let bundle = try ESPRuntimeBundle.open(at: artifact.rootURL)
        let config = bundle.config
        XCTAssertEqual(config.nLayer, 28)
        XCTAssertEqual(config.dModel, 1536)
        XCTAssertEqual(config.nHead, 12)
        XCTAssertEqual(config.nKVHead, 2)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.hiddenDim, 8960)
        XCTAssertEqual(config.vocab, 151_936)
        XCTAssertEqual(config.ropeTheta, 1_000_000, accuracy: 1)

        let weights = try RealModelInferenceEngine.loadHybridLayerWeightsLlamaForTesting(
            config: config,
            weightDir: bundle.archive.weightsURL.path,
            layer: 0
        )
        XCTAssertTrue(weights.hasQKVBias, "Qwen2.5-1.5B Q/K/V bias must be present")
        XCTAssertFalse(weights.hasQKNorm, "Qwen2.5 has no QK-Norm; serving graph does not apply it")

        let kernels = try FusedHybridDecodeLayerKernelSet(
            weights: weights,
            maxSeq: 32,
            nHeads: config.nHead,
            nKVHeads: config.nKVHead,
            headDim: config.headDim
        )
        XCTAssertEqual(kernels.dim, 1536)
        XCTAssertEqual(kernels.kvDim, 256)
        XCTAssertEqual(kernels.maxSeq, 32)
        XCTAssertEqual(FusedHybridDecodeLayerGenerator.phase11MaxN, 1)
        XCTAssertEqual(FusedHybridDecodeLayerGenerator.hopsPerToken(nLayer: 28), 28)
        XCTAssertEqual(FusedHybridDecodeLayerGenerator.decodePathLabel, "fused")
    }
}

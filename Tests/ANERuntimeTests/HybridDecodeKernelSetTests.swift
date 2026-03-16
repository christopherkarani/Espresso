import XCTest
import IOSurface
import ANETypes
@testable import ANERuntime

private func requireHybridANEHardware(file: StaticString = #filePath, line: UInt = #line) throws {
    guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
        throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run ANE hardware tests", file: file, line: line)
    }
}

private func makeHybridTestLayerWeights(value: Float = 0.01) -> LayerWeights {
    let weights = LayerWeights()
    func fill(_ buf: borrowing TensorBuffer, _ value: Float) {
        buf.withUnsafeMutableBufferPointer { ptr in
            for idx in ptr.indices {
                ptr[idx] = value
            }
        }
    }

    fill(weights.Wq, value)
    fill(weights.Wk, value)
    fill(weights.Wv, value)
    fill(weights.Wo, value)
    fill(weights.W1, value)
    fill(weights.W2, value)
    fill(weights.W3, value)
    fill(weights.rmsAtt, 1.0)
    fill(weights.rmsFfn, 1.0)
    return weights
}

final class HybridDecodeKernelSetTests: XCTestCase {
    func test_compile_specs_include_qkv_projection_and_ffn_kernels() {
        let weights = makeHybridTestLayerWeights()
        let specs = HybridDecodeKernelSet.compileSpecs(weights: weights, maxSeq: 17)

        XCTAssertEqual(specs.count, 3)
        XCTAssertEqual(specs[0].kind, .decodeQKVOnly)
        XCTAssertEqual(specs[1].kind, .decodeProjection)
        XCTAssertEqual(specs[2].kind, .decodeFFN)
        XCTAssertEqual(specs[0].inputSizes, [ModelConfig.dim * 32 * 2])
        XCTAssertEqual(
            specs[0].outputSizes,
            [ModelConfig.dim * 32 * 2, ModelConfig.dim * 32 * 2, ModelConfig.dim * 32 * 2]
        )
        XCTAssertEqual(specs[1].inputSizes, [ModelConfig.dim * 32 * 4, ModelConfig.dim * 32 * 2])
        XCTAssertEqual(specs[1].outputSizes, [ModelConfig.dim * 32 * 2])
        XCTAssertEqual(specs[0].weights.count, 4)
        XCTAssertEqual(specs[1].weights.count, 1)
        XCTAssertEqual(specs[2].weights.count, 4)
        XCTAssertFalse(specs[0].milText.contains("wo.bin"))
        XCTAssertTrue(specs[1].milText.contains("wo.bin"))
        XCTAssertTrue(specs[2].milText.contains("w2.bin"))
    }

    func test_compile_specs_include_gpt2_layernorm_and_bias_weights() {
        let weights = makeHybridTestLayerWeights().withArchitecture(.gpt2)
        let specs = HybridDecodeKernelSet.compileSpecs(weights: weights, maxSeq: 17)

        XCTAssertEqual(specs.count, 3)
        XCTAssertEqual(specs[0].weights.count, 8)
        XCTAssertEqual(specs[1].weights.count, 2)
        XCTAssertEqual(specs[2].weights.count, 6)
        XCTAssertTrue(specs[0].milText.contains("rms1_beta.bin"))
        XCTAssertTrue(specs[0].milText.contains("bq.bin"))
        XCTAssertTrue(specs[1].milText.contains("bo.bin"))
        XCTAssertTrue(specs[2].milText.contains("rms2_beta.bin"))
        XCTAssertTrue(specs[2].milText.contains("b1.bin"))
        XCTAssertTrue(specs[2].milText.contains("b2.bin"))
        XCTAssertFalse(specs[2].milText.contains("w3.bin"))
    }

    func test_hybrid_decode_kernel_set_compiles_on_hardware() throws {
        try requireHybridANEHardware()
        let weights = makeHybridTestLayerWeights()
        let kernels = try HybridDecodeKernelSet(weights: weights, maxSeq: 17)

        XCTAssertEqual(kernels.maxSeq, 17)
        XCTAssertEqual(kernels.laneSpatial, 32)
    }
}

private extension LayerWeights {
    func withArchitecture(_ architecture: LayerWeightsArchitecture) -> LayerWeights {
        let rewritten = LayerWeights(architecture: architecture, dim: dim, hiddenDim: hiddenDim)

        func copy(_ src: borrowing TensorBuffer, _ dst: borrowing TensorBuffer) {
            dst.withUnsafeMutableBufferPointer { dstPtr in
                src.withUnsafeBufferPointer { srcPtr in
                    dstPtr.baseAddress?.update(from: srcPtr.baseAddress!, count: srcPtr.count)
                }
            }
        }

        func fill(_ dst: borrowing TensorBuffer, _ value: Float) {
            dst.withUnsafeMutableBufferPointer { dstPtr in
                for idx in dstPtr.indices {
                    dstPtr[idx] = value
                }
            }
        }

        copy(Wq, rewritten.Wq)
        copy(Wk, rewritten.Wk)
        copy(Wv, rewritten.Wv)
        copy(Wo, rewritten.Wo)
        copy(W1, rewritten.W1)
        copy(W2, rewritten.W2)
        copy(W3, rewritten.W3)
        copy(rmsAtt, rewritten.rmsAtt)
        copy(rmsFfn, rewritten.rmsFfn)
        copy(rmsAtt, rewritten.attentionNormBeta)
        copy(rmsFfn, rewritten.ffnNormBeta)
        fill(rewritten.bq, 0.01)
        fill(rewritten.bk, 0.01)
        fill(rewritten.bv, 0.01)
        fill(rewritten.bo, 0.01)
        fill(rewritten.b1, 0.01)
        fill(rewritten.b2, 0.01)
        return rewritten
    }
}

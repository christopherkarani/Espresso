import XCTest
import IOSurface
import ANETypes
@testable import ANERuntime

private func requireHybridANEHardware(file: StaticString = #filePath, line: UInt = #line) throws {
    guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
        throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run ANE hardware tests", file: file, line: line)
    }
}

private func makeHybridTestLayerWeights(
    dim: Int = ModelConfig.dim,
    hiddenDim: Int = ModelConfig.hidden,
    qDim: Int? = nil,
    kvDim: Int? = nil,
    hasQKVBias: Bool = false,
    normEps: Float = 1e-5,
    value: Float = 0.01
) -> LayerWeights {
    let weights = LayerWeights(
        architecture: .rmsNormSwiGLU,
        dim: dim,
        hiddenDim: hiddenDim,
        qDim: qDim,
        kvDim: kvDim,
        normEps: normEps,
        hasQKVBias: hasQKVBias
    )
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
    if hasQKVBias {
        fill(weights.bq, value)
        fill(weights.bk, value)
        fill(weights.bv, value)
    }
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

    func test_compile_specs_thread_custom_norm_epsilon_into_hybrid_decode_generators() {
        let weights = makeHybridTestLayerWeights().withNormEps(1e-6)
        let specs = HybridDecodeKernelSet.compileSpecs(weights: weights, maxSeq: 17)

        XCTAssertEqual(specs.count, 3)
        XCTAssertTrue(specs[0].milText.contains("norm_scale"))
        XCTAssertTrue(specs[2].milText.contains("norm_scale"))
        XCTAssertFalse(specs[0].milText.contains("0.00001"))
        XCTAssertFalse(specs[2].milText.contains("0.00001"))
    }

    func test_compile_specs_qwen15b_widths_include_qkv_bias_and_swiglu_ffn() {
        let weights = makeHybridTestLayerWeights(
            dim: 1536,
            hiddenDim: 8960,
            qDim: 1536,
            kvDim: 256,
            hasQKVBias: true,
            normEps: 1e-6
        )
        let specs = HybridDecodeKernelSet.compileSpecs(weights: weights, maxSeq: 17)
        let lane = HybridDecodeKernelSet.resolvedLaneSpatialForCurrentProcess()

        XCTAssertEqual(specs.count, 3)
        XCTAssertEqual(specs[0].kind, .decodeQKVOnly)
        XCTAssertEqual(specs[1].kind, .decodeProjection)
        XCTAssertEqual(specs[2].kind, .decodeFFN)
        XCTAssertEqual(specs[0].inputSizes, [1536 * lane * 2])
        XCTAssertEqual(specs[0].outputSizes, [256 * lane * 2, 1536 * lane * 2, 256 * lane * 2])
        XCTAssertEqual(specs[2].inputSizes, [1536 * lane * 2])
        XCTAssertTrue(specs[0].milText.contains("bq.bin"))
        XCTAssertTrue(specs[0].milText.contains("bk.bin"))
        XCTAssertTrue(specs[0].milText.contains("bv.bin"))
        XCTAssertTrue(specs[0].milText.contains("tensor<fp16, [1, 1536, 1, \(lane)]> x"))
        XCTAssertTrue(specs[2].milText.contains("w1.bin"))
        XCTAssertTrue(specs[2].milText.contains("w3.bin"))
        XCTAssertTrue(specs[2].milText.contains("tensor<fp16, [8960, 1536, 1, 1]>"))
        XCTAssertFalse(specs[0].milText.contains("wo.bin"))
    }

    func test_hybrid_decode_kernel_set_compiles_on_hardware() throws {
        try requireHybridANEHardware()
        let weights = makeHybridTestLayerWeights()
        let kernels = try HybridDecodeKernelSet(weights: weights, maxSeq: 17)

        XCTAssertEqual(kernels.maxSeq, 17)
        XCTAssertEqual(kernels.laneSpatial, 32)
    }

    func test_qwen15b_widths_compile_qkv_and_ffn_on_hardware() throws {
        try requireHybridANEHardware()
        let weights = makeHybridTestLayerWeights(
            dim: 1536,
            hiddenDim: 8960,
            qDim: 1536,
            kvDim: 256,
            hasQKVBias: true,
            normEps: 1e-6
        )
        let kernels = try HybridDecodeKernelSet(weights: weights, maxSeq: 17)

        XCTAssertEqual(kernels.maxSeq, 17)
        XCTAssertEqual(kernels.laneSpatial, HybridDecodeKernelSet.resolvedLaneSpatialForCurrentProcess())
        XCTAssertFalse(kernels.donorHexIDs.decodeQKVOnly.isEmpty)
        XCTAssertFalse(kernels.donorHexIDs.decodeFFN.isEmpty)
    }
}

private extension LayerWeights {
    func withArchitecture(_ architecture: LayerWeightsArchitecture) -> LayerWeights {
        let rewritten = LayerWeights(
            architecture: architecture,
            dim: dim,
            hiddenDim: hiddenDim,
            normEps: normEps
        )

        func copy(_ src: borrowing TensorBuffer, _ dst: borrowing TensorBuffer) {
            dst.withUnsafeMutableBufferPointer { dstPtr in
                src.withUnsafeBufferPointer { srcPtr in
                    guard srcPtr.count > 0 else { return }
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

    func withNormEps(_ normEps: Float) -> LayerWeights {
        let rewritten = LayerWeights(
            architecture: architecture,
            dim: dim,
            hiddenDim: hiddenDim,
            qDim: qDim,
            kvDim: kvDim,
            normEps: normEps,
            qNormDim: hasQNorm ? qNorm.count : nil,
            kNormDim: hasKNorm ? kNorm.count : nil
        )

        func copy(_ src: borrowing TensorBuffer, _ dst: borrowing TensorBuffer) {
            dst.withUnsafeMutableBufferPointer { dstPtr in
                src.withUnsafeBufferPointer { srcPtr in
                    dstPtr.baseAddress?.update(from: srcPtr.baseAddress!, count: srcPtr.count)
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
        copy(qNorm, rewritten.qNorm)
        copy(kNorm, rewritten.kNorm)
        copy(attentionNormBeta, rewritten.attentionNormBeta)
        copy(ffnNormBeta, rewritten.ffnNormBeta)
        copy(bq, rewritten.bq)
        copy(bk, rewritten.bk)
        copy(bv, rewritten.bv)
        copy(bo, rewritten.bo)
        copy(b1, rewritten.b1)
        copy(b2, rewritten.b2)
        return rewritten
    }
}

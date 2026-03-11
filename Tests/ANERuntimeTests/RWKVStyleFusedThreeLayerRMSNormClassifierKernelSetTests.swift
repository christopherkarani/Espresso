import XCTest
import ANETypes
@testable import ANERuntime

private func makeFusedThreeRWKVStyleRecurrentWeights(value: Float = 0.01) -> RWKVStyleRecurrentWeights {
    let weights = RWKVStyleRecurrentWeights()

    func fill(_ buffer: borrowing TensorBuffer, _ fillValue: Float) {
        buffer.withUnsafeMutableBufferPointer { ptr in
            for idx in ptr.indices {
                ptr[idx] = fillValue
            }
        }
    }

    fill(weights.rms, 1.0)
    fill(weights.Wx, value)
    fill(weights.Ws, value)
    fill(weights.Wd, value)
    fill(weights.Wo, value)

    return weights
}

private func makeGenerationRMSNormWeights(value: Float = 1.0) -> TensorBuffer {
    let weights = TensorBuffer(count: ModelConfig.dim, zeroed: false)
    weights.withUnsafeMutableBufferPointer { ptr in
        for idx in ptr.indices {
            ptr[idx] = value
        }
    }
    return weights
}

private func makeGenerationRMSNormClassifierWeights(value: Float = 0.01) -> TensorBuffer {
    let weights = TensorBuffer(count: ModelConfig.vocab * ModelConfig.dim, zeroed: false)
    weights.withUnsafeMutableBufferPointer { ptr in
        for idx in ptr.indices {
            ptr[idx] = value
        }
    }
    return weights
}

final class RWKVStyleFusedThreeLayerRMSNormClassifierKernelSetTests: XCTestCase {
    func test_compile_specs_expose_single_fused_three_layer_rmsnorm_classifier_kernel() {
        let laneSpatial = 32
        let weights0 = makeFusedThreeRWKVStyleRecurrentWeights(value: 0.01)
        let weights1 = makeFusedThreeRWKVStyleRecurrentWeights(value: 0.02)
        let weights2 = makeFusedThreeRWKVStyleRecurrentWeights(value: 0.03)
        let rmsFinal = makeGenerationRMSNormWeights()
        let classifier = makeGenerationRMSNormClassifierWeights()

        let specs = RWKVStyleFusedThreeLayerRMSNormClassifierKernelSet.compileSpecs(
            weights0: weights0,
            weights1: weights1,
            weights2: weights2,
            rmsFinal: rmsFinal,
            classifier: classifier,
            vocabSize: ModelConfig.vocab,
            laneSpatial: laneSpatial
        )

        let stateBytes = ModelConfig.dim * laneSpatial * 2
        let logitsBytes = ModelConfig.vocab * laneSpatial * 2

        XCTAssertEqual(specs.count, 1)
        XCTAssertEqual(specs[0].inputSizes, [stateBytes, stateBytes, stateBytes, stateBytes])
        XCTAssertEqual(specs[0].outputSizes, [stateBytes, stateBytes, stateBytes, stateBytes, logitsBytes])
        XCTAssertEqual(specs[0].weights.count, 17)
        XCTAssertTrue(specs[0].milText.contains("rwkv_rms0.bin"))
        XCTAssertTrue(specs[0].milText.contains("rwkv_rms2.bin"))
        XCTAssertTrue(specs[0].milText.contains("rms_final.bin"))
        XCTAssertTrue(specs[0].milText.contains("classifier.bin"))
    }
}

import XCTest
import ANETypes
@testable import ANERuntime

private func makeFusedRWKVStyleRecurrentWeights(value: Float = 0.01) -> RWKVStyleRecurrentWeights {
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

final class RWKVStyleFusedTwoLayerKernelSetTests: XCTestCase {
    func test_compile_specs_expose_single_fused_recurrent_step_kernel() {
        let laneSpatial = 32
        let weights0 = makeFusedRWKVStyleRecurrentWeights(value: 0.01)
        let weights1 = makeFusedRWKVStyleRecurrentWeights(value: 0.02)
        let specs = RWKVStyleFusedTwoLayerKernelSet.compileSpecs(
            weights0: weights0,
            weights1: weights1,
            laneSpatial: laneSpatial
        )
        let bytes = ModelConfig.dim * laneSpatial * 2

        XCTAssertEqual(specs.count, 1)
        XCTAssertEqual(specs[0].inputSizes, [bytes, bytes, bytes])
        XCTAssertEqual(specs[0].outputSizes, [bytes, bytes, bytes])
        XCTAssertTrue(specs[0].milText.contains("rwkv_rms0.bin"))
        XCTAssertTrue(specs[0].milText.contains("wx0.bin"))
        XCTAssertTrue(specs[0].milText.contains("rwkv_rms1.bin"))
        XCTAssertTrue(specs[0].milText.contains("wx1.bin"))
    }
}

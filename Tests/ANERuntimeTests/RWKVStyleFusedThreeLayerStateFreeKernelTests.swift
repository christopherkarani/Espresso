import XCTest
import ANETypes
@testable import ANERuntime

private func makeFusedThreeStateFreeRWKVStyleRecurrentWeights(
    wxValue: Float = 1,
    woValue: Float = 0.25
) -> RWKVStyleRecurrentWeights {
    let weights = RWKVStyleRecurrentWeights()
    weights.rms.withUnsafeMutableBufferPointer { ptr in
        for idx in ptr.indices {
            ptr[idx] = 1
        }
    }
    weights.Wx.withUnsafeMutableBufferPointer { ptr in
        for idx in ptr.indices {
            ptr[idx] = 0
        }
        for idx in stride(from: 0, to: ptr.count, by: ModelConfig.dim + 1) {
            ptr[idx] = wxValue
        }
    }
    weights.Ws.withUnsafeMutableBufferPointer { ptr in
        for idx in ptr.indices {
            ptr[idx] = 0
        }
    }
    weights.Wd.withUnsafeMutableBufferPointer { ptr in
        for idx in ptr.indices {
            ptr[idx] = 0
        }
    }
    weights.Wo.withUnsafeMutableBufferPointer { ptr in
        for idx in ptr.indices {
            ptr[idx] = 0
        }
        for idx in stride(from: 0, to: ptr.count, by: ModelConfig.dim + 1) {
            ptr[idx] = woValue
        }
    }
    return weights
}

final class RWKVStyleFusedThreeLayerStateFreeKernelSetTests: XCTestCase {
    func test_compile_specs_expose_single_fused_three_layer_state_free_recurrent_step_kernel() {
        let laneSpatial = 32
        let weights0 = makeFusedThreeStateFreeRWKVStyleRecurrentWeights()
        let weights1 = makeFusedThreeStateFreeRWKVStyleRecurrentWeights()
        let weights2 = makeFusedThreeStateFreeRWKVStyleRecurrentWeights()
        let specs = RWKVStyleFusedThreeLayerStateFreeKernelSet.compileSpecs(
            weights0: weights0,
            weights1: weights1,
            weights2: weights2,
            laneSpatial: laneSpatial
        )
        let bytes = ModelConfig.dim * laneSpatial * 2

        XCTAssertEqual(specs.count, 1)
        XCTAssertEqual(specs[0].inputSizes, [bytes])
        XCTAssertEqual(specs[0].outputSizes, [bytes])
        XCTAssertTrue(specs[0].milText.contains("rwkv_rms0.bin"))
        XCTAssertTrue(specs[0].milText.contains("wx1.bin"))
        XCTAssertTrue(specs[0].milText.contains("wo2.bin"))
        XCTAssertFalse(specs[0].milText.contains("ws0.bin"))
        XCTAssertFalse(specs[0].milText.contains("wd1.bin"))
    }
}

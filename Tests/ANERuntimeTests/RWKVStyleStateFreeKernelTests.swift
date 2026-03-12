import XCTest
import ANETypes
@testable import ANERuntime

private func makeStateFreeRWKVStyleRecurrentWeights(wxValue: Float = 1, woValue: Float = 0.25) -> RWKVStyleRecurrentWeights {
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

final class RWKVStyleStateFreeKernelSetTests: XCTestCase {
    func test_compile_specs_expose_single_state_free_recurrent_step_kernel() {
        let laneSpatial = 32
        let weights = makeStateFreeRWKVStyleRecurrentWeights()
        let specs = RWKVStyleStateFreeKernelSet.compileSpecs(weights: weights, laneSpatial: laneSpatial)
        let bytes = ModelConfig.dim * laneSpatial * 2

        XCTAssertEqual(specs.count, 1)
        XCTAssertEqual(specs[0].inputSizes, [bytes])
        XCTAssertEqual(specs[0].outputSizes, [bytes])
        XCTAssertTrue(specs[0].milText.contains("rwkv_rms.bin"))
        XCTAssertTrue(specs[0].milText.contains("wx.bin"))
        XCTAssertTrue(specs[0].milText.contains("wo.bin"))
        XCTAssertFalse(specs[0].milText.contains("ws.bin"))
        XCTAssertFalse(specs[0].milText.contains("wd.bin"))
    }
}

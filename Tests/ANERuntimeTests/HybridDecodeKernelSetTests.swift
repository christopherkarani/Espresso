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
    func test_compile_specs_include_qkv_only_and_ffn_kernels() {
        let weights = makeHybridTestLayerWeights()
        let specs = HybridDecodeKernelSet.compileSpecs(weights: weights, maxSeq: 17)

        XCTAssertEqual(specs.count, 2)
        XCTAssertEqual(specs[0].kind, .decodeQKVOnly)
        XCTAssertEqual(specs[1].kind, .decodeFFN)
        XCTAssertEqual(specs[0].inputSizes, [ModelConfig.dim * 32 * 2])
        XCTAssertEqual(
            specs[0].outputSizes,
            [ModelConfig.dim * 32 * 2, ModelConfig.dim * 32 * 2, ModelConfig.dim * 32 * 2]
        )
        XCTAssertEqual(specs[0].weights.count, 4)
        XCTAssertEqual(specs[1].weights.count, 4)
        XCTAssertFalse(specs[0].milText.contains("wo.bin"))
        XCTAssertTrue(specs[1].milText.contains("w2.bin"))
    }

    func test_hybrid_decode_kernel_set_compiles_on_hardware() throws {
        try requireHybridANEHardware()
        let weights = makeHybridTestLayerWeights()
        let kernels = try HybridDecodeKernelSet(weights: weights, maxSeq: 17)

        XCTAssertEqual(kernels.maxSeq, 17)
        XCTAssertEqual(kernels.laneSpatial, 32)
    }
}

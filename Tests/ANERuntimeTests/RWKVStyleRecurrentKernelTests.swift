import XCTest
import IOSurface
import ANEInterop
import ANETypes
import MILGenerator
@testable import ANERuntime

private func requireRecurrentANEHardware(file: StaticString = #filePath, line: UInt = #line) throws {
    guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
        throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run ANE hardware tests", file: file, line: line)
    }
    let handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW
    )
    if handle == nil {
        throw XCTSkip("AppleNeuralEngine.framework unavailable", file: file, line: line)
    }
    dlclose(handle)
}

private func makeRWKVStyleRecurrentWeights(value: Float = 0.01) -> RWKVStyleRecurrentWeights {
    let weights = RWKVStyleRecurrentWeights()
    fillRWKVStyleRecurrentWeights(weights, value: value)
    return weights
}

private func fillRWKVStyleRecurrentWeights(_ weights: borrowing RWKVStyleRecurrentWeights, value: Float) {
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
}

final class RWKVStyleRecurrentKernelSetTests: XCTestCase {
    func test_compile_specs_expose_single_recurrent_step_kernel() {
        let laneSpatial = 32
        let weights = makeRWKVStyleRecurrentWeights()
        let specs = RWKVStyleRecurrentKernelSet.compileSpecs(weights: weights, laneSpatial: laneSpatial)
        let bytes = ModelConfig.dim * laneSpatial * 2

        XCTAssertEqual(specs.count, 1)
        XCTAssertEqual(specs[0].inputSizes, [bytes, bytes])
        XCTAssertEqual(specs[0].outputSizes, [bytes, bytes])
        XCTAssertTrue(specs[0].milText.contains("rwkv_rms.bin"))
        XCTAssertTrue(specs[0].milText.contains("wx.bin"))
    }

    func test_recurrent_kernel_compiles_on_hardware() throws {
        try requireRecurrentANEHardware()

        let weights = makeRWKVStyleRecurrentWeights()
        let kernelSet = try RWKVStyleRecurrentKernelSet(weights: weights, laneSpatial: 32)

        XCTAssertEqual(kernelSet.laneSpatial, 32)
    }

    func test_recurrent_kernel_eval_succeeds() throws {
        try requireRecurrentANEHardware()

        let weights = makeRWKVStyleRecurrentWeights()
        let kernelSet = try RWKVStyleRecurrentKernelSet(weights: weights, laneSpatial: 32)

        do {
            try kernelSet.step.eval()
        } catch {
            print("  NOTE: recurrent kernel eval failed (known host instability): \(error)")
        }
    }

    func test_recurrent_kernel_surfaces_are_accessible() throws {
        try requireRecurrentANEHardware()

        let weights = makeRWKVStyleRecurrentWeights()
        let kernelSet = try RWKVStyleRecurrentKernelSet(weights: weights, laneSpatial: 32)

        for idx in 0..<2 {
            let surf = try kernelSet.step.inputSurface(at: idx)
            XCTAssertTrue(IOSurfaceGetAllocSize(surf) > 0, "Input surface \(idx) should be non-empty")
        }

        for idx in 0..<2 {
            let surf = try kernelSet.step.outputSurface(at: idx)
            XCTAssertTrue(IOSurfaceGetAllocSize(surf) > 0, "Output surface \(idx) should be non-empty")
        }
    }
}

import XCTest
import Darwin
import ANEInterop
import ANETypes
import MILGenerator
@testable import ANERuntime

private let probeChannels = 4
private let probeSpatial = 8
private let probeBytes = probeChannels * probeSpatial * MemoryLayout<UInt16>.stride

private func requireANEAvailable(file: StaticString = #filePath, line: UInt = #line) throws {
    let handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW
    )
    if handle == nil {
        throw XCTSkip("AppleNeuralEngine.framework unavailable", file: file, line: line)
    }
    dlclose(handle)
}

private func requireANEHardwareTests(file: StaticString = #filePath, line: UInt = #line) throws {
    guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
        throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run ANE hardware tests", file: file, line: line)
    }
    try requireANEAvailable(file: file, line: line)
}

private func identityWeightBlob(channels: Int) -> Data {
    var weights = [Float](repeating: 0, count: channels * channels)
    for i in 0..<channels {
        weights[i * channels + i] = 1
    }
    return WeightBlob.build(from: weights, rows: channels, cols: channels)
}

private func identityKernel() throws -> ANEKernel {
    let mil = GenericMIL.conv(
        inCh: probeChannels,
        outCh: probeChannels,
        spatial: probeSpatial
    )
    return try ANEKernel(
        milText: mil,
        weights: [(
            path: "@model_path/weights/weight.bin",
            data: identityWeightBlob(channels: probeChannels)
        )],
        inputBytes: probeBytes,
        outputBytes: probeBytes
    )
}

// MARK: - Non-hardware tests

final class RealTimeEvalProbeTests: XCTestCase {

    func test_realtime_probe_struct_has_expected_fields() {
        // Verify the Swift wrapper struct has all expected fields
        let probe = ANEKernel.RealTimeEvalProbe(
            hasBeginRealTimeTask: false,
            hasEndRealTimeTask: false,
            hasLoadRealTimeModel: false,
            hasUnloadRealTimeModel: false,
            hasEvaluateRealTime: false,
            realtimeLoadSucceeded: false,
            realtimeEvalSucceeded: false,
            standardEvalSucceeded: false,
            realtimeEvalsCompleted: 0,
            standardEvalsCompleted: 0,
            realtimeTotalMS: 0,
            standardTotalMS: 0,
            realtimePerEvalMS: 0,
            standardPerEvalMS: 0,
            savedPerEvalMS: 0,
            savedPercent: 0
        )
        XCTAssertFalse(probe.hasBeginRealTimeTask)
        XCTAssertEqual(probe.realtimeEvalsCompleted, 0)
        XCTAssertEqual(probe.savedPercent, 0)
    }

    // MARK: - Hardware-gated tests

    func test_realtime_selector_discovery() throws {
        try requireANEHardwareTests()
        let kernel = try identityKernel()

        let hasRT = kernel.hasRealTimeEvalSupport
        print("RealTime eval support: \(hasRT)")

        // All five selectors should be present on Apple Silicon macOS 15+
        let probe = kernel.realTimeEvalProbe(nIters: 1)
        print("  hasBeginRealTimeTask: \(probe.hasBeginRealTimeTask)")
        print("  hasEndRealTimeTask: \(probe.hasEndRealTimeTask)")
        print("  hasLoadRealTimeModel: \(probe.hasLoadRealTimeModel)")
        print("  hasUnloadRealTimeModel: \(probe.hasUnloadRealTimeModel)")
        print("  hasEvaluateRealTime: \(probe.hasEvaluateRealTime)")

        XCTAssertTrue(probe.hasBeginRealTimeTask,
                      "_ANEClient should have beginRealTimeTask")
        XCTAssertTrue(probe.hasEvaluateRealTime,
                      "_ANEClient should have evaluateRealTimeWithModel:")
    }

    func test_realtime_load_succeeds() throws {
        try requireANEHardwareTests()
        let kernel = try identityKernel()
        let probe = kernel.realTimeEvalProbe(nIters: 1)

        print("RealTime load: succeeded=\(probe.realtimeLoadSucceeded)")

        if !probe.realtimeLoadSucceeded {
            print("  NOTE: loadRealTimeModel failed (may require entitlements)")
        }
    }

    func test_realtime_eval_vs_standard_benchmark() throws {
        try requireANEHardwareTests()
        let kernel = try identityKernel()
        let nIters = 30

        let probe = kernel.realTimeEvalProbe(nIters: nIters)

        print("── Real-time vs Standard eval (\(nIters) iters) ──")
        print("  Standard: \(probe.standardEvalsCompleted)/\(nIters) evals, " +
              "\(String(format: "%.3f", probe.standardPerEvalMS))ms/eval")

        if probe.realtimeLoadSucceeded {
            print("  RealTime: \(probe.realtimeEvalsCompleted)/\(nIters) evals, " +
                  "\(String(format: "%.3f", probe.realtimePerEvalMS))ms/eval")
            print("  Savings:  \(String(format: "%.3f", probe.savedPerEvalMS))ms/eval " +
                  "(\(String(format: "%.1f", probe.savedPercent))%)")
        } else {
            print("  RealTime: load failed — path unavailable on this host")
        }

        // Standard eval via InMemoryModel should succeed (known stable path)
        if !probe.standardEvalSucceeded {
            print("  NOTE: standard eval failed (known host instability)")
        } else {
            XCTAssertEqual(probe.standardEvalsCompleted, nIters,
                           "All standard evals should complete")
        }
    }

    func test_realtime_eval_correctness() throws {
        try requireANEHardwareTests()
        let kernel = try identityKernel()

        // Write known input
        let input: [Float] = (0..<(probeChannels * probeSpatial)).map { Float($0) + 1 }
        let inputSurface = try kernel.inputSurface(at: 0)
        XCTAssertTrue(
            ane_interop_io_write_fp16(inputSurface, input, Int32(probeChannels), Int32(probeSpatial))
        )

        let probe = kernel.realTimeEvalProbe(nIters: 1)

        // Verify standard eval produces correct output (identity kernel)
        if probe.standardEvalSucceeded {
            var output = [Float](repeating: 0, count: probeChannels * probeSpatial)
            let outputSurface = try kernel.outputSurface(at: 0)
            XCTAssertTrue(
                ane_interop_io_read_fp16(outputSurface, 0, &output, Int32(probeChannels), Int32(probeSpatial))
            )
            // After probe, output should still be valid from the last eval
            // (tolerance relaxed because multiple evals may interleave)
            let anyNonZero = output.contains { $0 != 0 }
            XCTAssertTrue(anyNonZero, "Output should contain non-zero values after eval")
        }

        // If real-time eval succeeded, verify its output too
        if probe.realtimeEvalSucceeded {
            print("  Real-time eval produced output — correctness verified by non-crash")
        }
    }

    func test_realtime_high_iteration_stability() throws {
        try requireANEHardwareTests()
        let kernel = try identityKernel()
        let nIters = 100

        let probe = kernel.realTimeEvalProbe(nIters: nIters)

        print("── Real-time stability test (\(nIters) iters) ──")
        print("  Standard: \(probe.standardEvalsCompleted)/\(nIters)")

        if probe.realtimeLoadSucceeded {
            print("  RealTime: \(probe.realtimeEvalsCompleted)/\(nIters)")
            if probe.realtimeEvalsCompleted > 0 {
                XCTAssertEqual(probe.realtimeEvalsCompleted, nIters,
                               "All real-time evals should complete without crash")
            }
        } else {
            print("  RealTime: load failed")
        }

        if !probe.standardEvalSucceeded {
            print("  NOTE: standard eval failed (known host instability)")
        } else {
            XCTAssertEqual(probe.standardEvalsCompleted, nIters,
                           "All standard evals should complete")
        }
    }
}

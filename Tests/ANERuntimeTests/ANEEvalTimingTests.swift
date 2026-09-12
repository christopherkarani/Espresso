import XCTest
@testable import ANERuntime

final class ANEEvalTimingTests: XCTestCase {
    func test_recorder_stores_wall_and_hw_from_shipped_record_call() {
        let recorder = ANEEvalTimingRecorder()
        recorder.record(wallNanoseconds: 5_000, hardwareNanoseconds: 3_000)
        XCTAssertEqual(recorder.lastWallNanoseconds, 5_000)
        XCTAssertEqual(recorder.lastWallMicroseconds, 5.0, accuracy: 1e-9)
        XCTAssertEqual(recorder.lastHWExecutionTimeNS, 3_000)
        XCTAssertGreaterThanOrEqual(recorder.lastWallNanoseconds, recorder.lastHWExecutionTimeNS)
    }

    func test_recorder_allows_zero_hw_when_perf_stats_off() {
        let recorder = ANEEvalTimingRecorder()
        recorder.record(wallNanoseconds: 2_000, hardwareNanoseconds: 0)
        XCTAssertEqual(recorder.lastWallMicroseconds, 2.0, accuracy: 1e-9)
        XCTAssertEqual(recorder.lastHWExecutionTimeNS, 0)
    }

    // Regression: with `evalTiming` default-initialized, Swift 6.2.4 -Onone
    // released an uninitialized slot on this early throw (SIGSEGV in init).
    func test_kernel_init_throws_cleanly_before_compile_on_empty_donor() {
        for _ in 0..<64 {
            do throws(ANEError) {
                _ = try ANEKernel(
                    milText: "program(1.0)",
                    weights: [],
                    inputSizes: [4],
                    outputSizes: [4],
                    donorHexId: ""
                )
                XCTFail("expected invalidArguments")
            } catch {
                guard case .invalidArguments = error else {
                    return XCTFail("unexpected error \(error)")
                }
            }
        }
    }
}

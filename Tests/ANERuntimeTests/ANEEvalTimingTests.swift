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
}

import XCTest
import Darwin
import ANEInterop

final class ANEChainingProbeConfigTests: XCTestCase {
    private func withEnvironmentValue(
        key: String,
        value: String?,
        file: StaticString = #filePath,
        line: UInt = #line,
        _ body: () -> Void
    ) {
        let previous = getenv(key)
        let previousValue = previous.map { String(cString: $0) }
        if let value {
            XCTAssertEqual(setenv(key, value, 1), 0, file: file, line: line)
        } else {
            XCTAssertEqual(unsetenv(key), 0, file: file, line: line)
        }
        defer {
            if let previousValue {
                XCTAssertEqual(setenv(key, previousValue, 1), 0, file: file, line: line)
            } else {
                XCTAssertEqual(unsetenv(key), 0, file: file, line: line)
            }
        }
        body()
    }

    func test_chaining_probe_stats_surface_mode_defaults_to_scratch() {
        withEnvironmentValue(key: "ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE", value: nil) {
            XCTAssertEqual(
                ane_interop_chaining_probe_stats_surface_mode(),
                ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE_SCRATCH
            )
        }
    }

    func test_chaining_probe_stats_surface_mode_reads_null() {
        withEnvironmentValue(key: "ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE", value: "null") {
            XCTAssertEqual(
                ane_interop_chaining_probe_stats_surface_mode(),
                ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE_NULL
            )
        }
    }

    func test_chaining_probe_stats_surface_mode_reads_output0() {
        withEnvironmentValue(key: "ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE", value: "output0") {
            XCTAssertEqual(
                ane_interop_chaining_probe_stats_surface_mode(),
                ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE_OUTPUT0
            )
        }
    }

    func test_chaining_probe_stats_surface_mode_reads_scratch() {
        withEnvironmentValue(key: "ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE", value: "scratch") {
            XCTAssertEqual(
                ane_interop_chaining_probe_stats_surface_mode(),
                ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE_SCRATCH
            )
        }
    }

    func test_chaining_probe_stats_surface_mode_rejects_unknown_values() {
        withEnvironmentValue(key: "ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE", value: "bogus") {
            XCTAssertEqual(
                ane_interop_chaining_probe_stats_surface_mode(),
                ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE_SCRATCH
            )
        }
    }
}

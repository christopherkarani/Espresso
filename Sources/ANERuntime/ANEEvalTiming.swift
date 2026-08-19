import Foundation

/// Per-`eval()` wall time next to driver `lastHWExecutionTimeNS`.
///
/// `lastHWExecutionTimeNS` is 0 when perf stats are off or unsupported.
public final class ANEEvalTimingRecorder: @unchecked Sendable {
    public private(set) var lastWallNanoseconds: UInt64 = 0
    public private(set) var lastWallMicroseconds: Double = 0
    public private(set) var lastHWExecutionTimeNS: UInt64 = 0

    public init() {}

    public func record(wallNanoseconds: UInt64, hardwareNanoseconds: UInt64) {
        lastWallNanoseconds = wallNanoseconds
        lastWallMicroseconds = Double(wallNanoseconds) / 1_000.0
        lastHWExecutionTimeNS = hardwareNanoseconds
    }
}

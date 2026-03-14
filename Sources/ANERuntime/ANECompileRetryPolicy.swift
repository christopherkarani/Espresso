import ANEInterop
import Darwin
import Foundation

enum ANECompileRetryPolicy {
    static let maxAttempts = 5
    static let initialDelayMicroseconds: useconds_t = 100_000
    static let maxDelayMicroseconds: useconds_t = 1_000_000

    static func shouldRetry(lastCompileError: Int32, attemptIndex: Int) -> Bool {
        guard attemptIndex >= 0 else { return false }
        guard lastCompileError == ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE else {
            return false
        }
        return attemptIndex + 1 < maxAttempts
    }

    static func delayMicroseconds(afterFailedAttempt attemptIndex: Int) -> useconds_t {
        guard attemptIndex > 0 else {
            return initialDelayMicroseconds
        }

        var delay = initialDelayMicroseconds
        for _ in 0..<attemptIndex {
            if delay >= maxDelayMicroseconds {
                return maxDelayMicroseconds
            }
            delay = min(delay * 2, maxDelayMicroseconds)
        }
        return delay
    }

    static func sleepAfterFailedAttempt(_ attemptIndex: Int) {
        usleep(delayMicroseconds(afterFailedAttempt: attemptIndex))
    }

    static func retryNotice(afterFailedAttempt attemptIndex: Int) -> String {
        let nextAttempt = min(attemptIndex + 2, maxAttempts)
        return "ANE compile retrying (\(nextAttempt)/\(maxAttempts)) after transient compiler failure"
    }
}

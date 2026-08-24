import Foundation

/// Prefill + first-token wall times for generate/chat.
///
/// `firstTokenLatencyMs` is TTFT including prefill: prompt-submit through the
/// first emitted token. Compile time is not included. Decode tok/s keeps a
/// separate post-prefill clock.
public struct GenerateTiming: Sendable, Equatable {
    public let prefillMs: Double
    public let firstTokenLatencyMs: Double

    public var ttftIncludingPrefillMs: Double { firstTokenLatencyMs }

    public init(prefillMs: Double, firstTokenLatencyIncludingPrefillMs: Double) {
        self.prefillMs = prefillMs
        self.firstTokenLatencyMs = max(firstTokenLatencyIncludingPrefillMs, prefillMs)
    }

    public static func milliseconds(from nanoseconds: UInt64) -> Double {
        Double(nanoseconds) / 1_000_000
    }
}

/// Wall clock from prompt submit, with an explicit prefill-end mark.
public struct GenerateClock: Sendable {
    public let submitNS: UInt64
    public private(set) var prefillEndNS: UInt64
    private let now: @Sendable () -> UInt64

    public init(now: @escaping @Sendable () -> UInt64 = { DispatchTime.now().uptimeNanoseconds }) {
        self.now = now
        let t = now()
        self.submitNS = t
        self.prefillEndNS = t
    }

    public init(submitNS: UInt64, now: @escaping @Sendable () -> UInt64 = { DispatchTime.now().uptimeNanoseconds }) {
        self.now = now
        self.submitNS = submitNS
        self.prefillEndNS = submitNS
    }

    public var decodeStartNS: UInt64 { prefillEndNS }

    public mutating func markPrefillEnd() {
        markPrefillEnd(at: now())
    }

    public mutating func markPrefillEnd(at ns: UInt64) {
        prefillEndNS = ns
    }

    public func prefillMs() -> Double {
        GenerateTiming.milliseconds(from: prefillEndNS - submitNS)
    }

    public func ttftIncludingPrefillMs(at emissionNS: UInt64) -> Double {
        let ttft = GenerateTiming.milliseconds(from: emissionNS - submitNS)
        return max(ttft, prefillMs())
    }

    public func timing(firstTokenNS: UInt64) -> GenerateTiming {
        GenerateTiming(
            prefillMs: prefillMs(),
            firstTokenLatencyIncludingPrefillMs: ttftIncludingPrefillMs(at: firstTokenNS)
        )
    }
}

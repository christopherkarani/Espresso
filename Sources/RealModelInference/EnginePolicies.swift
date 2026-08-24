import Foundation

/// Everything the process environment steers in this engine, resolved once
/// when an engine is built.
///
/// The interface used to be "three build parameters plus ~78 undeclared
/// environment variables read lazily mid-flight": callers had to `setenv`
/// before calling, hot decode loops consulted `ProcessInfo` per token, and
/// tests could only steer behavior by mutating process-global state. Now one
/// dictionary crosses the seam at build time and nothing inside reads live
/// process state while serving.
///
/// Typed fields exist for knobs consulted on hot or failure paths; every
/// other knob flows through the frozen snapshot into the pure
/// `(config:environment:)` policy functions, which stay injectable for tests.
struct EnginePolicies: Sendable {
    /// Frozen snapshot. Policy statics (`selectTrunk`, `resolvedTrunk`,
    /// `prefersCPUDecodeAttention`, ...) receive this instead of live
    /// `ProcessInfo`.
    let environment: [String: String]

    /// `ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK=1`: hybrid fast-path
    /// failures become hard errors instead of silent fallbacks.
    let disableHybridFallback: Bool

    /// `ESPRESSO_REALMODEL_DEBUG_HYBRID_CACHE=1`: print hybrid surface geometry.
    let debugHybridCacheDumps: Bool

    /// `ESPRESSO_DUMP_LM_HEAD_HIDDEN=<path>`: stream hidden states to disk.
    let lmHeadHiddenDumpPath: String?

    init(environment: [String: String]) {
        self.environment = environment
        self.disableHybridFallback = environment["ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK"] == "1"
        self.debugHybridCacheDumps = environment["ESPRESSO_REALMODEL_DEBUG_HYBRID_CACHE"] == "1"
        self.lmHeadHiddenDumpPath = environment["ESPRESSO_DUMP_LM_HEAD_HIDDEN"]
    }

    static func resolve(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> EnginePolicies {
        EnginePolicies(environment: environment)
    }
}

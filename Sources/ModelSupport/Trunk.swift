import Foundation

/// Which decode-step implementation a llama serving session uses for each token.
///
/// Matches the domain vocabulary in `CONTEXT.md`:
/// - ``fusedHybrid`` — one ANE program per transformer layer, attention included
/// - ``splitHybrid`` — ANE QKV, host attention, ANE FFN
/// - ``exactCPU`` — transformer layer on the CPU (also the Qwen oracle)
///
/// Telemetry / CLI labels use the stable `rawValue` strings (`fused`, `hybrid`, `exact_cpu`).
/// Artifact metadata still uses ``MultiModelConfig/PreferredDecodePath``, which only
/// distinguishes the hybrid family from exact-CPU; fused vs split is runtime policy.
public enum Trunk: String, Sendable, Equatable, CaseIterable {
    case fusedHybrid = "fused"
    case splitHybrid = "hybrid"
    case exactCPU = "exact_cpu"

    /// ANE hybrid trunks (fused or split), as opposed to exact-CPU.
    public var isHybrid: Bool {
        switch self {
        case .fusedHybrid, .splitHybrid:
            return true
        case .exactCPU:
            return false
        }
    }

    /// Stable label for `decode_path=` stderr / JSON contracts.
    public var telemetryLabel: String { rawValue }

    /// Parse a telemetry or CLI path label into a trunk.
    public static func parseTelemetryLabel(_ raw: String) throws -> Trunk {
        let normalized = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard let value = Trunk(rawValue: normalized) else {
            throw ParseError.unsupported(raw)
        }
        return value
    }

    public enum ParseError: Error, Sendable, Equatable, LocalizedError {
        case unsupported(String)

        public var errorDescription: String? {
            switch self {
            case let .unsupported(raw):
                return "Unsupported trunk label: \(raw) (expected \"fused\", \"hybrid\", or \"exact_cpu\")"
            }
        }
    }
}

extension MultiModelConfig.PreferredDecodePath {
    /// Coarse artifact preference maps onto the exact-CPU trunk or the hybrid family.
    /// Fused vs split is chosen later by runtime policy, not by metadata.
    public var exactCPUTrunk: Trunk? {
        self == .exactCPU ? .exactCPU : nil
    }

    public var prefersHybridFamily: Bool {
        self == .hybrid
    }
}

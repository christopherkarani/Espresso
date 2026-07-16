import ModelSupport

/// Selects between ANE and CPU exact classifier based on the model's
/// vocabulary size and embedding dimension.
///
/// The ANE classifier is faster but requires the full weight matrix to fit
/// in the Neural Engine's SRAM. When the weight matrix exceeds the SRAM
/// element limit (16M elements = 32MB fp16), we fall back to an exact CPU
/// path. Llama-family artifacts without an exact float32 sidecar can use the
/// FP16-tiled classifier directly over the packed blob weights.
public enum ClassifierStrategy: Sendable, Equatable {
    /// Use the ANE lane-packed classifier (fused RMSNorm + classifier head).
    case ane
    /// Use the exact block-pruned FP32 classifier on the CPU.
    case cpuPartitionedFP32
    /// Use the exact FP16-tiled classifier on the CPU.
    case cpuFP16Tiled

    /// Conservative SRAM element limit: 16M elements (32MB fp16).
    /// Leaves headroom for activations and intermediate buffers.
    private static let aneSRAMElementLimit: Int = 16_000_000

    public var usesANEClassifier: Bool {
        self == .ane
    }

    public var usesCPUExactClassifier: Bool {
        !usesANEClassifier
    }

    public var exactHeadBackendLabel: String {
        switch self {
        case .ane:
            return "ane_classifier"
        case .cpuPartitionedFP32:
            return "cpu_partitioned_fp32"
        case .cpuFP16Tiled:
            return "cpu_fp16_tiled"
        }
    }

    /// Select the appropriate classifier strategy for a given model configuration.
    ///
    /// - Parameter config: The model configuration containing vocab size and embedding dimension.
    /// - Parameter hasExactFloat32LMHead: Whether the artifact ships an exact float32 sidecar for the LM head.
    /// - Returns: `.ane` if the classifier weight matrix fits in SRAM, otherwise an exact CPU backend.
    public static func select(
        for config: MultiModelConfig,
        hasExactFloat32LMHead: Bool = false
    ) -> ClassifierStrategy {
        // Stories 110M benefits from the ANE classifier path even though the
        // raw vocab*dModel product exceeds the conservative global SRAM cutoff.
        // Keep this as an explicit allowlist entry until a broader policy is proven.
        // See `ModelFamily.isStories110MVariant` for the single name match.
        if config.architecture == .llama, ModelFamily.isStories110MVariant(config) {
            return .ane
        }
        let elements = config.vocab * config.dModel
        if elements <= aneSRAMElementLimit {
            return .ane
        }
        if config.architecture == .llama && !hasExactFloat32LMHead {
            return .cpuFP16Tiled
        }
        return .cpuPartitionedFP32
    }
}

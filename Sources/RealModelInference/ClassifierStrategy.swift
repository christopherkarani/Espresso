import Espresso
import ModelSupport

/// Selects between ANE and CPU-tiled classifier based on the model's
/// vocabulary size and embedding dimension.
///
/// The ANE classifier is faster but requires the full weight matrix to fit
/// in the Neural Engine's SRAM. When the weight matrix exceeds the SRAM
/// element limit (16M elements = 32MB fp16), we fall back to the CPU-tiled
/// path which processes the matrix in L2-friendly tiles via `FP16TiledClassifier`.
public enum ClassifierStrategy: Sendable, Equatable {
    /// Use the ANE lane-packed classifier (fused RMSNorm + classifier head).
    case ane
    /// Use `FP16TiledClassifier.tiledMatvecArgmax` on the CPU.
    case cpuTiled

    /// Conservative SRAM element limit: 16M elements (32MB fp16).
    /// Leaves headroom for activations and intermediate buffers.
    private static let aneSRAMElementLimit: Int = 16_000_000

    /// Select the appropriate classifier strategy for a given model configuration.
    ///
    /// - Parameter config: The model configuration containing vocab size and embedding dimension.
    /// - Returns: `.ane` if the classifier weight matrix fits in SRAM, `.cpuTiled` otherwise.
    public static func select(for config: MultiModelConfig) -> ClassifierStrategy {
        let elements = config.vocab * config.dModel
        return elements <= aneSRAMElementLimit ? .ane : .cpuTiled
    }
}

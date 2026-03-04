public enum ModelConfig {
    public static let dim = 768
    public static let hidden = 2048
    public static let heads = 12
    public static let seqLen = 256
    public static let nLayers = 12
    public static let vocab = 32_000

    public static let accumSteps = 10
    public static let maxCompiles = 100

    public static let kernelsPerLayer = 5
    public static let totalWeightKernels = kernelsPerLayer * nLayers

    public static let headDim = dim / heads
    public static let scoreCh = heads * seqLen

    // Per-layer weight sizes (Float32 element counts).
    public static let wqSize = dim * dim
    public static let woSize = dim * dim
    public static let w1Size = hidden * dim
    public static let w2Size = dim * hidden
    public static let w3Size = hidden * dim

    public static let layerParams = 4 * wqSize + w1Size + w2Size + w3Size + 2 * dim
    public static let totalParams = nLayers * layerParams + dim + vocab * dim
}

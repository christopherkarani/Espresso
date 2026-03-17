public enum LayerWeightsArchitecture: Sendable, Equatable {
    case rmsNormSwiGLU
    case gpt2
}

public struct LayerWeights: ~Copyable {
    public let architecture: LayerWeightsArchitecture
    public let dim: Int
    public let kvDim: Int
    public let hiddenDim: Int
    public let Wq: TensorBuffer
    public let Wk: TensorBuffer
    public let Wv: TensorBuffer
    public let Wo: TensorBuffer
    public let W1: TensorBuffer
    public let W2: TensorBuffer
    public let W3: TensorBuffer
    public let rmsAtt: TensorBuffer
    public let rmsFfn: TensorBuffer
    public let attentionNormBeta: TensorBuffer
    public let ffnNormBeta: TensorBuffer
    public let bq: TensorBuffer
    public let bk: TensorBuffer
    public let bv: TensorBuffer
    public let bo: TensorBuffer
    public let b1: TensorBuffer
    public let b2: TensorBuffer

    public init() {
        self.init(architecture: .rmsNormSwiGLU, dim: ModelConfig.dim, hiddenDim: ModelConfig.hidden)
    }

    public init(architecture: LayerWeightsArchitecture, dim: Int, hiddenDim: Int, kvDim: Int? = nil) {
        let resolvedKVDim = kvDim ?? dim
        self.architecture = architecture
        self.dim = dim
        self.kvDim = resolvedKVDim
        self.hiddenDim = hiddenDim
        self.Wq = TensorBuffer(count: dim * dim, zeroed: false)
        self.Wk = TensorBuffer(count: dim * resolvedKVDim, zeroed: false)
        self.Wv = TensorBuffer(count: dim * resolvedKVDim, zeroed: false)
        self.Wo = TensorBuffer(count: dim * dim, zeroed: false)
        self.W1 = TensorBuffer(count: hiddenDim * dim, zeroed: false)
        self.W2 = TensorBuffer(count: dim * hiddenDim, zeroed: false)
        self.W3 = TensorBuffer(count: hiddenDim * dim, zeroed: false)
        self.rmsAtt = TensorBuffer(count: dim, zeroed: false)
        self.rmsFfn = TensorBuffer(count: dim, zeroed: false)
        self.attentionNormBeta = TensorBuffer(count: dim, zeroed: false)
        self.ffnNormBeta = TensorBuffer(count: dim, zeroed: false)
        self.bq = TensorBuffer(count: dim, zeroed: false)
        self.bk = TensorBuffer(count: resolvedKVDim, zeroed: false)
        self.bv = TensorBuffer(count: resolvedKVDim, zeroed: false)
        self.bo = TensorBuffer(count: dim, zeroed: false)
        self.b1 = TensorBuffer(count: hiddenDim, zeroed: false)
        self.b2 = TensorBuffer(count: dim, zeroed: false)
    }
}

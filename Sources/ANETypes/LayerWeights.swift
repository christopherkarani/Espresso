public struct LayerWeights: ~Copyable {
    public let Wq: TensorBuffer
    public let Wk: TensorBuffer
    public let Wv: TensorBuffer
    public let Wo: TensorBuffer
    public let W1: TensorBuffer
    public let W2: TensorBuffer
    public let W3: TensorBuffer
    public let rmsAtt: TensorBuffer
    public let rmsFfn: TensorBuffer

    public init() {
        self.Wq = TensorBuffer(count: ModelConfig.wqSize, zeroed: false)
        self.Wk = TensorBuffer(count: ModelConfig.wqSize, zeroed: false)
        self.Wv = TensorBuffer(count: ModelConfig.wqSize, zeroed: false)
        self.Wo = TensorBuffer(count: ModelConfig.woSize, zeroed: false)
        self.W1 = TensorBuffer(count: ModelConfig.w1Size, zeroed: false)
        self.W2 = TensorBuffer(count: ModelConfig.w2Size, zeroed: false)
        self.W3 = TensorBuffer(count: ModelConfig.w3Size, zeroed: false)
        self.rmsAtt = TensorBuffer(count: ModelConfig.dim, zeroed: false)
        self.rmsFfn = TensorBuffer(count: ModelConfig.dim, zeroed: false)
    }
}


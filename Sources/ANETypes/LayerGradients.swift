public struct LayerGradients: ~Copyable {
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
        self.Wq = TensorBuffer(count: ModelConfig.wqSize, zeroed: true)
        self.Wk = TensorBuffer(count: ModelConfig.wqSize, zeroed: true)
        self.Wv = TensorBuffer(count: ModelConfig.wqSize, zeroed: true)
        self.Wo = TensorBuffer(count: ModelConfig.woSize, zeroed: true)
        self.W1 = TensorBuffer(count: ModelConfig.w1Size, zeroed: true)
        self.W2 = TensorBuffer(count: ModelConfig.w2Size, zeroed: true)
        self.W3 = TensorBuffer(count: ModelConfig.w3Size, zeroed: true)
        self.rmsAtt = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        self.rmsFfn = TensorBuffer(count: ModelConfig.dim, zeroed: true)
    }

    public func zero() {
        Wq.zero(); Wk.zero(); Wv.zero(); Wo.zero()
        W1.zero(); W2.zero(); W3.zero()
        rmsAtt.zero(); rmsFfn.zero()
    }
}


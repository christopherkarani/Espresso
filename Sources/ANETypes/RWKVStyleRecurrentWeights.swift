public struct RWKVStyleRecurrentWeights: ~Copyable {
    public let rms: TensorBuffer
    public let Wx: TensorBuffer
    public let Ws: TensorBuffer
    public let Wd: TensorBuffer
    public let Wo: TensorBuffer

    public init() {
        self.rms = TensorBuffer(count: ModelConfig.dim, zeroed: false)
        self.Wx = TensorBuffer(count: ModelConfig.wqSize, zeroed: false)
        self.Ws = TensorBuffer(count: ModelConfig.wqSize, zeroed: false)
        self.Wd = TensorBuffer(count: ModelConfig.wqSize, zeroed: false)
        self.Wo = TensorBuffer(count: ModelConfig.woSize, zeroed: false)
    }
}

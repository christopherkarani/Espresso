public struct LayerAdam: ~Copyable {
    public let Wq: AdamState
    public let Wk: AdamState
    public let Wv: AdamState
    public let Wo: AdamState
    public let W1: AdamState
    public let W2: AdamState
    public let W3: AdamState
    public let rmsAtt: AdamState
    public let rmsFfn: AdamState

    public init() {
        self.Wq = AdamState(count: ModelConfig.wqSize)
        self.Wk = AdamState(count: ModelConfig.wqSize)
        self.Wv = AdamState(count: ModelConfig.wqSize)
        self.Wo = AdamState(count: ModelConfig.woSize)
        self.W1 = AdamState(count: ModelConfig.w1Size)
        self.W2 = AdamState(count: ModelConfig.w2Size)
        self.W3 = AdamState(count: ModelConfig.w3Size)
        self.rmsAtt = AdamState(count: ModelConfig.dim)
        self.rmsFfn = AdamState(count: ModelConfig.dim)
    }
}


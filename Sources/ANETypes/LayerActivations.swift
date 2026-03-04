public struct LayerActivations: ~Copyable {
    private static let dimSeq = ModelConfig.dim * ModelConfig.seqLen
    private static let hidSeq = ModelConfig.hidden * ModelConfig.seqLen

    public let layerIn: TensorBuffer
    public let xnorm: TensorBuffer
    public let Q: TensorBuffer
    public let K: TensorBuffer
    public let V: TensorBuffer
    public let attnOut: TensorBuffer
    public let oOut: TensorBuffer
    public let x2: TensorBuffer
    public let x2norm: TensorBuffer
    public let h1: TensorBuffer
    public let h3: TensorBuffer
    public let siluOut: TensorBuffer
    public let ffnOut: TensorBuffer

    public init() {
        self.layerIn = TensorBuffer(count: Self.dimSeq, zeroed: false)
        self.xnorm = TensorBuffer(count: Self.dimSeq, zeroed: false)
        self.Q = TensorBuffer(count: Self.dimSeq, zeroed: false)
        self.K = TensorBuffer(count: Self.dimSeq, zeroed: false)
        self.V = TensorBuffer(count: Self.dimSeq, zeroed: false)
        self.attnOut = TensorBuffer(count: Self.dimSeq, zeroed: false)
        self.oOut = TensorBuffer(count: Self.dimSeq, zeroed: false)
        self.x2 = TensorBuffer(count: Self.dimSeq, zeroed: false)
        self.x2norm = TensorBuffer(count: Self.dimSeq, zeroed: false)
        self.h1 = TensorBuffer(count: Self.hidSeq, zeroed: false)
        self.h3 = TensorBuffer(count: Self.hidSeq, zeroed: false)
        self.siluOut = TensorBuffer(count: Self.hidSeq, zeroed: false)
        self.ffnOut = TensorBuffer(count: Self.dimSeq, zeroed: false)
    }
}

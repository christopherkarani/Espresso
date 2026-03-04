@frozen
public struct CheckpointHeader {
    public var magic: Int32
    public var version: Int32
    public var step: Int32
    public var totalSteps: Int32
    public var nLayers: Int32
    public var vocabSize: Int32
    public var dim: Int32
    public var hiddenDim: Int32
    public var nHeads: Int32
    public var seqLen: Int32
    public var lr: Float
    public var loss: Float
    public var cumCompile: Double
    public var cumTrain: Double
    public var cumWall: Double
    public var cumSteps: Int32
    public var cumBatches: Int32
    public var adamT: Int32
    public var pad0: Int32
    public var pad1: Int32
    public var pad2: Int32

    public init() {
        self.magic = 0x424C5A54
        self.version = 2
        self.step = 0
        self.totalSteps = 0
        self.nLayers = 0
        self.vocabSize = 0
        self.dim = 0
        self.hiddenDim = 0
        self.nHeads = 0
        self.seqLen = 0
        self.lr = 0
        self.loss = 0
        self.cumCompile = 0
        self.cumTrain = 0
        self.cumWall = 0
        self.cumSteps = 0
        self.cumBatches = 0
        self.adamT = 0
        self.pad0 = 0
        self.pad1 = 0
        self.pad2 = 0
    }

    public static func validateLayout() {
        precondition(MemoryLayout<Self>.size == 96)
        precondition(MemoryLayout<Self>.alignment == 8)

        func off<T>(_ kp: KeyPath<Self, T>) -> Int {
            guard let o = MemoryLayout<Self>.offset(of: kp) else {
                preconditionFailure("Missing offset for \(kp)")
            }
            return o
        }

        precondition(off(\.magic) == 0)
        precondition(off(\.version) == 4)
        precondition(off(\.step) == 8)
        precondition(off(\.totalSteps) == 12)
        precondition(off(\.nLayers) == 16)
        precondition(off(\.vocabSize) == 20)
        precondition(off(\.dim) == 24)
        precondition(off(\.hiddenDim) == 28)
        precondition(off(\.nHeads) == 32)
        precondition(off(\.seqLen) == 36)
        precondition(off(\.lr) == 40)
        precondition(off(\.loss) == 44)
        precondition(off(\.cumCompile) == 48)
        precondition(off(\.cumTrain) == 56)
        precondition(off(\.cumWall) == 64)
        precondition(off(\.cumSteps) == 72)
        precondition(off(\.cumBatches) == 76)
        precondition(off(\.adamT) == 80)
        precondition(off(\.pad0) == 84)
        precondition(off(\.pad1) == 88)
        precondition(off(\.pad2) == 92)
    }
}

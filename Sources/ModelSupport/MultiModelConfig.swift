import ANETypes

public struct MultiModelConfig: Sendable, Equatable {
    public let name: String
    public let nLayer: Int
    public let nHead: Int
    public let nKVHead: Int
    public let dModel: Int
    public let headDim: Int
    public let hiddenDim: Int
    public let vocab: Int
    public let maxSeq: Int
    public let normEps: Float
    public let ropeTheta: Float
    public let eosToken: TokenID?
    public let architecture: Architecture

    public enum Architecture: Sendable, Equatable {
        case gpt2
        case llama
    }

    public init(
        name: String,
        nLayer: Int,
        nHead: Int,
        nKVHead: Int,
        dModel: Int,
        headDim: Int,
        hiddenDim: Int,
        vocab: Int,
        maxSeq: Int,
        normEps: Float,
        ropeTheta: Float = 10_000.0,
        eosToken: TokenID? = nil,
        architecture: Architecture
    ) {
        self.name = name
        self.nLayer = nLayer
        self.nHead = nHead
        self.nKVHead = nKVHead
        self.dModel = dModel
        self.headDim = headDim
        self.hiddenDim = hiddenDim
        self.vocab = vocab
        self.maxSeq = maxSeq
        self.normEps = normEps
        self.ropeTheta = ropeTheta
        self.eosToken = eosToken
        self.architecture = architecture
    }
}

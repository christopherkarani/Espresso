import ANETypes
import Foundation

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

    /// Coarse artifact preference for decode placement (hybrid family vs exact-CPU).
    ///
    /// When an artifact states this, the runtime honours it instead of guessing from the
    /// model name. `nil` means the artifact is silent and legacy name-based routing
    /// applies, which keeps older bundles behaving exactly as before.
    ///
    /// Fused vs split hybrid is **not** declared here — that is runtime policy resolved
    /// into ``Trunk``. See ``Trunk`` and `CONTEXT.md`.
    public let preferredDecodePath: PreferredDecodePath?

    public var attentionDim: Int { nHead * headDim }
    public var kvDim: Int { nKVHead * headDim }

    public enum Architecture: Sendable, Equatable {
        case gpt2
        case llama
    }

    /// Wire-format preference in `metadata.json` / bundle manifests.
    ///
    /// Only distinguishes the ANE hybrid family from exact-CPU. Resolved serving uses
    /// ``Trunk`` (`fusedHybrid` | `splitHybrid` | `exactCPU`).
    public enum PreferredDecodePath: String, Sendable, Equatable {
        case hybrid
        case exactCPU = "exact_cpu"

        public enum ParseError: Error, Sendable, Equatable, LocalizedError {
            case unsupported(String)

            public var errorDescription: String? {
                switch self {
                case let .unsupported(raw):
                    return "Unsupported preferredDecodePath: \(raw) (expected \"hybrid\" or \"exact_cpu\")"
                }
            }
        }

        /// Trims and lowercases `raw`, then maps it onto the declared decode path.
        public static func parse(_ raw: String) throws -> PreferredDecodePath {
            let normalized = raw
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
            guard let value = PreferredDecodePath(rawValue: normalized) else {
                throw ParseError.unsupported(raw)
            }
            return value
        }
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
        architecture: Architecture,
        preferredDecodePath: PreferredDecodePath? = nil
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
        self.preferredDecodePath = preferredDecodePath
    }
}

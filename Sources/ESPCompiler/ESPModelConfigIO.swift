import ANETypes
import Foundation
import ModelSupport

public enum ESPModelConfigIO {
    public static func load(fromMetadataFile url: URL) throws -> MultiModelConfig {
        let data = try Data(contentsOf: url)
        let metadata = try JSONDecoder().decode(MetadataFile.self, from: data)
        return try metadata.asConfig()
    }

    private struct MetadataFile: Decodable {
        let name: String
        let nLayer: Int
        let nHead: Int
        let nKVHead: Int
        let dModel: Int
        let headDim: Int
        let hiddenDim: Int
        let vocab: Int
        let maxSeq: Int
        let normEps: Float
        let ropeTheta: Float?
        let eosToken: Int?
        let architecture: String
        let preferredDecodePath: String?

        func asConfig() throws -> MultiModelConfig {
            let parsedArchitecture: MultiModelConfig.Architecture
            switch architecture.lowercased() {
            case "gpt2":
                parsedArchitecture = .gpt2
            case "llama":
                parsedArchitecture = .llama
            default:
                throw NSError(
                    domain: "ESPModelConfigIO",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Unsupported metadata architecture: \(architecture)"]
                )
            }

            // Dropping this silently would route an artifact that declares the ANE hybrid
            // path back to the pure-CPU oracle once it is packed into a bundle.
            var parsedDecodePath: MultiModelConfig.PreferredDecodePath?
            if let preferredDecodePath {
                let normalized = preferredDecodePath
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased()
                guard let value = MultiModelConfig.PreferredDecodePath(rawValue: normalized) else {
                    throw NSError(
                        domain: "ESPModelConfigIO",
                        code: 2,
                        userInfo: [
                            NSLocalizedDescriptionKey:
                                "Unsupported metadata preferredDecodePath: \(preferredDecodePath) "
                                + "(expected \"hybrid\" or \"exact_cpu\")",
                        ]
                    )
                }
                parsedDecodePath = value
            }

            return MultiModelConfig(
                name: name,
                nLayer: nLayer,
                nHead: nHead,
                nKVHead: nKVHead,
                dModel: dModel,
                headDim: headDim,
                hiddenDim: hiddenDim,
                vocab: vocab,
                maxSeq: maxSeq,
                normEps: normEps,
                ropeTheta: ropeTheta ?? 10_000.0,
                eosToken: eosToken.map { TokenID($0) },
                architecture: parsedArchitecture,
                preferredDecodePath: parsedDecodePath
            )
        }
    }
}

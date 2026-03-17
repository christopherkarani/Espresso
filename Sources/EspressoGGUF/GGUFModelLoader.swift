import Foundation
import EspressoEdgeRunner
import EdgeRunnerIO
import ModelSupport
import ANETypes
import Metal

public enum GGUFModelLoaderError: Error, Sendable {
    case metalDeviceUnavailable
    case configMappingFailed(String)
    case unsupportedArchitecture(String)
    case conversionFailed(String)
    case noTensorsConverted
}

/// Loads a GGUF model file, converts weights to Espresso's BLOBFILE format,
/// and produces a weight directory + MultiModelConfig ready for RealModelInferenceEngine.build().
public enum GGUFModelLoader {

    /// Result of preparing a GGUF model for Espresso.
    public struct PreparedModel: Sendable {
        /// MultiModelConfig for passing to RealModelInferenceEngine.build()
        public let config: MultiModelConfig
        /// Path to the temporary directory containing BLOBFILE weights in Espresso layout
        public let weightDir: String
        /// Number of tensors converted
        public let tensorCount: Int
    }

    /// Load a GGUF file and convert all weights to Espresso BLOBFILE format.
    /// Returns a PreparedModel with the config and a temp directory of weight files.
    public static func prepare(ggufURL: URL) async throws -> PreparedModel {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw GGUFModelLoaderError.metalDeviceUnavailable
        }

        // 1. Load GGUF via EdgeRunner
        let ggufLoader = try GGUFLoader(url: ggufURL)
        let weightMap = try await ggufLoader.load(from: ggufURL)
        let erConfig = try EspressoModelConfig(from: ggufLoader.modelConfig)

        // 2. Map to Espresso's MultiModelConfig
        let arch: MultiModelConfig.Architecture
        switch erConfig.architectureName.lowercased() {
        case "gpt2":
            arch = .gpt2
        case "llama", "llama2", "llama3", "mistral", "qwen2":
            arch = .llama
        default:
            throw GGUFModelLoaderError.unsupportedArchitecture(erConfig.architectureName)
        }

        let config = MultiModelConfig(
            name: erConfig.architectureName,
            nLayer: erConfig.blockCount,
            nHead: erConfig.headCount,
            nKVHead: erConfig.kvHeadCount,
            dModel: erConfig.embeddingDim,
            headDim: erConfig.headDim,
            hiddenDim: erConfig.feedForwardLength,
            vocab: ggufLoader.modelConfig.metadata["tokenizer.ggml.tokens"]?.arrayValue?.count
                ?? ggufLoader.modelConfig.int(forKey: "\(erConfig.architectureName).vocab_size")
                ?? 0,
            maxSeq: erConfig.contextLength,
            normEps: erConfig.rmsNormEpsilon,
            architecture: arch
        )

        // 3. Convert weights to temp directory
        let outputDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("espresso_gguf_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

        let converter = try WeightConverter(device: device)
        let count = try await converter.convert(
            weightMap: weightMap,
            architecture: erConfig.architectureName,
            outputDirectory: outputDir
        )

        guard count > 0 else {
            throw GGUFModelLoaderError.noTensorsConverted
        }

        // 4. Write metadata.json for Espresso's validateMetadataIfPresent
        let metadata: [String: Any] = [
            "name": config.name,
            "nLayer": config.nLayer,
            "nHead": config.nHead,
            "nKVHead": config.nKVHead,
            "dModel": config.dModel,
            "headDim": config.headDim,
            "hiddenDim": config.hiddenDim,
            "vocab": config.vocab,
            "maxSeq": config.maxSeq,
            "normEps": config.normEps,
            "architecture": erConfig.architectureName,
        ]
        if let jsonData = try? JSONSerialization.data(withJSONObject: metadata, options: .prettyPrinted) {
            try? jsonData.write(to: outputDir.appendingPathComponent("metadata.json"))
        }

        return PreparedModel(
            config: config,
            weightDir: outputDir.path,
            tensorCount: count
        )
    }

    /// Clean up a temporary weight directory created by prepare().
    public static func cleanup(weightDir: String) {
        try? FileManager.default.removeItem(atPath: weightDir)
    }
}

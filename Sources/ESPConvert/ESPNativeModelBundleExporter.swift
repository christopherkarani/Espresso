import ESPBundle
import ESPCompiler
import Foundation
import ModelSupport

public enum ESPNativeModelBundleExportError: Error, Equatable {
    case missingModelMetadata
    case missingTokenizerAssets
}

public struct ESPNativeModelBundleExportOptions: Sendable, Equatable {
    public let contextTargetTokens: Int?
    public let modelTier: ESPModelTier
    public let behaviorClass: ESPBehaviorClass
    public let optimization: ESPOptimizationMetadata
    public let outputHead: ESPOutputHeadMetadata?
    public let draft: ESPDraftMetadata?

    public init(
        contextTargetTokens: Int? = nil,
        modelTier: ESPModelTier = .compat,
        behaviorClass: ESPBehaviorClass = .exact,
        optimization: ESPOptimizationMetadata = .init(recipe: "native-baseline", qualityGate: "exact"),
        outputHead: ESPOutputHeadMetadata? = nil,
        draft: ESPDraftMetadata? = nil
    ) {
        self.contextTargetTokens = contextTargetTokens
        self.modelTier = modelTier
        self.behaviorClass = behaviorClass
        self.optimization = optimization
        self.outputHead = outputHead
        self.draft = draft
    }
}

public enum ESPNativeModelBundleExporter {
    private static let recognizedTokenizerFiles: Set<String> = [
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    ]

    public static func exportModel(
        at modelDirectory: URL,
        tokenizerDirectory: URL? = nil,
        outputBundleURL: URL,
        options: ESPNativeModelBundleExportOptions = .init(),
        overwriteExisting: Bool = false,
        fileManager: FileManager = .default
    ) throws -> ESPBundleArchive {
        let modelDirectory = modelDirectory.standardizedFileURL
        let tokenizerDirectory = (tokenizerDirectory ?? modelDirectory).standardizedFileURL
        let metadataURL = modelDirectory.appendingPathComponent("metadata.json")

        guard fileManager.fileExists(atPath: metadataURL.path) else {
            throw ESPNativeModelBundleExportError.missingModelMetadata
        }

        guard containsTokenizerAssets(at: tokenizerDirectory, fileManager: fileManager) else {
            throw ESPNativeModelBundleExportError.missingTokenizerAssets
        }

        let config = try ESPModelConfigIO.load(fromMetadataFile: metadataURL)
        let manifest = try makeManifest(from: config, options: options)
        let stagingWeightsDirectory = try makeWeightsStagingDirectory(
            modelDirectory: modelDirectory,
            tokenizerDirectory: tokenizerDirectory,
            fileManager: fileManager
        )
        defer { try? fileManager.removeItem(at: stagingWeightsDirectory) }

        // When the tokenizer lives inside the model directory, copying that directory
        // verbatim would duplicate every weight blob under tokenizer/ and double the
        // bundle. Stage just the tokenizer assets instead.
        let stagingTokenizerDirectory: URL? = modelDirectory.path == tokenizerDirectory.path
            ? try makeTokenizerStagingDirectory(tokenizerDirectory: tokenizerDirectory, fileManager: fileManager)
            : nil
        defer {
            if let stagingTokenizerDirectory {
                try? fileManager.removeItem(at: stagingTokenizerDirectory)
            }
        }

        return try ESPBundleArchive.create(
            at: outputBundleURL,
            manifest: manifest,
            weightsDirectory: stagingWeightsDirectory,
            tokenizerDirectory: stagingTokenizerDirectory ?? tokenizerDirectory,
            overwriteExisting: overwriteExisting,
            fileManager: fileManager
        )
    }

    public static func makeManifest(
        from config: MultiModelConfig,
        options: ESPNativeModelBundleExportOptions = .init()
    ) throws -> ESPManifest {
        let family = inferFamily(from: config)
        let supportedProfiles = makeProfiles(maxContext: config.maxSeq)
        let supportedBackends: [ESPBackendKind] = switch family {
        case .gpt2:
            [.anePrivate]
        case .llama, .qwen:
            [.anePrivate, .cpuSafe]
        }
        let contextTargetTokens = options.contextTargetTokens ?? config.maxSeq
        let modelID = contextTargetTokens == config.maxSeq
            ? config.name
            : "\(config.name)-ctx\(contextTargetTokens)"

        let manifest = ESPManifest(
            formatVersion: "1.1.0",
            modelID: modelID,
            modelFamily: family,
            architectureVersion: "decoder-v1",
            tokenizerContract: tokenizerContract(for: family),
            supportedBackends: supportedBackends,
            supportedProfiles: supportedProfiles,
            maxContext: config.maxSeq,
            contextTargetTokens: contextTargetTokens,
            compressionPolicy: .init(name: "native-ane-fp16", weightBits: 16, activationBits: nil),
            modelTier: options.modelTier,
            behaviorClass: options.behaviorClass,
            adapterSlots: 0,
            optimization: options.optimization,
            outputHead: options.outputHead,
            draft: options.draft,
            accuracyBaselineRef: "benchmarks/accuracy.json",
            performanceBaselineRef: "benchmarks/perf.json",
            signatureRef: "signatures/\(ESPBundleLayout.signatureCatalogFileName)"
        )
        try manifest.validate()
        return manifest
    }

    public static func inferFamily(from config: MultiModelConfig) -> ESPModelFamily {
        if config.architecture == .gpt2 {
            return .gpt2
        }

        if config.name.lowercased().contains("qwen") {
            return .qwen
        }

        return .llama
    }

    private static func makeProfiles(maxContext: Int) -> [ESPProfile] {
        var profiles: [ESPProfile] = [.prefill256, .decode1]
        if maxContext >= 2048 {
            profiles.append(.prefill2048)
        }
        return profiles
    }

    private static func tokenizerContract(for family: ESPModelFamily) -> String {
        switch family {
        case .gpt2:
            "gpt2-bpe-v1"
        case .llama:
            "sentencepiece-v1"
        case .qwen:
            "qwen-bpe-v1"
        }
    }

    private static func containsTokenizerAssets(
        at tokenizerDirectory: URL,
        fileManager: FileManager
    ) -> Bool {
        guard let enumerator = fileManager.enumerator(
            at: tokenizerDirectory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return false
        }

        for case let fileURL as URL in enumerator {
            guard let isRegularFile = try? fileURL.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile,
                  isRegularFile == true else {
                continue
            }

            if recognizedTokenizerFiles.contains(fileURL.lastPathComponent) {
                return true
            }
        }

        return false
    }

    private static func makeWeightsStagingDirectory(
        modelDirectory: URL,
        tokenizerDirectory: URL,
        fileManager: FileManager
    ) throws -> URL {
        let stagingDirectory = fileManager.temporaryDirectory
            .appendingPathComponent("esp-native-weights-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(at: stagingDirectory, withIntermediateDirectories: true)

        guard let enumerator = fileManager.enumerator(
            at: modelDirectory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return stagingDirectory
        }

        let modelPath = modelDirectory.path
        let tokenizerPath = tokenizerDirectory.path
        let sharesTokenizerRoot = modelPath == tokenizerPath

        for case let sourceURL as URL in enumerator {
            let relativePath = relativePath(from: modelDirectory, to: sourceURL)
            guard !relativePath.isEmpty else {
                continue
            }

            if sharesTokenizerRoot && shouldSkipInWeights(relativePath: relativePath) {
                continue
            }

            let destinationURL = stagingDirectory.appendingPathComponent(relativePath)
            let values = try sourceURL.resourceValues(forKeys: [.isDirectoryKey])
            if values.isDirectory == true {
                try fileManager.createDirectory(at: destinationURL, withIntermediateDirectories: true)
            } else {
                try fileManager.createDirectory(
                    at: destinationURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                try fileManager.copyItem(at: sourceURL, to: destinationURL)
            }
        }

        return stagingDirectory
    }

    private static func makeTokenizerStagingDirectory(
        tokenizerDirectory: URL,
        fileManager: FileManager
    ) throws -> URL {
        let stagingDirectory = fileManager.temporaryDirectory
            .appendingPathComponent("esp-native-tokenizer-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(at: stagingDirectory, withIntermediateDirectories: true)

        for name in recognizedTokenizerFiles.sorted() {
            let sourceURL = tokenizerDirectory.appendingPathComponent(name)
            guard fileManager.fileExists(atPath: sourceURL.path) else { continue }
            try fileManager.copyItem(at: sourceURL, to: stagingDirectory.appendingPathComponent(name))
        }

        return stagingDirectory
    }

    private static func shouldSkipInWeights(relativePath: String) -> Bool {
        let pathComponents = NSString(string: relativePath).pathComponents
        guard pathComponents.count == 1 else {
            return false
        }
        return recognizedTokenizerFiles.contains(relativePath)
    }

    private static func relativePath(from baseURL: URL, to fileURL: URL) -> String {
        let baseComponents = baseURL.resolvingSymlinksInPath().standardizedFileURL.pathComponents
        let fileComponents = fileURL.resolvingSymlinksInPath().standardizedFileURL.pathComponents
        let relativeComponents = fileComponents.dropFirst(baseComponents.count)
        return relativeComponents.joined(separator: "/")
    }
}

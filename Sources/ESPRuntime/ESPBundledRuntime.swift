import Darwin
import ESPBundle
import ESPCompiler
import Foundation
import ModelSupport
import RealModelInference

public struct ESPRuntimeBundle: Sendable, Equatable {
    public let archive: ESPBundleArchive
    public let config: MultiModelConfig

    public init(archive: ESPBundleArchive, config: MultiModelConfig) {
        self.archive = archive
        self.config = config
    }

    public static func open(at bundleURL: URL) throws -> ESPRuntimeBundle {
        let archive = try ESPBundleArchive.open(at: bundleURL)
        let config = try ESPModelConfigIO.load(
            fromMetadataFile: archive.weightsURL.appendingPathComponent("metadata.json")
        )
        return ESPRuntimeBundle(archive: archive, config: config)
    }
}

public struct ESPRuntimeHost {
    public static func currentCapabilities(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> ESPDeviceCapabilities {
        ESPDeviceCapabilities(
            supportsANEPrivate: environment["ESPRESSO_DISABLE_ANE_PRIVATE"] != "1"
        )
    }
}

public enum ESPRuntimeRunner {
    public static func resolve(
        bundle: ESPRuntimeBundle,
        requestedContextTokens: Int? = nil
    ) throws -> ESPRuntimeSelection {
        try ESPRuntimeResolver.selectBackend(
            capabilities: ESPRuntimeHost.currentCapabilities(),
            manifest: bundle.archive.manifest,
            requestedContextTokens: requestedContextTokens
        )
    }

    public static func generate(
        bundle: ESPRuntimeBundle,
        prompt: String,
        maxTokens: Int,
        temperature: Float = 0
    ) throws -> GenerationResult {
        let promptTokenCount = try countPromptTokens(bundle: bundle, prompt: prompt)
        let requestedContextTokens = totalRequestedContextTokens(
            promptTokenCount: promptTokenCount,
            maxTokens: maxTokens
        )
        let selection = try resolve(bundle: bundle, requestedContextTokens: requestedContextTokens)
        // Manifest metadata crosses into the engine as parameters, not as
        // process-global mutation: merge overrides into one dictionary and
        // hand it to build(), which resolves everything once.
        var environment = ProcessInfo.processInfo.environment
        for (key, value) in environmentOverrides(for: selection) {
            environment[key] = value
        }
        switch selection.backend {
        case .anePrivate:
            var engine = try RealModelInferenceEngine.build(
                config: bundle.config,
                weightDir: bundle.archive.weightsURL.path,
                tokenizerDir: bundle.archive.tokenizerURL.path,
                environment: environment
            )
            return try engine.generate(prompt: prompt, maxTokens: maxTokens, temperature: temperature)
        case .cpuSafe:
            environment["ESPRESSO_USE_CPU_EXACT_DECODE"] = "1"
            var engine = try RealModelInferenceEngine.build(
                config: bundle.config,
                weightDir: bundle.archive.weightsURL.path,
                tokenizerDir: bundle.archive.tokenizerURL.path,
                environment: environment
            )
            return try engine.generate(prompt: prompt, maxTokens: maxTokens, temperature: temperature)
        }
    }

    private enum LoadedTokenizer {
        case gpt2(GPT2BPETokenizer)
        case sentencePiece(SentencePieceTokenizer)

        func encode(_ text: String) -> [Int] {
            switch self {
            case let .gpt2(tokenizer):
                return tokenizer.encode(text)
            case let .sentencePiece(tokenizer):
                return tokenizer.encode(text)
            }
        }
    }

    private static func countPromptTokens(bundle: ESPRuntimeBundle, prompt: String) throws -> Int {
        let tokenizer = try loadTokenizer(config: bundle.config, tokenizerDirURL: bundle.archive.tokenizerURL)
        return tokenizer.encode(prompt).count
    }

    private static func totalRequestedContextTokens(promptTokenCount: Int, maxTokens: Int) -> Int {
        let clampedMaxTokens = max(maxTokens, 0)
        let (sum, overflowed) = promptTokenCount.addingReportingOverflow(clampedMaxTokens)
        return overflowed ? Int.max : sum
    }

    private static func loadTokenizer(
        config: MultiModelConfig,
        tokenizerDirURL: URL
    ) throws -> LoadedTokenizer {
        let fileManager = FileManager.default

        switch config.architecture {
        case .gpt2:
            let vocabURL = tokenizerDirURL.appendingPathComponent("vocab.json")
            let mergesURL = tokenizerDirURL.appendingPathComponent("merges.txt")
            guard fileManager.fileExists(atPath: vocabURL.path),
                  fileManager.fileExists(atPath: mergesURL.path) else {
                throw RealModelInferenceError.runtimeFailure(
                    "Missing GPT-2 tokenizer assets in bundle tokenizer directory"
                )
            }
            return .gpt2(try GPT2BPETokenizer(vocabURL: vocabURL, mergesURL: mergesURL))
        case .llama:
            for candidate in ["tokenizer.model", "tokenizer.bin"] {
                let url = tokenizerDirURL.appendingPathComponent(candidate)
                if fileManager.fileExists(atPath: url.path) {
                    return .sentencePiece(try SentencePieceTokenizer(modelURL: url))
                }
            }

            let tokenizerJSONURL = tokenizerDirURL.appendingPathComponent("tokenizer.json")
            if fileManager.fileExists(atPath: tokenizerJSONURL.path) {
                return .gpt2(try GPT2BPETokenizer(tokenizerJSONURL: tokenizerJSONURL))
            }

            let vocabURL = tokenizerDirURL.appendingPathComponent("vocab.json")
            let mergesURL = tokenizerDirURL.appendingPathComponent("merges.txt")
            if fileManager.fileExists(atPath: vocabURL.path),
               fileManager.fileExists(atPath: mergesURL.path) {
                return .gpt2(try GPT2BPETokenizer(vocabURL: vocabURL, mergesURL: mergesURL))
            }

            throw RealModelInferenceError.runtimeFailure(
                "No tokenizer found in bundle tokenizer directory"
            )
        }
    }

    private static func environmentOverrides(for selection: ESPRuntimeSelection) -> [String: String] {
        var overrides: [String: String] = [:]
        if let outputHead = selection.outputHead,
           outputHead.kind == .factored,
           outputHead.behaviorClass != .approximate {
            overrides["ESPRESSO_BUNDLE_OUTPUT_HEAD_KIND"] = outputHead.kind.rawValue
            if let bottleneck = outputHead.bottleneck {
                overrides["ESPRESSO_BUNDLE_OUTPUT_HEAD_BOTTLENECK"] = String(bottleneck)
            }
            if let groups = outputHead.groups {
                overrides["ESPRESSO_BUNDLE_OUTPUT_HEAD_GROUPS"] = String(groups)
            }
            if let projectionRef = outputHead.projectionRef {
                overrides["ESPRESSO_BUNDLE_OUTPUT_HEAD_PROJECTION_REF"] = projectionRef
            }
            if let expansionRef = outputHead.expansionRef {
                overrides["ESPRESSO_BUNDLE_OUTPUT_HEAD_EXPANSION_REF"] = expansionRef
            }
        }
        if let draft = selection.draft,
           draft.behaviorClass != .approximate {
            overrides["ESPRESSO_BUNDLE_DRAFT_KIND"] = draft.kind.rawValue
            overrides["ESPRESSO_BUNDLE_DRAFT_HORIZON"] = String(draft.horizon)
            overrides["ESPRESSO_BUNDLE_DRAFT_VERIFIER"] = draft.verifier
            overrides["ESPRESSO_BUNDLE_DRAFT_ROLLBACK"] = draft.rollback
            overrides["ESPRESSO_BUNDLE_DRAFT_ARTIFACT_REF"] = draft.artifactRef
            overrides["ESPRESSO_BUNDLE_DRAFT_ACCEPTANCE_METRIC"] = draft.acceptanceMetric
        }
        return overrides
    }
}


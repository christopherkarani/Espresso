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
    public static func resolve(bundle: ESPRuntimeBundle) throws -> ESPRuntimeSelection {
        try ESPRuntimeResolver.selectBackend(
            capabilities: ESPRuntimeHost.currentCapabilities(),
            manifest: bundle.archive.manifest
        )
    }

    public static func generate(
        bundle: ESPRuntimeBundle,
        prompt: String,
        maxTokens: Int,
        temperature: Float = 0
    ) throws -> GenerationResult {
        let selection = try resolve(bundle: bundle)
        return try withTemporaryEnvironment(environmentOverrides(for: selection)) {
            switch selection.backend {
            case .anePrivate:
                var engine = try RealModelInferenceEngine.build(
                    config: bundle.config,
                    weightDir: bundle.archive.weightsURL.path,
                    tokenizerDir: bundle.archive.tokenizerURL.path
                )
                return try engine.generate(prompt: prompt, maxTokens: maxTokens, temperature: temperature)
            case .cpuSafe:
                return try withTemporaryEnvironment(["ESPRESSO_USE_CPU_EXACT_DECODE": "1"]) {
                    var engine = try RealModelInferenceEngine.build(
                        config: bundle.config,
                        weightDir: bundle.archive.weightsURL.path,
                        tokenizerDir: bundle.archive.tokenizerURL.path
                    )
                    return try engine.generate(prompt: prompt, maxTokens: maxTokens, temperature: temperature)
                }
            }
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

private func withTemporaryEnvironment<T>(
    _ overrides: [String: String],
    operation: () throws -> T
) throws -> T {
    var original: [String: String?] = [:]
    for key in overrides.keys {
        if let pointer = getenv(key) {
            original[key] = String(cString: pointer)
        } else {
            original[key] = nil
        }
    }

    for (key, value) in overrides {
        setenv(key, value, 1)
    }

    defer {
        for (key, originalValue) in original {
            if let originalValue {
                setenv(key, originalValue, 1)
            } else {
                unsetenv(key)
            }
        }
    }

    return try operation()
}

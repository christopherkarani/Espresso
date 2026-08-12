import CryptoKit
import Foundation

public enum ESPModelFamily: String, Sendable, Codable, CaseIterable {
    case gpt2
    case llama
    case qwen
}

public enum ESPBackendKind: String, Sendable, Codable, CaseIterable {
    case anePrivate = "ane-private"
    case cpuSafe = "cpu-safe"
}

public enum ESPProfile: String, Sendable, Codable, CaseIterable {
    case prefill256 = "prefill_256"
    case prefill2048 = "prefill_2048"
    case decode1 = "decode_1"
    case decode2 = "decode_2"
}

public enum ESPModelTier: String, Sendable, Codable, CaseIterable {
    case compat
    case optimized
    case nativeFast = "native_fast"
}

public enum ESPBehaviorClass: String, Sendable, Codable, CaseIterable {
    case exact
    case nearExact = "near_exact"
    case approximate
}

public enum ESPOutputHeadKind: String, Sendable, Codable, CaseIterable {
    case dense
    case factored
}

public struct ESPOutputHeadMetadata: Sendable, Codable, Equatable {
    public let kind: ESPOutputHeadKind
    public let behaviorClass: ESPBehaviorClass
    public let bottleneck: Int?
    public let groups: Int?
    public let projectionRef: String?
    public let expansionRef: String?

    public init(
        kind: ESPOutputHeadKind,
        behaviorClass: ESPBehaviorClass,
        bottleneck: Int? = nil,
        groups: Int? = nil,
        projectionRef: String? = nil,
        expansionRef: String? = nil
    ) {
        self.kind = kind
        self.behaviorClass = behaviorClass
        self.bottleneck = bottleneck
        self.groups = groups
        self.projectionRef = projectionRef
        self.expansionRef = expansionRef
    }
}

public enum ESPDraftKind: String, Sendable, Codable, CaseIterable {
    case exactTwoToken = "exact_two_token"
    case multiToken = "multi_token"
}

public struct ESPDraftMetadata: Sendable, Codable, Equatable {
    public let kind: ESPDraftKind
    public let behaviorClass: ESPBehaviorClass
    public let horizon: Int
    public let verifier: String
    public let rollback: String
    public let artifactRef: String
    public let acceptanceMetric: String

    public init(
        kind: ESPDraftKind,
        behaviorClass: ESPBehaviorClass,
        horizon: Int,
        verifier: String,
        rollback: String,
        artifactRef: String,
        acceptanceMetric: String
    ) {
        self.kind = kind
        self.behaviorClass = behaviorClass
        self.horizon = horizon
        self.verifier = verifier
        self.rollback = rollback
        self.artifactRef = artifactRef
        self.acceptanceMetric = acceptanceMetric
    }
}

public struct ESPCompressionPolicy: Sendable, Codable, Equatable {
    public let name: String
    public let weightBits: Int
    public let activationBits: Int?

    public init(name: String, weightBits: Int, activationBits: Int?) {
        self.name = name
        self.weightBits = weightBits
        self.activationBits = activationBits
    }
}

public struct ESPOptimizationMetadata: Sendable, Codable, Equatable {
    public let recipe: String
    public let qualityGate: String
    public let teacherModel: String?
    public let draftModel: String?
    public let performanceTarget: String?

    public init(
        recipe: String,
        qualityGate: String,
        teacherModel: String? = nil,
        draftModel: String? = nil,
        performanceTarget: String? = nil
    ) {
        self.recipe = recipe
        self.qualityGate = qualityGate
        self.teacherModel = teacherModel
        self.draftModel = draftModel
        self.performanceTarget = performanceTarget
    }

    public static let legacyDefaults = ESPOptimizationMetadata(
        recipe: "legacy",
        qualityGate: "legacy-compatible"
    )
}

public enum ESPBundleValidationError: Error, Equatable {
    case emptyField(String)
    case invalidMaxContext(Int)
    case invalidContextTarget(Int)
    case invalidAdapterSlots(Int)
    case missingBackends
    case missingProfiles
    case invalidWeightBits(Int)
    case invalidActivationBits(Int)
    case missingTopLevelEntry(String)
    case missingManifest
    case malformedLine(String)
    case malformedSection(String)
    case unsupportedValue(String)
    case invalidOutputHead(String)
    case invalidDraft(String)
    case invalidReferencedPath(String)
    case missingReferencedFile(String)
    case signatureMismatch(path: String)
}

public struct ESPManifest: Sendable, Codable, Equatable {
    public let formatVersion: String
    public let modelID: String
    public let modelFamily: ESPModelFamily
    public let architectureVersion: String
    public let tokenizerContract: String
    public let supportedBackends: [ESPBackendKind]
    public let supportedProfiles: [ESPProfile]
    public let maxContext: Int
    public let contextTargetTokens: Int
    public let compressionPolicy: ESPCompressionPolicy
    public let modelTier: ESPModelTier
    public let behaviorClass: ESPBehaviorClass
    public let adapterSlots: Int
    public let optimization: ESPOptimizationMetadata
    public let outputHead: ESPOutputHeadMetadata?
    public let draft: ESPDraftMetadata?
    public let accuracyBaselineRef: String
    public let performanceBaselineRef: String
    public let signatureRef: String

    public init(
        formatVersion: String,
        modelID: String,
        modelFamily: ESPModelFamily,
        architectureVersion: String,
        tokenizerContract: String,
        supportedBackends: [ESPBackendKind],
        supportedProfiles: [ESPProfile],
        maxContext: Int,
        contextTargetTokens: Int? = nil,
        compressionPolicy: ESPCompressionPolicy,
        modelTier: ESPModelTier = .compat,
        behaviorClass: ESPBehaviorClass = .exact,
        adapterSlots: Int,
        optimization: ESPOptimizationMetadata = .legacyDefaults,
        outputHead: ESPOutputHeadMetadata? = nil,
        draft: ESPDraftMetadata? = nil,
        accuracyBaselineRef: String,
        performanceBaselineRef: String,
        signatureRef: String
    ) {
        self.formatVersion = formatVersion
        self.modelID = modelID
        self.modelFamily = modelFamily
        self.architectureVersion = architectureVersion
        self.tokenizerContract = tokenizerContract
        self.supportedBackends = supportedBackends
        self.supportedProfiles = supportedProfiles
        self.maxContext = maxContext
        self.contextTargetTokens = contextTargetTokens ?? maxContext
        self.compressionPolicy = compressionPolicy
        self.modelTier = modelTier
        self.behaviorClass = behaviorClass
        self.adapterSlots = adapterSlots
        self.optimization = optimization
        self.outputHead = outputHead
        self.draft = draft
        self.accuracyBaselineRef = accuracyBaselineRef
        self.performanceBaselineRef = performanceBaselineRef
        self.signatureRef = signatureRef
    }

    public func validate() throws {
        try validateNonEmpty(formatVersion, field: "format_version")
        try validateNonEmpty(modelID, field: "model_id")
        try validateNonEmpty(architectureVersion, field: "architecture_version")
        try validateNonEmpty(tokenizerContract, field: "tokenizer_contract")
        try validateNonEmpty(accuracyBaselineRef, field: "accuracy_baseline_ref")
        try validateNonEmpty(performanceBaselineRef, field: "performance_baseline_ref")
        try validateNonEmpty(signatureRef, field: "signature_ref")

        guard !supportedBackends.isEmpty else {
            throw ESPBundleValidationError.missingBackends
        }

        guard !supportedProfiles.isEmpty else {
            throw ESPBundleValidationError.missingProfiles
        }

        guard maxContext > 0 else {
            throw ESPBundleValidationError.invalidMaxContext(maxContext)
        }

        guard contextTargetTokens > 0, contextTargetTokens <= maxContext else {
            throw ESPBundleValidationError.invalidContextTarget(contextTargetTokens)
        }

        guard adapterSlots >= 0 else {
            throw ESPBundleValidationError.invalidAdapterSlots(adapterSlots)
        }

        guard compressionPolicy.weightBits > 0 else {
            throw ESPBundleValidationError.invalidWeightBits(compressionPolicy.weightBits)
        }

        if let activationBits = compressionPolicy.activationBits, activationBits <= 0 {
            throw ESPBundleValidationError.invalidActivationBits(activationBits)
        }

        try validateNonEmpty(optimization.recipe, field: "optimization.recipe")
        try validateNonEmpty(optimization.qualityGate, field: "optimization.quality_gate")
        try validate(outputHead: outputHead)
        try validate(draft: draft)
    }

    public func renderTOML() -> String {
        let backendValues = supportedBackends.map(\.rawValue)
        let profileValues = supportedProfiles.map(\.rawValue)
        let rawLines: [String?] = [
            "format_version = \(quoted(formatVersion))",
            "model_id = \(quoted(modelID))",
            "model_family = \(quoted(modelFamily.rawValue))",
            "architecture_version = \(quoted(architectureVersion))",
            "tokenizer_contract = \(quoted(tokenizerContract))",
            "supported_backends = \(quotedArray(backendValues))",
            "supported_profiles = \(quotedArray(profileValues))",
            "max_context = \(maxContext)",
            "context_target_tokens = \(contextTargetTokens)",
            "model_tier = \(quoted(modelTier.rawValue))",
            "behavior_class = \(quoted(behaviorClass.rawValue))",
            "adapter_slots = \(adapterSlots)",
            "accuracy_baseline_ref = \(quoted(accuracyBaselineRef))",
            "performance_baseline_ref = \(quoted(performanceBaselineRef))",
            "signature_ref = \(quoted(signatureRef))",
            "[compression_policy]",
            "name = \(quoted(compressionPolicy.name))",
            "weight_bits = \(compressionPolicy.weightBits)",
            compressionPolicy.activationBits.map { "activation_bits = \($0)" },
            "[optimization]",
            "recipe = \(quoted(optimization.recipe))",
            "quality_gate = \(quoted(optimization.qualityGate))",
            optimization.teacherModel.map { "teacher_model = \(quoted($0))" },
            optimization.draftModel.map { "draft_model = \(quoted($0))" },
            optimization.performanceTarget.map { "performance_target = \(quoted($0))" },
            outputHead.map { _ in "[output_head]" },
            outputHead.map { "kind = \(quoted($0.kind.rawValue))" },
            outputHead.map { "behavior_class = \(quoted($0.behaviorClass.rawValue))" },
            outputHead?.bottleneck.map { "bottleneck = \($0)" },
            outputHead?.groups.map { "groups = \($0)" },
            outputHead?.projectionRef.map { "projection_ref = \(quoted($0))" },
            outputHead?.expansionRef.map { "expansion_ref = \(quoted($0))" },
            draft.map { _ in "[draft]" },
            draft.map { "kind = \(quoted($0.kind.rawValue))" },
            draft.map { "behavior_class = \(quoted($0.behaviorClass.rawValue))" },
            draft.map { "horizon = \($0.horizon)" },
            draft.map { "verifier = \(quoted($0.verifier))" },
            draft.map { "rollback = \(quoted($0.rollback))" },
            draft.map { "artifact_ref = \(quoted($0.artifactRef))" },
            draft.map { "acceptance_metric = \(quoted($0.acceptanceMetric))" },
        ]
        let lines = rawLines.compactMap { $0 }

        return lines.joined(separator: "\n") + "\n"
    }

    public static func parseTOML(_ text: String) throws -> ESPManifest {
        var topLevel: [String: String] = [:]
        var compressionSection: [String: String] = [:]
        var optimizationSection: [String: String] = [:]
        var outputHeadSection: [String: String] = [:]
        var draftSection: [String: String] = [:]
        var currentSection: String?

        for rawLine in text.split(whereSeparator: \.isNewline) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty || line.hasPrefix("#") {
                continue
            }
            if line.hasPrefix("[") {
                switch line {
                case "[compression_policy]":
                    currentSection = "compression_policy"
                case "[optimization]":
                    currentSection = "optimization"
                case "[output_head]":
                    currentSection = "output_head"
                case "[draft]":
                    currentSection = "draft"
                default:
                    throw ESPBundleValidationError.malformedSection(line)
                }
                continue
            }

            guard let separator = line.firstIndex(of: "=") else {
                throw ESPBundleValidationError.malformedLine(line)
            }

            let key = String(line[..<separator]).trimmingCharacters(in: .whitespacesAndNewlines)
            let value = String(line[line.index(after: separator)...]).trimmingCharacters(in: .whitespacesAndNewlines)
            if currentSection == "compression_policy" {
                compressionSection[key] = value
            } else if currentSection == "optimization" {
                optimizationSection[key] = value
            } else if currentSection == "output_head" {
                outputHeadSection[key] = value
            } else if currentSection == "draft" {
                draftSection[key] = value
            } else {
                topLevel[key] = value
            }
        }

        let manifest = ESPManifest(
            formatVersion: try parseString(topLevel, key: "format_version"),
            modelID: try parseString(topLevel, key: "model_id"),
            modelFamily: try parseEnum(topLevel, key: "model_family", as: ESPModelFamily.self),
            architectureVersion: try parseString(topLevel, key: "architecture_version"),
            tokenizerContract: try parseString(topLevel, key: "tokenizer_contract"),
            supportedBackends: try parseArray(topLevel, key: "supported_backends", as: ESPBackendKind.self),
            supportedProfiles: try parseArray(topLevel, key: "supported_profiles", as: ESPProfile.self),
            maxContext: try parseInt(topLevel, key: "max_context"),
            contextTargetTokens: topLevel["context_target_tokens"].flatMap(Int.init),
            compressionPolicy: ESPCompressionPolicy(
                name: try parseString(compressionSection, key: "name"),
                weightBits: try parseInt(compressionSection, key: "weight_bits"),
                activationBits: compressionSection["activation_bits"].flatMap(Int.init)
            ),
            modelTier: try parseEnum(topLevel, key: "model_tier", as: ESPModelTier.self, defaultValue: .compat),
            behaviorClass: try parseEnum(topLevel, key: "behavior_class", as: ESPBehaviorClass.self, defaultValue: .exact),
            adapterSlots: try parseInt(topLevel, key: "adapter_slots"),
            optimization: ESPOptimizationMetadata(
                recipe: try parseString(optimizationSection, key: "recipe", defaultValue: ESPOptimizationMetadata.legacyDefaults.recipe),
                qualityGate: try parseString(optimizationSection, key: "quality_gate", defaultValue: ESPOptimizationMetadata.legacyDefaults.qualityGate),
                teacherModel: optimizationSection["teacher_model"].flatMap { try? unquote($0) },
                draftModel: optimizationSection["draft_model"].flatMap { try? unquote($0) },
                performanceTarget: optimizationSection["performance_target"].flatMap { try? unquote($0) }
            ),
            outputHead: try parseOutputHeadSection(outputHeadSection),
            draft: try parseDraftSection(draftSection),
            accuracyBaselineRef: try parseString(topLevel, key: "accuracy_baseline_ref"),
            performanceBaselineRef: try parseString(topLevel, key: "performance_baseline_ref"),
            signatureRef: try parseString(topLevel, key: "signature_ref")
        )
        try manifest.validate()
        return manifest
    }

    private func validateNonEmpty(_ value: String, field: String) throws {
        guard !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ESPBundleValidationError.emptyField(field)
        }
    }

    private func validate(outputHead: ESPOutputHeadMetadata?) throws {
        guard let outputHead else {
            return
        }
        guard modelFamily == .llama else {
            throw ESPBundleValidationError.invalidOutputHead(
                "output-head metadata is supported for llama bundles only"
            )
        }

        switch outputHead.kind {
        case .dense:
            if outputHead.bottleneck != nil || outputHead.groups != nil ||
                outputHead.projectionRef != nil || outputHead.expansionRef != nil {
                throw ESPBundleValidationError.invalidOutputHead(
                    "dense output head cannot declare factored weights or dimensions"
                )
            }
        case .factored:
            guard let bottleneck = outputHead.bottleneck, bottleneck > 0 else {
                throw ESPBundleValidationError.invalidOutputHead(
                    "factored output head requires bottleneck > 0"
                )
            }
            guard let groups = outputHead.groups, groups > 0 else {
                throw ESPBundleValidationError.invalidOutputHead(
                    "factored output head requires groups > 0"
                )
            }
            try validateReference(outputHead.projectionRef, field: "output_head.projection_ref")
            try validateReference(outputHead.expansionRef, field: "output_head.expansion_ref")
        }
    }

    private func validate(draft: ESPDraftMetadata?) throws {
        guard let draft else {
            return
        }
        guard modelFamily == .llama else {
            throw ESPBundleValidationError.invalidDraft(
                "draft metadata is supported for llama bundles only"
            )
        }
        guard draft.kind == .exactTwoToken else {
            throw ESPBundleValidationError.invalidDraft(
                "only exact_two_token drafts are currently supported"
            )
        }

        guard draft.horizon > 1 else {
            throw ESPBundleValidationError.invalidDraft("draft horizon must be > 1")
        }
        if draft.kind == .exactTwoToken, draft.horizon != 2 {
            throw ESPBundleValidationError.invalidDraft(
                "exact_two_token draft requires horizon == 2"
            )
        }
        try validateNonEmpty(draft.verifier, field: "draft.verifier")
        try validateNonEmpty(draft.rollback, field: "draft.rollback")
        try validateNonEmpty(draft.acceptanceMetric, field: "draft.acceptance_metric")
        try validateReference(draft.artifactRef, field: "draft.artifact_ref")
    }

    private func validateReference(_ value: String?, field: String) throws {
        guard let value else {
            throw ESPBundleValidationError.emptyField(field)
        }
        try validateNonEmpty(value, field: field)
    }
}

public enum ESPBundleLayout {
    public static let manifestFileName = "manifest.toml"
    public static let signatureCatalogFileName = "content-hashes.json"

    /// Only directories that actually carry content. Older bundles may contain
    /// additional empty directories (arch, graphs, states, adapters, compiled,
    /// benchmarks, licenses); they are ignored, not required.
    public static let requiredTopLevelEntries = [
        "tokenizer",
        "weights",
        "signatures",
    ]
}

public struct ESPSignatureCatalog: Sendable, Codable, Equatable {
    public let algorithm: String
    public let manifestHash: String
    public let fileHashes: [String: String]

    public init(algorithm: String = "sha256", manifestHash: String, fileHashes: [String: String]) {
        self.algorithm = algorithm
        self.manifestHash = manifestHash
        self.fileHashes = fileHashes
    }
}

public struct ESPBundleArchive: Sendable, Equatable {
    public let bundleURL: URL
    public let manifest: ESPManifest

    public init(bundleURL: URL, manifest: ESPManifest) {
        self.bundleURL = bundleURL
        self.manifest = manifest
    }

    public var manifestURL: URL {
        bundleURL.appendingPathComponent(ESPBundleLayout.manifestFileName)
    }

    public var weightsURL: URL {
        bundleURL.appendingPathComponent("weights", isDirectory: true)
    }

    public var tokenizerURL: URL {
        bundleURL.appendingPathComponent("tokenizer", isDirectory: true)
    }

    public var signaturesURL: URL {
        bundleURL.appendingPathComponent("signatures", isDirectory: true)
    }

    public var signatureCatalogURL: URL {
        signaturesURL.appendingPathComponent(ESPBundleLayout.signatureCatalogFileName)
    }

    public static func create(
        at bundleURL: URL,
        manifest: ESPManifest,
        weightsDirectory: URL,
        tokenizerDirectory: URL,
        overwriteExisting: Bool = false,
        fileManager: FileManager = .default
    ) throws -> ESPBundleArchive {
        try manifest.validate()

        if fileManager.fileExists(atPath: bundleURL.path) {
            guard overwriteExisting else {
                throw CocoaError(.fileWriteFileExists)
            }
            try fileManager.removeItem(at: bundleURL)
        }

        try fileManager.createDirectory(at: bundleURL, withIntermediateDirectories: true)

        for entry in ESPBundleLayout.requiredTopLevelEntries {
            try fileManager.createDirectory(
                at: bundleURL.appendingPathComponent(entry, isDirectory: true),
                withIntermediateDirectories: true
            )
        }

        let archive = ESPBundleArchive(bundleURL: bundleURL, manifest: manifest)
        try manifest.renderTOML().write(to: archive.manifestURL, atomically: true, encoding: .utf8)
        try copyDirectoryContents(from: weightsDirectory, to: archive.weightsURL, fileManager: fileManager)
        try copyDirectoryContents(from: tokenizerDirectory, to: archive.tokenizerURL, fileManager: fileManager)
        try archive.writeSignatureCatalog(fileManager: fileManager)
        return archive
    }

    public static func open(
        at bundleURL: URL,
        verifySignatures: Bool = true,
        fileManager: FileManager = .default
    ) throws -> ESPBundleArchive {
        let manifestURL = bundleURL.appendingPathComponent(ESPBundleLayout.manifestFileName)
        guard fileManager.fileExists(atPath: manifestURL.path) else {
            throw ESPBundleValidationError.missingManifest
        }

        for entry in ESPBundleLayout.requiredTopLevelEntries {
            let url = bundleURL.appendingPathComponent(entry, isDirectory: true)
            var isDirectory = ObjCBool(false)
            guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
                throw ESPBundleValidationError.missingTopLevelEntry(entry)
            }
        }

        let manifest = try ESPManifest.parseTOML(String(contentsOf: manifestURL, encoding: .utf8))
        let archive = ESPBundleArchive(bundleURL: bundleURL, manifest: manifest)
        if verifySignatures {
            try archive.verifySignatures(fileManager: fileManager)
        }
        try archive.validateReferencedArtifacts(fileManager: fileManager)
        return archive
    }

    public func writeSignatureCatalog(fileManager: FileManager = .default) throws {
        let catalog = try computeSignatureCatalog(fileManager: fileManager)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(catalog)
        try data.write(to: signatureCatalogURL, options: .atomic)
    }

    public func verifySignatures(fileManager: FileManager = .default) throws {
        let data = try Data(contentsOf: signatureCatalogURL)
        let catalog = try JSONDecoder().decode(ESPSignatureCatalog.self, from: data)
        let computed = try computeSignatureCatalog(fileManager: fileManager)

        guard catalog.manifestHash == computed.manifestHash else {
            throw ESPBundleValidationError.signatureMismatch(path: ESPBundleLayout.manifestFileName)
        }

        for (path, expectedHash) in catalog.fileHashes.sorted(by: { $0.key < $1.key }) {
            guard computed.fileHashes[path] == expectedHash else {
                throw ESPBundleValidationError.signatureMismatch(path: path)
            }
        }
    }

    private func computeSignatureCatalog(fileManager: FileManager) throws -> ESPSignatureCatalog {
        let manifestHash = try sha256(of: manifestURL)
        var hashes: [String: String] = [:]
        for fileURL in try enumeratedFiles(fileManager: fileManager) {
            let relativePath = relativePath(from: bundleURL, to: fileURL)
            hashes[relativePath] = try sha256(of: fileURL)
        }
        return ESPSignatureCatalog(manifestHash: manifestHash, fileHashes: hashes)
    }

    private func enumeratedFiles(fileManager: FileManager) throws -> [URL] {
        guard let enumerator = fileManager.enumerator(at: bundleURL, includingPropertiesForKeys: [.isRegularFileKey]) else {
            return []
        }

        var urls: [URL] = []
        for case let url as URL in enumerator {
            if url.lastPathComponent == ESPBundleLayout.manifestFileName ||
                url.path == signatureCatalogURL.path {
                continue
            }

            let values = try url.resourceValues(forKeys: [.isRegularFileKey])
            if values.isRegularFile == true {
                urls.append(url)
            }
        }
        return urls.sorted { $0.path < $1.path }
    }

    private func validateReferencedArtifacts(fileManager: FileManager) throws {
        if let outputHead = manifest.outputHead, outputHead.kind == .factored {
            if let projectionRef = outputHead.projectionRef {
                try validateReferencedArtifact(
                    reference: projectionRef,
                    fileManager: fileManager
                )
            }
            if let expansionRef = outputHead.expansionRef {
                try validateReferencedArtifact(
                    reference: expansionRef,
                    fileManager: fileManager
                )
            }
        }
        if let draft = manifest.draft {
            try validateReferencedArtifact(reference: draft.artifactRef, fileManager: fileManager)
        }
    }

    private func validateReferencedArtifact(
        reference: String,
        fileManager: FileManager
    ) throws {
        let candidateURL = try resolveBundleRelativeFile(reference: reference)
        var isDirectory = ObjCBool(false)
        guard fileManager.fileExists(atPath: candidateURL.path, isDirectory: &isDirectory),
              !isDirectory.boolValue else {
            throw ESPBundleValidationError.missingReferencedFile(reference)
        }
    }

    private func resolveBundleRelativeFile(reference: String) throws -> URL {
        guard !reference.isEmpty,
              !reference.hasPrefix("/") else {
            throw ESPBundleValidationError.invalidReferencedPath(reference)
        }
        let resolvedURL = bundleURL.appendingPathComponent(reference).standardizedFileURL
        let rootPath = bundleURL.standardizedFileURL.path
        let resolvedPath = resolvedURL.path
        guard resolvedPath == rootPath || resolvedPath.hasPrefix(rootPath + "/") else {
            throw ESPBundleValidationError.invalidReferencedPath(reference)
        }
        return resolvedURL
    }
}

private func quoted(_ value: String) -> String {
    "\"\(value)\""
}

private func quotedArray(_ values: [String]) -> String {
    "[\(values.map(quoted).joined(separator: ", "))]"
}

private func parseString(_ dictionary: [String: String], key: String) throws -> String {
    guard let raw = dictionary[key] else {
        throw ESPBundleValidationError.emptyField(key)
    }
    return try unquote(raw)
}

private func parseString(
    _ dictionary: [String: String],
    key: String,
    defaultValue: String
) throws -> String {
    guard let raw = dictionary[key] else {
        return defaultValue
    }
    return try unquote(raw)
}

private func parseInt(_ dictionary: [String: String], key: String) throws -> Int {
    guard let raw = dictionary[key], let value = Int(raw) else {
        throw ESPBundleValidationError.unsupportedValue(key)
    }
    return value
}

private func parseEnum<T: RawRepresentable>(
    _ dictionary: [String: String],
    key: String,
    as type: T.Type
) throws -> T where T.RawValue == String {
    let raw = try parseString(dictionary, key: key)
    guard let value = T(rawValue: raw) else {
        throw ESPBundleValidationError.unsupportedValue(raw)
    }
    return value
}

private func parseOutputHeadSection(
    _ dictionary: [String: String]
) throws -> ESPOutputHeadMetadata? {
    guard !dictionary.isEmpty else {
        return nil
    }

    return ESPOutputHeadMetadata(
        kind: try parseEnum(dictionary, key: "kind", as: ESPOutputHeadKind.self),
        behaviorClass: try parseEnum(dictionary, key: "behavior_class", as: ESPBehaviorClass.self),
        bottleneck: dictionary["bottleneck"].flatMap(Int.init),
        groups: dictionary["groups"].flatMap(Int.init),
        projectionRef: dictionary["projection_ref"].flatMap { try? unquote($0) },
        expansionRef: dictionary["expansion_ref"].flatMap { try? unquote($0) }
    )
}

private func parseDraftSection(
    _ dictionary: [String: String]
) throws -> ESPDraftMetadata? {
    guard !dictionary.isEmpty else {
        return nil
    }

    return ESPDraftMetadata(
        kind: try parseEnum(dictionary, key: "kind", as: ESPDraftKind.self),
        behaviorClass: try parseEnum(dictionary, key: "behavior_class", as: ESPBehaviorClass.self),
        horizon: try parseInt(dictionary, key: "horizon"),
        verifier: try parseString(dictionary, key: "verifier"),
        rollback: try parseString(dictionary, key: "rollback"),
        artifactRef: try parseString(dictionary, key: "artifact_ref"),
        acceptanceMetric: try parseString(dictionary, key: "acceptance_metric")
    )
}

private func parseEnum<T: RawRepresentable>(
    _ dictionary: [String: String],
    key: String,
    as type: T.Type,
    defaultValue: T
) throws -> T where T.RawValue == String {
    guard dictionary[key] != nil else {
        return defaultValue
    }
    return try parseEnum(dictionary, key: key, as: type)
}

private func parseArray<T: RawRepresentable>(
    _ dictionary: [String: String],
    key: String,
    as type: T.Type
) throws -> [T] where T.RawValue == String {
    guard let raw = dictionary[key] else {
        throw ESPBundleValidationError.emptyField(key)
    }
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard trimmed.hasPrefix("["), trimmed.hasSuffix("]") else {
        throw ESPBundleValidationError.unsupportedValue(raw)
    }

    let body = String(trimmed.dropFirst().dropLast())
    if body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        return []
    }

    return try body
        .split(separator: ",")
        .map { try unquote(String($0).trimmingCharacters(in: .whitespacesAndNewlines)) }
        .map {
            guard let value = T(rawValue: $0) else {
                throw ESPBundleValidationError.unsupportedValue($0)
            }
            return value
        }
}

private func unquote(_ value: String) throws -> String {
    let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
    guard trimmed.hasPrefix("\""), trimmed.hasSuffix("\"") else {
        throw ESPBundleValidationError.unsupportedValue(value)
    }
    return String(trimmed.dropFirst().dropLast())
}

private func copyDirectoryContents(from source: URL, to destination: URL, fileManager: FileManager) throws {
    guard let enumerator = fileManager.enumerator(at: source, includingPropertiesForKeys: [.isDirectoryKey]) else {
        return
    }

    for case let itemURL as URL in enumerator {
        let relativePath = relativePath(from: source, to: itemURL)
        let targetURL = destination.appendingPathComponent(relativePath)
        let values = try itemURL.resourceValues(forKeys: [.isDirectoryKey])
        if values.isDirectory == true {
            try fileManager.createDirectory(at: targetURL, withIntermediateDirectories: true)
        } else {
            try fileManager.createDirectory(
                at: targetURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            if fileManager.fileExists(atPath: targetURL.path) {
                try fileManager.removeItem(at: targetURL)
            }
            try fileManager.copyItem(at: itemURL, to: targetURL)
        }
    }
}

private func sha256(of url: URL) throws -> String {
    let digest = SHA256.hash(data: try Data(contentsOf: url))
    return digest.map { String(format: "%02x", $0) }.joined()
}

private func relativePath(from baseURL: URL, to fileURL: URL) -> String {
    let baseComponents = baseURL.resolvingSymlinksInPath().standardizedFileURL.pathComponents
    let fileComponents = fileURL.resolvingSymlinksInPath().standardizedFileURL.pathComponents
    let relativeComponents = fileComponents.dropFirst(baseComponents.count)
    return relativeComponents.joined(separator: "/")
}

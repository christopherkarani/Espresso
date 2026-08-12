import Testing
import Foundation
@testable import ESPBundle

@Test func manifestRenderingIsDeterministic() throws {
    let manifest = ESPManifest(
        formatVersion: "1.1.0",
        modelID: "espresso.llama.1b",
        modelFamily: .llama,
        architectureVersion: "decoder-v1",
        tokenizerContract: "sentencepiece-v1",
        supportedBackends: [.anePrivate, .cpuSafe],
        supportedProfiles: [.prefill256, .prefill2048, .decode1],
        maxContext: 2048,
        contextTargetTokens: 1024,
        compressionPolicy: .init(name: "int4-palettized", weightBits: 4, activationBits: nil),
        modelTier: .optimized,
        behaviorClass: .nearExact,
        adapterSlots: 4,
        optimization: .init(
            recipe: "stories-gqa4-distilled",
            qualityGate: "short-long-prompt-parity",
            teacherModel: "teacher://qwen3-0.6b",
            draftModel: nil,
            performanceTarget: "110 tok/s"
        ),
        outputHead: .init(
            kind: .factored,
            behaviorClass: .nearExact,
            bottleneck: 128,
            groups: 1,
            projectionRef: "weights/cls_proj.bin",
            expansionRef: "weights/cls_expand.bin"
        ),
        draft: .init(
            kind: .exactTwoToken,
            behaviorClass: .exact,
            horizon: 2,
            verifier: "exact",
            rollback: "exact_replay",
            artifactRef: "weights/future-sidecar.bin",
            acceptanceMetric: "accepted_future_tokens"
        ),
        accuracyBaselineRef: "benchmarks/qwen-0.6b/accuracy.json",
        performanceBaselineRef: "benchmarks/qwen-0.6b/perf.json",
        signatureRef: "signatures/manifest.sig"
    )

    let renderedA = manifest.renderTOML()
    let renderedB = manifest.renderTOML()
    #expect(renderedA == renderedB)
    #expect(renderedA.contains("model_family = \"llama\""))
    #expect(renderedA.contains("model_tier = \"optimized\""))
    #expect(renderedA.contains("behavior_class = \"near_exact\""))
    #expect(renderedA.contains("context_target_tokens = 1024"))
    #expect(renderedA.contains("supported_backends = [\"ane-private\", \"cpu-safe\"]"))
    #expect(renderedA.contains("[output_head]"))
    #expect(renderedA.contains("projection_ref = \"weights/cls_proj.bin\""))
    #expect(renderedA.contains("[draft]"))
    #expect(renderedA.contains("artifact_ref = \"weights/future-sidecar.bin\""))
}

@Test func manifestValidationRejectsEmptyModelID() {
    let manifest = ESPManifest(
        formatVersion: "1.0.0",
        modelID: "",
        modelFamily: .llama,
        architectureVersion: "decoder-v1",
        tokenizerContract: "spm-v1",
        supportedBackends: [.anePrivate],
        supportedProfiles: [.decode1],
        maxContext: 2048,
        compressionPolicy: .init(name: "fp16", weightBits: 16, activationBits: nil),
        adapterSlots: 0,
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/manifest.sig"
    )

    do {
        try manifest.validate()
        #expect(Bool(false), "Expected validation failure for an empty model id")
    } catch let error as ESPBundleValidationError {
        #expect(error == .emptyField("model_id"))
    } catch {
        #expect(Bool(false), "Unexpected error: \(error)")
    }
}

@Test func bundleLayoutIncludesCanonicalTopLevelEntries() {
    #expect(ESPBundleLayout.manifestFileName == "manifest.toml")
    #expect(ESPBundleLayout.requiredTopLevelEntries == [
        "tokenizer",
        "weights",
        "signatures",
    ])
}

@Test func manifestRoundTripsThroughTOMLParser() throws {
    let manifest = ESPManifest(
        formatVersion: "1.1.0",
        modelID: "espresso.llama.1b",
        modelFamily: .llama,
        architectureVersion: "decoder-v1",
        tokenizerContract: "sentencepiece-v1",
        supportedBackends: [.anePrivate],
        supportedProfiles: [.prefill256, .decode1],
        maxContext: 1024,
        contextTargetTokens: 512,
        compressionPolicy: .init(name: "fp16", weightBits: 16, activationBits: nil),
        modelTier: .compat,
        behaviorClass: .exact,
        adapterSlots: 0,
        optimization: .init(
            recipe: "native-baseline",
            qualityGate: "exact",
            teacherModel: nil,
            draftModel: nil,
            performanceTarget: nil
        ),
        outputHead: .init(
            kind: .factored,
            behaviorClass: .nearExact,
            bottleneck: 96,
            groups: 3,
            projectionRef: "weights/cls_proj.bin",
            expansionRef: "weights/cls_expand.bin"
        ),
        draft: .init(
            kind: .exactTwoToken,
            behaviorClass: .exact,
            horizon: 2,
            verifier: "exact",
            rollback: "exact_replay",
            artifactRef: "weights/future-sidecar.bin",
            acceptanceMetric: "accepted_future_tokens"
        ),
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/content-hashes.json"
    )

    let parsed = try ESPManifest.parseTOML(manifest.renderTOML())
    #expect(parsed == manifest)
}

@Test func manifestValidationRejectsOutputHeadForNonLlamaBundles() {
    let manifest = ESPManifest(
        formatVersion: "1.1.0",
        modelID: "espresso.gpt2.factored",
        modelFamily: .gpt2,
        architectureVersion: "decoder-v1",
        tokenizerContract: "gpt2-bpe-v1",
        supportedBackends: [.anePrivate],
        supportedProfiles: [.prefill256, .decode1],
        maxContext: 256,
        contextTargetTokens: 256,
        compressionPolicy: .init(name: "fp16", weightBits: 16, activationBits: nil),
        modelTier: .optimized,
        behaviorClass: .nearExact,
        adapterSlots: 0,
        optimization: .init(recipe: "factored", qualityGate: "gated"),
        outputHead: .init(
            kind: .factored,
            behaviorClass: .nearExact,
            bottleneck: 64,
            groups: 1,
            projectionRef: "weights/cls_proj.bin",
            expansionRef: "weights/cls_expand.bin"
        ),
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/content-hashes.json"
    )

    do {
        try manifest.validate()
        #expect(Bool(false), "Expected non-llama output-head rejection")
    } catch let error as ESPBundleValidationError {
        #expect(error == .invalidOutputHead("output-head metadata is supported for llama bundles only"))
    } catch {
        #expect(Bool(false), "Unexpected error: \(error)")
    }
}

@Test func manifestValidationRejectsUnsupportedDraftKindsAndFamilies() {
    let qwenManifest = ESPManifest(
        formatVersion: "1.1.0",
        modelID: "espresso.qwen.draft",
        modelFamily: .qwen,
        architectureVersion: "decoder-v1",
        tokenizerContract: "qwen-bpe-v1",
        supportedBackends: [.anePrivate],
        supportedProfiles: [.prefill256, .decode1],
        maxContext: 256,
        contextTargetTokens: 256,
        compressionPolicy: .init(name: "fp16", weightBits: 16, activationBits: nil),
        modelTier: .optimized,
        behaviorClass: .exact,
        adapterSlots: 0,
        optimization: .init(recipe: "draft", qualityGate: "gated"),
        draft: .init(
            kind: .exactTwoToken,
            behaviorClass: .exact,
            horizon: 2,
            verifier: "exact",
            rollback: "exact_replay",
            artifactRef: "weights/future-sidecar.bin",
            acceptanceMetric: "accepted_future_tokens"
        ),
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/content-hashes.json"
    )

    do {
        try qwenManifest.validate()
        #expect(Bool(false), "Expected non-llama draft rejection")
    } catch let error as ESPBundleValidationError {
        #expect(error == .invalidDraft("draft metadata is supported for llama bundles only"))
    } catch {
        #expect(Bool(false), "Unexpected error: \(error)")
    }

    let llamaManifest = ESPManifest(
        formatVersion: "1.1.0",
        modelID: "espresso.llama.multi-token",
        modelFamily: .llama,
        architectureVersion: "decoder-v1",
        tokenizerContract: "sentencepiece-v1",
        supportedBackends: [.anePrivate],
        supportedProfiles: [.prefill256, .decode1],
        maxContext: 256,
        contextTargetTokens: 256,
        compressionPolicy: .init(name: "fp16", weightBits: 16, activationBits: nil),
        modelTier: .optimized,
        behaviorClass: .exact,
        adapterSlots: 0,
        optimization: .init(recipe: "draft", qualityGate: "gated"),
        draft: .init(
            kind: .multiToken,
            behaviorClass: .nearExact,
            horizon: 4,
            verifier: "exact",
            rollback: "exact_replay",
            artifactRef: "weights/future-sidecar.bin",
            acceptanceMetric: "accepted_future_tokens"
        ),
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/content-hashes.json"
    )

    do {
        try llamaManifest.validate()
        #expect(Bool(false), "Expected unsupported multi-token draft rejection")
    } catch let error as ESPBundleValidationError {
        #expect(error == .invalidDraft("only exact_two_token drafts are currently supported"))
    } catch {
        #expect(Bool(false), "Unexpected error: \(error)")
    }
}

@Test func manifestParserBackfillsDefaultsForLegacyV1Bundles() throws {
    let text = """
    format_version = "1.0.0"
    model_id = "legacy.stories"
    model_family = "llama"
    architecture_version = "decoder-v1"
    tokenizer_contract = "sentencepiece-v1"
    supported_backends = ["ane-private", "cpu-safe"]
    supported_profiles = ["prefill_256", "decode_1"]
    max_context = 256
    adapter_slots = 0
    accuracy_baseline_ref = "benchmarks/accuracy.json"
    performance_baseline_ref = "benchmarks/perf.json"
    signature_ref = "signatures/content-hashes.json"
    [compression_policy]
    name = "native-ane-fp16"
    weight_bits = 16
    """

    let manifest = try ESPManifest.parseTOML(text)

    #expect(manifest.modelTier == .compat)
    #expect(manifest.behaviorClass == .exact)
    #expect(manifest.contextTargetTokens == 256)
    #expect(manifest.optimization.recipe == "legacy")
    #expect(manifest.optimization.qualityGate == "legacy-compatible")
    #expect(manifest.outputHead == nil)
    #expect(manifest.draft == nil)
}

@Test func manifestValidationRejectsIncompleteFactoredHead() {
    let manifest = ESPManifest(
        formatVersion: "1.1.0",
        modelID: "espresso.llama.factored",
        modelFamily: .llama,
        architectureVersion: "decoder-v1",
        tokenizerContract: "sentencepiece-v1",
        supportedBackends: [.anePrivate],
        supportedProfiles: [.prefill256, .decode1],
        maxContext: 256,
        contextTargetTokens: 256,
        compressionPolicy: .init(name: "fp16", weightBits: 16, activationBits: nil),
        modelTier: .optimized,
        behaviorClass: .nearExact,
        adapterSlots: 0,
        optimization: .init(recipe: "factored", qualityGate: "gated"),
        outputHead: .init(
            kind: .factored,
            behaviorClass: .nearExact,
            bottleneck: 128,
            groups: 1,
            projectionRef: nil,
            expansionRef: "weights/cls_expand.bin"
        ),
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/content-hashes.json"
    )

    do {
        try manifest.validate()
        #expect(Bool(false), "Expected invalid factored head")
    } catch let error as ESPBundleValidationError {
        #expect(error == .emptyField("output_head.projection_ref"))
    } catch {
        #expect(Bool(false), "Unexpected error: \(error)")
    }
}

@Test func manifestValidationRejectsInvalidExactTwoTokenDraft() {
    let manifest = ESPManifest(
        formatVersion: "1.1.0",
        modelID: "espresso.llama.draft",
        modelFamily: .llama,
        architectureVersion: "decoder-v1",
        tokenizerContract: "sentencepiece-v1",
        supportedBackends: [.anePrivate],
        supportedProfiles: [.prefill256, .decode1],
        maxContext: 256,
        contextTargetTokens: 256,
        compressionPolicy: .init(name: "fp16", weightBits: 16, activationBits: nil),
        modelTier: .optimized,
        behaviorClass: .exact,
        adapterSlots: 0,
        optimization: .init(recipe: "draft", qualityGate: "gated"),
        draft: .init(
            kind: .exactTwoToken,
            behaviorClass: .exact,
            horizon: 3,
            verifier: "exact",
            rollback: "exact_replay",
            artifactRef: "weights/future-sidecar.bin",
            acceptanceMetric: "accepted_future_tokens"
        ),
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/content-hashes.json"
    )

    do {
        try manifest.validate()
        #expect(Bool(false), "Expected invalid draft horizon")
    } catch let error as ESPBundleValidationError {
        #expect(error == .invalidDraft("exact_two_token draft requires horizon == 2"))
    } catch {
        #expect(Bool(false), "Unexpected error: \(error)")
    }
}

@Test func bundleOpenRejectsMissingReferencedDraftArtifact() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    let weights = root.appendingPathComponent("weights-src", isDirectory: true)
    let tokenizer = root.appendingPathComponent("tokenizer-src", isDirectory: true)
    let bundle = root.appendingPathComponent("model.esp", isDirectory: true)

    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: weights.appendingPathComponent("metadata.json"))
    try Data("weights".utf8).write(to: weights.appendingPathComponent("lm_head.bin"))
    try Data("proj".utf8).write(to: weights.appendingPathComponent("cls_proj.bin"))
    try Data("expand".utf8).write(to: weights.appendingPathComponent("cls_expand.bin"))
    try Data("tokenizer".utf8).write(to: tokenizer.appendingPathComponent("tokenizer.model"))

    let manifest = ESPManifest(
        formatVersion: "1.1.0",
        modelID: "espresso.llama.test",
        modelFamily: .llama,
        architectureVersion: "decoder-v1",
        tokenizerContract: "sentencepiece-v1",
        supportedBackends: [.anePrivate, .cpuSafe],
        supportedProfiles: [.prefill256, .decode1],
        maxContext: 256,
        contextTargetTokens: 256,
        compressionPolicy: .init(name: "fp16", weightBits: 16, activationBits: nil),
        modelTier: .optimized,
        behaviorClass: .exact,
        adapterSlots: 0,
        optimization: .init(recipe: "stories-ctx256", qualityGate: "exact"),
        outputHead: .init(
            kind: .factored,
            behaviorClass: .nearExact,
            bottleneck: 64,
            groups: 1,
            projectionRef: "weights/cls_proj.bin",
            expansionRef: "weights/cls_expand.bin"
        ),
        draft: .init(
            kind: .exactTwoToken,
            behaviorClass: .exact,
            horizon: 2,
            verifier: "exact",
            rollback: "exact_replay",
            artifactRef: "weights/future-sidecar.bin",
            acceptanceMetric: "accepted_future_tokens"
        ),
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/content-hashes.json"
    )

    _ = try ESPBundleArchive.create(
        at: bundle,
        manifest: manifest,
        weightsDirectory: weights,
        tokenizerDirectory: tokenizer
    )

    do {
        _ = try ESPBundleArchive.open(at: bundle)
        #expect(Bool(false), "Expected missing referenced artifact rejection")
    } catch let error as ESPBundleValidationError {
        #expect(error == .missingReferencedFile("weights/future-sidecar.bin"))
    } catch {
        #expect(Bool(false), "Unexpected error: \(error)")
    }
}

@Test func bundleCreateAndOpenRoundTrip() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    let weights = root.appendingPathComponent("weights-src", isDirectory: true)
    let tokenizer = root.appendingPathComponent("tokenizer-src", isDirectory: true)
    let bundle = root.appendingPathComponent("model.esp", isDirectory: true)

    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: weights.appendingPathComponent("metadata.json"))
    try Data("weights".utf8).write(to: weights.appendingPathComponent("lm_head.bin"))
    try Data("tokenizer".utf8).write(to: tokenizer.appendingPathComponent("tokenizer.json"))

    let manifest = ESPManifest(
        formatVersion: "1.1.0",
        modelID: "espresso.llama.test",
        modelFamily: .llama,
        architectureVersion: "decoder-v1",
        tokenizerContract: "sentencepiece-v1",
        supportedBackends: [.anePrivate, .cpuSafe],
        supportedProfiles: [.prefill256, .decode1],
        maxContext: 2048,
        contextTargetTokens: 1024,
        compressionPolicy: .init(name: "fp16", weightBits: 16, activationBits: nil),
        modelTier: .optimized,
        behaviorClass: .exact,
        adapterSlots: 0,
        optimization: .init(
            recipe: "stories-ctx1024",
            qualityGate: "short-long-prompt-parity",
            teacherModel: nil,
            draftModel: nil,
            performanceTarget: "105 tok/s"
        ),
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/content-hashes.json"
    )

    _ = try ESPBundleArchive.create(
        at: bundle,
        manifest: manifest,
        weightsDirectory: weights,
        tokenizerDirectory: tokenizer
    )

    let opened = try ESPBundleArchive.open(at: bundle)
    #expect(opened.manifest == manifest)
    #expect(FileManager.default.fileExists(atPath: opened.signatureCatalogURL.path))
}

@Test func bundleOpenRejectsTamperedFileContent() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    let weights = root.appendingPathComponent("weights-src", isDirectory: true)
    let tokenizer = root.appendingPathComponent("tokenizer-src", isDirectory: true)
    let bundle = root.appendingPathComponent("model.esp", isDirectory: true)

    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: tokenizer, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: weights.appendingPathComponent("metadata.json"))
    try Data("weights".utf8).write(to: weights.appendingPathComponent("lm_head.bin"))
    try Data("tokenizer".utf8).write(to: tokenizer.appendingPathComponent("tokenizer.json"))

    let manifest = ESPManifest(
        formatVersion: "1.1.0",
        modelID: "espresso.llama.test",
        modelFamily: .llama,
        architectureVersion: "decoder-v1",
        tokenizerContract: "sentencepiece-v1",
        supportedBackends: [.anePrivate, .cpuSafe],
        supportedProfiles: [.prefill256, .decode1],
        maxContext: 2048,
        contextTargetTokens: 1024,
        compressionPolicy: .init(name: "fp16", weightBits: 16, activationBits: nil),
        modelTier: .optimized,
        behaviorClass: .exact,
        adapterSlots: 0,
        optimization: .init(
            recipe: "stories-ctx1024",
            qualityGate: "short-long-prompt-parity",
            teacherModel: nil,
            draftModel: nil,
            performanceTarget: "105 tok/s"
        ),
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/content-hashes.json"
    )

    let archive = try ESPBundleArchive.create(
        at: bundle,
        manifest: manifest,
        weightsDirectory: weights,
        tokenizerDirectory: tokenizer
    )
    try Data("tampered".utf8).write(to: archive.weightsURL.appendingPathComponent("lm_head.bin"))

    do {
        _ = try ESPBundleArchive.open(at: bundle)
        #expect(Bool(false), "Expected signature verification to fail")
    } catch let error as ESPBundleValidationError {
        #expect(error == .signatureMismatch(path: "weights/lm_head.bin"))
    } catch {
        #expect(Bool(false), "Unexpected error: \(error)")
    }
}

import Foundation
import Testing
import ANETypes
import ESPBenchSupport
import ESPBundle
import ModelSupport
@testable import EspressoGenerate

@Test func test_normalizeHiddenForCoreMLReferenceUsesLayerNormForGPT2() {
    let hidden: [Float] = [1, 2, 3, 4]
    let gamma: [Float] = [1, 1, 1, 1]
    let beta: [Float] = [0.5, 0.5, 0.5, 0.5]
    let output = normalizeHiddenForCoreMLReference(
        architecture: .gpt2,
        hidden: hidden,
        epsilon: 1e-5,
        gamma: gamma,
        beta: beta
    )

    let mean: Float = 2.5
    let variance: Float = 1.25
    let inverseStd = 1.0 / sqrtf(variance + 1e-5)
    let expected = hidden.map { (($0 - mean) * inverseStd) + 0.5 }

    for index in output.indices {
        #expect(abs(output[index] - expected[index]) < 1e-5)
    }
}

@Test func test_normalizeHiddenForCoreMLReferenceUsesRMSNormForLlama() {
    let hidden: [Float] = [1, 2, 3, 4]
    let gamma: [Float] = [1, 1, 1, 1]
    let output = normalizeHiddenForCoreMLReference(
        architecture: .llama,
        hidden: hidden,
        epsilon: 1e-5,
        gamma: gamma,
        beta: nil
    )

    let rms = sqrtf((1 + 4 + 9 + 16) / 4 + 1e-5)
    let expected = hidden.map { $0 / rms }

    for index in output.indices {
        #expect(abs(output[index] - expected[index]) < 1e-5)
    }
}

@Test func test_optionsParseSubcommandsAndFlags() throws {
    let options = try Options.parse([
        "espresso-generate",
        "compare",
        "--bench",
        "--power",
        "--output-dir", "/tmp/report",
        "--seed", "77",
        "--max-tokens", "16",
        "Hello",
    ])

    #expect(options.command == .compare)
    #expect(options.preferBenchCompare)
    #expect(!options.preferLiveCompare)
    #expect(options.powerMode == .on)
    #expect(options.outputDir == "/tmp/report")
    #expect(options.seed == 77)
    #expect(options.maxTokens == 16)
    #expect(options.positionalPrompt == ["Hello"])
}

@Test func test_optionsParseSuiteFlags() throws {
    let options = try Options.parse([
        "espresso-generate",
        "suite",
        "--prompts", "/tmp/prompts.txt",
        "--runs", "4",
        "--results-tsv", "/tmp/results.tsv",
        "--no-cold",
        "--compare-warmup", "2",
        "--compare-iterations", "5",
    ])

    #expect(options.command == .suite)
    #expect(options.promptsFile == "/tmp/prompts.txt")
    #expect(options.suiteRuns == 4)
    #expect(options.resultsTSV == "/tmp/results.tsv")
    #expect(!options.includeColdRun)
    #expect(options.compareWarmup == 2)
    #expect(options.compareIterations == 5)
}

@Test func test_optionsParseGenerateBenchmarkFlags() throws {
    let options = try Options.parse([
        "espresso-generate",
        "generate",
        "--benchmark-generate",
        "--compare-warmup", "2",
        "--compare-iterations", "4",
        "--json",
        "Hello",
    ])

    #expect(options.command == .generate)
    #expect(options.benchmarkGenerate)
    #expect(options.compareWarmup == 2)
    #expect(options.compareIterations == 4)
    #expect(options.jsonOutput)
    #expect(options.positionalPrompt == ["Hello"])
    #expect(!options.rawPrompt)
    #expect(!options.disableHybridFallback)
}

@Test func test_optionsParseRawPromptAndNoHybridFallback() throws {
    let options = try Options.parse([
        "espresso-generate",
        "generate",
        "--raw-prompt",
        "--no-hybrid-fallback",
        "--prompt", "Hello",
    ])

    #expect(options.command == .generate)
    #expect(options.rawPrompt)
    #expect(options.disableHybridFallback)
    #expect(options.prompt == "Hello")
}

@Test func test_optionsParseAttentionCompileProbeFlags() throws {
    let options = try Options.parse([
        "espresso-generate",
        "doctor",
        "--probe-gpt2-attention-compile",
        "--mil-deployment-target", "macos26",
        "--gpt2-norm", "rmsnorm",
    ])

    #expect(options.command == .doctor)
    #expect(options.probeGPT2AttentionCompile)
    #expect(options.milDeploymentTarget == "macos26")
    #expect(options.gpt2NormMode == "rmsnorm")
}

@Test func test_optionsParseBundleFlag() throws {
    let options = try Options.parse([
        "espresso-generate",
        "generate",
        "--bundle", "/tmp/model.esp",
        "--prompt", "Hello",
    ])

    #expect(options.bundlePath == "/tmp/model.esp")
    #expect(options.prompt == "Hello")
}

@Test func test_benchmarkFingerprintIncludesRequiredMetrics() {
    let metrics = BackendRunMetrics(
        backend: "ane-private",
        text: "Hello,",
        generatedTokens: [11],
        promptTokens: [1, 2, 3],
        compileTimeMs: 250,
        firstTokenLatencyMs: 12,
        tokensPerSecond: 80,
        medianTokenMs: 12,
        p95TokenMs: 12,
        totalTimeMs: 320,
        tokenLatenciesMs: [12],
        compileRetryCount: 0,
        compileFailureCount: 0
    )

    let fingerprint = benchmarkFingerprint(for: metrics)
    #expect(fingerprint.missingRequiredMetrics().isEmpty)
    #expect(fingerprint.metrics[.ttftMilliseconds] == 12)
    #expect(fingerprint.metrics[.tokensPerSecond] == 80)
}

@Test func test_backendRunMetricsCarriesDecodePath() {
    let metrics = BackendRunMetrics(
        backend: "espresso",
        text: "Paris",
        generatedTokens: [11],
        promptTokens: [1],
        compileTimeMs: 10,
        firstTokenLatencyMs: 5,
        tokensPerSecond: 20,
        medianTokenMs: 5,
        p95TokenMs: 5,
        totalTimeMs: 20,
        tokenLatenciesMs: [5],
        decodePath: "hybrid"
    )
    #expect(metrics.decodePath == "hybrid")
    #expect(metrics.hopsPerToken == nil)

    let fusedMetrics = BackendRunMetrics(
        backend: "espresso",
        text: "Paris",
        generatedTokens: [11],
        promptTokens: [1],
        compileTimeMs: 10,
        firstTokenLatencyMs: 5,
        tokensPerSecond: 20,
        medianTokenMs: 5,
        p95TokenMs: 5,
        totalTimeMs: 20,
        tokenLatenciesMs: [5],
        cachedBindingsEnabled: false,
        decodePath: "fused",
        hopsPerToken: 28
    )
    #expect(fusedMetrics.decodePath == "fused")
    #expect(fusedMetrics.hopsPerToken == 28)
    #expect(fusedMetrics.cachedBindingsEnabled == false)
}

@Test func test_generateDecodeProfileMeanLineExtractsContractLine() {
    let report = """
    decode_profile_mean_ms/token qkv=1.00 rope=2.00 attn=3.00 ffn=4.00 lm_head=5.00 io=6.00 n=31 exclude_ttft=1
    decode_profile_ttft_ms=90.00 lm_head=90.00
    decode_profile_token i=0 qkv=0.00 rope=0.00 attn=0.00 ffn=0.00 lm_head=90.00 io=0.00
    """
    #expect(
        generateDecodeProfileMeanLine(report)
            == "decode_profile_mean_ms/token qkv=1.00 rope=2.00 attn=3.00 ffn=4.00 lm_head=5.00 io=6.00 n=31 exclude_ttft=1"
    )
    #expect(generateDecodeProfileMeanLine(nil) == nil)
    #expect(generateDecodeProfileMeanLine("") == nil)
    #expect(generateDecodeProfileMeanLine("cached_bindings_enabled=false") == nil)

    #expect(
        espressoCompareLaneContractLines(
            decodePath: "fused",
            hopsPerToken: 28,
            decodeProfileReport: report
        ) == [
            "decode_path=fused",
            "hops/token=28",
            "decode_profile_mean_ms/token qkv=1.00 rope=2.00 attn=3.00 ffn=4.00 lm_head=5.00 io=6.00 n=31 exclude_ttft=1",
        ]
    )
    #expect(espressoCompareLaneContractLines(decodePath: nil, hopsPerToken: nil, decodeProfileReport: nil).isEmpty)

    let metrics = BackendRunMetrics(
        backend: "espresso",
        text: "Paris",
        generatedTokens: [11],
        promptTokens: [1],
        compileTimeMs: 10,
        firstTokenLatencyMs: 5,
        tokensPerSecond: 20,
        medianTokenMs: 5,
        p95TokenMs: 5,
        totalTimeMs: 20,
        tokenLatenciesMs: [5],
        decodePath: "hybrid",
        decodeProfileReport: report
    )
    #expect(metrics.decodeProfileReport == report)
}

@Test func test_metadataConfigFilePreservesOptionalRopeThetaAndEOSToken() throws {
    let metadata = MetadataConfigFile(
        name: "qwen3",
        nLayer: 28,
        nHead: 16,
        nKVHead: 8,
        dModel: 1024,
        headDim: 128,
        hiddenDim: 3072,
        vocab: 151936,
        maxSeq: 40960,
        normEps: 1e-6,
        ropeTheta: 1_000_000,
        eosToken: 151645,
        architecture: "llama",
        preferredDecodePath: nil
    )

    let config = try metadata.asConfig()
    #expect(config.ropeTheta == 1_000_000)
    #expect(config.eosToken == 151645)
    #expect(config.preferredDecodePath == nil)
}

@Test func test_metadataConfigFileParsesPreferredDecodePath() throws {
    let hybrid = MetadataConfigFile(
        name: "Qwen2.5-0.5B-Instruct",
        nLayer: 24,
        nHead: 14,
        nKVHead: 2,
        dModel: 896,
        headDim: 64,
        hiddenDim: 4864,
        vocab: 151936,
        maxSeq: 4096,
        normEps: 1e-6,
        ropeTheta: 1_000_000,
        eosToken: 151645,
        architecture: "llama",
        preferredDecodePath: " Hybrid "
    )
    #expect(try hybrid.asConfig().preferredDecodePath == .hybrid)

    let exact = MetadataConfigFile(
        name: "Qwen2.5-0.5B-Instruct",
        nLayer: 24,
        nHead: 14,
        nKVHead: 2,
        dModel: 896,
        headDim: 64,
        hiddenDim: 4864,
        vocab: 151936,
        maxSeq: 4096,
        normEps: 1e-6,
        ropeTheta: 1_000_000,
        eosToken: 151645,
        architecture: "llama",
        preferredDecodePath: "EXACT_CPU"
    )
    #expect(try exact.asConfig().preferredDecodePath == .exactCPU)

    let invalid = MetadataConfigFile(
        name: "Qwen2.5-0.5B-Instruct",
        nLayer: 24,
        nHead: 14,
        nKVHead: 2,
        dModel: 896,
        headDim: 64,
        hiddenDim: 4864,
        vocab: 151936,
        maxSeq: 4096,
        normEps: 1e-6,
        ropeTheta: 1_000_000,
        eosToken: 151645,
        architecture: "llama",
        preferredDecodePath: "metal"
    )
    #expect(throws: MultiModelConfig.PreferredDecodePath.ParseError.unsupported("metal")) {
        try invalid.asConfig()
    }
}

@Test func test_preparedGeneratePromptWrapsQwenUnlessRawPrompt() {
    let qwen = MultiModelConfig(
        name: "Qwen2.5-0.5B-Instruct",
        nLayer: 24,
        nHead: 14,
        nKVHead: 2,
        dModel: 896,
        headDim: 64,
        hiddenDim: 4864,
        vocab: 151_936,
        maxSeq: 4096,
        normEps: 1e-6,
        architecture: .llama
    )
    #expect(
        preparedGeneratePrompt("Hello", config: qwen, rawPrompt: false)
            == QwenInstructPrompt.wrapUserTurn("Hello")
    )
    #expect(preparedGeneratePrompt("Hello", config: qwen, rawPrompt: true) == "Hello")
    #expect(preparedGeneratePrompt("Hello", config: ModelRegistry.stories110m, rawPrompt: false) == "Hello")
}

@Test func test_resolveCoreMLModelPathUsesExplicitPathForLlama() throws {
    let fileManager = FileManager.default
    let root = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    let weightsDir = root.appendingPathComponent("weights", isDirectory: true)
    let tokenizerDir = root.appendingPathComponent("tokenizer", isDirectory: true)
    let explicitModel = root.appendingPathComponent("llama3_2_1b.mlpackage", isDirectory: true)
    try fileManager.createDirectory(at: weightsDir, withIntermediateDirectories: true)
    try fileManager.createDirectory(at: tokenizerDir, withIntermediateDirectories: true)
    try fileManager.createDirectory(at: explicitModel, withIntermediateDirectories: true)

    let defaults = DemoDefaults(
        repoRoot: root,
        workingDirectory: root,
        stateRoot: root,
        cacheRoot: root,
        reportsRoot: root,
        hfCacheRoot: root,
        weightsDir: weightsDir,
        tokenizerDir: tokenizerDir,
        coreMLDir: root,
        toolsVenvDir: root,
        scriptsDir: nil,
        legacyArtifactsRoot: nil
    )
    let invocation = ResolvedInvocation(
        config: try #require(ModelRegistry.config(named: "llama3_2_1b")),
        bundlePath: nil,
        weightsDir: weightsDir.path,
        tokenizerDir: tokenizerDir.path,
        prompt: "",
        maxTokens: 16,
        temperature: 0,
        showStats: false,
        coreMLModelPath: explicitModel.path,
        coreMLSequenceLength: nil,
        compareWarmup: 0,
        compareIterations: 1,
        coreMLComputeUnits: "cpu_only",
        allowBootstrap: false,
        seed: 1234,
        outputDir: nil
    )

    let resolved = try resolveCoreMLModelPath(invocation: invocation, defaults: defaults, sequenceLength: 64)
    #expect(resolved == explicitModel.standardizedFileURL.path)

    try? fileManager.removeItem(at: root)
}

@Test func test_sentencePieceTokenizerURLPrefersTokenizerModel() throws {
    let fileManager = FileManager.default
    let root = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
    let modelURL = root.appendingPathComponent("tokenizer.model")
    let binURL = root.appendingPathComponent("tokenizer.bin")
    fileManager.createFile(atPath: modelURL.path, contents: Data([0x01]))
    fileManager.createFile(atPath: binURL.path, contents: Data([0x02]))

    let resolved = sentencePieceTokenizerURL(in: root)
    #expect(resolved == modelURL)

    try? fileManager.removeItem(at: root)
}
@Test func test_resolveCoreMLModelPathRejectsLlamaWithoutExplicitModel() throws {
    let fileManager = FileManager.default
    let root = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    let weightsDir = root.appendingPathComponent("weights", isDirectory: true)
    let tokenizerDir = root.appendingPathComponent("tokenizer", isDirectory: true)
    try fileManager.createDirectory(at: weightsDir, withIntermediateDirectories: true)
    try fileManager.createDirectory(at: tokenizerDir, withIntermediateDirectories: true)

    let defaults = DemoDefaults(
        repoRoot: root,
        workingDirectory: root,
        stateRoot: root,
        cacheRoot: root,
        reportsRoot: root,
        hfCacheRoot: root,
        weightsDir: weightsDir,
        tokenizerDir: tokenizerDir,
        coreMLDir: root,
        toolsVenvDir: root,
        scriptsDir: nil,
        legacyArtifactsRoot: nil
    )
    let invocation = ResolvedInvocation(
        config: try #require(ModelRegistry.config(named: "llama3_2_1b")),
        bundlePath: nil,
        weightsDir: weightsDir.path,
        tokenizerDir: tokenizerDir.path,
        prompt: "",
        maxTokens: 16,
        temperature: 0,
        showStats: false,
        coreMLModelPath: nil,
        coreMLSequenceLength: nil,
        compareWarmup: 0,
        compareIterations: 1,
        coreMLComputeUnits: "cpu_only",
        allowBootstrap: false,
        seed: 1234,
        outputDir: nil
    )

    do {
        _ = try resolveCoreMLModelPath(invocation: invocation, defaults: defaults, sequenceLength: 64)
        Issue.record("Expected llama Core ML path resolution to require --coreml-model")
    } catch let error as CLIError {
        #expect(error.localizedDescription.contains("--coreml-model"))
    }

    try? fileManager.removeItem(at: root)
}

@Test func test_resolveCoreMLModelPathUsesExplicitOverrideForGPT2() throws {
    let fileManager = FileManager.default
    let root = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    let weightsDir = root.appendingPathComponent("weights", isDirectory: true)
    let tokenizerDir = root.appendingPathComponent("tokenizer", isDirectory: true)
    let explicitModel = root.appendingPathComponent("gpt2_seq128.mlpackage", isDirectory: true)
    try fileManager.createDirectory(at: weightsDir, withIntermediateDirectories: true)
    try fileManager.createDirectory(at: tokenizerDir, withIntermediateDirectories: true)
    try fileManager.createDirectory(at: explicitModel, withIntermediateDirectories: true)

    let defaults = DemoDefaults(
        repoRoot: root,
        workingDirectory: root,
        stateRoot: root,
        cacheRoot: root,
        reportsRoot: root,
        hfCacheRoot: root,
        weightsDir: weightsDir,
        tokenizerDir: tokenizerDir,
        coreMLDir: root,
        toolsVenvDir: root,
        scriptsDir: nil,
        legacyArtifactsRoot: nil
    )
    let invocation = ResolvedInvocation(
        config: try #require(ModelRegistry.config(named: "gpt2_124m")),
        bundlePath: nil,
        weightsDir: weightsDir.path,
        tokenizerDir: tokenizerDir.path,
        prompt: "",
        maxTokens: 16,
        temperature: 0,
        showStats: false,
        coreMLModelPath: explicitModel.path,
        coreMLSequenceLength: nil,
        compareWarmup: 0,
        compareIterations: 1,
        coreMLComputeUnits: "cpu_only",
        allowBootstrap: true,
        seed: 1234,
        outputDir: nil
    )

    let resolved = try resolveCoreMLModelPath(invocation: invocation, defaults: defaults, sequenceLength: 128)
    #expect(resolved == explicitModel.standardizedFileURL.path)

    try? fileManager.removeItem(at: root)
}

@Test func test_shouldUseDefaultGPT2DemoWhenNoWeightsProvided() {
    let options = Options()
    #expect(shouldUseDefaultGPT2Demo(options))

    var explicit = Options()
    explicit.weightsDir = "/tmp/weights"
    #expect(!shouldUseDefaultGPT2Demo(explicit))
}

@Test func test_shouldUseDefaultGPT2DemoIsDisabledForBundleInput() {
    var options = Options()
    options.bundlePath = "/tmp/model.esp"
    #expect(!shouldUseDefaultGPT2Demo(options))
}

@Test func test_shouldUseDefaultGPT2DemoIsDisabledForChat() {
    var options = Options()
    options.command = .chat
    #expect(!shouldUseDefaultGPT2Demo(options))
}

@Test func test_demoDefaultsTreatReferenceRunnerAsOptional() throws {
    let fileManager = FileManager.default
    let root = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    let scripts = root.appendingPathComponent("scripts", isDirectory: true)
    try fileManager.createDirectory(at: scripts, withIntermediateDirectories: true)
    try Data().write(to: scripts.appendingPathComponent("bootstrap_gpt2_demo.py"))
    try Data().write(to: scripts.appendingPathComponent("export_gpt2_coreml.py"))

    let defaults = DemoDefaults(
        repoRoot: root,
        workingDirectory: root,
        stateRoot: root,
        cacheRoot: root,
        reportsRoot: root,
        hfCacheRoot: root,
        weightsDir: root,
        tokenizerDir: root,
        coreMLDir: root,
        toolsVenvDir: root,
        scriptsDir: scripts,
        legacyArtifactsRoot: nil
    )

    #expect(defaults.bootstrapScriptAvailable)
    #expect(defaults.exportScriptAvailable)
    #expect(defaults.scriptsAvailable)
    #expect(!defaults.referenceScriptAvailable)
}

@Test func test_implicitPromptDefaultsDemoAndCompareToHelloForManagedDemo() {
    let options = Options()

    #expect(implicitPrompt(command: .demo, options: options) == "Hello")
    #expect(implicitPrompt(command: .compare, options: options) == "Hello")
    #expect(implicitPrompt(command: .generate, options: options) == nil)

    var explicit = Options()
    explicit.weightsDir = "/tmp/weights"
    #expect(implicitPrompt(command: .demo, options: explicit) == nil)
}

/// Writes a minimal but structurally valid `.esp` bundle and returns its URL.
private func makeStubBundle(
    at root: URL,
    named name: String = "stories.esp",
    preferredDecodePath: String? = nil
) throws -> URL {
    let fileManager = FileManager.default
    let weightsDir = root.appendingPathComponent("weights-src-\(name)", isDirectory: true)
    let tokenizerDir = root.appendingPathComponent("tokenizer-src-\(name)", isDirectory: true)
    let bundleURL = root.appendingPathComponent(name, isDirectory: true)
    try fileManager.createDirectory(at: weightsDir, withIntermediateDirectories: true)
    try fileManager.createDirectory(at: tokenizerDir, withIntermediateDirectories: true)
    let decodePathEntry = preferredDecodePath.map { ",\n  \"preferredDecodePath\": \"\($0)\"" } ?? ""
    try """
    {
      "name": "llama2.c-stories110M",
      "nLayer": 12,
      "nHead": 12,
      "nKVHead": 12,
      "dModel": 768,
      "headDim": 64,
      "hiddenDim": 2048,
      "vocab": 32000,
      "maxSeq": 256,
      "normEps": 0.00001,
      "architecture": "llama"\(decodePathEntry)
    }
    """.write(to: weightsDir.appendingPathComponent("metadata.json"), atomically: true, encoding: .utf8)
    try Data("weights".utf8).write(to: weightsDir.appendingPathComponent("lm_head.bin"))
    try Data("tokenizer".utf8).write(to: tokenizerDir.appendingPathComponent("tokenizer.model"))

    let manifest = ESPManifest(
        formatVersion: "1.0.0",
        modelID: "llama2.c-stories110M",
        modelFamily: .llama,
        architectureVersion: "decoder-v1",
        tokenizerContract: "sentencepiece-v1",
        supportedBackends: [.anePrivate, .cpuSafe],
        supportedProfiles: [.prefill256, .decode1],
        maxContext: 256,
        compressionPolicy: .init(name: "native-ane-fp16", weightBits: 16, activationBits: nil),
        adapterSlots: 0,
        accuracyBaselineRef: "benchmarks/accuracy.json",
        performanceBaselineRef: "benchmarks/perf.json",
        signatureRef: "signatures/content-hashes.json"
    )
    _ = try ESPBundleArchive.create(
        at: bundleURL,
        manifest: manifest,
        weightsDirectory: weightsDir,
        tokenizerDirectory: tokenizerDir
    )
    return bundleURL
}

private func makeStubDemoDefaults(root: URL) -> DemoDefaults {
    DemoDefaults(
        repoRoot: root,
        workingDirectory: root,
        stateRoot: root,
        cacheRoot: root,
        reportsRoot: root,
        hfCacheRoot: root,
        weightsDir: root,
        tokenizerDir: root,
        coreMLDir: root,
        toolsVenvDir: root,
        scriptsDir: nil,
        legacyArtifactsRoot: nil
    )
}

@Test func test_resolveInvocationUsesBundlePathsWhenBundleProvided() throws {
    let fileManager = FileManager.default
    let root = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
    let bundleURL = try makeStubBundle(at: root)

    let defaults = DemoDefaults(
        repoRoot: root,
        workingDirectory: root,
        stateRoot: root,
        cacheRoot: root,
        reportsRoot: root,
        hfCacheRoot: root,
        weightsDir: root,
        tokenizerDir: root,
        coreMLDir: root,
        toolsVenvDir: root,
        scriptsDir: nil,
        legacyArtifactsRoot: nil
    )
    var options = Options()
    options.bundlePath = bundleURL.path
    options.prompt = "Hello"

    let invocation = try resolveInvocation(from: options, demoDefaults: defaults, command: .generate)
    #expect(invocation.bundlePath == bundleURL.path)
    #expect(invocation.weightsDir == bundleURL.appendingPathComponent("weights", isDirectory: true).path)
    #expect(invocation.tokenizerDir == bundleURL.appendingPathComponent("tokenizer", isDirectory: true).path)
    #expect(invocation.config.name == "llama2.c-stories110M")
}

/// `--model <path-to.esp>` is the documented README command, so it must resolve the same
/// way `--bundle` does rather than being looked up as a registry key.
@Test func test_resolveInvocationTreatsModelPathEndingInESPAsBundle() throws {
    let fileManager = FileManager.default
    let root = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
    let bundleURL = try makeStubBundle(at: root, named: "qwen.esp", preferredDecodePath: "hybrid")

    var options = Options()
    options.modelName = bundleURL.path
    options.prompt = "Hello"

    let invocation = try resolveInvocation(
        from: options,
        demoDefaults: makeStubDemoDefaults(root: root),
        command: .generate
    )
    #expect(invocation.bundlePath == bundleURL.path)
    #expect(invocation.weightsDir == bundleURL.appendingPathComponent("weights", isDirectory: true).path)
    // A packed bundle must not lose the artifact's declared decode path, otherwise the
    // runtime would silently route it to the pure-CPU oracle.
    #expect(invocation.config.preferredDecodePath == .hybrid)
}

@Test func test_bundlePathFromModelArgumentLeavesRegistryNamesAlone() throws {
    #expect(try bundlePathFromModelArgument("gpt2") == nil)
    #expect(try bundlePathFromModelArgument("") == nil)
}

@Test func test_bundlePathFromModelArgumentReportsMissingAndMalformedBundles() throws {
    let fileManager = FileManager.default
    let root = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try fileManager.createDirectory(at: root, withIntermediateDirectories: true)

    let missing = root.appendingPathComponent("absent.esp", isDirectory: true)
    do {
        _ = try bundlePathFromModelArgument(missing.path)
        Issue.record("Expected a missing bundle to be rejected")
    } catch let error as CLIError {
        guard case let .usage(message) = error else {
            Issue.record("Expected usage error, got \(error)")
            return
        }
        #expect(message.contains("not found"))
    }

    let empty = root.appendingPathComponent("empty.esp", isDirectory: true)
    try fileManager.createDirectory(at: empty, withIntermediateDirectories: true)
    do {
        _ = try bundlePathFromModelArgument(empty.path)
        Issue.record("Expected a bundle without a manifest to be rejected")
    } catch let error as CLIError {
        guard case let .usage(message) = error else {
            Issue.record("Expected usage error, got \(error)")
            return
        }
        #expect(message.contains("manifest.toml"))
    }
}

@Test func test_resolveInvocationRejectsBundleMixedWithWeights() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let defaults = DemoDefaults(
        repoRoot: root,
        workingDirectory: root,
        stateRoot: root,
        cacheRoot: root,
        reportsRoot: root,
        hfCacheRoot: root,
        weightsDir: root,
        tokenizerDir: root,
        coreMLDir: root,
        toolsVenvDir: root,
        scriptsDir: nil,
        legacyArtifactsRoot: nil
    )
    var options = Options()
    options.bundlePath = "/tmp/model.esp"
    options.weightsDir = "/tmp/weights"
    options.prompt = "Hello"

    do {
        _ = try resolveInvocation(from: options, demoDefaults: defaults, command: .generate)
        Issue.record("Expected bundle + weights to be rejected")
    } catch let error as CLIError {
        guard case let .usage(message) = error else {
            Issue.record("Expected usage error, got \(error)")
            return
        }
        #expect(message.contains("--bundle"))
    }
}

@Test func test_powerCapabilityProbeTargetsPowermetricsNotGenericTrue() {
    #expect(PowerTelemetryCollector.capabilityProbeArguments.contains("/usr/bin/powermetrics"))
    #expect(!PowerTelemetryCollector.capabilityProbeArguments.contains("/usr/bin/true"))
}

@Test func test_parsePowermetricsSamplesParsesWattsAndMilliwatts() {
    let log = """
    CPU Power: 850 mW
    GPU Power: 0.20 W
    ANE Power: 1.60 W
    Combined Power: 3.25 W

    CPU Power: 900 mW
    GPU Power: 210 mW
    ANE Power: 1.55 W
    Package Power: 3.10 W

    """

    let samples = parsePowermetricsSamples(from: log)
    #expect(samples.count == 2)
    #expect(samples[0].cpuW == 0.85)
    #expect(samples[0].gpuW == 0.2)
    #expect(samples[0].aneW == 1.6)
    #expect(samples[0].packageW == 3.25)
    #expect(samples[1].cpuW == 0.9)
    #expect(samples[1].gpuW == 0.21)
    #expect(samples[1].aneW == 1.55)
    #expect(samples[1].packageW == 3.10)
}

@Test func test_parsePowermetricsSamplesKeepsANEDistinctFromCombinedPackage() {
    let log = """
    CPU Power: 2193 mW
    GPU Power: 19 mW
    ANE Power: 0 mW
    Combined Power (CPU + GPU + ANE): 2211 mW

    """

    let samples = parsePowermetricsSamples(from: log)
    #expect(samples.count == 1)
    #expect(samples[0].cpuW == 2.193)
    #expect(samples[0].gpuW == 0.019)
    #expect(samples[0].aneW == 0)
    #expect(samples[0].packageW == 2.211)
    #expect(samples[0].aneW != samples[0].packageW)
}

@Test func test_liveCompareRendererProducesSideBySideLayout() {
    var espresso = LiveLaneSnapshot(title: "Espresso / ANE", maxTokens: 32)
    espresso.status = .generating
    espresso.generatedTokenCount = 7
    espresso.lastToken = "but"
    espresso.text = "Hello, I'm sorry, but I'm"
    espresso.tokensPerSecond = 24.8
    espresso.ttftMs = 118
    espresso.compileMs = 842
    espresso.medianTokenMs = 41
    espresso.p95TokenMs = 56
    espresso.totalMs = 294
    espresso.power = PowerSummary(packageW: 4.2, cpuW: 0.9, gpuW: 0.2, aneW: 2.4, sampleCount: 3)

    var coreML = LiveLaneSnapshot(title: "Core ML", maxTokens: 32)
    coreML.status = .generating
    coreML.generatedTokenCount = 7
    coreML.lastToken = "but"
    coreML.text = "Hello, I'm sorry, but I'm"
    coreML.tokensPerSecond = 10.7
    coreML.ttftMs = 243
    coreML.compileMs = 1461
    coreML.medianTokenMs = 92
    coreML.p95TokenMs = 110
    coreML.totalMs = 651
    coreML.power = PowerSummary(packageW: 5.8, cpuW: 1.6, gpuW: 0.4, aneW: 1.1, sampleCount: 3)

    let snapshot = LiveCompareSnapshot(
        modelName: "gpt2_124m",
        prompt: "Hello",
        maxTokens: 32,
        elapsedMs: 1420,
        espresso: espresso,
        coreML: coreML,
        livePower: PowerSummary(packageW: 6.2, cpuW: 1.1, gpuW: 0.3, aneW: 2.8, sampleCount: 1),
        matchCount: 7,
        totalComparedTokens: 7,
        events: ["[Espresso] token 1 -> ,", "[Core ML] token 1 -> ,"]
    )

    let rendered = LiveCompareRenderer().render(snapshot: snapshot, size: TerminalSize(width: 140, height: 40))
    #expect(rendered.contains("ESPRESSO vs CORE ML LIVE GPT-2"))
    #expect(rendered.contains("ESPRESSO / ANE"))
    #expect(rendered.contains("CORE ML"))
    #expect(rendered.contains("TOKENS / SEC"))
    #expect(rendered.contains("POWER"))
    #expect(rendered.contains("Espresso preflight avg"))
    #expect(rendered.contains("Hello, I'm sorry, but I'm"))
}

@Test func test_aggregateBenchmarkRunsUsesWarmupAndAggregatesMeasuredLatencySamples() throws {
    var callCount = 0

    let result = try aggregateBenchmarkRuns(warmup: 1, iterations: 2) {
        callCount += 1
        let compileTimeMs = callCount == 1 ? 320.0 : 0.0
        let latencies: [Double]
        switch callCount {
        case 1:
            latencies = [1, 1]
        case 2:
            latencies = [3, 3]
        default:
            latencies = [7, 7]
        }
        return BackendRunMetrics(
            backend: "espresso",
            text: "Hello \(callCount)",
            generatedTokens: [TokenID(callCount)],
            promptTokens: [0],
            compileTimeMs: compileTimeMs,
            firstTokenLatencyMs: Double(100 + callCount),
            tokensPerSecond: Double(10 * callCount),
            medianTokenMs: 0,
            p95TokenMs: 0,
            totalTimeMs: Double(200 * callCount),
            tokenLatenciesMs: latencies
        )
    }

    #expect(callCount == 3)
    #expect(result.compileTimeMs == 320.0)
    #expect(result.firstTokenLatencyMs == 103.0)
    #expect(result.tokensPerSecond == 30.0)
    #expect(result.totalTimeMs == 600.0)
    #expect(result.medianTokenMs == 5.0)
    #expect(abs(result.p95TokenMs - 7.0) < 0.0001)
    #expect(result.tokenLatenciesMs == [7, 7])
}

@Test func test_resolvePowerEnabledRequiresCapabilityWhenExplicitlyRequested() {
    do {
        _ = try resolvePowerEnabled(
            command: .bench,
            powerMode: .on,
            capability: PowerCapability(available: false, message: "powermetrics unavailable")
        )
        Issue.record("Expected --power to fail when telemetry is unavailable")
    } catch let error as CLIError {
        guard case let .runtime(message) = error else {
            Issue.record("Expected runtime error, got \(error)")
            return
        }
        #expect(message.contains("powermetrics unavailable"))
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
}

@Test func test_resolvePowerEnabledAutoOnlyEnablesDefaultCommandsWhenCapabilityExists() throws {
    #expect(
        try resolvePowerEnabled(
            command: .bench,
            powerMode: .auto,
            capability: PowerCapability(available: true, message: "ready")
        )
    )
    #expect(
        !(try resolvePowerEnabled(
            command: .compare,
            powerMode: .auto,
            capability: PowerCapability(available: true, message: "ready")
        ))
    )
    #expect(
        !(try resolvePowerEnabled(
            command: .demo,
            powerMode: .auto,
            capability: PowerCapability(available: false, message: "missing"),
            emitWarnings: false
        ))
    )
    #expect(
        try resolvePowerEnabled(
            command: .chat,
            powerMode: .auto,
            capability: PowerCapability(available: true, message: "ready")
        )
    )
    #expect(
        !(try resolvePowerEnabled(
            command: .chat,
            powerMode: .auto,
            capability: PowerCapability(available: false, message: "powermetrics requires passwordless sudo or root"),
            emitWarnings: false
        ))
    )
}

@Test func test_resolvedANECompileCachePolicyDefaultsToPreferCachedWhenUnset() {
    #expect(resolvedANECompileCachePolicy(environment: [:]) == "preferCached")
    #expect(resolvedANECompileCachePolicy(environment: ["ANE_COMPILE_CACHE_POLICY": ""]) == "preferCached")
}

@Test func test_resolvedANECompileCachePolicyPreservesExplicitEnvironmentValue() {
    #expect(
        resolvedANECompileCachePolicy(environment: ["ANE_COMPILE_CACHE_POLICY": "forceRebuild"]) == "forceRebuild"
    )
    #expect(
        resolvedANECompileCachePolicy(environment: ["ANE_COMPILE_CACHE_POLICY": "preferCached"]) == "preferCached"
    )
}

@Test func test_loadPromptSuiteParsesCommentsAndPrompts() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let promptsURL = root.appendingPathComponent("prompts.txt")
    try """
    # benchmark prompts
    intro:Hello there

    story:Once upon a time: the lights flickered
    """.write(to: promptsURL, atomically: true, encoding: .utf8)

    let prompts = try loadPromptSuite(from: promptsURL.path)
    #expect(prompts == [
        PromptSuiteEntry(id: "intro", text: "Hello there"),
        PromptSuiteEntry(id: "story", text: "Once upon a time: the lights flickered"),
    ])
}

@Test func test_loadPromptSuiteRejectsDuplicateIDs() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let promptsURL = root.appendingPathComponent("prompts.txt")
    try """
    intro:Hello
    intro:World
    """.write(to: promptsURL, atomically: true, encoding: .utf8)

    do {
        _ = try loadPromptSuite(from: promptsURL.path)
        Issue.record("Expected duplicate prompt IDs to fail")
    } catch let error as CLIError {
        guard case let .usage(message) = error else {
            Issue.record("Expected usage error, got \(error)")
            return
        }
        #expect(message.contains("Duplicate prompt id"))
    }
}

@Test func test_resolvedSuiteCoreMLSequenceLengthRoundsAndClamps() throws {
    #expect(
        try resolvedSuiteCoreMLSequenceLength(
            explicitSequenceLength: nil,
            promptTokenCounts: [8, 23],
            maxTokens: 64,
            maxModelSequenceLength: 256
        ) == 128
    )
    #expect(
        try resolvedSuiteCoreMLSequenceLength(
            explicitSequenceLength: nil,
            promptTokenCounts: [129],
            maxTokens: 64,
            maxModelSequenceLength: 200
        ) == 193
    )
}

@Test func test_makePromptSuiteSummaryAggregatesPerPromptVerdicts() {
    let promptOrder = [
        PromptSuiteEntry(id: "alpha", text: "Hello"),
        PromptSuiteEntry(id: "beta", text: "World"),
    ]
    let reports = [
        PromptSuiteRunRecord(
            promptID: "alpha",
            report: CompareReport(
                model: "gpt2_124m",
                prompt: "Hello",
                maxTokens: 16,
                seed: 1234,
                espresso: BackendRunMetrics(
                    backend: "espresso",
                    text: "Hello",
                    generatedTokens: [1],
                    promptTokens: [10],
                    compileTimeMs: 50,
                    firstTokenLatencyMs: 10,
                    tokensPerSecond: 100,
                    medianTokenMs: 10,
                    p95TokenMs: 12,
                    totalTimeMs: 20,
                    tokenLatenciesMs: [10]
                ),
                coreML: BackendRunMetrics(
                    backend: "coreml",
                    text: "Hello",
                    generatedTokens: [1],
                    promptTokens: [10],
                    compileTimeMs: 40,
                    firstTokenLatencyMs: 12,
                    tokensPerSecond: 80,
                    medianTokenMs: 12,
                    p95TokenMs: 14,
                    totalTimeMs: 24,
                    tokenLatenciesMs: [12]
                ),
                tokenMatch: true,
                textMatch: true,
                coreMLComputeUnits: "cpu_and_neural_engine",
                coreMLSequenceLength: 64,
                espressoPower: nil,
                coreMLPower: nil,
                outputDirectory: "/tmp/alpha"
            )
        ),
        PromptSuiteRunRecord(
            promptID: "beta",
            report: CompareReport(
                model: "gpt2_124m",
                prompt: "World",
                maxTokens: 16,
                seed: 1234,
                espresso: BackendRunMetrics(
                    backend: "espresso",
                    text: "World a",
                    generatedTokens: [2],
                    promptTokens: [11],
                    compileTimeMs: 0,
                    firstTokenLatencyMs: 11,
                    tokensPerSecond: 90,
                    medianTokenMs: 11,
                    p95TokenMs: 13,
                    totalTimeMs: 21,
                    tokenLatenciesMs: [11]
                ),
                coreML: BackendRunMetrics(
                    backend: "coreml",
                    text: "World b",
                    generatedTokens: [3],
                    promptTokens: [11],
                    compileTimeMs: 0,
                    firstTokenLatencyMs: 13,
                    tokensPerSecond: 95,
                    medianTokenMs: 13,
                    p95TokenMs: 15,
                    totalTimeMs: 22,
                    tokenLatenciesMs: [13]
                ),
                tokenMatch: false,
                textMatch: false,
                coreMLComputeUnits: "cpu_and_neural_engine",
                coreMLSequenceLength: 64,
                espressoPower: nil,
                coreMLPower: nil,
                outputDirectory: "/tmp/beta"
            )
        ),
    ]

    let summary = makePromptSuiteSummary(
        promptOrder: promptOrder,
        reports: reports,
        commit: "abc123",
        timestamp: "2026-03-18T00:00:00Z",
        config: PromptSuiteConfig(runs: 1, warmup: 1, iterations: 3, maxTokens: 16)
    )

    #expect(summary.perPrompt.count == 2)
    #expect(summary.aggregate.nPrompts == 2)
    #expect(summary.aggregate.totalRuns == 2)
    #expect(!summary.aggregate.allTokenMatch)
    #expect(!summary.aggregate.allTextMatch)
    #expect(!summary.verdict.allCorrectnessGatesPass)
}

@Test func test_optionsParseChatFlags() throws {
    let options = try Options.parse([
        "espresso-generate",
        "chat",
        "--model", "/tmp/qwen.esp",
        "--plain",
        "--greedy",
        "--system", "Be terse.",
        "--top-p", "0.8",
        "--temperature", "0.2",
        "-n", "32",
    ])

    #expect(options.command == .chat)
    #expect(options.modelName == "/tmp/qwen.esp")
    #expect(options.plain)
    #expect(options.greedy)
    #expect(options.systemPrompt == "Be terse.")
    #expect(options.topP == 0.8)
    #expect(options.topPWasSet)
    #expect(options.temperature == 0.2)
    #expect(options.temperatureWasSet)
    #expect(options.maxTokens == 32)
    #expect(options.powerMode == .auto)
}

@Test func test_optionsParseChatPowerFlag() throws {
    let required = try Options.parse([
        "espresso-generate",
        "chat",
        "--power",
        "--model", "/tmp/qwen.esp",
    ])
    #expect(required.command == .chat)
    #expect(required.powerMode == .on)

    let disabled = try Options.parse([
        "espresso-generate",
        "chat",
        "--no-power",
        "--model", "/tmp/qwen.esp",
    ])
    #expect(disabled.powerMode == .off)
}

@Test func test_optionsParseGenerateStillWorksWithoutChatFlags() throws {
    let options = try Options.parse([
        "espresso-generate",
        "generate",
        "--model", "/tmp/qwen.esp",
        "--prompt", "Hello",
    ])

    #expect(options.command == .generate)
    #expect(!options.plain)
    #expect(!options.greedy)
    #expect(options.systemPrompt == nil)
    #expect(options.temperature == 0)
    #expect(!options.temperatureWasSet)
    #expect(options.topP == 1)
    #expect(!options.topPWasSet)
}

@Test func test_resolvedChatSamplingDefaultsAndGreedy() {
    var chat = Options()
    chat.command = .chat
    let defaults = resolvedSampling(command: .chat, options: chat)
    #expect(defaults.temperature == 0.7)
    #expect(defaults.topP == 0.9)
    #expect(!defaults.isGreedy)

    var greedy = chat
    greedy.greedy = true
    greedy.temperature = 0.2
    greedy.temperatureWasSet = true
    greedy.topP = 0.5
    greedy.topPWasSet = true
    let greedySampling = resolvedSampling(command: .chat, options: greedy)
    #expect(greedySampling.temperature == 0)
    #expect(greedySampling.topP == 1)
    #expect(greedySampling.isGreedy)

    var generate = Options()
    generate.command = .generate
    let generateSampling = resolvedSampling(command: .generate, options: generate)
    #expect(generateSampling.temperature == 0)
    #expect(generateSampling.topP == 1)
}

@Test func test_chatForcesHybridFallbackDisable() {
    var chat = Options()
    chat.command = .chat
    #expect(chatForcesHybridFallbackDisable(chat))

    var generate = Options()
    generate.command = .generate
    #expect(!chatForcesHybridFallbackDisable(generate))

    generate.disableHybridFallback = true
    #expect(!chatForcesHybridFallbackDisable(generate))
}

@Test func test_cliDecodePathOptionsForceChatFallbackDisable() {
    var chat = Options()
    chat.command = .chat
    #expect(cliDecodePathOptions(chat, environment: [:]).disableHybridFallback)

    var generate = Options()
    generate.command = .generate
    #expect(!cliDecodePathOptions(generate, environment: [:]).disableHybridFallback)

    generate.disableHybridFallback = true
    #expect(cliDecodePathOptions(generate, environment: [:]).disableHybridFallback)

    // Environment variables remain the fallback when the CLI does not own the flag.
    #expect(
        cliDecodePathOptions(
            generate,
            environment: ["ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK": "1"]
        ).disableHybridFallback
    )
}

@Test func test_implicitPromptIsNilForChat() {
    #expect(implicitPrompt(command: .chat, options: Options()) == nil)
}

@Test func test_chatSessionKeepsHistoryAndSlashCommands() {
    var session = ChatSession(system: "Be helpful.")
    #expect(session.messages.isEmpty)

    switch session.apply(.message("my name is Ada")) {
    case let .generate(prompt):
        #expect(prompt.contains("my name is Ada"))
        #expect(prompt.contains("Be helpful."))
        #expect(prompt.hasSuffix("<|im_start|>assistant\n"))
    default:
        Issue.record("Expected generate after a user turn")
    }

    session.appendAssistant("Hello Ada.")
    #expect(session.messages.count == 2)

    switch session.apply(.message("what is my name?")) {
    case let .generate(prompt):
        #expect(prompt.contains("my name is Ada"))
        #expect(prompt.contains("Hello Ada."))
        #expect(prompt.contains("what is my name?"))
    default:
        Issue.record("Expected generate to re-prefill the full history")
    }

    session.appendAssistant("Ada")
    switch session.apply(.retry) {
    case let .generate(prompt):
        #expect(prompt.contains("what is my name?"))
        #expect(!prompt.contains("<|im_start|>assistant\nAda<|im_end|>"))
        #expect(session.messages.last?.role == .user)
    default:
        Issue.record("Expected /retry to drop the last assistant turn and regenerate")
    }

    switch session.apply(.reset) {
    case .noop:
        #expect(session.messages.isEmpty)
        #expect(session.system == "Be helpful.")
    default:
        Issue.record("Expected /reset to clear history")
    }

    #expect(session.apply(.exit) == .exit)
}

@Test func test_chatSessionStripsChatMarkersFromAssistantText() {
    #expect(ChatSession.sanitizeAssistantText("Hello Ada.<|im_end|>\n<|im_start|>user") == "Hello Ada.")
}

@Test func test_chatCommandParseRecognizesSlashCommands() {
    #expect(ChatCommand.parse("/reset") == .reset)
    #expect(ChatCommand.parse("  /retry  ") == .retry)
    #expect(ChatCommand.parse("/exit") == .exit)
    #expect(ChatCommand.parse("") == .empty)
    #expect(ChatCommand.parse("hello") == .message("hello"))
}

@Test func test_chatFooterRendersLiveMetrics() {
    let footer = ChatStatusFooter(
        tokensPerSecond: 12.4,
        ttftMs: 180,
        decodePath: "hybrid",
        contextUsed: 142,
        contextMax: 1024
    )
    let rendered = footer.render()
    #expect(rendered.contains("tok/s 12.4"))
    #expect(rendered.contains("TTFT 180ms"))
    #expect(rendered.contains("path=hybrid"))
    #expect(rendered.contains("ctx 142/1024"))
    #expect(rendered.contains("power: unavailable"))
    #expect(!rendered.contains("W"))
    #expect(!rendered.contains("J/tok"))
}

@Test func test_joulesPerTokenIsPackageWattsDividedByTokensPerSecond() {
    #expect(joulesPerToken(packageWatts: 3.25, tokensPerSecond: 13) == 0.25)
    #expect(joulesPerToken(packageWatts: 4, tokensPerSecond: 8) == 0.5)
    #expect(joulesPerToken(packageWatts: 3.25, tokensPerSecond: 0) == nil)
    #expect(joulesPerToken(packageWatts: 3.25, tokensPerSecond: .nan) == nil)
}

@Test func test_chatPowerFooterUsesMeasuredSummaryForThatCompletionOnly() {
    let capability = PowerCapability(available: true, message: "powermetrics ready")
    let measured = chatPowerFooter(
        capability: capability,
        summary: PowerSummary(packageW: 3.25, cpuW: 0.85, gpuW: 0.20, aneW: 1.60, sampleCount: 4),
        tokensPerSecond: 13
    )
    guard case let .measured(packageW, cpuW, gpuW, aneW, joules) = measured else {
        Issue.record("Expected measured power for a live summary")
        return
    }
    #expect(packageW == 3.25)
    #expect(cpuW == 0.85)
    #expect(gpuW == 0.20)
    #expect(aneW == 1.60)
    #expect(joules == 0.25)

    let cleared = chatPowerFooter(capability: capability, summary: nil, tokensPerSecond: 13)
    guard case let .unavailable(message) = cleared else {
        Issue.record("Expected unavailable after a completion with no samples")
        return
    }
    #expect(message.contains("power: unavailable"))
    #expect(!message.contains("3.25"))
    #expect(!message.contains("1.60"))

    let emptySummary = chatPowerFooter(
        capability: capability,
        summary: .unavailable,
        tokensPerSecond: 13
    )
    guard case let .unavailable(emptyMessage) = emptySummary else {
        Issue.record("Expected unavailable for sampleCount=0; zeros are not watts")
        return
    }
    #expect(emptyMessage.contains("power: unavailable"))
    #expect(!emptyMessage.contains("0.00"))
}

@Test func test_chatPowerFooterUsesCapabilityMessageWhenTelemetryIsDown() {
    let sudo = chatPowerFooter(
        capability: PowerCapability(available: false, message: "powermetrics requires passwordless sudo or root"),
        summary: PowerSummary(packageW: 9, cpuW: 9, gpuW: 9, aneW: 9, sampleCount: 9),
        tokensPerSecond: 20
    )
    guard case let .unavailable(sudoMessage) = sudo else {
        Issue.record("Expected unavailable when capability is down")
        return
    }
    #expect(sudoMessage == "power: unavailable (sudo)")
    #expect(!sudoMessage.contains("9"))

    let host = chatPowerFooter(
        capability: PowerCapability(available: false, message: "powermetrics is unavailable on this host"),
        summary: nil,
        tokensPerSecond: 20
    )
    guard case let .unavailable(hostMessage) = host else {
        Issue.record("Expected capability.message when sudo is not the cause")
        return
    }
    #expect(hostMessage == "powermetrics is unavailable on this host")
}

@Test func test_chatFooterRendersMeasuredPowerAndJoulesPerToken() {
    let footer = ChatStatusFooter(
        tokensPerSecond: 13,
        ttftMs: 180,
        decodePath: "hybrid",
        contextUsed: 142,
        contextMax: 1024,
        power: .measured(packageW: 3.25, cpuW: 0.85, gpuW: 0.20, aneW: 1.60, joulesPerToken: 0.25)
    )
    let rendered = footer.render()
    #expect(rendered.contains("ANE 1.60W"))
    #expect(rendered.contains("CPU 0.85W"))
    #expect(rendered.contains("pkg 3.25W"))
    #expect(rendered.contains("0.250 J/tok"))
    #expect(!rendered.contains("power: unavailable"))
}

@Test func test_chatTUIRendererShowsConversationAndFooter() {
    let snapshot = ChatSnapshot(
        modelName: "Qwen2.5-1.5B-Instruct",
        turns: [
            ChatTurn(role: .user, content: "my name is Ada"),
            ChatTurn(role: .assistant, content: "Hello Ada."),
        ],
        streamingAssistant: "",
        status: .idle,
        footer: ChatStatusFooter(
            tokensPerSecond: 8.0,
            ttftMs: 90,
            decodePath: "hybrid",
            contextUsed: 40,
            contextMax: 1024
        )
    )
    let rendered = ChatTUIRenderer().render(snapshot: snapshot, size: TerminalSize(width: 80, height: 24))
    #expect(rendered.contains("Qwen2.5-1.5B-Instruct"))
    #expect(rendered.contains("my name is Ada"))
    #expect(rendered.contains("Hello Ada."))
    #expect(rendered.contains("path=hybrid"))
    #expect(rendered.contains("tok/s"))
    #expect(rendered.contains("TTFT"))
    #expect(rendered.contains("power: unavailable"))
    #expect(!rendered.contains("NOPASSWD"))
    #expect(!rendered.contains("/etc/sudoers"))
}

@Test func test_cliUsageDocumentsChatPowerFormulaAndSudo() {
    let usage = cliUsageText()
    #expect(usage.contains("J/tok = package_watts / tok_s"))
    #expect(usage.contains("compile"))
    #expect(usage.contains("--power"))
    #expect(usage.contains("/usr/bin/powermetrics"))
    #expect(usage.contains("passwordless sudo"))
    #expect(usage.contains("NOPASSWD"))
}

@Test func test_chatRefusesNonHybridDecodePath() {
    do {
        try assertChatDecodePathIsHybrid("exact_cpu")
        Issue.record("Expected exact_cpu to be rejected")
    } catch let error as CLIError {
        guard case let .runtime(message) = error else {
            Issue.record("Expected runtime error, got \(error)")
            return
        }
        #expect(message.contains("hybrid"))
        #expect(message.contains("exact_cpu"))
    } catch {
        Issue.record("Unexpected error: \(error)")
    }

    do {
        try assertChatDecodePathIsHybrid("hybrid")
    } catch {
        Issue.record("hybrid path should be accepted: \(error)")
    }

    do {
        try assertChatDecodePathIsHybrid("fused")
    } catch {
        Issue.record("fused path should be accepted: \(error)")
    }
}

@Test func test_optionsParseChatVsMLXFlags() throws {
    let options = try Options.parse([
        "espresso-generate",
        "chat",
        "--vs", "mlx",
        "--greedy",
        "--model", "/tmp/qwen.esp",
        "-n", "64",
    ])
    #expect(options.command == .chat)
    #expect(options.compareOpponent == .mlx)
    #expect(options.greedy)
    #expect(options.mlxQuant == nil)
    #expect(options.mlxModel == nil)
    #expect(options.maxTokens == 64)
}

@Test func test_optionsParseMLXQuantRequiresExplicitLabel() throws {
    let labeled = try Options.parse([
        "espresso-generate",
        "chat",
        "--vs", "mlx",
        "--mlx-quant", "4bit",
        "--greedy",
        "--model", "/tmp/qwen.esp",
    ])
    #expect(labeled.mlxQuant == "4bit")
    #expect(labeled.compareOpponent == .mlx)

    do {
        _ = try Options.parse([
            "espresso-generate",
            "chat",
            "--vs", "stories",
            "--greedy",
            "--model", "/tmp/qwen.esp",
        ])
        Issue.record("CoreML Stories must not be a chat --vs opponent")
    } catch let error as CLIError {
        guard case let .usage(message) = error else {
            Issue.record("Expected usage error, got \(error)")
            return
        }
        #expect(message.lowercased().contains("mlx"))
        #expect(message.lowercased().contains("compare"))
    }
}

@Test func test_compareCommandIsUnchangedByVsMLXFlags() throws {
    let options = try Options.parse([
        "espresso-generate",
        "compare",
        "--live",
        "--prompt", "Hello",
    ])
    #expect(options.command == .compare)
    #expect(options.preferLiveCompare)
    #expect(options.compareOpponent == nil)
    #expect(options.mlxQuant == nil)
}

@Test func test_chatVsMLXFairnessRequiresSameRepoGreedyAndNativePrecision() throws {
    let fairness = try makeChatVsMLXFairness(
        espressoModelName: "Qwen2.5-1.5B-Instruct",
        greedy: true,
        maxNewTokens: 32,
        mlxQuantFlag: nil,
        mlxModelOverride: nil
    )
    #expect(fairness.huggingfaceRepo == ChatVsMLXFairness.requiredHuggingFaceRepo)
    #expect(fairness.huggingfaceRepo == "Qwen/Qwen2.5-1.5B-Instruct")
    #expect(fairness.espressoModelName == "Qwen2.5-1.5B-Instruct")
    #expect(fairness.espressoPrecision == "fp16")
    #expect(fairness.mlxPrecisionLabel == "fp16")
    #expect(fairness.mlxQuantization == .native)
    #expect(fairness.greedy)
    #expect(fairness.maxNewTokens == 32)
    #expect(fairness.tokPerSecExcludesCompile)
    #expect(!fairness.mlxLaneHeader().lowercased().contains("4-bit"))
    #expect(!fairness.espressoLaneHeader().lowercased().contains("4-bit"))
}

@Test func test_chatVsMLXFairnessRejectsUnlabeledQuantAndNonGreedy() {
    do {
        _ = try makeChatVsMLXFairness(
            espressoModelName: "Qwen2.5-1.5B-Instruct",
            greedy: false,
            maxNewTokens: 32,
            mlxQuantFlag: nil,
            mlxModelOverride: nil
        )
        Issue.record("Expected --vs mlx without --greedy to fail")
    } catch let error as CLIError {
        guard case let .usage(message) = error else {
            Issue.record("Expected usage error, got \(error)")
            return
        }
        #expect(message.contains("--greedy"))
    } catch {
        Issue.record("Unexpected error: \(error)")
    }

    do {
        _ = try makeChatVsMLXFairness(
            espressoModelName: "gpt2_124m",
            greedy: true,
            maxNewTokens: 32,
            mlxQuantFlag: nil,
            mlxModelOverride: nil
        )
        Issue.record("Expected non-1.5B espresso model to fail")
    } catch let error as CLIError {
        guard case let .usage(message) = error else {
            Issue.record("Expected usage error, got \(error)")
            return
        }
        #expect(message.contains("Qwen/Qwen2.5-1.5B-Instruct"))
    } catch {
        Issue.record("Unexpected error: \(error)")
    }

    do {
        _ = try makeChatVsMLXFairness(
            espressoModelName: "Qwen2.5-1.5B-Instruct",
            greedy: true,
            maxNewTokens: 32,
            mlxQuantFlag: nil,
            mlxModelOverride: "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
        )
        Issue.record("Expected a different MLX repo without --mlx-quant to fail")
    } catch let error as CLIError {
        guard case let .usage(message) = error else {
            Issue.record("Expected usage error, got \(error)")
            return
        }
        #expect(message.contains("Qwen/Qwen2.5-1.5B-Instruct"))
        #expect(message.contains("--mlx-quant") || message.lowercased().contains("quant"))
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
}

@Test func test_chatVsMLXQuantMustBeExplicitAndLabeledOnBothLanes() throws {
    let fairness = try makeChatVsMLXFairness(
        espressoModelName: "Qwen2.5-1.5B-Instruct",
        greedy: true,
        maxNewTokens: 16,
        mlxQuantFlag: "4bit",
        mlxModelOverride: nil
    )
    #expect(fairness.mlxQuantization == .quantized(label: "4-bit"))
    #expect(fairness.mlxPrecisionLabel == "4-bit")
    #expect(fairness.espressoLaneHeader().contains("fp16"))
    #expect(fairness.espressoLaneHeader().contains("4-bit"))
    #expect(fairness.mlxLaneHeader().contains("4-bit"))
    #expect(fairness.title().contains("4-bit"))

    do {
        _ = try parseMLXQuantizationFlag("")
        Issue.record("Empty --mlx-quant must be rejected")
    } catch let error as CLIError {
        guard case let .usage(message) = error else {
            Issue.record("Expected usage error, got \(error)")
            return
        }
        #expect(message.contains("--mlx-quant"))
    }

    do {
        _ = try parseMLXQuantizationFlag("mystery")
        Issue.record("Unknown --mlx-quant must be rejected")
    } catch let error as CLIError {
        guard case let .usage(message) = error else {
            Issue.record("Expected usage error, got \(error)")
            return
        }
        #expect(message.contains("--mlx-quant"))
    }
}

@Test func test_chatVsMLXRejectsLoadedQuantWithoutFlag() {
    do {
        try assertMLXLoadMatchesFairness(
            quantized: true,
            precision: "4-bit",
            repo: ChatVsMLXFairness.requiredHuggingFaceRepo,
            fairness: try makeChatVsMLXFairness(
                espressoModelName: "Qwen2.5-1.5B-Instruct",
                greedy: true,
                maxNewTokens: 8,
                mlxQuantFlag: nil,
                mlxModelOverride: nil
            )
        )
        Issue.record("Unlabeled quantized MLX load must be impossible")
    } catch let error as CLIError {
        guard case let .runtime(message) = error else {
            Issue.record("Expected runtime error, got \(error)")
            return
        }
        #expect(message.contains("--mlx-quant"))
        #expect(message.lowercased().contains("4-bit") || message.lowercased().contains("quant"))
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
}

@Test func test_completionTokensPerSecondExcludesCompileWindow() {
    #expect(completionTokensPerSecond(generatedTokenCount: 13, completionMilliseconds: 1_000) == 13)
    #expect(completionTokensPerSecond(generatedTokenCount: 8, completionMilliseconds: 2_000) == 4)
    #expect(completionTokensPerSecond(generatedTokenCount: 0, completionMilliseconds: 1_000) == 0)
    #expect(completionTokensPerSecond(generatedTokenCount: 10, completionMilliseconds: 0) == 0)
    let compileMs = 8_400.0
    let completionMs = 500.0
    let tokPerSec = completionTokensPerSecond(generatedTokenCount: 10, completionMilliseconds: completionMs)
    #expect(tokPerSec == 20)
    #expect(tokPerSec != completionTokensPerSecond(generatedTokenCount: 10, completionMilliseconds: compileMs + completionMs))
}

@Test func test_chatVsMLXScoreboardNamesAWinnerPerMetricAndHidesMissingWatts() {
    let espresso = ChatVsMLXTurnMetrics(
        tokensPerSecond: 8,
        ttftMs: 180,
        compileMs: 1_200,
        packageW: 3.2,
        joulesPerToken: 0.4
    )
    let mlx = ChatVsMLXTurnMetrics(
        tokensPerSecond: 20,
        ttftMs: 40,
        compileMs: 800,
        packageW: 6.4,
        joulesPerToken: 0.32
    )
    let table = formatChatVsMLXScoreboard(espresso: espresso, mlx: mlx)
    #expect(table.contains("tok/s"))
    #expect(table.contains("TTFT"))
    #expect(table.contains("package W"))
    #expect(table.contains("J/tok"))
    #expect(table.contains("winner"))
    #expect(table.contains("mlx"))
    #expect(table.contains("espresso"))
    #expect(table.contains("8.0") || table.contains("8.00") || table.contains("8"))
    #expect(!table.contains("README"))

    let rows = chatVsMLXScoreboard(espresso: espresso, mlx: mlx)
    #expect(rows.first { $0.metric == "tok/s" }?.winner == "mlx")
    #expect(rows.first { $0.metric == "TTFT ms" }?.winner == "mlx")
    #expect(rows.first { $0.metric == "package W" }?.winner == "espresso")
    #expect(rows.first { $0.metric == "J/tok" }?.winner == "mlx")

    let noPower = ChatVsMLXTurnMetrics(
        tokensPerSecond: 8,
        ttftMs: 180,
        compileMs: 0,
        packageW: nil,
        joulesPerToken: nil
    )
    let unavailable = chatVsMLXScoreboard(espresso: noPower, mlx: mlx)
    #expect(unavailable.first { $0.metric == "package W" }?.espresso == "unavailable")
    #expect(unavailable.first { $0.metric == "J/tok" }?.espresso == "unavailable")
    #expect(unavailable.first { $0.metric == "package W" }?.winner == "—")
    #expect(unavailable.first { $0.metric == "J/tok" }?.winner == "—")
}

@Test func test_mlxMissingInstallCommandDoesNotSuggestQuantization() {
    let message = mlxInstallInstructions()
    #expect(message.contains("pip install mlx-lm"))
    #expect(message.contains("fp16") || message.contains("bf16") || message.contains("native"))
    #expect(!message.lowercased().contains("4-bit") || message.lowercased().contains("do not"))
}

@Test func test_resolveMLXPythonPicksFirstCandidateThatImports() {
    #expect(
        resolveMLXPython(candidates: ["/missing", "/good"], canImport: { $0 == "/good" }) == "/good"
    )
    #expect(resolveMLXPython(candidates: ["/missing"], canImport: { _ in false }) == nil)
    #expect(mlxPythonCandidates(environment: ["ESPRESSO_MLX_PYTHON": "/only"]) == ["/only"])
}

@Test func test_huggingFaceSnapshotRequiresConfigAndWeights() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    let snapshots = root
        .appendingPathComponent("models--Qwen--Qwen2.5-1.5B-Instruct", isDirectory: true)
        .appendingPathComponent("snapshots", isDirectory: true)
        .appendingPathComponent("abc123", isDirectory: true)
    try FileManager.default.createDirectory(at: snapshots, withIntermediateDirectories: true)
    #expect(
        huggingFaceHubSnapshot(repo: "Qwen/Qwen2.5-1.5B-Instruct", cacheRoot: root) == nil
    )
    try "{}".write(to: snapshots.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
    #expect(
        huggingFaceHubSnapshot(repo: "Qwen/Qwen2.5-1.5B-Instruct", cacheRoot: root) == nil
    )
    try Data().write(to: snapshots.appendingPathComponent("model.safetensors"))
    let found = huggingFaceHubSnapshot(repo: "Qwen/Qwen2.5-1.5B-Instruct", cacheRoot: root)
    #expect(found?.standardizedFileURL.path == snapshots.standardizedFileURL.path)
}

@Test func test_parseMLXStreamEventsCoverCompileTokenAndCompletion() throws {
    let hello = try parseMLXStreamEvent(
        #"{"type":"hello","precision":"float16","quantized":false,"repo":"Qwen/Qwen2.5-1.5B-Instruct"}"#
    )
    #expect(hello == .hello(precision: "float16", quantized: false, repo: "Qwen/Qwen2.5-1.5B-Instruct"))

    let compile = try parseMLXStreamEvent(#"{"type":"compile","compile_time_ms":812.5}"#)
    #expect(compile == .compile(ms: 812.5))

    let token = try parseMLXStreamEvent(
        #"{"type":"token","text":"Hello","token_index":1,"elapsed_ms":40,"token_latency_ms":40,"tokens_per_second":25}"#
    )
    #expect(
        token == .token(text: "Hello", tokenIndex: 1, elapsedMs: 40, tokenLatencyMs: 40, tokensPerSecond: 25)
    )

    let done = try parseMLXStreamEvent(
        #"{"type":"completed","text":"Hello","compile_time_ms":812.5,"first_token_latency_ms":40,"tokens_per_second":20,"generation_tokens":8}"#
    )
    #expect(
        done == .completed(
            text: "Hello",
            compileMs: 812.5,
            ttftMs: 40,
            tokensPerSecond: 20,
            tokenCount: 8
        )
    )
}

@Test func test_liveCompareRendererMLXModeShowsPerLaneEnergyAndNotCoreMLStories() throws {
    let fairness = try makeChatVsMLXFairness(
        espressoModelName: "Qwen2.5-1.5B-Instruct",
        greedy: true,
        maxNewTokens: 32,
        mlxQuantFlag: nil,
        mlxModelOverride: nil
    )
    var espresso = LiveLaneSnapshot(title: fairness.espressoLaneHeader(), maxTokens: 32)
    espresso.status = .generating
    espresso.generatedTokenCount = 4
    espresso.text = "Start with async/await"
    espresso.tokensPerSecond = 8.1
    espresso.ttftMs = 120
    espresso.compileMs = 900
    espresso.power = PowerSummary(packageW: 3.2, cpuW: 0.8, gpuW: 0.1, aneW: 1.7, sampleCount: 3)
    espresso.wattFocus = .ane

    var mlx = LiveLaneSnapshot(title: fairness.mlxLaneHeader(), maxTokens: 32)
    mlx.status = .generating
    mlx.generatedTokenCount = 4
    mlx.text = "Same completion streamed"
    mlx.tokensPerSecond = 19.4
    mlx.ttftMs = 45
    mlx.compileMs = 700
    mlx.power = PowerSummary(packageW: 6.4, cpuW: 1.1, gpuW: 4.2, aneW: 0.0, sampleCount: 3)
    mlx.wattFocus = .gpu

    let snapshot = LiveCompareSnapshot(
        modelName: "Qwen2.5-1.5B-Instruct",
        prompt: "what is a good way to learn Swift concurrency?",
        maxTokens: 32,
        elapsedMs: 800,
        espresso: espresso,
        coreML: mlx,
        livePower: nil,
        matchCount: 0,
        totalComparedTokens: 0,
        events: ["[Espresso] token 1", "[MLX] token 1"],
        display: .mlx(fairness)
    )
    let rendered = LiveCompareRenderer().render(snapshot: snapshot, size: TerminalSize(width: 140, height: 40))
    #expect(rendered.contains("ESPRESSO"))
    #expect(rendered.contains("MLX"))
    #expect(rendered.contains("fp16"))
    #expect(rendered.contains("TOKENS / SEC"))
    #expect(rendered.contains("compile"))
    #expect(rendered.contains("J/tok"))
    #expect(rendered.contains("Start with async/await"))
    #expect(rendered.contains("Same completion streamed"))
    #expect(!rendered.contains("CORE ML"))
    #expect(!rendered.contains("LIVE GPT-2"))
    #expect(!rendered.contains("Stories"))
    #expect(rendered.contains("Ctrl-C cancels the current lane"))
    #expect(!rendered.contains("Ctrl-C to quit"))
}

@Test func test_liveCompareRendererCoreMLStoriesPathStillSaysCoreML() {
    var espresso = LiveLaneSnapshot(title: "Espresso / ANE", maxTokens: 32)
    espresso.text = "Hello"
    var coreML = LiveLaneSnapshot(title: "Core ML", maxTokens: 32)
    coreML.text = "Hello"
    let snapshot = LiveCompareSnapshot(
        modelName: "gpt2_124m",
        prompt: "Hello",
        maxTokens: 32,
        elapsedMs: 10,
        espresso: espresso,
        coreML: coreML,
        livePower: nil,
        matchCount: 1,
        totalComparedTokens: 1,
        events: []
    )
    let rendered = LiveCompareRenderer().render(snapshot: snapshot, size: TerminalSize(width: 140, height: 40))
    #expect(rendered.contains("ESPRESSO vs CORE ML LIVE GPT-2"))
    #expect(rendered.contains("CORE ML"))
    #expect(rendered.contains("Espresso preflight avg"))
    #expect(!rendered.contains("MLX"))
    #expect(rendered.contains("Ctrl-C to quit"))
    #expect(!rendered.contains("Ctrl-C cancels the current lane"))
}

@Test func test_cliUsageDocumentsChatVsMLXFairness() {
    let usage = cliUsageText()
    #expect(usage.contains("--vs mlx"))
    #expect(usage.contains("--mlx-quant"))
    #expect(usage.contains("Qwen/Qwen2.5-1.5B-Instruct"))
    #expect(usage.contains("fp16"))
    #expect(usage.contains("compile"))
    #expect(usage.contains("J/tok"))
}

@Test func test_chatVsMLXAppliesLoadedNativePrecisionToBothLaneLabels() throws {
    let fairness = try makeChatVsMLXFairness(
        espressoModelName: "Qwen2.5-1.5B-Instruct",
        greedy: true,
        maxNewTokens: 8,
        mlxQuantFlag: nil,
        mlxModelOverride: nil
    )
    #expect(mlxNativePrecisionLabel("float16") == "fp16")
    #expect(mlxNativePrecisionLabel("bfloat16") == "bf16")
    #expect(mlxNativePrecisionLabel("bf16") == "bf16")

    let bf16 = fairness.applyingLoadedPrecision("bfloat16")
    #expect(bf16.mlxPrecisionLabel == "bf16")
    #expect(bf16.mlxLaneHeader().contains("bf16"))
    #expect(bf16.title().contains("bf16"))
    #expect(bf16.espressoLaneHeader().contains("fp16"))

    let labeled = try makeChatVsMLXFairness(
        espressoModelName: "Qwen2.5-1.5B-Instruct",
        greedy: true,
        maxNewTokens: 8,
        mlxQuantFlag: "4bit",
        mlxModelOverride: nil
    ).applyingLoadedPrecision("bfloat16")
    #expect(labeled.mlxPrecisionLabel == "4-bit")
}

@Test func test_pairedChatVsMLXMetricsDropsUnmatchedTurns() {
    let espressoOnly = ChatVsMLXTurnMetrics(
        tokensPerSecond: 10,
        ttftMs: 100,
        compileMs: 0,
        packageW: 4,
        joulesPerToken: 0.4
    )
    let espressoPaired = ChatVsMLXTurnMetrics(
        tokensPerSecond: 8,
        ttftMs: 180,
        compileMs: 0,
        packageW: 3.2,
        joulesPerToken: 0.4
    )
    let mlxPaired = ChatVsMLXTurnMetrics(
        tokensPerSecond: 20,
        ttftMs: 40,
        compileMs: 0,
        packageW: 6.4,
        joulesPerToken: 0.32
    )
    let paired = pairedChatVsMLXMetrics(
        espresso: [espressoPaired, espressoOnly],
        mlx: [mlxPaired]
    )
    #expect(paired.espresso.tokensPerSecond == 8)
    #expect(paired.mlx.tokensPerSecond == 20)
    #expect(paired.espresso.tokensPerSecond != 9)
}

@Test func test_laneJoulesPerTokenUsesPackageWattsOverTokPerSec() {
    var lane = LiveLaneSnapshot(title: "Espresso", maxTokens: 8)
    lane.tokensPerSecond = 8
    lane.power = PowerSummary(packageW: 4, cpuW: 1, gpuW: 0, aneW: 2, sampleCount: 2)
    #expect(laneJoulesPerToken(lane) == 0.5)

    var unavailable = LiveLaneSnapshot(title: "MLX", maxTokens: 8)
    unavailable.tokensPerSecond = 20
    unavailable.power = .unavailable
    #expect(laneJoulesPerToken(unavailable) == nil)
}

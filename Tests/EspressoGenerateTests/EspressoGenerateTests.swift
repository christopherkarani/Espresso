import Foundation
import Testing
import ANETypes
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
        architecture: "llama"
    )

    let config = try metadata.asConfig()
    #expect(config.ropeTheta == 1_000_000)
    #expect(config.eosToken == 151645)
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

@Test func test_resolveCoreMLReferenceModelContractDetectsStatelessInput() throws {
    let contract = try resolveCoreMLReferenceModelContract(
        inputSpecs: [
            CoreMLReferenceArraySpec(name: "input_ids", shape: [1, 64], dataType: .int32),
        ],
        outputFeatureNames: ["hidden_states"],
        stateSpecs: [],
        requestedSequenceLength: 32
    )

    #expect(
        contract == .stateless(
            StatelessCoreMLReferenceContract(
                tokenInput: CoreMLReferenceArraySpec(name: "input_ids", shape: [1, 64], dataType: .int32),
                outputFeatureName: "hidden_states",
                maxSequenceTokens: 64
            )
        )
    )
}

@Test func test_resolveCoreMLReferenceModelContractDetectsStatefulInputsAndCapacity() throws {
    let contract = try resolveCoreMLReferenceModelContract(
        inputSpecs: [
            CoreMLReferenceArraySpec(name: "cache_position", shape: [1], dataType: .int32),
            CoreMLReferenceArraySpec(name: "input_ids", shape: [1, 1], dataType: .int32),
        ],
        outputFeatureNames: ["hidden_states"],
        stateSpecs: [
            CoreMLReferenceArraySpec(name: "key_cache_0", shape: [1, 4, 128, 16], dataType: .float16),
            CoreMLReferenceArraySpec(name: "value_cache_0", shape: [1, 4, 128, 16], dataType: .float16),
        ],
        requestedSequenceLength: 96
    )

    #expect(
        contract == .stateful(
            StatefulCoreMLReferenceContract(
                tokenInput: CoreMLReferenceArraySpec(name: "input_ids", shape: [1, 1], dataType: .int32),
                cachePositionInput: CoreMLReferenceArraySpec(name: "cache_position", shape: [1], dataType: .int32),
                outputFeatureName: "hidden_states",
                maxSequenceTokens: 128
            )
        )
    )
}

@Test func test_resolveCoreMLReferenceModelContractRejectsMismatchedStateCapacities() throws {
    do {
        _ = try resolveCoreMLReferenceModelContract(
            inputSpecs: [
                CoreMLReferenceArraySpec(name: "cache_position", shape: [1], dataType: .int32),
                CoreMLReferenceArraySpec(name: "input_ids", shape: [1, 1], dataType: .int32),
            ],
            outputFeatureNames: ["hidden_states"],
            stateSpecs: [
                CoreMLReferenceArraySpec(name: "key_cache_0", shape: [1, 4, 128, 16], dataType: .float16),
                CoreMLReferenceArraySpec(name: "value_cache_0", shape: [1, 4, 64, 16], dataType: .float16),
            ],
            requestedSequenceLength: 32
        )
        Issue.record("Expected mismatched state capacities to be rejected.")
    } catch let error as CLIError {
        #expect(error.localizedDescription.contains("KV cache capacity"))
    }
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

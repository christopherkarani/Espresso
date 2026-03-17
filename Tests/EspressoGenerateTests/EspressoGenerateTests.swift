import Foundation
import Testing
import ANETypes
@testable import EspressoGenerate

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
            generatedTokens: [UInt32(callCount)],
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

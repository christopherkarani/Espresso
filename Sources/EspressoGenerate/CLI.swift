import Foundation
import Darwin
import ANERuntime
import ANETypes
import ESPBenchSupport
import ESPBundle
import ESPRuntime
import ModelSupport
import RealModelInference

enum CommandName: String {
    case demo
    case generate
    case compare
    case bench
    case suite
    case doctor
    case chat
}

enum PowerMode: Equatable {
    case auto
    case on
    case off
}

struct Options {
    var command: CommandName?
    var modelName: String? = ProcessInfo.processInfo.environment["ESPRESSO_MODEL"]
    var bundlePath: String? = ProcessInfo.processInfo.environment["ESPRESSO_BUNDLE_PATH"]
    var weightsDir: String? = ProcessInfo.processInfo.environment["ESPRESSO_WEIGHTS_DIR"]
    var tokenizerDir: String? = ProcessInfo.processInfo.environment["ESPRESSO_TOKENIZER_DIR"]
    var coreMLModelPath: String? = ProcessInfo.processInfo.environment["ESPRESSO_COREML_MODEL"]
    var prompt: String?
    var positionalPrompt: [String] = []
    var maxTokens: Int = Options.parseNonNegativeIntEnv("ESPRESSO_MAX_TOKENS") ?? 128
    var temperature: Float = Options.parseNonNegativeFloatEnv("ESPRESSO_TEMPERATURE") ?? 0
    var coreMLSequenceLength: Int? = Options.parseNonNegativeIntEnv("ESPRESSO_COREML_SEQ_LEN")
    var compareWarmup: Int = Options.parseNonNegativeIntEnv("ESPRESSO_COMPARE_WARMUP") ?? 1
    var compareIterations: Int = Options.parsePositiveIntEnv("ESPRESSO_COMPARE_ITERATIONS") ?? 3
    var suiteRuns: Int = Options.parsePositiveIntEnv("ESPRESSO_SUITE_RUNS") ?? 3
    var seed: Int = Options.parseIntEnv("ESPRESSO_COMPARE_SEED") ?? 1234
    var coreMLComputeUnits: String = ProcessInfo.processInfo.environment["ESPRESSO_COREML_COMPUTE_UNITS"] ?? "cpu_and_neural_engine"
    var showStats = true
    var listModels = false
    var prepareDemo = false
    var promptsFile: String?
    var resultsTSV: String?
    var allowBootstrap = true
    var includeColdRun = true
    var benchmarkGenerate = false
    var showHelp = false
    var forceTUI = false
    var disableTUI = false
    var preferLiveCompare = false
    var preferBenchCompare = false
    var powerMode: PowerMode = .auto
    var outputDir: String?
    var jsonOutput = false
    var probeGPT2AttentionCompile = false
    var milDeploymentTarget: String?
    var gpt2NormMode: String?
    var rawPrompt = false
    var disableHybridFallback = false
    var plain = false
    var greedy = false
    var systemPrompt: String?
    var topP: Float = Options.parseNonNegativeFloatEnv("ESPRESSO_TOP_P") ?? 1
    var topPWasSet = Options.parseNonNegativeFloatEnv("ESPRESSO_TOP_P") != nil
    var temperatureWasSet = Options.parseNonNegativeFloatEnv("ESPRESSO_TEMPERATURE") != nil
    var compareOpponent: CompareOpponent?
    var mlxQuant: String?
    var mlxModel: String?

    static func parse(_ argv: [String]) throws -> Options {
        var options = Options()
        var index = 1

        if index < argv.count, let command = CommandName(rawValue: argv[index]) {
            options.command = command
            index += 1
        }

        while index < argv.count {
            let argument = argv[index]
            if argument == "--" {
                options.positionalPrompt.append(contentsOf: argv[(index + 1)...])
                break
            }

            let (flag, inlineValue) = splitInlineValue(argument)
            switch flag {
            case "-m", "--model":
                options.modelName = try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index)
            case "-b", "--bundle":
                options.bundlePath = try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index)
            case "-w", "--weights":
                options.weightsDir = try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index)
            case "-t", "--tokenizer":
                options.tokenizerDir = try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index)
            case "--coreml-model":
                options.coreMLModelPath = try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index)
            case "-p", "--prompt":
                options.prompt = try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index)
            case "-n", "--max-tokens":
                options.maxTokens = try parseNonNegativeInt(
                    try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index),
                    flag: flag
                )
            case "--temperature":
                options.temperature = try parseNonNegativeFloat(
                    try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index),
                    flag: flag
                )
                options.temperatureWasSet = true
            case "--top-p":
                options.topP = try parseUnitInterval(
                    try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index),
                    flag: flag
                )
                options.topPWasSet = true
            case "--system":
                options.systemPrompt = try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index)
            case "--plain":
                options.plain = true
                options.disableTUI = true
                options.forceTUI = false
            case "--greedy":
                options.greedy = true
            case "--vs":
                options.compareOpponent = try parseCompareOpponent(
                    try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index)
                )
            case "--mlx-quant":
                options.mlxQuant = try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index)
            case "--mlx-model":
                options.mlxModel = try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index)
            case "--coreml-seq-len":
                options.coreMLSequenceLength = try parseNonNegativeInt(
                    try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index),
                    flag: flag
                )
            case "--compare-warmup":
                options.compareWarmup = try parseNonNegativeInt(
                    try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index),
                    flag: flag
                )
            case "--compare-iterations":
                options.compareIterations = try parsePositiveInt(
                    try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index),
                    flag: flag
                )
            case "--runs":
                options.suiteRuns = try parsePositiveInt(
                    try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index),
                    flag: flag
                )
            case "--coreml-compute-units":
                options.coreMLComputeUnits = try parseCoreMLComputeUnits(
                    try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index)
                )
            case "--seed":
                options.seed = try parseInt(
                    try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index),
                    flag: flag
                )
            case "--output-dir":
                options.outputDir = try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index)
            case "--prompts":
                options.promptsFile = try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index)
            case "--results-tsv":
                options.resultsTSV = try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index)
            case "--prepare-demo":
                options.prepareDemo = true
            case "--no-bootstrap":
                options.allowBootstrap = false
            case "--no-cold":
                options.includeColdRun = false
            case "--no-stats":
                options.showStats = false
            case "--benchmark-generate":
                options.benchmarkGenerate = true
            case "--list-models":
                options.listModels = true
            case "--tui":
                options.forceTUI = true
                options.disableTUI = false
            case "--no-tui":
                options.disableTUI = true
                options.forceTUI = false
            case "--live":
                options.preferLiveCompare = true
                options.preferBenchCompare = false
            case "--bench":
                options.preferBenchCompare = true
                options.preferLiveCompare = false
            case "--power":
                options.powerMode = .on
            case "--no-power":
                options.powerMode = .off
            case "--json":
                options.jsonOutput = true
            case "--probe-gpt2-attention-compile":
                options.probeGPT2AttentionCompile = true
            case "--mil-deployment-target":
                options.milDeploymentTarget = try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index)
            case "--gpt2-norm":
                options.gpt2NormMode = try parseGPT2NormMode(
                    try value(for: flag, inlineValue: inlineValue, argv: argv, index: &index)
                )
            case "--raw-prompt":
                options.rawPrompt = true
            case "--no-hybrid-fallback":
                options.disableHybridFallback = true
            case "-h", "--help":
                options.showHelp = true
            default:
                if argument.hasPrefix("-") {
                    throw CLIError.usage("Unknown argument: \(argument)")
                }
                options.positionalPrompt.append(argument)
            }
            index += 1
        }

        return options
    }

    private static func splitInlineValue(_ argument: String) -> (String, String?) {
        guard argument.hasPrefix("--"), let separator = argument.firstIndex(of: "=") else {
            return (argument, nil)
        }
        return (String(argument[..<separator]), String(argument[argument.index(after: separator)...]))
    }

    private static func value(
        for flag: String,
        inlineValue: String?,
        argv: [String],
        index: inout Int
    ) throws -> String {
        if let inlineValue {
            guard !inlineValue.isEmpty else {
                throw CLIError.usage("Expected a value for \(flag)")
            }
            return inlineValue
        }
        index += 1
        guard index < argv.count else {
            throw CLIError.usage("Expected a value for \(flag)")
        }
        return argv[index]
    }

    private static func parseNonNegativeInt(_ raw: String, flag: String) throws -> Int {
        guard let value = Int(raw), value >= 0 else {
            throw CLIError.usage("Expected a non-negative integer for \(flag)")
        }
        return value
    }

    private static func parseNonNegativeFloat(_ raw: String, flag: String) throws -> Float {
        guard let value = Float(raw), value.isFinite, value >= 0 else {
            throw CLIError.usage("Expected a finite non-negative number for \(flag)")
        }
        return value
    }

    private static func parseUnitInterval(_ raw: String, flag: String) throws -> Float {
        guard let value = Float(raw), value.isFinite, value > 0, value <= 1 else {
            throw CLIError.usage("Expected a number in (0, 1] for \(flag)")
        }
        return value
    }

    private static func parsePositiveInt(_ raw: String, flag: String) throws -> Int {
        guard let value = Int(raw), value > 0 else {
            throw CLIError.usage("Expected a positive integer for \(flag)")
        }
        return value
    }

    private static func parseInt(_ raw: String, flag: String) throws -> Int {
        guard let value = Int(raw) else {
            throw CLIError.usage("Expected an integer for \(flag)")
        }
        return value
    }

    static func parseNonNegativeIntEnv(_ key: String) -> Int? {
        guard let value = ProcessInfo.processInfo.environment[key], !value.isEmpty else {
            return nil
        }
        return Int(value).flatMap { $0 >= 0 ? $0 : nil }
    }

    static func parseNonNegativeFloatEnv(_ key: String) -> Float? {
        guard let value = ProcessInfo.processInfo.environment[key], !value.isEmpty else {
            return nil
        }
        return Float(value).flatMap { $0.isFinite && $0 >= 0 ? $0 : nil }
    }

    static func parsePositiveIntEnv(_ key: String) -> Int? {
        guard let value = ProcessInfo.processInfo.environment[key], !value.isEmpty else {
            return nil
        }
        return Int(value).flatMap { $0 > 0 ? $0 : nil }
    }

    static func parseIntEnv(_ key: String) -> Int? {
        guard let value = ProcessInfo.processInfo.environment[key], !value.isEmpty else {
            return nil
        }
        return Int(value)
    }

    private static func parseCoreMLComputeUnits(_ raw: String) throws -> String {
        switch raw {
        case "all", "cpu_and_neural_engine", "cpu_and_gpu", "cpu_only":
            return raw
        default:
            throw CLIError.usage("Expected --coreml-compute-units all|cpu_and_neural_engine|cpu_and_gpu|cpu_only")
        }
    }

    private static func parseGPT2NormMode(_ raw: String) throws -> String {
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "layernorm", "rmsnorm":
            return raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        default:
            throw CLIError.usage("Expected --gpt2-norm layernorm|rmsnorm")
        }
    }
}

enum CLIError: Error, LocalizedError {
    case usage(String)
    case runtime(String)

    var errorDescription: String? {
        switch self {
        case let .usage(message), let .runtime(message):
            return message
        }
    }
}

struct MetadataConfigFile: Decodable {
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
            throw CLIError.runtime("Unsupported architecture in metadata.json: \(architecture)")
        }
        let parsedDecodePath = try preferredDecodePath.map {
            try MultiModelConfig.PreferredDecodePath.parse($0)
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
            eosToken: eosToken.flatMap(TokenID.init),
            architecture: parsedArchitecture,
            preferredDecodePath: parsedDecodePath
        )
    }
}

struct ResolvedInvocation {
    let config: MultiModelConfig
    let bundlePath: String?
    let weightsDir: String
    let tokenizerDir: String
    let prompt: String
    let maxTokens: Int
    let temperature: Float
    let topP: Float
    let showStats: Bool
    let coreMLModelPath: String?
    let coreMLSequenceLength: Int?
    let compareWarmup: Int
    let compareIterations: Int
    let benchmarkGenerate: Bool
    let coreMLComputeUnits: String
    let allowBootstrap: Bool
    let seed: Int
    let outputDir: String?
    let rawPrompt: Bool

    init(
        config: MultiModelConfig,
        bundlePath: String?,
        weightsDir: String,
        tokenizerDir: String,
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float = 1,
        showStats: Bool,
        coreMLModelPath: String?,
        coreMLSequenceLength: Int?,
        compareWarmup: Int,
        compareIterations: Int,
        benchmarkGenerate: Bool = false,
        coreMLComputeUnits: String,
        allowBootstrap: Bool,
        seed: Int,
        outputDir: String?,
        rawPrompt: Bool = false
    ) {
        self.config = config
        self.bundlePath = bundlePath
        self.weightsDir = weightsDir
        self.tokenizerDir = tokenizerDir
        self.prompt = prompt
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.showStats = showStats
        self.coreMLModelPath = coreMLModelPath
        self.coreMLSequenceLength = coreMLSequenceLength
        self.compareWarmup = compareWarmup
        self.compareIterations = compareIterations
        self.benchmarkGenerate = benchmarkGenerate
        self.coreMLComputeUnits = coreMLComputeUnits
        self.allowBootstrap = allowBootstrap
        self.seed = seed
        self.outputDir = outputDir
        self.rawPrompt = rawPrompt
    }
}

extension ResolvedInvocation {
    func with(
        prompt: String? = nil,
        compareWarmup: Int? = nil,
        compareIterations: Int? = nil,
        outputDir: String? = nil
    ) -> ResolvedInvocation {
        ResolvedInvocation(
            config: config,
            bundlePath: bundlePath,
            weightsDir: weightsDir,
            tokenizerDir: tokenizerDir,
            prompt: prompt ?? self.prompt,
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            showStats: showStats,
            coreMLModelPath: coreMLModelPath,
            coreMLSequenceLength: coreMLSequenceLength,
            compareWarmup: compareWarmup ?? self.compareWarmup,
            compareIterations: compareIterations ?? self.compareIterations,
            benchmarkGenerate: benchmarkGenerate,
            coreMLComputeUnits: coreMLComputeUnits,
            allowBootstrap: allowBootstrap,
            seed: seed,
            outputDir: outputDir ?? self.outputDir,
            rawPrompt: rawPrompt
        )
    }
}

struct BackendRunMetrics: Sendable {
    let backend: String
    let text: String
    let generatedTokens: [TokenID]
    let promptTokens: [TokenID]
    let compileTimeMs: Double
    let firstTokenLatencyMs: Double
    let tokensPerSecond: Double
    let medianTokenMs: Double
    let p95TokenMs: Double
    let totalTimeMs: Double
    let tokenLatenciesMs: [Double]
    let compileRetryCount: Int
    let compileFailureCount: Int
    let compileBreakdown: [String: ANECompileLabelStatsSnapshot]?
    let exactHeadBackend: String?
    let cachedBindingsEnabled: Bool?
    let committedExactTokensPerPass: Double?
    let acceptedFutureTokensPerPass: Double?
    let decodePath: String?
    let hopsPerToken: Int?
    let decodeProfileReport: String?

    init(
        backend: String,
        text: String,
        generatedTokens: [TokenID],
        promptTokens: [TokenID],
        compileTimeMs: Double,
        firstTokenLatencyMs: Double,
        tokensPerSecond: Double,
        medianTokenMs: Double,
        p95TokenMs: Double,
        totalTimeMs: Double,
        tokenLatenciesMs: [Double],
        compileRetryCount: Int = 0,
        compileFailureCount: Int = 0,
        compileBreakdown: [String: ANECompileLabelStatsSnapshot]? = nil,
        exactHeadBackend: String? = nil,
        cachedBindingsEnabled: Bool? = nil,
        committedExactTokensPerPass: Double? = nil,
        acceptedFutureTokensPerPass: Double? = nil,
        decodePath: String? = nil,
        hopsPerToken: Int? = nil,
        decodeProfileReport: String? = nil
    ) {
        self.backend = backend
        self.text = text
        self.generatedTokens = generatedTokens
        self.promptTokens = promptTokens
        self.compileTimeMs = compileTimeMs
        self.firstTokenLatencyMs = firstTokenLatencyMs
        self.tokensPerSecond = tokensPerSecond
        self.medianTokenMs = medianTokenMs
        self.p95TokenMs = p95TokenMs
        self.totalTimeMs = totalTimeMs
        self.tokenLatenciesMs = tokenLatenciesMs
        self.compileRetryCount = compileRetryCount
        self.compileFailureCount = compileFailureCount
        self.compileBreakdown = compileBreakdown
        self.exactHeadBackend = exactHeadBackend
        self.cachedBindingsEnabled = cachedBindingsEnabled
        self.committedExactTokensPerPass = committedExactTokensPerPass
        self.acceptedFutureTokensPerPass = acceptedFutureTokensPerPass
        self.decodePath = decodePath
        self.hopsPerToken = hopsPerToken
        self.decodeProfileReport = decodeProfileReport
    }
}

func generateDecodeProfileMeanLine(_ report: String?) -> String? {
    guard let report, !report.isEmpty else { return nil }
    return report
        .split(whereSeparator: \.isNewline)
        .map(String.init)
        .first { $0.hasPrefix("decode_profile_mean_ms/token") }
}

func espressoCompareLaneContractLines(
    decodePath: String?,
    hopsPerToken: Int?,
    decodeProfileReport: String?
) -> [String] {
    var lines: [String] = []
    if let decodePath, !decodePath.isEmpty {
        lines.append("decode_path=\(decodePath)")
    }
    if let hopsPerToken {
        lines.append("hops/token=\(hopsPerToken)")
    }
    if let profile = generateDecodeProfileMeanLine(decodeProfileReport) {
        lines.append(profile)
    }
    return lines
}

struct CompareReport: Sendable {
    let model: String
    let prompt: String
    let maxTokens: Int
    let seed: Int
    let espresso: BackendRunMetrics
    let coreML: BackendRunMetrics
    let tokenMatch: Bool
    let textMatch: Bool
    let coreMLComputeUnits: String
    let coreMLSequenceLength: Int
    let espressoPower: PowerSummary?
    let coreMLPower: PowerSummary?
    let outputDirectory: String?
}

private struct CLITokenizer: @unchecked Sendable {
    let encodeImpl: (String) throws -> [Int]
    let decodeImpl: ([Int]) -> String

    func encode(_ text: String) throws -> [Int] {
        try encodeImpl(text)
    }

    func decode(_ tokens: [Int]) -> String {
        decodeImpl(tokens)
    }
}

private struct DoctorCheck: Sendable {
    enum Severity: String, Sendable {
        case ok
        case warning
        case error
    }

    let name: String
    let severity: Severity
    let detail: String
}

private struct GPT2AttentionCompileProbePayload: Sendable {
    let deploymentTarget: String
    let normKind: String
    let results: [RealModelInferenceEngine.GPT2AttentionCompileProbeResult]
}

private final class LockedValueBox<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: Value?

    func set(_ value: Value) {
        lock.lock()
        storage = value
        lock.unlock()
    }

    func get() -> Value? {
        lock.lock()
        let value = storage
        lock.unlock()
        return value
    }
}

func applyCLIExperimentEnvironment(_ options: Options) {
    if options.disableHybridFallback || chatForcesHybridFallbackDisable(options) {
        setenv("ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK", "1", 1)
    }
    if options.rawPrompt {
        setenv("ESPRESSO_RAW_PROMPT", "1", 1)
    }

    if let deploymentTarget = options.milDeploymentTarget?.trimmingCharacters(in: .whitespacesAndNewlines),
       !deploymentTarget.isEmpty {
        setenv("ESPRESSO_MIL_DEPLOYMENT_TARGET", deploymentTarget, 1)
    }

    guard let gpt2NormMode = options.gpt2NormMode else {
        return
    }
    setenv("ESPRESSO_GPT2_NORM", gpt2NormMode, 1)
    switch gpt2NormMode {
    case "rmsnorm":
        setenv("ESPRESSO_GPT2_USE_RMS_NORM", "1", 1)
    default:
        setenv("ESPRESSO_GPT2_USE_RMS_NORM", "0", 1)
    }
}

private let posixLocale = Locale(identifier: "en_US_POSIX")

private func formatCompileBreakdown(_ breakdown: [String: ANECompileLabelStatsSnapshot]) -> String {
    breakdown.keys.sorted().compactMap { key in
        guard let stats = breakdown[key] else {
            return nil
        }
        return "\(key):a\(stats.attemptCount)/s\(stats.successCount)/f\(stats.failureCount)/r\(stats.retryCount)"
    }.joined(separator: ",")
}
private let defaultDemoPrompt = "Hello"
private let explicitModelAliases: [String: String] = [
    "gpt2": "gpt2_124m",
    "stories": "stories110m",
    "smollm": "smolLM_135m",
    "smollm135m": "smolLM_135m",
    "tinyllama": "tinyLlama_1_1b",
    "tinyllama11b": "tinyLlama_1_1b",
]

private let normalizedModelNames: [String: String] = {
    var map: [String: String] = [:]
    for name in ModelRegistry.all.keys {
        map[normalizeModelKey(name)] = name
    }
    for (alias, name) in explicitModelAliases {
        map[normalizeModelKey(alias)] = name
    }
    return map
}()

func normalizeModelKey(_ raw: String) -> String {
    raw.lowercased().filter { $0.isLetter || $0.isNumber }
}

func canonicalModelName(for raw: String) -> String? {
    normalizedModelNames[normalizeModelKey(raw)]
}

func architectureLabel(_ architecture: MultiModelConfig.Architecture) -> String {
    switch architecture {
    case .gpt2:
        return "gpt2"
    case .llama:
        return "llama"
    }
}

func stderrLine(_ message: String) {
    message.withCString { cString in
        _ = fputs(cString, stderr)
        _ = fputc(0x0A, stderr)
    }
}

func cliUsageText() -> String {
    """
    Usage:
      espresso-generate demo [options] [prompt...]
      espresso-generate generate [options] [prompt...]
      espresso-generate chat [options]
      espresso-generate compare [options] [prompt...]
      espresso-generate bench [options] [prompt...]
      espresso-generate suite [options]
      espresso-generate doctor [--json]
      espresso-generate [options] [prompt...]

    Commands:
      demo       First-run GPT-2 experience. Boots assets if needed and runs live compare in a TTY.
      generate   Run Espresso ANE generation only.
      chat       Multi-turn conversation. TTY default is a compact TUI; --plain or a non-TTY uses readline.
      compare    Compare Espresso against a Core ML baseline. Use --live or --bench.
      bench      Run a fair sequential benchmark, export JSON/CSV/Markdown, and capture power when available.
      suite      Run a multi-prompt compare suite in one process and reuse the Core ML baseline across prompts.
      doctor     Validate hardware, scripts, caches, demo assets, Python tooling, and power telemetry support.

    Common options:
      -m, --model NAME|PATH    ModelRegistry key or alias, or a path to an `.esp` bundle
      -b, --bundle PATH        `.esp` bundle directory (preferred runtime input)
      -w, --weights DIR        Weights directory
      -t, --tokenizer DIR      Tokenizer directory or tokenizer asset path
          --coreml-model PATH  Override the exported Core ML baseline (required for non-GPT2 models)
      -p, --prompt TEXT        Prompt text; otherwise use trailing args or piped stdin
      -n, --max-tokens N       Max new tokens (default: 128)
          --temperature FLOAT  Sampling temperature; 0 = greedy (generate default: 0; chat default: 0.7)
          --top-p FLOAT        Nucleus sampling in (0, 1] (generate default: 1; chat default: 0.9)
          --greedy             Force greedy decode (temperature=0, top-p=1). Required for chat parity.
          --system TEXT        Optional system prompt for chat (Qwen Instruct default otherwise)
          --plain              Chat: readline/stdio instead of the TTY conversation TUI
          --vs mlx             Dual-pane Espresso vs MLX on Qwen/Qwen2.5-1.5B-Instruct.
                               Requires --greedy. Opponent is MLX, not Core ML Stories.
                               Both lanes use the same prompt, max new tokens, and argmax.
                               Espresso: fp16 .esp hybrid, fallback disabled.
                               MLX: native fp16/bf16. 4-bit is rejected unless --mlx-quant
                               is set and both footers label it.
                               tok/s is completion-only; compile ms is shown per lane.
                               J/tok = package_watts / tok_s when telemetry is up.
          --mlx-quant 4bit     Permit a labeled quantized MLX lane. Required to load 4-bit.
          --mlx-model PATH|ID  Optional MLX path. Native runs must still be
                               Qwen/Qwen2.5-1.5B-Instruct.
          --seed N             Sampling seed for compare/bench (default: 1234)
          --prepare-demo       Prepare the managed GPT-2 demo assets and exit when no prompt is provided
          --output-dir DIR     Report output directory for compare/bench exports
          --compare-warmup N   Warmup iterations for sequential compare/bench (default: 1)
          --compare-iterations N
                               Measured iterations for sequential compare/bench (default: 3)
          --runs N             Warm runs per prompt for suite mode (default: 3)
          --prompts FILE       Prompt suite file for suite mode (id:prompt_text per line)
          --results-tsv PATH   Append per-run suite measurements to a TSV file
          --no-cold            Skip the suite cold run
          --coreml-seq-len N   Override the fixed Core ML sequence length used for compare/export
          --coreml-compute-units all|cpu_and_neural_engine|cpu_and_gpu|cpu_only
          --tui                Force the live terminal dashboard
          --no-tui             Disable the live terminal dashboard
          --live               Prefer the live compare experience
          --bench              Prefer the fair sequential benchmark path
          --power              Require power telemetry; exits if unavailable
          --no-power           Disable power telemetry
                               Chat: J/tok = package_watts / tok_s for that completion
                               (generate window only; compile is excluded from tok/s and J/tok).
                               Enable passwordless sudo for /usr/bin/powermetrics on the recording machine:
                                 echo "$USER ALL=(root) NOPASSWD: /usr/bin/powermetrics" | sudo tee /etc/sudoers.d/espresso-powermetrics
                                 sudo chmod 440 /etc/sudoers.d/espresso-powermetrics
                               The TUI never prints these sudo instructions.
          --json               Emit JSON doctor/compare output to stdout
          --probe-gpt2-attention-compile
                               Doctor-only compile probe for GPT-2 attention MIL at spatial sizes 64, 128, and 256
          --mil-deployment-target TARGET
                               Override the emitted MIL deployment target (for example: ios18, ios19, macos26)
          --gpt2-norm layernorm|rmsnorm
                               Override GPT-2 normalization used by the real-model ANE path
          --list-models        Print available models and exit
          --no-bootstrap       Do not auto-install/download demo dependencies or assets
          --no-stats           Suppress timing stats on stderr
          --benchmark-generate Reuse one Espresso engine and report warm generate metrics using --compare-warmup/--compare-iterations
          --raw-prompt         Encode the prompt verbatim (skip the Qwen2.5-Instruct chat wrap)
          --no-hybrid-fallback Fail instead of falling back from the ANE hybrid decode path
                           Chat always sets ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK=1
      -h, --help               Show this help

    Environment:
      ESPRESSO_HOME                 Override the managed cache/report root
      ESPRESSO_CACHE_HOME           Override the managed cache directory
      ESPRESSO_REPO_ROOT            Override repository root detection
      ESPRESSO_SCRIPTS_DIR          Override the helper scripts directory
      ESPRESSO_MODEL
      ESPRESSO_BUNDLE_PATH
      ESPRESSO_WEIGHTS_DIR
      ESPRESSO_TOKENIZER_DIR
      ESPRESSO_COREML_MODEL
      ESPRESSO_COREML_SEQ_LEN
      ESPRESSO_COREML_COMPUTE_UNITS
      ESPRESSO_COMPARE_WARMUP
      ESPRESSO_COMPARE_ITERATIONS
      ESPRESSO_SUITE_RUNS
      ESPRESSO_COMPARE_SEED
      ESPRESSO_MAX_TOKENS
      ESPRESSO_TEMPERATURE
      ESPRESSO_TOP_P
      ESPRESSO_MIL_DEPLOYMENT_TARGET
      ESPRESSO_GPT2_NORM
      ESPRESSO_GPT2_USE_RMS_NORM
      ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK
      ESPRESSO_MLX_PYTHON

    Examples:
      ./espresso
      ./espresso "Hello"
      ./espresso doctor
      espresso-generate demo "Hello"
      espresso-generate doctor
      espresso-generate demo
      espresso-generate compare --live "Hello"
      espresso-generate bench --output-dir ./reports/gpt2 "Hello"
      espresso-generate suite --prompts ./scripts/benchmark-prompts.txt --no-power
      ./espresso chat --model ~/Library/Caches/Espresso/qwen25-15b/Qwen2.5-1.5B-Instruct.esp
      ./espresso chat --model <model.esp> --plain --greedy
      ./espresso chat --model <model.esp> --power
      ./espresso chat --vs mlx --greedy --model ~/Library/Caches/Espresso/qwen25-15b/Qwen2.5-1.5B-Instruct.esp
    """
}

private func printUsage() {
    print(cliUsageText())
}

private func printAvailableModels() {
    print("Available models:")
    for name in ModelRegistry.all.keys.sorted() {
        guard let config = ModelRegistry.all[name] else { continue }
        let status = "ready"
        let paddedName = name.padding(toLength: 16, withPad: " ", startingAt: 0)
        let paddedArchitecture = architectureLabel(config.architecture).padding(toLength: 5, withPad: " ", startingAt: 0)
        print("  \(paddedName)  arch=\(paddedArchitecture)  layers=\(config.nLayer)  dModel=\(config.dModel)  maxSeq=\(config.maxSeq)  \(status)")
    }
    print("")
    print("Aliases: gpt2, stories, smollm, tinyllama")
}

private func ensureDirectoryExists(_ path: String, label: String) throws -> String {
    let fileManager = FileManager()
    let expanded = NSString(string: path).expandingTildeInPath
    var isDirectory: ObjCBool = false
    guard fileManager.fileExists(atPath: expanded, isDirectory: &isDirectory), isDirectory.boolValue else {
        throw CLIError.usage("\(label) directory does not exist: \(path)")
    }
    return URL(fileURLWithPath: expanded, isDirectory: true).standardizedFileURL.path
}

private func parseMetadataConfig(from weightsDirURL: URL) throws -> MultiModelConfig? {
    let fileManager = FileManager()
    let metadataURL = weightsDirURL.appendingPathComponent("metadata.json")
    guard fileManager.fileExists(atPath: metadataURL.path) else {
        return nil
    }
    let data = try Data(contentsOf: metadataURL)
    let metadata = try JSONDecoder().decode(MetadataConfigFile.self, from: data)
    return try metadata.asConfig()
}

private func resolveConfig(modelName: String?, weightsDirURL: URL) throws -> MultiModelConfig {
    if let modelName, !modelName.isEmpty {
        guard let canonical = canonicalModelName(for: modelName), let config = ModelRegistry.config(named: canonical) else {
            throw CLIError.usage("Unknown model: \(modelName). Use --list-models to inspect available models.")
        }
        return config
    }

    if let metadataConfig = try parseMetadataConfig(from: weightsDirURL) {
        return metadataConfig
    }

    if let inferred = canonicalModelName(for: weightsDirURL.lastPathComponent),
       let config = ModelRegistry.config(named: inferred)
    {
        return config
    }

    throw CLIError.usage(
        "Unable to resolve a model config. Pass --model, set ESPRESSO_MODEL, or provide weights/metadata.json."
    )
}

func hasGPT2TokenizerAssets(in directory: URL) -> Bool {
    let fileManager = FileManager()
    return fileManager.fileExists(atPath: directory.appendingPathComponent("vocab.json").path) &&
        fileManager.fileExists(atPath: directory.appendingPathComponent("merges.txt").path)
}

func sentencePieceTokenizerURL(in directory: URL) -> URL? {
    let fileManager = FileManager()
    for filename in ["tokenizer.model", "tokenizer.bin"] {
        let candidate = directory.appendingPathComponent(filename)
        if fileManager.fileExists(atPath: candidate.path) {
            return candidate
        }
    }
    return nil
}

private func hasSentencePieceAssets(in directory: URL) -> Bool {
    sentencePieceTokenizerURL(in: directory) != nil
}

private func hasTokenizerJSONAssets(in directory: URL) -> Bool {
    FileManager.default.fileExists(atPath: directory.appendingPathComponent("tokenizer.json").path)
}

private func hasBPEPairAssets(in directory: URL) -> Bool {
    let fileManager = FileManager()
    return fileManager.fileExists(atPath: directory.appendingPathComponent("vocab.json").path) &&
        fileManager.fileExists(atPath: directory.appendingPathComponent("merges.txt").path)
}

private func directoryHasTokenizerAssets(_ directory: URL, for config: MultiModelConfig) -> Bool {
    switch config.architecture {
    case .gpt2:
        return hasGPT2TokenizerAssets(in: directory)
    case .llama:
        return hasSentencePieceAssets(in: directory) ||
            hasTokenizerJSONAssets(in: directory) ||
            hasBPEPairAssets(in: directory)
    }
}

private func tokenizerDirectoryCandidates(weightsDirURL: URL, config: MultiModelConfig) -> [URL] {
    let parent = weightsDirURL.deletingLastPathComponent()
    let architecture = architectureLabel(config.architecture)
    let names = [
        weightsDirURL.lastPathComponent,
        config.name,
        architecture,
    ]

    var candidates: [URL] = [
        weightsDirURL,
        weightsDirURL.appendingPathComponent("tokenizer", isDirectory: true),
        weightsDirURL.appendingPathComponent("tokenizer_files", isDirectory: true),
        parent,
        parent.appendingPathComponent("tokenizer", isDirectory: true),
    ]

    for name in names {
        candidates.append(parent.appendingPathComponent("\(name)_tokenizer", isDirectory: true))
        candidates.append(parent.appendingPathComponent("\(name)-tokenizer", isDirectory: true))
    }

    var seen: Set<String> = []
    return candidates.filter { seen.insert($0.standardizedFileURL.path).inserted }
}

private func normalizeTokenizerLocation(_ rawPath: String, config: MultiModelConfig) throws -> String {
    let fileManager = FileManager()
    let expanded = NSString(string: rawPath).expandingTildeInPath
    let url = URL(fileURLWithPath: expanded)

    var isDirectory: ObjCBool = false
    if fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) {
        if isDirectory.boolValue {
            let standardized = url.standardizedFileURL
            guard directoryHasTokenizerAssets(standardized, for: config) else {
                throw CLIError.usage("Tokenizer directory is missing required assets: \(rawPath)")
            }
            return standardized.path
        }

        let parent = url.deletingLastPathComponent().standardizedFileURL
        switch config.architecture {
        case .gpt2:
            if ["vocab.json", "merges.txt"].contains(url.lastPathComponent), hasGPT2TokenizerAssets(in: parent) {
                return parent.path
            }
        case .llama:
            if ["tokenizer.model", "tokenizer.bin"].contains(url.lastPathComponent), hasSentencePieceAssets(in: parent) {
                return parent.path
            }
            if url.lastPathComponent == "tokenizer.json", hasTokenizerJSONAssets(in: parent) {
                return parent.path
            }
            if ["vocab.json", "merges.txt"].contains(url.lastPathComponent), hasBPEPairAssets(in: parent) {
                return parent.path
            }
        }
        throw CLIError.usage("Tokenizer path must point to a tokenizer directory or one of its primary assets.")
    }

    throw CLIError.usage("Tokenizer path does not exist: \(rawPath)")
}

private func resolveTokenizerDirectory(
    tokenizerArgument: String?,
    weightsDirURL: URL,
    config: MultiModelConfig
) throws -> String {
    if let tokenizerArgument, !tokenizerArgument.isEmpty {
        return try normalizeTokenizerLocation(tokenizerArgument, config: config)
    }

    for candidate in tokenizerDirectoryCandidates(weightsDirURL: weightsDirURL, config: config) {
        if directoryHasTokenizerAssets(candidate, for: config) {
            return candidate.standardizedFileURL.path
        }
    }

    switch config.architecture {
    case .gpt2:
        throw CLIError.usage(
            "Unable to locate GPT-2 tokenizer assets. Pass --tokenizer or place vocab.json + merges.txt near the weights."
        )
    case .llama:
        throw CLIError.usage(
            "Unable to locate llama tokenizer assets. Pass --tokenizer or place tokenizer.model, tokenizer.bin, tokenizer.json, or vocab.json + merges.txt near the weights."
        )
    }
}

private func readPromptFromStandardInput() throws -> String? {
    guard isatty(STDIN_FILENO) == 0 else {
        return nil
    }
    let data = try FileHandle.standardInput.readToEnd() ?? Data()
    guard !data.isEmpty else {
        return nil
    }
    var prompt = String(decoding: data, as: UTF8.self)
    while prompt.last?.isNewline == true {
        prompt.removeLast()
    }
    return prompt.isEmpty ? nil : prompt
}

func implicitPrompt(
    command: CommandName,
    options: Options
) -> String? {
    guard shouldUseDefaultGPT2Demo(options) else {
        return nil
    }
    switch command {
    case .demo, .compare:
        return defaultDemoPrompt
    case .generate, .bench, .suite, .doctor, .chat:
        return nil
    }
}

private func commandRequiresPrompt(_ command: CommandName) -> Bool {
    switch command {
    case .doctor, .suite, .chat:
        return false
    case .demo, .generate, .compare, .bench:
        return true
    }
}

private func resolvePrompt(_ options: Options, command: CommandName) throws -> String {
    if let prompt = options.prompt, !prompt.isEmpty {
        return prompt
    }
    if !options.positionalPrompt.isEmpty {
        return options.positionalPrompt.joined(separator: " ")
    }
    if let stdinPrompt = try readPromptFromStandardInput() {
        return stdinPrompt
    }
    if let fallbackPrompt = implicitPrompt(command: command, options: options) {
        return fallbackPrompt
    }
    throw CLIError.usage("Missing prompt. Pass --prompt, provide trailing text, or pipe stdin.")
}

private func loadTokenizer(config: MultiModelConfig, tokenizerDir: String) throws -> CLITokenizer {
    let tokenizerDirURL = URL(fileURLWithPath: tokenizerDir, isDirectory: true)
    switch config.architecture {
    case .gpt2:
        let vocabURL = tokenizerDirURL.appendingPathComponent("vocab.json")
        let mergesURL = tokenizerDirURL.appendingPathComponent("merges.txt")
        let tokenizer = try GPT2BPETokenizer(vocabURL: vocabURL, mergesURL: mergesURL)
        return CLITokenizer(encodeImpl: tokenizer.encode, decodeImpl: tokenizer.decode)
    case .llama:
        if let sentencePieceURL = sentencePieceTokenizerURL(in: tokenizerDirURL) {
            let tokenizer = try SentencePieceTokenizer(modelURL: sentencePieceURL)
            return CLITokenizer(encodeImpl: tokenizer.encode, decodeImpl: tokenizer.decode)
        }

        let tokenizerJSONURL = tokenizerDirURL.appendingPathComponent("tokenizer.json")
        if FileManager.default.fileExists(atPath: tokenizerJSONURL.path) {
            let tokenizer = try GPT2BPETokenizer(tokenizerJSONURL: tokenizerJSONURL)
            return CLITokenizer(encodeImpl: tokenizer.encode, decodeImpl: tokenizer.decode)
        }

        let vocabURL = tokenizerDirURL.appendingPathComponent("vocab.json")
        let mergesURL = tokenizerDirURL.appendingPathComponent("merges.txt")
        let tokenizer = try GPT2BPETokenizer(vocabURL: vocabURL, mergesURL: mergesURL)
        return CLITokenizer(encodeImpl: tokenizer.encode, decodeImpl: tokenizer.decode)
    }
}

private func shouldExitAfterPreparingDemo(_ options: Options) -> Bool {
    guard options.prepareDemo else {
        return false
    }
    return options.prompt == nil && options.positionalPrompt.isEmpty
}

private func defaultCommand(for options: Options) -> CommandName {
    if options.prepareDemo {
        return .demo
    }
    if options.preferBenchCompare {
        return .bench
    }
    if options.preferLiveCompare || options.forceTUI {
        return .compare
    }
    if shouldUseDefaultGPT2Demo(options) && isatty(STDOUT_FILENO) == 1 {
        return .demo
    }
    return .generate
}

private func shouldUseTUI(command: CommandName, options: Options) -> Bool {
    if options.disableTUI {
        return false
    }
    if options.forceTUI {
        return true
    }
    guard isatty(STDOUT_FILENO) == 1 else {
        return false
    }
    guard let term = ProcessInfo.processInfo.environment["TERM"], term != "dumb" else {
        return false
    }
    switch command {
    case .demo:
        return true
    case .compare:
        return !options.preferBenchCompare
    case .chat:
        return !options.plain
    case .generate, .bench, .suite, .doctor:
        return false
    }
}

/// Treats a `--model` value as a bundle path when it names an `.esp` directory on disk.
///
/// Registry keys never contain a path separator and never end in `.esp`, so a value that
/// does is unambiguously a filesystem argument. Reporting a missing manifest here is more
/// useful than the registry's "Unknown model" error.
func bundlePathFromModelArgument(_ value: String, fileManager: FileManager = .default) throws -> String? {
    let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return nil }
    let expanded = (trimmed as NSString).expandingTildeInPath
    let looksLikePath = expanded.hasSuffix(".esp") || expanded.contains("/")
    guard looksLikePath else { return nil }

    var isDirectory = ObjCBool(false)
    guard fileManager.fileExists(atPath: expanded, isDirectory: &isDirectory), isDirectory.boolValue else {
        throw CLIError.usage("Model bundle not found: \(trimmed)")
    }
    let manifest = URL(fileURLWithPath: expanded, isDirectory: true)
        .appendingPathComponent(ESPBundleLayout.manifestFileName)
    guard fileManager.fileExists(atPath: manifest.path) else {
        throw CLIError.usage(
            "\(trimmed) is not an .esp bundle: missing \(ESPBundleLayout.manifestFileName)."
        )
    }
    return expanded
}

func resolveInvocation(from options: Options, demoDefaults: DemoDefaults, command: CommandName) throws -> ResolvedInvocation {
    var bundleArgument = options.bundlePath
    if let modelArgument = options.modelName,
       let modelBundle = try bundlePathFromModelArgument(modelArgument) {
        if let bundleArgument, !bundleArgument.isEmpty, bundleArgument != modelBundle {
            throw CLIError.usage("--model names an .esp bundle that conflicts with --bundle.")
        }
        bundleArgument = modelBundle
    }

    if let bundleArgument, !bundleArgument.isEmpty {
        if options.weightsDir != nil || options.tokenizerDir != nil {
            throw CLIError.usage("Pass either --bundle or --weights/--tokenizer, not both.")
        }

        let bundlePath = try ensureDirectoryExists(bundleArgument, label: "Bundle")
        let bundle = try ESPRuntimeBundle.open(at: URL(fileURLWithPath: bundlePath, isDirectory: true))
        let prompt = commandRequiresPrompt(command) ? try resolvePrompt(options, command: command) : ""
        let sampling = resolvedSampling(command: command, options: options)
        return ResolvedInvocation(
            config: bundle.config,
            bundlePath: bundlePath,
            weightsDir: bundle.archive.weightsURL.path,
            tokenizerDir: bundle.archive.tokenizerURL.path,
            prompt: prompt,
            maxTokens: options.maxTokens,
            temperature: sampling.temperature,
            topP: sampling.topP,
            showStats: options.showStats,
            coreMLModelPath: options.coreMLModelPath,
            coreMLSequenceLength: options.coreMLSequenceLength,
            compareWarmup: options.compareWarmup,
            compareIterations: options.compareIterations,
            benchmarkGenerate: options.benchmarkGenerate,
            coreMLComputeUnits: options.coreMLComputeUnits,
            allowBootstrap: options.allowBootstrap,
            seed: options.seed,
            outputDir: options.outputDir,
            rawPrompt: options.rawPrompt
        )
    }

    var modelName = options.modelName
    var weightsArgument = options.weightsDir
    var tokenizerArgument = options.tokenizerDir

    if shouldUseDefaultGPT2Demo(options) {
        modelName = modelName ?? "gpt2"
        if weightsArgument == nil {
            weightsArgument = demoDefaults.weightsDir.path
            tokenizerArgument = tokenizerArgument ?? demoDefaults.tokenizerDir.path
        }
    }

    guard let weightsArgument, !weightsArgument.isEmpty else {
        if command == .chat {
            throw CLIError.usage("chat requires --model <path.esp> (or --bundle / --weights).")
        }
        throw CLIError.usage("Missing weights directory. Pass --weights or set ESPRESSO_WEIGHTS_DIR.")
    }
    let weightsDir = try ensureDirectoryExists(weightsArgument, label: "Weights")
    let weightsDirURL = URL(fileURLWithPath: weightsDir, isDirectory: true)
    let config = try resolveConfig(modelName: modelName, weightsDirURL: weightsDirURL)
    let tokenizerDir = try resolveTokenizerDirectory(
        tokenizerArgument: tokenizerArgument,
        weightsDirURL: weightsDirURL,
        config: config
    )
    let prompt = commandRequiresPrompt(command) ? try resolvePrompt(options, command: command) : ""
    let sampling = resolvedSampling(command: command, options: options)
    return ResolvedInvocation(
        config: config,
        bundlePath: nil,
        weightsDir: weightsDir,
        tokenizerDir: tokenizerDir,
        prompt: prompt,
        maxTokens: options.maxTokens,
        temperature: sampling.temperature,
        topP: sampling.topP,
        showStats: options.showStats,
        coreMLModelPath: options.coreMLModelPath,
        coreMLSequenceLength: options.coreMLSequenceLength,
        compareWarmup: options.compareWarmup,
        compareIterations: options.compareIterations,
        benchmarkGenerate: options.benchmarkGenerate,
        coreMLComputeUnits: options.coreMLComputeUnits,
        allowBootstrap: options.allowBootstrap,
        seed: options.seed,
        outputDir: options.outputDir,
        rawPrompt: options.rawPrompt
    )
}

private func printPreparedDemoSummary(_ defaults: DemoDefaults) {
    stderrLine("Default GPT-2 demo assets are ready.")
    stderrLine("weights=\(defaults.weightsDir.path)")
    stderrLine("tokenizer=\(defaults.tokenizerDir.path)")
    stderrLine("next=./espresso")
}

private func makeBundleRuntimeMetrics(bundle: ESPRuntimeBundle, invocation: ResolvedInvocation) throws -> BackendRunMetrics {
    let selection = try ESPRuntimeRunner.resolve(bundle: bundle)
    let compileStatsBefore = ANECompileStats.snapshot()
    let started = DispatchTime.now().uptimeNanoseconds
    let result = try ESPRuntimeRunner.generate(
        bundle: bundle,
        prompt: preparedGeneratePrompt(
            invocation.prompt,
            config: invocation.config,
            rawPrompt: invocation.rawPrompt
        ),
        maxTokens: invocation.maxTokens,
        temperature: invocation.temperature
    )
    let totalTimeMs = millisecondsSince(started)
    let compileStatsAfter = ANECompileStats.snapshot()
    let compileStatsDelta = compileStatsAfter.subtracting(compileStatsBefore)
    let tokenLatenciesMs = result.tokenLatenciesMs
    return BackendRunMetrics(
        backend: selection.backend.rawValue,
        text: result.text,
        generatedTokens: result.tokens,
        promptTokens: result.promptTokens,
        compileTimeMs: result.compileTimeMs,
        firstTokenLatencyMs: result.firstTokenLatencyMs,
        tokensPerSecond: result.tokensPerSecond,
        medianTokenMs: percentile(tokenLatenciesMs, percentile: 0.5),
        p95TokenMs: percentile(tokenLatenciesMs, percentile: 0.95),
        totalTimeMs: totalTimeMs,
        tokenLatenciesMs: tokenLatenciesMs,
        compileRetryCount: compileStatsDelta.retryCount,
        compileFailureCount: compileStatsDelta.failureCount,
        compileBreakdown: compileStatsDelta.labelStats.isEmpty ? nil : compileStatsDelta.labelStats,
        exactHeadBackend: result.exactHeadBackend,
        cachedBindingsEnabled: result.cachedBindingsEnabled,
        committedExactTokensPerPass: result.committedExactTokensPerPass,
        acceptedFutureTokensPerPass: result.acceptedFutureTokensPerPass,
        decodePath: result.decodePath,
        hopsPerToken: result.hopsPerToken,
        decodeProfileReport: result.decodeProfileReport
    )
}

private func percentile(_ latencies: [Double], percentile: Double) -> Double {
    guard !latencies.isEmpty else { return 0 }
    let sorted = latencies.sorted()
    let clamped = min(max(percentile, 0), 1)
    let index = clamped * Double(sorted.count - 1)
    let lower = Int(index.rounded(.down))
    let upper = min(lower + 1, sorted.count - 1)
    if lower == upper {
        return sorted[lower]
    }
    let fraction = index - Double(lower)
    return sorted[lower] + ((sorted[upper] - sorted[lower]) * fraction)
}

func resolvedANECompileCachePolicy(environment: [String: String]) -> String {
    guard let explicit = environment["ANE_COMPILE_CACHE_POLICY"], !explicit.isEmpty else {
        return "preferCached"
    }
    return explicit
}

private func applyDefaultANECompileCachePolicyIfNeeded() {
    let policy = resolvedANECompileCachePolicy(environment: ProcessInfo.processInfo.environment)
    setenv("ANE_COMPILE_CACHE_POLICY", policy, 0)
}

private func makeEspressoEngine(invocation: ResolvedInvocation) throws -> RealModelInferenceEngine {
    applyDefaultANECompileCachePolicyIfNeeded()
    return try RealModelInferenceEngine.build(
        config: invocation.config,
        weightDir: invocation.weightsDir,
        tokenizerDir: invocation.tokenizerDir
    )
}

private func runEspressoGeneration(
    engine: inout RealModelInferenceEngine,
    invocation: ResolvedInvocation,
    onStep: ((GenerationStep) -> Void)? = nil
) throws -> BackendRunMetrics {
    var tokenLatenciesMs: [Double] = []
    var totalTimeMs = 0.0
    let compileStatsBefore = ANECompileStats.snapshot()
    let result = try engine.generate(
        prompt: preparedGeneratePrompt(
            invocation.prompt,
            config: invocation.config,
            rawPrompt: invocation.rawPrompt
        ),
        maxTokens: invocation.maxTokens,
        temperature: invocation.temperature,
        topP: invocation.topP,
        onStep: { step in
            tokenLatenciesMs.append(step.tokenLatencyMs)
            totalTimeMs = step.elapsedMs
            onStep?(step)
        }
    )
    let compileStatsAfter = ANECompileStats.snapshot()
    let compileStatsDelta = compileStatsAfter.subtracting(compileStatsBefore)
    return BackendRunMetrics(
        backend: "espresso",
        text: result.text,
        generatedTokens: result.tokens,
        promptTokens: result.promptTokens,
        compileTimeMs: result.compileTimeMs,
        firstTokenLatencyMs: result.firstTokenLatencyMs,
        tokensPerSecond: result.tokensPerSecond,
        medianTokenMs: percentile(tokenLatenciesMs, percentile: 0.5),
        p95TokenMs: percentile(tokenLatenciesMs, percentile: 0.95),
        totalTimeMs: totalTimeMs,
        tokenLatenciesMs: tokenLatenciesMs,
        compileRetryCount: compileStatsDelta.retryCount,
        compileFailureCount: compileStatsDelta.failureCount,
        compileBreakdown: compileStatsDelta.labelStats.isEmpty ? nil : compileStatsDelta.labelStats,
        exactHeadBackend: result.exactHeadBackend,
        cachedBindingsEnabled: result.cachedBindingsEnabled,
        committedExactTokensPerPass: result.committedExactTokensPerPass,
        acceptedFutureTokensPerPass: result.acceptedFutureTokensPerPass,
        decodePath: result.decodePath,
        hopsPerToken: result.hopsPerToken,
        decodeProfileReport: result.decodeProfileReport
    )
}

func aggregateBenchmarkRuns(
    warmup: Int,
    iterations: Int,
    run: () throws -> BackendRunMetrics
) throws -> BackendRunMetrics {
    guard warmup >= 0 else {
        throw CLIError.usage("Warmup iterations must be >= 0")
    }
    guard iterations > 0 else {
        throw CLIError.usage("Measured iterations must be > 0")
    }

    let totalIterations = warmup + iterations
    var lastMeasured: BackendRunMetrics?
    var compileTimeMs = 0.0
    var aggregatedLatencySamples: [Double] = []
    var compileRetryCount = 0
    var compileFailureCount = 0
    var compileBreakdown: [String: ANECompileLabelStatsSnapshot]?
    var exactHeadBackend: String?
    var cachedBindingsEnabled: Bool?
    var committedExactTokensPerPass: Double?
    var acceptedFutureTokensPerPass: Double?

    for iteration in 0..<totalIterations {
        let metrics = try run()
        if compileTimeMs == 0, metrics.compileTimeMs > 0 {
            compileTimeMs = metrics.compileTimeMs
            compileRetryCount = metrics.compileRetryCount
            compileFailureCount = metrics.compileFailureCount
            compileBreakdown = metrics.compileBreakdown
        }
        exactHeadBackend = exactHeadBackend ?? metrics.exactHeadBackend
        cachedBindingsEnabled = cachedBindingsEnabled ?? metrics.cachedBindingsEnabled
        committedExactTokensPerPass = committedExactTokensPerPass ?? metrics.committedExactTokensPerPass
        acceptedFutureTokensPerPass = acceptedFutureTokensPerPass ?? metrics.acceptedFutureTokensPerPass
        if iteration >= warmup {
            lastMeasured = metrics
            aggregatedLatencySamples.append(contentsOf: metrics.tokenLatenciesMs)
        }
    }

    guard let lastMeasured else {
        throw CLIError.runtime("Benchmark did not produce a measured run.")
    }

    return BackendRunMetrics(
        backend: lastMeasured.backend,
        text: lastMeasured.text,
        generatedTokens: lastMeasured.generatedTokens,
        promptTokens: lastMeasured.promptTokens,
        compileTimeMs: compileTimeMs,
        firstTokenLatencyMs: lastMeasured.firstTokenLatencyMs,
        tokensPerSecond: lastMeasured.tokensPerSecond,
        medianTokenMs: percentile(aggregatedLatencySamples, percentile: 0.5),
        p95TokenMs: percentile(aggregatedLatencySamples, percentile: 0.95),
        totalTimeMs: lastMeasured.totalTimeMs,
        tokenLatenciesMs: lastMeasured.tokenLatenciesMs,
        compileRetryCount: compileRetryCount != 0 ? compileRetryCount : lastMeasured.compileRetryCount,
        compileFailureCount: compileFailureCount != 0 ? compileFailureCount : lastMeasured.compileFailureCount,
        compileBreakdown: compileBreakdown ?? lastMeasured.compileBreakdown,
        exactHeadBackend: exactHeadBackend ?? lastMeasured.exactHeadBackend,
        cachedBindingsEnabled: cachedBindingsEnabled ?? lastMeasured.cachedBindingsEnabled,
        committedExactTokensPerPass: committedExactTokensPerPass ?? lastMeasured.committedExactTokensPerPass,
        acceptedFutureTokensPerPass: acceptedFutureTokensPerPass ?? lastMeasured.acceptedFutureTokensPerPass,
        decodePath: lastMeasured.decodePath,
        hopsPerToken: lastMeasured.hopsPerToken,
        decodeProfileReport: lastMeasured.decodeProfileReport
    )
}

private func runEspressoBenchmark(invocation: ResolvedInvocation) throws -> BackendRunMetrics {
    if let bundlePath = invocation.bundlePath {
        let bundle = try ESPRuntimeBundle.open(at: URL(fileURLWithPath: bundlePath, isDirectory: true))
        return try aggregateBenchmarkRuns(warmup: invocation.compareWarmup, iterations: invocation.compareIterations) {
            try makeBundleRuntimeMetrics(bundle: bundle, invocation: invocation)
        }
    }
    var engine = try makeEspressoEngine(invocation: invocation)
    return try aggregateBenchmarkRuns(warmup: invocation.compareWarmup, iterations: invocation.compareIterations) {
        try runEspressoGeneration(engine: &engine, invocation: invocation)
    }
}

private struct CoreMLRunOutput {
    let metrics: BackendRunMetrics
    let sequenceLength: Int
}

private func runCoreMLGeneration(
    invocation: ResolvedInvocation,
    defaults: DemoDefaults,
    promptTokens: [TokenID]
) throws -> CoreMLRunOutput {
    let sequenceLength = invocation.coreMLSequenceLength ?? nextPowerOfTwo(
        min(invocation.config.maxSeq, promptTokens.count + max(invocation.maxTokens, 1))
    )
    let coreMLModelPath = try resolveCoreMLModelPath(
        invocation: invocation,
        defaults: defaults,
        sequenceLength: sequenceLength
    )
    var runner = try ReusableCoreMLBenchmarkRunner(
        modelPath: coreMLModelPath,
        weightsDir: invocation.weightsDir,
        architecture: invocation.config.architecture,
        sequenceLength: sequenceLength,
        computeUnits: invocation.coreMLComputeUnits
    )
    let result = try runner.benchmark(
        promptTokens: promptTokens,
        maxTokens: invocation.maxTokens,
        temperature: invocation.temperature,
        warmup: invocation.compareWarmup,
        iterations: invocation.compareIterations,
        seed: invocation.seed
    )
    let tokenizer = try loadTokenizer(config: invocation.config, tokenizerDir: invocation.tokenizerDir)
    let generatedTokens = try result.generatedTokens.map(validateToken)
    let text = tokenizer.decode((promptTokens + generatedTokens).map(Int.init))
    return CoreMLRunOutput(
        metrics: BackendRunMetrics(
            backend: "coreml",
            text: text,
            generatedTokens: generatedTokens,
            promptTokens: promptTokens,
            compileTimeMs: result.compileTimeMs,
            firstTokenLatencyMs: result.firstTokenLatencyMs,
            tokensPerSecond: result.tokensPerSecond,
            medianTokenMs: result.medianTokenMs,
            p95TokenMs: result.p95TokenMs,
            totalTimeMs: result.totalTimeMs,
            tokenLatenciesMs: result.tokenLatenciesMs
        ),
        sequenceLength: result.seqLen
    )
}

private func validateToken(_ raw: Int) throws -> TokenID {
    guard raw >= 0, raw <= Int(TokenID.max) else {
        throw CLIError.runtime("Received invalid token id: \(raw)")
    }
    return TokenID(raw)
}

func resolvePowerEnabled(
    command: CommandName,
    powerMode: PowerMode,
    capability: PowerCapability = PowerTelemetryCollector.capability(),
    emitWarnings: Bool = true
) throws -> Bool {
    switch powerMode {
    case .off:
        return false
    case .on:
        guard capability.available else {
            throw CLIError.runtime("Power telemetry is unavailable: \(capability.message)")
        }
        return true
    case .auto:
        let enabledByDefault = command == .demo || command == .bench || command == .chat
        if emitWarnings, enabledByDefault, command != .chat, !capability.available {
            stderrLine("espresso-generate warning: power telemetry unavailable: \(capability.message)")
        }
        return enabledByDefault && capability.available
    }
}

private func maybeMeasurePower<T>(
    enabled: Bool,
    sampleIntervalMs: Int = 1_000,
    body: () throws -> T
) throws -> (result: T, power: PowerSummary?) {
    guard enabled else {
        return (try body(), nil)
    }
    let capability = PowerTelemetryCollector.capability()
    guard capability.available else {
        return (try body(), nil)
    }
    let collector = PowerTelemetryCollector(sampleIntervalMs: sampleIntervalMs)
    do {
        try collector.start { _ in }
        let result = try body()
        let summary = collector.stop()
        return (result, summary)
    } catch {
        _ = collector.stop()
        throw error
    }
}

private func compareReport(
    invocation: ResolvedInvocation,
    espresso: BackendRunMetrics,
    coreML: BackendRunMetrics,
    coreMLSequenceLength: Int,
    espressoPower: PowerSummary?,
    coreMLPower: PowerSummary?,
    outputDirectory: String?
) -> CompareReport {
    CompareReport(
        model: invocation.config.name,
        prompt: invocation.prompt,
        maxTokens: invocation.maxTokens,
        seed: invocation.seed,
        espresso: espresso,
        coreML: coreML,
        tokenMatch: espresso.generatedTokens == coreML.generatedTokens,
        textMatch: espresso.text == coreML.text,
        coreMLComputeUnits: invocation.coreMLComputeUnits,
        coreMLSequenceLength: coreMLSequenceLength,
        espressoPower: espressoPower,
        coreMLPower: coreMLPower,
        outputDirectory: outputDirectory
    )
}

private func printGenerateStats(_ invocation: ResolvedInvocation, result: BackendRunMetrics) {
    stderrLine(
        String(
            format: "model=%@ prompt_tokens=%d generated_tokens=%d compile_ms=%.2f compile_retries=%d compile_failures=%d first_token_ms=%.2f tok_per_s=%.2f median_token_ms=%.2f p95_token_ms=%.2f",
            locale: posixLocale,
            invocation.config.name,
            result.promptTokens.count,
            result.generatedTokens.count,
            result.compileTimeMs,
            result.compileRetryCount,
            result.compileFailureCount,
            result.firstTokenLatencyMs,
            result.tokensPerSecond,
            result.medianTokenMs,
            result.p95TokenMs
        )
    )
    if let exactHeadBackend = result.exactHeadBackend {
        stderrLine("exact_head_backend=\(exactHeadBackend)")
    }
    if let cachedBindingsEnabled = result.cachedBindingsEnabled {
        stderrLine("cached_bindings_enabled=\(cachedBindingsEnabled)")
    }
    if let hopsPerToken = result.hopsPerToken {
        stderrLine("hops/token=\(hopsPerToken)")
    }
    if let committedExactTokensPerPass = result.committedExactTokensPerPass {
        stderrLine(
            String(
                format: "committed_exact_tokens_per_pass=%.4f",
                locale: posixLocale,
                committedExactTokensPerPass
            )
        )
    }
    if let acceptedFutureTokensPerPass = result.acceptedFutureTokensPerPass {
        stderrLine(
            String(
                format: "accepted_future_tokens_per_pass=%.4f",
                locale: posixLocale,
                acceptedFutureTokensPerPass
            )
        )
    }
    if let compileBreakdown = result.compileBreakdown, !compileBreakdown.isEmpty {
        stderrLine("compile_breakdown=\(formatCompileBreakdown(compileBreakdown))")
    }
    if let decodePath = result.decodePath {
        stderrLine("decode_path=\(decodePath)")
    }
    if let decodeProfileMean = generateDecodeProfileMeanLine(result.decodeProfileReport) {
        stderrLine(decodeProfileMean)
    }
    if let bundlePath = invocation.bundlePath {
        stderrLine("bundle=\(bundlePath)")
    }
    stderrLine("weights=\(invocation.weightsDir)")
    stderrLine("tokenizer=\(invocation.tokenizerDir)")
}

private func printCompareSummary(_ report: CompareReport) {
    stderrLine(
        String(
            format: "espresso compile_ms=%.2f compile_retries=%d compile_failures=%d first_token_ms=%.2f tok_per_s=%.2f median_token_ms=%.2f p95_token_ms=%.2f",
            locale: posixLocale,
            report.espresso.compileTimeMs,
            report.espresso.compileRetryCount,
            report.espresso.compileFailureCount,
            report.espresso.firstTokenLatencyMs,
            report.espresso.tokensPerSecond,
            report.espresso.medianTokenMs,
            report.espresso.p95TokenMs
        )
    )
    stderrLine(
        String(
            format: "coreml  compute=%@ seq_len=%d compile_ms=%.2f first_token_ms=%.2f tok_per_s=%.2f median_token_ms=%.2f p95_token_ms=%.2f",
            locale: posixLocale,
            report.coreMLComputeUnits,
            report.coreMLSequenceLength,
            report.coreML.compileTimeMs,
            report.coreML.firstTokenLatencyMs,
            report.coreML.tokensPerSecond,
            report.coreML.medianTokenMs,
            report.coreML.p95TokenMs
        )
    )
    if report.coreML.tokensPerSecond > 0 {
        stderrLine(
            String(
                format: "speedup_vs_coreml=%.2fx",
                locale: posixLocale,
                report.espresso.tokensPerSecond / max(report.coreML.tokensPerSecond, 1e-9)
            )
        )
    }
    stderrLine("compare_match=tokens:\(report.tokenMatch) text:\(report.textMatch)")
    if let exactHeadBackend = report.espresso.exactHeadBackend {
        stderrLine("espresso_exact_head_backend=\(exactHeadBackend)")
    }
    if let cachedBindingsEnabled = report.espresso.cachedBindingsEnabled {
        stderrLine("espresso_cached_bindings_enabled=\(cachedBindingsEnabled)")
    }
    if let compileBreakdown = report.espresso.compileBreakdown, !compileBreakdown.isEmpty {
        stderrLine("espresso_compile_breakdown=\(formatCompileBreakdown(compileBreakdown))")
    }
    if let espressoPower = report.espressoPower, espressoPower.sampleCount > 0 {
        stderrLine(
            String(
                format: "power_espresso package_w=%.2f cpu_w=%.2f gpu_w=%.2f ane_w=%.2f samples=%d",
                locale: posixLocale,
                espressoPower.packageW,
                espressoPower.cpuW,
                espressoPower.gpuW,
                espressoPower.aneW,
                espressoPower.sampleCount
            )
        )
    }
    if let coreMLPower = report.coreMLPower, coreMLPower.sampleCount > 0 {
        stderrLine(
            String(
                format: "power_coreml package_w=%.2f cpu_w=%.2f gpu_w=%.2f ane_w=%.2f samples=%d",
                locale: posixLocale,
                coreMLPower.packageW,
                coreMLPower.cpuW,
                coreMLPower.gpuW,
                coreMLPower.aneW,
                coreMLPower.sampleCount
            )
        )
    }
    if let outputDirectory = report.outputDirectory {
        stderrLine("report_dir=\(outputDirectory)")
    }
}

private func writeCompareArtifacts(report: CompareReport, defaults: DemoDefaults, requestedOutputDir: String?) throws -> String {
    let outputDir: URL = {
        if let requestedOutputDir, !requestedOutputDir.isEmpty {
            return URL(fileURLWithPath: NSString(string: requestedOutputDir).expandingTildeInPath, isDirectory: true).standardizedFileURL
        }
        let timestamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "")
        return defaults.reportsRoot.appendingPathComponent("compare-\(timestamp)", isDirectory: true)
    }()
    let fileManager = FileManager()
    try fileManager.createDirectory(at: outputDir, withIntermediateDirectories: true)

    let jsonURL = outputDir.appendingPathComponent("compare.json")
    let csvURL = outputDir.appendingPathComponent("summary.csv")
    let markdownURL = outputDir.appendingPathComponent("summary.md")
    let espressoLatenciesURL = outputDir.appendingPathComponent("espresso_token_latencies.csv")
    let coreMLLatenciesURL = outputDir.appendingPathComponent("coreml_token_latencies.csv")
    let espressoFingerprintURL = outputDir.appendingPathComponent("espresso-benchmark-fingerprint.json")
    let coreMLFingerprintURL = outputDir.appendingPathComponent("coreml-benchmark-fingerprint.json")
    let espressoFingerprint = benchmarkFingerprint(for: report.espresso)
    let coreMLFingerprint = benchmarkFingerprint(for: report.coreML)

    let payload: [String: Any] = [
        "model": report.model,
        "prompt": report.prompt,
        "max_tokens": report.maxTokens,
        "seed": report.seed,
        "token_match": report.tokenMatch,
        "text_match": report.textMatch,
        "coreml_compute_units": report.coreMLComputeUnits,
        "coreml_sequence_length": report.coreMLSequenceLength,
        "espresso": backendPayload(report.espresso),
        "coreml": backendPayload(report.coreML),
        "benchmark_fingerprints": [
            "espresso": benchmarkFingerprintPayload(espressoFingerprint),
            "coreml": benchmarkFingerprintPayload(coreMLFingerprint),
        ],
        "power": [
            "espresso": powerPayload(report.espressoPower),
            "coreml": powerPayload(report.coreMLPower),
        ],
    ]
    let jsonData = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    try jsonData.write(to: jsonURL)

    let csv = """
    backend,compile_ms,compile_retry_count,compile_failure_count,first_token_ms,tokens_per_second,median_token_ms,p95_token_ms,total_time_ms,exact_head_backend,cached_bindings_enabled,package_w,cpu_w,gpu_w,ane_w
    espresso,\(report.espresso.compileTimeMs),\(report.espresso.compileRetryCount),\(report.espresso.compileFailureCount),\(report.espresso.firstTokenLatencyMs),\(report.espresso.tokensPerSecond),\(report.espresso.medianTokenMs),\(report.espresso.p95TokenMs),\(report.espresso.totalTimeMs),\(report.espresso.exactHeadBackend ?? ""),\(report.espresso.cachedBindingsEnabled.map(String.init(describing:)) ?? ""),\(report.espressoPower?.packageW ?? 0),\(report.espressoPower?.cpuW ?? 0),\(report.espressoPower?.gpuW ?? 0),\(report.espressoPower?.aneW ?? 0)
    coreml,\(report.coreML.compileTimeMs),\(report.coreML.compileRetryCount),\(report.coreML.compileFailureCount),\(report.coreML.firstTokenLatencyMs),\(report.coreML.tokensPerSecond),\(report.coreML.medianTokenMs),\(report.coreML.p95TokenMs),\(report.coreML.totalTimeMs),\(report.coreML.exactHeadBackend ?? ""),\(report.coreML.cachedBindingsEnabled.map(String.init(describing:)) ?? ""),\(report.coreMLPower?.packageW ?? 0),\(report.coreMLPower?.cpuW ?? 0),\(report.coreMLPower?.gpuW ?? 0),\(report.coreMLPower?.aneW ?? 0)
    """
    try (csv + "\n").write(to: csvURL, atomically: true, encoding: .utf8)

    let markdown = """
    # Espresso Compare Report

    - model: `\(report.model)`
    - prompt: `\(report.prompt)`
    - max tokens: `\(report.maxTokens)`
    - seed: `\(report.seed)`
    - token match: `\(report.tokenMatch)`
    - text match: `\(report.textMatch)`

    | Backend | Compile ms | TTFT ms | Tok/s | Median token ms | P95 token ms | Total ms |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: |
    | Espresso | \(formatMarkdown(report.espresso.compileTimeMs)) | \(formatMarkdown(report.espresso.firstTokenLatencyMs)) | \(formatMarkdown(report.espresso.tokensPerSecond)) | \(formatMarkdown(report.espresso.medianTokenMs)) | \(formatMarkdown(report.espresso.p95TokenMs)) | \(formatMarkdown(report.espresso.totalTimeMs)) |
    | Core ML | \(formatMarkdown(report.coreML.compileTimeMs)) | \(formatMarkdown(report.coreML.firstTokenLatencyMs)) | \(formatMarkdown(report.coreML.tokensPerSecond)) | \(formatMarkdown(report.coreML.medianTokenMs)) | \(formatMarkdown(report.coreML.p95TokenMs)) | \(formatMarkdown(report.coreML.totalTimeMs)) |
    """
    try (markdown + "\n").write(to: markdownURL, atomically: true, encoding: .utf8)

    try writeLatencyCSV(latencies: report.espresso.tokenLatenciesMs, to: espressoLatenciesURL)
    try writeLatencyCSV(latencies: report.coreML.tokenLatenciesMs, to: coreMLLatenciesURL)
    try writeBenchmarkFingerprint(espressoFingerprint, to: espressoFingerprintURL)
    try writeBenchmarkFingerprint(coreMLFingerprint, to: coreMLFingerprintURL)
    return outputDir.path
}

private func writeGenerateArtifacts(
    invocation: ResolvedInvocation,
    result: BackendRunMetrics,
    defaults: DemoDefaults,
    requestedOutputDir: String?
) throws -> String? {
    let requestedOutputDir = requestedOutputDir?.trimmingCharacters(in: .whitespacesAndNewlines)
    guard let requestedOutputDir, !requestedOutputDir.isEmpty else {
        return nil
    }

    let outputDir = URL(
        fileURLWithPath: NSString(string: requestedOutputDir).expandingTildeInPath,
        isDirectory: true
    ).standardizedFileURL
    try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)

    let fingerprint = benchmarkFingerprint(for: result)
    let payload: [String: Any] = [
        "model": invocation.config.name,
        "bundle": invocation.bundlePath ?? NSNull(),
        "backend": backendPayload(result),
        "benchmark_fingerprint": benchmarkFingerprintPayload(fingerprint),
        "reports_root": defaults.reportsRoot.path,
    ]
    let jsonData = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    try jsonData.write(to: outputDir.appendingPathComponent("generate.json"))
    try writeLatencyCSV(
        latencies: result.tokenLatenciesMs,
        to: outputDir.appendingPathComponent("espresso_token_latencies.csv")
    )
    try writeBenchmarkFingerprint(
        fingerprint,
        to: outputDir.appendingPathComponent("benchmark-fingerprint.json")
    )
    return outputDir.path
}

private func backendPayload(_ backend: BackendRunMetrics) -> [String: Any] {
    let fingerprint = benchmarkFingerprint(for: backend)
    return [
        "compile_time_ms": backend.compileTimeMs,
        "compile_retry_count": backend.compileRetryCount,
        "compile_failure_count": backend.compileFailureCount,
        "first_token_latency_ms": backend.firstTokenLatencyMs,
        "tokens_per_second": backend.tokensPerSecond,
        "median_token_ms": backend.medianTokenMs,
        "p95_token_ms": backend.p95TokenMs,
        "total_time_ms": backend.totalTimeMs,
        "generated_tokens": backend.generatedTokens.map(Int.init),
        "token_latencies_ms": backend.tokenLatenciesMs,
        "text": backend.text,
        "benchmark_fingerprint": benchmarkFingerprintPayload(fingerprint),
        "compile_breakdown": backend.compileBreakdown?.mapValues { stats in
            [
                "attempt_count": stats.attemptCount,
                "success_count": stats.successCount,
                "failure_count": stats.failureCount,
                "retry_count": stats.retryCount,
            ]
        } ?? NSNull(),
        "exact_head_backend": backend.exactHeadBackend ?? NSNull(),
        "cached_bindings_enabled": backend.cachedBindingsEnabled ?? NSNull(),
        "decode_path": backend.decodePath ?? NSNull(),
        "hops_per_token": backend.hopsPerToken ?? NSNull(),
    ]
}

private func writeBenchmarkFingerprint(_ fingerprint: ESPBenchmarkFingerprint, to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let payload = benchmarkFingerprintPayload(fingerprint)
    try encoder.encode(payload).write(to: url)
}

private func powerPayload(_ power: PowerSummary?) -> [String: Any] {
    guard let power else {
        return ["available": false]
    }
    return [
        "available": power.sampleCount > 0,
        "package_w": power.packageW,
        "cpu_w": power.cpuW,
        "gpu_w": power.gpuW,
        "ane_w": power.aneW,
        "sample_count": power.sampleCount,
    ]
}

private func formatMarkdown(_ value: Double) -> String {
    String(format: "%.2f", locale: posixLocale, value)
}

private func writeLatencyCSV(latencies: [Double], to url: URL) throws {
    var rows = ["token_index,latency_ms"]
    rows.reserveCapacity(latencies.count + 1)
    for (index, latency) in latencies.enumerated() {
        rows.append("\(index + 1),\(latency)")
    }
    try (rows.joined(separator: "\n") + "\n").write(to: url, atomically: true, encoding: .utf8)
}

private func runDoctor(defaults: DemoDefaults, options: Options) throws -> Int32 {
    if options.probeGPT2AttentionCompile {
        let invocation = try resolveInvocation(from: options, demoDefaults: defaults, command: .doctor)
        guard invocation.config.architecture == .gpt2 else {
            throw CLIError.usage("GPT-2 attention compile probe requires a GPT-2 model/config.")
        }
        let payload = try runGPT2AttentionCompileProbe(invocation: invocation)
        if options.jsonOutput {
            let resultPayload = payload.results.map { result in
                [
                    "spatial": result.spatial,
                    "compile_succeeded": result.compileSucceeded,
                    "deployment_target": result.deploymentTarget,
                    "norm_kind": result.normKind,
                    "diagnostics": result.diagnostics,
                    "error": result.error as Any,
                ] as [String: Any]
            }
            let data = try JSONSerialization.data(withJSONObject: [
                "deployment_target": payload.deploymentTarget,
                "norm_kind": payload.normKind,
                "results": resultPayload,
            ], options: [.prettyPrinted, .sortedKeys])
            print(String(decoding: data, as: UTF8.self))
        } else {
            print("GPT-2 Attention Compile Probe")
            print("============================")
            print("deployment_target: \(payload.deploymentTarget)")
            print("norm_kind: \(payload.normKind)")
            for result in payload.results {
                let status = result.compileSucceeded ? "OK" : "FAIL"
                print("\(status.padding(toLength: 5, withPad: " ", startingAt: 0)) spatial=\(result.spatial)")
                if !result.diagnostics.isEmpty {
                    print("      diagnostics: \(result.diagnostics.joined(separator: " | "))")
                }
                if let error = result.error {
                    print("      error: \(error)")
                }
            }
        }
        return payload.results.allSatisfy(\.compileSucceeded) ? 0 : 1
    }

    let capability = PowerTelemetryCollector.capability()
    let checks: [DoctorCheck] = [
        DoctorCheck(
            name: "repository",
            severity: defaults.repoRoot == nil ? .warning : .ok,
            detail: defaults.repoRoot?.path ?? "repository root not detected; set ESPRESSO_REPO_ROOT if you need bootstrap helpers"
        ),
        DoctorCheck(
            name: "scripts",
            severity: defaults.scriptsAvailable ? .ok : .error,
            detail: defaults.scriptsDir?.path ?? "helper scripts unavailable"
        ),
        DoctorCheck(
            name: "state_root",
            severity: .ok,
            detail: defaults.stateRoot.path
        ),
        DoctorCheck(
            name: "demo_assets",
            severity: (hasGPT2TokenizerAssets(in: defaults.tokenizerDir) && FileManager().fileExists(atPath: defaults.weightsDir.appendingPathComponent("metadata.json").path)) ? .ok : .warning,
            detail: "weights=\(defaults.weightsDir.path) tokenizer=\(defaults.tokenizerDir.path)"
        ),
        DoctorCheck(
            name: "ane_framework",
            severity: aneFrameworkAvailable() ? .ok : .error,
            detail: aneFrameworkAvailable() ? "AppleNeuralEngine.framework available" : "AppleNeuralEngine.framework unavailable"
        ),
        DoctorCheck(
            name: "python",
            severity: (try? preferredBootstrapPython(defaults: defaults)) != nil ? .ok : .error,
            detail: (try? preferredBootstrapPython(defaults: defaults)) ?? "No suitable python3.13/python3.12/python3 found"
        ),
        DoctorCheck(
            name: "powermetrics",
            severity: capability.available ? .ok : .warning,
            detail: capability.message
        ),
    ]

    if options.jsonOutput {
        let payload = checks.map { ["name": $0.name, "severity": $0.severity.rawValue, "detail": $0.detail] }
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        print(String(decoding: data, as: UTF8.self))
    } else {
        print("Espresso Doctor")
        print("================")
        for check in checks {
            let glyph: String
            switch check.severity {
            case .ok: glyph = "OK"
            case .warning: glyph = "WARN"
            case .error: glyph = "ERROR"
            }
            print("\(glyph.padding(toLength: 5, withPad: " ", startingAt: 0)) \(check.name): \(check.detail)")
        }
    }

    return checks.contains(where: { $0.severity == .error }) ? 1 : 0
}

private func runGPT2AttentionCompileProbe(invocation: ResolvedInvocation) throws -> GPT2AttentionCompileProbePayload {
    let results = try RealModelInferenceEngine.probeGPT2AttentionCompilation(
        config: invocation.config,
        weightDir: invocation.weightsDir,
        spatials: [64, 128, 256]
    )
    let first = results.first
    return GPT2AttentionCompileProbePayload(
        deploymentTarget: first?.deploymentTarget ?? "ios18",
        normKind: first?.normKind ?? "layernorm",
        results: results
    )
}

private func aneFrameworkAvailable() -> Bool {
    let handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW
    )
    guard handle != nil else {
        return false
    }
    dlclose(handle)
    return true
}

private func runLiveCompare(invocation: ResolvedInvocation, defaults: DemoDefaults, powerEnabled: Bool) throws -> CompareReport {
    guard invocation.config.architecture == .gpt2 else {
        throw CLIError.usage("Live compare currently supports GPT-2 only.")
    }

    let tokenizer = try loadTokenizer(config: invocation.config, tokenizerDir: invocation.tokenizerDir)
    let sequenceLength = invocation.coreMLSequenceLength ?? nextPowerOfTwo(
        min(invocation.config.maxSeq, invocation.maxTokens + 8)
    )
    let coreMLModelPath = try ensureGPT2CoreMLModel(
        defaults: defaults,
        weightsDir: invocation.weightsDir,
        sequenceLength: sequenceLength,
        explicitModelPath: invocation.coreMLModelPath,
        allowBootstrap: invocation.allowBootstrap
    )

    let espressoCalibration = try maybeMeasurePower(enabled: powerEnabled) {
        var engine = try makeEspressoEngine(invocation: invocation)
        return try engine.generate(
            prompt: preparedGeneratePrompt(
                invocation.prompt,
                config: invocation.config,
                rawPrompt: invocation.rawPrompt
            ),
            maxTokens: min(invocation.maxTokens, 4),
            temperature: invocation.temperature
        )
    }.power

    let coreMLCalibration = try maybeMeasurePower(enabled: powerEnabled) {
        try runGPT2CoreMLReference(
            defaults: defaults,
            coreMLModelPath: coreMLModelPath,
            weightsDir: invocation.weightsDir,
            promptTokens: try encodePromptTokens(
                invocation.prompt,
                config: invocation.config,
                tokenizerDir: invocation.tokenizerDir,
                rawPrompt: invocation.rawPrompt
            ),
            sequenceLength: sequenceLength,
            maxTokens: min(invocation.maxTokens, 4),
            temperature: invocation.temperature,
            warmup: 0,
            iterations: 1,
            computeUnits: invocation.coreMLComputeUnits,
            seed: invocation.seed,
            allowBootstrap: invocation.allowBootstrap
        )
    }.power

    let initialSnapshot = LiveCompareSnapshot(
        modelName: invocation.config.name,
        prompt: invocation.prompt,
        maxTokens: invocation.maxTokens,
        elapsedMs: 0,
        espresso: LiveLaneSnapshot(title: "Espresso / ANE", maxTokens: invocation.maxTokens),
        coreML: LiveLaneSnapshot(title: "Core ML", maxTokens: invocation.maxTokens),
        livePower: nil,
        matchCount: 0,
        totalComparedTokens: 0,
        events: ["Preparing compare lanes"]
    )
    let store = LiveCompareStateStore(snapshot: initialSnapshot)
    store.mutate {
        $0.espresso.power = espressoCalibration
        $0.coreML.power = coreMLCalibration
    }

    let display = TerminalDisplay()
    display.start()
    let renderer = LiveCompareRenderer()
    defer { display.stop() }

    let powerCapability = PowerTelemetryCollector.capability()
    let powerCollector: PowerTelemetryCollector? = powerEnabled && powerCapability.available ? PowerTelemetryCollector() : nil
    if let powerCollector {
        try? powerCollector.start { sample in
            store.mutate { snapshot in
                snapshot.livePower = PowerSummary(
                    packageW: sample.packageW,
                    cpuW: sample.cpuW,
                    gpuW: sample.gpuW,
                    aneW: sample.aneW,
                    sampleCount: 1
                )
            }
        }
    }
    defer {
        if let powerCollector {
            _ = powerCollector.stop()
        }
    }

    let started = DispatchTime.now().uptimeNanoseconds
    let promptTokens = try encodePromptTokens(
        invocation.prompt,
        config: invocation.config,
        tokenizerDir: invocation.tokenizerDir,
        rawPrompt: invocation.rawPrompt
    )

    let coreMLMetricsBox = LockedValueBox<BackendRunMetrics>()
    let espressoMetricsBox = LockedValueBox<BackendRunMetrics>()
    let coreMLTokensBox = LockedValueBox<[TokenID]>()
    let espressoTokensBox = LockedValueBox<[TokenID]>()
    let errorBox = LockedValueBox<Error>()
    let renderStopBox = LockedValueBox<Bool>()
    renderStopBox.set(false)

    let renderGroup = DispatchGroup()
    renderGroup.enter()
    DispatchQueue.global(qos: .userInitiated).async {
        while renderStopBox.get() != true {
            display.render(renderer.render(snapshot: store.read(), size: .current()))
            Thread.sleep(forTimeInterval: 0.12)
        }
        display.render(renderer.render(snapshot: store.read(), size: .current()))
        renderGroup.leave()
    }
    defer {
        renderStopBox.set(true)
        renderGroup.wait()
    }

    let coreMLGroup = DispatchGroup()
    coreMLGroup.enter()
    DispatchQueue.global(qos: .userInitiated).async {
        do {
            store.mutate { snapshot in
                snapshot.coreML.status = .compiling
                snapshot.events.append("[Core ML] loading baseline")
                trimEvents(&snapshot.events)
            }
            var generated: [TokenID] = []
            let result = try runGPT2CoreMLReferenceStreaming(
                defaults: defaults,
                coreMLModelPath: coreMLModelPath,
                weightsDir: invocation.weightsDir,
                promptTokens: promptTokens,
                sequenceLength: sequenceLength,
                maxTokens: invocation.maxTokens,
                temperature: invocation.temperature,
                computeUnits: invocation.coreMLComputeUnits,
                seed: invocation.seed,
                allowBootstrap: invocation.allowBootstrap
            ) { event in
                store.mutate { snapshot in
                    snapshot.elapsedMs = millisecondsSince(started)
                    switch event {
                    case let .compile(compileTimeMs, _, _):
                        snapshot.coreML.compileMs = compileTimeMs
                        snapshot.coreML.status = .generating
                        snapshot.events.append("[Core ML] compile complete")
                    case let .token(token, tokenIndex, elapsedMs, tokenLatencyMs, tokensPerSecond):
                        generated.append(token)
                        coreMLTokensBox.set(generated)
                        snapshot.coreML.status = .generating
                        snapshot.coreML.generatedTokenCount = tokenIndex
                        snapshot.coreML.lastToken = decodeToken(token, tokenizer: tokenizer)
                        snapshot.coreML.text = tokenizer.decode((promptTokens + generated).map(Int.init))
                        snapshot.coreML.tokensPerSecond = tokensPerSecond
                        snapshot.coreML.totalMs = elapsedMs
                        snapshot.coreML.ttftMs = tokenIndex == 1 ? tokenLatencyMs : snapshot.coreML.ttftMs
                        updateMatchCounts(
                            snapshot: &snapshot,
                            espressoTokens: espressoTokensBox.get(),
                            coreMLTokens: generated
                        )
                        snapshot.events.append("[Core ML] token \(tokenIndex) -> \(snapshot.coreML.lastToken)")
                    case let .completed(final):
                        snapshot.coreML.status = .completed
                        snapshot.coreML.compileMs = final.compileTimeMs
                        snapshot.coreML.generatedTokenCount = final.generatedTokens.count
                        snapshot.coreML.text = tokenizer.decode((promptTokens + generated).map(Int.init))
                        snapshot.coreML.ttftMs = final.firstTokenLatencyMs
                        snapshot.coreML.tokensPerSecond = final.tokensPerSecond
                        snapshot.coreML.medianTokenMs = final.medianTokenMs
                        snapshot.coreML.p95TokenMs = final.p95TokenMs
                        snapshot.coreML.totalMs = final.totalTimeMs
                        updateMatchCounts(
                            snapshot: &snapshot,
                            espressoTokens: espressoTokensBox.get(),
                            coreMLTokens: generated
                        )
                        snapshot.events.append("[Core ML] completed")
                    }
                    trimEvents(&snapshot.events)
                }
            }
            let generatedTokens = try result.generatedTokens.map(validateToken)
            coreMLTokensBox.set(generatedTokens)
            coreMLMetricsBox.set(
                BackendRunMetrics(
                backend: "coreml",
                text: tokenizer.decode((promptTokens + generatedTokens).map(Int.init)),
                generatedTokens: generatedTokens,
                promptTokens: promptTokens,
                compileTimeMs: result.compileTimeMs,
                firstTokenLatencyMs: result.firstTokenLatencyMs,
                tokensPerSecond: result.tokensPerSecond,
                medianTokenMs: result.medianTokenMs,
                p95TokenMs: result.p95TokenMs,
                totalTimeMs: result.totalTimeMs,
                tokenLatenciesMs: result.tokenLatenciesMs
                )
            )
        } catch {
            errorBox.set(error)
            store.mutate { snapshot in
                snapshot.coreML.status = .failed
                snapshot.events.append("[Core ML] failed: \(error)")
                trimEvents(&snapshot.events)
            }
        }
        coreMLGroup.leave()
    }

    let espressoGroup = DispatchGroup()
    espressoGroup.enter()
    DispatchQueue.global(qos: .userInitiated).async {
        do {
            store.mutate { snapshot in
                snapshot.espresso.status = .compiling
                snapshot.events.append("[Espresso] loading ANE pipeline")
                trimEvents(&snapshot.events)
            }
            var liveEngine = try makeEspressoEngine(invocation: invocation)
            let espressoMetrics = try runEspressoGeneration(engine: &liveEngine, invocation: invocation) { step in
                espressoTokensBox.set(step.generatedTokens)
                store.mutate { snapshot in
                    snapshot.elapsedMs = millisecondsSince(started)
                    snapshot.espresso.status = .generating
                    snapshot.espresso.generatedTokenCount = step.generatedTokens.count
                    snapshot.espresso.lastToken = decodeToken(step.token, tokenizer: tokenizer)
                    snapshot.espresso.text = step.text
                    snapshot.espresso.ttftMs = step.firstTokenLatencyMs
                    snapshot.espresso.tokensPerSecond = step.tokensPerSecond
                    snapshot.espresso.totalMs = step.elapsedMs
                    updateMatchCounts(
                        snapshot: &snapshot,
                        espressoTokens: step.generatedTokens,
                        coreMLTokens: coreMLTokensBox.get() ?? []
                    )
                    snapshot.events.append("[Espresso] token \(step.generatedTokens.count) -> \(snapshot.espresso.lastToken)")
                    trimEvents(&snapshot.events)
                }
            }
            espressoMetricsBox.set(espressoMetrics)
            espressoTokensBox.set(espressoMetrics.generatedTokens)
            store.mutate { snapshot in
                snapshot.elapsedMs = millisecondsSince(started)
                snapshot.espresso.status = .completed
                snapshot.espresso.generatedTokenCount = espressoMetrics.generatedTokens.count
                snapshot.espresso.text = espressoMetrics.text
                snapshot.espresso.compileMs = espressoMetrics.compileTimeMs
                snapshot.espresso.ttftMs = espressoMetrics.firstTokenLatencyMs
                snapshot.espresso.tokensPerSecond = espressoMetrics.tokensPerSecond
                snapshot.espresso.medianTokenMs = espressoMetrics.medianTokenMs
                snapshot.espresso.p95TokenMs = espressoMetrics.p95TokenMs
                snapshot.espresso.totalMs = espressoMetrics.totalTimeMs
                updateMatchCounts(
                    snapshot: &snapshot,
                    espressoTokens: espressoMetrics.generatedTokens,
                    coreMLTokens: coreMLTokensBox.get() ?? []
                )
                snapshot.events.append("[Espresso] completed")
                trimEvents(&snapshot.events)
            }
        } catch {
            errorBox.set(error)
            store.mutate { snapshot in
                snapshot.espresso.status = .failed
                snapshot.events.append("[Espresso] failed: \(error)")
                trimEvents(&snapshot.events)
            }
        }
        espressoGroup.leave()
    }

    while coreMLGroup.wait(timeout: .now()) != .success || espressoGroup.wait(timeout: .now()) != .success {
        Thread.sleep(forTimeInterval: 0.05)
    }

    if let taskError = errorBox.get() {
        throw taskError
    }
    guard let coreMLMetrics = coreMLMetricsBox.get() else {
        throw CLIError.runtime("Live compare did not produce both backend results.")
    }
    guard let espressoMetrics = espressoMetricsBox.get() else {
        throw CLIError.runtime("Live compare did not produce both backend results.")
    }

    store.mutate { snapshot in
        updateMatchCounts(snapshot: &snapshot, espressoTokens: espressoMetrics.generatedTokens, coreMLTokens: coreMLMetrics.generatedTokens)
    }

    let report = compareReport(
        invocation: invocation,
        espresso: espressoMetrics,
        coreML: coreMLMetrics,
        coreMLSequenceLength: sequenceLength,
        espressoPower: espressoCalibration,
        coreMLPower: coreMLCalibration,
        outputDirectory: nil
    )
    return report
}

private func decodeToken(_ token: TokenID, tokenizer: CLITokenizer) -> String {
    tokenizer.decode([Int(token)]).replacingOccurrences(of: "\n", with: "\\n")
}

private func updateMatchCounts(snapshot: inout LiveCompareSnapshot, espressoTokens: [TokenID]?, coreMLTokens: [TokenID]) {
    guard let espressoTokens else {
        snapshot.matchCount = 0
        snapshot.totalComparedTokens = 0
        return
    }
    let compared = min(espressoTokens.count, coreMLTokens.count)
    var matched = 0
    for index in 0..<compared where espressoTokens[index] == coreMLTokens[index] {
        matched += 1
    }
    snapshot.matchCount = matched
    snapshot.totalComparedTokens = compared
}

private func trimEvents(_ events: inout [String]) {
    if events.count > 8 {
        events.removeFirst(events.count - 8)
    }
}

private func millisecondsSince(_ started: UInt64) -> Double {
    Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000.0
}

func preparedGeneratePrompt(_ prompt: String, config: MultiModelConfig, rawPrompt: Bool) -> String {
    if !rawPrompt, QwenInstructPrompt.shouldWrap(config: config) {
        return QwenInstructPrompt.wrapUserTurn(prompt)
    }
    return prompt
}

private func encodePromptTokens(
    _ prompt: String,
    config: MultiModelConfig,
    tokenizerDir: String,
    rawPrompt: Bool = false
) throws -> [TokenID] {
    let tokenizer = try loadTokenizer(config: config, tokenizerDir: tokenizerDir)
    return try tokenizer.encode(preparedGeneratePrompt(prompt, config: config, rawPrompt: rawPrompt)).map(validateToken)
}

private func metricsOrZero(_ value: Double, fallback: Double) -> Double {
    value > 0 ? value : fallback
}

private func currentResidentMemoryBytes() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info_data_t>.size / MemoryLayout<natural_t>.size)
    let status: kern_return_t = withUnsafeMutablePointer(to: &info) { pointer in
        pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), rebound, &count)
        }
    }
    guard status == KERN_SUCCESS else {
        return 0
    }
    return UInt64(info.resident_size)
}

private func compileCacheHitRate(for backend: BackendRunMetrics) -> Double {
    backend.compileTimeMs > 0 ? 0.0 : 1.0
}

func benchmarkFingerprint(for backend: BackendRunMetrics) -> ESPBenchmarkFingerprint {
    ESPBenchmarkFingerprint(metrics: [
        .ttftMilliseconds: backend.firstTokenLatencyMs,
        .tokensPerSecond: backend.tokensPerSecond,
        .coldLoadMilliseconds: backend.compileTimeMs + backend.firstTokenLatencyMs,
        .warmLoadMilliseconds: backend.firstTokenLatencyMs,
        .compileCacheHitRate: compileCacheHitRate(for: backend),
        .peakResidentMemoryBytes: Double(currentResidentMemoryBytes()),
    ])
}

private func benchmarkFingerprintPayload(_ fingerprint: ESPBenchmarkFingerprint) -> [String: Double] {
    var payload: [String: Double] = [:]
    for metric in ESPBenchmarkMetric.allCases {
        if let value = fingerprint.metrics[metric] {
            payload[metric.rawValue] = value
        }
    }
    return payload
}

private func runGenerate(invocation: ResolvedInvocation) throws -> BackendRunMetrics {
    if invocation.benchmarkGenerate {
        return try runEspressoBenchmark(invocation: invocation)
    }
    if let bundlePath = invocation.bundlePath {
        let bundle = try ESPRuntimeBundle.open(at: URL(fileURLWithPath: bundlePath, isDirectory: true))
        return try makeBundleRuntimeMetrics(bundle: bundle, invocation: invocation)
    }
    var engine = try makeEspressoEngine(invocation: invocation)
    return try runEspressoGeneration(engine: &engine, invocation: invocation)
}

private func runCompareOrBench(
    invocation: ResolvedInvocation,
    defaults: DemoDefaults,
    powerEnabled: Bool
) throws -> CompareReport {
    let espressoMeasured = try maybeMeasurePower(enabled: powerEnabled) {
        try runEspressoBenchmark(invocation: invocation)
    }
    let coreMLMeasured = try maybeMeasurePower(enabled: powerEnabled) {
        try runCoreMLGeneration(invocation: invocation, defaults: defaults, promptTokens: espressoMeasured.result.promptTokens)
    }
    return compareReport(
        invocation: invocation,
        espresso: espressoMeasured.result,
        coreML: coreMLMeasured.result.metrics,
        coreMLSequenceLength: coreMLMeasured.result.sequenceLength,
        espressoPower: espressoMeasured.power,
        coreMLPower: coreMLMeasured.power,
        outputDirectory: nil
    )
}

private func finalizedCompareReport(_ report: CompareReport, outputDirectory: String) -> CompareReport {
    CompareReport(
        model: report.model,
        prompt: report.prompt,
        maxTokens: report.maxTokens,
        seed: report.seed,
        espresso: report.espresso,
        coreML: report.coreML,
        tokenMatch: report.tokenMatch,
        textMatch: report.textMatch,
        coreMLComputeUnits: report.coreMLComputeUnits,
        coreMLSequenceLength: report.coreMLSequenceLength,
        espressoPower: report.espressoPower,
        coreMLPower: report.coreMLPower,
        outputDirectory: outputDirectory
    )
}

private func coreMLMetrics(
    from result: CoreMLComparisonResult,
    tokenizer: CLITokenizer,
    promptTokens: [TokenID]
) throws -> BackendRunMetrics {
    let generatedTokens = try result.generatedTokens.map(validateToken)
    return BackendRunMetrics(
        backend: "coreml",
        text: tokenizer.decode((promptTokens + generatedTokens).map(Int.init)),
        generatedTokens: generatedTokens,
        promptTokens: promptTokens,
        compileTimeMs: result.compileTimeMs,
        firstTokenLatencyMs: result.firstTokenLatencyMs,
        tokensPerSecond: result.tokensPerSecond,
        medianTokenMs: result.medianTokenMs,
        p95TokenMs: result.p95TokenMs,
        totalTimeMs: result.totalTimeMs,
        tokenLatenciesMs: result.tokenLatenciesMs
    )
}

private func suiteDirectoryTimestamp() -> String {
    let formatter = DateFormatter()
    formatter.calendar = Calendar(identifier: .gregorian)
    formatter.locale = posixLocale
    formatter.timeZone = TimeZone(secondsFromGMT: 0)
    formatter.dateFormat = "yyyyMMdd-HHmmss"
    return formatter.string(from: Date())
}

private func suiteSummaryTimestamp() -> String {
    ISO8601DateFormatter().string(from: Date())
}

private func gitShortCommit(repoRoot: URL?) -> String {
    guard let repoRoot else {
        return "unknown"
    }

    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
    process.arguments = ["-C", repoRoot.path, "rev-parse", "--short", "HEAD"]
    let stdout = Pipe()
    process.standardOutput = stdout
    process.standardError = Pipe()

    do {
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            return "unknown"
        }
        return String(decoding: stdout.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    } catch {
        return "unknown"
    }
}

private func resolvePromptSuiteOutputDirectory(defaults: DemoDefaults, requestedOutputDir: String?) -> URL {
    if let requestedOutputDir, !requestedOutputDir.isEmpty {
        return URL(fileURLWithPath: NSString(string: requestedOutputDir).expandingTildeInPath, isDirectory: true)
            .standardizedFileURL
    }

    let baseDirectory: URL
    if let repoRoot = defaults.repoRoot {
        baseDirectory = repoRoot.appendingPathComponent("results/autoresearch", isDirectory: true)
    } else {
        baseDirectory = defaults.reportsRoot
    }
    return baseDirectory.appendingPathComponent("suite-\(suiteDirectoryTimestamp())", isDirectory: true)
}

func resolveCoreMLModelPath(
    invocation: ResolvedInvocation,
    defaults: DemoDefaults,
    sequenceLength: Int
) throws -> String {
    if let explicitModelPath = invocation.coreMLModelPath, !explicitModelPath.isEmpty {
        let expanded = NSString(string: explicitModelPath).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: expanded) else {
            throw CLIError.usage("Core ML model path does not exist: \(explicitModelPath)")
        }
        return URL(fileURLWithPath: expanded).standardizedFileURL.path
    }

    switch invocation.config.architecture {
    case .gpt2:
        return try ensureGPT2CoreMLModel(
            defaults: defaults,
            weightsDir: invocation.weightsDir,
            sequenceLength: sequenceLength,
            explicitModelPath: nil,
            allowBootstrap: invocation.allowBootstrap
        )
    case .llama:
        throw CLIError.usage(
            "Model \(invocation.config.name) requires --coreml-model PATH to an exported Core ML package."
        )
    }
}

private func prepareResultsTSV(at path: String?) throws -> URL? {
    guard let path, !path.isEmpty else {
        return nil
    }

    let url = URL(fileURLWithPath: NSString(string: path).expandingTildeInPath).standardizedFileURL
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    if !FileManager.default.fileExists(atPath: url.path) {
        try (promptSuiteResultsTSVHeader() + "\n").write(to: url, atomically: true, encoding: .utf8)
    }
    return url
}

private func appendLine(_ line: String, to url: URL) throws {
    let existing = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
    let separator = existing.isEmpty || existing.hasSuffix("\n") ? "" : "\n"
    try (existing + separator + line + "\n").write(to: url, atomically: true, encoding: .utf8)
}

private func runPromptSuiteCompare(
    invocation: ResolvedInvocation,
    tokenizer: CLITokenizer,
    promptTokens: [TokenID],
    espressoRun: () throws -> BackendRunMetrics,
    coreMLRunner: inout ReusableCoreMLBenchmarkRunner,
    sequenceLength: Int,
    powerEnabled: Bool
) throws -> CompareReport {
    let espressoMeasured = try maybeMeasurePower(enabled: powerEnabled) {
        try aggregateBenchmarkRuns(warmup: invocation.compareWarmup, iterations: invocation.compareIterations) {
            try espressoRun()
        }
    }
    let coreMLMeasured = try maybeMeasurePower(enabled: powerEnabled) {
        let result = try coreMLRunner.benchmark(
            promptTokens: promptTokens,
            maxTokens: invocation.maxTokens,
            temperature: invocation.temperature,
            warmup: invocation.compareWarmup,
            iterations: invocation.compareIterations,
            seed: invocation.seed
        )
        return try coreMLMetrics(from: result, tokenizer: tokenizer, promptTokens: promptTokens)
    }
    return compareReport(
        invocation: invocation,
        espresso: espressoMeasured.result,
        coreML: coreMLMeasured.result,
        coreMLSequenceLength: sequenceLength,
        espressoPower: espressoMeasured.power,
        coreMLPower: coreMLMeasured.power,
        outputDirectory: nil
    )
}

private func runPromptSuite(
    invocation: ResolvedInvocation,
    options: Options,
    defaults: DemoDefaults,
    powerEnabled: Bool
) throws -> Int32 {
    guard let promptsFile = options.promptsFile, !promptsFile.isEmpty else {
        throw CLIError.usage("Suite mode requires --prompts FILE.")
    }

    let prompts = try loadPromptSuite(from: promptsFile)
    let tokenizer = try loadTokenizer(config: invocation.config, tokenizerDir: invocation.tokenizerDir)
    let promptTokensByID = try Dictionary(uniqueKeysWithValues: prompts.map { prompt in
        (prompt.id, try tokenizer.encode(prompt.text).map(validateToken))
    })
    let promptTokenCounts = prompts.compactMap { promptTokensByID[$0.id]?.count }
    let sequenceLength = try resolvedSuiteCoreMLSequenceLength(
        explicitSequenceLength: invocation.coreMLSequenceLength,
        promptTokenCounts: promptTokenCounts,
        maxTokens: invocation.maxTokens,
        maxModelSequenceLength: invocation.config.maxSeq
    )
    let coreMLModelPath = try resolveCoreMLModelPath(
        invocation: invocation,
        defaults: defaults,
        sequenceLength: sequenceLength
    )

    let outputRoot = resolvePromptSuiteOutputDirectory(defaults: defaults, requestedOutputDir: invocation.outputDir)
    try FileManager.default.createDirectory(at: outputRoot, withIntermediateDirectories: true)

    let commit = gitShortCommit(repoRoot: defaults.repoRoot)
    let timestamp = suiteSummaryTimestamp()
    let config = PromptSuiteConfig(
        runs: options.suiteRuns,
        warmup: invocation.compareWarmup,
        iterations: invocation.compareIterations,
        maxTokens: invocation.maxTokens
    )
    let metadata = PromptSuiteMetadata(
        commit: commit,
        timestamp: timestamp,
        promptsFile: NSString(string: promptsFile).expandingTildeInPath,
        promptIDs: prompts.map(\.id),
        runs: options.suiteRuns,
        warmup: invocation.compareWarmup,
        iterations: invocation.compareIterations,
        maxTokens: invocation.maxTokens,
        coldRun: options.includeColdRun
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(metadata).write(to: outputRoot.appendingPathComponent("metadata.json"))

    let resultsTSVURL = try prepareResultsTSV(at: options.resultsTSV)
    let bundle = try invocation.bundlePath.map {
        try ESPRuntimeBundle.open(at: URL(fileURLWithPath: $0, isDirectory: true))
    }
    let espressoRun: (ResolvedInvocation) throws -> BackendRunMetrics
    if let bundle {
        espressoRun = { promptInvocation in
            try makeBundleRuntimeMetrics(bundle: bundle, invocation: promptInvocation)
        }
    } else {
        var engine = try makeEspressoEngine(invocation: invocation)
        espressoRun = { promptInvocation in
            try runEspressoGeneration(engine: &engine, invocation: promptInvocation)
        }
    }
    var coreMLRunner = try ReusableCoreMLBenchmarkRunner(
        modelPath: coreMLModelPath,
        weightsDir: invocation.weightsDir,
        architecture: invocation.config.architecture,
        sequenceLength: sequenceLength,
        computeUnits: invocation.coreMLComputeUnits
    )

    if options.includeColdRun, let coldPrompt = prompts.first, let coldTokens = promptTokensByID[coldPrompt.id] {
        let coldInvocation = invocation.with(prompt: coldPrompt.text, compareWarmup: 0, compareIterations: 1)
        let coldReport = try runPromptSuiteCompare(
            invocation: coldInvocation,
            tokenizer: tokenizer,
            promptTokens: coldTokens,
            espressoRun: { try espressoRun(coldInvocation) },
            coreMLRunner: &coreMLRunner,
            sequenceLength: sequenceLength,
            powerEnabled: powerEnabled
        )
        _ = try writeCompareArtifacts(
            report: coldReport,
            defaults: defaults,
            requestedOutputDir: outputRoot.appendingPathComponent("cold/\(coldPrompt.id)", isDirectory: true).path
        )
    }

    var records: [PromptSuiteRunRecord] = []
    records.reserveCapacity(prompts.count * options.suiteRuns)
    for runIndex in 1...options.suiteRuns {
        for prompt in prompts {
            guard let promptTokens = promptTokensByID[prompt.id] else {
                throw CLIError.runtime("Missing prompt tokens for suite entry \(prompt.id)")
            }
            let promptInvocation = invocation.with(prompt: prompt.text)
            let report = try runPromptSuiteCompare(
                invocation: promptInvocation,
                tokenizer: tokenizer,
                promptTokens: promptTokens,
                espressoRun: { try espressoRun(promptInvocation) },
                coreMLRunner: &coreMLRunner,
                sequenceLength: sequenceLength,
                powerEnabled: powerEnabled
            )
            let requestedOutputDir = outputRoot
                .appendingPathComponent("run-\(runIndex)", isDirectory: true)
                .appendingPathComponent(prompt.id, isDirectory: true)
                .path
            let outputDirectory = try writeCompareArtifacts(
                report: report,
                defaults: defaults,
                requestedOutputDir: requestedOutputDir
            )
            let finalized = finalizedCompareReport(report, outputDirectory: outputDirectory)
            records.append(PromptSuiteRunRecord(promptID: prompt.id, report: finalized))
            if let resultsTSVURL {
                try appendLine(
                    promptSuiteResultsTSVRow(
                        timestamp: timestamp,
                        commit: commit,
                        promptID: prompt.id,
                        outputDirectory: outputDirectory,
                        report: finalized
                    ),
                    to: resultsTSVURL
                )
            }
        }
    }

    let summary = makePromptSuiteSummary(
        promptOrder: prompts,
        reports: records,
        commit: commit,
        timestamp: timestamp,
        config: config
    )
    let summaryURL = outputRoot.appendingPathComponent("suite-summary.json")
    try encoder.encode(summary).write(to: summaryURL)

    if options.jsonOutput {
        print(String(decoding: try Data(contentsOf: summaryURL), as: UTF8.self))
    } else {
        for line in promptSuiteSummaryLines(summary) {
            print(line)
        }
        stderrLine("suite_dir=\(outputRoot.path)")
    }

    return summary.verdict.allCorrectnessGatesPass ? 0 : 1
}

private func readChatLine() -> String? {
    var buffer = [CChar](repeating: 0, count: 4096)
    while true {
        errno = 0
        guard fgets(&buffer, Int32(buffer.count), stdin) != nil else {
            if feof(stdin) != 0 {
                return nil
            }
            if errno == EINTR {
                continue
            }
            return nil
        }
        let end = buffer.firstIndex(of: 0) ?? buffer.endIndex
        let bytes = buffer[..<end].map { UInt8(bitPattern: $0) }
        return String(decoding: bytes, as: UTF8.self).trimmingCharacters(in: CharacterSet.newlines)
    }
}

private func installChatSIGINT(_ cancel: ChatCancelState) -> DispatchSourceSignal {
    signal(SIGINT, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: SIGINT, queue: .global())
    source.setEventHandler {
        _ = cancel.requestCancel(userInterrupt: true)
    }
    source.resume()
    return source
}

private func runChatVsMLX(
    invocation: ResolvedInvocation,
    options: Options,
    defaults: DemoDefaults,
    powerEnabled: Bool
) throws -> Int32 {
    applyCLIExperimentEnvironment(options)
    var fairness = try makeChatVsMLXFairness(
        espressoModelName: invocation.config.name,
        greedy: options.greedy,
        maxNewTokens: invocation.maxTokens,
        mlxQuantFlag: options.mlxQuant,
        mlxModelOverride: options.mlxModel
    )
    let python = try requireMLXPython(defaults: defaults)
    let script = try mlxStreamScriptURL(defaults: defaults)
    let snapshot = huggingFaceHubSnapshot(
        repo: fairness.huggingfaceRepo,
        cacheRoot: defaultHuggingFaceCacheRoot()
    )
    let modelPath = options.mlxModel?.trimmingCharacters(in: .whitespacesAndNewlines)
    let resolvedModelPath: String
    if let modelPath, !modelPath.isEmpty {
        resolvedModelPath = modelPath
    } else if let snapshot {
        resolvedModelPath = snapshot.path
    } else {
        resolvedModelPath = fairness.huggingfaceRepo
    }

    var environment = ProcessInfo.processInfo.environment
    environment["PYTHONUNBUFFERED"] = "1"
    if snapshot != nil, options.mlxModel == nil {
        environment["HF_HUB_OFFLINE"] = "1"
    }

    let mlx = try MLXLaneSession.start(
        python: python,
        script: script,
        fairness: fairness,
        modelPath: resolvedModelPath,
        workingDirectory: defaults.workingDirectory,
        environment: environment
    )
    defer { mlx.close() }
    fairness = fairness.applyingLoadedPrecision(mlx.loadedPrecision)

    var engine = try makeEspressoEngine(invocation: invocation)
    let compileStarted = DispatchTime.now().uptimeNanoseconds
    try engine.precompileHybridDecode(covering: invocation.config.maxSeq)
    let espressoCompileMs = millisecondsSince(compileStarted)
    let tokenizer = try loadTokenizer(config: invocation.config, tokenizerDir: invocation.tokenizerDir)
    var session = ChatSession(system: options.systemPrompt ?? QwenInstructPrompt.systemBanner)
    let sampling = resolvedSampling(command: .chat, options: options)
    let useTUI = shouldUseTUI(command: .chat, options: options)
    let cancel = ChatCancelState()
    let sigint = installChatSIGINT(cancel)
    defer { sigint.cancel() }
    let powerCapability = PowerTelemetryCollector.capability()

    let display = TerminalDisplay()
    let renderer = LiveCompareRenderer()
    if useTUI {
        display.start()
    }
    defer {
        if useTUI {
            display.stop()
        }
    }

    var espressoLane = LiveLaneSnapshot(title: fairness.espressoLaneHeader(), maxTokens: invocation.maxTokens)
    espressoLane.wattFocus = .ane
    espressoLane.compileMs = espressoCompileMs
    var mlxLane = LiveLaneSnapshot(title: fairness.mlxLaneHeader(), maxTokens: invocation.maxTokens)
    mlxLane.wattFocus = .gpu
    let store = LiveCompareStateStore(
        snapshot: LiveCompareSnapshot(
            modelName: invocation.config.name,
            prompt: "",
            maxTokens: invocation.maxTokens,
            elapsedMs: 0,
            espresso: espressoLane,
            coreML: mlxLane,
            livePower: nil,
            matchCount: 0,
            totalComparedTokens: 0,
            events: [
                "fairness \(fairness.huggingfaceRepo)  espresso=\(fairness.espressoPrecision)  mlx=\(fairness.mlxPrecisionLabel)  greedy  compile excluded from tok/s"
            ],
            display: .mlx(fairness)
        )
    )

    var espressoTurns: [ChatVsMLXTurnMetrics] = []
    var mlxTurns: [ChatVsMLXTurnMetrics] = []

    func writePlain(_ text: String, terminator: String = "\n") {
        guard !useTUI else { return }
        FileHandle.standardOutput.write(Data((text + terminator).utf8))
        fflush(stdout)
    }

    func renderFrame() {
        guard useTUI else { return }
        display.render(renderer.render(snapshot: store.read(), size: TerminalSize.current()))
    }

    if !useTUI {
        writePlain(
            "espresso chat --vs mlx  model=\(invocation.config.name)  repo=\(fairness.huggingfaceRepo)  precision=\(fairness.espressoPrecision) vs \(fairness.mlxPrecisionLabel)  greedy  fallback=disabled"
        )
        writePlain("commands: /reset /retry /exit   Ctrl-C cancels the current lane")
    } else {
        renderFrame()
    }

    while true {
        if useTUI {
            renderFrame()
            FileHandle.standardOutput.write(Data("you> ".utf8))
            fflush(stdout)
        } else {
            writePlain("you> ", terminator: "")
        }
        guard let line = readChatLine() else {
            break
        }
        if !useTUI, isatty(STDIN_FILENO) == 0 {
            writePlain(line)
        }
        let action = session.apply(ChatCommand.parse(line))
        switch action {
        case .exit:
            printScoreboardAndReturn()
            return 0
        case let .noop(message):
            if !message.isEmpty {
                writePlain(message)
            }
            renderFrame()
        case let .generate(prompt):
            let userText = session.messages.last(where: { $0.role == .user })?.content ?? prompt
            store.mutate { snapshot in
                snapshot.prompt = userText
                snapshot.espresso.status = .waiting
                snapshot.espresso.text = ""
                snapshot.espresso.lastToken = "—"
                snapshot.espresso.generatedTokenCount = 0
                snapshot.espresso.tokensPerSecond = 0
                snapshot.espresso.ttftMs = 0
                snapshot.espresso.power = nil
                snapshot.coreML.status = .waiting
                snapshot.coreML.text = ""
                snapshot.coreML.lastToken = "—"
                snapshot.coreML.generatedTokenCount = 0
                snapshot.coreML.tokensPerSecond = 0
                snapshot.coreML.ttftMs = 0
                snapshot.coreML.power = nil
                snapshot.events.append("[shared] \(userText)")
                trimEvents(&snapshot.events)
            }
            renderFrame()

            cancel.beginGeneration()
            defer { cancel.endGeneration() }

            do {
                let espressoMetrics = try runChatVsMLXEspressoLane(
                    engine: &engine,
                    prompt: prompt,
                    invocation: invocation,
                    sampling: sampling,
                    tokenizer: tokenizer,
                    compileMs: espressoCompileMs,
                    powerEnabled: powerEnabled,
                    powerCapability: powerCapability,
                    cancel: cancel,
                    store: store,
                    useTUI: useTUI,
                    writePlain: writePlain,
                    renderFrame: renderFrame
                )
                session.appendAssistant(store.read().espresso.text)

                let mlxMetrics = try runChatVsMLXMLXLane(
                    mlx: mlx,
                    prompt: prompt,
                    maxTokens: invocation.maxTokens,
                    powerEnabled: powerEnabled,
                    powerCapability: powerCapability,
                    cancel: cancel,
                    store: store,
                    useTUI: useTUI,
                    writePlain: writePlain,
                    renderFrame: renderFrame
                )
                espressoTurns.append(espressoMetrics)
                mlxTurns.append(mlxMetrics)
            } catch let error as RealModelInferenceError {
                if case .cancelled = error {
                    writePlain("^C cancelled")
                    store.mutate { snapshot in
                        snapshot.events.append("[cancel] current lane stopped")
                        trimEvents(&snapshot.events)
                    }
                    renderFrame()
                    continue
                }
                throw error
            } catch let error as CLIError {
                if case let .runtime(message) = error, message.contains("cancelled") {
                    writePlain("^C cancelled")
                    continue
                }
                throw error
            }
        }
    }

    printScoreboardAndReturn()
    return 0

    func printScoreboardAndReturn() {
        guard !espressoTurns.isEmpty || !mlxTurns.isEmpty else { return }
        let paired = pairedChatVsMLXMetrics(espresso: espressoTurns, mlx: mlxTurns)
        let espresso = paired.espresso
        let mlx = paired.mlx
        let table = formatChatVsMLXScoreboard(espresso: espresso, mlx: mlx)
        if useTUI {
            display.stop()
        }
        print(table)
    }
}

private func runChatVsMLXEspressoLane(
    engine: inout RealModelInferenceEngine,
    prompt: String,
    invocation: ResolvedInvocation,
    sampling: ChatSampling,
    tokenizer: CLITokenizer,
    compileMs: Double,
    powerEnabled: Bool,
    powerCapability: PowerCapability,
    cancel: ChatCancelState,
    store: LiveCompareStateStore,
    useTUI: Bool,
    writePlain: @escaping (String, String) -> Void,
    renderFrame: @escaping () -> Void
) throws -> ChatVsMLXTurnMetrics {
    store.mutate { snapshot in
        snapshot.espresso.status = .compiling
        snapshot.espresso.compileMs = compileMs
        snapshot.events.append("[Espresso] generating")
        trimEvents(&snapshot.events)
    }
    renderFrame()
    if !useTUI {
        writePlain("espresso> ", "")
    }

    let measured = try maybeMeasurePower(enabled: powerEnabled, sampleIntervalMs: 200) {
        try engine.generate(
            prompt: prompt,
            maxTokens: invocation.maxTokens,
            temperature: sampling.temperature,
            topP: sampling.topP,
            onStep: { step in
                let text = ChatSession.sanitizeAssistantText(
                    assistantTextFromGeneratedTokens(
                        step.generatedTokens,
                        decode: tokenizer.decodeImpl,
                        eosToken: invocation.config.eosToken
                    )
                )
                store.mutate { snapshot in
                    snapshot.espresso.status = .generating
                    snapshot.espresso.generatedTokenCount = step.generatedTokens.count
                    snapshot.espresso.lastToken = tokenizer.decode([Int(step.token)])
                    snapshot.espresso.text = text
                    snapshot.espresso.ttftMs = step.firstTokenLatencyMs
                    snapshot.espresso.tokensPerSecond = step.tokensPerSecond
                    snapshot.espresso.totalMs = step.elapsedMs
                    snapshot.espresso.compileMs = compileMs
                }
                if useTUI {
                    renderFrame()
                } else {
                    writePlain(tokenizer.decode([Int(step.token)]), "")
                }
            },
            isCancelled: { cancel.isCancelled }
        )
    }
    try assertChatDecodePathIsHybrid(measured.result.decodePath)
    if !useTUI {
        for line in espressoCompareLaneContractLines(
            decodePath: measured.result.decodePath,
            hopsPerToken: measured.result.hopsPerToken,
            decodeProfileReport: measured.result.decodeProfileReport
        ) {
            writePlain(line, "\n")
        }
    }
    store.mutate { snapshot in
        snapshot.espresso.status = .completed
        snapshot.espresso.tokensPerSecond = measured.result.tokensPerSecond
        snapshot.espresso.ttftMs = measured.result.firstTokenLatencyMs
        snapshot.espresso.compileMs = measured.result.compileTimeMs > 0 ? measured.result.compileTimeMs : compileMs
        snapshot.espresso.power = measured.power
        snapshot.espresso.text = ChatSession.sanitizeAssistantText(
            assistantTextFromGeneratedTokens(
                measured.result.tokens,
                decode: tokenizer.decodeImpl,
                eosToken: invocation.config.eosToken
            )
        )
        snapshot.events.append("[Espresso] completed")
        trimEvents(&snapshot.events)
    }
    renderFrame()
    if !useTUI {
        writePlain("", "\n")
        writePlain(plainLaneFooter(lane: store.read().espresso, focus: .ane, capability: powerCapability), "\n")
    }
    return metricsFromLane(store.read().espresso)
}

private func runChatVsMLXMLXLane(
    mlx: MLXLaneSession,
    prompt: String,
    maxTokens: Int,
    powerEnabled: Bool,
    powerCapability: PowerCapability,
    cancel: ChatCancelState,
    store: LiveCompareStateStore,
    useTUI: Bool,
    writePlain: @escaping (String, String) -> Void,
    renderFrame: @escaping () -> Void
) throws -> ChatVsMLXTurnMetrics {
    store.mutate { snapshot in
        snapshot.coreML.status = .compiling
        snapshot.events.append("[MLX] generating")
        trimEvents(&snapshot.events)
    }
    renderFrame()
    if !useTUI {
        writePlain("mlx> ", "")
    }

    var streamed = ""
    let measured = try maybeMeasurePower(enabled: powerEnabled, sampleIntervalMs: 200) {
        try mlx.generate(
            prompt: prompt,
            maxTokens: maxTokens,
            isCancelled: { cancel.isCancelled },
            onEvent: { event in
                store.mutate { snapshot in
                    switch event {
                    case let .compile(ms):
                        snapshot.coreML.compileMs = ms
                        snapshot.coreML.status = .generating
                    case let .token(text, tokenIndex, elapsedMs, tokenLatencyMs, tokensPerSecond):
                        streamed += text
                        snapshot.coreML.status = .generating
                        snapshot.coreML.generatedTokenCount = tokenIndex
                        snapshot.coreML.lastToken = text
                        snapshot.coreML.text = ChatSession.sanitizeAssistantText(streamed)
                        snapshot.coreML.tokensPerSecond = tokensPerSecond
                        snapshot.coreML.totalMs = elapsedMs
                        if tokenIndex == 1 {
                            snapshot.coreML.ttftMs = tokenLatencyMs
                        }
                    case let .completed(text, compileMs, ttftMs, tokensPerSecond, tokenCount):
                        streamed = text
                        snapshot.coreML.status = .completed
                        snapshot.coreML.text = ChatSession.sanitizeAssistantText(text)
                        snapshot.coreML.compileMs = compileMs
                        snapshot.coreML.ttftMs = ttftMs
                        snapshot.coreML.tokensPerSecond = tokensPerSecond
                        snapshot.coreML.generatedTokenCount = tokenCount
                    default:
                        break
                    }
                }
                if useTUI {
                    renderFrame()
                } else if case let .token(text, _, _, _, _) = event {
                    writePlain(text, "")
                }
            }
        )
    }
    store.mutate { snapshot in
        snapshot.coreML.status = .completed
        snapshot.coreML.text = ChatSession.sanitizeAssistantText(measured.result.text)
        snapshot.coreML.compileMs = measured.result.compileMs
        snapshot.coreML.ttftMs = measured.result.ttftMs
        snapshot.coreML.tokensPerSecond = measured.result.tokensPerSecond
        snapshot.coreML.generatedTokenCount = measured.result.tokenCount
        snapshot.coreML.power = measured.power
        snapshot.events.append("[MLX] completed")
        trimEvents(&snapshot.events)
    }
    renderFrame()
    if !useTUI {
        writePlain("", "\n")
        writePlain(plainLaneFooter(lane: store.read().coreML, focus: .gpu, capability: powerCapability), "\n")
    }
    return metricsFromLane(store.read().coreML)
}

private func plainLaneFooter(lane: LiveLaneSnapshot, focus: LaneWattFocus, capability: PowerCapability) -> String {
    let tok = String(format: "%.1f", lane.tokensPerSecond)
    let ttft = String(format: "%.0f", lane.ttftMs)
    let compile = String(format: "%.0f", lane.compileMs)
    let metrics = "tok/s \(tok)  TTFT \(ttft)ms  compile \(compile)ms"
    let power = chatPowerFooter(
        capability: capability,
        summary: lane.power,
        tokensPerSecond: lane.tokensPerSecond
    )
    switch power {
    case let .unavailable(message):
        return metrics + "  " + message
    case .pending:
        return metrics + "  power: sampling"
    case let .measured(packageW, cpuW, gpuW, aneW, joulesPerToken):
        let watts: String
        switch focus {
        case .ane:
            watts = String(format: "ANE %.2fW  CPU %.2fW  pkg %.2fW", aneW, cpuW, packageW)
        case .gpu:
            watts = String(format: "GPU %.2fW  CPU %.2fW  pkg %.2fW", gpuW, cpuW, packageW)
        }
        if let joulesPerToken {
            return metrics + "  " + watts + String(format: "  %.3f J/tok", joulesPerToken)
        }
        return metrics + "  " + watts + "  J/tok unavailable"
    }
}

private func runChat(invocation: ResolvedInvocation, options: Options, powerEnabled: Bool) throws -> Int32 {
    applyCLIExperimentEnvironment(options)
    var engine = try makeEspressoEngine(invocation: invocation)
    try engine.precompileHybridDecode(covering: invocation.config.maxSeq)
    let tokenizer = try loadTokenizer(config: invocation.config, tokenizerDir: invocation.tokenizerDir)
    var session = ChatSession(system: options.systemPrompt ?? QwenInstructPrompt.systemBanner)
    let sampling = resolvedSampling(command: .chat, options: options)
    let useTUI = shouldUseTUI(command: .chat, options: options)
    let cancel = ChatCancelState()
    let sigint = installChatSIGINT(cancel)
    defer { sigint.cancel() }
    let powerCapability = PowerTelemetryCollector.capability()

    let display = TerminalDisplay()
    let renderer = ChatTUIRenderer()
    if useTUI {
        display.start()
    }

    var footer = ChatStatusFooter(
        tokensPerSecond: 0,
        ttftMs: 0,
        decodePath: "hybrid",
        contextUsed: 0,
        contextMax: invocation.config.maxSeq,
        power: chatPowerFooter(capability: powerCapability, summary: nil, tokensPerSecond: 0)
    )
    var streamingAssistant = ""

    func currentSnapshot(status: ChatLaneStatus) -> ChatSnapshot {
        ChatSnapshot(
            modelName: invocation.config.name,
            turns: session.messages,
            streamingAssistant: streamingAssistant,
            status: status,
            footer: footer
        )
    }

    func renderTUI(status: ChatLaneStatus) {
        guard useTUI else { return }
        display.render(renderer.render(snapshot: currentSnapshot(status: status), size: TerminalSize.current()))
    }

    func writePlain(_ text: String, terminator: String = "\n") {
        guard !useTUI else { return }
        FileHandle.standardOutput.write(Data((text + terminator).utf8))
        fflush(stdout)
    }

    if !useTUI {
        writePlain(
            "espresso chat  model=\(invocation.config.name)  sampling=\(sampling.isGreedy ? "greedy" : "t=\(sampling.temperature) top-p=\(sampling.topP)")  fallback=disabled"
        )
        writePlain("commands: /reset /retry /exit   Ctrl-C cancels the current completion")
    } else {
        renderTUI(status: .idle)
    }

    while true {
        if useTUI {
            renderTUI(status: .idle)
            FileHandle.standardOutput.write(Data("you> ".utf8))
            fflush(stdout)
        } else {
            writePlain("you> ", terminator: "")
        }

        guard let line = readChatLine() else {
            break
        }
        if !useTUI {
            if isatty(STDIN_FILENO) == 0 {
                writePlain(line)
            }
        }
        let action = session.apply(ChatCommand.parse(line))
        switch action {
        case .exit:
            if useTUI {
                display.stop()
            }
            return 0
        case let .noop(message):
            if !message.isEmpty {
                writePlain(message)
            }
            renderTUI(status: .idle)
        case let .generate(prompt):
            streamingAssistant = ""
            cancel.beginGeneration()
            defer { cancel.endGeneration() }
            footer.power = powerEnabled && powerCapability.available
                ? .pending
                : chatPowerFooter(capability: powerCapability, summary: nil, tokensPerSecond: 0)
            if !useTUI {
                writePlain("qwen> ", terminator: "")
            } else {
                renderTUI(status: .generating)
            }
            do {
                let promptTokenCount = (try? tokenizer.encode(prompt).count) ?? 0
                footer.contextUsed = promptTokenCount
                footer.contextMax = invocation.config.maxSeq
                let measured = try maybeMeasurePower(enabled: powerEnabled, sampleIntervalMs: 200) {
                    try engine.generate(
                        prompt: prompt,
                        maxTokens: invocation.maxTokens,
                        temperature: sampling.temperature,
                        topP: sampling.topP,
                        onStep: { step in
                            let rawAssistant = assistantTextFromGeneratedTokens(
                                step.generatedTokens,
                                decode: tokenizer.decodeImpl,
                                eosToken: invocation.config.eosToken
                            )
                            streamingAssistant = ChatSession.sanitizeAssistantText(rawAssistant)
                            if rawAssistant.contains("<|im_end|>") || rawAssistant.contains("<|endoftext|>") {
                                cancel.requestCancel()
                            }
                            footer.tokensPerSecond = step.tokensPerSecond
                            footer.ttftMs = step.firstTokenLatencyMs
                            footer.contextUsed = promptTokenCount + step.generatedTokens.count
                            if useTUI {
                                renderTUI(status: .generating)
                            } else {
                                writePlain(tokenizer.decode([Int(step.token)]), terminator: "")
                            }
                        },
                        isCancelled: { cancel.isCancelled }
                    )
                }
                let result = measured.result
                try assertChatDecodePathIsHybrid(result.decodePath)
                footer.decodePath = result.decodePath
                footer.tokensPerSecond = result.tokensPerSecond
                footer.ttftMs = result.firstTokenLatencyMs
                footer.contextUsed = result.promptTokens.count + result.tokens.count
                footer.power = chatPowerFooter(
                    capability: powerCapability,
                    summary: measured.power,
                    tokensPerSecond: result.tokensPerSecond
                )
                let text = assistantTextFromGeneratedTokens(
                    result.tokens,
                    decode: tokenizer.decodeImpl,
                    eosToken: invocation.config.eosToken
                )
                session.appendAssistant(text)
                streamingAssistant = ""
                if !useTUI {
                    writePlain("")
                    writePlain(footer.render())
                } else {
                    renderTUI(status: .idle)
                }
            } catch let error as RealModelInferenceError {
                if case .cancelled = error {
                    let partial = ChatSession.sanitizeAssistantText(streamingAssistant)
                    if !partial.isEmpty {
                        session.appendAssistant(partial)
                    }
                    streamingAssistant = ""
                    footer.power = chatPowerFooter(
                        capability: powerCapability,
                        summary: nil,
                        tokensPerSecond: footer.tokensPerSecond
                    )
                    if cancel.wasUserInterrupt {
                        if !useTUI {
                            writePlain("")
                            writePlain("^C cancelled")
                        } else {
                            renderTUI(status: .cancelled)
                        }
                    } else if !useTUI {
                        writePlain("")
                        writePlain(footer.render())
                    } else {
                        renderTUI(status: .idle)
                    }
                    continue
                }
                throw error
            }
        }
    }

    if useTUI {
        display.stop()
    }
    return 0
}

enum EspressoGenerateCLI {
    static func main() -> Int32 {
        do {
            let options = try Options.parse(CommandLine.arguments)
            try validateChatVsMLXFlags(options)
            let defaults = detectDemoDefaults()

            if options.showHelp {
                printUsage()
                return 0
            }
            if options.listModels {
                printAvailableModels()
                return 0
            }

            applyCLIExperimentEnvironment(options)

            let command = options.command ?? defaultCommand(for: options)
            let powerEnabled = try resolvePowerEnabled(command: command, powerMode: options.powerMode)

            if shouldUseDefaultGPT2Demo(options) {
                try ensureGPT2DemoWeightsAndTokenizer(defaults: defaults, allowBootstrap: options.allowBootstrap)
            }

            if shouldExitAfterPreparingDemo(options) {
                printPreparedDemoSummary(defaults)
                return 0
            }

            if command == .doctor {
                return try runDoctor(defaults: defaults, options: options)
            }

            let invocation = try resolveInvocation(from: options, demoDefaults: defaults, command: command)

            switch command {
            case .chat:
                if options.compareOpponent == .mlx {
                    return try runChatVsMLX(
                        invocation: invocation,
                        options: options,
                        defaults: defaults,
                        powerEnabled: powerEnabled
                    )
                }
                return try runChat(invocation: invocation, options: options, powerEnabled: powerEnabled)
            case .generate:
                let result = try runGenerate(invocation: invocation)
                let artifactDirectory = try writeGenerateArtifacts(
                    invocation: invocation,
                    result: result,
                    defaults: defaults,
                    requestedOutputDir: invocation.outputDir
                )
                if options.jsonOutput {
                    var payloadObject: [String: Any] = [
                        "model": invocation.config.name,
                        "backend": backendPayload(result),
                    ]
                    if let artifactDirectory {
                        payloadObject["report_dir"] = artifactDirectory
                    }
                    let payload = try JSONSerialization.data(
                        withJSONObject: payloadObject,
                        options: [.prettyPrinted, .sortedKeys]
                    )
                    print(String(decoding: payload, as: UTF8.self))
                } else {
                    print(result.text)
                    if invocation.showStats {
                        printGenerateStats(invocation, result: result)
                    }
                    if let artifactDirectory {
                        stderrLine("report_dir=\(artifactDirectory)")
                    }
                }
                return 0
            case .demo, .compare:
                let useTUI = invocation.bundlePath == nil
                    && invocation.config.architecture == .gpt2
                    && shouldUseTUI(command: command, options: options)
                let report = if useTUI {
                    try runLiveCompare(invocation: invocation, defaults: defaults, powerEnabled: powerEnabled)
                } else {
                    try runCompareOrBench(invocation: invocation, defaults: defaults, powerEnabled: powerEnabled)
                }
                let outputDirectory = try writeCompareArtifacts(report: report, defaults: defaults, requestedOutputDir: invocation.outputDir)
                let finalized = finalizedCompareReport(report, outputDirectory: outputDirectory)
                if options.jsonOutput {
                    let payload = try JSONSerialization.data(withJSONObject: [
                        "model": finalized.model,
                        "token_match": finalized.tokenMatch,
                        "text_match": finalized.textMatch,
                        "report_dir": outputDirectory,
                        "espresso": backendPayload(finalized.espresso),
                        "coreml": backendPayload(finalized.coreML),
                    ], options: [.prettyPrinted, .sortedKeys])
                    print(String(decoding: payload, as: UTF8.self))
                } else {
                    print(finalized.espresso.text)
                    if invocation.showStats {
                        printCompareSummary(finalized)
                    }
                }
                return 0
            case .bench:
                let report = try runCompareOrBench(invocation: invocation, defaults: defaults, powerEnabled: powerEnabled)
                let outputDirectory = try writeCompareArtifacts(report: report, defaults: defaults, requestedOutputDir: invocation.outputDir)
                let finalized = finalizedCompareReport(report, outputDirectory: outputDirectory)
                if options.jsonOutput {
                    let payload = try JSONSerialization.data(withJSONObject: [
                        "report_dir": outputDirectory,
                        "espresso": backendPayload(finalized.espresso),
                        "coreml": backendPayload(finalized.coreML),
                    ], options: [.prettyPrinted, .sortedKeys])
                    print(String(decoding: payload, as: UTF8.self))
                } else {
                    printCompareSummary(finalized)
                }
                return 0
            case .suite:
                return try runPromptSuite(
                    invocation: invocation,
                    options: options,
                    defaults: defaults,
                    powerEnabled: powerEnabled
                )
            case .doctor:
                return try runDoctor(defaults: defaults, options: options)
            }
        } catch let error as CLIError {
            stderrLine("espresso-generate error: \(error.localizedDescription)")
            if case .usage = error {
                stderrLine("Run `espresso-generate --help` for usage.")
            }
            return 1
        } catch {
            let message = (error as? LocalizedError)?.errorDescription ?? String(describing: error)
            stderrLine("espresso-generate error: \(message)")
            return 1
        }
    }
}

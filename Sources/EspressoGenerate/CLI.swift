import Foundation
import Darwin
import ANETypes
import ModelSupport
import RealModelInference

enum CommandName: String {
    case demo
    case generate
    case compare
    case bench
    case doctor
}

enum PowerMode: Equatable {
    case auto
    case on
    case off
}

struct Options {
    var command: CommandName?
    var modelName: String? = ProcessInfo.processInfo.environment["ESPRESSO_MODEL"]
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
    var seed: Int = Options.parseIntEnv("ESPRESSO_COMPARE_SEED") ?? 1234
    var coreMLComputeUnits: String = ProcessInfo.processInfo.environment["ESPRESSO_COREML_COMPUTE_UNITS"] ?? "cpu_and_neural_engine"
    var showStats = true
    var listModels = false
    var prepareDemo = false
    var allowBootstrap = true
    var showHelp = false
    var forceTUI = false
    var disableTUI = false
    var preferLiveCompare = false
    var preferBenchCompare = false
    var powerMode: PowerMode = .auto
    var outputDir: String?
    var jsonOutput = false

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
            case "--prepare-demo":
                options.prepareDemo = true
            case "--no-bootstrap":
                options.allowBootstrap = false
            case "--no-stats":
                options.showStats = false
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

private struct MetadataConfigFile: Decodable {
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
    let architecture: String

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
            architecture: parsedArchitecture
        )
    }
}

struct ResolvedInvocation {
    let config: MultiModelConfig
    let weightsDir: String
    let tokenizerDir: String
    let prompt: String
    let maxTokens: Int
    let temperature: Float
    let showStats: Bool
    let coreMLModelPath: String?
    let coreMLSequenceLength: Int?
    let compareWarmup: Int
    let compareIterations: Int
    let coreMLComputeUnits: String
    let allowBootstrap: Bool
    let seed: Int
    let outputDir: String?
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
    let decodeImpl: ([Int]) -> String

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

private let posixLocale = Locale(identifier: "en_US_POSIX")
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

private func printUsage() {
    let usage = """
    Usage:
      espresso-generate demo [options] [prompt...]
      espresso-generate generate [options] [prompt...]
      espresso-generate compare [options] [prompt...]
      espresso-generate bench [options] [prompt...]
      espresso-generate doctor [--json]
      espresso-generate [options] [prompt...]

    Commands:
      demo       First-run GPT-2 experience. Boots assets if needed and runs live compare in a TTY.
      generate   Run Espresso ANE generation only.
      compare    Compare Espresso against the same GPT-2 Core ML baseline. Use --live or --bench.
      bench      Run a fair sequential benchmark, export JSON/CSV/Markdown, and capture power when available.
      doctor     Validate hardware, scripts, caches, demo assets, Python tooling, and power telemetry support.

    Common options:
      -m, --model NAME         ModelRegistry key or alias
      -w, --weights DIR        Weights directory
      -t, --tokenizer DIR      Tokenizer directory or tokenizer asset path
          --coreml-model PATH  Override the exported GPT-2 Core ML baseline
      -p, --prompt TEXT        Prompt text; otherwise use trailing args or piped stdin
      -n, --max-tokens N       Max new tokens (default: 128)
          --temperature FLOAT  Sampling temperature; 0 = greedy (default: 0)
          --seed N             Sampling seed for compare/bench (default: 1234)
          --prepare-demo       Prepare the managed GPT-2 demo assets and exit when no prompt is provided
          --output-dir DIR     Report output directory for compare/bench exports
          --compare-warmup N   Warmup iterations for sequential compare/bench (default: 1)
          --compare-iterations N
                               Measured iterations for sequential compare/bench (default: 3)
          --coreml-seq-len N   Override the fixed Core ML sequence length used for compare/export
          --coreml-compute-units all|cpu_and_neural_engine|cpu_and_gpu|cpu_only
          --tui                Force the live terminal dashboard
          --no-tui             Disable the live terminal dashboard
          --live               Prefer the live compare experience
          --bench              Prefer the fair sequential benchmark path
          --power              Require power telemetry; exits if unavailable
          --no-power           Disable power telemetry
          --json               Emit JSON doctor/compare output to stdout
          --list-models        Print available models and exit
          --no-bootstrap       Do not auto-install/download demo dependencies or assets
          --no-stats           Suppress timing stats on stderr
      -h, --help               Show this help

    Environment:
      ESPRESSO_HOME                 Override the managed cache/report root
      ESPRESSO_CACHE_HOME           Override the managed cache directory
      ESPRESSO_REPO_ROOT            Override repository root detection
      ESPRESSO_SCRIPTS_DIR          Override the helper scripts directory
      ESPRESSO_MODEL
      ESPRESSO_WEIGHTS_DIR
      ESPRESSO_TOKENIZER_DIR
      ESPRESSO_COREML_MODEL
      ESPRESSO_COREML_SEQ_LEN
      ESPRESSO_COREML_COMPUTE_UNITS
      ESPRESSO_COMPARE_WARMUP
      ESPRESSO_COMPARE_ITERATIONS
      ESPRESSO_COMPARE_SEED
      ESPRESSO_MAX_TOKENS
      ESPRESSO_TEMPERATURE

    Examples:
      ./espresso
      ./espresso "Hello"
      ./espresso doctor
      espresso-generate demo "Hello"
      espresso-generate doctor
      espresso-generate demo
      espresso-generate compare --live "Hello"
      espresso-generate bench --output-dir ./reports/gpt2 "Hello"
    """
    print(usage)
}

private func printAvailableModels() {
    print("Available models:")
    for name in ModelRegistry.all.keys.sorted() {
        guard let config = ModelRegistry.all[name] else { continue }
        let status = config.architecture == .gpt2 ? "ready" : "RoPE pending"
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

private func hasSentencePieceAssets(in directory: URL) -> Bool {
    let fileManager = FileManager()
    return fileManager.fileExists(atPath: directory.appendingPathComponent("tokenizer.bin").path)
}

private func directoryHasTokenizerAssets(_ directory: URL, for config: MultiModelConfig) -> Bool {
    switch config.architecture {
    case .gpt2:
        return hasGPT2TokenizerAssets(in: directory)
    case .llama:
        return hasSentencePieceAssets(in: directory)
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
            if url.lastPathComponent == "tokenizer.bin", hasSentencePieceAssets(in: parent) {
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
            "Unable to locate SentencePiece tokenizer assets. Pass --tokenizer or place tokenizer.bin near the weights."
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
    case .generate, .bench, .doctor:
        return nil
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
        return CLITokenizer(decodeImpl: tokenizer.decode)
    case .llama:
        let tokenizerURL = tokenizerDirURL.appendingPathComponent("tokenizer.bin")
        let tokenizer = try SentencePieceTokenizer(modelURL: tokenizerURL)
        return CLITokenizer(decodeImpl: tokenizer.decode)
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
    case .generate, .bench, .doctor:
        return false
    }
}

private func resolveInvocation(from options: Options, demoDefaults: DemoDefaults, command: CommandName) throws -> ResolvedInvocation {
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
    let prompt = command == .doctor ? "" : try resolvePrompt(options, command: command)
    return ResolvedInvocation(
        config: config,
        weightsDir: weightsDir,
        tokenizerDir: tokenizerDir,
        prompt: prompt,
        maxTokens: options.maxTokens,
        temperature: options.temperature,
        showStats: options.showStats,
        coreMLModelPath: options.coreMLModelPath,
        coreMLSequenceLength: options.coreMLSequenceLength,
        compareWarmup: options.compareWarmup,
        compareIterations: options.compareIterations,
        coreMLComputeUnits: options.coreMLComputeUnits,
        allowBootstrap: options.allowBootstrap,
        seed: options.seed,
        outputDir: options.outputDir
    )
}

private func printPreparedDemoSummary(_ defaults: DemoDefaults) {
    stderrLine("Default GPT-2 demo assets are ready.")
    stderrLine("weights=\(defaults.weightsDir.path)")
    stderrLine("tokenizer=\(defaults.tokenizerDir.path)")
    stderrLine("next=./espresso")
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
    let result = try engine.generate(
        prompt: invocation.prompt,
        maxTokens: invocation.maxTokens,
        temperature: invocation.temperature,
        onStep: { step in
            tokenLatenciesMs.append(step.tokenLatencyMs)
            totalTimeMs = step.elapsedMs
            onStep?(step)
        }
    )
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
        tokenLatenciesMs: tokenLatenciesMs
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

    for iteration in 0..<totalIterations {
        let metrics = try run()
        if compileTimeMs == 0, metrics.compileTimeMs > 0 {
            compileTimeMs = metrics.compileTimeMs
        }
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
        tokenLatenciesMs: lastMeasured.tokenLatenciesMs
    )
}

private func runEspressoBenchmark(invocation: ResolvedInvocation) throws -> BackendRunMetrics {
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
    let coreMLModelPath = try ensureGPT2CoreMLModel(
        defaults: defaults,
        weightsDir: invocation.weightsDir,
        sequenceLength: sequenceLength,
        explicitModelPath: invocation.coreMLModelPath,
        allowBootstrap: invocation.allowBootstrap
    )
    let result = try runGPT2CoreMLReference(
        defaults: defaults,
        coreMLModelPath: coreMLModelPath,
        weightsDir: invocation.weightsDir,
        promptTokens: promptTokens,
        sequenceLength: sequenceLength,
        maxTokens: invocation.maxTokens,
        temperature: invocation.temperature,
        warmup: invocation.compareWarmup,
        iterations: invocation.compareIterations,
        computeUnits: invocation.coreMLComputeUnits,
        seed: invocation.seed,
        allowBootstrap: invocation.allowBootstrap
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
        let enabledByDefault = command == .demo || command == .bench
        if emitWarnings, enabledByDefault, !capability.available {
            stderrLine("espresso-generate warning: power telemetry unavailable: \(capability.message)")
        }
        return enabledByDefault && capability.available
    }
}

private func maybeMeasurePower<T>(
    enabled: Bool,
    body: () throws -> T
) throws -> (result: T, power: PowerSummary?) {
    guard enabled else {
        return (try body(), nil)
    }
    let capability = PowerTelemetryCollector.capability()
    guard capability.available else {
        return (try body(), nil)
    }
    let collector = PowerTelemetryCollector()
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
            format: "model=%@ prompt_tokens=%d generated_tokens=%d compile_ms=%.2f first_token_ms=%.2f tok_per_s=%.2f median_token_ms=%.2f p95_token_ms=%.2f",
            locale: posixLocale,
            invocation.config.name,
            result.promptTokens.count,
            result.generatedTokens.count,
            result.compileTimeMs,
            result.firstTokenLatencyMs,
            result.tokensPerSecond,
            result.medianTokenMs,
            result.p95TokenMs
        )
    )
    stderrLine("weights=\(invocation.weightsDir)")
    stderrLine("tokenizer=\(invocation.tokenizerDir)")
}

private func printCompareSummary(_ report: CompareReport) {
    stderrLine(
        String(
            format: "espresso compile_ms=%.2f first_token_ms=%.2f tok_per_s=%.2f median_token_ms=%.2f p95_token_ms=%.2f",
            locale: posixLocale,
            report.espresso.compileTimeMs,
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
        "power": [
            "espresso": powerPayload(report.espressoPower),
            "coreml": powerPayload(report.coreMLPower),
        ],
    ]
    let jsonData = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
    try jsonData.write(to: jsonURL)

    let csv = """
    backend,compile_ms,first_token_ms,tokens_per_second,median_token_ms,p95_token_ms,total_time_ms,package_w,cpu_w,gpu_w,ane_w
    espresso,\(report.espresso.compileTimeMs),\(report.espresso.firstTokenLatencyMs),\(report.espresso.tokensPerSecond),\(report.espresso.medianTokenMs),\(report.espresso.p95TokenMs),\(report.espresso.totalTimeMs),\(report.espressoPower?.packageW ?? 0),\(report.espressoPower?.cpuW ?? 0),\(report.espressoPower?.gpuW ?? 0),\(report.espressoPower?.aneW ?? 0)
    coreml,\(report.coreML.compileTimeMs),\(report.coreML.firstTokenLatencyMs),\(report.coreML.tokensPerSecond),\(report.coreML.medianTokenMs),\(report.coreML.p95TokenMs),\(report.coreML.totalTimeMs),\(report.coreMLPower?.packageW ?? 0),\(report.coreMLPower?.cpuW ?? 0),\(report.coreMLPower?.gpuW ?? 0),\(report.coreMLPower?.aneW ?? 0)
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
    return outputDir.path
}

private func backendPayload(_ backend: BackendRunMetrics) -> [String: Any] {
    [
        "compile_time_ms": backend.compileTimeMs,
        "first_token_latency_ms": backend.firstTokenLatencyMs,
        "tokens_per_second": backend.tokensPerSecond,
        "median_token_ms": backend.medianTokenMs,
        "p95_token_ms": backend.p95TokenMs,
        "total_time_ms": backend.totalTimeMs,
        "generated_tokens": backend.generatedTokens.map(Int.init),
        "token_latencies_ms": backend.tokenLatenciesMs,
        "text": backend.text,
    ]
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
            prompt: invocation.prompt,
            maxTokens: min(invocation.maxTokens, 4),
            temperature: invocation.temperature
        )
    }.power

    let coreMLCalibration = try maybeMeasurePower(enabled: powerEnabled) {
        try runGPT2CoreMLReference(
            defaults: defaults,
            coreMLModelPath: coreMLModelPath,
            weightsDir: invocation.weightsDir,
            promptTokens: try encodePromptTokens(invocation.prompt, config: invocation.config, tokenizerDir: invocation.tokenizerDir),
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
    let promptTokens = try encodePromptTokens(invocation.prompt, config: invocation.config, tokenizerDir: invocation.tokenizerDir)

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

private func encodePromptTokens(_ prompt: String, config: MultiModelConfig, tokenizerDir: String) throws -> [TokenID] {
    let tokenizer = try loadTokenizer(config: config, tokenizerDir: tokenizerDir)
    let decoded = tokenizer.decode // silence unused? no
    _ = decoded
    switch config.architecture {
    case .gpt2:
        let tokenizerDirURL = URL(fileURLWithPath: tokenizerDir, isDirectory: true)
        let tokenizer = try GPT2BPETokenizer(
            vocabURL: tokenizerDirURL.appendingPathComponent("vocab.json"),
            mergesURL: tokenizerDirURL.appendingPathComponent("merges.txt")
        )
        return try tokenizer.encode(prompt).map(validateToken)
    case .llama:
        let tokenizerDirURL = URL(fileURLWithPath: tokenizerDir, isDirectory: true)
        let tokenizer = try SentencePieceTokenizer(modelURL: tokenizerDirURL.appendingPathComponent("tokenizer.bin"))
        return try tokenizer.encode(prompt).map(validateToken)
    }
}

private func metricsOrZero(_ value: Double, fallback: Double) -> Double {
    value > 0 ? value : fallback
}

private func runGenerate(invocation: ResolvedInvocation) throws -> BackendRunMetrics {
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

enum EspressoGenerateCLI {
    static func main() -> Int32 {
        do {
            let options = try Options.parse(CommandLine.arguments)
            let defaults = detectDemoDefaults()

            if options.showHelp {
                printUsage()
                return 0
            }
            if options.listModels {
                printAvailableModels()
                return 0
            }

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
            case .generate:
                let result = try runGenerate(invocation: invocation)
                print(result.text)
                if invocation.showStats {
                    printGenerateStats(invocation, result: result)
                }
                return 0
            case .demo, .compare:
                let useTUI = shouldUseTUI(command: command, options: options)
                let report = if useTUI {
                    try runLiveCompare(invocation: invocation, defaults: defaults, powerEnabled: powerEnabled)
                } else {
                    try runCompareOrBench(invocation: invocation, defaults: defaults, powerEnabled: powerEnabled)
                }
                let outputDirectory = try writeCompareArtifacts(report: report, defaults: defaults, requestedOutputDir: invocation.outputDir)
                let finalized = CompareReport(
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
                let finalized = CompareReport(
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

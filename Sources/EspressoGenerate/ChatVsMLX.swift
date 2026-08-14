import Foundation

enum CompareOpponent: String, Equatable, Sendable {
    case mlx
}

enum MLXWeightPrecision: Equatable, Sendable {
    case native
    case quantized(label: String)
}

struct ChatVsMLXFairness: Equatable, Sendable {
    static let requiredHuggingFaceRepo = "Qwen/Qwen2.5-1.5B-Instruct"
    static let requiredEspressoName = "Qwen2.5-1.5B-Instruct"
    static let espressoPrecision = "fp16"

    let huggingfaceRepo: String
    let espressoModelName: String
    let espressoPrecision: String
    let mlxPrecisionLabel: String
    let mlxQuantization: MLXWeightPrecision
    let greedy: Bool
    let maxNewTokens: Int
    let tokPerSecExcludesCompile: Bool

    func title() -> String {
        "ESPRESSO \(espressoPrecision) vs MLX \(mlxPrecisionLabel)"
    }

    func espressoLaneHeader() -> String {
        switch mlxQuantization {
        case .native:
            return "ESPRESSO / ANE  \(espressoPrecision)"
        case let .quantized(label):
            return "ESPRESSO / ANE  \(espressoPrecision)  vs \(label)"
        }
    }

    func mlxLaneHeader() -> String {
        "MLX / GPU  \(mlxPrecisionLabel)"
    }

    func applyingLoadedPrecision(_ loadedPrecision: String) -> ChatVsMLXFairness {
        switch mlxQuantization {
        case .quantized:
            return self
        case .native:
            let label = mlxNativePrecisionLabel(loadedPrecision)
            return ChatVsMLXFairness(
                huggingfaceRepo: huggingfaceRepo,
                espressoModelName: espressoModelName,
                espressoPrecision: espressoPrecision,
                mlxPrecisionLabel: label,
                mlxQuantization: mlxQuantization,
                greedy: greedy,
                maxNewTokens: maxNewTokens,
                tokPerSecExcludesCompile: tokPerSecExcludesCompile
            )
        }
    }
}

func mlxNativePrecisionLabel(_ raw: String) -> String {
    switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
    case "float16", "fp16":
        return "fp16"
    case "bfloat16", "bf16":
        return "bf16"
    default:
        return raw
    }
}

struct ChatVsMLXTurnMetrics: Equatable, Sendable {
    var tokensPerSecond: Double
    var ttftMs: Double
    var compileMs: Double
    var packageW: Double?
    var joulesPerToken: Double?
}

struct ChatVsMLXScoreboardRow: Equatable, Sendable {
    var metric: String
    var espresso: String
    var mlx: String
    var winner: String
}

func parseCompareOpponent(_ raw: String) throws -> CompareOpponent {
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    if trimmed == CompareOpponent.mlx.rawValue {
        return .mlx
    }
    throw CLIError.usage(
        "chat --vs only supports mlx. Core ML Stories compare is `espresso compare`, not the HN path."
    )
}

func parseMLXQuantizationFlag(_ raw: String?) throws -> MLXWeightPrecision {
    guard let raw else {
        return .native
    }
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else {
        throw CLIError.usage("--mlx-quant requires a label such as 4bit")
    }
    switch trimmed.lowercased() {
    case "4", "4bit", "4-bit", "q4", "int4", "q4_0":
        return .quantized(label: "4-bit")
    default:
        throw CLIError.usage(
            "Unsupported --mlx-quant \(raw). Use 4bit, or omit the flag for native fp16/bf16."
        )
    }
}

func makeChatVsMLXFairness(
    espressoModelName: String,
    greedy: Bool,
    maxNewTokens: Int,
    mlxQuantFlag: String?,
    mlxModelOverride: String?
) throws -> ChatVsMLXFairness {
    guard greedy else {
        throw CLIError.usage(
            "chat --vs mlx requires --greedy so both lanes use identical argmax decoding."
        )
    }
    guard maxNewTokens > 0 else {
        throw CLIError.usage("chat --vs mlx requires --max-tokens > 0")
    }
    let espressoName = espressoModelName.trimmingCharacters(in: .whitespacesAndNewlines)
    guard espressoName == ChatVsMLXFairness.requiredEspressoName else {
        throw CLIError.usage(
            "chat --vs mlx compares \(ChatVsMLXFairness.requiredHuggingFaceRepo) only; got espresso model \(espressoName)."
        )
    }

    let quantization = try parseMLXQuantizationFlag(mlxQuantFlag)
    if let override = mlxModelOverride?.trimmingCharacters(in: .whitespacesAndNewlines), !override.isEmpty {
        let sameRepo = override == ChatVsMLXFairness.requiredHuggingFaceRepo
        switch quantization {
        case .native:
            guard sameRepo else {
                throw CLIError.usage(
                    "Unlabeled MLX must load \(ChatVsMLXFairness.requiredHuggingFaceRepo). A different --mlx-model requires --mlx-quant so both footers label the quantization."
                )
            }
        case .quantized:
            break
        }
    }

    let mlxLabel: String
    switch quantization {
    case .native:
        mlxLabel = "fp16"
    case let .quantized(label):
        mlxLabel = label
    }

    return ChatVsMLXFairness(
        huggingfaceRepo: ChatVsMLXFairness.requiredHuggingFaceRepo,
        espressoModelName: espressoName,
        espressoPrecision: ChatVsMLXFairness.espressoPrecision,
        mlxPrecisionLabel: mlxLabel,
        mlxQuantization: quantization,
        greedy: true,
        maxNewTokens: maxNewTokens,
        tokPerSecExcludesCompile: true
    )
}

func validateChatVsMLXFlags(_ options: Options) throws {
    if options.mlxQuant != nil || options.mlxModel != nil {
        guard options.compareOpponent == .mlx else {
            throw CLIError.usage("--mlx-quant and --mlx-model require --vs mlx")
        }
    }
    if options.compareOpponent == .mlx, let command = options.command, command != .chat {
        throw CLIError.usage(
            "--vs mlx is a chat flag. Core ML Stories compare is `espresso compare`, not the HN path."
        )
    }
}

func assertMLXLoadMatchesFairness(
    quantized: Bool,
    precision: String,
    repo: String,
    fairness: ChatVsMLXFairness
) throws {
    if quantized {
        guard case .quantized = fairness.mlxQuantization else {
            throw CLIError.runtime(
                "MLX loaded a quantized checkpoint (\(precision)) without --mlx-quant. Unlabeled 4-bit is rejected. Re-run with --mlx-quant 4bit so both footers label it, or load native fp16/bf16."
            )
        }
    } else if case .quantized = fairness.mlxQuantization {
        throw CLIError.runtime(
            "chat --vs mlx --mlx-quant requested \(fairness.mlxPrecisionLabel) but MLX loaded native \(precision). Pass a quantized MLX directory via --mlx-model."
        )
    }
    if case .native = fairness.mlxQuantization {
        let allowed = ["float16", "fp16", "bfloat16", "bf16"]
        let normalized = precision.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard allowed.contains(normalized) else {
            throw CLIError.runtime(
                "MLX native load must be fp16 or bf16; got \(precision). Do not silently switch to 4-bit."
            )
        }
        guard repo == fairness.huggingfaceRepo else {
            throw CLIError.runtime(
                "MLX repo \(repo) does not match Espresso fairness repo \(fairness.huggingfaceRepo)."
            )
        }
    }
}

func completionTokensPerSecond(generatedTokenCount: Int, completionMilliseconds: Double) -> Double {
    guard generatedTokenCount > 0, completionMilliseconds.isFinite, completionMilliseconds > 0 else {
        return 0
    }
    return Double(generatedTokenCount) / (completionMilliseconds / 1_000.0)
}

func laneJoulesPerToken(_ lane: LiveLaneSnapshot) -> Double? {
    guard let power = lane.power, power.sampleCount > 0 else {
        return nil
    }
    return joulesPerToken(packageWatts: power.packageW, tokensPerSecond: lane.tokensPerSecond)
}

func metricsFromLane(_ lane: LiveLaneSnapshot) -> ChatVsMLXTurnMetrics {
    ChatVsMLXTurnMetrics(
        tokensPerSecond: lane.tokensPerSecond,
        ttftMs: lane.ttftMs,
        compileMs: lane.compileMs,
        packageW: (lane.power?.sampleCount ?? 0) > 0 ? lane.power?.packageW : nil,
        joulesPerToken: laneJoulesPerToken(lane)
    )
}

func pairedChatVsMLXMetrics(
    espresso: [ChatVsMLXTurnMetrics],
    mlx: [ChatVsMLXTurnMetrics]
) -> (espresso: ChatVsMLXTurnMetrics, mlx: ChatVsMLXTurnMetrics) {
    let count = min(espresso.count, mlx.count)
    return (
        averageChatVsMLXMetrics(Array(espresso.prefix(count))),
        averageChatVsMLXMetrics(Array(mlx.prefix(count)))
    )
}

func averageChatVsMLXMetrics(_ turns: [ChatVsMLXTurnMetrics]) -> ChatVsMLXTurnMetrics {
    guard !turns.isEmpty else {
        return ChatVsMLXTurnMetrics(tokensPerSecond: 0, ttftMs: 0, compileMs: 0, packageW: nil, joulesPerToken: nil)
    }
    let count = Double(turns.count)
    let packageValues = turns.compactMap(\.packageW)
    let jouleValues = turns.compactMap(\.joulesPerToken)
    return ChatVsMLXTurnMetrics(
        tokensPerSecond: turns.map(\.tokensPerSecond).reduce(0, +) / count,
        ttftMs: turns.map(\.ttftMs).reduce(0, +) / count,
        compileMs: turns.map(\.compileMs).reduce(0, +) / count,
        packageW: packageValues.isEmpty ? nil : packageValues.reduce(0, +) / Double(packageValues.count),
        joulesPerToken: jouleValues.isEmpty ? nil : jouleValues.reduce(0, +) / Double(jouleValues.count)
    )
}

func chatVsMLXScoreboard(
    espresso: ChatVsMLXTurnMetrics,
    mlx: ChatVsMLXTurnMetrics
) -> [ChatVsMLXScoreboardRow] {
    [
        scoreboardRow(
            metric: "tok/s",
            espresso: espresso.tokensPerSecond,
            mlx: mlx.tokensPerSecond,
            higherWins: true
        ),
        scoreboardRow(
            metric: "TTFT ms",
            espresso: espresso.ttftMs,
            mlx: mlx.ttftMs,
            higherWins: false
        ),
        optionalScoreboardRow(
            metric: "package W",
            espresso: espresso.packageW,
            mlx: mlx.packageW,
            higherWins: false
        ),
        optionalScoreboardRow(
            metric: "J/tok",
            espresso: espresso.joulesPerToken,
            mlx: mlx.joulesPerToken,
            higherWins: false
        ),
    ]
}

func formatChatVsMLXScoreboard(espresso: ChatVsMLXTurnMetrics, mlx: ChatVsMLXTurnMetrics) -> String {
    let rows = chatVsMLXScoreboard(espresso: espresso, mlx: mlx)
    let header = padded("metric", 12) + padded("espresso", 16) + padded("mlx", 16) + "winner"
    let body = rows.map { row in
        padded(row.metric, 12) + padded(row.espresso, 16) + padded(row.mlx, 16) + row.winner
    }
    return ([header] + body).joined(separator: "\n")
}

func mlxInstallInstructions() -> String {
    """
    MLX is not installed. Install the native-precision (fp16/bf16) runtime, then retry:
      python3 -m pip install mlx-lm
    Do not install a 4-bit quantized build to make this pane work.
    Then re-run: ./espresso chat --vs mlx --greedy --model <path.esp>
    """
}

func resolveMLXPython(candidates: [String], canImport: (String) -> Bool) -> String? {
    candidates.first(where: canImport)
}

func mlxPythonCandidates(environment: [String: String] = ProcessInfo.processInfo.environment) -> [String] {
    if let override = environment["ESPRESSO_MLX_PYTHON"], !override.isEmpty {
        return [override]
    }
    return ["python3.13", "python3.12", "python3"]
}

func huggingFaceHubSnapshot(repo: String, cacheRoot: URL) -> URL? {
    let slug = "models--" + repo.replacingOccurrences(of: "/", with: "--")
    let snapshotsRoot = cacheRoot
        .appendingPathComponent(slug, isDirectory: true)
        .appendingPathComponent("snapshots", isDirectory: true)
    let fileManager = FileManager()
    guard let snapshots = try? fileManager.contentsOfDirectory(
        at: snapshotsRoot,
        includingPropertiesForKeys: [.isDirectoryKey],
        options: [.skipsHiddenFiles]
    ) else {
        return nil
    }
    for snapshot in snapshots {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: snapshot.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            continue
        }
        let config = snapshot.appendingPathComponent("config.json")
        let weights = snapshot.appendingPathComponent("model.safetensors")
        if fileManager.fileExists(atPath: config.path), fileManager.fileExists(atPath: weights.path) {
            return snapshot.standardizedFileURL
        }
    }
    return nil
}

func defaultHuggingFaceCacheRoot() -> URL {
    URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
        .appendingPathComponent(".cache/huggingface/hub", isDirectory: true)
}

private func scoreboardRow(
    metric: String,
    espresso: Double,
    mlx: Double,
    higherWins: Bool
) -> ChatVsMLXScoreboardRow {
    ChatVsMLXScoreboardRow(
        metric: metric,
        espresso: formatScore(espresso),
        mlx: formatScore(mlx),
        winner: winnerName(espresso: espresso, mlx: mlx, higherWins: higherWins)
    )
}

private func optionalScoreboardRow(
    metric: String,
    espresso: Double?,
    mlx: Double?,
    higherWins: Bool
) -> ChatVsMLXScoreboardRow {
    guard let espresso, let mlx else {
        return ChatVsMLXScoreboardRow(
            metric: metric,
            espresso: espresso.map(formatScore) ?? "unavailable",
            mlx: mlx.map(formatScore) ?? "unavailable",
            winner: "—"
        )
    }
    return scoreboardRow(metric: metric, espresso: espresso, mlx: mlx, higherWins: higherWins)
}

private func winnerName(espresso: Double, mlx: Double, higherWins: Bool) -> String {
    if abs(espresso - mlx) < 1e-9 {
        return "tie"
    }
    if higherWins {
        return espresso > mlx ? "espresso" : "mlx"
    }
    return espresso < mlx ? "espresso" : "mlx"
}

private func formatScore(_ value: Double) -> String {
    String(format: "%.2f", value)
}

private func padded(_ value: String, _ width: Int) -> String {
    if value.count >= width {
        return String(value.prefix(width - 1)) + " "
    }
    return value + String(repeating: " ", count: width - value.count)
}

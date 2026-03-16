import Foundation
import ModelSupport

struct DemoDefaults {
    let repoRoot: URL?
    let workingDirectory: URL
    let stateRoot: URL
    let cacheRoot: URL
    let reportsRoot: URL
    let hfCacheRoot: URL
    let weightsDir: URL
    let tokenizerDir: URL
    let coreMLDir: URL
    let toolsVenvDir: URL
    let scriptsDir: URL?
    let legacyArtifactsRoot: URL?

    var scriptsAvailable: Bool {
        guard let scriptsDir else { return false }
        let fileManager = FileManager()
        return fileManager.fileExists(atPath: scriptsDir.appendingPathComponent("bootstrap_gpt2_demo.py").path) &&
            fileManager.fileExists(atPath: scriptsDir.appendingPathComponent("export_gpt2_coreml.py").path) &&
            fileManager.fileExists(atPath: scriptsDir.appendingPathComponent("run_gpt2_coreml_reference.py").path)
    }

    func coreMLModelURL(sequenceLength: Int, weightsDirURL: URL? = nil) -> URL {
        if let weightsDirURL {
            let parent = weightsDirURL.deletingLastPathComponent()
            let basename = weightsDirURL.lastPathComponent
            let directoryName = basename == "gpt2_124m" ? "gpt2_coreml" : "\(basename)_coreml"
            return parent
                .appendingPathComponent(directoryName, isDirectory: true)
                .appendingPathComponent("gpt2_seq\(sequenceLength).mlpackage")
        }
        return coreMLDir.appendingPathComponent("gpt2_seq\(sequenceLength).mlpackage")
    }
}

struct CoreMLComparisonResult: Decodable, Sendable {
    let generatedTokens: [Int]
    let compileTimeMs: Double
    let firstTokenLatencyMs: Double
    let tokensPerSecond: Double
    let medianTokenMs: Double
    let p95TokenMs: Double
    let tokenLatenciesMs: [Double]
    let totalTimeMs: Double
    let computeUnits: String
    let seqLen: Int

    private enum CodingKeys: String, CodingKey {
        case generatedTokens = "generated_tokens"
        case compileTimeMs = "compile_time_ms"
        case firstTokenLatencyMs = "first_token_latency_ms"
        case tokensPerSecond = "tokens_per_second"
        case medianTokenMs = "median_token_ms"
        case p95TokenMs = "p95_token_ms"
        case tokenLatenciesMs = "token_latencies_ms"
        case totalTimeMs = "total_time_ms"
        case computeUnits = "compute_units"
        case seqLen = "seq_len"
    }
}

enum CoreMLStreamEvent: Sendable {
    case compile(compileTimeMs: Double, computeUnits: String, seqLen: Int)
    case token(token: UInt16, tokenIndex: Int, elapsedMs: Double, tokenLatencyMs: Double, tokensPerSecond: Double)
    case completed(CoreMLComparisonResult)
}

private struct ProcessOutput {
    let status: Int32
    let stdout: String
    let stderr: String
}

private struct RawCoreMLStreamEvent: Decodable {
    let type: String
    let token: Int?
    let tokenIndex: Int?
    let elapsedMs: Double?
    let tokenLatencyMs: Double?
    let tokensPerSecond: Double?
    let compileTimeMs: Double?
    let computeUnits: String?
    let seqLen: Int?
    let generatedTokens: [Int]?
    let firstTokenLatencyMs: Double?
    let medianTokenMs: Double?
    let p95TokenMs: Double?
    let tokenLatenciesMs: [Double]?
    let totalTimeMs: Double?

    private enum CodingKeys: String, CodingKey {
        case type
        case token
        case tokenIndex = "token_index"
        case elapsedMs = "elapsed_ms"
        case tokenLatencyMs = "token_latency_ms"
        case tokensPerSecond = "tokens_per_second"
        case compileTimeMs = "compile_time_ms"
        case computeUnits = "compute_units"
        case seqLen = "seq_len"
        case generatedTokens = "generated_tokens"
        case firstTokenLatencyMs = "first_token_latency_ms"
        case medianTokenMs = "median_token_ms"
        case p95TokenMs = "p95_token_ms"
        case tokenLatenciesMs = "token_latencies_ms"
        case totalTimeMs = "total_time_ms"
    }
}

func detectDemoDefaults() -> DemoDefaults {
    let fileManager = FileManager()
    let repoRoot = locateRepoRoot(fileManager: fileManager)
    let stateRoot = preferredStateRoot(fileManager: fileManager)
    let cacheRoot = preferredCacheRoot(fileManager: fileManager)
    let scriptsDir = resolvedScriptsDirectory(repoRoot: repoRoot)
    let legacyArtifactsRoot = repoRoot?.appendingPathComponent(".artifacts", isDirectory: true)
    let workingDirectory = repoRoot ?? stateRoot

    return DemoDefaults(
        repoRoot: repoRoot,
        workingDirectory: workingDirectory,
        stateRoot: stateRoot,
        cacheRoot: cacheRoot,
        reportsRoot: stateRoot.appendingPathComponent("reports", isDirectory: true),
        hfCacheRoot: cacheRoot.appendingPathComponent("huggingface", isDirectory: true),
        weightsDir: stateRoot.appendingPathComponent("demo/gpt2_124m", isDirectory: true),
        tokenizerDir: stateRoot.appendingPathComponent("demo/gpt2_tokenizer", isDirectory: true),
        coreMLDir: stateRoot.appendingPathComponent("coreml/gpt2_124m", isDirectory: true),
        toolsVenvDir: stateRoot.appendingPathComponent("tools/python/gpt2-tools-venv", isDirectory: true),
        scriptsDir: scriptsDir,
        legacyArtifactsRoot: legacyArtifactsRoot
    )
}

func shouldUseDefaultGPT2Demo(_ options: Options) -> Bool {
    if options.prepareDemo || options.command == .demo {
        return true
    }
    guard options.weightsDir == nil else {
        return false
    }
    guard let modelName = options.modelName else {
        return true
    }
    return canonicalModelName(for: modelName) == ModelRegistry.gpt2_124m.name
}

func nextPowerOfTwo(_ value: Int) -> Int {
    guard value > 1 else { return 1 }
    var result = 1
    while result < value {
        result <<= 1
    }
    return result
}

func ensureGPT2DemoWeightsAndTokenizer(
    defaults: DemoDefaults,
    allowBootstrap: Bool
) throws {
    try ensureStateDirectories(defaults)
    try migrateLegacyArtifactsIfNeeded(defaults)

    if hasGPT2TokenizerAssets(in: defaults.tokenizerDir),
       FileManager().fileExists(atPath: defaults.weightsDir.appendingPathComponent("metadata.json").path)
    {
        return
    }

    guard allowBootstrap else {
        throw CLIError.usage(
            "Missing default GPT-2 demo artifacts. Re-run without --no-bootstrap, run `espresso-generate doctor`, or pass explicit --weights/--tokenizer paths."
        )
    }

    let scriptsDir = try requireScriptsDirectory(defaults)
    stderrLine("Preparing default GPT-2 demo artifacts in \(defaults.stateRoot.path)")
    let python = try ensurePythonEnvironment(
        defaults: defaults,
        requiredModules: ["numpy", "torch", "transformers"]
    )
    try runProcessStreaming(
        executable: python,
        arguments: [
            scriptsDir.appendingPathComponent("bootstrap_gpt2_demo.py").path,
            "--weights-out", defaults.weightsDir.path,
            "--tokenizer-out", defaults.tokenizerDir.path,
            "--cache-dir", defaults.hfCacheRoot.path,
        ],
        workingDirectory: defaults.workingDirectory,
        environment: pythonEnvironment(defaults: defaults)
    )
}

func ensureGPT2CoreMLModel(
    defaults: DemoDefaults,
    weightsDir: String,
    sequenceLength: Int,
    explicitModelPath: String?,
    allowBootstrap: Bool
) throws -> String {
    try ensureStateDirectories(defaults)
    try migrateLegacyArtifactsIfNeeded(defaults)

    if let explicitModelPath, !explicitModelPath.isEmpty {
        let expanded = NSString(string: explicitModelPath).expandingTildeInPath
        guard FileManager().fileExists(atPath: expanded) else {
            throw CLIError.usage("Core ML model path does not exist: \(explicitModelPath)")
        }
        return URL(fileURLWithPath: expanded).standardizedFileURL.path
    }

    let weightsURL = URL(fileURLWithPath: weightsDir, isDirectory: true).standardizedFileURL
    let modelURL = defaults.coreMLModelURL(sequenceLength: sequenceLength, weightsDirURL: weightsURL)
    if FileManager().fileExists(atPath: modelURL.path) {
        return modelURL.path
    }

    if let legacyArtifactsRoot = defaults.legacyArtifactsRoot {
        let legacyModel = legacyArtifactsRoot
            .appendingPathComponent("gpt2_coreml", isDirectory: true)
            .appendingPathComponent("gpt2_seq\(sequenceLength).mlpackage")
        if FileManager().fileExists(atPath: legacyModel.path) {
            try copyItemIfMissing(from: legacyModel, to: modelURL)
            return modelURL.path
        }
    }

    guard allowBootstrap else {
        throw CLIError.usage(
            "Missing GPT-2 Core ML baseline for sequence length \(sequenceLength). Re-run without --no-bootstrap, run `espresso-generate doctor`, or pass --coreml-model."
        )
    }

    let scriptsDir = try requireScriptsDirectory(defaults)
    stderrLine("Exporting GPT-2 Core ML baseline for seq_len=\(sequenceLength)")
    let python = try ensurePythonEnvironment(
        defaults: defaults,
        requiredModules: ["numpy", "torch", "coremltools"]
    )
    try runProcessStreaming(
        executable: python,
        arguments: [
            scriptsDir.appendingPathComponent("export_gpt2_coreml.py").path,
            "--weights", weightsURL.path,
            "--output", modelURL.path,
            "--seq-len", String(sequenceLength),
        ],
        workingDirectory: defaults.workingDirectory,
        environment: pythonEnvironment(defaults: defaults)
    )
    return modelURL.path
}

func runGPT2CoreMLReference(
    defaults: DemoDefaults,
    coreMLModelPath: String,
    weightsDir: String,
    promptTokens: [UInt16],
    sequenceLength: Int,
    maxTokens: Int,
    temperature: Float,
    warmup: Int,
    iterations: Int,
    computeUnits: String,
    seed: Int,
    allowBootstrap: Bool
) throws -> CoreMLComparisonResult {
    guard !promptTokens.isEmpty else {
        throw CLIError.runtime("Cannot compare Core ML without prompt tokens.")
    }
    let scriptsDir = try requireScriptsDirectory(defaults)
    let python = try ensurePythonEnvironment(
        defaults: defaults,
        requiredModules: ["numpy", "coremltools"]
    )
    let promptTokenString = promptTokens.map(String.init).joined(separator: ",")
    let output = try runProcessCaptured(
        executable: python,
        arguments: [
            scriptsDir.appendingPathComponent("run_gpt2_coreml_reference.py").path,
            "--coreml-model", coreMLModelPath,
            "--weights", weightsDir,
            "--prompt-tokens", promptTokenString,
            "--seq-len", String(sequenceLength),
            "--max-tokens", String(maxTokens),
            "--temperature", String(temperature),
            "--warmup", String(warmup),
            "--iterations", String(iterations),
            "--seed", String(seed),
            "--compute-units", computeUnits,
        ],
        workingDirectory: defaults.workingDirectory,
        environment: pythonEnvironment(defaults: defaults),
        allowBootstrap: allowBootstrap
    )

    guard output.status == 0 else {
        throw CLIError.runtime(
            output.stderr.isEmpty
                ? "Core ML reference runner failed with status \(output.status)"
                : output.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
        )
    }

    let decoder = JSONDecoder()
    do {
        return try decoder.decode(CoreMLComparisonResult.self, from: Data(output.stdout.utf8))
    } catch {
        throw CLIError.runtime("Failed to decode Core ML comparison output: \(error)")
    }
}

func runGPT2CoreMLReferenceStreaming(
    defaults: DemoDefaults,
    coreMLModelPath: String,
    weightsDir: String,
    promptTokens: [UInt16],
    sequenceLength: Int,
    maxTokens: Int,
    temperature: Float,
    computeUnits: String,
    seed: Int,
    allowBootstrap: Bool,
    onEvent: @escaping (CoreMLStreamEvent) -> Void
) throws -> CoreMLComparisonResult {
    guard !promptTokens.isEmpty else {
        throw CLIError.runtime("Cannot compare Core ML without prompt tokens.")
    }
    let scriptsDir = try requireScriptsDirectory(defaults)
    let python = try ensurePythonEnvironment(
        defaults: defaults,
        requiredModules: ["numpy", "coremltools"]
    )
    let promptTokenString = promptTokens.map(String.init).joined(separator: ",")

    var completed: CoreMLComparisonResult?
    let output = try runProcessStreamingLinesCapture(
        executable: python,
        arguments: [
            scriptsDir.appendingPathComponent("run_gpt2_coreml_reference.py").path,
            "--coreml-model", coreMLModelPath,
            "--weights", weightsDir,
            "--prompt-tokens", promptTokenString,
            "--seq-len", String(sequenceLength),
            "--max-tokens", String(maxTokens),
            "--temperature", String(temperature),
            "--warmup", "0",
            "--iterations", "1",
            "--seed", String(seed),
            "--compute-units", computeUnits,
            "--emit-events",
        ],
        workingDirectory: defaults.workingDirectory,
        environment: pythonEnvironment(defaults: defaults),
        allowBootstrap: allowBootstrap,
        onStdoutLine: { line in
            guard !line.isEmpty else { return }
            do {
                let raw = try JSONDecoder().decode(RawCoreMLStreamEvent.self, from: Data(line.utf8))
                switch raw.type {
                case "compile":
                    if let compileTimeMs = raw.compileTimeMs,
                       let computeUnits = raw.computeUnits,
                       let seqLen = raw.seqLen
                    {
                        onEvent(.compile(compileTimeMs: compileTimeMs, computeUnits: computeUnits, seqLen: seqLen))
                    }
                case "token":
                    guard let token = raw.token,
                          let tokenIndex = raw.tokenIndex,
                          let elapsedMs = raw.elapsedMs,
                          let tokenLatencyMs = raw.tokenLatencyMs,
                          let tokensPerSecond = raw.tokensPerSecond,
                          token >= 0,
                          token <= Int(UInt16.max)
                    else {
                        return
                    }
                    onEvent(
                        .token(
                            token: UInt16(token),
                            tokenIndex: tokenIndex,
                            elapsedMs: elapsedMs,
                            tokenLatencyMs: tokenLatencyMs,
                            tokensPerSecond: tokensPerSecond
                        )
                    )
                case "completed":
                    let result = try JSONDecoder().decode(CoreMLComparisonResult.self, from: Data(line.utf8))
                    completed = result
                    onEvent(.completed(result))
                default:
                    break
                }
            } catch {
                stderrLine("espresso-generate warning: failed to parse Core ML stream event: \(error)")
            }
        }
    )

    guard output.status == 0 else {
        throw CLIError.runtime(
            output.stderr.isEmpty
                ? "Core ML reference runner failed with status \(output.status)"
                : output.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
        )
    }
    guard let completed else {
        throw CLIError.runtime("Core ML reference runner did not emit a completion event.")
    }
    return completed
}

private func locateRepoRoot(fileManager: FileManager) -> URL? {
    if let override = ProcessInfo.processInfo.environment["ESPRESSO_REPO_ROOT"], !override.isEmpty {
        let candidate = URL(fileURLWithPath: NSString(string: override).expandingTildeInPath, isDirectory: true)
        if isRepoRoot(candidate, fileManager: fileManager) {
            return candidate.standardizedFileURL
        }
    }

    let currentDirectory = URL(fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true).standardizedFileURL
    if let resolved = ascendToRepoRoot(start: currentDirectory, fileManager: fileManager) {
        return resolved
    }

    if let executableURL = Bundle.main.executableURL?.deletingLastPathComponent().standardizedFileURL,
       let resolved = ascendToRepoRoot(start: executableURL, fileManager: fileManager)
    {
        return resolved
    }

    let argv0 = CommandLine.arguments.first ?? ""
    if !argv0.isEmpty {
        let executableURL = URL(fileURLWithPath: NSString(string: argv0).expandingTildeInPath)
            .deletingLastPathComponent()
            .standardizedFileURL
        if let resolved = ascendToRepoRoot(start: executableURL, fileManager: fileManager) {
            return resolved
        }
    }

    return nil
}

private func ascendToRepoRoot(start: URL, fileManager: FileManager) -> URL? {
    var current = start
    while true {
        if isRepoRoot(current, fileManager: fileManager) {
            return current
        }
        let parent = current.deletingLastPathComponent()
        if parent.path == current.path {
            return nil
        }
        current = parent
    }
}

private func isRepoRoot(_ candidate: URL, fileManager: FileManager) -> Bool {
    var isDirectory: ObjCBool = false
    let scriptsDir = candidate.appendingPathComponent("scripts", isDirectory: true)
    return fileManager.fileExists(atPath: candidate.appendingPathComponent("Package.swift").path) &&
        fileManager.fileExists(atPath: scriptsDir.path, isDirectory: &isDirectory) &&
        isDirectory.boolValue &&
        fileManager.fileExists(atPath: scriptsDir.appendingPathComponent("bootstrap_gpt2_demo.py").path) &&
        fileManager.fileExists(atPath: scriptsDir.appendingPathComponent("export_gpt2_coreml.py").path) &&
        fileManager.fileExists(atPath: scriptsDir.appendingPathComponent("run_gpt2_coreml_reference.py").path)
}

private func preferredStateRoot(fileManager: FileManager) -> URL {
    if let override = ProcessInfo.processInfo.environment["ESPRESSO_HOME"], !override.isEmpty {
        return URL(fileURLWithPath: NSString(string: override).expandingTildeInPath, isDirectory: true).standardizedFileURL
    }
    let base = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first ??
        URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support", isDirectory: true)
    return base.appendingPathComponent("Espresso", isDirectory: true)
}

private func preferredCacheRoot(fileManager: FileManager) -> URL {
    if let override = ProcessInfo.processInfo.environment["ESPRESSO_CACHE_HOME"], !override.isEmpty {
        return URL(fileURLWithPath: NSString(string: override).expandingTildeInPath, isDirectory: true).standardizedFileURL
    }
    let base = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first ??
        URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Caches", isDirectory: true)
    return base.appendingPathComponent("Espresso", isDirectory: true)
}

private func resolvedScriptsDirectory(repoRoot: URL?) -> URL? {
    if let override = ProcessInfo.processInfo.environment["ESPRESSO_SCRIPTS_DIR"], !override.isEmpty {
        return URL(fileURLWithPath: NSString(string: override).expandingTildeInPath, isDirectory: true).standardizedFileURL
    }
    return repoRoot?.appendingPathComponent("scripts", isDirectory: true)
}

private func requireScriptsDirectory(_ defaults: DemoDefaults) throws -> URL {
    guard let scriptsDir = defaults.scriptsDir, defaults.scriptsAvailable else {
        throw CLIError.runtime(
            "Espresso helper scripts are unavailable. Run from a repository checkout or set ESPRESSO_SCRIPTS_DIR to the scripts directory."
        )
    }
    return scriptsDir
}

private func ensureStateDirectories(_ defaults: DemoDefaults) throws {
    let fileManager = FileManager()
    for directory in [
        defaults.stateRoot,
        defaults.cacheRoot,
        defaults.reportsRoot,
        defaults.hfCacheRoot,
        defaults.weightsDir.deletingLastPathComponent(),
        defaults.tokenizerDir.deletingLastPathComponent(),
        defaults.coreMLDir,
        defaults.toolsVenvDir.deletingLastPathComponent(),
    ] {
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true, attributes: nil)
    }
}

private func migrateLegacyArtifactsIfNeeded(_ defaults: DemoDefaults) throws {
    guard let legacyArtifactsRoot = defaults.legacyArtifactsRoot else {
        return
    }
    let legacyWeights = legacyArtifactsRoot.appendingPathComponent("gpt2_124m", isDirectory: true)
    let legacyTokenizer = legacyArtifactsRoot.appendingPathComponent("gpt2_tokenizer", isDirectory: true)
    let legacyCoreML = legacyArtifactsRoot.appendingPathComponent("gpt2_coreml", isDirectory: true)

    if FileManager().fileExists(atPath: legacyWeights.path),
       !FileManager().fileExists(atPath: defaults.weightsDir.path)
    {
        try copyItemIfMissing(from: legacyWeights, to: defaults.weightsDir)
    }
    if FileManager().fileExists(atPath: legacyTokenizer.path),
       !FileManager().fileExists(atPath: defaults.tokenizerDir.path)
    {
        try copyItemIfMissing(from: legacyTokenizer, to: defaults.tokenizerDir)
    }
    if FileManager().fileExists(atPath: legacyCoreML.path),
       !FileManager().fileExists(atPath: defaults.coreMLDir.path)
    {
        try copyItemIfMissing(from: legacyCoreML, to: defaults.coreMLDir)
    }
}

private func copyItemIfMissing(from source: URL, to destination: URL) throws {
    let fileManager = FileManager()
    if fileManager.fileExists(atPath: destination.path) {
        return
    }
    try fileManager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true, attributes: nil)
    try fileManager.copyItem(at: source, to: destination)
}

private func ensurePythonEnvironment(
    defaults: DemoDefaults,
    requiredModules: [String]
) throws -> String {
    if let override = ProcessInfo.processInfo.environment["ESPRESSO_TOOLS_PYTHON"], !override.isEmpty {
        if try pythonSupportsModules(override, requiredModules: requiredModules, defaults: defaults) {
            return override
        }
        throw CLIError.runtime("ESPRESSO_TOOLS_PYTHON is set but missing required modules: \(requiredModules.joined(separator: ", "))")
    }

    let bootstrapPython = try preferredBootstrapPython(defaults: defaults)
    let versionTag = try pythonVersionTag(bootstrapPython, defaults: defaults)
    let managedVenvDir = defaults.toolsVenvDir.deletingLastPathComponent()
        .appendingPathComponent("gpt2-tools-\(versionTag)", isDirectory: true)
    let managedPython = managedVenvDir.appendingPathComponent("bin/python3").path
    if try pythonSupportsModules(managedPython, requiredModules: requiredModules, defaults: defaults) {
        return managedPython
    }

    let fileManager = FileManager()
    try fileManager.createDirectory(
        at: managedVenvDir.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    if !fileManager.fileExists(atPath: managedPython) {
        stderrLine("Creating managed Python environment at \(managedVenvDir.path)")
        try runProcessStreaming(
            executable: bootstrapPython,
            arguments: ["-m", "venv", managedVenvDir.path],
            workingDirectory: defaults.workingDirectory,
            environment: pythonEnvironment(defaults: defaults)
        )
    }

    try runProcessStreaming(
        executable: managedPython,
        arguments: ["-m", "pip", "install", "--upgrade", "pip"],
        workingDirectory: defaults.workingDirectory,
        environment: pythonEnvironment(defaults: defaults)
    )

    let packages = packagesForModules(requiredModules)
    stderrLine("Installing Python packages: \(packages.joined(separator: ", "))")
    try runProcessStreaming(
        executable: managedPython,
        arguments: ["-m", "pip", "install"] + packages,
        workingDirectory: defaults.workingDirectory,
        environment: pythonEnvironment(defaults: defaults)
    )

    guard try pythonSupportsModules(managedPython, requiredModules: requiredModules, defaults: defaults) else {
        throw CLIError.runtime("Managed Python environment is still missing required modules after installation.")
    }
    return managedPython
}

func preferredBootstrapPython(defaults: DemoDefaults) throws -> String {
    if let override = ProcessInfo.processInfo.environment["ESPRESSO_BOOTSTRAP_PYTHON"], !override.isEmpty {
        return override
    }

    for candidate in ["python3.13", "python3.12", "python3"] {
        let output = try runProcessCaptured(
            executable: candidate,
            arguments: ["--version"],
            workingDirectory: defaults.workingDirectory,
            environment: pythonEnvironment(defaults: defaults),
            allowBootstrap: true
        )
        if output.status == 0 {
            return candidate
        }
    }

    throw CLIError.runtime("Unable to find python3.13, python3.12, or python3 for GPT-2 demo setup.")
}

private func pythonVersionTag(_ executable: String, defaults: DemoDefaults) throws -> String {
    let output = try runProcessCaptured(
        executable: executable,
        arguments: ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        workingDirectory: defaults.workingDirectory,
        environment: pythonEnvironment(defaults: defaults),
        allowBootstrap: true
    )
    guard output.status == 0 else {
        throw CLIError.runtime("Unable to determine Python version for \(executable)")
    }
    let trimmed = output.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
    return trimmed.replacingOccurrences(of: ".", with: "_")
}

private func packagesForModules(_ modules: [String]) -> [String] {
    var packages: Set<String> = ["numpy"]
    for module in modules {
        switch module {
        case "numpy":
            packages.insert("numpy")
        case "torch":
            packages.insert("torch")
        case "transformers":
            packages.insert("transformers")
        case "coremltools":
            packages.insert("coremltools")
        default:
            packages.insert(module)
        }
    }
    return packages.sorted()
}

private func pythonSupportsModules(
    _ executable: String,
    requiredModules: [String],
    defaults: DemoDefaults
) throws -> Bool {
    let fileManager = FileManager()
    if executable.contains("/"), !fileManager.fileExists(atPath: executable) {
        return false
    }
    let command = "import " + requiredModules.joined(separator: ", ")
    let output = try runProcessCaptured(
        executable: executable,
        arguments: ["-c", command],
        workingDirectory: defaults.workingDirectory,
        environment: pythonEnvironment(defaults: defaults),
        allowBootstrap: true
    )
    return output.status == 0
}

private func pythonEnvironment(defaults: DemoDefaults) -> [String: String] {
    var environment = ProcessInfo.processInfo.environment
    environment["HF_HOME"] = defaults.hfCacheRoot.path
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment
}

private func runProcessStreaming(
    executable: String,
    arguments: [String],
    workingDirectory: URL,
    environment: [String: String]
) throws {
    let process = configuredProcess(
        executable: executable,
        arguments: arguments,
        workingDirectory: workingDirectory,
        environment: environment
    )
    process.standardOutput = FileHandle.standardError
    process.standardError = FileHandle.standardError
    try process.run()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
        throw CLIError.runtime("Process failed (\(process.terminationStatus)): \(executable) \(arguments.joined(separator: " "))")
    }
}

private func runProcessCaptured(
    executable: String,
    arguments: [String],
    workingDirectory: URL,
    environment: [String: String],
    allowBootstrap: Bool
) throws -> ProcessOutput {
    let process = configuredProcess(
        executable: executable,
        arguments: arguments,
        workingDirectory: workingDirectory,
        environment: environment
    )
    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe
    do {
        try process.run()
    } catch {
        if allowBootstrap {
            throw error
        }
        throw CLIError.runtime("\(error)")
    }
    process.waitUntilExit()
    let stdout = String(decoding: stdoutPipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
    let stderr = String(decoding: stderrPipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
    return ProcessOutput(status: process.terminationStatus, stdout: stdout, stderr: stderr)
}

private func runProcessStreamingLinesCapture(
    executable: String,
    arguments: [String],
    workingDirectory: URL,
    environment: [String: String],
    allowBootstrap: Bool,
    onStdoutLine: @escaping (String) -> Void
) throws -> ProcessOutput {
    let process = configuredProcess(
        executable: executable,
        arguments: arguments,
        workingDirectory: workingDirectory,
        environment: environment
    )
    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe

    let stdoutAccumulator = LockedStringBuffer()
    let stderrAccumulator = LockedStringBuffer()
    let stdoutParser = LineStreamParser { line in
        stdoutAccumulator.append(line + "\n")
        onStdoutLine(line)
    }
    let stderrParser = LineStreamParser { line in
        stderrAccumulator.append(line + "\n")
        stderrLine(line)
    }

    stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
        let data = handle.availableData
        if data.isEmpty {
            return
        }
        stdoutParser.append(data)
    }
    stderrPipe.fileHandleForReading.readabilityHandler = { handle in
        let data = handle.availableData
        if data.isEmpty {
            return
        }
        stderrParser.append(data)
    }

    do {
        try process.run()
    } catch {
        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        stderrPipe.fileHandleForReading.readabilityHandler = nil
        if allowBootstrap {
            throw error
        }
        throw CLIError.runtime("\(error)")
    }

    process.waitUntilExit()
    stdoutPipe.fileHandleForReading.readabilityHandler = nil
    stderrPipe.fileHandleForReading.readabilityHandler = nil
    stdoutParser.finish()
    stderrParser.finish()

    return ProcessOutput(
        status: process.terminationStatus,
        stdout: stdoutAccumulator.value,
        stderr: stderrAccumulator.value
    )
}

private func configuredProcess(
    executable: String,
    arguments: [String],
    workingDirectory: URL,
    environment: [String: String]
) -> Process {
    let process = Process()
    if executable.contains("/") {
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
    } else {
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [executable] + arguments
    }
    process.currentDirectoryURL = workingDirectory
    process.environment = environment
    return process
}

private final class LockedStringBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var storage = ""

    func append(_ text: String) {
        lock.lock()
        storage.append(text)
        lock.unlock()
    }

    var value: String {
        lock.lock()
        let snapshot = storage
        lock.unlock()
        return snapshot
    }
}

private final class LineStreamParser: @unchecked Sendable {
    private let lock = NSLock()
    private let onLine: (String) -> Void
    private var buffer = Data()

    init(onLine: @escaping (String) -> Void) {
        self.onLine = onLine
    }

    func append(_ data: Data) {
        lock.lock()
        buffer.append(data)
        emitLockedLines()
        lock.unlock()
    }

    func finish() {
        lock.lock()
        if !buffer.isEmpty {
            let line = String(decoding: buffer, as: UTF8.self).trimmingCharacters(in: .newlines)
            if !line.isEmpty {
                onLine(line)
            }
            buffer.removeAll(keepingCapacity: false)
        }
        lock.unlock()
    }

    private func emitLockedLines() {
        while let newlineIndex = buffer.firstIndex(of: 0x0A) {
            let lineData = buffer[..<newlineIndex]
            buffer.removeSubrange(...newlineIndex)
            let line = String(decoding: lineData, as: UTF8.self).trimmingCharacters(in: .newlines)
            onLine(line)
        }
    }
}

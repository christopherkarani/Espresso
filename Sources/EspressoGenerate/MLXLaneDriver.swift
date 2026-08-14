import Foundation

enum MLXStreamEvent: Equatable, Sendable {
    case hello(precision: String, quantized: Bool, repo: String)
    case compile(ms: Double)
    case token(text: String, tokenIndex: Int, elapsedMs: Double, tokenLatencyMs: Double, tokensPerSecond: Double)
    case completed(text: String, compileMs: Double, ttftMs: Double, tokensPerSecond: Double, tokenCount: Int)
    case ready
    case error(String)
}

struct MLXCompletion: Equatable, Sendable {
    var text: String
    var compileMs: Double
    var ttftMs: Double
    var tokensPerSecond: Double
    var tokenCount: Int
}

func parseMLXStreamEvent(_ line: String) throws -> MLXStreamEvent? {
    let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else {
        return nil
    }
    guard let data = trimmed.data(using: .utf8) else {
        return nil
    }
    let object = try JSONSerialization.jsonObject(with: data)
    guard let payload = object as? [String: Any], let type = payload["type"] as? String else {
        throw CLIError.runtime("MLX stream event is not a JSON object: \(trimmed)")
    }
    switch type {
    case "hello":
        return .hello(
            precision: stringValue(payload["precision"]) ?? "unknown",
            quantized: boolValue(payload["quantized"]) ?? false,
            repo: stringValue(payload["repo"]) ?? ""
        )
    case "compile":
        return .compile(ms: doubleValue(payload["compile_time_ms"]) ?? 0)
    case "token":
        return .token(
            text: stringValue(payload["text"]) ?? "",
            tokenIndex: intValue(payload["token_index"]) ?? 0,
            elapsedMs: doubleValue(payload["elapsed_ms"]) ?? 0,
            tokenLatencyMs: doubleValue(payload["token_latency_ms"]) ?? 0,
            tokensPerSecond: doubleValue(payload["tokens_per_second"]) ?? 0
        )
    case "completed":
        return .completed(
            text: stringValue(payload["text"]) ?? "",
            compileMs: doubleValue(payload["compile_time_ms"]) ?? 0,
            ttftMs: doubleValue(payload["first_token_latency_ms"]) ?? 0,
            tokensPerSecond: doubleValue(payload["tokens_per_second"]) ?? 0,
            tokenCount: intValue(payload["generation_tokens"]) ?? 0
        )
    case "ready":
        return .ready
    case "error":
        return .error(stringValue(payload["message"]) ?? trimmed)
    default:
        return nil
    }
}

func requireMLXPython(
    defaults: DemoDefaults,
    environment: [String: String] = ProcessInfo.processInfo.environment
) throws -> String {
    let candidates = mlxPythonCandidates(environment: environment)
    if let python = resolveMLXPython(candidates: candidates, canImport: { candidate in
        mlxPythonCanImport(candidate, defaults: defaults)
    }) {
        return python
    }
    throw CLIError.runtime(mlxInstallInstructions())
}

private struct MLXLaneLaunch: Sendable {
    let python: String
    let script: URL
    let fairness: ChatVsMLXFairness
    let modelPath: String
    let workingDirectory: URL
    let environment: [String: String]
}

final class MLXLaneSession: @unchecked Sendable {
    private let launch: MLXLaneLaunch
    private var process: Process?
    private var stdinHandle: FileHandle?
    private let stdoutParser = MLXLineBuffer()
    private let lock = NSLock()
    private var pending = [MLXStreamEvent]()
    private var stderrText = ""
    private var hello: MLXStreamEvent?
    private var closed = false
    private(set) var loadedPrecision: String = "fp16"

    static func start(
        python: String,
        script: URL,
        fairness: ChatVsMLXFairness,
        modelPath: String,
        workingDirectory: URL,
        environment: [String: String]
    ) throws -> MLXLaneSession {
        let session = MLXLaneSession(
            launch: MLXLaneLaunch(
                python: python,
                script: script,
                fairness: fairness,
                modelPath: modelPath,
                workingDirectory: workingDirectory,
                environment: environment
            )
        )
        try session.launchProcess()
        return session
    }

    private init(launch: MLXLaneLaunch) {
        self.launch = launch
    }

    func generate(
        prompt: String,
        maxTokens: Int,
        isCancelled: () -> Bool,
        onEvent: (MLXStreamEvent) -> Void
    ) throws -> MLXCompletion {
        try ensureRunning()
        guard let stdinHandle else {
            throw CLIError.runtime("MLX lane stdin is closed.")
        }
        let payload: [String: Any] = [
            "prompt": prompt,
            "max_tokens": maxTokens,
        ]
        let data = try JSONSerialization.data(withJSONObject: payload)
        stdinHandle.write(data)
        stdinHandle.write(Data("\n".utf8))

        var completion: MLXCompletion?
        while completion == nil {
            if isCancelled() {
                interruptCurrentGenerate()
                throw CLIError.runtime("MLX generate cancelled")
            }
            if process?.isRunning != true {
                throw CLIError.runtime(mlxFailureMessage())
            }
            guard let event = waitForEvent(timeout: 0.05) else {
                continue
            }
            switch event {
            case .ready, .hello:
                continue
            case let .error(message):
                throw CLIError.runtime(message)
            case let .completed(text, compileMs, ttftMs, tokensPerSecond, tokenCount):
                completion = MLXCompletion(
                    text: text,
                    compileMs: compileMs,
                    ttftMs: ttftMs,
                    tokensPerSecond: tokensPerSecond,
                    tokenCount: tokenCount
                )
                onEvent(event)
            default:
                onEvent(event)
            }
        }
        return completion!
    }

    func close() {
        lock.lock()
        closed = true
        lock.unlock()
        interruptCurrentGenerate()
    }

    private func ensureRunning() throws {
        lock.lock()
        let isClosed = closed
        let running = process?.isRunning == true
        lock.unlock()
        if isClosed {
            throw CLIError.runtime("MLX lane is closed")
        }
        if running {
            return
        }
        try launchProcess()
    }

    private func launchProcess() throws {
        interruptCurrentGenerate()
        lock.lock()
        pending.removeAll()
        stderrText = ""
        hello = nil
        lock.unlock()

        let process = Process()
        if launch.python.contains("/") {
            process.executableURL = URL(fileURLWithPath: launch.python)
            process.arguments = [launch.script.path]
        } else {
            process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            process.arguments = [launch.python, launch.script.path]
        }
        var arguments = process.arguments ?? []
        arguments.append(contentsOf: [
            "--repo", launch.fairness.huggingfaceRepo,
            "--model-path", launch.modelPath,
            "--max-tokens", String(launch.fairness.maxNewTokens),
        ])
        if case .quantized = launch.fairness.mlxQuantization {
            arguments.append("--allow-quant")
        }
        process.arguments = arguments
        process.currentDirectoryURL = launch.workingDirectory
        process.environment = launch.environment

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            self?.ingest(handle.availableData)
        }
        stderrPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            self?.ingestStderr(handle.availableData)
        }

        do {
            try process.run()
        } catch {
            throw CLIError.runtime("Failed to start MLX lane: \(error)\n\(mlxInstallInstructions())")
        }

        lock.lock()
        self.process = process
        self.stdinHandle = stdinPipe.fileHandleForWriting
        lock.unlock()

        let hello = try waitForHello()
        if case let .hello(precision, quantized, repo) = hello {
            try assertMLXLoadMatchesFairness(
                quantized: quantized,
                precision: precision,
                repo: repo,
                fairness: launch.fairness
            )
            loadedPrecision = precision
        }
    }

    private func interruptCurrentGenerate() {
        lock.lock()
        let process = self.process
        let stdinHandle = self.stdinHandle
        self.process = nil
        self.stdinHandle = nil
        pending.removeAll()
        lock.unlock()
        if let process, process.isRunning {
            process.terminate()
            process.waitUntilExit()
        }
        stdinHandle?.closeFile()
    }

    private func waitForHello() throws -> MLXStreamEvent {
        let deadline = Date().addingTimeInterval(600)
        while Date() < deadline {
            if process?.isRunning != true, hello == nil, pending.isEmpty {
                throw CLIError.runtime(mlxFailureMessage())
            }
            if let event = waitForEvent(timeout: 0.1) {
                if case .hello = event {
                    hello = event
                    return event
                }
                if case let .error(message) = event {
                    throw CLIError.runtime(message)
                }
            }
        }
        throw CLIError.runtime("MLX lane did not become ready.\n\(mlxInstallInstructions())")
    }

    private func waitForEvent(timeout: TimeInterval) -> MLXStreamEvent? {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            lock.lock()
            if !pending.isEmpty {
                let event = pending.removeFirst()
                lock.unlock()
                return event
            }
            lock.unlock()
            Thread.sleep(forTimeInterval: 0.01)
        }
        lock.lock()
        let event = pending.isEmpty ? nil : pending.removeFirst()
        lock.unlock()
        return event
    }

    private func ingest(_ data: Data) {
        guard !data.isEmpty else { return }
        let lines = stdoutParser.append(data)
        lock.lock()
        for line in lines {
            if let event = try? parseMLXStreamEvent(line) {
                pending.append(event)
            }
        }
        lock.unlock()
    }

    private func ingestStderr(_ data: Data) {
        guard !data.isEmpty else { return }
        let text = String(decoding: data, as: UTF8.self)
        lock.lock()
        stderrText.append(text)
        lock.unlock()
    }

    private func mlxFailureMessage() -> String {
        lock.lock()
        let stderr = stderrText.trimmingCharacters(in: .whitespacesAndNewlines)
        lock.unlock()
        if stderr.lowercased().contains("modulenotfounderror") || stderr.lowercased().contains("no module named") {
            return mlxInstallInstructions()
        }
        if stderr.isEmpty {
            return "MLX lane exited early.\n\(mlxInstallInstructions())"
        }
        return stderr
    }
}

func mlxStreamScriptURL(defaults: DemoDefaults) throws -> URL {
    guard let scriptsDir = defaults.scriptsDir else {
        throw CLIError.runtime("helper scripts directory is unavailable; cannot start the MLX lane.")
    }
    let script = scriptsDir.appendingPathComponent("mlx_qwen_stream.py")
    guard FileManager().fileExists(atPath: script.path) else {
        throw CLIError.runtime("Missing \(script.path)")
    }
    return script
}

private func mlxPythonCanImport(_ executable: String, defaults: DemoDefaults) -> Bool {
    let process = Process()
    if executable.contains("/") {
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = ["-c", "import mlx_lm"]
    } else {
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [executable, "-c", "import mlx_lm"]
    }
    process.currentDirectoryURL = defaults.workingDirectory
    process.standardOutput = Pipe()
    process.standardError = Pipe()
    do {
        try process.run()
    } catch {
        return false
    }
    process.waitUntilExit()
    return process.terminationStatus == 0
}

private final class MLXLineBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer = Data()

    func append(_ data: Data) -> [String] {
        lock.lock()
        defer { lock.unlock() }
        buffer.append(data)
        var lines: [String] = []
        while let newline = buffer.firstIndex(of: 0x0A) {
            let lineData = buffer[..<newline]
            buffer.removeSubrange(...newline)
            let line = String(decoding: lineData, as: UTF8.self).trimmingCharacters(in: .newlines)
            if !line.isEmpty {
                lines.append(line)
            }
        }
        return lines
    }
}

private func stringValue(_ value: Any?) -> String? {
    value as? String
}

private func boolValue(_ value: Any?) -> Bool? {
    if let bool = value as? Bool {
        return bool
    }
    if let number = value as? NSNumber {
        return number.boolValue
    }
    return nil
}

private func doubleValue(_ value: Any?) -> Double? {
    if let number = value as? NSNumber {
        return number.doubleValue
    }
    if let string = value as? String {
        return Double(string)
    }
    return nil
}

private func intValue(_ value: Any?) -> Int? {
    if let number = value as? NSNumber {
        return number.intValue
    }
    if let string = value as? String {
        return Int(string)
    }
    return nil
}

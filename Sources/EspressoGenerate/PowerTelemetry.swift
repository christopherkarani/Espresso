import Foundation

struct PowerSample: Sendable {
    let timestamp: Date
    let packageW: Double
    let cpuW: Double
    let gpuW: Double
    let aneW: Double
}

struct PowerSummary: Sendable {
    let packageW: Double
    let cpuW: Double
    let gpuW: Double
    let aneW: Double
    let sampleCount: Int

    static let unavailable = PowerSummary(packageW: 0, cpuW: 0, gpuW: 0, aneW: 0, sampleCount: 0)
}

struct PowerCapability: Sendable {
    let available: Bool
    let message: String
}

final class PowerTelemetryCollector: @unchecked Sendable {
    private let sampleIntervalMs: Int
    private let outputBuffer = LockedPowerBuffer()
    private let parser = PowermetricsParser()
    private var process: Process?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?

    init(sampleIntervalMs: Int = 1_000) {
        self.sampleIntervalMs = sampleIntervalMs
    }

    static func capability() -> PowerCapability {
        let fileManager = FileManager()
        guard fileManager.fileExists(atPath: "/usr/bin/powermetrics") else {
            return PowerCapability(available: false, message: "powermetrics is unavailable on this host")
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/sudo")
        process.arguments = ["-n", "/usr/bin/true"]
        let stderrPipe = Pipe()
        process.standardError = stderrPipe
        do {
            try process.run()
        } catch {
            return PowerCapability(available: false, message: "powermetrics requires sudo and could not be probed: \(error)")
        }
        process.waitUntilExit()
        if process.terminationStatus == 0 {
            return PowerCapability(available: true, message: "powermetrics ready")
        }
        let stderr = String(decoding: stderrPipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let message = stderr.isEmpty ? "powermetrics requires passwordless sudo or root" : stderr
        return PowerCapability(available: false, message: message)
    }

    func start(onSample: @escaping @Sendable (PowerSample) -> Void) throws {
        guard process == nil else {
            return
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/sudo")
        process.arguments = [
            "-n",
            "/usr/bin/powermetrics",
            "--samplers", "cpu_power,gpu_power,ane_power",
            "--sample-interval", String(sampleIntervalMs),
        ]
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        stdoutPipe.fileHandleForReading.readabilityHandler = { [parser, outputBuffer, onSample] handle in
            let data = handle.availableData
            if data.isEmpty {
                return
            }
            parser.append(data) { sample in
                outputBuffer.append(sample)
                onSample(sample)
            }
        }
        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            let text = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty {
                stderrLine("powermetrics: \(text)")
            }
        }

        try process.run()
        self.process = process
        self.stdoutPipe = stdoutPipe
        self.stderrPipe = stderrPipe
    }

    func stop() -> PowerSummary {
        guard let process else {
            return outputBuffer.summary
        }
        process.terminate()
        process.waitUntilExit()
        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        stderrPipe?.fileHandleForReading.readabilityHandler = nil
        parser.finish { [self] sample in
            self.outputBuffer.append(sample)
        }
        self.process = nil
        self.stdoutPipe = nil
        self.stderrPipe = nil
        return outputBuffer.summary
    }
}

func parsePowermetricsSamples(from text: String) -> [PowerSample] {
    let parser = PowermetricsParser()
    var samples: [PowerSample] = []
    parser.append(Data(text.utf8)) { sample in
        samples.append(sample)
    }
    parser.finish { sample in
        samples.append(sample)
    }
    return samples
}

private final class LockedPowerBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var samples: [PowerSample] = []

    func append(_ sample: PowerSample) {
        lock.lock()
        samples.append(sample)
        lock.unlock()
    }

    var summary: PowerSummary {
        lock.lock()
        let snapshot = samples
        lock.unlock()
        guard !snapshot.isEmpty else {
            return .unavailable
        }
        let count = Double(snapshot.count)
        return PowerSummary(
            packageW: snapshot.reduce(0) { $0 + $1.packageW } / count,
            cpuW: snapshot.reduce(0) { $0 + $1.cpuW } / count,
            gpuW: snapshot.reduce(0) { $0 + $1.gpuW } / count,
            aneW: snapshot.reduce(0) { $0 + $1.aneW } / count,
            sampleCount: snapshot.count
        )
    }
}

private final class PowermetricsParser: @unchecked Sendable {
    private let lock = NSLock()
    private var buffer = ""
    private let powerPattern = try! NSRegularExpression(
        pattern: #"(?im)^\s*(CPU Power|GPU Power|ANE Power|Package Power|Combined Power|SoC Power)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(mW|W)\s*$"#
    )

    func append(_ data: Data, onSample: @escaping (PowerSample) -> Void) {
        guard let text = String(data: data, encoding: .utf8) else {
            return
        }
        lock.lock()
        buffer.append(text)
        emitLockedSamples(onSample: onSample)
        lock.unlock()
    }

    func finish(onSample: @escaping (PowerSample) -> Void) {
        lock.lock()
        if let sample = parseSample(from: buffer) {
            onSample(sample)
        }
        buffer.removeAll(keepingCapacity: false)
        lock.unlock()
    }

    private func emitLockedSamples(onSample: @escaping (PowerSample) -> Void) {
        while let separatorRange = buffer.range(of: "\n\n") {
            let chunk = String(buffer[..<separatorRange.lowerBound])
            buffer.removeSubrange(..<separatorRange.upperBound)
            if let sample = parseSample(from: chunk) {
                onSample(sample)
            }
        }
    }

    private func parseSample(from text: String) -> PowerSample? {
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        let matches = powerPattern.matches(in: text, range: range)
        guard !matches.isEmpty else {
            return nil
        }

        var cpuW = 0.0
        var gpuW = 0.0
        var aneW = 0.0
        var packageW: Double?

        for match in matches {
            guard let nameRange = Range(match.range(at: 1), in: text),
                  let valueRange = Range(match.range(at: 2), in: text),
                  let unitRange = Range(match.range(at: 3), in: text),
                  let rawValue = Double(text[valueRange])
            else {
                continue
            }
            let name = String(text[nameRange]).lowercased()
            let unit = String(text[unitRange]).lowercased()
            let watts = unit == "mw" ? rawValue / 1_000.0 : rawValue
            switch name {
            case "cpu power":
                cpuW = watts
            case "gpu power":
                gpuW = watts
            case "ane power":
                aneW = watts
            case "package power", "combined power", "soc power":
                packageW = watts
            default:
                break
            }
        }

        let resolvedPackage = packageW ?? (cpuW + gpuW + aneW)
        return PowerSample(timestamp: Date(), packageW: resolvedPackage, cpuW: cpuW, gpuW: gpuW, aneW: aneW)
    }
}

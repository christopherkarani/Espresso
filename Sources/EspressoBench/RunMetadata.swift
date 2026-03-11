import Foundation
import Espresso

enum RunMetadata {
    static func base(mode: String, options: BenchmarkOptions) -> [String: Any] {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return [
            "schema_version": 2,
            "timestamp": formatter.string(from: Date()),
            "mode": mode,
            "git_sha": gitSHA() ?? "unknown",
            "device": [
                "chip": ResultsFormatter.chipName(),
                "os": ProcessInfo.processInfo.operatingSystemVersionString,
            ],
            "ane_effective_options": ANEOptionSnapshot
                .fromEnvironment(ProcessInfo.processInfo.environment)
                .asJSON(),
            "options": [
                "warmup": options.warmup,
                "iterations": options.iterations,
                "layers": options.layers,
                "sustained": options.sustained,
                "output_directory": options.outputDirectory ?? "",
                "model_path": options.modelPath,
                "ane_only": options.aneOnly,
            ],
            "env": envSnapshot(),
        ]
    }

    static func writeJSON(_ object: [String: Any], to path: String) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))
    }

    private static func envSnapshot() -> [String: String] {
        let env = ProcessInfo.processInfo.environment
        let prefixes = ["ANE_", "ESPRESSO_"]
        var out: [String: String] = [:]
        for (key, value) in env {
            if prefixes.contains(where: { key.hasPrefix($0) }) {
                out[key] = value
            }
        }
        return out
    }

    private static func gitSHA() -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["git", "rev-parse", "HEAD"]
        process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

        let out = Pipe()
        let err = Pipe()
        process.standardOutput = out
        process.standardError = err

        do {
            try process.run()
            process.waitUntilExit()
            guard process.terminationStatus == 0 else { return nil }
            let data = out.fileHandleForReading.readDataToEndOfFile()
            let text = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
            return text?.isEmpty == false ? text : nil
        } catch {
            return nil
        }
    }
}

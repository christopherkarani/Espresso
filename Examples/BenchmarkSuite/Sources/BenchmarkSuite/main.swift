/// BenchmarkSuite — compare Espresso vs CoreML throughput side-by-side.
///
/// Usage:
///   swift run BenchmarkSuite
///   ESPRESSO_WEIGHTS_DIR=~/.espresso/models/gpt2 swift run BenchmarkSuite
///
/// This wraps `espresso bench` for side-by-side comparison.
/// For the full interactive TUI use `./espresso compare`.

import Foundation

// ── Helpers ───────────────────────────────────────────────────────────────────

func pad(_ s: String, _ n: Int) -> String {
    s + String(repeating: " ", count: max(0, n - s.count))
}

@discardableResult
func shell(_ args: String...) -> Int32 {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    p.arguments = args
    try? p.run(); p.waitUntilExit()
    return p.terminationStatus
}

// ── Find espresso CLI ─────────────────────────────────────────────────────────

func findEspresso() -> String? {
    // Walk up from this file looking for ./espresso
    var url = URL(fileURLWithPath: #file)
    for _ in 0 ..< 10 {
        url = url.deletingLastPathComponent()
        let candidate = url.appendingPathComponent("espresso")
        if FileManager.default.fileExists(atPath: candidate.path) { return candidate.path }
    }
    // Fall back to PATH
    let task = Process()
    task.executableURL = URL(fileURLWithPath: "/usr/bin/which")
    task.arguments = ["espresso"]
    let pipe = Pipe()
    task.standardOutput = pipe
    try? task.run(); task.waitUntilExit()
    let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
        .trimmingCharacters(in: .whitespacesAndNewlines)
    return out?.isEmpty == false ? out : nil
}

// ── Main ──────────────────────────────────────────────────────────────────────

print("Espresso · BenchmarkSuite")
print("Compares Espresso ANE vs CoreML using the project's built-in bench command.\n")

guard let espresso = findEspresso() else {
    fputs("Error: could not locate the `espresso` script.\n", stderr)
    fputs("Run this example from within the Espresso checkout, or add the Espresso bin to your PATH.\n", stderr)
    exit(1)
}

let prompt = CommandLine.arguments.dropFirst().first ?? "The Neural Engine is"
print("Prompt: \"\(prompt)\"\n")

// Run bench mode — prints a machine-readable JSON summary to stdout
let status = shell(espresso, "bench", "--json", prompt)
if status != 0 {
    fputs("Benchmark failed (exit \(status)). Run `\(espresso) doctor` to diagnose.\n", stderr)
    exit(status)
}

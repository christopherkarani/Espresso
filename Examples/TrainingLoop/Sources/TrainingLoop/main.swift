/// TrainingLoop — fine-tune a small transformer on the ANE.
///
/// This example drives `espresso-train` on a local text corpus.
/// All heavy lifting (ANE compilation, forward/backward, Adam) is inside
/// the Espresso package; this file is the entry-point glue.
///
/// Usage:
///   swift run TrainingLoop
///   CORPUS_DIR=~/my-text swift run TrainingLoop
///
/// Requires the Espresso repo to be cloned locally so the espresso-train
/// binary can be built.  The first run will compile and cache ANE kernels;
/// subsequent runs reuse the system cache.

import Foundation

// ── Configuration ────────────────────────────────────────────────────────────

let corpusDir   = ProcessInfo.processInfo.environment["CORPUS_DIR"]
                  ?? FileManager.default.currentDirectoryPath
let steps       = ProcessInfo.processInfo.environment["TRAIN_STEPS"].flatMap(Int.init) ?? 100
let learningRate = ProcessInfo.processInfo.environment["LEARNING_RATE"] ?? "3e-4"
let layerCount  = ProcessInfo.processInfo.environment["LAYER_COUNT"].flatMap(Int.init) ?? 6
let outputPath  = ProcessInfo.processInfo.environment["CHECKPOINT_PATH"] ?? "checkpoint.bin"

// ── Find espresso-train binary ────────────────────────────────────────────────

func findBinary(named name: String) -> String? {
    var url = URL(fileURLWithPath: #file)
    for _ in 0 ..< 10 {
        url = url.deletingLastPathComponent()
        // Built by SPM in .build/release or .build/debug
        for config in ["release", "debug"] {
            let p = url.appendingPathComponent(".build/\(config)/\(name)").path
            if FileManager.default.fileExists(atPath: p) { return p }
        }
    }
    let t = Process()
    t.executableURL = URL(fileURLWithPath: "/usr/bin/which")
    t.arguments = [name]
    let pipe = Pipe()
    t.standardOutput = pipe
    try? t.run(); t.waitUntilExit()
    let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
        .trimmingCharacters(in: .whitespacesAndNewlines)
    return out?.isEmpty == false ? out : nil
}

// ── Build if needed ───────────────────────────────────────────────────────────

print("TrainingLoop — Espresso ANE fine-tuning example")
print(String(repeating: "─", count: 50))
print("Corpus  : \(corpusDir)")
print("Steps   : \(steps)")
print("LR      : \(learningRate)")
print("Layers  : \(layerCount)")
print("Output  : \(outputPath)\n")

var trainBin = findBinary(named: "espresso-train")
if trainBin == nil {
    print("Building espresso-train (first run)…")
    let build = Process()
    build.executableURL = URL(fileURLWithPath: "/usr/bin/swift")
    build.arguments = ["build", "-c", "release", "--product", "espresso-train"]
    build.currentDirectoryURL = URL(fileURLWithPath: #file)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    try? build.run(); build.waitUntilExit()
    trainBin = findBinary(named: "espresso-train")
}

guard let bin = trainBin else {
    fputs("Error: espresso-train binary not found.\n", stderr)
    fputs("Clone the Espresso repo and run `swift build -c release --product espresso-train`.\n", stderr)
    exit(1)
}

// ── Launch training ───────────────────────────────────────────────────────────

let train = Process()
train.executableURL = URL(fileURLWithPath: bin)
train.arguments = [
    "--total-steps", "\(steps)",
    "--lr", learningRate,
    "--artifact-layer-count", "\(layerCount)",
    "--local-text-dataset", corpusDir,
    "--local-bigram-prefix", "example-model",
    "--generation-model-export", outputPath,
]
print("Launching: \(bin)")
try? train.run(); train.waitUntilExit()

if train.terminationStatus == 0 {
    print("\nTraining complete. Checkpoint saved to: \(outputPath)")
} else {
    fputs("Training exited with status \(train.terminationStatus).\n", stderr)
    exit(train.terminationStatus)
}

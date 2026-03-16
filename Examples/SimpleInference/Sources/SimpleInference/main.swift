/// SimpleInference — generate text with Espresso on the Neural Engine.
///
/// Usage:
///   ESPRESSO_WEIGHTS_DIR=~/.espresso/models/gpt2 swift run SimpleInference
///   ESPRESSO_WEIGHTS_DIR=~/.espresso/models/gpt2 swift run SimpleInference "The sky is"
///
/// Run `./espresso install` from the Espresso repo first to download GPT-2 weights.

import Foundation
import RealModelInference
import ModelSupport

// ── Configuration ─────────────────────────────────────────────────────────────

let weightsDir = ProcessInfo.processInfo.environment["ESPRESSO_WEIGHTS_DIR"]
                 ?? (NSHomeDirectory() + "/.espresso/models/gpt2")
let prompt     = CommandLine.arguments.dropFirst().first ?? "The Neural Engine is"
let maxTokens  = 64

// ── Inference ─────────────────────────────────────────────────────────────────

print("Espresso · SimpleInference")
print("Prompt : \"\(prompt)\"")
print("Weights: \(weightsDir)\n")

do {
    var engine = try RealModelInferenceEngine.build(
        config: ModelRegistry.gpt2_124m,
        weightDir: weightsDir,
        tokenizerDir: weightsDir
    )

    let result = try engine.generate(prompt: prompt, maxTokens: maxTokens) { step in
        print(step.text, terminator: ""); fflush(stdout)
    }

    print("\n\n── Stats ──────────────────────────────────")
    print(String(format: "Compile   : %.1f ms", result.compileTimeMs))
    print(String(format: "First tok : %.1f ms", result.firstTokenLatencyMs))
    print(String(format: "Throughput: %.1f tok/s", result.tokensPerSecond))

} catch {
    fputs("\nError: \(error)\n", stderr)
    fputs("Tip: run `./espresso install` from the Espresso repo to download GPT-2 weights.\n", stderr)
    exit(1)
}

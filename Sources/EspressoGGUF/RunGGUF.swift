import Foundation
import ModelSupport
import RealModelInference

/// Standalone entry point for testing GGUF → ANE inference.
/// Run with: swift run EspressoGGUFRunner <path-to-gguf> <tokenizer-dir> [prompt]
public enum RunGGUF {
    public static func main(ggufPath: String, tokenizerDir: String, prompt: String, maxTokens: Int) async throws {
        let url = URL(fileURLWithPath: ggufPath)
        fputs("Loading GGUF: \(ggufPath)\n", stderr)

        let prepareStart = DispatchTime.now()
        let prepared = try await GGUFModelLoader.prepare(ggufURL: url)
        let prepareMs = Double(DispatchTime.now().uptimeNanoseconds - prepareStart.uptimeNanoseconds) / 1_000_000
        fputs("Prepared: \(prepared.config.name), \(prepared.config.nLayer) layers, d=\(prepared.config.dModel), vocab=\(prepared.config.vocab), \(prepared.tensorCount) tensors in \(String(format: "%.0f", prepareMs))ms\n", stderr)
        fputs("Architecture: \(prepared.config.architecture == .gpt2 ? "gpt2" : "llama")\n", stderr)
        fputs("Weight dir: \(prepared.weightDir)\n", stderr)

        // Import RealModelInference dynamically would be ideal, but since we're in EspressoGGUF
        // which depends on RealModelInference, we can use it directly
        fputs("Building engine...\n", stderr)

        let buildStart = DispatchTime.now()
        // RealModelInferenceEngine.build needs a tokenizer dir.
        // For the GGUF path, the tokenizer is embedded in the GGUF metadata.
        // We need to extract it or use a separate tokenizer dir.
        var engine = try RealModelInferenceEngine.build(
            config: prepared.config,
            weightDir: prepared.weightDir,
            tokenizerDir: tokenizerDir
        )
        let buildMs = Double(DispatchTime.now().uptimeNanoseconds - buildStart.uptimeNanoseconds) / 1_000_000
        fputs("Engine built in \(String(format: "%.0f", buildMs))ms\n", stderr)

        fputs("Generating from prompt: \"\(prompt)\"\n", stderr)
        let result = try engine.generate(
            prompt: prompt,
            maxTokens: maxTokens,
            temperature: 0.0
        )

        // Output
        print(result.text)
        fputs("\n--- Stats ---\n", stderr)
        fputs("Tokens generated: \(result.tokens.count)\n", stderr)
        fputs("Tokens/sec: \(String(format: "%.2f", result.tokensPerSecond))\n", stderr)
        fputs("Compile time: \(String(format: "%.0f", result.compileTimeMs))ms\n", stderr)
        fputs("First token latency: \(String(format: "%.2f", result.firstTokenLatencyMs))ms\n", stderr)

        GGUFModelLoader.cleanup(weightDir: prepared.weightDir)
    }
}

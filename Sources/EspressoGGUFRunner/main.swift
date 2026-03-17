import Foundation
import EspressoGGUF
import RealModelInference
import ModelSupport

@main
struct GGUFRunner {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count >= 4 else {
            fputs("Usage: EspressoGGUFRunner <gguf-path> <tokenizer-dir> <prompt> [max-tokens]\n", stderr)
            Foundation.exit(1)
        }

        let ggufPath = args[1]
        let tokenizerDir = args[2]
        let prompt = args[3]
        let maxTokens = args.count > 4 ? Int(args[4]) ?? 32 : 32

        do {
            try await RunGGUF.main(
                ggufPath: ggufPath,
                tokenizerDir: tokenizerDir,
                prompt: prompt,
                maxTokens: maxTokens
            )
        } catch {
            fputs("Error: \(error)\n", stderr)
            Foundation.exit(1)
        }
    }
}

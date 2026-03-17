import Foundation
import EspressoEdgeRunner
import EdgeRunnerIO
import ModelSupport
import Metal

/// Quick benchmark: load GGUF → prepare weights → report config.
/// Full inference requires RealModelInferenceEngine.build() which needs
/// a tokenizer directory + ANE hardware.
public enum GGUFBenchmark {

    public struct BenchmarkResult: Sendable {
        public let modelName: String
        public let architecture: String
        public let nLayer: Int
        public let dModel: Int
        public let vocab: Int
        public let tensorCount: Int
        public let loadTimeMs: Double
        public let convertTimeMs: Double
        public let totalTimeMs: Double
    }

    public static func run(ggufPath: String) async throws -> BenchmarkResult {
        let url = URL(fileURLWithPath: ggufPath)
        let totalStart = DispatchTime.now()

        // Load GGUF
        let loadStart = DispatchTime.now()
        let ggufLoader = try GGUFLoader(url: url)
        let weightMap = try await ggufLoader.load(from: url)
        let loadMs = Double(DispatchTime.now().uptimeNanoseconds - loadStart.uptimeNanoseconds) / 1_000_000

        // Convert
        let convertStart = DispatchTime.now()
        let prepared = try await GGUFModelLoader.prepare(ggufURL: url)
        let convertMs = Double(DispatchTime.now().uptimeNanoseconds - convertStart.uptimeNanoseconds) / 1_000_000

        let totalMs = Double(DispatchTime.now().uptimeNanoseconds - totalStart.uptimeNanoseconds) / 1_000_000

        return BenchmarkResult(
            modelName: prepared.config.name,
            architecture: prepared.config.architecture == .gpt2 ? "gpt2" : "llama",
            nLayer: prepared.config.nLayer,
            dModel: prepared.config.dModel,
            vocab: prepared.config.vocab,
            tensorCount: prepared.tensorCount,
            loadTimeMs: loadMs,
            convertTimeMs: convertMs,
            totalTimeMs: totalMs
        )
    }
}

import Testing
import Foundation
@testable import EspressoGGUF
import ModelSupport

@Suite("GGUFModelLoader")
struct GGUFModelLoaderTests {

    @Test("PreparedModel has expected fields")
    func preparedModelFields() {
        let config = MultiModelConfig(
            name: "test", nLayer: 12, nHead: 12, nKVHead: 12,
            dModel: 768, headDim: 64, hiddenDim: 3072,
            vocab: 50257, maxSeq: 1024, normEps: 1e-5,
            architecture: .gpt2
        )
        let prepared = GGUFModelLoader.PreparedModel(
            config: config, weightDir: "/tmp/test", tensorCount: 42
        )
        #expect(prepared.config.architecture == .gpt2)
        #expect(prepared.weightDir == "/tmp/test")
        #expect(prepared.tensorCount == 42)
    }

    @Test("Cleanup removes directory")
    func cleanupRemovesDir() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("espresso_gguf_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let marker = dir.appendingPathComponent("test.bin")
        try Data([0x01]).write(to: marker)

        #expect(FileManager.default.fileExists(atPath: dir.path))
        GGUFModelLoader.cleanup(weightDir: dir.path)
        #expect(!FileManager.default.fileExists(atPath: dir.path))
    }

    @Test("Full GGUF prepare pipeline",
          .enabled(if: ProcessInfo.processInfo.environment["GGUF_MODEL_PATH"] != nil))
    func fullPipelineWithRealModel() async throws {
        let modelPath = ProcessInfo.processInfo.environment["GGUF_MODEL_PATH"]!
        let prepared = try await GGUFModelLoader.prepare(
            ggufURL: URL(fileURLWithPath: modelPath)
        )
        #expect(prepared.tensorCount > 0)
        #expect(prepared.config.nLayer > 0)
        #expect(prepared.config.dModel > 0)
        #expect(FileManager.default.fileExists(atPath: prepared.weightDir))
        GGUFModelLoader.cleanup(weightDir: prepared.weightDir)
    }
}

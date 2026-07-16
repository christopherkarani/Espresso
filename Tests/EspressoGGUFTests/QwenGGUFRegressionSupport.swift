import ANETypes
import EspressoGGUF
import Foundation
import ModelSupport
import Testing
@testable import RealModelInference

let defaultQwenTokenizerDir =
    "/Users/chriskarani/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-8bit/snapshots/11de96878523501bcaa86104e3c186de07ff9068"
let defaultQwen060BModelPath = "/tmp/edgerunner-models/Qwen3-0.6B-Q8_0.gguf"

private func qwenGGUFRegressionInputsAvailable(
    modelEnvKey: String,
    defaultModelPath: String,
    tokenizerEnvKey: String = "QWEN_TOKENIZER_DIR",
    defaultTokenizerDir: String = defaultQwenTokenizerDir,
    env: [String: String] = ProcessInfo.processInfo.environment
) -> Bool {
    let modelPath = env[modelEnvKey] ?? defaultModelPath
    let tokenizerDir = env[tokenizerEnvKey] ?? defaultTokenizerDir
    var isDirectory: ObjCBool = false
    return FileManager.default.fileExists(atPath: modelPath) &&
        FileManager.default.fileExists(atPath: tokenizerDir, isDirectory: &isDirectory) &&
        isDirectory.boolValue
}

let qwen060BRegressionEnabled = qwenGGUFRegressionInputsAvailable(
    modelEnvKey: "QWEN_0_6B_GGUF_MODEL_PATH",
    defaultModelPath: defaultQwen060BModelPath
)

enum QwenGGUFRegressionSupport {
    static let helloPrompt = "Hello"
    static let helloPromptTokens: [TokenID] = [9707]
    static let helloContinuationTokens: [TokenID] = [21806, 11, 358, 2776, 14589, 369, 279, 3681]
    static let helloContinuationText = "Hello Answer, I'm sorry for the previous"
    static let latePrefixPrompt = "Hello Answer, I'm sorry for the"
    static let latePrefixTokens: [TokenID] = [9707, 21806, 11, 358, 2776, 14589, 369, 279]
    static let latePrefixExpectedToken: TokenID = 3681
    static let helloFirstToken: TokenID = 21806

    struct Fixture: Sendable {
        let modelURL: URL
        let tokenizerDir: URL
    }

    static func fixture(
        modelEnvKey: String,
        defaultModelPath: String,
        tokenizerEnvKey: String = "QWEN_TOKENIZER_DIR",
        defaultTokenizerDir: String = defaultQwenTokenizerDir,
        env: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> Fixture {
        let rawModelPath = env[modelEnvKey] ?? defaultModelPath
        let rawTokenizerDir = env[tokenizerEnvKey] ?? defaultTokenizerDir

        let modelURL = URL(fileURLWithPath: rawModelPath)
        let tokenizerDir = URL(fileURLWithPath: rawTokenizerDir, isDirectory: true)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw NSError(domain: "QwenGGUFRegressionTests", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Missing Qwen GGUF model at \(modelURL.path)"
            ])
        }
        guard FileManager.default.fileExists(atPath: tokenizerDir.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw NSError(domain: "QwenGGUFRegressionTests", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "Missing tokenizer directory at \(tokenizerDir.path)"
            ])
        }

        return Fixture(modelURL: modelURL, tokenizerDir: tokenizerDir)
    }
}

actor QwenGGUFPreparedModelCache {
    static let shared = QwenGGUFPreparedModelCache()

    private var preparedByModelPath: [String: GGUFModelLoader.PreparedModel] = [:]

    func preparedModel(for fixture: QwenGGUFRegressionSupport.Fixture) async throws -> GGUFModelLoader.PreparedModel {
        if let prepared = preparedByModelPath[fixture.modelURL.path] {
            return prepared
        }

        let prepared = try await GGUFModelLoader.prepare(ggufURL: fixture.modelURL)
        preparedByModelPath[fixture.modelURL.path] = prepared
        return prepared
    }
}

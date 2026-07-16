import ANETypes
import EspressoGGUF
import Foundation
import Testing
@testable import RealModelInference

@Suite("QwenGGUFRegression", .serialized)
struct QwenGGUFRegressionTests {
    private func manualFixture(
        env: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> QwenGGUFRegressionSupport.Fixture {
        guard let modelPath = env["ESPRESSO_QWEN_MANUAL_MODEL_PATH"], !modelPath.isEmpty else {
            throw NSError(domain: "QwenGGUFRegressionTests", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "ESPRESSO_QWEN_MANUAL_MODEL_PATH is required"
            ])
        }

        let tokenizerDir = env["QWEN_TOKENIZER_DIR"] ?? defaultQwenTokenizerDir
        let modelURL = URL(fileURLWithPath: modelPath)
        let tokenizerURL = URL(fileURLWithPath: tokenizerDir, isDirectory: true)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw NSError(domain: "QwenGGUFRegressionTests", code: 4, userInfo: [
                NSLocalizedDescriptionKey: "Missing manual Qwen GGUF model at \(modelURL.path)"
            ])
        }
        guard FileManager.default.fileExists(atPath: tokenizerURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw NSError(domain: "QwenGGUFRegressionTests", code: 5, userInfo: [
                NSLocalizedDescriptionKey: "Missing tokenizer directory at \(tokenizerURL.path)"
            ])
        }

        return .init(modelURL: modelURL, tokenizerDir: tokenizerURL)
    }

    private func parseExpectedTokens(
        envKey: String,
        env: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> [TokenID] {
        guard let raw = env[envKey], !raw.isEmpty else {
            throw NSError(domain: "QwenGGUFRegressionTests", code: 6, userInfo: [
                NSLocalizedDescriptionKey: "\(envKey) is required"
            ])
        }
        let tokens = raw
            .split(separator: ",")
            .compactMap { TokenID($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
        guard !tokens.isEmpty else {
            throw NSError(domain: "QwenGGUFRegressionTests", code: 7, userInfo: [
                NSLocalizedDescriptionKey: "\(envKey) must contain at least one token"
            ])
        }
        return tokens
    }

    private func preparedModel(
        modelEnvKey: String,
        defaultModelPath: String
    ) async throws -> (QwenGGUFRegressionSupport.Fixture, GGUFModelLoader.PreparedModel) {
        let fixture = try QwenGGUFRegressionSupport.fixture(
            modelEnvKey: modelEnvKey,
            defaultModelPath: defaultModelPath
        )
        let prepared = try await QwenGGUFPreparedModelCache.shared.preparedModel(for: fixture)
        return (fixture, prepared)
    }

    private func assertPreparedArtifactRegressionGate(
        modelEnvKey: String,
        defaultModelPath: String
    ) async throws {
        let (fixture, prepared) = try await preparedModel(
            modelEnvKey: modelEnvKey,
            defaultModelPath: defaultModelPath
        )

        #expect(FileManager.default.fileExists(atPath: prepared.weightDir))
        #expect(
            prepared.weightDir.contains("espresso_gguf_")
                || prepared.weightDir.contains("/gguf-prepared/")
        )

        var engine = try RealModelInferenceEngine.build(
            config: prepared.config,
            weightDir: prepared.weightDir,
            tokenizerDir: fixture.tokenizerDir.path
        )
        let coldStart = try engine.generate(
            prompt: QwenGGUFRegressionSupport.helloPrompt,
            maxTokens: 1,
            temperature: 0
        )
        #expect(coldStart.tokens == [QwenGGUFRegressionSupport.helloFirstToken])
        #expect(coldStart.text == "Hello Answer")

        let latePrefix = try engine.generate(
            prompt: QwenGGUFRegressionSupport.latePrefixPrompt,
            maxTokens: 1,
            temperature: 0
        )
        #expect(latePrefix.promptTokens == QwenGGUFRegressionSupport.latePrefixTokens)
        #expect(latePrefix.tokens == [QwenGGUFRegressionSupport.latePrefixExpectedToken])
        #expect(!RealModelInferenceEngine.shouldRoundCPUExactDecodeIntermediatesToFP16(env: [:]))

        let hello = try engine.generate(
            prompt: QwenGGUFRegressionSupport.helloPrompt,
            maxTokens: QwenGGUFRegressionSupport.helloContinuationTokens.count,
            temperature: 0
        )

        #expect(hello.tokens == QwenGGUFRegressionSupport.helloContinuationTokens)
        #expect(hello.tokens.first == QwenGGUFRegressionSupport.helloFirstToken)
        #expect(hello.text == QwenGGUFRegressionSupport.helloContinuationText)
    }

    @Test(
        "Qwen 0.6B prepared artifact Hello and late-prefix parity match raw GGUF golden",
        .enabled(if: qwen060BRegressionEnabled)
    )
    func qwen060BPreparedArtifactRegressionGateMatchesGolden() async throws {
        try await assertPreparedArtifactRegressionGate(
            modelEnvKey: "QWEN_0_6B_GGUF_MODEL_PATH",
            defaultModelPath: defaultQwen060BModelPath
        )
    }

    @Test(
        "Manual Qwen exact CPU late-prefix token matches expected",
        .enabled(if: ProcessInfo.processInfo.environment["ESPRESSO_QWEN_MANUAL_MODEL_PATH"] != nil
            && ProcessInfo.processInfo.environment["ESPRESSO_QWEN_MANUAL_EXPECTED_LATE_PREFIX_TOKEN"] != nil)
    )
    func qwenManualExactCPULatePrefixMatchesExpected() async throws {
        let env = ProcessInfo.processInfo.environment
        let fixture = try manualFixture(env: env)
        let prepared = try await QwenGGUFPreparedModelCache.shared.preparedModel(for: fixture)
        let expected = try parseExpectedTokens(
            envKey: "ESPRESSO_QWEN_MANUAL_EXPECTED_LATE_PREFIX_TOKEN",
            env: env
        )
        #expect(expected.count == 1)

        let token = try RealModelInferenceEngine.generateNextTokenExactCPUForTesting(
            config: prepared.config,
            weightDir: prepared.weightDir,
            promptTokens: QwenGGUFRegressionSupport.latePrefixTokens
        )

        #expect(token == expected[0])
    }

    @Test(
        "Manual Qwen Hello continuation matches expected tokens",
        .enabled(if: ProcessInfo.processInfo.environment["ESPRESSO_QWEN_MANUAL_MODEL_PATH"] != nil
            && ProcessInfo.processInfo.environment["ESPRESSO_QWEN_MANUAL_EXPECTED_HELLO_TOKENS"] != nil)
    )
    func qwenManualHelloContinuationMatchesExpected() async throws {
        let env = ProcessInfo.processInfo.environment
        let fixture = try manualFixture(env: env)
        let prepared = try await QwenGGUFPreparedModelCache.shared.preparedModel(for: fixture)
        let expectedTokens = try parseExpectedTokens(
            envKey: "ESPRESSO_QWEN_MANUAL_EXPECTED_HELLO_TOKENS",
            env: env
        )
        let tokens = try RealModelInferenceEngine.generateTokensExactCPUForTesting(
            config: prepared.config,
            weightDir: prepared.weightDir,
            promptTokens: QwenGGUFRegressionSupport.helloPromptTokens,
            maxTokens: expectedTokens.count
        )

        #expect(tokens == expectedTokens)
    }

}

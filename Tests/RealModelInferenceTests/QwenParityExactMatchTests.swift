import ANETypes
import Darwin
import ESPRuntime
import Foundation
import ModelSupport
import Testing
@testable import RealModelInference

// Hardware-gated greedy parity for Qwen2.5-0.5B-Instruct against a PyTorch fp32 reference.
//
// The fixture is produced by:
//   scripts/qwen25_pytorch_reference.py fixtures \
//     --output Tests/RealModelInferenceTests/Fixtures/qwen25-05b-greedy-reference.json \
//     --max-new-tokens 32
// Real fixtures flags: --source-dir, --prompts, --output, --max-new-tokens,
// --min-prompts, --raw-prompt, --eos-token-ids.
//
// The converted artifact is produced by:
//   scripts/convert_qwen25_05b_to_esp.py
//
// Run with:
//   ANE_HARDWARE_TESTS=1 ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK=1 \
//     swift test --filter QwenGreedyParity

struct QwenGreedyReferenceFixture: Decodable {
    struct Case: Decodable {
        let index: Int
        let prompt: String
        let promptTokens: [Int]
        let expectedTokens: [Int]
        let expectedText: String
        let stoppedOnEOS: Bool
        /// Per-step top-1/top-2 logit gap the reference saw.
        let topLogitGaps: [Double]
        /// Per-step second-choice token the reference would have picked.
        let runnerUpTokens: [Int]
        /// Smallest top-1/top-2 logit gap the reference saw while producing this case.
        let minTopLogitGap: Double?
    }

    let model: String
    let reference: String
    let chatTemplate: Bool
    let maxNewTokens: Int
    let eosTokenIds: [Int]
    let cases: [Case]
}

private func qwenHardwareTestsEnabled() -> Bool {
    ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" && qwenANEIsAvailable()
}

private func qwenANEIsAvailable() -> Bool {
    let handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW
    )
    if handle == nil {
        return false
    }
    dlclose(handle)
    return ["_ANEInMemoryModelDescriptor", "_ANEInMemoryModel", "_ANERequest", "_ANEIOSurfaceObject"]
        .allSatisfy { NSClassFromString($0) != nil }
}

/// Locates the converted Qwen2.5-0.5B packed `.esp` artifact (the shipped surface).
/// Override the bundle path with `ESPRESSO_QWEN_BUNDLE`.
struct QwenParityArtifact {
    let rootURL: URL
    let weightDirectory: URL

    static func cacheRoot(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> URL {
        if let override = environment["ESPRESSO_CACHE_HOME"], !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        return homeDirectory.appendingPathComponent("Library/Caches/Espresso", isDirectory: true)
    }

    static func resolve(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default,
        homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> QwenParityArtifact? {
        if let override = environment["ESPRESSO_QWEN_BUNDLE"], !override.isEmpty {
            return bundleArtifact(at: URL(fileURLWithPath: override, isDirectory: true), fileManager: fileManager)
        }

        let cache = cacheRoot(environment: environment, homeDirectory: homeDirectory)
            .appendingPathComponent("qwen25-05b", isDirectory: true)
        return bundleArtifact(
            at: cache.appendingPathComponent("Qwen2.5-0.5B-Instruct.esp", isDirectory: true),
            fileManager: fileManager
        )
    }

    private static func bundleArtifact(at url: URL, fileManager: FileManager) -> QwenParityArtifact? {
        let manifest = url.appendingPathComponent("manifest.toml")
        let metadata = url.appendingPathComponent("weights/metadata.json")
        guard fileManager.fileExists(atPath: manifest.path),
              fileManager.fileExists(atPath: metadata.path)
        else {
            return nil
        }
        return QwenParityArtifact(
            rootURL: url,
            weightDirectory: url.appendingPathComponent("weights", isDirectory: true)
        )
    }
}

private func loadQwenGreedyFixture() throws -> QwenGreedyReferenceFixture? {
    var candidates: [URL] = [
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/qwen25-05b-greedy-reference.json"),
    ]
    if let resource = Bundle.module.url(
        forResource: "qwen25-05b-greedy-reference",
        withExtension: "json",
        subdirectory: "Fixtures"
    ) {
        candidates.append(resource)
    }
    for url in candidates where FileManager.default.fileExists(atPath: url.path) {
        return try JSONDecoder().decode(QwenGreedyReferenceFixture.self, from: Data(contentsOf: url))
    }
    return nil
}

@Test func test_qwenGreedyParityFixtureCoversTheRequiredSuite() throws {
    guard let fixture = try loadQwenGreedyFixture() else {
        // The fixture is committed, so absence means a packaging problem worth surfacing
        // even on machines without an ANE.
        Issue.record("qwen25-05b-greedy-reference.json fixture is missing")
        return
    }
    #expect(fixture.model == "Qwen2.5-0.5B-Instruct")
    #expect(fixture.cases.count >= 8)
    #expect(fixture.maxNewTokens >= 32)
    for testCase in fixture.cases {
        #expect(testCase.expectedTokens.count >= 32)
        #expect(!testCase.promptTokens.isEmpty)
        #expect(testCase.topLogitGaps.count == testCase.expectedTokens.count)
        #expect(testCase.runnerUpTokens.count == testCase.expectedTokens.count)
    }
}

@Test func test_qwenParityArtifactPrefersPackedESPBundleOverNativeDir() throws {
    let fileManager = FileManager.default
    let root = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    defer { try? fileManager.removeItem(at: root) }

    let cache = root.appendingPathComponent("Caches/Espresso", isDirectory: true)
    let native = cache
        .appendingPathComponent("qwen25-05b", isDirectory: true)
        .appendingPathComponent("Qwen2.5-0.5B-Instruct-native", isDirectory: true)
    let bundle = cache
        .appendingPathComponent("qwen25-05b", isDirectory: true)
        .appendingPathComponent("Qwen2.5-0.5B-Instruct.esp", isDirectory: true)
    try fileManager.createDirectory(at: native, withIntermediateDirectories: true)
    try fileManager.createDirectory(
        at: bundle.appendingPathComponent("weights", isDirectory: true),
        withIntermediateDirectories: true
    )
    try Data("{}".utf8).write(to: native.appendingPathComponent("metadata.json"))
    try Data("format_version = \"1.1.0\"\n".utf8).write(to: bundle.appendingPathComponent("manifest.toml"))
    try Data("{}".utf8).write(to: bundle.appendingPathComponent("weights/metadata.json"))

    let resolved = QwenParityArtifact.resolve(
        environment: ["ESPRESSO_CACHE_HOME": cache.path],
        fileManager: fileManager,
        homeDirectory: root
    )
    #expect(resolved?.rootURL.path == bundle.path)
    #expect(resolved?.weightDirectory.lastPathComponent == "weights")
}

@Test func test_generationResultWithDecodePathPreservesTokens() {
    let original = GenerationResult(
        text: "hi",
        tokens: [1, 2],
        promptTokens: [9],
        tokensPerSecond: 0,
        compileTimeMs: 1,
        firstTokenLatencyMs: 1,
        exactHeadBackend: "cpu_fp16_tiled"
    )
    #expect(original.decodePath == "unknown")
    let labeled = original.withDecodePath("hybrid")
    #expect(labeled.decodePath == "hybrid")
    #expect(labeled.tokens == original.tokens)
    #expect(labeled.exactHeadBackend == "cpu_fp16_tiled")
}

/// Greedy decoding on the ANE hybrid path must reproduce the PyTorch reference token IDs.
///
/// Generation is driven from the fixture's prompt token IDs so this measures the model, not
/// the tokenizer. The converted `.esp` is the artifact under test (same path `./espresso
/// generate --model` uses). Tokenizer agreement is covered separately.
@Test func test_qwenGreedyParityMatchesPyTorchReferenceOnANE() throws {
    guard qwenHardwareTestsEnabled() else { return }
    guard let artifact = QwenParityArtifact.resolve() else {
        Issue.record(
            """
            Converted Qwen .esp bundle not found. Run scripts/convert_qwen25_05b_to_esp.py \
            or set ESPRESSO_QWEN_BUNDLE.
            """
        )
        return
    }
    guard let fixture = try loadQwenGreedyFixture() else {
        Issue.record("qwen25-05b-greedy-reference.json fixture is missing")
        return
    }

    let bundle = try ESPRuntimeBundle.open(at: artifact.rootURL)
    let config = bundle.config
    let weightDir = bundle.archive.weightsURL.path
    #expect(config.preferredDecodePath == .hybrid)
    // With fallback disabled this artifact must resolve to the ANE hybrid path, never CPU.
    #expect(
        try RealModelInferenceEngine.resolvedLlamaGenerationPath(
            config: config,
            environment: ["ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK": "1"]
        ) == .hybrid
    )

    let cases = fixture.cases
    let results = try RealModelInferenceEngine.generateFromTokenSuiteForTesting(
        config: config,
        weightDir: weightDir,
        promptTokenSuite: cases.map { $0.promptTokens.map(TokenID.init) },
        maxTokens: fixture.maxNewTokens
    )
    #expect(results.count == cases.count)
    #expect(results.allSatisfy { $0.decodePath == "hybrid" })

    var matchingPrefixLengths: [Int] = []
    var exactMatches = 0
    var unexplainedDivergences: [String] = []
    for (testCase, result) in zip(cases, results) {
        let expected = testCase.expectedTokens.map(TokenID.init)
        let actual = result.tokens
        var prefix = 0
        while prefix < min(expected.count, actual.count), expected[prefix] == actual[prefix] {
            prefix += 1
        }
        matchingPrefixLengths.append(prefix)
        if expected == actual {
            exactMatches += 1
            continue
        }

        // A divergence is only acceptable if it is a near-tie the reference itself was
        // unsure about, and the runtime picked precisely the reference's runner-up. Anything
        // else is a wrong answer, not a rounding difference.
        let gap = prefix < testCase.topLogitGaps.count ? testCase.topLogitGaps[prefix] : .infinity
        let runnerUp = prefix < testCase.runnerUpTokens.count
            ? TokenID(testCase.runnerUpTokens[prefix])
            : nil
        let chose = prefix < actual.count ? actual[prefix] : nil
        let choseRunnerUp = chose != nil && chose == runnerUp
        let withinLogitError = gap <= maxObservedANELogitError

        fputs(
            """
            [qwen-greedy-parity] case \(testCase.index) diverged at token \(prefix)/\(expected.count)
              prompt: \(testCase.prompt)
              reference top-1 \(expected[prefix]) runner-up \(runnerUp.map(String.init) ?? "?") \
            gap \(gap)
              runtime chose \(chose.map(String.init) ?? "?") \
            (runner-up: \(choseRunnerUp), gap within logit error: \(withinLogitError))

            """,
            stderr
        )

        if !choseRunnerUp || !withinLogitError {
            unexplainedDivergences.append(
                "case \(testCase.index) token \(prefix): chose \(chose.map(String.init) ?? "?"), "
                    + "reference top-1 \(expected[prefix]), runner-up "
                    + "\(runnerUp.map(String.init) ?? "?"), gap \(gap)"
            )
        }
    }

    let totalExpected = cases.reduce(0) { $0 + $1.expectedTokens.count }
    let totalMatching = matchingPrefixLengths.reduce(0, +)
    fputs(
        """
        [qwen-greedy-parity] exact cases \(exactMatches)/\(cases.count) \
        matching-prefix tokens \(totalMatching)/\(totalExpected) \
        path=\(results.first?.decodePath ?? "?") \
        artifact=esp \
        head=\(results.first?.exactHeadBackend ?? "?")

        """,
        stderr
    )

    // Every divergence must be an explained near-tie flip.
    #expect(
        unexplainedDivergences.isEmpty,
        "unexplained greedy divergences: \(unexplainedDivergences.joined(separator: "; "))"
    )
    // fp16 arithmetic on the ANE flips only near-ties, so the overwhelming majority of
    // sequences still reproduce exactly. A collapse here means a real regression.
    #expect(exactMatches >= 10)
    #expect(Double(exactMatches) / Double(cases.count) >= minimumExactSequenceRatio)
    #expect(Double(totalMatching) / Double(totalExpected) >= minimumTokenAgreementRatio)

    // Two short greedy runs of the first fixture prompt must emit identical tokens.
    let determinismPrompt = [cases[0].promptTokens.map(TokenID.init)]
    let firstRun = try RealModelInferenceEngine.generateFromTokenSuiteForTesting(
        config: config,
        weightDir: weightDir,
        promptTokenSuite: determinismPrompt,
        maxTokens: 8
    )
    let secondRun = try RealModelInferenceEngine.generateFromTokenSuiteForTesting(
        config: config,
        weightDir: weightDir,
        promptTokenSuite: determinismPrompt,
        maxTokens: 8
    )
    #expect(firstRun.count == 1)
    #expect(secondRun.count == 1)
    #expect(!firstRun[0].tokens.isEmpty)
    #expect(firstRun[0].tokens == secondRun[0].tokens)
}

/// Largest end-to-end logit error measured for the ANE hybrid stack against PyTorch fp32:
/// 0.955 over the 12 fixture prompts, recorded in `docs/qwen-logit-parity.json` and
/// regenerated by `scripts/qwen25_pytorch_reference.py logit-parity`. Greedy choices can
/// only flip where the reference's own top-1/top-2 gap is inside this band, so a
/// divergence at a wider gap is a bug rather than rounding.
private let maxObservedANELogitError = 0.96

/// Published floor: 10/12 sequences must reproduce exactly.
private let minimumExactSequenceRatio = 10.0 / 12

/// Published floor: 341/384 tokens must agree, and full agreement before any flip.
private let minimumTokenAgreementRatio = 341.0 / 384

import ANETypes
import Darwin
import Foundation
import ModelSupport
import Testing
@testable import RealModelInference

// Hardware-gated greedy parity for Qwen2.5-0.5B-Instruct against a PyTorch fp32 reference.
//
// The fixture is produced by:
//   scripts/qwen25_pytorch_reference.py fixtures \
//     --output Tests/RealModelInferenceTests/Fixtures/qwen25-05b-greedy-reference.json \
//     --max-new-tokens 32 --forbid-early-stop
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

/// The converted native directory. Defaults to the converter's cache location.
private func qwenNativeDirectory() -> URL? {
    if let override = ProcessInfo.processInfo.environment["ESPRESSO_QWEN_NATIVE_DIR"], !override.isEmpty {
        return URL(fileURLWithPath: override, isDirectory: true)
    }
    let cacheRoot: URL
    if let override = ProcessInfo.processInfo.environment["ESPRESSO_CACHE_HOME"], !override.isEmpty {
        cacheRoot = URL(fileURLWithPath: override, isDirectory: true)
    } else {
        cacheRoot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Caches/Espresso", isDirectory: true)
    }
    let candidate = cacheRoot
        .appendingPathComponent("qwen25-05b", isDirectory: true)
        .appendingPathComponent("Qwen2.5-0.5B-Instruct-native", isDirectory: true)
    return FileManager.default.fileExists(atPath: candidate.appendingPathComponent("metadata.json").path)
        ? candidate
        : nil
}

private func loadQwenGreedyFixture() throws -> QwenGreedyReferenceFixture? {
    let url = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures/qwen25-05b-greedy-reference.json")
    guard FileManager.default.fileExists(atPath: url.path) else { return nil }
    return try JSONDecoder().decode(QwenGreedyReferenceFixture.self, from: Data(contentsOf: url))
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

/// Greedy decoding on the ANE hybrid path must reproduce the PyTorch reference token IDs.
///
/// Generation is driven from the fixture's prompt token IDs so this measures the model, not
/// the tokenizer. Tokenizer agreement is covered separately.
@Test func test_qwenGreedyParityMatchesPyTorchReferenceOnANE() throws {
    guard qwenHardwareTestsEnabled() else { return }
    guard let nativeDir = qwenNativeDirectory() else {
        Issue.record(
            """
            Converted Qwen native directory not found. Run scripts/convert_qwen25_05b_to_esp.py \
            or set ESPRESSO_QWEN_NATIVE_DIR.
            """
        )
        return
    }
    guard let fixture = try loadQwenGreedyFixture() else {
        Issue.record("qwen25-05b-greedy-reference.json fixture is missing")
        return
    }

    let config = try QwenLayerParityProbe.loadConfig(nativeDir: nativeDir.path)
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
        weightDir: nativeDir.path,
        promptTokenSuite: cases.map { $0.promptTokens.map(TokenID.init) },
        maxTokens: fixture.maxNewTokens
    )
    #expect(results.count == cases.count)

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
    #expect(Double(exactMatches) / Double(cases.count) >= minimumExactSequenceRatio)
    #expect(Double(totalMatching) / Double(totalExpected) >= minimumTokenAgreementRatio)
}

/// Largest end-to-end logit error measured for the ANE hybrid stack against PyTorch fp32:
/// 0.955 over the 12 fixture prompts, recorded in `docs/qwen-logit-parity.json` and
/// regenerated by `scripts/qwen25_pytorch_reference.py logit-parity`. Greedy choices can
/// only flip where the reference's own top-1/top-2 gap is inside this band, so a
/// divergence at a wider gap is a bug rather than rounding.
private let maxObservedANELogitError = 0.96

/// Measured on Apple M-series: 10/12 sequences reproduce exactly.
private let minimumExactSequenceRatio = 0.75

/// Measured on Apple M-series: 341/384 tokens agree, and full agreement before any flip.
private let minimumTokenAgreementRatio = 0.85

import ANETypes
import Darwin
import ESPRuntime
import Foundation
import ModelSupport
import Testing
@testable import RealModelInference

// Hardware-gated greedy parity for Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct
// against a PyTorch fp32 reference.
//
// Fixtures are produced by:
//   scripts/qwen25_pytorch_reference.py fixtures \
//     --output Tests/RealModelInferenceTests/Fixtures/qwen25-05b-greedy-reference.json \
//     --max-new-tokens 32
//   scripts/qwen25_pytorch_reference.py --model Qwen/Qwen2.5-1.5B-Instruct fixtures \
//     --output Tests/RealModelInferenceTests/Fixtures/qwen25-15b-greedy-reference.json \
//     --max-new-tokens 32
// Real fixtures flags: --model, --source-dir, --prompts, --output, --max-new-tokens,
// --min-prompts, --raw-prompt, --eos-token-ids.
//
// Converted artifacts:
//   scripts/convert_qwen25_05b_to_esp.py
//   scripts/convert_qwen25_05b_to_esp.py --model Qwen/Qwen2.5-1.5B-Instruct
//
// Run with:
//   ANE_HARDWARE_TESTS=1 ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK=1 \
//     swift test --filter qwenGreedyParity

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

/// Cache layout and greedy-contract constants for one Qwen2.5 Instruct size.
struct QwenParityProfile: Equatable, Sendable {
    let displayName: String
    let cacheSlug: String
    let bundleFileName: String
    let fixtureFileName: String
    /// Largest end-to-end ANE logit error measured against PyTorch fp32 for this size.
    let maxObservedANELogitError: Double
    let minimumExactMatches: Int
    let minimumExactSequenceRatio: Double
    let minimumTokenAgreementRatio: Double

    var fixtureResourceName: String {
        URL(fileURLWithPath: fixtureFileName).deletingPathExtension().lastPathComponent
    }

    static let qwen25_05b = QwenParityProfile(
        displayName: "Qwen2.5-0.5B-Instruct",
        cacheSlug: "qwen25-05b",
        bundleFileName: "Qwen2.5-0.5B-Instruct.esp",
        fixtureFileName: "qwen25-05b-greedy-reference.json",
        maxObservedANELogitError: 0.96,
        minimumExactMatches: 10,
        minimumExactSequenceRatio: 10.0 / 12,
        minimumTokenAgreementRatio: 341.0 / 384
    )

    /// Floors from the first hybrid suite (2026-08-14): 9/12 exact, 328/384 prefix
    /// tokens. `maxObservedANELogitError` is the worst ANE `|dlogit|` in
    /// `docs/qwen15b-logit-parity.json` (cases 0 and 5; chained hybrid + Python LM head).
    static let qwen25_15b = QwenParityProfile(
        displayName: "Qwen2.5-1.5B-Instruct",
        cacheSlug: "qwen25-15b",
        bundleFileName: "Qwen2.5-1.5B-Instruct.esp",
        fixtureFileName: "qwen25-15b-greedy-reference.json",
        maxObservedANELogitError: 0.24,
        minimumExactMatches: 9,
        minimumExactSequenceRatio: 9.0 / 12,
        minimumTokenAgreementRatio: 328.0 / 384
    )
}

/// Locates a converted Qwen2.5 packed `.esp` artifact (the shipped surface).
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
        profile: QwenParityProfile = .qwen25_05b,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileManager: FileManager = .default,
        homeDirectory: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> QwenParityArtifact? {
        if let override = environment["ESPRESSO_QWEN_BUNDLE"], !override.isEmpty {
            return bundleArtifact(at: URL(fileURLWithPath: override, isDirectory: true), fileManager: fileManager)
        }

        let cache = cacheRoot(environment: environment, homeDirectory: homeDirectory)
            .appendingPathComponent(profile.cacheSlug, isDirectory: true)
        return bundleArtifact(
            at: cache.appendingPathComponent(profile.bundleFileName, isDirectory: true),
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

private func loadQwenGreedyFixture(
    profile: QwenParityProfile = .qwen25_05b
) throws -> QwenGreedyReferenceFixture? {
    var candidates: [URL] = [
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/\(profile.fixtureFileName)"),
    ]
    if let resource = Bundle.module.url(
        forResource: profile.fixtureResourceName,
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

private func writeFakeQwenBundle(
    profile: QwenParityProfile,
    cache: URL,
    fileManager: FileManager
) throws -> (native: URL, bundle: URL) {
    let native = cache
        .appendingPathComponent(profile.cacheSlug, isDirectory: true)
        .appendingPathComponent("\(profile.displayName)-native", isDirectory: true)
    let bundle = cache
        .appendingPathComponent(profile.cacheSlug, isDirectory: true)
        .appendingPathComponent(profile.bundleFileName, isDirectory: true)
    try fileManager.createDirectory(at: native, withIntermediateDirectories: true)
    try fileManager.createDirectory(
        at: bundle.appendingPathComponent("weights", isDirectory: true),
        withIntermediateDirectories: true
    )
    try Data("{}".utf8).write(to: native.appendingPathComponent("metadata.json"))
    try Data("format_version = \"1.1.0\"\n".utf8).write(to: bundle.appendingPathComponent("manifest.toml"))
    try Data("{}".utf8).write(to: bundle.appendingPathComponent("weights/metadata.json"))
    return (native, bundle)
}

@Test func test_qwenGreedyParityFixtureCoversTheRequiredSuite() throws {
    try assertQwenGreedyFixtureCoversTheRequiredSuite(profile: .qwen25_05b)
}

@Test func test_qwenGreedyParity15bFixtureCoversTheRequiredSuite() throws {
    try assertQwenGreedyFixtureCoversTheRequiredSuite(profile: .qwen25_15b)
}

private func assertQwenGreedyFixtureCoversTheRequiredSuite(profile: QwenParityProfile) throws {
    guard let fixture = try loadQwenGreedyFixture(profile: profile) else {
        // The fixture is committed, so absence means a packaging problem worth surfacing
        // even on machines without an ANE.
        Issue.record("\(profile.fixtureFileName) fixture is missing")
        return
    }
    #expect(fixture.model == profile.displayName)
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
    let bundle = try writeFakeQwenBundle(
        profile: .qwen25_05b,
        cache: cache,
        fileManager: fileManager
    ).bundle

    let resolved = QwenParityArtifact.resolve(
        profile: .qwen25_05b,
        environment: ["ESPRESSO_CACHE_HOME": cache.path],
        fileManager: fileManager,
        homeDirectory: root
    )
    #expect(resolved?.rootURL.path == bundle.path)
    #expect(resolved?.weightDirectory.lastPathComponent == "weights")
}

@Test func test_qwenParityArtifactResolves15bPackedESPBundle() throws {
    let fileManager = FileManager.default
    let root = fileManager.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    defer { try? fileManager.removeItem(at: root) }

    let cache = root.appendingPathComponent("Caches/Espresso", isDirectory: true)
    let written = try writeFakeQwenBundle(
        profile: .qwen25_15b,
        cache: cache,
        fileManager: fileManager
    )
    _ = try writeFakeQwenBundle(
        profile: .qwen25_05b,
        cache: cache,
        fileManager: fileManager
    )

    let resolved05b = QwenParityArtifact.resolve(
        profile: .qwen25_05b,
        environment: ["ESPRESSO_CACHE_HOME": cache.path],
        fileManager: fileManager,
        homeDirectory: root
    )
    let resolved15b = QwenParityArtifact.resolve(
        profile: .qwen25_15b,
        environment: ["ESPRESSO_CACHE_HOME": cache.path],
        fileManager: fileManager,
        homeDirectory: root
    )
    #expect(resolved05b?.rootURL.lastPathComponent == QwenParityProfile.qwen25_05b.bundleFileName)
    #expect(resolved15b?.rootURL.path == written.bundle.path)
    #expect(resolved15b?.rootURL.path != resolved05b?.rootURL.path)
}

@Test func test_generationResultWithDecodePathPreservesTokens() throws {
    let original = GenerationResult(
        text: "hi",
        tokens: [1, 2],
        promptTokens: [9],
        tokensPerSecond: 0,
        compileTimeMs: 1,
        firstTokenLatencyMs: 1,
        exactHeadBackend: "cpu_fp16_tiled",
        decodeProfileReport: "decode_profile_mean_ms/token qkv=1.00 rope=2.00 attn=3.00 ffn=4.00 lm_head=5.00 io=6.00 n=1 exclude_ttft=1"
    )
    #expect(original.trunk == nil)
    #expect(original.decodePath == "unknown")
    #expect(original.hopsPerToken == nil)
    let labeled = original.withTrunk(.splitHybrid)
    #expect(labeled.trunk == .splitHybrid)
    #expect(labeled.decodePath == "hybrid")
    #expect(labeled.hopsPerToken == nil)
    #expect(labeled.tokens == original.tokens)
    #expect(labeled.exactHeadBackend == "cpu_fp16_tiled")
    #expect(labeled.decodeProfileReport == original.decodeProfileReport)
    let fused = try GenerationResult(
        text: "hi",
        tokens: [1],
        promptTokens: [9],
        tokensPerSecond: 0,
        compileTimeMs: 1,
        firstTokenLatencyMs: 1,
        exactHeadBackend: "cpu_fp16_tiled",
        cachedBindingsEnabled: false,
        decodePath: "fused",
        hopsPerToken: 28
    )
    #expect(try fused.withDecodePath("fused").hopsPerToken == 28)
    #expect(throws: Trunk.ParseError.unsupported("metal")) {
        try GenerationResult(
            text: "hi",
            tokens: [1],
            promptTokens: [9],
            tokensPerSecond: 0,
            compileTimeMs: 1,
            firstTokenLatencyMs: 1,
            decodePath: "metal"
        )
    }
}

/// Greedy decoding on the ANE hybrid path must reproduce the PyTorch reference token IDs.
///
/// Generation is driven from the fixture's prompt token IDs so this measures the model, not
/// the tokenizer. The converted `.esp` is the artifact under test (same path `./espresso
/// generate --model` uses). Tokenizer agreement is covered separately.
@Test func test_qwenGreedyParityMatchesPyTorchReferenceOnANE() throws {
    try assertQwenGreedyParityMatchesPyTorchReferenceOnANE(profile: .qwen25_05b)
}

@Test func test_qwenGreedyParity15bMatchesPyTorchReferenceOnANE() throws {
    try assertQwenGreedyParityMatchesPyTorchReferenceOnANE(profile: .qwen25_15b)
}

private func assertQwenGreedyParityMatchesPyTorchReferenceOnANE(
    profile: QwenParityProfile
) throws {
    guard qwenHardwareTestsEnabled() else { return }
    guard let artifact = QwenParityArtifact.resolve(profile: profile) else {
        Issue.record(
            """
            Converted \(profile.displayName) .esp bundle not found. Run \
            scripts/convert_qwen25_05b_to_esp.py --model Qwen/\(profile.displayName) \
            or set ESPRESSO_QWEN_BUNDLE.
            """
        )
        return
    }
    guard let fixture = try loadQwenGreedyFixture(profile: profile) else {
        Issue.record("\(profile.fixtureFileName) fixture is missing")
        return
    }
    #expect(fixture.model == profile.displayName)

    let bundle = try ESPRuntimeBundle.open(at: artifact.rootURL)
    let config = bundle.config
    let weightDir = bundle.archive.weightsURL.path
    #expect(config.preferredDecodePath == .hybrid)
    #expect(config.name == profile.displayName)
    // With fallback disabled this artifact must resolve to an ANE hybrid trunk, never CPU.
    let expectedTrunk: Trunk = ModelFamily.isQwen15BVariant(config) ? .fusedHybrid : .splitHybrid
    #expect(
        try RealModelInferenceEngine.resolvedTrunk(
            config: config,
            environment: ["ESPRESSO_REALMODEL_DISABLE_HYBRID_FALLBACK": "1"]
        ) == expectedTrunk
    )

    let cases = fixture.cases
    // Determinism shares this engine. A second generateFromTokenSuiteForTesting
    // call recompiles and exhausts the per-process ANE budget (0.5B died at
    // layer 16 on the follow-up run after a cold RMSNorm-hash compile).
    let determinismPrompt = cases[0].promptTokens.map(TokenID.init)
    let allResults = try RealModelInferenceEngine.generateFromTokenSuiteForTesting(
        config: config,
        weightDir: weightDir,
        promptTokenSuite: cases.map { $0.promptTokens.map(TokenID.init) } + [
            determinismPrompt,
            determinismPrompt,
        ],
        maxTokens: fixture.maxNewTokens
    )
    let snapshots = allResults.map { result in
        (
            decodePath: result.decodePath,
            hopsPerToken: result.hopsPerToken,
            exactHeadBackend: result.exactHeadBackend,
            cachedBindingsEnabled: result.cachedBindingsEnabled,
            tokens: result.tokens
        )
    }
    #expect(snapshots.count == cases.count + 2)
    let expectedDecodePath = expectedTrunk.telemetryLabel
    #expect(snapshots.allSatisfy { $0.decodePath == expectedDecodePath })
    if expectedTrunk == .fusedHybrid {
        #expect(snapshots.allSatisfy { $0.hopsPerToken == 28 })
        #expect(snapshots.allSatisfy { $0.cachedBindingsEnabled == false })
        #expect(snapshots.allSatisfy { $0.exactHeadBackend == "cpu_fp16_tiled" })
    }
    let results = Array(snapshots.prefix(cases.count))

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
        let withinLogitError = gap <= profile.maxObservedANELogitError

        fputs(
            """
            [qwen-greedy-parity] \(profile.displayName) case \(testCase.index) \
            diverged at token \(prefix)/\(expected.count)
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
        [qwen-greedy-parity] \(profile.displayName) exact cases \(exactMatches)/\(cases.count) \
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
    if profile.minimumExactMatches > 0 {
        #expect(exactMatches >= profile.minimumExactMatches)
        #expect(Double(exactMatches) / Double(cases.count) >= profile.minimumExactSequenceRatio)
        #expect(Double(totalMatching) / Double(totalExpected) >= profile.minimumTokenAgreementRatio)
    }

    // Two greedy runs of the first fixture prompt must emit identical tokens.
    let firstRun = allResults[cases.count]
    let secondRun = allResults[cases.count + 1]
    #expect(!firstRun.tokens.isEmpty)
    #expect(firstRun.tokens == secondRun.tokens)
}

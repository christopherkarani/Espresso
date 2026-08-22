import ANETypes
import Darwin
import ESPRuntime
import Foundation
import ModelSupport
import Testing
@testable import RealModelInference

typealias TraceTokenID = TokenID

// Golden decode traces: greedy token sequences recorded from the serving engine,
// replayed to prove the trunk-runtime refactor is behavior-preserving.
//
// Record:  ESPRESSO_RECORD_GOLDEN_TRACES=1 swift test --filter GoldenTraceTests
//          (optionally ESPRESSO_GOLDEN_TRACE_CASE=<id> to record one case per run)
// Replay:  `swift test --filter goldenTraceReplay` — the synthetic exact-CPU case
//          runs everywhere (no ANE, no downloads); real-artifact cases require the
//          local artifacts and are skipped loudly when absent.

struct GoldenTraceCase: Codable, Sendable {
    let id: String
    let model: String
    let trunkLabel: String
    let promptTokens: [Int]
    let maxTokens: Int
    let expectedTokens: [Int]

    init(id: String, model: String, trunkLabel: String, promptTokens: [Int], maxTokens: Int, expectedTokens: [Int]) {
        self.id = id
        self.model = model
        self.trunkLabel = trunkLabel
        self.promptTokens = promptTokens
        self.maxTokens = maxTokens
        self.expectedTokens = expectedTokens
    }
}

private struct GoldenTraceFixture: Codable, Sendable {
    let format: String
    let cases: [GoldenTraceCase]

    static let formatID = "golden-decode-traces-v1"
}

private let syntheticPrompt: [Int] = [1, 2, 3, 4, 5, 6, 7, 8]
private let syntheticMaxTokens = 24

private func fixtureURL() -> URL {
    URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures/golden-decode-traces.json")
}

private func loadFixture() throws -> GoldenTraceFixture? {
    let url = fixtureURL()
    guard FileManager.default.fileExists(atPath: url.path) else { return nil }
    return try JSONDecoder().decode(GoldenTraceFixture.self, from: Data(contentsOf: url))
}

/// Loud skip marker so CI logs show *why* a case did not run (anti silent-skip).
private func skipCase(_ id: String, _ reason: String) {
    print("[golden-traces] SKIP \(id): \(reason)")
}

private func aneIsAvailable() -> Bool {
    let handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW
    )
    guard handle != nil else { return false }
    dlclose(handle)
    return true
}

private func demoDirectory(_ subdirectory: String) -> URL {
    FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/Espresso/demo/\(subdirectory)", isDirectory: true)
}

struct QwenBundleLocation {
    let rootURL: URL
    let weightsPath: String
    let config: MultiModelConfig

    static func resolve(profileSlug: String, bundleFileName: String) -> QwenBundleLocation? {
        let root = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Caches/Espresso/\(profileSlug)/\(bundleFileName)", isDirectory: true)
        guard FileManager.default.fileExists(atPath: root.appendingPathComponent("manifest.toml").path),
              FileManager.default.fileExists(atPath: root.appendingPathComponent("weights/metadata.json").path)
        else { return nil }
        do {
            let bundle = try ESPRuntimeBundle.open(at: root)
            return QwenBundleLocation(rootURL: root, weightsPath: bundle.archive.weightsURL.path, config: bundle.config)
        } catch {
            print("[golden-traces] WARN failed to open \(root.path): \(error)")
            return nil
        }
    }
}

enum GoldenTraceRunner {

    // MARK: - Case execution

    static func run(case traceCase: GoldenTraceCase) throws -> (tokens: [Int], trunkLabel: String) {
        switch traceCase.model {
        case SyntheticLlamaMicro.name:
            let root = FileManager.default.temporaryDirectory
                .appendingPathComponent("espresso-golden-\(UUID().uuidString)", isDirectory: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let weightDir = try SyntheticLlamaMicro.makeBundle(in: root)
            let results = try RealModelInferenceEngine.generateFromTokenSuiteForTesting(
                config: SyntheticLlamaMicro.config,
                weightDir: weightDir.path,
                promptTokenSuite: [traceCase.promptTokens.map(TraceTokenID.init)],
                maxTokens: traceCase.maxTokens,
                options: DecodePathOptions(useCPUExactDecode: true)
            )
            return (results[0].tokens.map(Int.init), results[0].decodePath)

        case "stories110m":
            let dir = demoDirectory("stories110m")
            let metadata = dir.appendingPathComponent("metadata.json")
            guard FileManager.default.fileExists(atPath: metadata.path) else {
                throw CaseUnavailable("stories110m demo artifacts not present")
            }
            let config = try RealModelInferenceEngine.loadConfigFromMetadataFile(at: metadata)
            let results = try RealModelInferenceEngine.generateFromTokenSuiteForTesting(
                config: config,
                weightDir: dir.path,
                promptTokenSuite: [traceCase.promptTokens.map(TraceTokenID.init)],
                maxTokens: traceCase.maxTokens
            )
            return (results[0].tokens.map(Int.init), results[0].decodePath)

        case "qwen25-05b", "qwen25-15b":
            let slug = traceCase.model == "qwen25-05b" ? "qwen25-05b" : "qwen25-15b"
            let fileName = traceCase.model == "qwen25-05b"
                ? "Qwen2.5-0.5B-Instruct.esp" : "Qwen2.5-1.5B-Instruct.esp"
            guard let location = QwenBundleLocation.resolve(profileSlug: slug, bundleFileName: fileName) else {
                throw CaseUnavailable("\(traceCase.model) .esp bundle not present")
            }
            let results = try RealModelInferenceEngine.generateFromTokenSuiteForTesting(
                config: location.config,
                weightDir: location.weightsPath,
                promptTokenSuite: [traceCase.promptTokens.map(TraceTokenID.init)],
                maxTokens: traceCase.maxTokens
            )
            return (results[0].tokens.map(Int.init), results[0].decodePath)

        default:
            throw CaseUnavailable("unknown model kind \(traceCase.model)")
        }
    }

    /// Cases this machine can execute right now.
    static func availableCases() -> [GoldenTraceCase] {
        var cases: [GoldenTraceCase] = [
            GoldenTraceCase(
                id: "synthetic-exact-cpu",
                model: SyntheticLlamaMicro.name,
                trunkLabel: "cpu_exact",
                promptTokens: syntheticPrompt,
                maxTokens: syntheticMaxTokens,
                expectedTokens: []
            ),
        ]
        if aneIsAvailable() {
            cases.append(GoldenTraceCase(
                id: "stories110m-split-hybrid",
                model: "stories110m",
                trunkLabel: "hybrid",
                promptTokens: [1, 9856, 13, 338, 4099],
                maxTokens: 24,
                expectedTokens: []
            ))
            cases.append(GoldenTraceCase(
                id: "qwen05b-split-hybrid",
                model: "qwen25-05b",
                trunkLabel: "hybrid",
                promptTokens: [9707, 11, 1879, 374, 279, 5301, 315, 6524, 151645],
                maxTokens: 20,
                expectedTokens: []
            ))
            cases.append(GoldenTraceCase(
                id: "qwen15b-fused-hybrid",
                model: "qwen25-15b",
                trunkLabel: "fused",
                promptTokens: [9707, 11, 1879, 374, 279, 5301, 315, 6524, 151645],
                maxTokens: 20,
                expectedTokens: []
            ))
        }
        return cases
    }

    struct CaseUnavailable: Error, CustomStringConvertible {
        let description: String
        init(_ description: String) { self.description = description }
    }
}

// MARK: - Tests

@Test func goldenTraceRecord() throws {
    guard ProcessInfo.processInfo.environment["ESPRESSO_RECORD_GOLDEN_TRACES"] == "1" else {
        // Recording is an explicit operator action on ANE hardware; replay is the gate.
        print("[golden-traces] SKIP goldenTraceRecord: set ESPRESSO_RECORD_GOLDEN_TRACES=1 to record")
        return
    }
    let only = ProcessInfo.processInfo.environment["ESPRESSO_GOLDEN_TRACE_CASE"]

    var existing: [String: GoldenTraceCase] = [:]
    if let fixture = try loadFixture() {
        for traceCase in fixture.cases { existing[traceCase.id] = traceCase }
    }

    var recorded: [GoldenTraceCase] = []
    for candidate in GoldenTraceRunner.availableCases() where only == nil || only == candidate.id {
        let result = try GoldenTraceRunner.run(case: candidate)
        recorded.append(GoldenTraceCase(
            id: candidate.id,
            model: candidate.model,
            trunkLabel: result.trunkLabel,
            promptTokens: candidate.promptTokens,
            maxTokens: candidate.maxTokens,
            expectedTokens: result.tokens
        ))
        print("[golden-traces] recorded \(candidate.id): \(result.tokens.count) tokens via \(result.trunkLabel)")
    }

    guard !recorded.isEmpty else {
        throw GoldenTraceRunner.CaseUnavailable("no cases matched ESPRESSO_GOLDEN_TRACE_CASE=\(only ?? "*")")
    }

    for updated in recorded { existing[updated.id] = updated }
    let ordered = ["synthetic-exact-cpu", "stories110m-split-hybrid", "qwen05b-split-hybrid", "qwen15b-fused-hybrid"]
    let cases = ordered.compactMap { existing[$0] }

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(GoldenTraceFixture(format: GoldenTraceFixture.formatID, cases: cases))
    try data.write(to: fixtureURL())
    print("[golden-traces] wrote \(cases.count) cases to \(fixtureURL().path)")
}

@Test func goldenTraceReplay() throws {
    guard let fixture = try loadFixture() else {
        throw GoldenTraceRunner.CaseUnavailable(
            "\(fixtureURL().path) missing — record traces with ESPRESSO_RECORD_GOLDEN_TRACES=1 on ANE hardware"
        )
    }

    for traceCase in fixture.cases {
        let isSynthetic = traceCase.model == SyntheticLlamaMicro.name
        if !isSynthetic {
            guard aneIsAvailable() else {
                skipCase(traceCase.id, "ANE unavailable on this host")
                continue
            }
        }

        let result = try GoldenTraceRunner.run(case: traceCase)

        #expect(result.tokens == traceCase.expectedTokens, """
        golden trace drift in \(traceCase.id):
          expected \(traceCase.expectedTokens.prefix(12))…
          actual   \(result.tokens.prefix(12))…
        """)
        if result.trunkLabel != traceCase.trunkLabel {
            Issue.record("""
            trunk label changed for \(traceCase.id): \
            recorded \(traceCase.trunkLabel), now \(result.trunkLabel)
            """)
        }
    }
}

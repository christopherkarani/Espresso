import XCTest
@testable import Espresso

private func requireRWKVStyleRecurrentHardware(file: StaticString = #filePath, line: UInt = #line) throws {
    guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
        throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run recurrent hardware tests", file: file, line: line)
    }
}

final class RWKVStyleRecurrentPrototypeHardwareTests: XCTestCase {
    func test_context_scaling_benchmark_reports_requested_contexts_on_hardware() throws {
        try requireRWKVStyleRecurrentHardware()

        let contexts = [32, 256, 1024, 4096]
        let report = try RWKVStyleRecurrentBench.runContextScaling(
            contexts: contexts,
            warmup: 3,
            iterations: 20,
            transformerLayers: 6
        )

        XCTAssertEqual(report.recurrentSamples.map(\.effectiveContext), contexts)
        XCTAssertTrue(report.recurrentSamples.allSatisfy { $0.medianLatencyMs > 0 })
        XCTAssertTrue(report.recurrentSamples.allSatisfy { $0.tokensPerSecond > 0 })
        XCTAssertTrue(
            Set(report.transformerSamples.map(\.effectiveContext)).isSubset(of: Set(contexts))
        )

        for sample in report.recurrentSamples {
            print(
                "recurrent context=\(sample.effectiveContext) median=\(sample.medianLatencyMs) ms/token tok/s=\(sample.tokensPerSecond)"
            )
        }
        for sample in report.transformerSamples {
            print(
                "transformer context=\(sample.effectiveContext) median=\(sample.medianLatencyMs) ms/token tok/s=\(sample.tokensPerSecond)"
            )
        }
        if !report.skippedTransformerContexts.isEmpty {
            print("skipped transformer contexts=\(report.skippedTransformerContexts)")
        }
    }
}

import Foundation
import Testing
@testable import ESPBenchSupport

@Test func benchmarkGuardRejectsConcurrentExecutionOnSameLockPath() throws {
    let lockPath = FileManager.default.temporaryDirectory
        .appendingPathComponent("espresso-benchmark-lock-\(UUID().uuidString)")
        .path

    let firstGuard = try ESPBenchmarkExecutionGuard(lockPath: lockPath)
    #expect(!firstGuard.thermalBefore.isEmpty)

    do {
        _ = try ESPBenchmarkExecutionGuard(lockPath: lockPath)
        Issue.record("Expected concurrent benchmark rejection")
    } catch let error as ESPBenchmarkGuardError {
        #expect(error == .benchmarkAlreadyRunning(lockPath: lockPath))
    }
}

@Test func benchmarkThermalStateBlocksSeriousAndCriticalStarts() {
    #expect(ESPBenchmarkThermalState.nominal.permitsBenchmarkStart)
    #expect(ESPBenchmarkThermalState.fair.permitsBenchmarkStart)
    #expect(!ESPBenchmarkThermalState.serious.permitsBenchmarkStart)
    #expect(!ESPBenchmarkThermalState.critical.permitsBenchmarkStart)
}

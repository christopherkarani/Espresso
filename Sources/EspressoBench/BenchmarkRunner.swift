import Foundation
import os

// MARK: - Statistics

struct BenchmarkResult: Sendable {
    let label: String
    let latencies: [Double]  // milliseconds
    let warmupCount: Int
    let iterationCount: Int
    private let _sorted: [Double]

    init(label: String, latencies: [Double], warmupCount: Int, iterationCount: Int) {
        self.label = label
        self.latencies = latencies
        self.warmupCount = warmupCount
        self.iterationCount = iterationCount
        self._sorted = latencies.sorted()
    }

    var sorted: [Double] { _sorted }

    var mean: Double { latencies.reduce(0, +) / Double(latencies.count) }

    var median: Double { percentile(0.50) }
    var p50: Double { percentile(0.50) }
    var p95: Double { percentile(0.95) }
    var p99: Double { percentile(0.99) }
    var min: Double { _sorted.first ?? 0 }
    var max: Double { _sorted.last ?? 0 }

    var stddev: Double {
        let m = mean
        let variance = latencies.reduce(0.0) { $0 + ($1 - m) * ($1 - m) } / Double(latencies.count)
        return variance.squareRoot()
    }

    func percentile(_ p: Double) -> Double {
        guard !_sorted.isEmpty else { return 0 }
        let index = p * Double(_sorted.count - 1)
        let lower = Int(index)
        let upper = Swift.min(lower + 1, _sorted.count - 1)
        let frac = index - Double(lower)
        return _sorted[lower] * (1 - frac) + _sorted[upper] * frac
    }
}

// MARK: - Runner

/// Generic measurement harness for Copyable closures (Core ML, etc.).
/// ANEDirectBench inlines its own loop to avoid ~Copyable closure capture issues.
struct BenchmarkRunner: Sendable {
    let warmup: Int
    let iterations: Int
    let signposter: OSSignposter

    init(warmup: Int = 50, iterations: Int = 1000) {
        self.warmup = warmup
        self.iterations = iterations
        self.signposter = OSSignposter(subsystem: "com.espresso.bench", category: .pointsOfInterest)
    }

    func run(label: String, body: () throws -> Void) rethrows -> BenchmarkResult {
        // Warmup
        for _ in 0..<warmup {
            try body()
        }

        // Measured
        var latencies: [Double] = []
        latencies.reserveCapacity(iterations)

        for i in 0..<iterations {
            let state = signposter.beginInterval("Iteration")
            let start = ContinuousClock.now

            try body()

            let elapsed = ContinuousClock.now - start
            signposter.endInterval("Iteration", state)

            latencies.append(durationMs(elapsed))

            if (i + 1) % 100 == 0 {
                let currentMean = latencies.reduce(0, +) / Double(latencies.count)
                printStderr(String(format: "  [%@] %d/%d — mean: %.3f ms", label, i + 1, iterations, currentMean))
            }
        }

        return BenchmarkResult(
            label: label,
            latencies: latencies,
            warmupCount: warmup,
            iterationCount: iterations
        )
    }
}

// MARK: - Helpers

func durationMs(_ duration: Duration) -> Double {
    Double(duration.components.seconds) * 1000.0
        + Double(duration.components.attoseconds) / 1_000_000_000_000_000.0
}

func printStderr(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

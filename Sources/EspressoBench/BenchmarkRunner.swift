import Foundation
import os.signpost

struct BenchmarkResult: Sendable {
    let label: String
    let latencies: [Double]
    let warmupCount: Int
    let iterationCount: Int

    private let sortedLatencies: [Double]

    init(label: String, latencies: [Double], warmupCount: Int, iterationCount: Int) {
        self.label = label
        self.latencies = latencies
        self.warmupCount = warmupCount
        self.iterationCount = iterationCount
        self.sortedLatencies = latencies.sorted()
    }

    var mean: Double {
        guard !latencies.isEmpty else { return 0 }
        return latencies.reduce(0, +) / Double(latencies.count)
    }

    var median: Double { percentile(0.5) }
    var p50: Double { percentile(0.5) }
    var p95: Double { percentile(0.95) }
    var p99: Double { percentile(0.99) }
    var min: Double { sortedLatencies.first ?? 0 }
    var max: Double { sortedLatencies.last ?? 0 }

    var stddev: Double {
        guard !latencies.isEmpty else { return 0 }
        let mean = self.mean
        let variance = latencies.reduce(into: 0.0) { partialResult, latency in
            let delta = latency - mean
            partialResult += delta * delta
        } / Double(latencies.count)
        return variance.squareRoot()
    }

    func percentile(_ value: Double) -> Double {
        guard !sortedLatencies.isEmpty else { return 0 }

        let clamped = Swift.min(Swift.max(value, 0), 1)
        let index = clamped * Double(sortedLatencies.count - 1)
        let lowerIndex = Int(index)
        let upperIndex = Swift.min(lowerIndex + 1, sortedLatencies.count - 1)

        if lowerIndex == upperIndex {
            return sortedLatencies[lowerIndex]
        }

        let fraction = index - Double(lowerIndex)
        let lower = sortedLatencies[lowerIndex]
        let upper = sortedLatencies[upperIndex]
        return lower + ((upper - lower) * fraction)
    }
}

struct BenchmarkRunner: Sendable {
    private static let progressLocale = Locale(identifier: "en_US_POSIX")

    let warmup: Int
    let iterations: Int
    let log: OSLog

    init(warmup: Int = 50, iterations: Int = 1000, log: OSLog = .init(subsystem: "com.espresso.bench", category: "benchmark")) {
        self.warmup = warmup
        self.iterations = iterations
        self.log = log
    }

    func run(label: String, body: () throws -> Void) throws -> BenchmarkResult {
        for _ in 0..<warmup {
            try body()
        }

        var latencies: [Double] = []
        latencies.reserveCapacity(iterations)

        for iteration in 0..<iterations {
            let signpostID = OSSignpostID(log: log)
            let start = ContinuousClock.now
            os_signpost(.begin, log: log, name: "BenchmarkIteration", signpostID: signpostID, "%{public}s", label)

            try body()

            os_signpost(.end, log: log, name: "BenchmarkIteration", signpostID: signpostID, "%{public}s", label)
            let elapsed = ContinuousClock.now - start
            latencies.append(durationMs(elapsed))

            if (iteration + 1).isMultiple(of: 100) {
                let runningMean = latencies.reduce(0, +) / Double(latencies.count)
                printStderr(
                    String(
                        format: "[%@] %d/%d mean %.3f ms",
                        locale: Self.progressLocale,
                        label,
                        iteration + 1,
                        iterations,
                        runningMean
                    )
                )
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

@inline(__always)
func durationMs(_ duration: Duration) -> Double {
    let seconds = Double(duration.components.seconds) * 1_000.0
    let attoseconds = Double(duration.components.attoseconds) / 1_000_000_000_000_000.0
    return seconds + attoseconds
}

func printStderr(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

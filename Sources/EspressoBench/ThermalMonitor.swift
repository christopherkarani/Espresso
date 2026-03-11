import Foundation

enum ThermalMonitor {
    static func currentState() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:
            return "nominal"
        case .fair:
            return "fair"
        case .serious:
            return "serious"
        case .critical:
            return "critical"
        @unknown default:
            return "unknown"
        }
    }

    static func sustainedRun(
        duration: TimeInterval = 60.0,
        body: () throws -> Void
    ) throws -> (before: String, after: String, samples: [(time: Double, state: String)]) {
        precondition(duration > 0)

        let before = currentState()
        let start = ContinuousClock.now
        let deadline = start + .milliseconds(Int(duration * 1_000.0))
        var nextSampleTime = start + .seconds(1)
        var samples: [(time: Double, state: String)] = [(time: 0, state: before)]

        while ContinuousClock.now < deadline {
            try body()

            let now = ContinuousClock.now
            while now >= nextSampleTime {
                let elapsed = durationMs(nextSampleTime - start) / 1_000.0
                samples.append((time: elapsed, state: currentState()))
                nextSampleTime += .seconds(1)
            }
        }

        let after = currentState()
        let finalElapsed = durationMs(ContinuousClock.now - start) / 1_000.0
        if samples.last?.time != finalElapsed || samples.last?.state != after {
            samples.append((time: finalElapsed, state: after))
        }

        return (before: before, after: after, samples: samples)
    }
}

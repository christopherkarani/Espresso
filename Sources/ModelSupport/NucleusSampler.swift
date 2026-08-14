import Darwin

/// Temperature + nucleus (top-p) sampling over raw logits.
public enum NucleusSampler: Sendable {
    public static func sample<R: RandomNumberGenerator>(
        logits: [Float],
        temperature: Float,
        topP: Float,
        using rng: inout R
    ) -> Int {
        selectIndex(
            logits: logits,
            temperature: temperature,
            topP: topP,
            unitSample: Double.random(in: 0..<1, using: &rng)
        )
    }

    public static func selectIndex(
        logits: [Float],
        temperature: Float,
        topP: Float,
        unitSample: Double
    ) -> Int {
        guard !logits.isEmpty else { return 0 }
        if !temperature.isFinite || temperature <= 0 {
            return argmax(logits)
        }

        let maxLogit = logits.max() ?? 0
        var weights = [Double](repeating: 0, count: logits.count)
        var total = 0.0
        for index in logits.indices {
            let value = exp(Double((logits[index] - maxLogit) / temperature))
            weights[index] = value
            total += value
        }
        guard total.isFinite, total > 0 else {
            return argmax(logits)
        }
        for index in weights.indices {
            weights[index] /= total
        }

        var order = Array(logits.indices)
        order.sort { weights[$0] > weights[$1] }

        let clippedTopP = Double(min(max(topP, 0), 1))
        var nucleus: [Int] = []
        if clippedTopP >= 1 {
            nucleus = order
        } else {
            var cumulative = 0.0
            for index in order {
                nucleus.append(index)
                cumulative += weights[index]
                if cumulative >= clippedTopP {
                    break
                }
            }
        }
        if nucleus.isEmpty {
            nucleus = [order[0]]
        }

        let nucleusMass = nucleus.reduce(0.0) { $0 + weights[$1] }
        guard nucleusMass > 0 else {
            return nucleus[0]
        }
        var threshold = min(max(unitSample, 0), 0.999999999) * nucleusMass
        for index in nucleus {
            threshold -= weights[index]
            if threshold <= 0 {
                return index
            }
        }
        return nucleus[nucleus.count - 1]
    }

    private static func argmax(_ logits: [Float]) -> Int {
        logits.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
    }
}

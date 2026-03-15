import Foundation

public struct MILDiff: Sendable {
    public init() {}

    public static func textDiff(_ lhs: String, _ rhs: String) -> [String] {
        let left = normalizedLines(lhs)
        let right = normalizedLines(rhs)
        let sharedCount = min(left.count, right.count)

        var diff: [String] = []
        diff.reserveCapacity(max(left.count, right.count))

        for index in 0..<sharedCount where left[index] != right[index] {
            diff.append("-\(left[index])")
            diff.append("+\(right[index])")
        }

        if left.count > sharedCount {
            diff.append(contentsOf: left[sharedCount...].map { "-\($0)" })
        }

        if right.count > sharedCount {
            diff.append(contentsOf: right[sharedCount...].map { "+\($0)" })
        }

        return diff
    }

    public static func structuralEquiv(_ lhs: String, _ rhs: String) -> Bool {
        extractOps(lhs) == extractOps(rhs)
    }

    public static func extractOps(_ mil: String) -> [String] {
        let pattern = #"=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\("#
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return []
        }

        return mil.split(whereSeparator: \.isNewline).compactMap { rawLine in
            let line = String(rawLine)
            let range = NSRange(line.startIndex..<line.endIndex, in: line)
            guard let match = regex.firstMatch(in: line, range: range) else {
                return nil
            }
            guard let opRange = Range(match.range(at: 1), in: line) else {
                return nil
            }

            let op = String(line[opRange])
            return op == "const" ? nil : op
        }
    }

    private static func normalizedLines(_ text: String) -> [String] {
        text.split(whereSeparator: \.isNewline)
            .map { line in
                line.reduce(into: "") { partialResult, character in
                    if !character.isWhitespace {
                        partialResult.append(character)
                    }
                }
            }
            .filter { !$0.isEmpty }
    }
}

import Foundation

public protocol FutureTokenProposingLanguageModel: ~Copyable, DirectTokenSelectingLanguageModel {
    mutating func proposeFutureToken(
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> UInt16
}

public struct OfflineExactAcceptanceTrace: Sendable, Equatable {
    public let promptTokens: [UInt16]
    public let generatedTokens: [UInt16]
    public let committedExactTokenCounts: [Int]
    public let acceptedFutureTokenCounts: [Int]
    public let parityMatchedAllCommittedTokens: Bool

    public init(
        promptTokens: [UInt16],
        generatedTokens: [UInt16],
        committedExactTokenCounts: [Int],
        acceptedFutureTokenCounts: [Int],
        parityMatchedAllCommittedTokens: Bool
    ) {
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
        self.committedExactTokenCounts = committedExactTokenCounts
        self.acceptedFutureTokenCounts = acceptedFutureTokenCounts
        self.parityMatchedAllCommittedTokens = parityMatchedAllCommittedTokens
    }

    public var committedExactTokensPerPass: Double {
        guard !committedExactTokenCounts.isEmpty else { return 0 }
        let total = committedExactTokenCounts.reduce(0, +)
        return Double(total) / Double(committedExactTokenCounts.count)
    }

    public var acceptedFutureTokensPerPass: Double {
        guard !acceptedFutureTokenCounts.isEmpty else { return 0 }
        let total = acceptedFutureTokenCounts.reduce(0, +)
        return Double(total) / Double(acceptedFutureTokenCounts.count)
    }
}

public enum OfflineExactAcceptanceEvaluator {
    public static func evaluate<Teacher, Student>(
        teacher: inout Teacher,
        student: inout Student,
        promptTokens: [UInt16],
        maxNewTokens: Int,
        strategy: TokenSelectionStrategy
    ) throws(GenerationError) -> OfflineExactAcceptanceTrace
    where Teacher: DirectTokenSelectingLanguageModel, Teacher: ~Copyable,
          Student: FutureTokenProposingLanguageModel, Student: ~Copyable {
        guard !promptTokens.isEmpty else {
            throw .invalidArguments("promptTokens must not be empty")
        }
        guard maxNewTokens > 0 else {
            throw .invalidArguments("maxNewTokens must be > 0")
        }
        guard teacher.vocabSize == student.vocabSize else {
            throw .invalidArguments("teacher/student vocab sizes must match")
        }

        try teacher.reset()
        try student.reset()

        var teacherCurrent = try teacher.prefillSelectedToken(promptTokens: promptTokens, strategy: strategy)
        var studentCurrent = try student.prefillSelectedToken(promptTokens: promptTokens, strategy: strategy)

        var generatedTokens: [UInt16] = []
        var committedExactTokenCounts: [Int] = []
        var acceptedFutureTokenCounts: [Int] = []
        var parityMatchedAllCommittedTokens = (teacherCurrent == studentCurrent)

        if !parityMatchedAllCommittedTokens {
            return OfflineExactAcceptanceTrace(
                promptTokens: promptTokens,
                generatedTokens: [],
                committedExactTokenCounts: [],
                acceptedFutureTokenCounts: [],
                parityMatchedAllCommittedTokens: false
            )
        }

        while generatedTokens.count < maxNewTokens {
            let remainingTokenBudget = maxNewTokens - generatedTokens.count
            let proposedFuture = try student.proposeFutureToken(strategy: strategy)

            let teacherNext = try teacher.decodeSelectedToken(nextToken: teacherCurrent, strategy: strategy)
            let studentNext = try student.decodeSelectedToken(nextToken: studentCurrent, strategy: strategy)

            generatedTokens.append(teacherCurrent)

            let nextParityMatches = (teacherNext == studentNext)
            if !nextParityMatches {
                committedExactTokenCounts.append(1)
                acceptedFutureTokenCounts.append(0)
                parityMatchedAllCommittedTokens = false
                break
            }

            let canAcceptFuture = remainingTokenBudget > 1 && proposedFuture == teacherNext
            if canAcceptFuture {
                generatedTokens.append(teacherNext)
                committedExactTokenCounts.append(2)
                acceptedFutureTokenCounts.append(1)

                let teacherFuture = try teacher.decodeSelectedToken(nextToken: teacherNext, strategy: strategy)
                let studentFuture = try student.decodeSelectedToken(nextToken: studentNext, strategy: strategy)

                teacherCurrent = teacherFuture
                studentCurrent = studentFuture
                if teacherFuture != studentFuture {
                    parityMatchedAllCommittedTokens = false
                    break
                }
                continue
            }

            committedExactTokenCounts.append(1)
            acceptedFutureTokenCounts.append(0)
            teacherCurrent = teacherNext
            studentCurrent = studentNext
        }

        return OfflineExactAcceptanceTrace(
            promptTokens: promptTokens,
            generatedTokens: generatedTokens,
            committedExactTokenCounts: committedExactTokenCounts,
            acceptedFutureTokenCounts: acceptedFutureTokenCounts,
            parityMatchedAllCommittedTokens: parityMatchedAllCommittedTokens
        )
    }
}

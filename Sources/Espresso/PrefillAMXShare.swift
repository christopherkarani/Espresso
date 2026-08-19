import Accelerate
import Foundation
import Dispatch
import ANEInterop

/// Leftover output channels of a **batched** prefill GEMM (seq=64/256), never N=1 decode FFN.
public enum PrefillAMXShareTarget: String, Sendable, Equatable {
    case batchedPrefillLeftoverChannels
}

public struct ChatChunkMixTrace: Sendable, Equatable {
    public let promptTokenCounts: [Int]

    public init(promptTokenCounts: [Int]) {
        self.promptTokenCounts = promptTokenCounts
    }

    public var medianPromptTokens: Int {
        guard !promptTokenCounts.isEmpty else { return 0 }
        let sorted = promptTokenCounts.sorted()
        return sorted[sorted.count / 2]
    }
}

public struct PrefillAMXShareDecision: Sendable, Equatable {
    public let enabled: Bool
    public let leftoverChannelFraction: Double
    public let leftoverChannelCount: Int
    public let target: PrefillAMXShareTarget

    public static let off = PrefillAMXShareDecision(
        enabled: false,
        leftoverChannelFraction: 0,
        leftoverChannelCount: 0,
        target: .batchedPrefillLeftoverChannels
    )
}

/// Hardware-local leftover share. Fractions come from ANE channel alignment remainder
/// and the actual chat chunk mix — never omlx's 14/20/13 table.
public struct PrefillAMXShareTuner: Sendable {
    public let aneChannelAlignment: Int

    public init(aneChannelAlignment: Int = 32) {
        self.aneChannelAlignment = max(aneChannelAlignment, 1)
    }

    public func decide(trace: ChatChunkMixTrace, outputChannels: Int) -> PrefillAMXShareDecision {
        guard outputChannels > 0 else { return .off }
        let leftover = outputChannels % aneChannelAlignment
        let decodeLike = trace.promptTokenCounts.isEmpty
            || trace.promptTokenCounts.allSatisfy { $0 <= 4 }
        let prefillHeavy = trace.medianPromptTokens >= 32
        if leftover == 0 || decodeLike || !prefillHeavy {
            return .off
        }
        return PrefillAMXShareDecision(
            enabled: true,
            leftoverChannelFraction: Double(leftover) / Double(outputChannels),
            leftoverChannelCount: leftover,
            target: .batchedPrefillLeftoverChannels
        )
    }
}

public enum BatchedPrefillGEMM {
    /// Batched prefill GEMM. When share is on, leftover **output channels** go to
    /// 8-shard BNNS AMX; the prefix stays on the batched GEMM (ANE stand-in).
    public static func multiply(
        sequenceLength: Int,
        dim: Int,
        outputChannels: Int,
        input: [Float],
        weights: [Float],
        share: PrefillAMXShareDecision
    ) -> [Float] {
        precondition(QwenPrefillCompare.sequenceLengths.contains(sequenceLength))
        precondition(dim > 0)
        precondition(outputChannels > 0)
        precondition(input.count == sequenceLength * dim)
        precondition(weights.count == outputChannels * dim)
        precondition(share.target == .batchedPrefillLeftoverChannels)

        let leftover = share.enabled ? share.leftoverChannelCount : 0
        let aneChannels = outputChannels - leftover
        if leftover <= 0 || aneChannels <= 0 {
            return gemm(sequenceLength: sequenceLength, dim: dim, outputChannels: outputChannels, input: input, weights: weights)
        }

        let aneWeights = Array(weights.prefix(aneChannels * dim))
        let amxWeights = Array(weights.suffix(leftover * dim))
        let aneOut = gemm(
            sequenceLength: sequenceLength,
            dim: dim,
            outputChannels: aneChannels,
            input: input,
            weights: aneWeights
        )
        let amxOut = leftoverAMX(
            sequenceLength: sequenceLength,
            dim: dim,
            leftoverChannels: leftover,
            input: input,
            weights: amxWeights
        )

        var stitched = [Float](repeating: 0, count: sequenceLength * outputChannels)
        for token in 0..<sequenceLength {
            let dst = token * outputChannels
            let aneSrc = token * aneChannels
            stitched.replaceSubrange(dst..<(dst + aneChannels), with: aneOut[aneSrc..<(aneSrc + aneChannels)])
            let amxSrc = token * leftover
            stitched.replaceSubrange(
                (dst + aneChannels)..<(dst + outputChannels),
                with: amxOut[amxSrc..<(amxSrc + leftover)]
            )
        }
        return stitched
    }

    private static func gemm(
        sequenceLength: Int,
        dim: Int,
        outputChannels: Int,
        input: [Float],
        weights: [Float]
    ) -> [Float] {
        var out = [Float](repeating: 0, count: sequenceLength * outputChannels)
        input.withUnsafeBufferPointer { x in
            weights.withUnsafeBufferPointer { w in
                out.withUnsafeMutableBufferPointer { y in
                    BLAS.sgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasTrans,
                        m: Int32(sequenceLength),
                        n: Int32(outputChannels),
                        k: Int32(dim),
                        alpha: 1.0,
                        a: x.baseAddress!,
                        lda: Int32(dim),
                        b: w.baseAddress!,
                        ldb: Int32(dim),
                        beta: 0.0,
                        c: y.baseAddress!,
                        ldc: Int32(outputChannels)
                    )
                }
            }
        }
        return out
    }

    private static func leftoverAMX(
        sequenceLength: Int,
        dim: Int,
        leftoverChannels: Int,
        input: [Float],
        weights: [Float]
    ) -> [Float] {
        let outPtr = UnsafeMutablePointer<Float>.allocate(capacity: sequenceLength * leftoverChannels)
        outPtr.initialize(repeating: 0, count: sequenceLength * leftoverChannels)
        defer {
            outPtr.deinitialize(count: sequenceLength * leftoverChannels)
            outPtr.deallocate()
        }
        let shardCount = FP16TiledClassifier.eightShardCount
        nonisolated(unsafe) let outUnsafe = outPtr
        DispatchQueue.concurrentPerform(iterations: shardCount) { shard in
            let range = FP16TiledClassifier.shardRowRange(
                vocabSize: leftoverChannels,
                shard: shard,
                shardCount: shardCount
            )
            guard !range.isEmpty else { return }
            _ = ane_interop_amx_shared_resource_hint(1, Int32(shard), 2)
            defer { _ = ane_interop_amx_shared_resource_hint(0, Int32(shard), 2) }

            let rows = range.count
            let shardOut = UnsafeMutablePointer<Float>.allocate(capacity: sequenceLength * rows)
            defer { shardOut.deallocate() }
            weights.withUnsafeBufferPointer { w in
                input.withUnsafeBufferPointer { x in
                    BLAS.sgemm(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasTrans,
                        m: Int32(sequenceLength),
                        n: Int32(rows),
                        k: Int32(dim),
                        alpha: 1.0,
                        a: x.baseAddress!,
                        lda: Int32(dim),
                        b: w.baseAddress!.advanced(by: range.lowerBound * dim),
                        ldb: Int32(dim),
                        beta: 0.0,
                        c: shardOut,
                        ldc: Int32(rows)
                    )
                }
            }
            for token in 0..<sequenceLength {
                let dst = outUnsafe.advanced(by: token * leftoverChannels + range.lowerBound)
                let src = shardOut.advanced(by: token * rows)
                dst.update(from: src, count: rows)
            }
        }
        return Array(UnsafeBufferPointer(start: outPtr, count: sequenceLength * leftoverChannels))
    }
}

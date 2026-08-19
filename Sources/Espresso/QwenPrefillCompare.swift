import Accelerate
import Foundation
import IOSurface
import ANERuntime
import ANETypes

/// N=1 token loop vs one batched GEMM for Qwen prefill leftover-channel work.
///
/// Sequence lengths are the fixed ANE prefill shapes (64 and 256). Outputs are
/// token-major `[seq * outputChannels]`.
public struct QwenPrefillCompareResult: Sendable, Equatable {
    public let sequenceLength: Int
    public let n1Hidden: [Float]
    public let batchedHidden: [Float]
    public let n1WallMs: Double
    public let batchedWallMs: Double
    public let matchTolerance: Float

    public var outputsMatch: Bool {
        n1Hidden.count == batchedHidden.count &&
            zip(n1Hidden, batchedHidden).allSatisfy { abs($0 - $1) <= matchTolerance }
    }
}

public enum QwenPrefillCompare {
    public static let sequenceLengths: [Int] = HybridDecodeKernelSet.prefillSequenceLengths

    /// Compiled seq=64/256 QKV kernel (one `eval`) vs the N=1 hybrid QKV loop
    /// (`decodeQKVOnly.eval` once per prompt token) on the same prompt IDs.
    public static func compareHybridQKV(
        sequenceLength: Int,
        weights: borrowing LayerWeights
    ) throws -> QwenPrefillCompareResult {
        guard sequenceLengths.contains(sequenceLength) else {
            throw ANEError.invalidArguments(
                "Qwen prefill sequenceLength must be 64 or 256, got \(sequenceLength)"
            )
        }
        let dim = weights.dim
        let qDim = weights.qDim
        let n1Kernels = try HybridDecodeKernelSet(weights: weights, maxSeq: sequenceLength)
        let batchedKernels = try HybridDecodeKernelSet(
            prefillWeights: weights,
            sequenceLength: sequenceLength
        )
        let n1Handles = try HybridDecodeSurfaceHandles(
            kernels: n1Kernels,
            logicalMaxSeq: sequenceLength,
            dim: dim,
            qDim: qDim,
            kvDim: weights.kvDim
        )
        let batchedHandles = try HybridDecodeSurfaceHandles(
            kernels: batchedKernels,
            logicalMaxSeq: sequenceLength,
            dim: dim,
            qDim: qDim,
            kvDim: weights.kvDim
        )

        var tokens = [[Float]](repeating: [Float](repeating: 0, count: dim), count: sequenceLength)
        for token in 0..<sequenceLength {
            for channel in 0..<dim {
                tokens[token][channel] = Float((token * 17 + channel) % 13) * 0.01
            }
        }

        let n1Start = DispatchTime.now().uptimeNanoseconds
        var n1Hidden = [Float](repeating: 0, count: sequenceLength * qDim)
        for token in 0..<sequenceLength {
            try mapSurfaceIOToANEError {
                try tokens[token].withUnsafeBufferPointer { buf in
                    try SurfaceIO.writeFP16SpatialSlice(
                        to: n1Handles.qkvIn,
                        channelOffset: 0,
                        spatialIndex: 0,
                        spatial: n1Handles.laneSpatial,
                        data: buf,
                        channels: dim
                    )
                }
            }
            try n1Kernels.decodeQKVOnly.eval()
            try mapSurfaceIOToANEError {
                try n1Hidden.withUnsafeMutableBufferPointer { out in
                    let slice = UnsafeMutableBufferPointer(
                        start: out.baseAddress!.advanced(by: token * qDim),
                        count: qDim
                    )
                    try SurfaceIO.readFP16SpatialSlice(
                        from: n1Handles.qOut,
                        channelOffset: 0,
                        spatialIndex: 0,
                        spatial: n1Handles.laneSpatial,
                        into: slice,
                        channels: qDim
                    )
                }
            }
        }
        let n1Ms = Double(DispatchTime.now().uptimeNanoseconds - n1Start) / 1_000_000

        let batchedStart = DispatchTime.now().uptimeNanoseconds
        for token in 0..<sequenceLength {
            try mapSurfaceIOToANEError {
                try tokens[token].withUnsafeBufferPointer { buf in
                    try SurfaceIO.writeFP16SpatialSlice(
                        to: batchedHandles.qkvIn,
                        channelOffset: 0,
                        spatialIndex: token,
                        spatial: batchedHandles.laneSpatial,
                        data: buf,
                        channels: dim
                    )
                }
            }
        }
        try batchedKernels.decodeQKVOnly.eval()
        var batchedHidden = [Float](repeating: 0, count: sequenceLength * qDim)
        for token in 0..<sequenceLength {
            try mapSurfaceIOToANEError {
                try batchedHidden.withUnsafeMutableBufferPointer { out in
                    let slice = UnsafeMutableBufferPointer(
                        start: out.baseAddress!.advanced(by: token * qDim),
                        count: qDim
                    )
                    try SurfaceIO.readFP16SpatialSlice(
                        from: batchedHandles.qOut,
                        channelOffset: 0,
                        spatialIndex: token,
                        spatial: batchedHandles.laneSpatial,
                        into: slice,
                        channels: qDim
                    )
                }
            }
        }
        let batchedMs = Double(DispatchTime.now().uptimeNanoseconds - batchedStart) / 1_000_000

        return QwenPrefillCompareResult(
            sequenceLength: sequenceLength,
            n1Hidden: n1Hidden,
            batchedHidden: batchedHidden,
            n1WallMs: n1Ms,
            batchedWallMs: batchedMs,
            matchTolerance: 2e-2
        )
    }

    /// Sequential N=1 GEMVs vs one GEMM on the same prompt IDs / weights.
    public static func compareCPUMatmul(
        sequenceLength: Int,
        dim: Int,
        outputChannels: Int,
        input: [Float],
        weights: [Float]
    ) -> QwenPrefillCompareResult {
        precondition(sequenceLengths.contains(sequenceLength))
        precondition(dim > 0)
        precondition(outputChannels > 0)
        precondition(input.count == sequenceLength * dim)
        precondition(weights.count == outputChannels * dim)

        let n1Start = DispatchTime.now().uptimeNanoseconds
        var n1 = [Float](repeating: 0, count: sequenceLength * outputChannels)
        input.withUnsafeBufferPointer { x in
            weights.withUnsafeBufferPointer { w in
                n1.withUnsafeMutableBufferPointer { y in
                    for token in 0..<sequenceLength {
                        let xTok = x.baseAddress!.advanced(by: token * dim)
                        let yTok = y.baseAddress!.advanced(by: token * outputChannels)
                        BLAS.sgemm(
                            CblasRowMajor,
                            CblasNoTrans,
                            CblasTrans,
                            m: 1,
                            n: Int32(outputChannels),
                            k: Int32(dim),
                            alpha: 1.0,
                            a: xTok,
                            lda: Int32(dim),
                            b: w.baseAddress!,
                            ldb: Int32(dim),
                            beta: 0.0,
                            c: yTok,
                            ldc: Int32(outputChannels)
                        )
                    }
                }
            }
        }
        let n1Ms = Double(DispatchTime.now().uptimeNanoseconds - n1Start) / 1_000_000

        let batchedStart = DispatchTime.now().uptimeNanoseconds
        var batched = [Float](repeating: 0, count: sequenceLength * outputChannels)
        input.withUnsafeBufferPointer { x in
            weights.withUnsafeBufferPointer { w in
                batched.withUnsafeMutableBufferPointer { y in
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
        let batchedMs = Double(DispatchTime.now().uptimeNanoseconds - batchedStart) / 1_000_000

        return QwenPrefillCompareResult(
            sequenceLength: sequenceLength,
            n1Hidden: n1,
            batchedHidden: batched,
            n1WallMs: n1Ms,
            batchedWallMs: batchedMs,
            matchTolerance: 1e-4
        )
    }
}

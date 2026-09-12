import Foundation
import Metal

public enum MetalLMHeadError: Error, Equatable {
    case metalUnavailable
    case commandQueueUnavailable
    case libraryBuildFailed(String)
    case pipelineBuildFailed(String)
    case bufferAllocationFailed
    case commandBufferUnavailable
    case commandEncoderUnavailable
    case commandExecutionFailed(String)
    case invalidArguments(String)
}

/// Metal GPU greedy LM head: `argmax(W · h)` over an FP16 `[vocab, dim]` weight
/// matrix and an FP32 hidden vector, returning only the winning token index.
///
/// Replaces the CPU `FP16TiledClassifier` for heads too large for ANE SRAM
/// (Qwen2.5-1.5B: 151 936 × 1 536 ≈ 467 MB). The weight matrix is streamed
/// exactly once per token from GPU-shared memory; the full logit vector is
/// never materialized.
///
/// Two-stage parallel reduction:
///  1. `lm_head_gemv_argmax_partial` — one SIMD group per vocab row (32 lanes
///     read 256 contiguous FP16 values per iteration, coalesced), FP32 FMA
///     accumulation, `simd_sum`; each threadgroup reduces its rows to one
///     `(max, index)` partial.
///  2. `lm_head_argmax_reduce` — a single threadgroup folds all partials into
///     one `UInt32` token id.
///
/// Ties resolve to the lowest vocab index, matching the first-max rule of the
/// CPU tiled path. Accumulation order differs from BLAS so logits that sit
/// inside FMA-association noise (≈1e-6 relative) may still disagree.
public final class MetalLMHeadArgmax {
    /// Threads per stage-1 threadgroup (8 SIMD groups of 32 lanes).
    static let stage1Threads = 256
    /// Vocab rows handled by one stage-1 threadgroup (8 SIMD groups × 8 rows).
    static let rowsPerThreadgroup = 16
    static let stage2Threads = 1024

    private struct Partial {
        var value: Float
        var index: UInt32
    }

    private struct Params {
        var vocab: UInt32
        var dim: UInt32
        var rowsPerThreadgroup: UInt32
        var partialCount: UInt32
    }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let partialPipeline: MTLComputePipelineState
    private let reducePipeline: MTLComputePipelineState
    private let weightBuffer: MTLBuffer
    private let hiddenBuffer: MTLBuffer
    private let partialBuffer: MTLBuffer
    private let outputBuffer: MTLBuffer
    private let params: Params
    private let threadgroupCount: Int

    public let vocabSize: Int
    public let dim: Int

    /// Bytes of FP16 weights streamed per `argmax` call.
    public var weightBytes: Int { vocabSize * dim * MemoryLayout<UInt16>.stride }

    /// Static shape constraint: each of the 32 lanes owns whole `half8` chunks per iteration.
    public static func supports(dim: Int) -> Bool {
        dim > 0 && dim % 256 == 0
    }

    /// - Parameters:
    ///   - weightsFP16: Row-major `[vocab, dim]` IEEE FP16 bit patterns. Copied once
    ///     into a GPU-shared `MTLBuffer`.
    ///   - vocabSize: Number of rows (token ids).
    ///   - dim: Hidden size; must be a multiple of 256 so each SIMD lane owns a
    ///     whole `half8` per iteration.
    public init(
        weightsFP16: UnsafeBufferPointer<UInt16>,
        vocabSize: Int,
        dim: Int
    ) throws(MetalLMHeadError) {
        guard vocabSize > 0, dim > 0 else {
            throw .invalidArguments("vocabSize and dim must be > 0")
        }
        guard Self.supports(dim: dim) else {
            throw .invalidArguments("dim \(dim) must be a multiple of 256")
        }
        guard vocabSize <= Int(UInt32.max), dim <= Int(UInt32.max) else {
            throw .invalidArguments("vocabSize \(vocabSize) and dim \(dim) must fit UInt32")
        }
        let (elementCount, elementOverflow) = vocabSize.multipliedReportingOverflow(by: dim)
        let (weightByteCount, byteOverflow) = elementCount.multipliedReportingOverflow(by: MemoryLayout<UInt16>.stride)
        guard !elementOverflow, !byteOverflow else {
            throw .invalidArguments("vocabSize \(vocabSize) x dim \(dim) overflows the weight byte count")
        }
        guard weightsFP16.count >= elementCount, let weightBase = weightsFP16.baseAddress else {
            throw .invalidArguments("weightsFP16 has \(weightsFP16.count) elements, need \(elementCount)")
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw .metalUnavailable
        }
        let hiddenThreadgroupBytes = dim * MemoryLayout<Float>.stride
        guard hiddenThreadgroupBytes <= device.maxThreadgroupMemoryLength else {
            throw .invalidArguments(
                "dim \(dim) needs \(hiddenThreadgroupBytes) threadgroup bytes, device allows \(device.maxThreadgroupMemoryLength)"
            )
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw .commandQueueUnavailable
        }

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: Self.shaderSource, options: nil)
        } catch {
            throw .libraryBuildFailed(String(describing: error))
        }
        guard let partialFn = library.makeFunction(name: "lm_head_gemv_argmax_partial") else {
            throw .libraryBuildFailed("missing lm_head_gemv_argmax_partial")
        }
        guard let reduceFn = library.makeFunction(name: "lm_head_argmax_reduce") else {
            throw .libraryBuildFailed("missing lm_head_argmax_reduce")
        }
        let partialPipeline: MTLComputePipelineState
        let reducePipeline: MTLComputePipelineState
        do {
            partialPipeline = try device.makeComputePipelineState(function: partialFn)
            reducePipeline = try device.makeComputePipelineState(function: reduceFn)
        } catch {
            throw .pipelineBuildFailed(String(describing: error))
        }
        guard partialPipeline.maxTotalThreadsPerThreadgroup >= Self.stage1Threads,
              reducePipeline.maxTotalThreadsPerThreadgroup >= Self.stage2Threads else {
            throw .pipelineBuildFailed("device threadgroup limit below required stage sizes")
        }

        guard let weightBuffer = device.makeBuffer(
            bytes: UnsafeRawPointer(weightBase),
            length: weightByteCount,
            options: .storageModeShared
        ) else {
            throw .bufferAllocationFailed
        }
        guard let hiddenBuffer = device.makeBuffer(
            length: dim * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw .bufferAllocationFailed
        }
        let threadgroupCount = (vocabSize + Self.rowsPerThreadgroup - 1) / Self.rowsPerThreadgroup
        guard let partialBuffer = device.makeBuffer(
            length: threadgroupCount * MemoryLayout<Partial>.stride,
            options: .storageModeShared
        ) else {
            throw .bufferAllocationFailed
        }
        guard let outputBuffer = device.makeBuffer(
            length: MemoryLayout<UInt32>.stride,
            options: .storageModeShared
        ) else {
            throw .bufferAllocationFailed
        }

        self.device = device
        self.commandQueue = commandQueue
        self.partialPipeline = partialPipeline
        self.reducePipeline = reducePipeline
        self.weightBuffer = weightBuffer
        self.hiddenBuffer = hiddenBuffer
        self.partialBuffer = partialBuffer
        self.outputBuffer = outputBuffer
        self.threadgroupCount = threadgroupCount
        self.vocabSize = vocabSize
        self.dim = dim
        self.params = Params(
            vocab: UInt32(vocabSize),
            dim: UInt32(dim),
            rowsPerThreadgroup: UInt32(Self.rowsPerThreadgroup),
            partialCount: UInt32(threadgroupCount)
        )
    }

    /// Greedy token for one FP32 hidden vector of length `dim`.
    public func argmax(hidden: UnsafeBufferPointer<Float>) throws(MetalLMHeadError) -> Int {
        guard hidden.count == dim, let hiddenBase = hidden.baseAddress else {
            throw .invalidArguments("hidden has \(hidden.count) elements, expected \(dim)")
        }
        memcpy(hiddenBuffer.contents(), hiddenBase, dim * MemoryLayout<Float>.stride)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw .commandBufferUnavailable
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        var params = self.params

        encoder.setComputePipelineState(partialPipeline)
        encoder.setBuffer(weightBuffer, offset: 0, index: 0)
        encoder.setBuffer(hiddenBuffer, offset: 0, index: 1)
        encoder.setBuffer(partialBuffer, offset: 0, index: 2)
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 3)
        encoder.setThreadgroupMemoryLength(dim * MemoryLayout<Float>.stride, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: threadgroupCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.stage1Threads, height: 1, depth: 1)
        )

        encoder.setComputePipelineState(reducePipeline)
        encoder.setBuffer(partialBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.stage2Threads, height: 1, depth: 1)
        )
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if commandBuffer.status != .completed {
            throw .commandExecutionFailed(
                commandBuffer.error?.localizedDescription ?? "status=\(commandBuffer.status.rawValue)"
            )
        }
        return Int(outputBuffer.contents().load(as: UInt32.self))
    }

    /// Convenience over a Swift array.
    public func argmax(hidden: [Float]) throws(MetalLMHeadError) -> Int {
        try hidden.withUnsafeBufferPointer { (buffer) throws(MetalLMHeadError) -> Int in
            try argmax(hidden: buffer)
        }
    }

    // MARK: - Shaders

    static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct LMHeadParams {
        uint vocab;
        uint dim;
        uint rowsPerThreadgroup;
        uint partialCount;
    };

    struct Partial {
        float value;
        uint index;
    };

    /// Lowest index wins ties so the result is order independent.
    inline void take_better(thread float &bestVal, thread uint &bestIdx, float val, uint idx) {
        if (val > bestVal || (val == bestVal && idx < bestIdx)) {
            bestVal = val;
            bestIdx = idx;
        }
    }

    /// Stage 1: each SIMD group streams whole rows; 32 lanes × half8 = 256
    /// contiguous FP16 per iteration. Threadgroup emits one (max, index).
    kernel void lm_head_gemv_argmax_partial(
        const device half *weights [[buffer(0)]],
        const device float *hidden [[buffer(1)]],
        device Partial *partials [[buffer(2)]],
        constant LMHeadParams &params [[buffer(3)]],
        threadgroup float *sharedHidden [[threadgroup(0)]],
        uint groupId [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint tpg [[threads_per_threadgroup]],
        uint simdId [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]
    ) {
        const uint dim = params.dim;
        for (uint c = tid; c < dim; c += tpg) {
            sharedHidden[c] = hidden[c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint simdCount = tpg / 32;
        const uint rowBase = groupId * params.rowsPerThreadgroup;
        const uint rowEnd = min(rowBase + params.rowsPerThreadgroup, params.vocab);
        const uint lanesStride = 32 * 8;

        float bestVal = -INFINITY;
        uint bestIdx = 0xFFFFFFFFu;

        for (uint row = rowBase + simdId; row < rowEnd; row += simdCount) {
            const device half *wRow = weights + (ulong)row * dim;
            float acc = 0.0f;
            for (uint base = lane * 8; base < dim; base += lanesStride) {
                half4 w0 = *((const device half4 *)(wRow + base));
                half4 w1 = *((const device half4 *)(wRow + base + 4));
                float4 h0 = *((threadgroup float4 *)(sharedHidden + base));
                float4 h1 = *((threadgroup float4 *)(sharedHidden + base + 4));
                acc = fma(float(w0.x), h0.x, acc);
                acc = fma(float(w0.y), h0.y, acc);
                acc = fma(float(w0.z), h0.z, acc);
                acc = fma(float(w0.w), h0.w, acc);
                acc = fma(float(w1.x), h1.x, acc);
                acc = fma(float(w1.y), h1.y, acc);
                acc = fma(float(w1.z), h1.z, acc);
                acc = fma(float(w1.w), h1.w, acc);
            }
            float logit = simd_sum(acc);
            take_better(bestVal, bestIdx, logit, row);
        }

        // One (max, index) per SIMD group; lane 0 holds the group's result.
        threadgroup float sharedMax[32];
        threadgroup uint sharedIdx[32];
        if (lane == 0) {
            sharedMax[simdId] = bestVal;
            sharedIdx[simdId] = bestIdx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float groupVal = sharedMax[0];
            uint groupIdx = sharedIdx[0];
            for (uint s = 1; s < simdCount; s++) {
                take_better(groupVal, groupIdx, sharedMax[s], sharedIdx[s]);
            }
            partials[groupId].value = groupVal;
            partials[groupId].index = groupIdx;
        }
    }

    /// Stage 2: single threadgroup folds every partial into one token id.
    kernel void lm_head_argmax_reduce(
        const device Partial *partials [[buffer(0)]],
        device uint *tokenOut [[buffer(1)]],
        constant LMHeadParams &params [[buffer(2)]],
        uint tid [[thread_index_in_threadgroup]],
        uint tpg [[threads_per_threadgroup]],
        uint simdId [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]
    ) {
        float bestVal = -INFINITY;
        uint bestIdx = 0xFFFFFFFFu;
        for (uint p = tid; p < params.partialCount; p += tpg) {
            take_better(bestVal, bestIdx, partials[p].value, partials[p].index);
        }

        for (ushort offset = 16; offset > 0; offset >>= 1) {
            float otherVal = simd_shuffle_down(bestVal, offset);
            uint otherIdx = simd_shuffle_down(bestIdx, offset);
            take_better(bestVal, bestIdx, otherVal, otherIdx);
        }

        threadgroup float sharedMax[32];
        threadgroup uint sharedIdx[32];
        if (lane == 0) {
            sharedMax[simdId] = bestVal;
            sharedIdx[simdId] = bestIdx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            const uint simdCount = tpg / 32;
            float finalVal = sharedMax[0];
            uint finalIdx = sharedIdx[0];
            for (uint s = 1; s < simdCount; s++) {
                take_better(finalVal, finalIdx, sharedMax[s], sharedIdx[s]);
            }
            tokenOut[0] = finalIdx;
        }
    }
    """
}

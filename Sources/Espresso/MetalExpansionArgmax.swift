import Foundation
import Metal
import MetalPerformanceShaders
import IOSurface
import ANETypes

@inline(__always)
private func requireUInt16TokenIDs(vocabSize: Int) throws(MetalExpansionArgmaxError) {
    guard vocabSize > 0, vocabSize <= Int(UInt16.max) + 1 else {
        throw .invalidArguments("vocabSize \(vocabSize) exceeds UInt16 token-id capacity")
    }
}

@inline(__always)
private func requiredFP16SurfaceBytes(channels: Int, spatial: Int) throws(MetalExpansionArgmaxError) -> Int {
    let elementCount = channels.multipliedReportingOverflow(by: spatial)
    guard !elementCount.overflow else {
        throw .invalidArguments("surface shape overflow for channels \(channels) and spatial \(spatial)")
    }
    let bytes = elementCount.partialValue.multipliedReportingOverflow(by: MemoryLayout<Float16>.stride)
    guard !bytes.overflow else {
        throw .invalidArguments("surface byte count overflow for channels \(channels) and spatial \(spatial)")
    }
    return bytes.partialValue
}

@inline(__always)
private func validatedLockedSurfaceBaseAddress(
    _ surface: IOSurfaceRef,
    requiredBytes: Int
) throws(MetalExpansionArgmaxError) -> UnsafeMutableRawPointer {
    let baseAddress = IOSurfaceGetBaseAddress(surface)
    guard requiredBytes <= IOSurfaceGetAllocSize(surface) else {
        throw .invalidArguments("IOSurface allocation \(IOSurfaceGetAllocSize(surface)) is smaller than required \(requiredBytes) bytes")
    }
    return baseAddress
}

public enum MetalExpansionArgmaxError: Error, Equatable {
    case metalUnavailable
    case commandQueueUnavailable
    case libraryBuildFailed(String)
    case pipelineBuildFailed(String)
    case bufferAllocationFailed
    case surfaceLockFailed(Int32)
    case bufferBindingFailed
    case commandBufferUnavailable
    case commandEncoderUnavailable
    case commandExecutionFailed(String)
    case invalidArguments(String)
}

/// MPS-based expansion [bneck→vocab] via optimized matmul + fused GPU argmax.
///
/// Two-pass design:
///  1. MPSMatrixMultiplication: proj^T [spatial, bneck] × w^T [bneck, vocab] → logits [spatial, vocab] fp16
///  2. Custom Metal kernel: row-wise argmax reduction → uint16 token IDs
///
/// Input: IOSurface [1, bneck, 1, spatial] fp16 channel-first (ANE native layout).
/// Output: uint16 token IDs, one per spatial lane.
public final class MPSExpansionArgmax {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let matmul: MPSMatrixMultiplication
    private let argmaxPipeline: MTLComputePipelineState
    private let projBuffer: MTLBuffer
    private let wBuffer: MTLBuffer
    private let logitsBuffer: MTLBuffer
    private let outputBuffer: MTLBuffer

    public let bottleneck: Int
    public let vocabSize: Int
    public let spatial: Int

    public init(
        wExpand: UnsafeBufferPointer<Float16>,
        bottleneck: Int,
        vocabSize: Int,
        spatial: Int
    ) throws(MetalExpansionArgmaxError) {
        guard bottleneck > 0, vocabSize > 0, spatial > 0 else {
            throw .invalidArguments("all dimensions must be > 0")
        }
        try requireUInt16TokenIDs(vocabSize: vocabSize)
        guard wExpand.count >= vocabSize * bottleneck else {
            throw .invalidArguments("wExpand too small")
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw .metalUnavailable
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw .commandQueueUnavailable
        }

        // MPS matmul: proj^T [spatial, bneck] × w^T [bneck, vocab] → [spatial, vocab]
        let mpsMatmul = MPSMatrixMultiplication(
            device: device,
            transposeLeft: true,
            transposeRight: true,
            resultRows: spatial,
            resultColumns: vocabSize,
            interiorColumns: bottleneck,
            alpha: 1.0,
            beta: 0.0)

        // Argmax kernel
        let argmaxLib: MTLLibrary
        do {
            argmaxLib = try device.makeLibrary(source: Self.argmaxShaderSource, options: nil)
        } catch {
            throw .libraryBuildFailed(String(describing: error))
        }
        guard let argmaxFn = argmaxLib.makeFunction(name: "row_argmax_fp16") else {
            throw .libraryBuildFailed("missing row_argmax_fp16")
        }
        let pso: MTLComputePipelineState
        do {
            pso = try device.makeComputePipelineState(function: argmaxFn)
        } catch {
            throw .pipelineBuildFailed(String(describing: error))
        }

        // Buffers
        let projSize = bottleneck * spatial * MemoryLayout<Float16>.stride
        guard let projBuf = device.makeBuffer(length: projSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        let wSize = vocabSize * bottleneck * MemoryLayout<Float16>.stride
        guard let wBuf = device.makeBuffer(bytes: wExpand.baseAddress!, length: wSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        let logitsSize = spatial * vocabSize * MemoryLayout<Float16>.stride
        guard let logBuf = device.makeBuffer(length: logitsSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        let outSize = spatial * MemoryLayout<UInt16>.stride
        guard let outBuf = device.makeBuffer(length: outSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }

        self.device = device
        self.commandQueue = commandQueue
        self.matmul = mpsMatmul
        self.argmaxPipeline = pso
        self.projBuffer = projBuf
        self.wBuffer = wBuf
        self.logitsBuffer = logBuf
        self.outputBuffer = outBuf
        self.bottleneck = bottleneck
        self.vocabSize = vocabSize
        self.spatial = spatial
    }

    /// Run MPS matmul + GPU argmax on projected ANE output surface.
    public func run(projectedSurface: IOSurfaceRef) throws(MetalExpansionArgmaxError) -> [TokenID] {
        let length = try requiredFP16SurfaceBytes(channels: bottleneck, spatial: spatial)

        // Copy IOSurface data into Metal buffer
        let status = IOSurfaceLock(projectedSurface, [.readOnly], nil)
        guard status == 0 else { throw .surfaceLockFailed(status) }
        let baseAddress: UnsafeMutableRawPointer
        do {
            baseAddress = try validatedLockedSurfaceBaseAddress(projectedSurface, requiredBytes: length)
        } catch {
            IOSurfaceUnlock(projectedSurface, [.readOnly], nil)
            throw error
        }
        memcpy(projBuffer.contents(), baseAddress, length)
        IOSurfaceUnlock(projectedSurface, [.readOnly], nil)

        // MPS matrix descriptors
        let projDesc = MPSMatrixDescriptor(
            rows: bottleneck, columns: spatial,
            rowBytes: spatial * MemoryLayout<Float16>.stride,
            dataType: .float16)
        let wDesc = MPSMatrixDescriptor(
            rows: vocabSize, columns: bottleneck,
            rowBytes: bottleneck * MemoryLayout<Float16>.stride,
            dataType: .float16)
        let outDesc = MPSMatrixDescriptor(
            rows: spatial, columns: vocabSize,
            rowBytes: vocabSize * MemoryLayout<Float16>.stride,
            dataType: .float16)

        let projMat = MPSMatrix(buffer: projBuffer, descriptor: projDesc)
        let wMat = MPSMatrix(buffer: wBuffer, descriptor: wDesc)
        let logitsMat = MPSMatrix(buffer: logitsBuffer, descriptor: outDesc)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw .commandBufferUnavailable
        }

        // Pass 1: MPS matmul
        matmul.encode(commandBuffer: commandBuffer, leftMatrix: projMat, rightMatrix: wMat, resultMatrix: logitsMat)

        // Pass 2: row-wise argmax
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        encoder.setComputePipelineState(argmaxPipeline)
        encoder.setBuffer(logitsBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        var vocabU32 = UInt32(vocabSize)
        encoder.setBytes(&vocabU32, length: 4, index: 2)
        // One threadgroup per row, 256 threads per group
        let tpg = min(256, Int(argmaxPipeline.maxTotalThreadsPerThreadgroup))
        encoder.dispatchThreadgroups(
            MTLSize(width: spatial, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1))
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if commandBuffer.status != .completed {
            throw .commandExecutionFailed(
                commandBuffer.error?.localizedDescription ?? "status=\(commandBuffer.status.rawValue)")
        }

        let ptr = outputBuffer.contents().assumingMemoryBound(to: UInt16.self)
        return UnsafeBufferPointer(start: ptr, count: spatial).map { TokenID($0) }
    }

    // MARK: - GPU Argmax Shader

    static let argmaxShaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    /// Row-wise argmax over [rows, vocab] fp16 matrix.
    /// One threadgroup per row, cooperative reduction within the group.
    kernel void row_argmax_fp16(
        const device half *logits [[buffer(0)]],
        device ushort *tokenIds [[buffer(1)]],
        constant uint &vocab [[buffer(2)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint tpg [[threads_per_threadgroup]]
    ) {
        const device half *rowPtr = logits + row * vocab;

        // Each thread finds local max over its portion
        float localMax = -INFINITY;
        uint localIdx = 0;

        for (uint v = tid; v < vocab; v += tpg) {
            float val = float(rowPtr[v]);
            if (val > localMax) {
                localMax = val;
                localIdx = v;
            }
        }

        // SIMD reduction within each simd_group
        for (ushort offset = 16; offset > 0; offset >>= 1) {
            float otherVal = simd_shuffle_down(localMax, offset);
            uint otherIdx = simd_shuffle_down(localIdx, offset);
            if (otherVal > localMax) {
                localMax = otherVal;
                localIdx = otherIdx;
            }
        }

        // Threadgroup reduction: lane 0 of each simd_group writes to shared mem
        threadgroup float sharedMax[8];  // up to 256 threads / 32 = 8 simd groups
        threadgroup uint sharedIdx[8];

        uint simdGroupId = tid / 32;
        uint laneInSimd = tid % 32;

        if (laneInSimd == 0) {
            sharedMax[simdGroupId] = localMax;
            sharedIdx[simdGroupId] = localIdx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Final reduction by thread 0
        if (tid == 0) {
            float bestVal = sharedMax[0];
            uint bestIdx = sharedIdx[0];
            uint numSimdGroups = (tpg + 31) / 32;
            for (uint s = 1; s < numSimdGroups; s++) {
                if (sharedMax[s] > bestVal) {
                    bestVal = sharedMax[s];
                    bestIdx = sharedIdx[s];
                }
            }
            tokenIds[row] = ushort(bestIdx);
        }
    }
    """
}

/// GPU-side full head: RMSNorm + projection + expansion + argmax.
/// Takes raw trunk output [1, dim, 1, spatial] from IOSurface, produces token IDs.
/// Eliminates the ANE head dispatch by doing RMSNorm + projection on GPU.
///
/// Pipeline:
///  1. memcpy IOSurface [dim, spatial] → Metal buffer
///  2. GPU RMSNorm: in-place normalize + scale
///  3. MPS matmul: norm^T [spatial, dim] × projW^T [dim, bneck] → proj [spatial, bneck]
///  4. MPS matmul: proj [spatial, bneck] × expandW^T [bneck, vocab] → logits [spatial, vocab]
///  5. GPU argmax: logits → uint16 token IDs
public final class GPUFullHeadArgmax {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let rmsNormPipeline: MTLComputePipelineState
    private let argmaxPipeline: MTLComputePipelineState
    private let projMatmul: MPSMatrixMultiplication
    private let expandMatmul: MPSMatrixMultiplication

    private let trunkBuffer: MTLBuffer      // [dim, spatial] fp16
    private let gammaBuffer: MTLBuffer      // [dim] fp16
    private let projWBuffer: MTLBuffer      // [dim, bneck] fp16
    private let projOutBuffer: MTLBuffer    // [spatial, bneck] fp16
    private let expandWBuffer: MTLBuffer    // [vocab, bneck] fp16
    private let logitsBuffer: MTLBuffer     // [spatial, vocab] fp16
    private let outputBuffer: MTLBuffer     // [spatial] uint16

    public let dim: Int
    public let bottleneck: Int
    public let vocabSize: Int
    public let spatial: Int

    public init(
        rmsGamma: UnsafeBufferPointer<Float16>,
        wProject: UnsafeBufferPointer<Float16>,
        wExpand: UnsafeBufferPointer<Float16>,
        dim: Int,
        bottleneck: Int,
        vocabSize: Int,
        spatial: Int
    ) throws(MetalExpansionArgmaxError) {
        guard dim > 0, bottleneck > 0, vocabSize > 0, spatial > 0 else {
            throw .invalidArguments("all dimensions must be > 0")
        }
        try requireUInt16TokenIDs(vocabSize: vocabSize)
        guard rmsGamma.count >= dim else { throw .invalidArguments("rmsGamma too small") }
        guard wProject.count >= dim * bottleneck else { throw .invalidArguments("wProject too small") }
        guard wExpand.count >= vocabSize * bottleneck else { throw .invalidArguments("wExpand too small") }

        guard let device = MTLCreateSystemDefaultDevice() else { throw .metalUnavailable }
        guard let queue = device.makeCommandQueue() else { throw .commandQueueUnavailable }

        // Compile RMSNorm kernel
        let rmsLib: MTLLibrary
        do { rmsLib = try device.makeLibrary(source: Self.rmsNormSource(dim: dim), options: nil) }
        catch { throw .libraryBuildFailed("RMSNorm: \(error)") }
        guard let rmsFn = rmsLib.makeFunction(name: "rms_norm_channelfirst") else {
            throw .libraryBuildFailed("missing rms_norm_channelfirst")
        }
        let rmsPSO: MTLComputePipelineState
        do { rmsPSO = try device.makeComputePipelineState(function: rmsFn) }
        catch { throw .pipelineBuildFailed("RMSNorm: \(error)") }

        // Compile argmax kernel
        let argLib: MTLLibrary
        do { argLib = try device.makeLibrary(source: MPSExpansionArgmax.argmaxShaderSource, options: nil) }
        catch { throw .libraryBuildFailed("argmax: \(error)") }
        guard let argFn = argLib.makeFunction(name: "row_argmax_fp16") else {
            throw .libraryBuildFailed("missing row_argmax_fp16")
        }
        let argPSO: MTLComputePipelineState
        do { argPSO = try device.makeComputePipelineState(function: argFn) }
        catch { throw .pipelineBuildFailed("argmax: \(error)") }

        // MPS matmuls
        let projMM = MPSMatrixMultiplication(
            device: device, transposeLeft: true, transposeRight: true,
            resultRows: spatial, resultColumns: bottleneck, interiorColumns: dim,
            alpha: 1.0, beta: 0.0)
        let expandMM = MPSMatrixMultiplication(
            device: device, transposeLeft: false, transposeRight: true,
            resultRows: spatial, resultColumns: vocabSize, interiorColumns: bottleneck,
            alpha: 1.0, beta: 0.0)

        // Allocate buffers
        let trunkSize = dim * spatial * MemoryLayout<Float16>.stride
        guard let trunkBuf = device.makeBuffer(length: trunkSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        guard let gammaBuf = device.makeBuffer(bytes: rmsGamma.baseAddress!, length: dim * 2, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        let projWSize = dim * bottleneck * MemoryLayout<Float16>.stride
        guard let projWBuf = device.makeBuffer(bytes: wProject.baseAddress!, length: projWSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        let projOutSize = spatial * bottleneck * MemoryLayout<Float16>.stride
        guard let projOutBuf = device.makeBuffer(length: projOutSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        let expandWSize = vocabSize * bottleneck * MemoryLayout<Float16>.stride
        guard let expandWBuf = device.makeBuffer(bytes: wExpand.baseAddress!, length: expandWSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        let logitsSize = spatial * vocabSize * MemoryLayout<Float16>.stride
        guard let logitsBuf = device.makeBuffer(length: logitsSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        guard let outBuf = device.makeBuffer(length: spatial * 2, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }

        self.device = device
        self.commandQueue = queue
        self.rmsNormPipeline = rmsPSO
        self.argmaxPipeline = argPSO
        self.projMatmul = projMM
        self.expandMatmul = expandMM
        self.trunkBuffer = trunkBuf
        self.gammaBuffer = gammaBuf
        self.projWBuffer = projWBuf
        self.projOutBuffer = projOutBuf
        self.expandWBuffer = expandWBuf
        self.logitsBuffer = logitsBuf
        self.outputBuffer = outBuf
        self.dim = dim
        self.bottleneck = bottleneck
        self.vocabSize = vocabSize
        self.spatial = spatial
    }

    /// Run full GPU head pipeline on trunk output surface.
    public func run(trunkSurface: IOSurfaceRef) throws(MetalExpansionArgmaxError) -> [TokenID] {
        let trunkBytes = try requiredFP16SurfaceBytes(channels: dim, spatial: spatial)

        // 1. Copy trunk output to GPU buffer
        let lockStatus = IOSurfaceLock(trunkSurface, [.readOnly], nil)
        guard lockStatus == 0 else { throw .surfaceLockFailed(lockStatus) }
        let baseAddress: UnsafeMutableRawPointer
        do {
            baseAddress = try validatedLockedSurfaceBaseAddress(trunkSurface, requiredBytes: trunkBytes)
        } catch {
            IOSurfaceUnlock(trunkSurface, [.readOnly], nil)
            throw error
        }
        memcpy(trunkBuffer.contents(), baseAddress, trunkBytes)
        IOSurfaceUnlock(trunkSurface, [.readOnly], nil)

        guard let cb = commandQueue.makeCommandBuffer() else { throw .commandBufferUnavailable }

        // 2. GPU RMSNorm in-place
        guard let rmsEnc = cb.makeComputeCommandEncoder() else { throw .commandEncoderUnavailable }
        rmsEnc.setComputePipelineState(rmsNormPipeline)
        rmsEnc.setBuffer(trunkBuffer, offset: 0, index: 0)
        rmsEnc.setBuffer(gammaBuffer, offset: 0, index: 1)
        var dims = (UInt32(dim), UInt32(spatial))
        withUnsafeBytes(of: &dims) { rmsEnc.setBytes($0.baseAddress!, length: $0.count, index: 2) }
        let rmsTpg = min(256, Int(rmsNormPipeline.maxTotalThreadsPerThreadgroup))
        rmsEnc.dispatchThreadgroups(
            MTLSize(width: (spatial + rmsTpg - 1) / rmsTpg, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: rmsTpg, height: 1, depth: 1))
        rmsEnc.endEncoding()

        // 3. MPS projection: norm^T [S, dim] × projW^T [dim, bneck] → [S, bneck]
        let trunkDesc = MPSMatrixDescriptor(rows: dim, columns: spatial,
            rowBytes: spatial * 2, dataType: .float16)
        let projWDesc = MPSMatrixDescriptor(rows: bottleneck, columns: dim,
            rowBytes: dim * 2, dataType: .float16)
        let projOutDesc = MPSMatrixDescriptor(rows: spatial, columns: bottleneck,
            rowBytes: bottleneck * 2, dataType: .float16)

        let trunkMat = MPSMatrix(buffer: trunkBuffer, descriptor: trunkDesc)
        let projWMat = MPSMatrix(buffer: projWBuffer, descriptor: projWDesc)
        let projOutMat = MPSMatrix(buffer: projOutBuffer, descriptor: projOutDesc)
        projMatmul.encode(commandBuffer: cb, leftMatrix: trunkMat, rightMatrix: projWMat, resultMatrix: projOutMat)

        // 4. MPS expansion: proj [S, bneck] × expandW^T [bneck, vocab] → logits [S, vocab]
        let projDesc2 = MPSMatrixDescriptor(rows: spatial, columns: bottleneck,
            rowBytes: bottleneck * 2, dataType: .float16)
        let expandWDesc = MPSMatrixDescriptor(rows: vocabSize, columns: bottleneck,
            rowBytes: bottleneck * 2, dataType: .float16)
        let logitsDesc = MPSMatrixDescriptor(rows: spatial, columns: vocabSize,
            rowBytes: vocabSize * 2, dataType: .float16)

        let projMat2 = MPSMatrix(buffer: projOutBuffer, descriptor: projDesc2)
        let expandWMat = MPSMatrix(buffer: expandWBuffer, descriptor: expandWDesc)
        let logitsMat = MPSMatrix(buffer: logitsBuffer, descriptor: logitsDesc)
        expandMatmul.encode(commandBuffer: cb, leftMatrix: projMat2, rightMatrix: expandWMat, resultMatrix: logitsMat)

        // 5. GPU argmax
        guard let argEnc = cb.makeComputeCommandEncoder() else { throw .commandEncoderUnavailable }
        argEnc.setComputePipelineState(argmaxPipeline)
        argEnc.setBuffer(logitsBuffer, offset: 0, index: 0)
        argEnc.setBuffer(outputBuffer, offset: 0, index: 1)
        var vocabU32 = UInt32(vocabSize)
        argEnc.setBytes(&vocabU32, length: 4, index: 2)
        let argTpg = min(256, Int(argmaxPipeline.maxTotalThreadsPerThreadgroup))
        argEnc.dispatchThreadgroups(
            MTLSize(width: spatial, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: argTpg, height: 1, depth: 1))
        argEnc.endEncoding()

        cb.commit()
        cb.waitUntilCompleted()
        if cb.status != .completed {
            throw .commandExecutionFailed(cb.error?.localizedDescription ?? "status=\(cb.status.rawValue)")
        }

        let ptr = outputBuffer.contents().assumingMemoryBound(to: UInt16.self)
        return UnsafeBufferPointer(start: ptr, count: spatial).map { TokenID($0) }
    }

    // MARK: - RMSNorm Shader

    internal static func publicRmsNormSource(dim: Int) -> String { rmsNormSource(dim: dim) }
    private static func rmsNormSource(dim: Int) -> String {
        """
        #include <metal_stdlib>
        using namespace metal;

        #define DIM \(dim)

        /// RMSNorm on channel-first [dim, spatial] data. One thread per spatial position.
        /// Adjacent threads access adjacent spatial positions → coalesced reads.
        kernel void rms_norm_channelfirst(
            device half *x [[buffer(0)]],
            constant half *gamma [[buffer(1)]],
            constant uint2 &dims [[buffer(2)]],
            uint s [[thread_position_in_grid]]
        ) {
            uint spatial = dims.y;
            if (s >= spatial) return;

            // Compute sum of squares across channels
            float sum_sq = 0.0f;
            for (uint c = 0; c < DIM; c++) {
                float v = float(x[c * spatial + s]);
                sum_sq += v * v;
            }
            float rms_inv = rsqrt(sum_sq / float(DIM) + 1e-5f);

            // Normalize and scale
            for (uint c = 0; c < DIM; c++) {
                float v = float(x[c * spatial + s]);
                x[c * spatial + s] = half(v * rms_inv * float(gamma[c]));
            }
        }
        """
    }
}

/// GPU full pipeline head: RMSNorm → proj → expand → argmax → embed write.
/// All stages execute in a single Metal command buffer. The embed write places
/// token embeddings directly into an ANE input IOSurface in channel-first layout,
/// eliminating the CPU embed write roundtrip.
///
/// Input: trunk IOSurface [1, dim, 1, spatial] (ANE output), embed IOSurface [1, dim, 1, spatial] (ANE input)
/// Output: token IDs (uint16, still returned for verification/generation tracking)
public final class GPUPipelinedHead {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let rmsNormPipeline: MTLComputePipelineState
    private let argmaxPipeline: MTLComputePipelineState
    private let embedPipeline: MTLComputePipelineState
    private let projMatmul: MPSMatrixMultiplication
    private let expandMatmul: MPSMatrixMultiplication

    private let trunkBuffer: MTLBuffer      // [dim, spatial] fp16
    private let gammaBuffer: MTLBuffer      // [dim] fp16
    private let projWBuffer: MTLBuffer      // [dim, bneck] fp16
    private let projOutBuffer: MTLBuffer    // [spatial, bneck] fp16
    private let expandWBuffer: MTLBuffer    // [vocab, bneck] fp16
    private let logitsBuffer: MTLBuffer     // [spatial, vocab] fp16
    private let outputBuffer: MTLBuffer     // [spatial] uint16
    private let embeddingBuffer: MTLBuffer  // [vocab, dim] fp16
    private let embedOutBuffer: MTLBuffer   // [dim, spatial] fp16 staging for embed write

    public let dim: Int
    public let bottleneck: Int
    public let vocabSize: Int
    public let spatial: Int

    public init(
        rmsGamma: UnsafeBufferPointer<Float16>,
        wProject: UnsafeBufferPointer<Float16>,
        wExpand: UnsafeBufferPointer<Float16>,
        embeddingFP16: UnsafeBufferPointer<Float16>,
        dim: Int,
        bottleneck: Int,
        vocabSize: Int,
        spatial: Int
    ) throws(MetalExpansionArgmaxError) {
        guard dim > 0, bottleneck > 0, vocabSize > 0, spatial > 0 else {
            throw .invalidArguments("all dimensions must be > 0")
        }
        try requireUInt16TokenIDs(vocabSize: vocabSize)
        guard rmsGamma.count >= dim else { throw .invalidArguments("rmsGamma too small") }
        guard wProject.count >= dim * bottleneck else { throw .invalidArguments("wProject too small") }
        guard wExpand.count >= vocabSize * bottleneck else { throw .invalidArguments("wExpand too small") }
        guard embeddingFP16.count >= vocabSize * dim else { throw .invalidArguments("embeddingFP16 too small") }

        guard let device = MTLCreateSystemDefaultDevice() else { throw .metalUnavailable }
        guard let queue = device.makeCommandQueue() else { throw .commandQueueUnavailable }

        // Compile RMSNorm kernel
        let rmsLib: MTLLibrary
        do { rmsLib = try device.makeLibrary(source: GPUFullHeadArgmax.publicRmsNormSource(dim: dim), options: nil) }
        catch { throw .libraryBuildFailed("RMSNorm: \(error)") }
        guard let rmsFn = rmsLib.makeFunction(name: "rms_norm_channelfirst") else {
            throw .libraryBuildFailed("missing rms_norm_channelfirst")
        }
        let rmsPSO: MTLComputePipelineState
        do { rmsPSO = try device.makeComputePipelineState(function: rmsFn) }
        catch { throw .pipelineBuildFailed("RMSNorm: \(error)") }

        // Compile argmax kernel
        let argLib: MTLLibrary
        do { argLib = try device.makeLibrary(source: MPSExpansionArgmax.argmaxShaderSource, options: nil) }
        catch { throw .libraryBuildFailed("argmax: \(error)") }
        guard let argFn = argLib.makeFunction(name: "row_argmax_fp16") else {
            throw .libraryBuildFailed("missing row_argmax_fp16")
        }
        let argPSO: MTLComputePipelineState
        do { argPSO = try device.makeComputePipelineState(function: argFn) }
        catch { throw .pipelineBuildFailed("argmax: \(error)") }

        // Compile embed write kernel
        let embedLib: MTLLibrary
        do { embedLib = try device.makeLibrary(source: Self.embedSource(dim: dim), options: nil) }
        catch { throw .libraryBuildFailed("embed: \(error)") }
        guard let embedFn = embedLib.makeFunction(name: "embed_write_channelfirst") else {
            throw .libraryBuildFailed("missing embed_write_channelfirst")
        }
        let embedPSO: MTLComputePipelineState
        do { embedPSO = try device.makeComputePipelineState(function: embedFn) }
        catch { throw .pipelineBuildFailed("embed: \(error)") }

        // MPS matmuls
        let projMM = MPSMatrixMultiplication(
            device: device, transposeLeft: true, transposeRight: true,
            resultRows: spatial, resultColumns: bottleneck, interiorColumns: dim,
            alpha: 1.0, beta: 0.0)
        let expandMM = MPSMatrixMultiplication(
            device: device, transposeLeft: false, transposeRight: true,
            resultRows: spatial, resultColumns: vocabSize, interiorColumns: bottleneck,
            alpha: 1.0, beta: 0.0)

        // Allocate buffers
        let trunkSize = dim * spatial * MemoryLayout<Float16>.stride
        guard let trunkBuf = device.makeBuffer(length: trunkSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        guard let gammaBuf = device.makeBuffer(bytes: rmsGamma.baseAddress!, length: dim * 2, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        let projWSize = dim * bottleneck * MemoryLayout<Float16>.stride
        guard let projWBuf = device.makeBuffer(bytes: wProject.baseAddress!, length: projWSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        let projOutSize = spatial * bottleneck * MemoryLayout<Float16>.stride
        guard let projOutBuf = device.makeBuffer(length: projOutSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        let expandWSize = vocabSize * bottleneck * MemoryLayout<Float16>.stride
        guard let expandWBuf = device.makeBuffer(bytes: wExpand.baseAddress!, length: expandWSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        let logitsSize = spatial * vocabSize * MemoryLayout<Float16>.stride
        guard let logitsBuf = device.makeBuffer(length: logitsSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        guard let outBuf = device.makeBuffer(length: spatial * 2, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        let embSize = vocabSize * dim * MemoryLayout<Float16>.stride
        guard let embBuf = device.makeBuffer(bytes: embeddingFP16.baseAddress!, length: embSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        guard let embedOutBuf = device.makeBuffer(length: trunkSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }

        self.device = device
        self.commandQueue = queue
        self.rmsNormPipeline = rmsPSO
        self.argmaxPipeline = argPSO
        self.embedPipeline = embedPSO
        self.projMatmul = projMM
        self.expandMatmul = expandMM
        self.trunkBuffer = trunkBuf
        self.gammaBuffer = gammaBuf
        self.projWBuffer = projWBuf
        self.projOutBuffer = projOutBuf
        self.expandWBuffer = expandWBuf
        self.logitsBuffer = logitsBuf
        self.outputBuffer = outBuf
        self.embeddingBuffer = embBuf
        self.embedOutBuffer = embedOutBuf
        self.dim = dim
        self.bottleneck = bottleneck
        self.vocabSize = vocabSize
        self.spatial = spatial
    }

    /// Run full pipeline: RMSNorm → proj → expand → argmax → embed write.
    /// Writes embedded tokens directly to `embedSurface` in channel-first layout.
    /// Returns token IDs as a copied array.
    public func runAndEmbed(
        trunkSurface: IOSurfaceRef,
        embedSurface: IOSurfaceRef
    ) throws(MetalExpansionArgmaxError) -> [TokenID] {
        let trunkBytes = try requiredFP16SurfaceBytes(channels: dim, spatial: spatial)

        // 1. Copy trunk output to GPU buffer (lock/unlock immediately)
        let lockStatus = IOSurfaceLock(trunkSurface, [.readOnly], nil)
        guard lockStatus == 0 else { throw .surfaceLockFailed(lockStatus) }
        let trunkBaseAddress: UnsafeMutableRawPointer
        do {
            trunkBaseAddress = try validatedLockedSurfaceBaseAddress(trunkSurface, requiredBytes: trunkBytes)
        } catch {
            IOSurfaceUnlock(trunkSurface, [.readOnly], nil)
            throw error
        }
        memcpy(trunkBuffer.contents(), trunkBaseAddress, trunkBytes)
        IOSurfaceUnlock(trunkSurface, [.readOnly], nil)

        guard let cb = commandQueue.makeCommandBuffer() else {
            throw .commandBufferUnavailable
        }

        // 2. GPU RMSNorm in-place
        guard let rmsEnc = cb.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        rmsEnc.setComputePipelineState(rmsNormPipeline)
        rmsEnc.setBuffer(trunkBuffer, offset: 0, index: 0)
        rmsEnc.setBuffer(gammaBuffer, offset: 0, index: 1)
        var dims = (UInt32(dim), UInt32(spatial))
        withUnsafeBytes(of: &dims) { rmsEnc.setBytes($0.baseAddress!, length: $0.count, index: 2) }
        let rmsTpg = min(256, Int(rmsNormPipeline.maxTotalThreadsPerThreadgroup))
        rmsEnc.dispatchThreadgroups(
            MTLSize(width: (spatial + rmsTpg - 1) / rmsTpg, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: rmsTpg, height: 1, depth: 1))
        rmsEnc.endEncoding()

        // 3. MPS projection
        let trunkDesc = MPSMatrixDescriptor(rows: dim, columns: spatial, rowBytes: spatial * 2, dataType: .float16)
        let projWDesc = MPSMatrixDescriptor(rows: bottleneck, columns: dim, rowBytes: dim * 2, dataType: .float16)
        let projOutDesc = MPSMatrixDescriptor(rows: spatial, columns: bottleneck, rowBytes: bottleneck * 2, dataType: .float16)
        let trunkMat = MPSMatrix(buffer: trunkBuffer, descriptor: trunkDesc)
        let projWMat = MPSMatrix(buffer: projWBuffer, descriptor: projWDesc)
        let projOutMat = MPSMatrix(buffer: projOutBuffer, descriptor: projOutDesc)
        projMatmul.encode(commandBuffer: cb, leftMatrix: trunkMat, rightMatrix: projWMat, resultMatrix: projOutMat)

        // 4. MPS expansion
        let projDesc2 = MPSMatrixDescriptor(rows: spatial, columns: bottleneck, rowBytes: bottleneck * 2, dataType: .float16)
        let expandWDesc = MPSMatrixDescriptor(rows: vocabSize, columns: bottleneck, rowBytes: bottleneck * 2, dataType: .float16)
        let logitsDesc = MPSMatrixDescriptor(rows: spatial, columns: vocabSize, rowBytes: vocabSize * 2, dataType: .float16)
        let projMat2 = MPSMatrix(buffer: projOutBuffer, descriptor: projDesc2)
        let expandWMat = MPSMatrix(buffer: expandWBuffer, descriptor: expandWDesc)
        let logitsMat = MPSMatrix(buffer: logitsBuffer, descriptor: logitsDesc)
        expandMatmul.encode(commandBuffer: cb, leftMatrix: projMat2, rightMatrix: expandWMat, resultMatrix: logitsMat)

        // 5. GPU argmax
        guard let argEnc = cb.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        argEnc.setComputePipelineState(argmaxPipeline)
        argEnc.setBuffer(logitsBuffer, offset: 0, index: 0)
        argEnc.setBuffer(outputBuffer, offset: 0, index: 1)
        var vocabU32 = UInt32(vocabSize)
        argEnc.setBytes(&vocabU32, length: 4, index: 2)
        let argTpg = min(256, Int(argmaxPipeline.maxTotalThreadsPerThreadgroup))
        argEnc.dispatchThreadgroups(
            MTLSize(width: spatial, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: argTpg, height: 1, depth: 1))
        argEnc.endEncoding()

        // 6. GPU embed write to staging buffer (not directly to IOSurface)
        guard let embedEnc = cb.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        embedEnc.setComputePipelineState(embedPipeline)
        embedEnc.setBuffer(outputBuffer, offset: 0, index: 0)
        embedEnc.setBuffer(embeddingBuffer, offset: 0, index: 1)
        embedEnc.setBuffer(embedOutBuffer, offset: 0, index: 2)
        var sp = UInt32(spatial)
        embedEnc.setBytes(&sp, length: 4, index: 3)
        let embedTpg = min(256, Int(embedPipeline.maxTotalThreadsPerThreadgroup))
        embedEnc.dispatchThreadgroups(
            MTLSize(width: (spatial + embedTpg - 1) / embedTpg, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: embedTpg, height: 1, depth: 1))
        embedEnc.endEncoding()

        cb.commit()
        cb.waitUntilCompleted()

        if cb.status != .completed {
            throw .commandExecutionFailed(cb.error?.localizedDescription ?? "status=\(cb.status.rawValue)")
        }

        // 7. Copy embed results back to IOSurface (lock only after GPU completes)
        let embedWriteLock = IOSurfaceLock(embedSurface, [], nil)
        guard embedWriteLock == 0 else { throw .surfaceLockFailed(embedWriteLock) }
        memcpy(IOSurfaceGetBaseAddress(embedSurface), embedOutBuffer.contents(), trunkBytes)
        IOSurfaceUnlock(embedSurface, [], nil)

        let ptr = outputBuffer.contents().assumingMemoryBound(to: UInt16.self)
        return UnsafeBufferPointer(start: ptr, count: spatial).map { TokenID($0) }
    }

    /// Pre-bind IOSurfaces for zero-copy access. Returns opaque binding tokens.
    /// Call once at pipeline setup; use `runPreBound` in the hot loop.
    public func preBind(
        trunkSurface: IOSurfaceRef,
        embedSurface: IOSurfaceRef
    ) throws(MetalExpansionArgmaxError) -> PreBoundSurfaces {
        let trunkBytes = dim * spatial * MemoryLayout<Float16>.stride

        IOSurfaceLock(trunkSurface, [.readOnly], nil)
        guard let trunkBuf = device.makeBuffer(
            bytesNoCopy: IOSurfaceGetBaseAddress(trunkSurface),
            length: trunkBytes, options: .storageModeShared, deallocator: nil
        ) else {
            IOSurfaceUnlock(trunkSurface, [.readOnly], nil)
            throw .bufferBindingFailed
        }
        IOSurfaceUnlock(trunkSurface, [.readOnly], nil)

        IOSurfaceLock(embedSurface, [], nil)
        guard let embedBuf = device.makeBuffer(
            bytesNoCopy: IOSurfaceGetBaseAddress(embedSurface),
            length: trunkBytes, options: .storageModeShared, deallocator: nil
        ) else {
            IOSurfaceUnlock(embedSurface, [], nil)
            throw .bufferBindingFailed
        }
        IOSurfaceUnlock(embedSurface, [], nil)

        return PreBoundSurfaces(trunkBuffer: trunkBuf, embedBuffer: embedBuf)
    }

    /// Opaque binding for pre-bound IOSurfaces.
    public struct PreBoundSurfaces {
        internal let trunkBuffer: MTLBuffer
        internal let embedBuffer: MTLBuffer
    }

    /// Zero-copy hot path: reads trunk directly from pre-bound IOSurface buffer,
    /// writes embed directly to pre-bound IOSurface buffer. No CPU memcpy or
    /// IOSurface lock/unlock in the critical path.
    public func runPreBound(
        _ binding: PreBoundSurfaces
    ) throws(MetalExpansionArgmaxError) -> [TokenID] {
        guard let cb = commandQueue.makeCommandBuffer() else { throw .commandBufferUnavailable }

        // 1. Blit trunk surface → trunkBuffer (GPU copy, avoids CPU memcpy)
        guard let blit = cb.makeBlitCommandEncoder() else { throw .commandEncoderUnavailable }
        blit.copy(from: binding.trunkBuffer, sourceOffset: 0,
                  to: trunkBuffer, destinationOffset: 0,
                  size: dim * spatial * MemoryLayout<Float16>.stride)
        blit.endEncoding()

        // 2. GPU RMSNorm in-place on trunkBuffer
        guard let rmsEnc = cb.makeComputeCommandEncoder() else { throw .commandEncoderUnavailable }
        rmsEnc.setComputePipelineState(rmsNormPipeline)
        rmsEnc.setBuffer(trunkBuffer, offset: 0, index: 0)
        rmsEnc.setBuffer(gammaBuffer, offset: 0, index: 1)
        var dims = (UInt32(dim), UInt32(spatial))
        withUnsafeBytes(of: &dims) { rmsEnc.setBytes($0.baseAddress!, length: $0.count, index: 2) }
        let rmsTpg = min(256, Int(rmsNormPipeline.maxTotalThreadsPerThreadgroup))
        rmsEnc.dispatchThreadgroups(
            MTLSize(width: (spatial + rmsTpg - 1) / rmsTpg, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: rmsTpg, height: 1, depth: 1))
        rmsEnc.endEncoding()

        // 3. MPS projection
        let trunkDesc = MPSMatrixDescriptor(rows: dim, columns: spatial, rowBytes: spatial * 2, dataType: .float16)
        let projWDesc = MPSMatrixDescriptor(rows: bottleneck, columns: dim, rowBytes: dim * 2, dataType: .float16)
        let projOutDesc = MPSMatrixDescriptor(rows: spatial, columns: bottleneck, rowBytes: bottleneck * 2, dataType: .float16)
        let trunkMat = MPSMatrix(buffer: trunkBuffer, descriptor: trunkDesc)
        let projWMat = MPSMatrix(buffer: projWBuffer, descriptor: projWDesc)
        let projOutMat = MPSMatrix(buffer: projOutBuffer, descriptor: projOutDesc)
        projMatmul.encode(commandBuffer: cb, leftMatrix: trunkMat, rightMatrix: projWMat, resultMatrix: projOutMat)

        // 4. MPS expansion
        let projDesc2 = MPSMatrixDescriptor(rows: spatial, columns: bottleneck, rowBytes: bottleneck * 2, dataType: .float16)
        let expandWDesc = MPSMatrixDescriptor(rows: vocabSize, columns: bottleneck, rowBytes: bottleneck * 2, dataType: .float16)
        let logitsDesc = MPSMatrixDescriptor(rows: spatial, columns: vocabSize, rowBytes: vocabSize * 2, dataType: .float16)
        let projMat2 = MPSMatrix(buffer: projOutBuffer, descriptor: projDesc2)
        let expandWMat = MPSMatrix(buffer: expandWBuffer, descriptor: expandWDesc)
        let logitsMat = MPSMatrix(buffer: logitsBuffer, descriptor: logitsDesc)
        expandMatmul.encode(commandBuffer: cb, leftMatrix: projMat2, rightMatrix: expandWMat, resultMatrix: logitsMat)

        // 5. GPU argmax
        guard let argEnc = cb.makeComputeCommandEncoder() else { throw .commandEncoderUnavailable }
        argEnc.setComputePipelineState(argmaxPipeline)
        argEnc.setBuffer(logitsBuffer, offset: 0, index: 0)
        argEnc.setBuffer(outputBuffer, offset: 0, index: 1)
        var vocabU32 = UInt32(vocabSize)
        argEnc.setBytes(&vocabU32, length: 4, index: 2)
        let argTpg = min(256, Int(argmaxPipeline.maxTotalThreadsPerThreadgroup))
        argEnc.dispatchThreadgroups(
            MTLSize(width: spatial, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: argTpg, height: 1, depth: 1))
        argEnc.endEncoding()

        // 6. GPU embed write directly to pre-bound ANE input surface
        guard let embedEnc = cb.makeComputeCommandEncoder() else { throw .commandEncoderUnavailable }
        embedEnc.setComputePipelineState(embedPipeline)
        embedEnc.setBuffer(outputBuffer, offset: 0, index: 0)
        embedEnc.setBuffer(embeddingBuffer, offset: 0, index: 1)
        embedEnc.setBuffer(binding.embedBuffer, offset: 0, index: 2)
        var sp = UInt32(spatial)
        embedEnc.setBytes(&sp, length: 4, index: 3)
        let embedTpg = min(256, Int(embedPipeline.maxTotalThreadsPerThreadgroup))
        embedEnc.dispatchThreadgroups(
            MTLSize(width: (spatial + embedTpg - 1) / embedTpg, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: embedTpg, height: 1, depth: 1))
        embedEnc.endEncoding()

        cb.commit()
        cb.waitUntilCompleted()

        if cb.status != .completed {
            throw .commandExecutionFailed(cb.error?.localizedDescription ?? "status=\(cb.status.rawValue)")
        }

        let ptr = outputBuffer.contents().assumingMemoryBound(to: UInt16.self)
        return UnsafeBufferPointer(start: ptr, count: spatial).map { TokenID($0) }
    }

    // MARK: - Embed Write Shader

    private static func embedSource(dim: Int) -> String {
        """
        #include <metal_stdlib>
        using namespace metal;

        #define DIM \(dim)

        /// Write token embeddings to ANE input surface in channel-first [dim, spatial] layout.
        /// One thread per spatial lane. Reads token ID, gathers embedding, scatters to output.
        kernel void embed_write_channelfirst(
            const device ushort *tokenIds [[buffer(0)]],
            const device half *embTable [[buffer(1)]],
            device half *output [[buffer(2)]],
            constant uint &spatial [[buffer(3)]],
            uint lane [[thread_position_in_grid]]
        ) {
            if (lane >= spatial) return;
            uint tokenId = uint(tokenIds[lane]);
            const device half *emb = embTable + tokenId * DIM;
            for (uint c = 0; c < DIM; c++) {
                output[c * spatial + lane] = emb[c];
            }
        }
        """
    }
}

/// Fused matmul+argmax: computes proj[spatial,bneck] × w[bneck,vocab] → argmax without
/// materializing the full [spatial,vocab] logits matrix. Eliminates ~1GB intermediate buffer.
///
/// One threadgroup per spatial row. Each thread handles a stripe of vocab entries,
/// computing dot products and tracking the local max. Cooperative SIMD+threadgroup
/// reduction finds the global argmax per row.
///
/// Input: IOSurface [1, bneck, 1, spatial] fp16 channel-first (ANE native layout).
/// Output: uint16 token IDs, one per spatial lane.
public final class FusedExpansionArgmax {

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let projBuffer: MTLBuffer
    private let wBuffer: MTLBuffer
    private let outputBuffer: MTLBuffer
    private let threadsPerGroup: Int

    public let bottleneck: Int
    public let vocabSize: Int
    public let spatial: Int

    public init(
        wExpand: UnsafeBufferPointer<Float16>,
        bottleneck: Int,
        vocabSize: Int,
        spatial: Int
    ) throws(MetalExpansionArgmaxError) {
        guard bottleneck > 0, vocabSize > 0, spatial > 0 else {
            throw .invalidArguments("all dimensions must be > 0")
        }
        try requireUInt16TokenIDs(vocabSize: vocabSize)
        guard wExpand.count >= vocabSize * bottleneck else {
            throw .invalidArguments("wExpand too small")
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw .metalUnavailable
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw .commandQueueUnavailable
        }

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: Self.shaderSource(bneck: bottleneck), options: nil)
        } catch {
            throw .libraryBuildFailed(String(describing: error))
        }
        guard let fn = library.makeFunction(name: "fused_expansion_argmax") else {
            throw .libraryBuildFailed("missing fused_expansion_argmax")
        }
        let pso: MTLComputePipelineState
        do {
            pso = try device.makeComputePipelineState(function: fn)
        } catch {
            throw .pipelineBuildFailed(String(describing: error))
        }

        let projSize = bottleneck * spatial * MemoryLayout<Float16>.stride
        guard let projBuf = device.makeBuffer(length: projSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        let wSize = vocabSize * bottleneck * MemoryLayout<Float16>.stride
        guard let wBuf = device.makeBuffer(bytes: wExpand.baseAddress!, length: wSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }
        let outSize = spatial * MemoryLayout<UInt16>.stride
        guard let outBuf = device.makeBuffer(length: outSize, options: .storageModeShared) else {
            throw .bufferAllocationFailed
        }

        self.device = device
        self.commandQueue = commandQueue
        self.pipeline = pso
        self.projBuffer = projBuf
        self.wBuffer = wBuf
        self.outputBuffer = outBuf
        self.bottleneck = bottleneck
        self.vocabSize = vocabSize
        self.spatial = spatial
        self.threadsPerGroup = min(1024, Int(pso.maxTotalThreadsPerThreadgroup))
    }

    /// Run fused matmul+argmax on projected ANE output surface.
    public func run(projectedSurface: IOSurfaceRef) throws(MetalExpansionArgmaxError) -> [TokenID] {
        let length = try requiredFP16SurfaceBytes(channels: bottleneck, spatial: spatial)

        let status = IOSurfaceLock(projectedSurface, [.readOnly], nil)
        guard status == 0 else { throw .surfaceLockFailed(status) }
        let baseAddress: UnsafeMutableRawPointer
        do {
            baseAddress = try validatedLockedSurfaceBaseAddress(projectedSurface, requiredBytes: length)
        } catch {
            IOSurfaceUnlock(projectedSurface, [.readOnly], nil)
            throw error
        }
        memcpy(projBuffer.contents(), baseAddress, length)
        IOSurfaceUnlock(projectedSurface, [.readOnly], nil)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw .commandBufferUnavailable
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(projBuffer, offset: 0, index: 0)
        encoder.setBuffer(wBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var dims = (UInt32(vocabSize), UInt32(spatial))
        withUnsafeBytes(of: &dims) { raw in
            encoder.setBytes(raw.baseAddress!, length: raw.count, index: 3)
        }
        encoder.dispatchThreadgroups(
            MTLSize(width: spatial, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if commandBuffer.status != .completed {
            throw .commandExecutionFailed(
                commandBuffer.error?.localizedDescription ?? "status=\(commandBuffer.status.rawValue)")
        }

        let ptr = outputBuffer.contents().assumingMemoryBound(to: UInt16.self)
        return UnsafeBufferPointer(start: ptr, count: spatial).map { TokenID($0) }
    }

    // MARK: - Fused Shader

    private static func shaderSource(bneck: Int) -> String {
        """
        #include <metal_stdlib>
        using namespace metal;

        #define BNECK \(bneck)

        /// Fused matmul+argmax: one threadgroup per spatial row.
        /// Loads projection vector into shared memory, each thread handles a stripe
        /// of vocab entries computing dot product + tracking max. Cooperative reduction
        /// finds the global argmax. Never materializes full [spatial,vocab] logits.
        ///
        /// proj layout: channel-first [1, bneck, 1, spatial] → proj[k * spatial + row]
        /// wExpand layout: row-major [vocab, bneck] → wExpand[v * bneck + k]
        kernel void fused_expansion_argmax(
            const device half *proj [[buffer(0)]],
            const device half *wExpand [[buffer(1)]],
            device ushort *tokenIds [[buffer(2)]],
            constant uint2 &dims [[buffer(3)]],
            uint row [[threadgroup_position_in_grid]],
            uint tid [[thread_index_in_threadgroup]],
            uint tpg [[threads_per_threadgroup]]
        ) {
            uint vocab = dims.x;
            uint spatial = dims.y;

            // Load projection vector into threadgroup shared memory (all threads cooperate)
            threadgroup float sharedProj[BNECK];
            if (tid < BNECK) {
                sharedProj[tid] = float(proj[tid * spatial + row]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Each thread handles a stripe of vocab entries
            float localMax = -INFINITY;
            uint localIdx = 0;

            for (uint v = tid; v < vocab; v += tpg) {
                float dot = 0.0f;
                const device half *wRow = wExpand + v * BNECK;
                for (uint k = 0; k < BNECK; k++) {
                    dot = fma(sharedProj[k], float(wRow[k]), dot);
                }
                if (dot > localMax) {
                    localMax = dot;
                    localIdx = v;
                }
            }

            // SIMD reduction within each simd_group
            for (ushort offset = 16; offset > 0; offset >>= 1) {
                float otherVal = simd_shuffle_down(localMax, offset);
                uint otherIdx = simd_shuffle_down(localIdx, offset);
                if (otherVal > localMax) {
                    localMax = otherVal;
                    localIdx = otherIdx;
                }
            }

            // Threadgroup reduction: lane 0 of each simd_group writes to shared mem
            threadgroup float sharedMax[32];  // up to 1024 threads / 32 = 32 simd groups
            threadgroup uint sharedArgmax[32];

            uint simdGroupId = tid / 32;
            uint laneInSimd = tid % 32;

            if (laneInSimd == 0) {
                sharedMax[simdGroupId] = localMax;
                sharedArgmax[simdGroupId] = localIdx;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Final reduction by thread 0
            if (tid == 0) {
                float bestVal = sharedMax[0];
                uint bestIdx = sharedArgmax[0];
                uint numSimdGroups = (tpg + 31) / 32;
                for (uint s = 1; s < numSimdGroups; s++) {
                    if (sharedMax[s] > bestVal) {
                        bestVal = sharedMax[s];
                        bestIdx = sharedArgmax[s];
                    }
                }
                tokenIds[row] = ushort(bestIdx);
            }
        }
        """
    }
}

/// Metal GPU expansion [bneck→vocab] + argmax on projected ANE output (single-pass, legacy).
///
/// Input: IOSurface [1, bneck, 1, spatial] fp16 channel-first (ANE native layout).
/// Output: uint16 token IDs, one per spatial lane.
///
/// Single-pass design: one thread per spatial lane. Adjacent SIMD threads read
/// adjacent proj elements (perfect coalescing). All threads read the same weight
/// row (broadcast). Weights stay in GPU L2 after first SIMD group.
public final class MetalExpansionArgmax {
    private struct Params {
        var vocab: UInt32
        var spatial: UInt32
    }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let wExpandBuffer: MTLBuffer
    private let outputBuffer: MTLBuffer
    /// Pre-allocated shared input buffer for the pre-bound path (avoids per-call IOSurface lock + makeBuffer).
    public let inputBuffer: MTLBuffer
    private let params: Params

    public let bottleneck: Int
    public let vocabSize: Int
    public let spatial: Int

    public init(
        wExpand: UnsafeBufferPointer<Float16>,
        bottleneck: Int,
        vocabSize: Int,
        spatial: Int,
        threadsPerGroup: Int = 256
    ) throws(MetalExpansionArgmaxError) {
        guard bottleneck > 0, vocabSize > 0, spatial > 0 else {
            throw .invalidArguments("all dimensions must be > 0")
        }
        try requireUInt16TokenIDs(vocabSize: vocabSize)
        guard bottleneck <= 64 else {
            throw .invalidArguments("bottleneck must be <= 64")
        }
        guard wExpand.count >= vocabSize * bottleneck else {
            throw .invalidArguments("wExpand too small")
        }

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw .metalUnavailable
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw .commandQueueUnavailable
        }

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: Self.shaderSource(bneck: bottleneck), options: nil)
        } catch {
            throw .libraryBuildFailed(String(describing: error))
        }

        guard let fn = library.makeFunction(name: "expansion_argmax") else {
            throw .libraryBuildFailed("missing expansion_argmax")
        }

        let pso: MTLComputePipelineState
        do {
            pso = try device.makeComputePipelineState(function: fn)
        } catch {
            throw .pipelineBuildFailed(String(describing: error))
        }

        let expandSize = vocabSize * bottleneck * MemoryLayout<Float16>.stride
        guard let expandBuf = device.makeBuffer(
            bytes: wExpand.baseAddress!, length: expandSize, options: .storageModeShared
        ) else { throw .bufferAllocationFailed }

        let outputSize = spatial * MemoryLayout<UInt16>.stride
        guard let outBuf = device.makeBuffer(
            length: outputSize, options: .storageModeShared
        ) else { throw .bufferAllocationFailed }

        let inputSize = bottleneck * spatial * MemoryLayout<Float16>.stride
        guard let inBuf = device.makeBuffer(
            length: inputSize, options: .storageModeShared
        ) else { throw .bufferAllocationFailed }

        self.device = device
        self.commandQueue = commandQueue
        self.pipeline = pso
        self.wExpandBuffer = expandBuf
        self.outputBuffer = outBuf
        self.inputBuffer = inBuf
        self.bottleneck = bottleneck
        self.vocabSize = vocabSize
        self.spatial = spatial
        self.params = Params(
            vocab: UInt32(vocabSize),
            spatial: UInt32(spatial)
        )
    }

    /// Run expansion+argmax on projected ANE output surface.
    /// Returns token IDs as a copied array.
    public func run(projectedSurface: IOSurfaceRef) throws(MetalExpansionArgmaxError) -> [TokenID] {
        let length = try requiredFP16SurfaceBytes(channels: bottleneck, spatial: spatial)

        let status = IOSurfaceLock(projectedSurface, [.readOnly], nil)
        guard status == 0 else { throw .surfaceLockFailed(status) }
        let baseAddress: UnsafeMutableRawPointer
        do {
            baseAddress = try validatedLockedSurfaceBaseAddress(projectedSurface, requiredBytes: length)
        } catch {
            IOSurfaceUnlock(projectedSurface, [.readOnly], nil)
            throw error
        }
        guard let inputBuffer = device.makeBuffer(
            bytesNoCopy: baseAddress,
            length: length,
            options: .storageModeShared,
            deallocator: nil
        ) else {
            IOSurfaceUnlock(projectedSurface, [.readOnly], nil)
            throw .bufferBindingFailed
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            IOSurfaceUnlock(projectedSurface, [.readOnly], nil)
            throw .commandBufferUnavailable
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            IOSurfaceUnlock(projectedSurface, [.readOnly], nil)
            throw .commandEncoderUnavailable
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(wExpandBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var p = params
        withUnsafeBytes(of: &p) { rawBytes in
            encoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 3)
        }
        let tpg = min(spatial, Int(pipeline.maxTotalThreadsPerThreadgroup))
        let groups = (spatial + tpg - 1) / tpg
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1)
        )
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        IOSurfaceUnlock(projectedSurface, [.readOnly], nil)

        if commandBuffer.status != .completed {
            throw .commandExecutionFailed(
                commandBuffer.error?.localizedDescription ?? "status=\(commandBuffer.status.rawValue)"
            )
        }

        let ptr = outputBuffer.contents().assumingMemoryBound(to: UInt16.self)
        return UnsafeBufferPointer(start: ptr, count: spatial).map { TokenID($0) }
    }

    /// Run expansion+argmax from pre-allocated inputBuffer (caller must fill it first).
    /// Avoids IOSurface lock and per-call buffer creation overhead.
    public func runPreBound() throws(MetalExpansionArgmaxError) -> [TokenID] {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw .commandBufferUnavailable
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(wExpandBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        var p = params
        withUnsafeBytes(of: &p) { rawBytes in
            encoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 3)
        }
        let tpg = min(spatial, Int(pipeline.maxTotalThreadsPerThreadgroup))
        let groups = (spatial + tpg - 1) / tpg
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tpg, height: 1, depth: 1)
        )
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if commandBuffer.status != .completed {
            throw .commandExecutionFailed(
                commandBuffer.error?.localizedDescription ?? "status=\(commandBuffer.status.rawValue)"
            )
        }

        let ptr = outputBuffer.contents().assumingMemoryBound(to: UInt16.self)
        return UnsafeBufferPointer(start: ptr, count: spatial).map { TokenID($0) }
    }

    // MARK: - Metal Shader

    private static func shaderSource(bneck: Int) -> String {
        """
        #include <metal_stdlib>
        using namespace metal;

        #define BNECK \(bneck)

        struct ExpParams {
            uint vocab;
            uint spatial;
        };

        /// Single-pass expansion [bneck → vocab] + fused argmax.
        ///
        /// BNECK is injected as a compile-time constant so the Metal compiler can:
        ///  1. Fully unroll the inner k-loop
        ///  2. Allocate projLocal in registers (not thread-private device memory)
        ///  3. Use fused multiply-add instructions
        ///
        /// One thread per spatial lane. Adjacent SIMD threads read adjacent
        /// proj elements (coalesced). All threads read the same weight row (broadcast).
        /// The 4MB weight tensor stays in GPU L2 after the first SIMD group.
        ///
        /// proj layout: channel-first [1, bneck, 1, spatial] → proj[k * spatial + lane]
        /// wExpand layout: row-major [vocab, bneck] → wExpand[v * bneck + k]
        kernel void expansion_argmax(
            const device half *proj [[buffer(0)]],
            const device half *wExpand [[buffer(1)]],
            device ushort *tokenIds [[buffer(2)]],
            constant ExpParams &params [[buffer(3)]],
            uint lane [[thread_position_in_grid]]
        ) {
            if (lane >= params.spatial) return;

            uint vocab = params.vocab;
            uint spatial = params.spatial;

            // Preload projection vector for this lane into registers.
            // BNECK is compile-time constant → compiler allocates registers, not device memory.
            float projLocal[BNECK];
            for (uint k = 0; k < BNECK; k++) {
                projLocal[k] = float(proj[k * spatial + lane]);
            }

            float bestVal = -INFINITY;
            uint bestIdx = 0;

            for (uint v = 0; v < vocab; v++) {
                float dot = 0.0f;
                const device half *wRow = wExpand + v * BNECK;
                for (uint k = 0; k < BNECK; k++) {
                    dot = fma(projLocal[k], float(wRow[k]), dot);
                }
                if (dot > bestVal) {
                    bestVal = dot;
                    bestIdx = v;
                }
            }

            tokenIds[lane] = ushort(bestIdx);
        }
        """
    }
}

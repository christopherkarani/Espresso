import Foundation
import Darwin
import Metal
import IOSurface
import ANETypes
import ANERuntime

public enum MetalAttentionError: Error, Equatable {
    case invalidShape(String)
    case invalidInputCount(String)
    case metalUnavailable
    case commandQueueUnavailable
    case libraryBuildFailed(String)
    case pipelineBuildFailed(String)
    case surfaceCreateFailed
    case surfaceLockFailed(Int32)
    case surfaceBaseAddressNil
    case bufferBindingFailed
    case temporaryBufferAllocationFailed
    case commandBufferUnavailable
    case commandEncoderUnavailable
    case commandExecutionFailed(String)
}

public struct MetalAttentionShape: Sendable, Equatable {
    public let heads: Int
    public let headDim: Int
    public let seqLen: Int

    public init(heads: Int, headDim: Int, seqLen: Int) throws(MetalAttentionError) {
        guard heads > 0 else {
            throw .invalidShape("heads must be > 0")
        }
        guard headDim > 0 else {
            throw .invalidShape("headDim must be > 0")
        }
        guard seqLen > 0 else {
            throw .invalidShape("seqLen must be > 0")
        }
        self.heads = heads
        self.headDim = headDim
        self.seqLen = seqLen
    }
}

public struct MetalAttentionBenchmarkResult: Sendable, Equatable {
    public let meanMs: Double
    public let medianMs: Double
    public let warmup: Int
    public let iterations: Int
    public let zeroCopyBindings: Bool
}

public struct MetalDecodeAttentionShape: Sendable, Equatable {
    public let heads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let visibleTokens: Int
    public let tokenBase: Int
    public let cacheStride: Int
    public let laneStride: Int

    public init(
        heads: Int,
        kvHeads: Int? = nil,
        headDim: Int,
        visibleTokens: Int,
        tokenBase: Int = 0,
        cacheStride: Int,
        laneStride: Int
    ) throws(MetalAttentionError) {
        let resolvedKVHeads = kvHeads ?? heads
        guard heads > 0 else {
            throw .invalidShape("heads must be > 0")
        }
        guard resolvedKVHeads > 0 else {
            throw .invalidShape("kvHeads must be > 0")
        }
        guard heads % resolvedKVHeads == 0 else {
            throw .invalidShape("heads must be divisible by kvHeads")
        }
        guard headDim > 0 else {
            throw .invalidShape("headDim must be > 0")
        }
        guard visibleTokens > 0 else {
            throw .invalidShape("visibleTokens must be > 0")
        }
        guard tokenBase >= 0 else {
            throw .invalidShape("tokenBase must be >= 0")
        }
        guard cacheStride >= visibleTokens else {
            throw .invalidShape("cacheStride must be >= visibleTokens")
        }
        guard tokenBase + visibleTokens <= cacheStride else {
            throw .invalidShape("tokenBase + visibleTokens must be <= cacheStride")
        }
        guard laneStride > 0 else {
            throw .invalidShape("laneStride must be > 0")
        }
        self.heads = heads
        self.kvHeads = resolvedKVHeads
        self.headDim = headDim
        self.visibleTokens = visibleTokens
        self.tokenBase = tokenBase
        self.cacheStride = cacheStride
        self.laneStride = laneStride
    }
}

public final class MetalAttentionKernel {
    static let moduloKVHeadMappingEnvKey = "ESPRESSO_METAL_KV_HEAD_MODULO"

    enum KVHeadMappingMode: Sendable, Equatable {
        case groupedContiguous
        case moduloInterleaved
    }

    static func resolvedKVHeadMappingMode(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> KVHeadMappingMode {
        guard let rawValue = environment[moduloKVHeadMappingEnvKey] else {
            return .groupedContiguous
        }
        switch rawValue.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1", "true", "yes", "on":
            return .moduloInterleaved
        default:
            return .groupedContiguous
        }
    }

    static func kvHeadIndex(
        queryHead: Int,
        heads: Int,
        kvHeads: Int,
        mode: KVHeadMappingMode
    ) -> Int {
        switch mode {
        case .groupedContiguous:
            return queryHead / (heads / kvHeads)
        case .moduloInterleaved:
            return queryHead % kvHeads
        }
    }

    private static var decodeKVHeadExpression: String {
        switch resolvedKVHeadMappingMode() {
        case .groupedContiguous:
            return "head / (params.heads / params.kvHeads)"
        case .moduloInterleaved:
            return "head % params.kvHeads"
        }
    }

    private struct AttentionParams {
        var heads: UInt32
        var headDim: UInt32
        var seqLen: UInt32
        var pad0: UInt32 = 0
        var scale: Float
        var pad1: Float = 0
        var pad2: Float = 0
        var pad3: Float = 0
    }

    fileprivate final class SurfaceBinding {
        let surface: IOSurfaceRef
        let buffer: MTLBuffer

        init(surface: IOSurfaceRef, byteCount: Int, device: MTLDevice) throws(MetalAttentionError) {
            self.surface = surface
            let status = IOSurfaceLock(surface, [], nil)
            guard status == 0 else {
                throw .surfaceLockFailed(status)
            }

            let baseAddress = IOSurfaceGetBaseAddress(surface)
            guard let buffer = device.makeBuffer(
                bytesNoCopy: baseAddress,
                length: byteCount,
                options: .storageModeShared,
                deallocator: nil
            ) else {
                IOSurfaceUnlock(surface, [], nil)
                throw .bufferBindingFailed
            }

            self.buffer = buffer
        }

        deinit {
            IOSurfaceUnlock(surface, [], nil)
        }

        convenience init(surface: IOSurfaceRef, elementCount: Int, device: MTLDevice) throws(MetalAttentionError) {
            try self.init(
                surface: surface,
                byteCount: elementCount * MemoryLayout<UInt16>.stride,
                device: device
            )
        }
    }

    private final class RunResources {
        let qSurface: IOSurfaceRef
        let kSurface: IOSurfaceRef
        let vSurface: IOSurfaceRef
        let maskSurface: IOSurfaceRef
        let outputSurface: IOSurfaceRef
        let qBinding: SurfaceBinding
        let kBinding: SurfaceBinding
        let vBinding: SurfaceBinding
        let maskBinding: SurfaceBinding
        let outputBinding: SurfaceBinding
        let scoresBuffer: MTLBuffer
        let weightsBuffer: MTLBuffer

        init(device: MTLDevice, shape: MetalAttentionShape) throws(MetalAttentionError) {
            let qCount = shape.heads * shape.headDim
            let kvCount = shape.heads * shape.seqLen * shape.headDim
            let maskCount = shape.heads * shape.seqLen
            let outputCount = qCount

            qSurface = try Self.makeSurface(channels: shape.heads, spatial: shape.headDim)
            kSurface = try Self.makeSurface(channels: shape.heads * shape.seqLen, spatial: shape.headDim)
            vSurface = try Self.makeSurface(channels: shape.heads * shape.seqLen, spatial: shape.headDim)
            maskSurface = try Self.makeSurface(channels: shape.heads, spatial: shape.seqLen)
            outputSurface = try Self.makeSurface(channels: shape.heads, spatial: shape.headDim)

            qBinding = try SurfaceBinding(surface: qSurface, elementCount: qCount, device: device)
            kBinding = try SurfaceBinding(surface: kSurface, elementCount: kvCount, device: device)
            vBinding = try SurfaceBinding(surface: vSurface, elementCount: kvCount, device: device)
            maskBinding = try SurfaceBinding(surface: maskSurface, elementCount: maskCount, device: device)
            outputBinding = try SurfaceBinding(surface: outputSurface, elementCount: outputCount, device: device)

            let floatStride = MemoryLayout<Float>.stride
            guard let scoresBuffer = device.makeBuffer(
                length: maskCount * floatStride,
                options: .storageModeShared
            ) else {
                throw .temporaryBufferAllocationFailed
            }
            guard let weightsBuffer = device.makeBuffer(
                length: maskCount * floatStride,
                options: .storageModeShared
            ) else {
                throw .temporaryBufferAllocationFailed
            }

            self.scoresBuffer = scoresBuffer
            self.weightsBuffer = weightsBuffer
        }

        private static func makeSurface(channels: Int, spatial: Int) throws(MetalAttentionError) -> IOSurfaceRef {
            guard channels > 0, spatial > 0 else {
                throw .surfaceCreateFailed
            }
            let bytesPerElement = MemoryLayout<UInt16>.stride
            let bytesPerRow = spatial * bytesPerElement
            let allocSize = channels * bytesPerRow
            let properties: [CFString: Any] = [
                kIOSurfaceWidth: spatial,
                kIOSurfaceHeight: channels,
                kIOSurfaceBytesPerElement: bytesPerElement,
                kIOSurfaceBytesPerRow: bytesPerRow,
                kIOSurfaceAllocSize: allocSize,
            ]
            guard let surface = IOSurfaceCreate(properties as CFDictionary) else {
                throw .surfaceCreateFailed
            }
            return surface
        }
    }

    private struct DecodeParams {
        var heads: UInt32
        var headDim: UInt32
        var visibleTokens: UInt32
        var cacheStride: UInt32
        var laneStride: UInt32
        var kvHeads: UInt32
        var tokenBase: UInt32
        var pad0: UInt32 = 0
        var scale: Float
        var pad1: Float = 0
    }

    private struct RoPEParams {
        var nHeads: UInt32
        var nKVHeads: UInt32
        var headDim: UInt32
        var position: UInt32
        var laneStride: UInt32
        var theta: Float
        var pad0: UInt32 = 0
        var pad1: UInt32 = 0
    }

    private struct KVScatterParams {
        var kvDim: UInt32
        var tokenIndex: UInt32
        var cacheStride: UInt32
        var laneStride: UInt32
    }

    /// RoPE configuration for Metal-side rotation.
    public struct MetalRoPEConfig {
        public let nHeads: Int
        public let nKVHeads: Int
        public let headDim: Int
        public let theta: Float

        public init(nHeads: Int, nKVHeads: Int, headDim: Int, theta: Float) {
            self.nHeads = nHeads
            self.nKVHeads = nKVHeads
            self.headDim = headDim
            self.theta = theta
        }
    }

    private final class DecodeScratch {
        let scoresBuffer: MTLBuffer
        let weightsBuffer: MTLBuffer
        let contextBuffer: MTLBuffer

        init(device: MTLDevice, heads: Int, cacheStride: Int, dim: Int) throws(MetalAttentionError) {
            let scoreLength = heads * cacheStride * MemoryLayout<Float>.stride
            let contextLength = dim * MemoryLayout<Float>.stride
            guard let scoresBuffer = device.makeBuffer(length: scoreLength, options: .storageModeShared),
                  let weightsBuffer = device.makeBuffer(length: scoreLength, options: .storageModeShared),
                  let contextBuffer = device.makeBuffer(length: contextLength, options: .storageModeShared) else {
                throw .temporaryBufferAllocationFailed
            }
            self.scoresBuffer = scoresBuffer
            self.weightsBuffer = weightsBuffer
            self.contextBuffer = contextBuffer
        }
    }

    private struct DecodeProjectionBuffers {
        let weights: MTLBuffer
        let bias: MTLBuffer
    }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let logitsPipeline: MTLComputePipelineState
    private let softmaxPipeline: MTLComputePipelineState
    private let outputPipeline: MTLComputePipelineState
    private let decodeLogitsPipeline: MTLComputePipelineState
    private let decodeOutputPipeline: MTLComputePipelineState
    private let decodeOutputStridedPipeline: MTLComputePipelineState
    private let decodeProjectionPipeline: MTLComputePipelineState
    private let fusedDecodeSDPAPipeline: MTLComputePipelineState
    private let ropeDecodePipeline: MTLComputePipelineState
    private let kvCacheScatterPipeline: MTLComputePipelineState
    private var decodeScratchBuffers: [Int: DecodeScratch] = [:]
    private var projectionBuffers: [String: DecodeProjectionBuffers] = [:]

    public init() throws(MetalAttentionError) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw .metalUnavailable
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

        guard let logitsFunction = library.makeFunction(name: "attention_logits") else {
            throw .libraryBuildFailed("missing attention_logits")
        }
        guard let softmaxFunction = library.makeFunction(name: "attention_softmax") else {
            throw .libraryBuildFailed("missing attention_softmax")
        }
        guard let outputFunction = library.makeFunction(name: "attention_output") else {
            throw .libraryBuildFailed("missing attention_output")
        }
        guard let decodeLogitsFunction = library.makeFunction(name: "decode_attention_logits") else {
            throw .libraryBuildFailed("missing decode_attention_logits")
        }
        guard let decodeOutputFunction = library.makeFunction(name: "decode_attention_output") else {
            throw .libraryBuildFailed("missing decode_attention_output")
        }
        guard let decodeOutputStridedFunction = library.makeFunction(name: "decode_attention_output_strided") else {
            throw .libraryBuildFailed("missing decode_attention_output_strided")
        }
        guard let decodeProjectionFunction = library.makeFunction(name: "decode_output_projection") else {
            throw .libraryBuildFailed("missing decode_output_projection")
        }
        guard let fusedDecodeSDPAFunction = library.makeFunction(name: "fused_decode_sdpa") else {
            throw .libraryBuildFailed("missing fused_decode_sdpa")
        }
        guard let ropeDecodeFunction = library.makeFunction(name: "rope_decode") else {
            throw .libraryBuildFailed("missing rope_decode")
        }
        guard let kvCacheScatterFunction = library.makeFunction(name: "kv_cache_scatter") else {
            throw .libraryBuildFailed("missing kv_cache_scatter")
        }

        do {
            logitsPipeline = try device.makeComputePipelineState(function: logitsFunction)
            softmaxPipeline = try device.makeComputePipelineState(function: softmaxFunction)
            outputPipeline = try device.makeComputePipelineState(function: outputFunction)
            decodeLogitsPipeline = try device.makeComputePipelineState(function: decodeLogitsFunction)
            decodeOutputPipeline = try device.makeComputePipelineState(function: decodeOutputFunction)
            decodeOutputStridedPipeline = try device.makeComputePipelineState(function: decodeOutputStridedFunction)
            decodeProjectionPipeline = try device.makeComputePipelineState(function: decodeProjectionFunction)
            fusedDecodeSDPAPipeline = try device.makeComputePipelineState(function: fusedDecodeSDPAFunction)
            ropeDecodePipeline = try device.makeComputePipelineState(function: ropeDecodeFunction)
            kvCacheScatterPipeline = try device.makeComputePipelineState(function: kvCacheScatterFunction)
        } catch {
            throw .pipelineBuildFailed(String(describing: error))
        }

        self.device = device
        self.commandQueue = commandQueue
    }

    public func run(
        q: [Float],
        k: [Float],
        v: [Float],
        mask: [Float],
        shape: MetalAttentionShape
    ) throws(MetalAttentionError) -> [Float] {
        try validateInputCounts(q: q, k: k, v: v, mask: mask, shape: shape)
        let resources = try RunResources(device: device, shape: shape)
        try loadInputs(q: q, k: k, v: v, mask: mask, resources: resources, shape: shape)
        try encodeAndWait(resources: resources, shape: shape)
        return readOutput(resources: resources, shape: shape)
    }

    public func benchmark(
        shape: MetalAttentionShape,
        warmup: Int,
        iterations: Int,
        seed: UInt64
    ) throws(MetalAttentionError) -> MetalAttentionBenchmarkResult {
        guard warmup >= 0 else {
            throw .invalidShape("warmup must be >= 0")
        }
        guard iterations > 0 else {
            throw .invalidShape("iterations must be > 0")
        }

        let qCount = shape.heads * shape.headDim
        let kvCount = shape.heads * shape.seqLen * shape.headDim
        let maskCount = shape.heads * shape.seqLen

        let q = Self.makeInput(count: qCount, seed: seed ^ 0x13579BDF)
        let k = Self.makeInput(count: kvCount, seed: seed ^ 0x2468ACE0)
        let v = Self.makeInput(count: kvCount, seed: seed ^ 0xDEADBEEF)
        let mask = Self.makeMask(count: maskCount)

        let resources = try RunResources(device: device, shape: shape)
        try loadInputs(q: q, k: k, v: v, mask: mask, resources: resources, shape: shape)

        for _ in 0..<warmup {
            try encodeAndWait(resources: resources, shape: shape)
        }

        var samples: [Double] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let start = mach_absolute_time()
            try encodeAndWait(resources: resources, shape: shape)
            let end = mach_absolute_time()
            samples.append(Self.elapsedMilliseconds(start: start, end: end))
        }

        let mean = samples.reduce(0, +) / Double(samples.count)
        let median = Self.median(samples)
        return MetalAttentionBenchmarkResult(
            meanMs: mean,
            medianMs: median,
            warmup: warmup,
            iterations: iterations,
            zeroCopyBindings: true
        )
    }

    public func runDecode(
        qSurface: IOSurfaceRef,
        kCacheSurface: IOSurfaceRef,
        vCacheSurface: IOSurfaceRef,
        residualSurface: IOSurfaceRef,
        outputSurface: IOSurfaceRef,
        shape: MetalDecodeAttentionShape,
        projection: HybridOutputProjectionWeights
    ) throws(MetalAttentionError) {
        let attentionDim = shape.heads * shape.headDim
        let kvDim = shape.kvHeads * shape.headDim
        guard projection.rowMajorWeights.count == projection.outputDim * projection.inputDim else {
            throw .invalidInputCount(
                "output projection count \(projection.rowMajorWeights.count) != expected \(projection.outputDim * projection.inputDim)"
            )
        }
        guard projection.rowMajorBias.count == projection.outputDim else {
            throw .invalidInputCount(
                "output projection bias count \(projection.rowMajorBias.count) != expected \(projection.outputDim)"
            )
        }

        let qLaneElementCount = attentionDim * shape.laneStride
        let kvCacheElementCount = kvDim * shape.cacheStride
        let outputLaneElementCount = projection.outputDim * shape.laneStride
        let qBinding = try SurfaceBinding(surface: qSurface, elementCount: qLaneElementCount, device: device)
        let kBinding = try SurfaceBinding(surface: kCacheSurface, elementCount: kvCacheElementCount, device: device)
        let vBinding = try SurfaceBinding(surface: vCacheSurface, elementCount: kvCacheElementCount, device: device)
        let residualBinding = try SurfaceBinding(surface: residualSurface, elementCount: outputLaneElementCount, device: device)
        let outputBinding = try SurfaceBinding(surface: outputSurface, elementCount: outputLaneElementCount, device: device)
        let scratch = try decodeScratch(for: shape, dim: attentionDim)
        let projectionBuffers = try decodeProjectionBuffers(for: projection)
        try encodeDecodeAndWait(
            qBinding: qBinding,
            kBinding: kBinding,
            vBinding: vBinding,
            residualBinding: residualBinding,
            outputBinding: outputBinding,
            scratch: scratch,
            projectionWeightsBuffer: projectionBuffers.weights,
            projectionBiasBuffer: projectionBuffers.bias,
            shape: shape
        )
    }

    public func runDecodeContext(
        qSurface: IOSurfaceRef,
        kCacheSurface: IOSurfaceRef,
        vCacheSurface: IOSurfaceRef,
        shape: MetalDecodeAttentionShape
    ) throws(MetalAttentionError) -> [Float] {
        let dim = shape.heads * shape.headDim
        let kvDim = shape.kvHeads * shape.headDim
        let laneElementCount = dim * shape.laneStride
        let kvCacheElementCount = kvDim * shape.cacheStride
        let qBinding = try SurfaceBinding(surface: qSurface, elementCount: laneElementCount, device: device)
        let kBinding = try SurfaceBinding(surface: kCacheSurface, elementCount: kvCacheElementCount, device: device)
        let vBinding = try SurfaceBinding(surface: vCacheSurface, elementCount: kvCacheElementCount, device: device)
        let scratch = try decodeScratch(for: shape, dim: dim)
        try encodeDecodeContextAndWait(
            qBinding: qBinding,
            kBinding: kBinding,
            vBinding: vBinding,
            contextBuffer: scratch.contextBuffer,
            outputPipeline: decodeOutputPipeline,
            scratch: scratch,
            shape: shape
        )
        let pointer = scratch.contextBuffer.contents().assumingMemoryBound(to: Float.self)
        return Array(UnsafeBufferPointer(start: pointer, count: dim))
    }

    public func runDecodeContextIntoSurface(
        qSurface: IOSurfaceRef,
        kCacheSurface: IOSurfaceRef,
        vCacheSurface: IOSurfaceRef,
        contextSurface: IOSurfaceRef,
        shape: MetalDecodeAttentionShape
    ) throws(MetalAttentionError) {
        let dim = shape.heads * shape.headDim
        let kvDim = shape.kvHeads * shape.headDim
        let laneElementCount = dim * shape.laneStride
        let kvCacheElementCount = kvDim * shape.cacheStride
        let qBinding = try SurfaceBinding(surface: qSurface, elementCount: laneElementCount, device: device)
        let kBinding = try SurfaceBinding(surface: kCacheSurface, elementCount: kvCacheElementCount, device: device)
        let vBinding = try SurfaceBinding(surface: vCacheSurface, elementCount: kvCacheElementCount, device: device)
        let contextBinding = try SurfaceBinding(
            surface: contextSurface,
            byteCount: laneElementCount * MemoryLayout<Float>.stride,
            device: device
        )
        let scratch = try decodeScratch(for: shape, dim: dim)
        try encodeDecodeContextAndWait(
            qBinding: qBinding,
            kBinding: kBinding,
            vBinding: vBinding,
            contextBuffer: contextBinding.buffer,
            outputPipeline: decodeOutputStridedPipeline,
            scratch: scratch,
            shape: shape
        )
    }

    // MARK: - Cached Surface Metadata (Phase 1 optimization)

    fileprivate final class TransientAttentionBindings {
        let qBinding: SurfaceBinding
        let kBinding: SurfaceBinding
        let vBinding: SurfaceBinding
        let contextBinding: SurfaceBinding

        init(cached: CachedLayerBindings, device: MTLDevice) throws(MetalAttentionError) {
            qBinding = try SurfaceBinding(
                surface: cached.qSurface,
                byteCount: cached.qByteCount,
                device: device
            )
            kBinding = try SurfaceBinding(
                surface: cached.kCacheSurface,
                byteCount: cached.kvCacheByteCount,
                device: device
            )
            vBinding = try SurfaceBinding(
                surface: cached.vCacheSurface,
                byteCount: cached.kvCacheByteCount,
                device: device
            )
            contextBinding = try SurfaceBinding(
                surface: cached.contextSurface,
                byteCount: cached.contextByteCount,
                device: device
            )
        }

        var all: [SurfaceBinding] {
            [qBinding, kBinding, vBinding, contextBinding]
        }
    }

    fileprivate final class TransientKVCacheBindings {
        let kBinding: SurfaceBinding
        let vBinding: SurfaceBinding

        init(cached: CachedLayerBindings, device: MTLDevice) throws(MetalAttentionError) {
            kBinding = try SurfaceBinding(
                surface: cached.kCacheSurface,
                byteCount: cached.kvCacheByteCount,
                device: device
            )
            vBinding = try SurfaceBinding(
                surface: cached.vCacheSurface,
                byteCount: cached.kvCacheByteCount,
                device: device
            )
        }

        var all: [SurfaceBinding] {
            [kBinding, vBinding]
        }
    }

    fileprivate final class RetainedSurfaceBindings: @unchecked Sendable {
        let bindings: [SurfaceBinding]

        init(_ bindings: [SurfaceBinding]) {
            self.bindings = bindings
        }
    }

    /// Cached per-layer surface metadata for decode bindings.
    /// Mutable IOSurfaces are rebound per dispatch so Metal never holds a
    /// long-lived `MTLBuffer(bytesNoCopy:)` across ANE/CPU writes.
    public final class CachedLayerBindings {
        fileprivate let qSurface: IOSurfaceRef
        fileprivate let kOutputSurface: IOSurfaceRef?
        fileprivate let vOutputSurface: IOSurfaceRef?
        fileprivate let kCacheSurface: IOSurfaceRef
        fileprivate let vCacheSurface: IOSurfaceRef
        fileprivate let contextSurface: IOSurfaceRef
        fileprivate let qByteCount: Int
        fileprivate let kvOutputByteCount: Int
        fileprivate let kvCacheByteCount: Int
        fileprivate let contextByteCount: Int

        public init(
            qSurface: IOSurfaceRef,
            kOutputSurface: IOSurfaceRef? = nil,
            vOutputSurface: IOSurfaceRef? = nil,
            kCacheSurface: IOSurfaceRef,
            vCacheSurface: IOSurfaceRef,
            contextSurface: IOSurfaceRef,
            dim: Int,
            kvDim: Int,
            laneStride: Int,
            cacheStride: Int,
            device: MTLDevice
        ) throws(MetalAttentionError) {
            let laneElementCount = dim * laneStride
            let kvCacheElementCount = kvDim * cacheStride
            _ = device
            self.qSurface = qSurface
            self.kOutputSurface = kOutputSurface
            self.vOutputSurface = vOutputSurface
            self.kCacheSurface = kCacheSurface
            self.vCacheSurface = vCacheSurface
            self.contextSurface = contextSurface
            self.qByteCount = laneElementCount * MemoryLayout<UInt16>.stride
            self.kvOutputByteCount = kvDim * laneStride * MemoryLayout<UInt16>.stride
            self.kvCacheByteCount = kvCacheElementCount * MemoryLayout<UInt16>.stride
            self.contextByteCount = laneElementCount * MemoryLayout<Float>.stride
        }

        fileprivate func makeTransientAttentionBindings(device: MTLDevice) throws(MetalAttentionError) -> TransientAttentionBindings {
            try TransientAttentionBindings(cached: self, device: device)
        }

        fileprivate func makeTransientKVCacheBindings(device: MTLDevice) throws(MetalAttentionError) -> TransientKVCacheBindings {
            try TransientKVCacheBindings(cached: self, device: device)
        }

        fileprivate func makeTransientKVOutputBindings(device: MTLDevice) throws(MetalAttentionError) -> TransientKVOutputBindings? {
            guard let kOutputSurface, let vOutputSurface else {
                return nil
            }
            return try TransientKVOutputBindings(
                kOutputSurface: kOutputSurface,
                vOutputSurface: vOutputSurface,
                byteCount: kvOutputByteCount,
                device: device
            )
        }
    }

    fileprivate final class TransientKVOutputBindings {
        let kOutBinding: SurfaceBinding
        let vOutBinding: SurfaceBinding

        init(
            kOutputSurface: IOSurfaceRef,
            vOutputSurface: IOSurfaceRef,
            byteCount: Int,
            device: MTLDevice
        ) throws(MetalAttentionError) {
            kOutBinding = try SurfaceBinding(surface: kOutputSurface, byteCount: byteCount, device: device)
            vOutBinding = try SurfaceBinding(surface: vOutputSurface, byteCount: byteCount, device: device)
        }

        var all: [SurfaceBinding] {
            [kOutBinding, vOutBinding]
        }
    }

    private static func retain(bindings: [SurfaceBinding], until commandBuffer: MTLCommandBuffer) {
        let retained = RetainedSurfaceBindings(bindings)
        commandBuffer.addCompletedHandler { _ in
            _ = retained
        }
    }

    /// Cache immutable surface metadata for a layer's decode surfaces. Call once at engine init time.
    public func createCachedLayerBindings(
        qSurface: IOSurfaceRef,
        kOutputSurface: IOSurfaceRef? = nil,
        vOutputSurface: IOSurfaceRef? = nil,
        kCacheSurface: IOSurfaceRef,
        vCacheSurface: IOSurfaceRef,
        contextSurface: IOSurfaceRef,
        dim: Int,
        kvDim: Int,
        laneStride: Int,
        cacheStride: Int
    ) throws(MetalAttentionError) -> CachedLayerBindings {
        try CachedLayerBindings(
            qSurface: qSurface,
            kOutputSurface: kOutputSurface,
            vOutputSurface: vOutputSurface,
            kCacheSurface: kCacheSurface,
            vCacheSurface: vCacheSurface,
            contextSurface: contextSurface,
            dim: dim,
            kvDim: kvDim,
            laneStride: laneStride,
            cacheStride: cacheStride,
            device: device
        )
    }

    /// Fast-path decode using cached surface metadata. Mutable surfaces are rebound
    /// on demand so each dispatch gets fresh Metal views over the latest contents.
    public func runDecodeContextFromCachedBindings(
        cached: CachedLayerBindings,
        shape: MetalDecodeAttentionShape
    ) throws(MetalAttentionError) {
        let dim = shape.heads * shape.headDim
        let scratch = try decodeScratch(for: shape, dim: dim)
        let bindings = try cached.makeTransientAttentionBindings(device: device)
        try encodeDecodeContextAndWait(
            qBinding: bindings.qBinding,
            kBinding: bindings.kBinding,
            vBinding: bindings.vBinding,
            contextBuffer: bindings.contextBinding.buffer,
            outputPipeline: decodeOutputStridedPipeline,
            scratch: scratch,
            shape: shape
        )
    }

    // MARK: - Fused SDPA (Phase 3 optimization)

    /// Fused decode SDPA from cached bindings: single Metal dispatch per head
    /// instead of 3 separate dispatches (logits, softmax, output).
    public func runFusedDecodeSDPAFromCachedBindings(
        cached: CachedLayerBindings,
        shape: MetalDecodeAttentionShape
    ) throws(MetalAttentionError) {
        let bindings = try cached.makeTransientAttentionBindings(device: device)
        try encodeFusedDecodeSDPAAndWait(
            qBinding: bindings.qBinding,
            kBinding: bindings.kBinding,
            vBinding: bindings.vBinding,
            contextBuffer: bindings.contextBinding.buffer,
            shape: shape
        )
    }

    /// Fused decode SDPA from raw surfaces (creates temporary bindings).
    public func runFusedDecodeSDPAIntoSurface(
        qSurface: IOSurfaceRef,
        kCacheSurface: IOSurfaceRef,
        vCacheSurface: IOSurfaceRef,
        contextSurface: IOSurfaceRef,
        shape: MetalDecodeAttentionShape
    ) throws(MetalAttentionError) {
        let dim = shape.heads * shape.headDim
        let kvDim = shape.kvHeads * shape.headDim
        let laneElementCount = dim * shape.laneStride
        let kvCacheElementCount = kvDim * shape.cacheStride
        let qBinding = try SurfaceBinding(surface: qSurface, elementCount: laneElementCount, device: device)
        let kBinding = try SurfaceBinding(surface: kCacheSurface, elementCount: kvCacheElementCount, device: device)
        let vBinding = try SurfaceBinding(surface: vCacheSurface, elementCount: kvCacheElementCount, device: device)
        let contextBinding = try SurfaceBinding(
            surface: contextSurface,
            byteCount: laneElementCount * MemoryLayout<Float>.stride,
            device: device
        )
        try encodeFusedDecodeSDPAAndWait(
            qBinding: qBinding,
            kBinding: kBinding,
            vBinding: vBinding,
            contextBuffer: contextBinding.buffer,
            shape: shape
        )
    }

    private func encodeFusedDecodeSDPA(
        qBinding: SurfaceBinding,
        kBinding: SurfaceBinding,
        vBinding: SurfaceBinding,
        contextBuffer: MTLBuffer,
        shape: MetalDecodeAttentionShape
    ) throws(MetalAttentionError) -> MTLCommandBuffer {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw .commandBufferUnavailable
        }

        let decodeParams = DecodeParams(
            heads: UInt32(shape.heads),
            headDim: UInt32(shape.headDim),
            visibleTokens: UInt32(shape.visibleTokens),
            cacheStride: UInt32(shape.cacheStride),
            laneStride: UInt32(shape.laneStride),
            kvHeads: UInt32(shape.kvHeads),
            tokenBase: UInt32(shape.tokenBase),
            scale: 1.0 / sqrt(Float(shape.headDim))
        )

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        encoder.setComputePipelineState(fusedDecodeSDPAPipeline)
        encoder.setBuffer(qBinding.buffer, offset: 0, index: 0)
        encoder.setBuffer(kBinding.buffer, offset: 0, index: 1)
        encoder.setBuffer(vBinding.buffer, offset: 0, index: 2)
        encoder.setBuffer(contextBuffer, offset: 0, index: 3)
        withUnsafeBytes(of: decodeParams) { rawBytes in
            encoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 4)
        }
        // Threadgroup memory for scores: visibleTokens floats
        let tgMemSize = shape.visibleTokens * MemoryLayout<Float>.stride
        encoder.setThreadgroupMemoryLength(tgMemSize, index: 0)

        let tgWidth = min(shape.visibleTokens, fusedDecodeSDPAPipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreadgroups(
            MTLSize(width: shape.heads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1)
        )
        encoder.endEncoding()
        return commandBuffer
    }

    private func encodeFusedDecodeSDPAAndWait(
        qBinding: SurfaceBinding,
        kBinding: SurfaceBinding,
        vBinding: SurfaceBinding,
        contextBuffer: MTLBuffer,
        shape: MetalDecodeAttentionShape
    ) throws(MetalAttentionError) {
        let commandBuffer = try encodeFusedDecodeSDPA(
            qBinding: qBinding, kBinding: kBinding, vBinding: vBinding,
            contextBuffer: contextBuffer, shape: shape
        )
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if commandBuffer.status != .completed {
            throw .commandExecutionFailed(commandBuffer.error?.localizedDescription ?? "status=\(commandBuffer.status.rawValue)")
        }
    }

    // MARK: - Pipelined Submission (Phase 4)

    /// Submit fused SDPA without waiting — returns the committed command buffer.
    /// Caller must call `waitForMetalCompletion(_:)` before reading the context surface.
    public func submitFusedDecodeSDPAFromCachedBindings(
        cached: CachedLayerBindings,
        shape: MetalDecodeAttentionShape
    ) throws(MetalAttentionError) -> MTLCommandBuffer {
        let bindings = try cached.makeTransientAttentionBindings(device: device)
        let cb = try encodeFusedDecodeSDPA(
            qBinding: bindings.qBinding,
            kBinding: bindings.kBinding,
            vBinding: bindings.vBinding,
            contextBuffer: bindings.contextBinding.buffer,
            shape: shape
        )
        Self.retain(bindings: bindings.all, until: cb)
        cb.commit()
        return cb
    }

    /// Wait for a previously submitted command buffer to complete.
    public func waitForMetalCompletion(_ commandBuffer: MTLCommandBuffer) throws(MetalAttentionError) {
        commandBuffer.waitUntilCompleted()
        if commandBuffer.status != .completed {
            throw .commandExecutionFailed(
                commandBuffer.error?.localizedDescription ?? "status=\(commandBuffer.status.rawValue)"
            )
        }
    }

    /// Submit all layers' fused SDPA as a SINGLE command buffer.
    /// Eliminates per-layer command buffer creation/commit/sync overhead.
    public func submitBatchedFusedDecodeSDPA(
        cachedLayers: [CachedLayerBindings],
        shape: MetalDecodeAttentionShape
    ) throws(MetalAttentionError) -> MTLCommandBuffer {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw .commandBufferUnavailable
        }

        let decodeParams = DecodeParams(
            heads: UInt32(shape.heads),
            headDim: UInt32(shape.headDim),
            visibleTokens: UInt32(shape.visibleTokens),
            cacheStride: UInt32(shape.cacheStride),
            laneStride: UInt32(shape.laneStride),
            kvHeads: UInt32(shape.kvHeads),
            tokenBase: UInt32(shape.tokenBase),
            scale: 1.0 / sqrt(Float(shape.headDim))
        )

        var retainedBindings: [SurfaceBinding] = []
        for cached in cachedLayers {
            let bindings = try cached.makeTransientAttentionBindings(device: device)
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw .commandEncoderUnavailable
            }
            encoder.setComputePipelineState(fusedDecodeSDPAPipeline)
            encoder.setBuffer(bindings.qBinding.buffer, offset: 0, index: 0)
            encoder.setBuffer(bindings.kBinding.buffer, offset: 0, index: 1)
            encoder.setBuffer(bindings.vBinding.buffer, offset: 0, index: 2)
            encoder.setBuffer(bindings.contextBinding.buffer, offset: 0, index: 3)
            withUnsafeBytes(of: decodeParams) { rawBytes in
                encoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 4)
            }
            let tgMemSize = shape.visibleTokens * MemoryLayout<Float>.stride
            encoder.setThreadgroupMemoryLength(tgMemSize, index: 0)
            let tgWidth = min(shape.visibleTokens, fusedDecodeSDPAPipeline.maxTotalThreadsPerThreadgroup)
            encoder.dispatchThreadgroups(
                MTLSize(width: shape.heads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1)
            )
            encoder.endEncoding()
            retainedBindings.append(contentsOf: bindings.all)
        }
        Self.retain(bindings: retainedBindings, until: commandBuffer)
        commandBuffer.commit()
        return commandBuffer
    }

    // MARK: - Fused RoPE + KV Scatter + SDPA (Phase 5)

    /// Submit a single command buffer that encodes:
    /// 1. RoPE rotation on Q and K surfaces in-place
    /// 2. KV cache scatter (copy K,V at lane 0 to cache at tokenIndex)
    /// 3. Fused SDPA (Q*K^T, softmax, weighted V)
    /// Returns the committed (non-blocking) command buffer.
    public func submitFullPipelinedDecodeFromCachedBindings(
        cached: CachedLayerBindings,
        shape: MetalDecodeAttentionShape,
        ropeConfig: MetalRoPEConfig,
        tokenIndex: Int
    ) throws(MetalAttentionError) -> MTLCommandBuffer {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw .commandBufferUnavailable
        }
        let attentionBindings = try cached.makeTransientAttentionBindings(device: device)
        guard let kvOutputBindings = try cached.makeTransientKVOutputBindings(device: device) else {
            throw .invalidInputCount("cached bindings are missing K/V output surfaces required for Metal RoPE fast path")
        }

        // Encode RoPE
        let ropeParams = RoPEParams(
            nHeads: UInt32(ropeConfig.nHeads),
            nKVHeads: UInt32(ropeConfig.nKVHeads),
            headDim: UInt32(ropeConfig.headDim),
            position: UInt32(tokenIndex),
            laneStride: UInt32(shape.laneStride),
            theta: ropeConfig.theta
        )
        guard let ropeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        ropeEncoder.setComputePipelineState(ropeDecodePipeline)
        ropeEncoder.setBuffer(attentionBindings.qBinding.buffer, offset: 0, index: 0)
        ropeEncoder.setBuffer(kvOutputBindings.kOutBinding.buffer, offset: 0, index: 1)
        withUnsafeBytes(of: ropeParams) { rawBytes in
            ropeEncoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 2)
        }
        let maxHeads = max(ropeConfig.nHeads, ropeConfig.nKVHeads)
        let halfDim = ropeConfig.headDim / 2
        ropeEncoder.dispatchThreads(
            MTLSize(width: halfDim, height: maxHeads, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(halfDim, ropeDecodePipeline.threadExecutionWidth),
                height: 1,
                depth: 1
            )
        )
        ropeEncoder.endEncoding()

        // Encode KV scatter from ANE K/V outputs into the cache surfaces.
        let scatterParams = KVScatterParams(
            kvDim: UInt32(shape.kvHeads * shape.headDim),
            tokenIndex: UInt32(tokenIndex),
            cacheStride: UInt32(shape.cacheStride),
            laneStride: UInt32(shape.laneStride)
        )
        guard let scatterEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        scatterEncoder.setComputePipelineState(kvCacheScatterPipeline)
        scatterEncoder.setBuffer(kvOutputBindings.kOutBinding.buffer, offset: 0, index: 0)
        scatterEncoder.setBuffer(kvOutputBindings.vOutBinding.buffer, offset: 0, index: 1)
        scatterEncoder.setBuffer(attentionBindings.kBinding.buffer, offset: 0, index: 2)
        scatterEncoder.setBuffer(attentionBindings.vBinding.buffer, offset: 0, index: 3)
        withUnsafeBytes(of: scatterParams) { rawBytes in
            scatterEncoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 4)
        }
        let kvDim = shape.kvHeads * shape.headDim
        scatterEncoder.dispatchThreads(
            MTLSize(width: kvDim, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(kvDim, kvCacheScatterPipeline.threadExecutionWidth),
                height: 1,
                depth: 1
            )
        )
        scatterEncoder.endEncoding()

        // Encode fused SDPA
        let decodeParams = DecodeParams(
            heads: UInt32(shape.heads),
            headDim: UInt32(shape.headDim),
            visibleTokens: UInt32(shape.visibleTokens),
            cacheStride: UInt32(shape.cacheStride),
            laneStride: UInt32(shape.laneStride),
            kvHeads: UInt32(shape.kvHeads),
            tokenBase: UInt32(shape.tokenBase),
            scale: 1.0 / sqrt(Float(shape.headDim))
        )
        guard let sdpaEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        sdpaEncoder.setComputePipelineState(fusedDecodeSDPAPipeline)
        sdpaEncoder.setBuffer(attentionBindings.qBinding.buffer, offset: 0, index: 0)
        sdpaEncoder.setBuffer(attentionBindings.kBinding.buffer, offset: 0, index: 1)
        sdpaEncoder.setBuffer(attentionBindings.vBinding.buffer, offset: 0, index: 2)
        sdpaEncoder.setBuffer(attentionBindings.contextBinding.buffer, offset: 0, index: 3)
        withUnsafeBytes(of: decodeParams) { rawBytes in
            sdpaEncoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 4)
        }
        let tgMemSize = shape.visibleTokens * MemoryLayout<Float>.stride
        sdpaEncoder.setThreadgroupMemoryLength(tgMemSize, index: 0)
        let tgWidth = min(shape.visibleTokens, fusedDecodeSDPAPipeline.maxTotalThreadsPerThreadgroup)
        sdpaEncoder.dispatchThreadgroups(
            MTLSize(width: shape.heads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgWidth, height: 1, depth: 1)
        )
        sdpaEncoder.endEncoding()

        Self.retain(bindings: attentionBindings.all + kvOutputBindings.all, until: commandBuffer)
        commandBuffer.commit()
        return commandBuffer
    }

    /// Encode KV cache scatter into an existing command buffer.
    /// Called separately because K/V output surfaces are not in CachedLayerBindings.
    public func encodeKVCacheScatter(
        commandBuffer: MTLCommandBuffer,
        kOutputSurface: IOSurfaceRef,
        vOutputSurface: IOSurfaceRef,
        kCacheBinding: CachedLayerBindings,
        shape: MetalDecodeAttentionShape,
        kvDim: Int,
        tokenIndex: Int
    ) throws(MetalAttentionError) {
        let kvOutputElementCount = kvDim * shape.laneStride
        let kOutBinding = try SurfaceBinding(surface: kOutputSurface, elementCount: kvOutputElementCount, device: device)
        let vOutBinding = try SurfaceBinding(surface: vOutputSurface, elementCount: kvOutputElementCount, device: device)
        let cacheBindings = try kCacheBinding.makeTransientKVCacheBindings(device: device)

        let params = KVScatterParams(
            kvDim: UInt32(kvDim),
            tokenIndex: UInt32(tokenIndex),
            cacheStride: UInt32(shape.cacheStride),
            laneStride: UInt32(shape.laneStride)
        )
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        encoder.setComputePipelineState(kvCacheScatterPipeline)
        encoder.setBuffer(kOutBinding.buffer, offset: 0, index: 0)
        encoder.setBuffer(vOutBinding.buffer, offset: 0, index: 1)
        encoder.setBuffer(cacheBindings.kBinding.buffer, offset: 0, index: 2)
        encoder.setBuffer(cacheBindings.vBinding.buffer, offset: 0, index: 3)
        withUnsafeBytes(of: params) { rawBytes in
            encoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 4)
        }
        encoder.dispatchThreads(
            MTLSize(width: kvDim, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: min(kvDim, kvCacheScatterPipeline.threadExecutionWidth),
                height: 1,
                depth: 1
            )
        )
        encoder.endEncoding()

        Self.retain(
            bindings: [kOutBinding, vOutBinding] + cacheBindings.all,
            until: commandBuffer
        )
    }

    private func validateInputCounts(
        q: [Float],
        k: [Float],
        v: [Float],
        mask: [Float],
        shape: MetalAttentionShape
    ) throws(MetalAttentionError) {
        let qExpected = shape.heads * shape.headDim
        let kvExpected = shape.heads * shape.seqLen * shape.headDim
        let maskExpected = shape.heads * shape.seqLen
        guard q.count == qExpected else {
            throw .invalidInputCount("q count \(q.count) != expected \(qExpected)")
        }
        guard k.count == kvExpected else {
            throw .invalidInputCount("k count \(k.count) != expected \(kvExpected)")
        }
        guard v.count == kvExpected else {
            throw .invalidInputCount("v count \(v.count) != expected \(kvExpected)")
        }
        guard mask.count == maskExpected else {
            throw .invalidInputCount("mask count \(mask.count) != expected \(maskExpected)")
        }
    }

    private func loadInputs(
        q: [Float],
        k: [Float],
        v: [Float],
        mask: [Float],
        resources: RunResources,
        shape: MetalAttentionShape
    ) throws(MetalAttentionError) {
        q.withUnsafeBufferPointer { ptr in
            SurfaceIO.writeFP16(to: resources.qSurface, data: ptr, channels: shape.heads, spatial: shape.headDim)
        }
        k.withUnsafeBufferPointer { ptr in
            SurfaceIO.writeFP16(
                to: resources.kSurface,
                data: ptr,
                channels: shape.heads * shape.seqLen,
                spatial: shape.headDim
            )
        }
        v.withUnsafeBufferPointer { ptr in
            SurfaceIO.writeFP16(
                to: resources.vSurface,
                data: ptr,
                channels: shape.heads * shape.seqLen,
                spatial: shape.headDim
            )
        }
        mask.withUnsafeBufferPointer { ptr in
            SurfaceIO.writeFP16(to: resources.maskSurface, data: ptr, channels: shape.heads, spatial: shape.seqLen)
        }
    }

    private func readOutput(resources: RunResources, shape: MetalAttentionShape) -> [Float] {
        var output = [Float](repeating: 0, count: shape.heads * shape.headDim)
        output.withUnsafeMutableBufferPointer { ptr in
            SurfaceIO.readFP16(
                from: resources.outputSurface,
                into: ptr,
                channelOffset: 0,
                channels: shape.heads,
                spatial: shape.headDim
            )
        }
        return output
    }

    private func decodeScratch(for shape: MetalDecodeAttentionShape, dim: Int) throws(MetalAttentionError) -> DecodeScratch {
        let key = shape.heads << 20 ^ shape.cacheStride << 8 ^ shape.laneStride
        if let scratch = decodeScratchBuffers[key] {
            return scratch
        }
        let scratch = try DecodeScratch(device: device, heads: shape.heads, cacheStride: shape.cacheStride, dim: dim)
        decodeScratchBuffers[key] = scratch
        return scratch
    }

    private func decodeProjectionBuffers(
        for projection: HybridOutputProjectionWeights
    ) throws(MetalAttentionError) -> DecodeProjectionBuffers {
        if let buffers = projectionBuffers[projection.cacheKey] {
            return buffers
        }
        let weightsLength = projection.rowMajorWeights.count * MemoryLayout<Float>.stride
        let biasLength = projection.rowMajorBias.count * MemoryLayout<Float>.stride
        let weightsBuffer = projection.rowMajorWeights.withUnsafeBytes { rawBytes -> MTLBuffer? in
            device.makeBuffer(bytes: rawBytes.baseAddress!, length: weightsLength, options: .storageModeShared)
        }
        let biasBuffer = projection.rowMajorBias.withUnsafeBytes { rawBytes -> MTLBuffer? in
            device.makeBuffer(bytes: rawBytes.baseAddress!, length: biasLength, options: .storageModeShared)
        }
        guard let weightsBuffer, let biasBuffer else {
            throw .bufferBindingFailed
        }
        let buffers = DecodeProjectionBuffers(weights: weightsBuffer, bias: biasBuffer)
        projectionBuffers[projection.cacheKey] = buffers
        return buffers
    }

    private func encodeAndWait(resources: RunResources, shape: MetalAttentionShape) throws(MetalAttentionError) {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw .commandBufferUnavailable
        }
        let params = AttentionParams(
            heads: UInt32(shape.heads),
            headDim: UInt32(shape.headDim),
            seqLen: UInt32(shape.seqLen),
            scale: 1.0 / sqrt(Float(shape.headDim))
        )

        guard let logitsEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        logitsEncoder.setComputePipelineState(logitsPipeline)
        logitsEncoder.setBuffer(resources.qBinding.buffer, offset: 0, index: 0)
        logitsEncoder.setBuffer(resources.kBinding.buffer, offset: 0, index: 1)
        logitsEncoder.setBuffer(resources.maskBinding.buffer, offset: 0, index: 2)
        logitsEncoder.setBuffer(resources.scoresBuffer, offset: 0, index: 3)
        withUnsafeBytes(of: params) { rawBytes in
            logitsEncoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 4)
        }
        logitsEncoder.dispatchThreads(
            MTLSize(width: shape.seqLen, height: shape.heads, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: max(1, min(logitsPipeline.threadExecutionWidth, shape.seqLen)),
                height: 1,
                depth: 1
            )
        )
        logitsEncoder.endEncoding()

        guard let softmaxEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        softmaxEncoder.setComputePipelineState(softmaxPipeline)
        softmaxEncoder.setBuffer(resources.scoresBuffer, offset: 0, index: 0)
        softmaxEncoder.setBuffer(resources.weightsBuffer, offset: 0, index: 1)
        withUnsafeBytes(of: params) { rawBytes in
            softmaxEncoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 2)
        }
        softmaxEncoder.dispatchThreads(
            MTLSize(width: shape.heads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: max(1, min(softmaxPipeline.maxTotalThreadsPerThreadgroup, shape.heads)),
                height: 1,
                depth: 1
            )
        )
        softmaxEncoder.endEncoding()

        guard let outputEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        outputEncoder.setComputePipelineState(outputPipeline)
        outputEncoder.setBuffer(resources.weightsBuffer, offset: 0, index: 0)
        outputEncoder.setBuffer(resources.vBinding.buffer, offset: 0, index: 1)
        outputEncoder.setBuffer(resources.outputBinding.buffer, offset: 0, index: 2)
        withUnsafeBytes(of: params) { rawBytes in
            outputEncoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 3)
        }
        outputEncoder.dispatchThreads(
            MTLSize(width: shape.headDim, height: shape.heads, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: max(1, min(outputPipeline.threadExecutionWidth, shape.headDim)),
                height: 1,
                depth: 1
            )
        )
        outputEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if commandBuffer.status != .completed {
            throw .commandExecutionFailed(commandBuffer.error?.localizedDescription ?? "status=\(commandBuffer.status.rawValue)")
        }
    }

    private func encodeDecodeAndWait(
        qBinding: SurfaceBinding,
        kBinding: SurfaceBinding,
        vBinding: SurfaceBinding,
        residualBinding: SurfaceBinding,
        outputBinding: SurfaceBinding,
        scratch: DecodeScratch,
        projectionWeightsBuffer: MTLBuffer,
        projectionBiasBuffer: MTLBuffer,
        shape: MetalDecodeAttentionShape
    ) throws(MetalAttentionError) {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw .commandBufferUnavailable
        }

        let decodeParams = DecodeParams(
            heads: UInt32(shape.heads),
            headDim: UInt32(shape.headDim),
            visibleTokens: UInt32(shape.visibleTokens),
            cacheStride: UInt32(shape.cacheStride),
            laneStride: UInt32(shape.laneStride),
            kvHeads: UInt32(shape.kvHeads),
            tokenBase: UInt32(shape.tokenBase),
            scale: 1.0 / sqrt(Float(shape.headDim))
        )
        let softmaxParams = AttentionParams(
            heads: UInt32(shape.heads),
            headDim: UInt32(shape.headDim),
            seqLen: UInt32(shape.visibleTokens),
            scale: 1.0 / sqrt(Float(shape.headDim))
        )

        guard let logitsEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        logitsEncoder.setComputePipelineState(decodeLogitsPipeline)
        logitsEncoder.setBuffer(qBinding.buffer, offset: 0, index: 0)
        logitsEncoder.setBuffer(kBinding.buffer, offset: 0, index: 1)
        logitsEncoder.setBuffer(scratch.scoresBuffer, offset: 0, index: 2)
        withUnsafeBytes(of: decodeParams) { rawBytes in
            logitsEncoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 3)
        }
        logitsEncoder.dispatchThreads(
            MTLSize(width: shape.visibleTokens, height: shape.heads, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: max(1, min(decodeLogitsPipeline.threadExecutionWidth, shape.visibleTokens)),
                height: 1,
                depth: 1
            )
        )
        logitsEncoder.endEncoding()

        guard let softmaxEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        softmaxEncoder.setComputePipelineState(softmaxPipeline)
        softmaxEncoder.setBuffer(scratch.scoresBuffer, offset: 0, index: 0)
        softmaxEncoder.setBuffer(scratch.weightsBuffer, offset: 0, index: 1)
        withUnsafeBytes(of: softmaxParams) { rawBytes in
            softmaxEncoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 2)
        }
        softmaxEncoder.dispatchThreads(
            MTLSize(width: shape.heads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: max(1, min(softmaxPipeline.maxTotalThreadsPerThreadgroup, shape.heads)),
                height: 1,
                depth: 1
            )
        )
        softmaxEncoder.endEncoding()

        guard let outputEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        outputEncoder.setComputePipelineState(decodeOutputPipeline)
        outputEncoder.setBuffer(scratch.weightsBuffer, offset: 0, index: 0)
        outputEncoder.setBuffer(vBinding.buffer, offset: 0, index: 1)
        outputEncoder.setBuffer(scratch.contextBuffer, offset: 0, index: 2)
        withUnsafeBytes(of: decodeParams) { rawBytes in
            outputEncoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 3)
        }
        outputEncoder.dispatchThreads(
            MTLSize(width: shape.headDim, height: shape.heads, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: max(1, min(decodeOutputPipeline.threadExecutionWidth, shape.headDim)),
                height: 1,
                depth: 1
            )
        )
        outputEncoder.endEncoding()

        guard let projectionEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        projectionEncoder.setComputePipelineState(decodeProjectionPipeline)
        projectionEncoder.setBuffer(scratch.contextBuffer, offset: 0, index: 0)
        projectionEncoder.setBuffer(projectionWeightsBuffer, offset: 0, index: 1)
        projectionEncoder.setBuffer(projectionBiasBuffer, offset: 0, index: 2)
        projectionEncoder.setBuffer(residualBinding.buffer, offset: 0, index: 3)
        projectionEncoder.setBuffer(outputBinding.buffer, offset: 0, index: 4)
        withUnsafeBytes(of: decodeParams) { rawBytes in
            projectionEncoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 5)
        }
        projectionEncoder.dispatchThreads(
            MTLSize(width: shape.heads * shape.headDim, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: max(1, min(decodeProjectionPipeline.threadExecutionWidth, shape.heads * shape.headDim)),
                height: 1,
                depth: 1
            )
        )
        projectionEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if commandBuffer.status != .completed {
            throw .commandExecutionFailed(commandBuffer.error?.localizedDescription ?? "status=\(commandBuffer.status.rawValue)")
        }
    }

    /// Batched decode attention: encodes ALL layers' attention into ONE command buffer.
    /// Eliminates per-layer command buffer creation/commit/sync overhead.
    struct LayerAttentionBindings {
        fileprivate let qBinding: SurfaceBinding
        fileprivate let kBinding: SurfaceBinding
        fileprivate let vBinding: SurfaceBinding
        let contextBuffer: MTLBuffer
    }

    struct BatchedDecodeScratch {
        let scoresBuffer: MTLBuffer
        let weightsBuffer: MTLBuffer

        init(device: MTLDevice, nLayers: Int, heads: Int, visibleTokens: Int) throws(MetalAttentionError) {
            let perLayerElements = heads * visibleTokens
            let totalElements = perLayerElements * nLayers
            let byteCount = totalElements * MemoryLayout<Float>.stride
            guard let scoresBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared),
                  let weightsBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
                throw .temporaryBufferAllocationFailed
            }
            self.scoresBuffer = scoresBuffer
            self.weightsBuffer = weightsBuffer
        }
    }

    func encodeBatchedDecodeContext(
        layers: [LayerAttentionBindings],
        shape: MetalDecodeAttentionShape,
        scratch: BatchedDecodeScratch
    ) throws(MetalAttentionError) {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw .commandBufferUnavailable
        }

        let decodeParams = DecodeParams(
            heads: UInt32(shape.heads),
            headDim: UInt32(shape.headDim),
            visibleTokens: UInt32(shape.visibleTokens),
            cacheStride: UInt32(shape.cacheStride),
            laneStride: UInt32(shape.laneStride),
            kvHeads: UInt32(shape.kvHeads),
            tokenBase: UInt32(shape.tokenBase),
            scale: 1.0 / sqrt(Float(shape.headDim))
        )
        let softmaxParams = AttentionParams(
            heads: UInt32(shape.heads),
            headDim: UInt32(shape.headDim),
            seqLen: UInt32(shape.visibleTokens),
            scale: 1.0 / sqrt(Float(shape.headDim))
        )

        let perLayerElements = shape.heads * shape.visibleTokens

        // Encode all layers' logits + softmax + output into single command buffer
        for (layerIdx, layer) in layers.enumerated() {
            // Retain bindings until command buffer completes
            Self.retain(bindings: [layer.qBinding, layer.kBinding, layer.vBinding], until: commandBuffer)

            let scoresOffset = layerIdx * perLayerElements * MemoryLayout<Float>.stride

            // Logits: Q × K^T → scores
            guard let logitsEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw .commandEncoderUnavailable
            }
            logitsEncoder.setComputePipelineState(decodeLogitsPipeline)
            logitsEncoder.setBuffer(layer.qBinding.buffer, offset: 0, index: 0)
            logitsEncoder.setBuffer(layer.kBinding.buffer, offset: 0, index: 1)
            logitsEncoder.setBuffer(scratch.scoresBuffer, offset: scoresOffset, index: 2)
            withUnsafeBytes(of: decodeParams) { rawBytes in
                logitsEncoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 3)
            }
            logitsEncoder.dispatchThreads(
                MTLSize(width: shape.visibleTokens, height: shape.heads, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: max(1, min(decodeLogitsPipeline.threadExecutionWidth, shape.visibleTokens)),
                    height: 1, depth: 1
                )
            )
            logitsEncoder.endEncoding()

            // Softmax: scores → weights
            guard let softmaxEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw .commandEncoderUnavailable
            }
            softmaxEncoder.setComputePipelineState(softmaxPipeline)
            softmaxEncoder.setBuffer(scratch.scoresBuffer, offset: scoresOffset, index: 0)
            softmaxEncoder.setBuffer(scratch.weightsBuffer, offset: scoresOffset, index: 1)
            withUnsafeBytes(of: softmaxParams) { rawBytes in
                softmaxEncoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 2)
            }
            softmaxEncoder.dispatchThreads(
                MTLSize(width: shape.heads, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: max(1, min(softmaxPipeline.maxTotalThreadsPerThreadgroup, shape.heads)),
                    height: 1, depth: 1
                )
            )
            softmaxEncoder.endEncoding()

            // Output: weights × V → context
            guard let outputEncoder = commandBuffer.makeComputeCommandEncoder() else {
                throw .commandEncoderUnavailable
            }
            outputEncoder.setComputePipelineState(decodeOutputStridedPipeline)
            outputEncoder.setBuffer(scratch.weightsBuffer, offset: scoresOffset, index: 0)
            outputEncoder.setBuffer(layer.vBinding.buffer, offset: 0, index: 1)
            outputEncoder.setBuffer(layer.contextBuffer, offset: 0, index: 2)
            withUnsafeBytes(of: decodeParams) { rawBytes in
                outputEncoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 3)
            }
            outputEncoder.dispatchThreads(
                MTLSize(width: shape.headDim, height: shape.heads, depth: 1),
                threadsPerThreadgroup: MTLSize(
                    width: max(1, min(outputPipeline.threadExecutionWidth, shape.headDim)),
                    height: 1, depth: 1
                )
            )
            outputEncoder.endEncoding()
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if commandBuffer.status != .completed {
            throw .commandExecutionFailed(commandBuffer.error?.localizedDescription ?? "status=\(commandBuffer.status.rawValue)")
        }
    }

    private func encodeDecodeContextAndWait(
        qBinding: SurfaceBinding,
        kBinding: SurfaceBinding,
        vBinding: SurfaceBinding,
        contextBuffer: MTLBuffer,
        outputPipeline: MTLComputePipelineState,
        scratch: DecodeScratch,
        shape: MetalDecodeAttentionShape
    ) throws(MetalAttentionError) {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw .commandBufferUnavailable
        }

        let decodeParams = DecodeParams(
            heads: UInt32(shape.heads),
            headDim: UInt32(shape.headDim),
            visibleTokens: UInt32(shape.visibleTokens),
            cacheStride: UInt32(shape.cacheStride),
            laneStride: UInt32(shape.laneStride),
            kvHeads: UInt32(shape.kvHeads),
            tokenBase: UInt32(shape.tokenBase),
            scale: 1.0 / sqrt(Float(shape.headDim))
        )
        let softmaxParams = AttentionParams(
            heads: UInt32(shape.heads),
            headDim: UInt32(shape.headDim),
            seqLen: UInt32(shape.visibleTokens),
            scale: 1.0 / sqrt(Float(shape.headDim))
        )

        guard let logitsEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        logitsEncoder.setComputePipelineState(decodeLogitsPipeline)
        logitsEncoder.setBuffer(qBinding.buffer, offset: 0, index: 0)
        logitsEncoder.setBuffer(kBinding.buffer, offset: 0, index: 1)
        logitsEncoder.setBuffer(scratch.scoresBuffer, offset: 0, index: 2)
        withUnsafeBytes(of: decodeParams) { rawBytes in
            logitsEncoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 3)
        }
        logitsEncoder.dispatchThreads(
            MTLSize(width: shape.visibleTokens, height: shape.heads, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: max(1, min(decodeLogitsPipeline.threadExecutionWidth, shape.visibleTokens)),
                height: 1,
                depth: 1
            )
        )
        logitsEncoder.endEncoding()

        guard let softmaxEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        softmaxEncoder.setComputePipelineState(softmaxPipeline)
        softmaxEncoder.setBuffer(scratch.scoresBuffer, offset: 0, index: 0)
        softmaxEncoder.setBuffer(scratch.weightsBuffer, offset: 0, index: 1)
        withUnsafeBytes(of: softmaxParams) { rawBytes in
            softmaxEncoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 2)
        }
        softmaxEncoder.dispatchThreads(
            MTLSize(width: shape.heads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: max(1, min(softmaxPipeline.maxTotalThreadsPerThreadgroup, shape.heads)),
                height: 1,
                depth: 1
            )
        )
        softmaxEncoder.endEncoding()

        guard let outputEncoder = commandBuffer.makeComputeCommandEncoder() else {
            throw .commandEncoderUnavailable
        }
        outputEncoder.setComputePipelineState(outputPipeline)
        outputEncoder.setBuffer(scratch.weightsBuffer, offset: 0, index: 0)
        outputEncoder.setBuffer(vBinding.buffer, offset: 0, index: 1)
        outputEncoder.setBuffer(contextBuffer, offset: 0, index: 2)
        withUnsafeBytes(of: decodeParams) { rawBytes in
            outputEncoder.setBytes(rawBytes.baseAddress!, length: rawBytes.count, index: 3)
        }
        outputEncoder.dispatchThreads(
            MTLSize(width: shape.headDim, height: shape.heads, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: max(1, min(outputPipeline.threadExecutionWidth, shape.headDim)),
                height: 1,
                depth: 1
            )
        )
        outputEncoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if commandBuffer.status != .completed {
            throw .commandExecutionFailed(commandBuffer.error?.localizedDescription ?? "status=\(commandBuffer.status.rawValue)")
        }
    }

    private static func makeInput(count: Int, seed: UInt64) -> [Float] {
        var values = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let mixed = UInt64(i) &* 6364136223846793005 &+ 1442695040888963407 &+ seed
            let unit = Float(mixed & 0xffff) / Float(0xffff)
            values[i] = unit * 2 - 1
        }
        return values
    }

    private static func makeMask(count: Int) -> [Float] {
        [Float](repeating: 0, count: count)
    }

    private static func median(_ values: [Double]) -> Double {
        precondition(!values.isEmpty)
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[mid - 1] + sorted[mid]) * 0.5
        }
        return sorted[mid]
    }

    private static func elapsedMilliseconds(start: UInt64, end: UInt64) -> Double {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        let elapsed = end &- start
        let nanos = elapsed &* UInt64(info.numer) / UInt64(info.denom)
        return Double(nanos) / 1_000_000.0
    }

    private static var shaderSource: String {
        let decodeKVHeadExpression = Self.decodeKVHeadExpression
        return """
    #include <metal_stdlib>
    using namespace metal;

    struct AttentionParams {
        uint heads;
        uint headDim;
        uint seqLen;
        uint pad0;
        float scale;
        float pad1;
        float pad2;
        float pad3;
    };

    struct DecodeParams {
        uint heads;
        uint headDim;
        uint visibleTokens;
        uint cacheStride;
        uint laneStride;
        uint kvHeads;
        uint tokenBase;
        uint pad0;
        float scale;
        float pad1;
    };

    kernel void attention_logits(
        const device half *q [[buffer(0)]],
        const device half *k [[buffer(1)]],
        const device half *mask [[buffer(2)]],
        device float *scores [[buffer(3)]],
        constant AttentionParams &params [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint token = gid.x;
        uint head = gid.y;
        if (head >= params.heads || token >= params.seqLen) {
            return;
        }

        uint qBase = head * params.headDim;
        uint kBase = (head * params.seqLen + token) * params.headDim;
        float dot = 0.0f;
        for (uint dim = 0; dim < params.headDim; ++dim) {
            dot += float(q[qBase + dim]) * float(k[kBase + dim]);
        }

        uint maskIndex = head * params.seqLen + token;
        scores[maskIndex] = dot * params.scale + float(mask[maskIndex]);
    }

    kernel void attention_softmax(
        const device float *scores [[buffer(0)]],
        device float *weights [[buffer(1)]],
        constant AttentionParams &params [[buffer(2)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint head = gid;
        if (head >= params.heads) {
            return;
        }

        uint base = head * params.seqLen;
        float maxScore = -INFINITY;
        for (uint token = 0; token < params.seqLen; ++token) {
            maxScore = max(maxScore, scores[base + token]);
        }

        float denom = 0.0f;
        for (uint token = 0; token < params.seqLen; ++token) {
            float value = exp(scores[base + token] - maxScore);
            weights[base + token] = value;
            denom += value;
        }

        float invDenom = denom > 0.0f ? (1.0f / denom) : 0.0f;
        for (uint token = 0; token < params.seqLen; ++token) {
            weights[base + token] *= invDenom;
        }
    }

    kernel void attention_output(
        const device float *weights [[buffer(0)]],
        const device half *v [[buffer(1)]],
        device half *output [[buffer(2)]],
        constant AttentionParams &params [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint dim = gid.x;
        uint head = gid.y;
        if (head >= params.heads || dim >= params.headDim) {
            return;
        }

        float accum = 0.0f;
        uint weightBase = head * params.seqLen;
        for (uint token = 0; token < params.seqLen; ++token) {
            uint vBase = (head * params.seqLen + token) * params.headDim;
            accum += weights[weightBase + token] * float(v[vBase + dim]);
        }

        output[head * params.headDim + dim] = half(accum);
    }

    kernel void decode_attention_logits(
        const device half *q [[buffer(0)]],
        const device half *kCache [[buffer(1)]],
        device float *scores [[buffer(2)]],
        constant DecodeParams &params [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint token = gid.x;
        uint head = gid.y;
        if (head >= params.heads || token >= params.visibleTokens) {
            return;
        }

        uint kvHead = \(decodeKVHeadExpression);
        float dot = 0.0f;
        uint qHeadOffset = head * params.headDim;
        uint kHeadOffset = kvHead * params.headDim;
        for (uint dim = 0; dim < params.headDim; ++dim) {
            uint cacheToken = params.tokenBase + token;
            uint qIndex = (qHeadOffset + dim) * params.laneStride;
            uint kIndex = (kHeadOffset + dim) * params.cacheStride + cacheToken;
            dot += float(q[qIndex]) * float(kCache[kIndex]);
        }

        scores[head * params.visibleTokens + token] = dot * params.scale;
    }

    kernel void decode_attention_output(
        const device float *weights [[buffer(0)]],
        const device half *vCache [[buffer(1)]],
        device float *context [[buffer(2)]],
        constant DecodeParams &params [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint dim = gid.x;
        uint head = gid.y;
        if (head >= params.heads || dim >= params.headDim) {
            return;
        }

        uint kvHead = \(decodeKVHeadExpression);
        uint vChannel = kvHead * params.headDim + dim;
        float accum = 0.0f;
        uint weightBase = head * params.visibleTokens;
        uint valueBase = vChannel * params.cacheStride;
        for (uint token = 0; token < params.visibleTokens; ++token) {
            uint cacheToken = params.tokenBase + token;
            accum += weights[weightBase + token] * float(vCache[valueBase + cacheToken]);
        }
        uint outChannel = head * params.headDim + dim;
        context[outChannel] = accum;
    }

    kernel void decode_attention_output_strided(
        const device float *weights [[buffer(0)]],
        const device half *vCache [[buffer(1)]],
        device float *context [[buffer(2)]],
        constant DecodeParams &params [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint dim = gid.x;
        uint head = gid.y;
        if (head >= params.heads || dim >= params.headDim) {
            return;
        }

        uint kvHead = \(decodeKVHeadExpression);
        uint vChannel = kvHead * params.headDim + dim;
        float accum = 0.0f;
        uint weightBase = head * params.visibleTokens;
        uint valueBase = vChannel * params.cacheStride;
        for (uint token = 0; token < params.visibleTokens; ++token) {
            uint cacheToken = params.tokenBase + token;
            accum += weights[weightBase + token] * float(vCache[valueBase + cacheToken]);
        }
        uint outChannel = head * params.headDim + dim;
        context[outChannel * params.laneStride] = accum;
    }

    kernel void decode_output_projection(
        const device float *context [[buffer(0)]],
        const device float *projection [[buffer(1)]],
        const device float *bias [[buffer(2)]],
        const device half *residual [[buffer(3)]],
        device half *output [[buffer(4)]],
        constant DecodeParams &params [[buffer(5)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint dim = gid;
        uint totalDim = params.heads * params.headDim;
        if (dim >= totalDim) {
            return;
        }

        float accum = 0.0f;
        uint rowBase = dim * totalDim;
        for (uint col = 0; col < totalDim; ++col) {
            accum += projection[rowBase + col] * context[col];
        }

        uint laneIndex = dim * params.laneStride;
        output[laneIndex] = half(accum + bias[dim] + float(residual[laneIndex]));
    }

    // RoPE rotation applied to Q and K surfaces in-place.
    // Llama-family RoPE uses half-split pairs: (0, halfDim), (1, halfDim+1), ...
    // Grid: (headDim/2, max(nHeads, nKVHeads), 1)
    struct RoPEParams {
        uint nHeads;
        uint nKVHeads;
        uint headDim;
        uint position;
        uint laneStride;
        float theta;
        uint pad0;
        uint pad1;
    };

    kernel void rope_decode(
        device half *q [[buffer(0)]],
        device half *k [[buffer(1)]],
        constant RoPEParams &params [[buffer(2)]],
        uint2 gid [[thread_position_in_grid]]
    ) {
        uint dimPair = gid.x;   // 0..<headDim/2
        uint headIdx = gid.y;   // 0..<max(nHeads, nKVHeads)
        uint halfDim = params.headDim / 2;
        if (dimPair >= halfDim) return;

        float freqBase = pow(params.theta, -float(dimPair) / float(halfDim));
        float angle = float(params.position) * freqBase;
        float cosA = cos(angle);
        float sinA = sin(angle);

        uint d0 = dimPair;
        uint d1 = dimPair + halfDim;

        // Apply to Q if headIdx < nHeads
        if (headIdx < params.nHeads) {
            uint qIdx0 = (headIdx * params.headDim + d0) * params.laneStride;
            uint qIdx1 = (headIdx * params.headDim + d1) * params.laneStride;
            float q0 = float(q[qIdx0]);
            float q1 = float(q[qIdx1]);
            q[qIdx0] = half(q0 * cosA - q1 * sinA);
            q[qIdx1] = half(q0 * sinA + q1 * cosA);
        }

        // Apply to K if headIdx < nKVHeads
        if (headIdx < params.nKVHeads) {
            uint kIdx0 = (headIdx * params.headDim + d0) * params.laneStride;
            uint kIdx1 = (headIdx * params.headDim + d1) * params.laneStride;
            float k0 = float(k[kIdx0]);
            float k1 = float(k[kIdx1]);
            k[kIdx0] = half(k0 * cosA - k1 * sinA);
            k[kIdx1] = half(k0 * sinA + k1 * cosA);
        }
    }

    // KV cache scatter: copy K and V outputs at lane 0 into their cache at tokenIndex.
    // Grid: (kvDim, 1, 1) — one thread per channel.
    struct KVScatterParams {
        uint kvDim;
        uint tokenIndex;
        uint cacheStride;
        uint laneStride;
    };

    kernel void kv_cache_scatter(
        const device half *kSrc [[buffer(0)]],
        const device half *vSrc [[buffer(1)]],
        device half *kCache [[buffer(2)]],
        device half *vCache [[buffer(3)]],
        constant KVScatterParams &params [[buffer(4)]],
        uint gid [[thread_position_in_grid]]
    ) {
        uint ch = gid;
        if (ch >= params.kvDim) return;
        uint srcIdx = ch * params.laneStride;  // lane 0 of source surface
        uint dstIdx = ch * params.cacheStride + params.tokenIndex;
        kCache[dstIdx] = kSrc[srcIdx];
        vCache[dstIdx] = vSrc[srcIdx];
    }

    // Fused decode SDPA: Q*K^T, softmax, weighted V sum in one dispatch per head.
    // One threadgroup per head. Uses threadgroup memory for the scores vector.
    // Grid: (heads, 1, 1), Threadgroup: (min(visibleTokens, 256), 1, 1)
    kernel void fused_decode_sdpa(
        const device half *q [[buffer(0)]],
        const device half *kCache [[buffer(1)]],
        const device half *vCache [[buffer(2)]],
        device float *context [[buffer(3)]],
        constant DecodeParams &params [[buffer(4)]],
        uint head [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint tgSize [[threads_per_threadgroup]],
        threadgroup float *scores [[threadgroup_buffer(0)]]
    ) {
        if (head >= params.heads) return;

        uint kvHead = \(decodeKVHeadExpression);
        uint qHeadOffset = head * params.headDim;
        uint kHeadOffset = kvHead * params.headDim;

        // Step 1: Compute Q*K^T scores (cooperative across threads in threadgroup)
        for (uint token = tid; token < params.visibleTokens; token += tgSize) {
            float dot = 0.0f;
            uint cacheToken = params.tokenBase + token;
            for (uint d = 0; d < params.headDim; ++d) {
                uint qIndex = (qHeadOffset + d) * params.laneStride;
                uint kIndex = (kHeadOffset + d) * params.cacheStride + cacheToken;
                dot += float(q[qIndex]) * float(kCache[kIndex]);
            }
            scores[token] = dot * params.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 2: Softmax (single-thread reduction for correctness, visibleTokens is small for decode)
        if (tid == 0) {
            float maxScore = -INFINITY;
            for (uint t = 0; t < params.visibleTokens; ++t) {
                maxScore = max(maxScore, scores[t]);
            }
            float denom = 0.0f;
            for (uint t = 0; t < params.visibleTokens; ++t) {
                float val = exp(scores[t] - maxScore);
                scores[t] = val;
                denom += val;
            }
            float invDenom = denom > 0.0f ? (1.0f / denom) : 0.0f;
            for (uint t = 0; t < params.visibleTokens; ++t) {
                scores[t] *= invDenom;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 3: Weighted V sum (cooperative across headDim)
        uint vHeadOffset = kvHead * params.headDim;
        for (uint d = tid; d < params.headDim; d += tgSize) {
            uint vChannel = vHeadOffset + d;
            uint valueBase = vChannel * params.cacheStride;
            float accum = 0.0f;
            for (uint t = 0; t < params.visibleTokens; ++t) {
                uint cacheToken = params.tokenBase + t;
                accum += scores[t] * float(vCache[valueBase + cacheToken]);
            }
            uint outChannel = head * params.headDim + d;
            context[outChannel * params.laneStride] = accum;
        }
    }
    """
    }
}

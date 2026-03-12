import Foundation
import IOSurface
import ANEInterop
import ANERuntime
import ANETypes

public struct RWKVStyleRecurrentBenchmarkSample: Sendable {
    public let effectiveContext: Int
    public let medianLatencyMs: Double
    public let tokensPerSecond: Double
}

public struct RWKVStyleRecurrentScalingReport: Sendable {
    public let recurrentSamples: [RWKVStyleRecurrentBenchmarkSample]
    public let transformerSamples: [RWKVStyleRecurrentBenchmarkSample]
    public let skippedTransformerContexts: [Int]
}

@inline(__always)
func makeRWKVZeroLaneSurface(laneSpatial: Int) throws(ANEError) -> IOSurfaceRef {
    guard let zeroLane = ane_interop_create_surface(ModelConfig.dim * laneSpatial * 2) else {
        throw .surfaceAllocationFailed
    }

    let zeroValues = Array(repeating: Float(0), count: ModelConfig.dim * laneSpatial)
    zeroValues.withUnsafeBufferPointer { src in
        SurfaceIO.writeFP16(to: zeroLane, data: src, channels: ModelConfig.dim, spatial: laneSpatial)
    }
    return zeroLane
}

public struct RWKVStyleRecurrentSurfaceHandles {
    public let xIn: IOSurfaceRef
    public let stateIn: IOSurfaceRef
    public let xOut: IOSurfaceRef
    public let stateOut: IOSurfaceRef
    public let zeroLane: IOSurfaceRef
    public let laneSpatial: Int

    public init(kernels: borrowing RWKVStyleRecurrentKernelSet) throws(ANEError) {
        self.xIn = try kernels.step.inputSurface(at: 0)
        self.stateIn = try kernels.step.inputSurface(at: 1)
        self.xOut = try kernels.step.outputSurface(at: 0)
        self.stateOut = try kernels.step.outputSurface(at: 1)
        self.laneSpatial = kernels.laneSpatial
        self.zeroLane = try makeRWKVZeroLaneSurface(laneSpatial: kernels.laneSpatial)
    }
}

public struct RWKVStyleRecurrentSession: ~Copyable {
    public let kernels: RWKVStyleRecurrentKernelSet
    public let handles: RWKVStyleRecurrentSurfaceHandles
    public private(set) var stepCount: Int

    public init(
        weights: borrowing RWKVStyleRecurrentWeights,
        laneSpatial: Int = RWKVStyleRecurrentKernelSet.defaultLaneSpatial
    ) throws(ANEError) {
        let kernels = try RWKVStyleRecurrentKernelSet(weights: weights, laneSpatial: laneSpatial)
        let handles = try RWKVStyleRecurrentSurfaceHandles(kernels: kernels)
        self.kernels = kernels
        self.handles = handles
        self.stepCount = 0
    }

    public mutating func reset() throws(ANEError) {
        do {
            try SurfaceIO.copyFP16(
                dst: handles.xIn,
                dstChannelOffset: 0,
                src: handles.zeroLane,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
            try SurfaceIO.copyFP16(
                dst: handles.stateIn,
                dstChannelOffset: 0,
                src: handles.zeroLane,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
        } catch {
            throw .invalidArguments("recurrent zero reset failed: \(error)")
        }
        self.stepCount = 0
    }

    public mutating func step(
        tokenInput: borrowing TensorBuffer,
        output: borrowing TensorBuffer,
        timings: inout StepTimingBreakdown
    ) throws(ANEError) {
        precondition(tokenInput.count == ModelConfig.dim)
        precondition(output.count == ModelConfig.dim)

        var t0 = RuntimeClock.now()
        do {
            try SurfaceIO.copyFP16(
                dst: handles.xIn,
                dstChannelOffset: 0,
                src: handles.zeroLane,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
            try tokenInput.withUnsafeBufferPointer { tokenBuf in
                try SurfaceIO.writeFP16SpatialSlice(
                    to: handles.xIn,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: handles.laneSpatial,
                    data: tokenBuf,
                    channels: ModelConfig.dim
                )
            }
        } catch {
            throw .invalidArguments("recurrent input write failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

        t0 = RuntimeClock.now()
        do {
            try kernels.step.eval()
        } catch {
            throw .invalidArguments("recurrent step eval failed at step \(stepCount): \(error)")
        }
        timings.tAne += RuntimeClock.ms(RuntimeClock.now() - t0)

        t0 = RuntimeClock.now()
        do {
            try SurfaceIO.copyFP16(
                dst: handles.stateIn,
                dstChannelOffset: 0,
                src: handles.stateOut,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
            try output.withUnsafeMutableBufferPointer { outBuf in
                try SurfaceIO.readFP16SpatialSlice(
                    from: handles.xOut,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: handles.laneSpatial,
                    into: outBuf,
                    channels: ModelConfig.dim
                )
            }
        } catch {
            throw .invalidArguments("recurrent output readback failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)
        self.stepCount += 1
    }
}

public struct RWKVStyleStateFreeSurfaceHandles {
    public let xIn: IOSurfaceRef
    public let xOut: IOSurfaceRef
    public let zeroLane: IOSurfaceRef
    public let laneSpatial: Int

    public init(kernels: borrowing RWKVStyleStateFreeKernelSet) throws(ANEError) {
        self.xIn = try kernels.step.inputSurface(at: 0)
        self.xOut = try kernels.step.outputSurface(at: 0)
        self.laneSpatial = kernels.laneSpatial
        self.zeroLane = try makeRWKVZeroLaneSurface(laneSpatial: kernels.laneSpatial)
    }
}

public struct RWKVStyleStateFreeSession: ~Copyable {
    public let kernels: RWKVStyleStateFreeKernelSet
    public let handles: RWKVStyleStateFreeSurfaceHandles
    public private(set) var stepCount: Int

    public init(
        weights: borrowing RWKVStyleRecurrentWeights,
        laneSpatial: Int = RWKVStyleStateFreeKernelSet.defaultLaneSpatial
    ) throws(ANEError) {
        let kernels = try RWKVStyleStateFreeKernelSet(weights: weights, laneSpatial: laneSpatial)
        let handles = try RWKVStyleStateFreeSurfaceHandles(kernels: kernels)
        self.kernels = kernels
        self.handles = handles
        self.stepCount = 0
    }

    public mutating func reset() throws(ANEError) {
        do {
            try SurfaceIO.copyFP16(
                dst: handles.xIn,
                dstChannelOffset: 0,
                src: handles.zeroLane,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
        } catch {
            throw .invalidArguments("state-free recurrent zero reset failed: \(error)")
        }
        self.stepCount = 0
    }

    public mutating func step(
        tokenInput: borrowing TensorBuffer,
        output: borrowing TensorBuffer,
        timings: inout StepTimingBreakdown
    ) throws(ANEError) {
        precondition(tokenInput.count == ModelConfig.dim)
        precondition(output.count == ModelConfig.dim)

        var t0 = RuntimeClock.now()
        do {
            try SurfaceIO.copyFP16(
                dst: handles.xIn,
                dstChannelOffset: 0,
                src: handles.zeroLane,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
            try tokenInput.withUnsafeBufferPointer { tokenBuf in
                try SurfaceIO.writeFP16SpatialSlice(
                    to: handles.xIn,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: handles.laneSpatial,
                    data: tokenBuf,
                    channels: ModelConfig.dim
                )
            }
        } catch {
            throw .invalidArguments("state-free recurrent input write failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

        t0 = RuntimeClock.now()
        do {
            try kernels.step.eval()
        } catch {
            throw .invalidArguments("state-free recurrent step eval failed at step \(stepCount): \(error)")
        }
        timings.tAne += RuntimeClock.ms(RuntimeClock.now() - t0)

        t0 = RuntimeClock.now()
        do {
            try output.withUnsafeMutableBufferPointer { outBuf in
                try SurfaceIO.readFP16SpatialSlice(
                    from: handles.xOut,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: handles.laneSpatial,
                    into: outBuf,
                    channels: ModelConfig.dim
                )
            }
        } catch {
            throw .invalidArguments("state-free recurrent output readback failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)
        self.stepCount += 1
    }
}

public struct RWKVStyleFusedTwoLayerSurfaceHandles {
    public let xIn: IOSurfaceRef
    public let stateIn0: IOSurfaceRef
    public let stateIn1: IOSurfaceRef
    public let xOut: IOSurfaceRef
    public let stateOut0: IOSurfaceRef
    public let stateOut1: IOSurfaceRef
    public let zeroLane: IOSurfaceRef
    public let laneSpatial: Int

    public init(kernels: borrowing RWKVStyleFusedTwoLayerKernelSet) throws(ANEError) {
        self.xIn = try kernels.step.inputSurface(at: 0)
        self.stateIn0 = try kernels.step.inputSurface(at: 1)
        self.stateIn1 = try kernels.step.inputSurface(at: 2)
        self.xOut = try kernels.step.outputSurface(at: 0)
        self.stateOut0 = try kernels.step.outputSurface(at: 1)
        self.stateOut1 = try kernels.step.outputSurface(at: 2)
        self.laneSpatial = kernels.laneSpatial
        self.zeroLane = try makeRWKVZeroLaneSurface(laneSpatial: kernels.laneSpatial)
    }
}

public struct RWKVStyleFusedTwoLayerSession: ~Copyable {
    public let kernels: RWKVStyleFusedTwoLayerKernelSet
    public let handles: RWKVStyleFusedTwoLayerSurfaceHandles
    public private(set) var stepCount: Int

    public init(
        weights0: borrowing RWKVStyleRecurrentWeights,
        weights1: borrowing RWKVStyleRecurrentWeights,
        laneSpatial: Int = RWKVStyleFusedTwoLayerKernelSet.defaultLaneSpatial
    ) throws(ANEError) {
        let kernels = try RWKVStyleFusedTwoLayerKernelSet(
            weights0: weights0,
            weights1: weights1,
            laneSpatial: laneSpatial
        )
        let handles = try RWKVStyleFusedTwoLayerSurfaceHandles(kernels: kernels)
        self.kernels = kernels
        self.handles = handles
        self.stepCount = 0
    }

    public mutating func reset() throws(ANEError) {
        do {
            try SurfaceIO.copyFP16(
                dst: handles.xIn,
                dstChannelOffset: 0,
                src: handles.zeroLane,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
            try SurfaceIO.copyFP16(
                dst: handles.stateIn0,
                dstChannelOffset: 0,
                src: handles.zeroLane,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
            try SurfaceIO.copyFP16(
                dst: handles.stateIn1,
                dstChannelOffset: 0,
                src: handles.zeroLane,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
        } catch {
            throw .invalidArguments("fused recurrent zero reset failed: \(error)")
        }
        self.stepCount = 0
    }

    public mutating func step(
        tokenInput: borrowing TensorBuffer,
        output: borrowing TensorBuffer,
        timings: inout StepTimingBreakdown
    ) throws(ANEError) {
        precondition(tokenInput.count == ModelConfig.dim)
        precondition(output.count == ModelConfig.dim)

        var t0 = RuntimeClock.now()
        do {
            try SurfaceIO.copyFP16(
                dst: handles.xIn,
                dstChannelOffset: 0,
                src: handles.zeroLane,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
            try tokenInput.withUnsafeBufferPointer { tokenBuf in
                try SurfaceIO.writeFP16SpatialSlice(
                    to: handles.xIn,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: handles.laneSpatial,
                    data: tokenBuf,
                    channels: ModelConfig.dim
                )
            }
        } catch {
            throw .invalidArguments("fused recurrent input write failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

        t0 = RuntimeClock.now()
        do {
            try kernels.step.eval()
        } catch {
            throw .invalidArguments("fused recurrent step eval failed at step \(stepCount): \(error)")
        }
        timings.tAne += RuntimeClock.ms(RuntimeClock.now() - t0)

        t0 = RuntimeClock.now()
        do {
            try SurfaceIO.copyFP16(
                dst: handles.stateIn0,
                dstChannelOffset: 0,
                src: handles.stateOut0,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
            try SurfaceIO.copyFP16(
                dst: handles.stateIn1,
                dstChannelOffset: 0,
                src: handles.stateOut1,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
            try output.withUnsafeMutableBufferPointer { outBuf in
                try SurfaceIO.readFP16SpatialSlice(
                    from: handles.xOut,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: handles.laneSpatial,
                    into: outBuf,
                    channels: ModelConfig.dim
                )
            }
        } catch {
            throw .invalidArguments("fused recurrent output readback failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)
        self.stepCount += 1
    }
}

public struct RWKVStyleFusedThreeLayerSurfaceHandles {
    public let xIn: IOSurfaceRef
    public let stateIn0: IOSurfaceRef
    public let stateIn1: IOSurfaceRef
    public let stateIn2: IOSurfaceRef
    public let xOut: IOSurfaceRef
    public let stateOut0: IOSurfaceRef
    public let stateOut1: IOSurfaceRef
    public let stateOut2: IOSurfaceRef
    public let zeroLane: IOSurfaceRef
    public let laneSpatial: Int

    public init(kernels: borrowing RWKVStyleFusedThreeLayerKernelSet) throws(ANEError) {
        self.xIn = try kernels.step.inputSurface(at: 0)
        self.stateIn0 = try kernels.step.inputSurface(at: 1)
        self.stateIn1 = try kernels.step.inputSurface(at: 2)
        self.stateIn2 = try kernels.step.inputSurface(at: 3)
        self.xOut = try kernels.step.outputSurface(at: 0)
        self.stateOut0 = try kernels.step.outputSurface(at: 1)
        self.stateOut1 = try kernels.step.outputSurface(at: 2)
        self.stateOut2 = try kernels.step.outputSurface(at: 3)
        self.laneSpatial = kernels.laneSpatial
        self.zeroLane = try makeRWKVZeroLaneSurface(laneSpatial: kernels.laneSpatial)
    }
}

public struct RWKVStyleFusedThreeLayerSession: ~Copyable {
    public let kernels: RWKVStyleFusedThreeLayerKernelSet
    public let handles: RWKVStyleFusedThreeLayerSurfaceHandles
    public private(set) var stepCount: Int

    public init(
        weights0: borrowing RWKVStyleRecurrentWeights,
        weights1: borrowing RWKVStyleRecurrentWeights,
        weights2: borrowing RWKVStyleRecurrentWeights,
        laneSpatial: Int = RWKVStyleFusedThreeLayerKernelSet.defaultLaneSpatial
    ) throws(ANEError) {
        let kernels = try RWKVStyleFusedThreeLayerKernelSet(
            weights0: weights0,
            weights1: weights1,
            weights2: weights2,
            laneSpatial: laneSpatial
        )
        let handles = try RWKVStyleFusedThreeLayerSurfaceHandles(kernels: kernels)
        self.kernels = kernels
        self.handles = handles
        self.stepCount = 0
    }

    public mutating func reset() throws(ANEError) {
        do {
            try SurfaceIO.copyFP16(
                dst: handles.xIn,
                dstChannelOffset: 0,
                src: handles.zeroLane,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
            try SurfaceIO.copyFP16(
                dst: handles.stateIn0,
                dstChannelOffset: 0,
                src: handles.zeroLane,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
            try SurfaceIO.copyFP16(
                dst: handles.stateIn1,
                dstChannelOffset: 0,
                src: handles.zeroLane,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
            try SurfaceIO.copyFP16(
                dst: handles.stateIn2,
                dstChannelOffset: 0,
                src: handles.zeroLane,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
        } catch {
            throw .invalidArguments("fused three-layer recurrent zero reset failed: \(error)")
        }
        self.stepCount = 0
    }

    public mutating func step(
        tokenInput: borrowing TensorBuffer,
        output: borrowing TensorBuffer,
        timings: inout StepTimingBreakdown
    ) throws(ANEError) {
        precondition(tokenInput.count == ModelConfig.dim)
        precondition(output.count == ModelConfig.dim)

        var t0 = RuntimeClock.now()
        do {
            try SurfaceIO.copyFP16(
                dst: handles.xIn,
                dstChannelOffset: 0,
                src: handles.zeroLane,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
            try tokenInput.withUnsafeBufferPointer { tokenBuf in
                try SurfaceIO.writeFP16SpatialSlice(
                    to: handles.xIn,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: handles.laneSpatial,
                    data: tokenBuf,
                    channels: ModelConfig.dim
                )
            }
        } catch {
            throw .invalidArguments("fused three-layer recurrent input write failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

        t0 = RuntimeClock.now()
        do {
            try kernels.step.eval()
        } catch {
            throw .invalidArguments("fused three-layer recurrent step eval failed at step \(stepCount): \(error)")
        }
        timings.tAne += RuntimeClock.ms(RuntimeClock.now() - t0)

        t0 = RuntimeClock.now()
        do {
            try SurfaceIO.copyFP16(
                dst: handles.stateIn0,
                dstChannelOffset: 0,
                src: handles.stateOut0,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
            try SurfaceIO.copyFP16(
                dst: handles.stateIn1,
                dstChannelOffset: 0,
                src: handles.stateOut1,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
            try SurfaceIO.copyFP16(
                dst: handles.stateIn2,
                dstChannelOffset: 0,
                src: handles.stateOut2,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
            try output.withUnsafeMutableBufferPointer { outBuf in
                try SurfaceIO.readFP16SpatialSlice(
                    from: handles.xOut,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: handles.laneSpatial,
                    into: outBuf,
                    channels: ModelConfig.dim
                )
            }
        } catch {
            throw .invalidArguments("fused three-layer recurrent output readback failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)
        self.stepCount += 1
    }
}

public struct RWKVStyleFusedThreeLayerStateFreeSurfaceHandles {
    public let xIn: IOSurfaceRef
    public let xOut: IOSurfaceRef
    public let zeroLane: IOSurfaceRef
    public let laneSpatial: Int

    public init(kernels: borrowing RWKVStyleFusedThreeLayerStateFreeKernelSet) throws(ANEError) {
        self.xIn = try kernels.step.inputSurface(at: 0)
        self.xOut = try kernels.step.outputSurface(at: 0)
        self.laneSpatial = kernels.laneSpatial
        self.zeroLane = try makeRWKVZeroLaneSurface(laneSpatial: kernels.laneSpatial)
    }
}

public struct RWKVStyleFusedThreeLayerStateFreeSession: ~Copyable {
    public let kernels: RWKVStyleFusedThreeLayerStateFreeKernelSet
    public let handles: RWKVStyleFusedThreeLayerStateFreeSurfaceHandles
    public private(set) var stepCount: Int

    public init(
        weights0: borrowing RWKVStyleRecurrentWeights,
        weights1: borrowing RWKVStyleRecurrentWeights,
        weights2: borrowing RWKVStyleRecurrentWeights,
        laneSpatial: Int = RWKVStyleFusedThreeLayerStateFreeKernelSet.defaultLaneSpatial
    ) throws(ANEError) {
        let kernels = try RWKVStyleFusedThreeLayerStateFreeKernelSet(
            weights0: weights0,
            weights1: weights1,
            weights2: weights2,
            laneSpatial: laneSpatial
        )
        let handles = try RWKVStyleFusedThreeLayerStateFreeSurfaceHandles(kernels: kernels)
        self.kernels = kernels
        self.handles = handles
        self.stepCount = 0
    }

    public mutating func reset() throws(ANEError) {
        do {
            try SurfaceIO.copyFP16(
                dst: handles.xIn,
                dstChannelOffset: 0,
                src: handles.zeroLane,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
        } catch {
            throw .invalidArguments("fused three-layer state-free recurrent zero reset failed: \(error)")
        }
        self.stepCount = 0
    }

    public mutating func step(
        tokenInput: borrowing TensorBuffer,
        output: borrowing TensorBuffer,
        timings: inout StepTimingBreakdown
    ) throws(ANEError) {
        precondition(tokenInput.count == ModelConfig.dim)
        precondition(output.count == ModelConfig.dim)

        var t0 = RuntimeClock.now()
        do {
            try SurfaceIO.copyFP16(
                dst: handles.xIn,
                dstChannelOffset: 0,
                src: handles.zeroLane,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
            try tokenInput.withUnsafeBufferPointer { tokenBuf in
                try SurfaceIO.writeFP16SpatialSlice(
                    to: handles.xIn,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: handles.laneSpatial,
                    data: tokenBuf,
                    channels: ModelConfig.dim
                )
            }
        } catch {
            throw .invalidArguments("fused three-layer state-free recurrent input write failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

        t0 = RuntimeClock.now()
        do {
            try kernels.step.eval()
        } catch {
            throw .invalidArguments("fused three-layer state-free recurrent eval failed at step \(stepCount): \(error)")
        }
        timings.tAne += RuntimeClock.ms(RuntimeClock.now() - t0)

        t0 = RuntimeClock.now()
        do {
            try output.withUnsafeMutableBufferPointer { outBuf in
                try SurfaceIO.readFP16SpatialSlice(
                    from: handles.xOut,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: handles.laneSpatial,
                    into: outBuf,
                    channels: ModelConfig.dim
                )
            }
        } catch {
            throw .invalidArguments("fused three-layer state-free recurrent output readback failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)
        self.stepCount += 1
    }
}

public enum RWKVStyleRecurrentBench {
    public static func runContextScaling(
        contexts: [Int],
        warmup: Int,
        iterations: Int,
        transformerLayers: Int = 6
    ) throws(ANEError) -> RWKVStyleRecurrentScalingReport {
        try validate(contexts: contexts, warmup: warmup, iterations: iterations, transformerLayers: transformerLayers)

        let recurrentWeights = makeConstantRecurrentWeights()
        var recurrentSession = try RWKVStyleRecurrentSession(weights: recurrentWeights)
        let tokenInputs = makeTokenInputs(stepCount: contexts.max() ?? 0)

        var recurrentSamples: [RWKVStyleRecurrentBenchmarkSample] = []
        recurrentSamples.reserveCapacity(contexts.count)
        for context in contexts {
            recurrentSamples.append(
                try benchmarkRecurrentContext(
                    effectiveContext: context,
                    warmup: warmup,
                    iterations: iterations,
                    tokenInputs: tokenInputs,
                    session: &recurrentSession
                )
            )
        }

        let transformerWeights = makeConstantTransformerWeights(layerCount: transformerLayers)
        let transformerMaxContext = contexts.max() ?? 0
        let transformerKernels = LayerStorage<DecodeKernelSet>(count: transformerLayers) { idx in
            try! DecodeKernelSet(weights: transformerWeights[idx], maxSeq: transformerMaxContext)
        }

        var transformerSamples: [RWKVStyleRecurrentBenchmarkSample] = []
        var skippedTransformerContexts: [Int] = []
        transformerSamples.reserveCapacity(contexts.count)
        skippedTransformerContexts.reserveCapacity(contexts.count)

        for context in contexts {
            do {
                if let sample = try benchmarkTransformerContext(
                    effectiveContext: context,
                    warmup: warmup,
                    iterations: iterations,
                    tokenInputs: tokenInputs,
                    kernels: transformerKernels
                ) {
                    transformerSamples.append(sample)
                } else {
                    skippedTransformerContexts.append(context)
                }
            } catch {
                skippedTransformerContexts.append(context)
            }
        }

        return RWKVStyleRecurrentScalingReport(
            recurrentSamples: recurrentSamples,
            transformerSamples: transformerSamples,
            skippedTransformerContexts: skippedTransformerContexts
        )
    }

    private static func benchmarkRecurrentContext(
        effectiveContext: Int,
        warmup: Int,
        iterations: Int,
        tokenInputs: [Float],
        session: inout RWKVStyleRecurrentSession
    ) throws(ANEError) -> RWKVStyleRecurrentBenchmarkSample {
        try session.reset()
        let prefixSteps = effectiveContext - warmup - iterations
        let tokenInput = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        let output = TensorBuffer(count: ModelConfig.dim, zeroed: true)

        for step in 0..<prefixSteps {
            loadToken(step: step, tokenInputs: tokenInputs, into: tokenInput)
            var timings = StepTimingBreakdown()
            try session.step(tokenInput: tokenInput, output: output, timings: &timings)
        }

        for step in prefixSteps..<(prefixSteps + warmup) {
            loadToken(step: step, tokenInputs: tokenInputs, into: tokenInput)
            var timings = StepTimingBreakdown()
            try session.step(tokenInput: tokenInput, output: output, timings: &timings)
        }

        var latencies: [Double] = []
        latencies.reserveCapacity(iterations)
        for step in (prefixSteps + warmup)..<effectiveContext {
            loadToken(step: step, tokenInputs: tokenInputs, into: tokenInput)
            var timings = StepTimingBreakdown()
            let start = RuntimeClock.now()
            try session.step(tokenInput: tokenInput, output: output, timings: &timings)
            latencies.append(RuntimeClock.ms(RuntimeClock.now() - start))
        }

        return makeSample(effectiveContext: effectiveContext, latencies: latencies)
    }

    private static func benchmarkTransformerContext(
        effectiveContext: Int,
        warmup: Int,
        iterations: Int,
        tokenInputs: [Float],
        kernels: borrowing LayerStorage<DecodeKernelSet>
    ) throws(ANEError) -> RWKVStyleRecurrentBenchmarkSample? {
        var handles: [DecodeSurfaceHandles] = []
        handles.reserveCapacity(kernels.count)
        for idx in 0..<kernels.count {
            handles.append(try DecodeSurfaceHandles(kernels: kernels[idx], logicalMaxSeq: effectiveContext))
        }
        ForwardPass.initializeDecodeCachesAndMask(surfaceHandles: handles)

        var decodeState = try DecodeState(maxSeq: effectiveContext)
        let prefixSteps = effectiveContext - warmup - iterations
        let xCur = TensorBuffer(count: ModelConfig.dim, zeroed: true)

        for step in 0..<prefixSteps {
            loadToken(step: step, tokenInputs: tokenInputs, into: xCur)
            var timings = StepTimingBreakdown()
            try ForwardPass.runDecodeTimed(
                xCur: xCur,
                kernels: kernels,
                surfaceHandles: handles,
                decodeState: &decodeState,
                timings: &timings
            )
        }

        for step in prefixSteps..<(prefixSteps + warmup) {
            loadToken(step: step, tokenInputs: tokenInputs, into: xCur)
            var timings = StepTimingBreakdown()
            try ForwardPass.runDecodeTimed(
                xCur: xCur,
                kernels: kernels,
                surfaceHandles: handles,
                decodeState: &decodeState,
                timings: &timings
            )
        }

        var latencies: [Double] = []
        latencies.reserveCapacity(iterations)
        for step in (prefixSteps + warmup)..<effectiveContext {
            loadToken(step: step, tokenInputs: tokenInputs, into: xCur)
            var timings = StepTimingBreakdown()
            let start = RuntimeClock.now()
            try ForwardPass.runDecodeTimed(
                xCur: xCur,
                kernels: kernels,
                surfaceHandles: handles,
                decodeState: &decodeState,
                timings: &timings
            )
            latencies.append(RuntimeClock.ms(RuntimeClock.now() - start))
        }

        return makeSample(effectiveContext: effectiveContext, latencies: latencies)
    }

    private static func makeSample(
        effectiveContext: Int,
        latencies: [Double]
    ) -> RWKVStyleRecurrentBenchmarkSample {
        let medianLatencyMs = median(latencies)
        let tokensPerSecond = medianLatencyMs > 0 ? 1000.0 / medianLatencyMs : 0
        return RWKVStyleRecurrentBenchmarkSample(
            effectiveContext: effectiveContext,
            medianLatencyMs: medianLatencyMs,
            tokensPerSecond: tokensPerSecond
        )
    }

    private static func validate(
        contexts: [Int],
        warmup: Int,
        iterations: Int,
        transformerLayers: Int
    ) throws(ANEError) {
        guard !contexts.isEmpty else {
            throw .invalidArguments("contexts must not be empty")
        }
        guard warmup > 0 else {
            throw .invalidArguments("warmup must be > 0")
        }
        guard iterations > 0 else {
            throw .invalidArguments("iterations must be > 0")
        }
        guard transformerLayers > 0 else {
            throw .invalidArguments("transformerLayers must be > 0")
        }

        let laneSpatial = DecodeKernelSet.resolvedLaneSpatialForCurrentProcess()
        for context in contexts {
            guard context >= warmup + iterations else {
                throw .invalidArguments(
                    "context \(context) must be >= warmup + iterations (\(warmup + iterations))"
                )
            }
            guard context >= laneSpatial else {
                throw .invalidArguments("context \(context) must be >= lane spatial \(laneSpatial)")
            }
            guard context % laneSpatial == 0 else {
                throw .invalidArguments("context \(context) must be a multiple of lane spatial \(laneSpatial)")
            }
        }
    }

    private static func makeConstantRecurrentWeights(value: Float = 0.01) -> RWKVStyleRecurrentWeights {
        let weights = RWKVStyleRecurrentWeights()
        fill(weights.rms, value: 1.0)
        fill(weights.Wx, value: value)
        fill(weights.Ws, value: value)
        fill(weights.Wd, value: value)
        fill(weights.Wo, value: value)
        return weights
    }

    private static func makeConstantTransformerWeights(layerCount: Int, value: Float = 0.01) -> LayerStorage<LayerWeights> {
        LayerStorage<LayerWeights>(count: layerCount) { _ in
            let weights = LayerWeights()
            fill(weights.Wq, value: value)
            fill(weights.Wk, value: value)
            fill(weights.Wv, value: value)
            fill(weights.Wo, value: value)
            fill(weights.W1, value: value)
            fill(weights.W2, value: value)
            fill(weights.W3, value: value)
            fill(weights.rmsAtt, value: 1.0)
            fill(weights.rmsFfn, value: 1.0)
            return weights
        }
    }

    private static func makeTokenInputs(stepCount: Int) -> [Float] {
        let total = stepCount * ModelConfig.dim
        var inputs = [Float](repeating: 0, count: total)
        for idx in 0..<total {
            let bucket = Float((idx % 29) - 14)
            inputs[idx] = bucket * 0.01
        }
        return inputs
    }

    private static func loadToken(
        step: Int,
        tokenInputs: [Float],
        into buffer: borrowing TensorBuffer
    ) {
        let dim = ModelConfig.dim
        let base = step * dim
        precondition(base + dim <= tokenInputs.count)
        precondition(buffer.count == dim)
        buffer.withUnsafeMutablePointer { dst in
            for idx in 0..<dim {
                dst[idx] = tokenInputs[base + idx]
            }
        }
    }

    private static func fill(_ buffer: borrowing TensorBuffer, value: Float) {
        buffer.withUnsafeMutablePointer { ptr in
            for idx in 0..<buffer.count {
                ptr[idx] = value
            }
        }
    }

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[mid - 1] + sorted[mid]) * 0.5
        }
        return sorted[mid]
    }
}

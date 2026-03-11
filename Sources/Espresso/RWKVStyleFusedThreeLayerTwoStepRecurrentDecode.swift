import Foundation
import IOSurface
import ANEInterop
import ANERuntime
import ANETypes

public struct RWKVStyleFusedThreeLayerTwoStepSurfaceHandles {
    public let x0In: IOSurfaceRef
    public let x1In: IOSurfaceRef
    public let stateIn0: IOSurfaceRef
    public let stateIn1: IOSurfaceRef
    public let stateIn2: IOSurfaceRef
    public let x0Out: IOSurfaceRef
    public let x1Out: IOSurfaceRef
    public let stateMid0: IOSurfaceRef
    public let stateMid1: IOSurfaceRef
    public let stateMid2: IOSurfaceRef
    public let stateOut0: IOSurfaceRef
    public let stateOut1: IOSurfaceRef
    public let stateOut2: IOSurfaceRef
    public let zeroLane: IOSurfaceRef
    public let laneSpatial: Int

    public init(kernels: borrowing RWKVStyleFusedThreeLayerTwoStepKernelSet) throws(ANEError) {
        self.x0In = try kernels.step.inputSurface(at: 0)
        self.x1In = try kernels.step.inputSurface(at: 1)
        self.stateIn0 = try kernels.step.inputSurface(at: 2)
        self.stateIn1 = try kernels.step.inputSurface(at: 3)
        self.stateIn2 = try kernels.step.inputSurface(at: 4)
        self.x0Out = try kernels.step.outputSurface(at: 0)
        self.x1Out = try kernels.step.outputSurface(at: 1)
        self.stateMid0 = try kernels.step.outputSurface(at: 2)
        self.stateMid1 = try kernels.step.outputSurface(at: 3)
        self.stateMid2 = try kernels.step.outputSurface(at: 4)
        self.stateOut0 = try kernels.step.outputSurface(at: 5)
        self.stateOut1 = try kernels.step.outputSurface(at: 6)
        self.stateOut2 = try kernels.step.outputSurface(at: 7)
        self.laneSpatial = kernels.laneSpatial
        self.zeroLane = try makeRWKVTwoStepZeroLaneSurface(laneSpatial: kernels.laneSpatial)
    }
}

public struct RWKVStyleFusedThreeLayerTwoStepSession: ~Copyable {
    public let kernels: RWKVStyleFusedThreeLayerTwoStepKernelSet
    public let handles: RWKVStyleFusedThreeLayerTwoStepSurfaceHandles
    public private(set) var stepCount: Int
    private var hasPreparedState: Bool

    public init(
        weights0: borrowing RWKVStyleRecurrentWeights,
        weights1: borrowing RWKVStyleRecurrentWeights,
        weights2: borrowing RWKVStyleRecurrentWeights,
        laneSpatial: Int = RWKVStyleFusedThreeLayerTwoStepKernelSet.defaultLaneSpatial
    ) throws(ANEError) {
        let kernels = try RWKVStyleFusedThreeLayerTwoStepKernelSet(
            weights0: weights0,
            weights1: weights1,
            weights2: weights2,
            laneSpatial: laneSpatial
        )
        let handles = try RWKVStyleFusedThreeLayerTwoStepSurfaceHandles(kernels: kernels)
        self.kernels = kernels
        self.handles = handles
        self.stepCount = 0
        self.hasPreparedState = false
    }

    public mutating func reset() throws(ANEError) {
        do {
            try SurfaceIO.copyFP16(dst: handles.x0In, dstChannelOffset: 0, src: handles.zeroLane, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
            try SurfaceIO.copyFP16(dst: handles.x1In, dstChannelOffset: 0, src: handles.zeroLane, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
            try SurfaceIO.copyFP16(dst: handles.stateIn0, dstChannelOffset: 0, src: handles.zeroLane, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
            try SurfaceIO.copyFP16(dst: handles.stateIn1, dstChannelOffset: 0, src: handles.zeroLane, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
            try SurfaceIO.copyFP16(dst: handles.stateIn2, dstChannelOffset: 0, src: handles.zeroLane, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
        } catch {
            throw .invalidArguments("fused three-layer two-step recurrent zero reset failed: \(error)")
        }
        self.stepCount = 0
        self.hasPreparedState = false
    }

    public mutating func prepare(
        tokenInput0: borrowing TensorBuffer,
        tokenInput1: borrowing TensorBuffer,
        output0: borrowing TensorBuffer,
        output1: borrowing TensorBuffer,
        timings: inout StepTimingBreakdown
    ) throws(ANEError) {
        precondition(tokenInput0.count == ModelConfig.dim)
        precondition(tokenInput1.count == ModelConfig.dim)
        precondition(output0.count == ModelConfig.dim)
        precondition(output1.count == ModelConfig.dim)

        var t0 = RuntimeClock.now()
        do {
            try SurfaceIO.copyFP16(dst: handles.x0In, dstChannelOffset: 0, src: handles.zeroLane, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
            try SurfaceIO.copyFP16(dst: handles.x1In, dstChannelOffset: 0, src: handles.zeroLane, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
            try tokenInput0.withUnsafeBufferPointer { tokenBuf in
                try SurfaceIO.writeFP16SpatialSlice(
                    to: handles.x0In,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: handles.laneSpatial,
                    data: tokenBuf,
                    channels: ModelConfig.dim
                )
            }
            try tokenInput1.withUnsafeBufferPointer { tokenBuf in
                try SurfaceIO.writeFP16SpatialSlice(
                    to: handles.x1In,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: handles.laneSpatial,
                    data: tokenBuf,
                    channels: ModelConfig.dim
                )
            }
        } catch {
            throw .invalidArguments("fused three-layer two-step recurrent input write failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

        t0 = RuntimeClock.now()
        do {
            try kernels.step.eval()
        } catch {
            throw .invalidArguments("fused three-layer two-step recurrent eval failed at step \(stepCount): \(error)")
        }
        timings.tAne += RuntimeClock.ms(RuntimeClock.now() - t0)

        t0 = RuntimeClock.now()
        do {
            try output0.withUnsafeMutableBufferPointer { outBuf in
                try SurfaceIO.readFP16SpatialSlice(
                    from: handles.x0Out,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: handles.laneSpatial,
                    into: outBuf,
                    channels: ModelConfig.dim
                )
            }
            try output1.withUnsafeMutableBufferPointer { outBuf in
                try SurfaceIO.readFP16SpatialSlice(
                    from: handles.x1Out,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: handles.laneSpatial,
                    into: outBuf,
                    channels: ModelConfig.dim
                )
            }
        } catch {
            throw .invalidArguments("fused three-layer two-step recurrent output readback failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)
        self.hasPreparedState = true
    }

    public mutating func promotePreparedState(commitCount: Int) throws(ANEError) {
        guard hasPreparedState else {
            throw .invalidArguments("fused three-layer two-step recurrent state promotion requested without a prepared branch")
        }
        guard commitCount == 1 || commitCount == 2 else {
            throw .invalidArguments("fused three-layer two-step recurrent promotion commitCount must be 1 or 2")
        }

        let source0 = commitCount == 1 ? handles.stateMid0 : handles.stateOut0
        let source1 = commitCount == 1 ? handles.stateMid1 : handles.stateOut1
        let source2 = commitCount == 1 ? handles.stateMid2 : handles.stateOut2
        do {
            try SurfaceIO.copyFP16(dst: handles.stateIn0, dstChannelOffset: 0, src: source0, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
            try SurfaceIO.copyFP16(dst: handles.stateIn1, dstChannelOffset: 0, src: source1, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
            try SurfaceIO.copyFP16(dst: handles.stateIn2, dstChannelOffset: 0, src: source2, srcChannelOffset: 0, channels: ModelConfig.dim, spatial: handles.laneSpatial)
        } catch {
            throw .invalidArguments("fused three-layer two-step recurrent state promotion failed: \(error)")
        }

        self.stepCount += commitCount
        self.hasPreparedState = false
    }
}

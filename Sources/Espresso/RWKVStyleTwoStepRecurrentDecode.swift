import Foundation
import IOSurface
import ANEInterop
import ANERuntime
import ANETypes

@inline(__always)
private func makeRWKVTwoStepZeroLaneSurface(laneSpatial: Int) throws(ANEError) -> IOSurfaceRef {
    guard let zeroLane = ane_interop_create_surface(ModelConfig.dim * laneSpatial * 2) else {
        throw .surfaceAllocationFailed
    }

    let zeroValues = Array(repeating: Float(0), count: ModelConfig.dim * laneSpatial)
    zeroValues.withUnsafeBufferPointer { src in
        SurfaceIO.writeFP16(to: zeroLane, data: src, channels: ModelConfig.dim, spatial: laneSpatial)
    }
    return zeroLane
}

public struct RWKVStyleTwoStepRecurrentSurfaceHandles {
    public let x0In: IOSurfaceRef
    public let x1In: IOSurfaceRef
    public let stateIn: IOSurfaceRef
    public let x0Out: IOSurfaceRef
    public let x1Out: IOSurfaceRef
    public let stateMid: IOSurfaceRef
    public let stateOut: IOSurfaceRef
    public let zeroLane: IOSurfaceRef
    public let laneSpatial: Int

    public init(kernels: borrowing RWKVStyleTwoStepRecurrentKernelSet) throws(ANEError) {
        self.x0In = try kernels.step.inputSurface(at: 0)
        self.x1In = try kernels.step.inputSurface(at: 1)
        self.stateIn = try kernels.step.inputSurface(at: 2)
        self.x0Out = try kernels.step.outputSurface(at: 0)
        self.x1Out = try kernels.step.outputSurface(at: 1)
        self.stateMid = try kernels.step.outputSurface(at: 2)
        self.stateOut = try kernels.step.outputSurface(at: 3)
        self.laneSpatial = kernels.laneSpatial
        self.zeroLane = try makeRWKVTwoStepZeroLaneSurface(laneSpatial: kernels.laneSpatial)
    }
}

public struct RWKVStyleTwoStepRecurrentSession: ~Copyable {
    public let kernels: RWKVStyleTwoStepRecurrentKernelSet
    public let handles: RWKVStyleTwoStepRecurrentSurfaceHandles
    public private(set) var stepCount: Int
    private var hasPreparedState: Bool

    public init(
        weights: borrowing RWKVStyleRecurrentWeights,
        laneSpatial: Int = RWKVStyleTwoStepRecurrentKernelSet.defaultLaneSpatial
    ) throws(ANEError) {
        let kernels = try RWKVStyleTwoStepRecurrentKernelSet(weights: weights, laneSpatial: laneSpatial)
        let handles = try RWKVStyleTwoStepRecurrentSurfaceHandles(kernels: kernels)
        self.kernels = kernels
        self.handles = handles
        self.stepCount = 0
        self.hasPreparedState = false
    }

    public mutating func reset() throws(ANEError) {
        do {
            try SurfaceIO.copyFP16(
                dst: handles.x0In,
                dstChannelOffset: 0,
                src: handles.zeroLane,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
            try SurfaceIO.copyFP16(
                dst: handles.x1In,
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
            throw .invalidArguments("two-step recurrent zero reset failed: \(error)")
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
            try SurfaceIO.copyFP16(
                dst: handles.x0In,
                dstChannelOffset: 0,
                src: handles.zeroLane,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
            try SurfaceIO.copyFP16(
                dst: handles.x1In,
                dstChannelOffset: 0,
                src: handles.zeroLane,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
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
            throw .invalidArguments("two-step recurrent input write failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)

        t0 = RuntimeClock.now()
        do {
            try kernels.step.eval()
        } catch {
            throw .invalidArguments("two-step recurrent eval failed at step \(stepCount): \(error)")
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
            throw .invalidArguments("two-step recurrent output readback failed: \(error)")
        }
        timings.tIO += RuntimeClock.ms(RuntimeClock.now() - t0)
        self.hasPreparedState = true
    }

    public mutating func promotePreparedState(commitCount: Int) throws(ANEError) {
        guard hasPreparedState else {
            throw .invalidArguments("two-step recurrent state promotion requested without a prepared branch")
        }
        guard commitCount == 1 || commitCount == 2 else {
            throw .invalidArguments("two-step recurrent promotion commitCount must be 1 or 2")
        }

        let source = commitCount == 1 ? handles.stateMid : handles.stateOut
        do {
            try SurfaceIO.copyFP16(
                dst: handles.stateIn,
                dstChannelOffset: 0,
                src: source,
                srcChannelOffset: 0,
                channels: ModelConfig.dim,
                spatial: handles.laneSpatial
            )
        } catch {
            throw .invalidArguments("two-step recurrent state promotion failed: \(error)")
        }

        self.stepCount += commitCount
        self.hasPreparedState = false
    }
}

import IOSurface
import ANERuntime
import ANETypes

public struct LayerSurfaceHandles {
    public let fwdAttnIn: IOSurfaceRef
    public let fwdAttnOut: IOSurfaceRef
    public let fwdFFNIn: IOSurfaceRef
    public let fwdFFNOut: IOSurfaceRef
    public let ffnBwdIn: IOSurfaceRef
    public let ffnBwdOut: IOSurfaceRef
    public let sdpaBwd1In: IOSurfaceRef
    public let sdpaBwd1Out: IOSurfaceRef
    public let qkvBwdIn: IOSurfaceRef
    public let qkvBwdOut: IOSurfaceRef
    public let sdpaBwd2In: IOSurfaceRef
    public let sdpaBwd2Out: IOSurfaceRef

    public init(kernels: borrowing LayerKernelSet, staticKernel: borrowing StaticKernel) throws(ANEError) {
        self.fwdAttnIn = try kernels.fwdAttn.inputSurface(at: 0)
        self.fwdAttnOut = try kernels.fwdAttn.outputSurface(at: 0)
        self.fwdFFNIn = try kernels.fwdFFN.inputSurface(at: 0)
        self.fwdFFNOut = try kernels.fwdFFN.outputSurface(at: 0)
        self.ffnBwdIn = try kernels.ffnBwd.inputSurface(at: 0)
        self.ffnBwdOut = try kernels.ffnBwd.outputSurface(at: 0)
        self.sdpaBwd1In = try kernels.sdpaBwd1.inputSurface(at: 0)
        self.sdpaBwd1Out = try kernels.sdpaBwd1.outputSurface(at: 0)
        self.qkvBwdIn = try kernels.qkvBwd.inputSurface(at: 0)
        self.qkvBwdOut = try kernels.qkvBwd.outputSurface(at: 0)
        self.sdpaBwd2In = try staticKernel.kernel.inputSurface(at: 0)
        self.sdpaBwd2Out = try staticKernel.kernel.outputSurface(at: 0)
    }
}

public enum SurfaceHandleCache {
    public static func build(
        kernels: borrowing LayerStorage<LayerKernelSet>,
        staticKernels: borrowing LayerStorage<StaticKernel>
    ) throws(ANEError) -> [LayerSurfaceHandles] {
        precondition(kernels.count == staticKernels.count)
        var handles: [LayerSurfaceHandles] = []
        handles.reserveCapacity(kernels.count)
        for i in 0..<kernels.count {
            handles.append(try LayerSurfaceHandles(kernels: kernels[i], staticKernel: staticKernels[i]))
        }
        return handles
    }
}

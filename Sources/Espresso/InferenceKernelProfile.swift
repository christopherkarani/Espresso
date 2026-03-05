/// Microsecond-level profiling for the inference forward pass.
///
/// This intentionally lives in the Espresso module (not EspressoBench) so the hot-path
/// forward pass can record events without exposing internal implementation details via
/// a protocol or closure.
public struct InferenceKernelProfile: Sendable {
    public struct LayerSamples: Sendable {
        public var attnWriteUS: [Double] = []
        public var attnEvalUS: [Double] = []
        public var attnReadUS: [Double] = []

        /// Either a CPU round-trip write (fp32->fp16) or 0 if not performed.
        public var ffnWriteUS: [Double] = []
        /// FP16 surface-to-surface copy time (attnOut -> ffnIn), or 0 if not performed.
        public var ffnCopyUS: [Double] = []

        public var ffnEvalUS: [Double] = []
        public var ffnReadUS: [Double] = []

        /// Host-visible time between `attn.eval` return and `ffn.eval` call entry.
        public var gapAttnToFfnUS: [Double] = []
    }

    public private(set) var layers: [LayerSamples]

    public init(layerCount: Int, reservedSamplesPerLayer: Int) {
        precondition(layerCount >= 0)
        self.layers = Array(repeating: LayerSamples(), count: layerCount)
        if reservedSamplesPerLayer > 0 {
            for i in layers.indices {
                layers[i].attnWriteUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].attnEvalUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].attnReadUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].ffnWriteUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].ffnCopyUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].ffnEvalUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].ffnReadUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].gapAttnToFfnUS.reserveCapacity(reservedSamplesPerLayer)
            }
        }
    }

    public mutating func record(
        layerIndex: Int,
        attnWriteUS: Double,
        attnEvalUS: Double,
        attnReadUS: Double,
        ffnWriteUS: Double,
        ffnCopyUS: Double,
        ffnEvalUS: Double,
        ffnReadUS: Double,
        gapAttnToFfnUS: Double
    ) {
        precondition(layerIndex >= 0 && layerIndex < layers.count)
        layers[layerIndex].attnWriteUS.append(attnWriteUS)
        layers[layerIndex].attnEvalUS.append(attnEvalUS)
        layers[layerIndex].attnReadUS.append(attnReadUS)
        layers[layerIndex].ffnWriteUS.append(ffnWriteUS)
        layers[layerIndex].ffnCopyUS.append(ffnCopyUS)
        layers[layerIndex].ffnEvalUS.append(ffnEvalUS)
        layers[layerIndex].ffnReadUS.append(ffnReadUS)
        layers[layerIndex].gapAttnToFfnUS.append(gapAttnToFfnUS)
    }
}

/// Reference wrapper used by the hot path to append samples without `inout`.
public final class InferenceKernelProfiler: @unchecked Sendable {
    public private(set) var profile: InferenceKernelProfile

    public init(layerCount: Int, reservedSamplesPerLayer: Int) {
        self.profile = InferenceKernelProfile(layerCount: layerCount, reservedSamplesPerLayer: reservedSamplesPerLayer)
    }

    public func record(
        layerIndex: Int,
        attnWriteUS: Double,
        attnEvalUS: Double,
        attnReadUS: Double,
        ffnWriteUS: Double,
        ffnCopyUS: Double,
        ffnEvalUS: Double,
        ffnReadUS: Double,
        gapAttnToFfnUS: Double
    ) {
        profile.record(
            layerIndex: layerIndex,
            attnWriteUS: attnWriteUS,
            attnEvalUS: attnEvalUS,
            attnReadUS: attnReadUS,
            ffnWriteUS: ffnWriteUS,
            ffnCopyUS: ffnCopyUS,
            ffnEvalUS: ffnEvalUS,
            ffnReadUS: ffnReadUS,
            gapAttnToFfnUS: gapAttnToFfnUS
        )
    }
}

/// Microsecond-level profiling for the inference forward pass.
///
/// This intentionally lives in the Espresso module (not EspressoBench) so the hot-path
/// forward pass can record events without exposing internal implementation details via
/// a protocol or closure.
public struct InferenceKernelProfile: Sendable {
    public struct LayerSamples: Sendable {
        public var attnWriteUS: [Double] = []
        public var attnWriteLockUS: [Double] = []
        public var attnWriteBodyUS: [Double] = []
        public var attnWriteUnlockUS: [Double] = []
        public var attnEvalUS: [Double] = []
        /// Driver-reported on-chip execution time for attn eval (ns), or 0 if unavailable.
        public var attnHwNS: [UInt64] = []
        /// Host overhead for attn eval (`attnEvalUS - attnHwNS/1000`), clamped at 0.
        public var attnHostOverheadUS: [Double] = []
        public var attnReadUS: [Double] = []
        public var attnReadLockUS: [Double] = []
        public var attnReadBodyUS: [Double] = []
        public var attnReadUnlockUS: [Double] = []

        /// Either a CPU round-trip write (fp32->fp16) or 0 if not performed.
        public var ffnWriteUS: [Double] = []
        public var ffnWriteLockUS: [Double] = []
        public var ffnWriteBodyUS: [Double] = []
        public var ffnWriteUnlockUS: [Double] = []
        /// FP16 surface-to-surface copy time (attnOut -> ffnIn), or 0 if not performed.
        public var ffnCopyUS: [Double] = []

        public var ffnEvalUS: [Double] = []
        /// Driver-reported on-chip execution time for FFN eval (ns), or 0 if unavailable.
        public var ffnHwNS: [UInt64] = []
        /// Host overhead for FFN eval (`ffnEvalUS - ffnHwNS/1000`), clamped at 0.
        public var ffnHostOverheadUS: [Double] = []
        public var ffnReadUS: [Double] = []
        public var ffnReadLockUS: [Double] = []
        public var ffnReadBodyUS: [Double] = []
        public var ffnReadUnlockUS: [Double] = []

        /// Host-visible time between `attn.eval` return and `ffn.eval` call entry.
        public var gapAttnToFfnUS: [Double] = []
    }

    public struct LayerAverages: Sendable {
        public let sampleCount: Int
        public let attnWriteUS: Double
        public let attnWriteLockUS: Double
        public let attnWriteBodyUS: Double
        public let attnWriteUnlockUS: Double
        public let attnEvalUS: Double
        public let attnHwUS: Double
        public let attnHostOverheadUS: Double
        public let attnReadUS: Double
        public let attnReadLockUS: Double
        public let attnReadBodyUS: Double
        public let attnReadUnlockUS: Double
        public let gapAttnToFfnUS: Double
        public let ffnWriteUS: Double
        public let ffnWriteLockUS: Double
        public let ffnWriteBodyUS: Double
        public let ffnWriteUnlockUS: Double
        public let ffnCopyUS: Double
        public let ffnEvalUS: Double
        public let ffnHwUS: Double
        public let ffnHostOverheadUS: Double
        public let ffnReadUS: Double
        public let ffnReadLockUS: Double
        public let ffnReadBodyUS: Double
        public let ffnReadUnlockUS: Double

        public var attnIOLockUS: Double { attnWriteLockUS + attnReadLockUS }
        public var attnIOBodyUS: Double { attnWriteBodyUS + attnReadBodyUS }
        public var attnIOUnlockUS: Double { attnWriteUnlockUS + attnReadUnlockUS }

        /// CPU round-trip handoff cost (`attnOut` read + FFN input write).
        public var handoffCPUUS: Double { attnReadUS + ffnWriteUS }
        /// FP16 surface-to-surface handoff cost.
        public var handoffFP16CopyUS: Double { ffnCopyUS }

        public var ffnIOLockUS: Double { ffnWriteLockUS + ffnReadLockUS }
        public var ffnIOBodyUS: Double { ffnWriteBodyUS + ffnReadBodyUS }
        public var ffnIOUnlockUS: Double { ffnWriteUnlockUS + ffnReadUnlockUS }
    }

    public private(set) var layers: [LayerSamples]

    public init(layerCount: Int, reservedSamplesPerLayer: Int) {
        precondition(layerCount >= 0)
        self.layers = Array(repeating: LayerSamples(), count: layerCount)
        if reservedSamplesPerLayer > 0 {
            for i in layers.indices {
                layers[i].attnWriteUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].attnWriteLockUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].attnWriteBodyUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].attnWriteUnlockUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].attnEvalUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].attnHwNS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].attnHostOverheadUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].attnReadUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].attnReadLockUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].attnReadBodyUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].attnReadUnlockUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].ffnWriteUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].ffnWriteLockUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].ffnWriteBodyUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].ffnWriteUnlockUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].ffnCopyUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].ffnEvalUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].ffnHwNS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].ffnHostOverheadUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].ffnReadUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].ffnReadLockUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].ffnReadBodyUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].ffnReadUnlockUS.reserveCapacity(reservedSamplesPerLayer)
                layers[i].gapAttnToFfnUS.reserveCapacity(reservedSamplesPerLayer)
            }
        }
    }

    public mutating func record(
        layerIndex: Int,
        attnWriteUS: Double,
        attnWriteLockUS: Double,
        attnWriteBodyUS: Double,
        attnWriteUnlockUS: Double,
        attnEvalUS: Double,
        attnHwNS: UInt64,
        attnHostOverheadUS: Double,
        attnReadUS: Double,
        attnReadLockUS: Double,
        attnReadBodyUS: Double,
        attnReadUnlockUS: Double,
        ffnWriteUS: Double,
        ffnWriteLockUS: Double,
        ffnWriteBodyUS: Double,
        ffnWriteUnlockUS: Double,
        ffnCopyUS: Double,
        ffnEvalUS: Double,
        ffnHwNS: UInt64,
        ffnHostOverheadUS: Double,
        ffnReadUS: Double,
        ffnReadLockUS: Double,
        ffnReadBodyUS: Double,
        ffnReadUnlockUS: Double,
        gapAttnToFfnUS: Double
    ) {
        precondition(layerIndex >= 0 && layerIndex < layers.count)
        layers[layerIndex].attnWriteUS.append(attnWriteUS)
        layers[layerIndex].attnWriteLockUS.append(attnWriteLockUS)
        layers[layerIndex].attnWriteBodyUS.append(attnWriteBodyUS)
        layers[layerIndex].attnWriteUnlockUS.append(attnWriteUnlockUS)
        layers[layerIndex].attnEvalUS.append(attnEvalUS)
        layers[layerIndex].attnHwNS.append(attnHwNS)
        layers[layerIndex].attnHostOverheadUS.append(attnHostOverheadUS)
        layers[layerIndex].attnReadUS.append(attnReadUS)
        layers[layerIndex].attnReadLockUS.append(attnReadLockUS)
        layers[layerIndex].attnReadBodyUS.append(attnReadBodyUS)
        layers[layerIndex].attnReadUnlockUS.append(attnReadUnlockUS)
        layers[layerIndex].ffnWriteUS.append(ffnWriteUS)
        layers[layerIndex].ffnWriteLockUS.append(ffnWriteLockUS)
        layers[layerIndex].ffnWriteBodyUS.append(ffnWriteBodyUS)
        layers[layerIndex].ffnWriteUnlockUS.append(ffnWriteUnlockUS)
        layers[layerIndex].ffnCopyUS.append(ffnCopyUS)
        layers[layerIndex].ffnEvalUS.append(ffnEvalUS)
        layers[layerIndex].ffnHwNS.append(ffnHwNS)
        layers[layerIndex].ffnHostOverheadUS.append(ffnHostOverheadUS)
        layers[layerIndex].ffnReadUS.append(ffnReadUS)
        layers[layerIndex].ffnReadLockUS.append(ffnReadLockUS)
        layers[layerIndex].ffnReadBodyUS.append(ffnReadBodyUS)
        layers[layerIndex].ffnReadUnlockUS.append(ffnReadUnlockUS)
        layers[layerIndex].gapAttnToFfnUS.append(gapAttnToFfnUS)
    }

    public func averageLayerMetrics(layerIndex: Int) -> LayerAverages {
        precondition(layerIndex >= 0 && layerIndex < layers.count)
        let layer = layers[layerIndex]
        let sampleCount = layer.attnWriteUS.count
        precondition(layer.attnWriteLockUS.count == sampleCount)
        precondition(layer.attnWriteBodyUS.count == sampleCount)
        precondition(layer.attnWriteUnlockUS.count == sampleCount)
        precondition(layer.attnEvalUS.count == sampleCount)
        precondition(layer.attnHwNS.count == sampleCount)
        precondition(layer.attnHostOverheadUS.count == sampleCount)
        precondition(layer.attnReadUS.count == sampleCount)
        precondition(layer.attnReadLockUS.count == sampleCount)
        precondition(layer.attnReadBodyUS.count == sampleCount)
        precondition(layer.attnReadUnlockUS.count == sampleCount)
        precondition(layer.ffnWriteUS.count == sampleCount)
        precondition(layer.ffnWriteLockUS.count == sampleCount)
        precondition(layer.ffnWriteBodyUS.count == sampleCount)
        precondition(layer.ffnWriteUnlockUS.count == sampleCount)
        precondition(layer.ffnCopyUS.count == sampleCount)
        precondition(layer.ffnEvalUS.count == sampleCount)
        precondition(layer.ffnHwNS.count == sampleCount)
        precondition(layer.ffnHostOverheadUS.count == sampleCount)
        precondition(layer.ffnReadUS.count == sampleCount)
        precondition(layer.ffnReadLockUS.count == sampleCount)
        precondition(layer.ffnReadBodyUS.count == sampleCount)
        precondition(layer.ffnReadUnlockUS.count == sampleCount)
        precondition(layer.gapAttnToFfnUS.count == sampleCount)

        func mean(_ values: [Double]) -> Double {
            guard !values.isEmpty else { return 0 }
            return values.reduce(0, +) / Double(values.count)
        }
        func meanHwUS(_ values: [UInt64]) -> Double {
            guard !values.isEmpty else { return 0 }
            let total = values.reduce(0) { $0 + Double($1) / 1_000.0 }
            return total / Double(values.count)
        }

        return LayerAverages(
            sampleCount: sampleCount,
            attnWriteUS: mean(layer.attnWriteUS),
            attnWriteLockUS: mean(layer.attnWriteLockUS),
            attnWriteBodyUS: mean(layer.attnWriteBodyUS),
            attnWriteUnlockUS: mean(layer.attnWriteUnlockUS),
            attnEvalUS: mean(layer.attnEvalUS),
            attnHwUS: meanHwUS(layer.attnHwNS),
            attnHostOverheadUS: mean(layer.attnHostOverheadUS),
            attnReadUS: mean(layer.attnReadUS),
            attnReadLockUS: mean(layer.attnReadLockUS),
            attnReadBodyUS: mean(layer.attnReadBodyUS),
            attnReadUnlockUS: mean(layer.attnReadUnlockUS),
            gapAttnToFfnUS: mean(layer.gapAttnToFfnUS),
            ffnWriteUS: mean(layer.ffnWriteUS),
            ffnWriteLockUS: mean(layer.ffnWriteLockUS),
            ffnWriteBodyUS: mean(layer.ffnWriteBodyUS),
            ffnWriteUnlockUS: mean(layer.ffnWriteUnlockUS),
            ffnCopyUS: mean(layer.ffnCopyUS),
            ffnEvalUS: mean(layer.ffnEvalUS),
            ffnHwUS: meanHwUS(layer.ffnHwNS),
            ffnHostOverheadUS: mean(layer.ffnHostOverheadUS),
            ffnReadUS: mean(layer.ffnReadUS),
            ffnReadLockUS: mean(layer.ffnReadLockUS),
            ffnReadBodyUS: mean(layer.ffnReadBodyUS),
            ffnReadUnlockUS: mean(layer.ffnReadUnlockUS)
        )
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
        attnWriteLockUS: Double,
        attnWriteBodyUS: Double,
        attnWriteUnlockUS: Double,
        attnEvalUS: Double,
        attnHwNS: UInt64,
        attnHostOverheadUS: Double,
        attnReadUS: Double,
        attnReadLockUS: Double,
        attnReadBodyUS: Double,
        attnReadUnlockUS: Double,
        ffnWriteUS: Double,
        ffnWriteLockUS: Double,
        ffnWriteBodyUS: Double,
        ffnWriteUnlockUS: Double,
        ffnCopyUS: Double,
        ffnEvalUS: Double,
        ffnHwNS: UInt64,
        ffnHostOverheadUS: Double,
        ffnReadUS: Double,
        ffnReadLockUS: Double,
        ffnReadBodyUS: Double,
        ffnReadUnlockUS: Double,
        gapAttnToFfnUS: Double
    ) {
        profile.record(
            layerIndex: layerIndex,
            attnWriteUS: attnWriteUS,
            attnWriteLockUS: attnWriteLockUS,
            attnWriteBodyUS: attnWriteBodyUS,
            attnWriteUnlockUS: attnWriteUnlockUS,
            attnEvalUS: attnEvalUS,
            attnHwNS: attnHwNS,
            attnHostOverheadUS: attnHostOverheadUS,
            attnReadUS: attnReadUS,
            attnReadLockUS: attnReadLockUS,
            attnReadBodyUS: attnReadBodyUS,
            attnReadUnlockUS: attnReadUnlockUS,
            ffnWriteUS: ffnWriteUS,
            ffnWriteLockUS: ffnWriteLockUS,
            ffnWriteBodyUS: ffnWriteBodyUS,
            ffnWriteUnlockUS: ffnWriteUnlockUS,
            ffnCopyUS: ffnCopyUS,
            ffnEvalUS: ffnEvalUS,
            ffnHwNS: ffnHwNS,
            ffnHostOverheadUS: ffnHostOverheadUS,
            ffnReadUS: ffnReadUS,
            ffnReadLockUS: ffnReadLockUS,
            ffnReadBodyUS: ffnReadBodyUS,
            ffnReadUnlockUS: ffnReadUnlockUS,
            gapAttnToFfnUS: gapAttnToFfnUS
        )
    }
}

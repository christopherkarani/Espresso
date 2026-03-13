import Foundation
import IOSurface
import os.lock
import ANERuntime
import ANETypes

public enum GenerationOutputHeadBackend: Sendable {
    case cpu
    case cpuExactStaged
    case cpuExactClustered
    case aneClassifier
    case aneRMSNormClassifier
    /// CPU fallback that opportunistically uses an ANE RMSNorm+classifier head when it compiles.
    case cpuThenANE
    /// Vocabulary-partitioned argmax with Cauchy-Schwarz pruning.
    case cpuPartitionedArgmax
    /// FP16 tiled classifier: converts FP16 weights in L2-sized tiles, runs sgemm on L2-resident FP32 data.
    /// Halves DRAM bandwidth for classifier weights compared to full FP32 sgemm.
    case cpuFP16Tiled
}

// MARK: - Deferred ANE head

/// Thread-safe container for an optional `ANEGenerationRMSNormClassifierHead`.
///
/// Readers on the hot token-selection path call `readyHead`, which performs a single
/// `os_unfair_lock`-protected load. If the ANE head never compiles, callers continue
/// using the CPU fallback path.
public final class DeferredANEHead: @unchecked Sendable {
    private var _lock: os_unfair_lock = os_unfair_lock()
    private var _head: ANEGenerationRMSNormClassifierHead?

    public init() {}

    /// Store the compiled head. Safe to call from any thread exactly once.
    func store(_ head: consuming ANEGenerationRMSNormClassifierHead) {
        os_unfair_lock_lock(&_lock)
        _head = consume head
        os_unfair_lock_unlock(&_lock)
    }

    /// Returns the compiled head if background compilation has finished, otherwise `nil`.
    /// This is the hot-path read — lock acquisition is extremely brief (pointer load).
    func readyHead() -> ANEGenerationRMSNormClassifierHead? {
        os_unfair_lock_lock(&_lock)
        let h = _head
        os_unfair_lock_unlock(&_lock)
        return h
    }
}

enum ANEGenerationOutputHeadIO {
    static func initializeInputSurface(
        _ surface: IOSurfaceRef,
        laneSpatial: Int
    ) throws(GenerationError) {
        guard laneSpatial > 1 else { return }
        let zeroInput = TensorBuffer(count: ModelConfig.dim * laneSpatial, zeroed: true)
        zeroInput.withUnsafeBufferPointer { zeroPtr in
            SurfaceIO.writeFP16(
                to: surface,
                data: zeroPtr,
                channels: ModelConfig.dim,
                spatial: laneSpatial
            )
        }
    }

    static func writeSingleToken(
        _ input: borrowing TensorBuffer,
        to surface: IOSurfaceRef,
        laneSpatial: Int
    ) throws(GenerationError) {
        precondition(input.count == ModelConfig.dim)
        if laneSpatial == 1 {
            input.withUnsafeBufferPointer { src in
                SurfaceIO.writeFP16(to: surface, data: src, channels: ModelConfig.dim, spatial: 1)
            }
            return
        }

        do {
            try input.withUnsafeBufferPointer { src in
                try SurfaceIO.writeFP16SpatialSlice(
                    to: surface,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSpatial,
                    data: src,
                    channels: ModelConfig.dim
                )
            }
        } catch {
            throw .runtimeFailure("ANE output-head input write failed: \(error)")
        }
    }

    static func writeTokenPair(
        _ inputA: borrowing TensorBuffer,
        _ inputB: borrowing TensorBuffer,
        to surface: IOSurfaceRef,
        laneSpatial: Int
    ) throws(GenerationError) {
        precondition(inputA.count == ModelConfig.dim)
        precondition(inputB.count == ModelConfig.dim)
        guard laneSpatial >= 2 else {
            throw .invalidArguments("ANE output-head pair write requires laneSpatial >= 2")
        }

        do {
            try inputA.withUnsafeBufferPointer { src in
                try SurfaceIO.writeFP16SpatialSlice(
                    to: surface,
                    channelOffset: 0,
                    spatialIndex: 0,
                    spatial: laneSpatial,
                    data: src,
                    channels: ModelConfig.dim
                )
            }
            try inputB.withUnsafeBufferPointer { src in
                try SurfaceIO.writeFP16SpatialSlice(
                    to: surface,
                    channelOffset: 0,
                    spatialIndex: 1,
                    spatial: laneSpatial,
                    data: src,
                    channels: ModelConfig.dim
                )
            }
        } catch {
            throw .runtimeFailure("ANE output-head pair input write failed: \(error)")
        }
    }

    static func readSingleTokenLogits(
        from surface: IOSurfaceRef,
        into logits: borrowing TensorBuffer,
        vocabSize: Int,
        laneSpatial: Int
    ) throws(GenerationError) {
        precondition(logits.count == vocabSize)
        do {
            if laneSpatial == 1 {
                logits.withUnsafeMutableBufferPointer { dst in
                    SurfaceIO.readFP16(
                        from: surface,
                        into: dst,
                        channelOffset: 0,
                        channels: vocabSize,
                        spatial: 1
                    )
                }
            } else {
                try logits.withUnsafeMutableBufferPointer { dst in
                    try SurfaceIO.readFP16SpatialSlice(
                        from: surface,
                        channelOffset: 0,
                        spatialIndex: 0,
                        spatial: laneSpatial,
                        into: dst,
                        channels: vocabSize
                    )
                }
            }
        } catch {
            throw .runtimeFailure("ANE output-head output read failed: \(error)")
        }
    }

    static func argmaxSingleTokenLogits(
        from surface: IOSurfaceRef,
        vocabSize: Int,
        laneSpatial: Int
    ) throws(GenerationError) -> UInt16 {
        do {
            let result = try SurfaceIO.argmaxFP16SpatialSlice(
                from: surface,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: laneSpatial,
                channels: vocabSize
            )
            guard let token = UInt16(exactly: result.index) else {
                throw GenerationError.invalidArguments(
                    "selected token index \(result.index) exceeds UInt16 range"
                )
            }
            return token
        } catch let error as GenerationError {
            throw error
        } catch {
            throw .runtimeFailure("ANE output-head argmax failed: \(error)")
        }
    }

    static func argmaxSingleTokenLogitsWithHint(
        from surface: IOSurfaceRef,
        vocabSize: Int,
        laneSpatial: Int,
        hintSurface: IOSurfaceRef
    ) throws(GenerationError) -> UInt16 {
        do {
            let result = try SurfaceIO.argmaxFP16SpatialSliceWithHint(
                from: surface,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: laneSpatial,
                channels: vocabSize,
                hintSurface: hintSurface,
                hintSpatialIndex: 0,
                hintSpatial: laneSpatial
            )
            guard let token = UInt16(exactly: result.index) else {
                throw GenerationError.invalidArguments(
                    "selected token index \(result.index) exceeds UInt16 range"
                )
            }
            return token
        } catch let error as GenerationError {
            throw error
        } catch {
            throw .runtimeFailure("ANE output-head argmax with hint failed: \(error)")
        }
    }

    static func argmaxTokenPairLogitsWithHint(
        from surface: IOSurfaceRef,
        vocabSize: Int,
        laneSpatial: Int,
        hintSurface: IOSurfaceRef
    ) throws(GenerationError) -> (UInt16, UInt16) {
        guard laneSpatial >= 2 else {
            throw .invalidArguments("ANE output-head pair argmax requires laneSpatial >= 2")
        }

        func convert(_ result: SurfaceIO.FP16ArgmaxResult) throws(GenerationError) -> UInt16 {
            guard let token = UInt16(exactly: result.index) else {
                throw .invalidArguments("selected token index \(result.index) exceeds UInt16 range")
            }
            return token
        }

        do {
            let first = try SurfaceIO.argmaxFP16SpatialSliceWithHint(
                from: surface,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: laneSpatial,
                channels: vocabSize,
                hintSurface: hintSurface,
                hintSpatialIndex: 0,
                hintSpatial: laneSpatial
            )
            let second = try SurfaceIO.argmaxFP16SpatialSliceWithHint(
                from: surface,
                channelOffset: 0,
                spatialIndex: 1,
                spatial: laneSpatial,
                channels: vocabSize,
                hintSurface: hintSurface,
                hintSpatialIndex: 1,
                hintSpatial: laneSpatial
            )
            return (try convert(first), try convert(second))
        } catch let error as GenerationError {
            throw error
        } catch {
            throw .runtimeFailure("ANE output-head pair argmax with hint failed: \(error)")
        }
    }

    static func argmaxTokenPairLogits(
        from surface: IOSurfaceRef,
        vocabSize: Int,
        laneSpatial: Int
    ) throws(GenerationError) -> (UInt16, UInt16) {
        guard laneSpatial >= 2 else {
            throw .invalidArguments("ANE output-head pair argmax requires laneSpatial >= 2")
        }

        func convert(_ result: SurfaceIO.FP16ArgmaxResult) throws(GenerationError) -> UInt16 {
            guard let token = UInt16(exactly: result.index) else {
                throw .invalidArguments("selected token index \(result.index) exceeds UInt16 range")
            }
            return token
        }

        do {
            let first = try SurfaceIO.argmaxFP16SpatialSlice(
                from: surface,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: laneSpatial,
                channels: vocabSize
            )
            let second = try SurfaceIO.argmaxFP16SpatialSlice(
                from: surface,
                channelOffset: 0,
                spatialIndex: 1,
                spatial: laneSpatial,
                channels: vocabSize
            )
            return (try convert(first), try convert(second))
        } catch let error as GenerationError {
            throw error
        } catch {
            throw .runtimeFailure("ANE output-head pair argmax failed: \(error)")
        }
    }
}

final class ANEGenerationClassifierHead {
    private static let defaultLaneSpatial = 32

    let kernelSet: GenerationClassifierKernelSet
    let inputSurface: IOSurfaceRef
    let outputSurface: IOSurfaceRef
    let vocabSize: Int
    let laneSpatial: Int

    init(
        classifierWeights: borrowing TensorBuffer,
        vocabSize: Int,
        laneSpatial: Int = defaultLaneSpatial
    ) throws(GenerationError) {
        do {
            let kernelSet = try GenerationClassifierKernelSet(
                classifier: classifierWeights,
                vocabSize: vocabSize,
                laneSpatial: laneSpatial
            )
            self.inputSurface = try kernelSet.classifier.inputSurface(at: 0)
            self.outputSurface = try kernelSet.classifier.outputSurface(at: 0)
            self.kernelSet = kernelSet
            self.vocabSize = vocabSize
            self.laneSpatial = laneSpatial
            try ANEGenerationOutputHeadIO.initializeInputSurface(self.inputSurface, laneSpatial: laneSpatial)
        } catch {
            throw .runtimeFailure("ANE classifier setup failed: \(error)")
        }
    }

    func project(
        normalizedInput: borrowing TensorBuffer,
        logits: borrowing TensorBuffer
    ) throws(GenerationError) {
        precondition(normalizedInput.count == ModelConfig.dim)
        precondition(logits.count == vocabSize)

        try ANEGenerationOutputHeadIO.writeSingleToken(
            normalizedInput,
            to: inputSurface,
            laneSpatial: laneSpatial
        )

        do {
            try kernelSet.classifier.eval()
        } catch {
            throw .runtimeFailure("ANE classifier eval failed: \(error)")
        }

        try ANEGenerationOutputHeadIO.readSingleTokenLogits(
            from: outputSurface,
            into: logits,
            vocabSize: vocabSize,
            laneSpatial: laneSpatial
        )
    }

    func selectArgmax(
        normalizedInput: borrowing TensorBuffer
    ) throws(GenerationError) -> UInt16 {
        precondition(normalizedInput.count == ModelConfig.dim)

        try ANEGenerationOutputHeadIO.writeSingleToken(
            normalizedInput,
            to: inputSurface,
            laneSpatial: laneSpatial
        )

        do {
            try kernelSet.classifier.eval()
        } catch {
            throw .runtimeFailure("ANE classifier eval failed: \(error)")
        }

        return try ANEGenerationOutputHeadIO.argmaxSingleTokenLogits(
            from: outputSurface,
            vocabSize: vocabSize,
            laneSpatial: laneSpatial
        )
    }

    func selectArgmaxPair(
        normalizedInputA: borrowing TensorBuffer,
        normalizedInputB: borrowing TensorBuffer
    ) throws(GenerationError) -> (UInt16, UInt16) {
        try ANEGenerationOutputHeadIO.writeTokenPair(
            normalizedInputA,
            normalizedInputB,
            to: inputSurface,
            laneSpatial: laneSpatial
        )

        do {
            try kernelSet.classifier.eval()
        } catch {
            throw .runtimeFailure("ANE classifier pair eval failed: \(error)")
        }

        return try ANEGenerationOutputHeadIO.argmaxTokenPairLogits(
            from: outputSurface,
            vocabSize: vocabSize,
            laneSpatial: laneSpatial
        )
    }
}

final class ANEGenerationRMSNormClassifierHead {
    private static let defaultLaneSpatial = 32

    let kernelSet: GenerationRMSNormClassifierKernelSet
    let inputSurface: IOSurfaceRef
    let outputSurface: IOSurfaceRef
    /// Second output surface from the ANE reduce_max: [1, 1, 1, laneSpatial].
    let maxValSurface: IOSurfaceRef
    let vocabSize: Int
    let laneSpatial: Int

    init(
        rmsFinal: borrowing TensorBuffer,
        classifierWeights: borrowing TensorBuffer,
        vocabSize: Int,
        laneSpatial: Int = defaultLaneSpatial
    ) throws(GenerationError) {
        do {
            let kernelSet = try GenerationRMSNormClassifierKernelSet(
                rmsFinal: rmsFinal,
                classifier: classifierWeights,
                vocabSize: vocabSize,
                laneSpatial: laneSpatial
            )
            self.inputSurface = try kernelSet.rmsNormClassifier.inputSurface(at: 0)
            self.outputSurface = try kernelSet.rmsNormClassifier.outputSurface(at: 0)
            self.maxValSurface = try kernelSet.rmsNormClassifier.outputSurface(at: 1)
            self.kernelSet = kernelSet
            self.vocabSize = vocabSize
            self.laneSpatial = laneSpatial
            try ANEGenerationOutputHeadIO.initializeInputSurface(self.inputSurface, laneSpatial: laneSpatial)
        } catch {
            throw .runtimeFailure("ANE fused output-head setup failed: \(error)")
        }
    }

    func project(
        rawInput: borrowing TensorBuffer,
        logits: borrowing TensorBuffer
    ) throws(GenerationError) {
        precondition(rawInput.count == ModelConfig.dim)
        precondition(logits.count == vocabSize)

        try ANEGenerationOutputHeadIO.writeSingleToken(
            rawInput,
            to: inputSurface,
            laneSpatial: laneSpatial
        )

        do {
            try kernelSet.rmsNormClassifier.eval()
        } catch {
            throw .runtimeFailure("ANE fused output-head eval failed: \(error)")
        }

        try ANEGenerationOutputHeadIO.readSingleTokenLogits(
            from: outputSurface,
            into: logits,
            vocabSize: vocabSize,
            laneSpatial: laneSpatial
        )
    }

    func selectArgmax(
        rawInput: borrowing TensorBuffer
    ) throws(GenerationError) -> UInt16 {
        precondition(rawInput.count == ModelConfig.dim)

        try ANEGenerationOutputHeadIO.writeSingleToken(
            rawInput,
            to: inputSurface,
            laneSpatial: laneSpatial
        )

        do {
            try kernelSet.rmsNormClassifier.eval()
        } catch {
            throw .runtimeFailure("ANE fused output-head eval failed: \(error)")
        }

        return try ANEGenerationOutputHeadIO.argmaxSingleTokenLogitsWithHint(
            from: outputSurface,
            vocabSize: vocabSize,
            laneSpatial: laneSpatial,
            hintSurface: maxValSurface
        )
    }

    func selectArgmaxPair(
        rawInputA: borrowing TensorBuffer,
        rawInputB: borrowing TensorBuffer
    ) throws(GenerationError) -> (UInt16, UInt16) {
        try ANEGenerationOutputHeadIO.writeTokenPair(
            rawInputA,
            rawInputB,
            to: inputSurface,
            laneSpatial: laneSpatial
        )

        do {
            try kernelSet.rmsNormClassifier.eval()
        } catch {
            throw .runtimeFailure("ANE fused output-head pair eval failed: \(error)")
        }

        return try ANEGenerationOutputHeadIO.argmaxTokenPairLogitsWithHint(
            from: outputSurface,
            vocabSize: vocabSize,
            laneSpatial: laneSpatial,
            hintSurface: maxValSurface
        )
    }
}

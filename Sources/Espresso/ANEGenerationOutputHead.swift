import Foundation
import IOSurface
import ANERuntime
import ANETypes

public enum GenerationOutputHeadBackend: Sendable {
    case cpu
    case cpuExactStaged
    case cpuExactClustered
    case aneClassifier
    case aneRMSNormClassifier
}

private enum ANEGenerationOutputHeadIO {
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
}

final class ANEGenerationRMSNormClassifierHead {
    private static let defaultLaneSpatial = 32

    let kernelSet: GenerationRMSNormClassifierKernelSet
    let inputSurface: IOSurfaceRef
    let outputSurface: IOSurfaceRef
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

        return try ANEGenerationOutputHeadIO.argmaxSingleTokenLogits(
            from: outputSurface,
            vocabSize: vocabSize,
            laneSpatial: laneSpatial
        )
    }
}

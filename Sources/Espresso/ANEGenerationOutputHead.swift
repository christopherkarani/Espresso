import Foundation
import IOSurface
import ANERuntime
import ANETypes

public enum GenerationOutputHeadBackend: Sendable {
    case cpu
    case aneClassifier
}

final class ANEGenerationClassifierHead {
    private static let defaultLaneSpatial = 32

    let kernelSet: GenerationClassifierKernelSet
    let inputSurface: IOSurfaceRef
    let outputSurface: IOSurfaceRef
    let vocabSize: Int
    let laneSpatial: Int
    let zeroInput: TensorBuffer

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
            self.zeroInput = TensorBuffer(count: ModelConfig.dim * laneSpatial, zeroed: true)
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

        do {
            if laneSpatial == 1 {
                normalizedInput.withUnsafeBufferPointer { src in
                    SurfaceIO.writeFP16(to: inputSurface, data: src, channels: ModelConfig.dim, spatial: 1)
                }
            } else {
                zeroInput.withUnsafeBufferPointer { zeroPtr in
                    SurfaceIO.writeFP16(
                        to: inputSurface,
                        data: zeroPtr,
                        channels: ModelConfig.dim,
                        spatial: laneSpatial
                    )
                }
                try normalizedInput.withUnsafeBufferPointer { src in
                    try SurfaceIO.writeFP16SpatialSlice(
                        to: inputSurface,
                        channelOffset: 0,
                        spatialIndex: 0,
                        spatial: laneSpatial,
                        data: src,
                        channels: ModelConfig.dim
                    )
                }
            }
        } catch {
            throw .runtimeFailure("ANE classifier input write failed: \(error)")
        }

        do {
            try kernelSet.classifier.eval()
        } catch {
            throw .runtimeFailure("ANE classifier eval failed: \(error)")
        }

        do {
            if laneSpatial == 1 {
                logits.withUnsafeMutableBufferPointer { dst in
                    SurfaceIO.readFP16(
                        from: outputSurface,
                        into: dst,
                        channelOffset: 0,
                        channels: vocabSize,
                        spatial: 1
                    )
                }
            } else {
                try logits.withUnsafeMutableBufferPointer { dst in
                    try SurfaceIO.readFP16SpatialSlice(
                        from: outputSurface,
                        channelOffset: 0,
                        spatialIndex: 0,
                        spatial: laneSpatial,
                        into: dst,
                        channels: vocabSize
                    )
                }
            }
        } catch {
            throw .runtimeFailure("ANE classifier output read failed: \(error)")
        }
    }
}

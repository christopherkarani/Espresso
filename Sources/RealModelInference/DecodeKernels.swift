import Accelerate
import ANETypes
import Foundation

// Shared decode math and sampling primitives (extracted from
// RealModelInferenceEngine). Pure functions used by every trunk runtime;
// the engine extension keeps the call surface frozen.

extension RealModelInferenceEngine {
    static func multiplyRowMajorMatrix(
        matrix: [Float],
        rows: Int,
        cols: Int,
        vector: UnsafeBufferPointer<Float>,
        into output: UnsafeMutableBufferPointer<Float>
    ) {
        precondition(matrix.count == rows * cols)
        precondition(vector.count == cols)
        precondition(output.count == rows)
        matrix.withUnsafeBufferPointer { matrixBuffer in
            vDSP_mmul(
                matrixBuffer.baseAddress!,
                1,
                vector.baseAddress!,
                1,
                output.baseAddress!,
                1,
                vDSP_Length(rows),
                1,
                vDSP_Length(cols)
            )
        }
    }

    static func multiplyRowMajorMatrix(
        matrix: [Float],
        rows: Int,
        cols: Int,
        vector: [Float]
    ) -> [Float] {
        var output = [Float](repeating: 0, count: rows)
        output.withUnsafeMutableBufferPointer { outputBuffer in
            vector.withUnsafeBufferPointer { vectorBuffer in
                multiplyRowMajorMatrix(
                    matrix: matrix,
                    rows: rows,
                    cols: cols,
                    vector: vectorBuffer,
                    into: outputBuffer
                )
            }
        }
        return output
    }

    static func addBiasInPlace(_ bias: [Float], into output: UnsafeMutableBufferPointer<Float>) {
        precondition(bias.count == output.count)
        bias.withUnsafeBufferPointer { biasBuffer in
            vDSP_vadd(
                output.baseAddress!,
                1,
                biasBuffer.baseAddress!,
                1,
                output.baseAddress!,
                1,
                vDSP_Length(output.count)
            )
        }
    }

    /// Row-major `(out, in)` projection with an optional additive bias, matching a single
    /// `nn.Linear` on the PyTorch side and the conv+add pair the ANE kernel emits.
    static func projectRowMajorMatrix(
        matrix: [Float],
        rows: Int,
        cols: Int,
        vector: [Float],
        bias: [Float]?
    ) -> [Float] {
        var output = multiplyRowMajorMatrix(matrix: matrix, rows: rows, cols: cols, vector: vector)
        guard let bias else { return output }
        precondition(bias.count == rows)
        for index in 0..<rows {
            output[index] += bias[index]
        }
        return output
    }


    static func rmsNorm(_ input: [Float], weight: [Float], eps: Float) -> [Float] {
        precondition(input.count == weight.count)
        var normalized = [Float](repeating: 0, count: input.count)
        var sumSq: Float = 0
        input.withUnsafeBufferPointer { inputBuffer in
            vDSP_dotpr(inputBuffer.baseAddress!, 1, inputBuffer.baseAddress!, 1, &sumSq, vDSP_Length(input.count))
        }
        var invRms = 1.0 / sqrtf(sumSq / Float(input.count) + eps)
        input.withUnsafeBufferPointer { inputBuffer in
            normalized.withUnsafeMutableBufferPointer { normalizedBuffer in
                vDSP_vsmul(inputBuffer.baseAddress!, 1, &invRms, normalizedBuffer.baseAddress!, 1, vDSP_Length(input.count))
            }
        }
        weight.withUnsafeBufferPointer { weightBuffer in
            normalized.withUnsafeMutableBufferPointer { normalizedBuffer in
                vDSP_vmul(normalizedBuffer.baseAddress!, 1, weightBuffer.baseAddress!, 1, normalizedBuffer.baseAddress!, 1, vDSP_Length(input.count))
            }
        }
        return normalized
    }

    static func applyHalfSplitRoPEPerHead(
        _ input: [Float],
        heads: Int,
        headDim: Int,
        position: Int,
        theta: Float
    ) -> [Float] {
        precondition(input.count == heads * headDim)
        precondition(headDim % 2 == 0)
        let halfDim = headDim / 2
        var output = input
        for head in 0..<heads {
            let base = head * headDim
            for dimPair in 0..<halfDim {
                let frequency = 1.0 / pow(theta, Float(2 * dimPair) / Float(headDim))
                let angle = Float(position) * frequency
                let cosv = cos(angle)
                let sinv = sin(angle)
                let i0 = base + dimPair
                let i1 = base + dimPair + halfDim
                let v0 = output[i0]
                let v1 = output[i1]
                output[i0] = v0 * cosv - v1 * sinv
                output[i1] = v0 * sinv + v1 * cosv
            }
        }
        return output
    }


    static func silu(_ value: Float) -> Float {
        0.5 * value * (1 + tanh(0.5 * value))
    }


    static func partitionedArgmax(
        classifier: UnsafePointer<Float>,
        input: UnsafePointer<Float>,
        logitsScratch: UnsafeMutablePointer<Float>,
        blockMaxNorms: UnsafePointer<Float>,
        vocabSize: Int,
        dim: Int,
        blockSize: Int
    ) -> Int {
        var inputNormSquared: Float = 0
        vDSP_svesq(input, 1, &inputNormSquared, vDSP_Length(dim))
        let inputNorm = sqrtf(inputNormSquared)

        var bestIndex = 0
        var bestValue: Float = -.infinity
        var blockIndex = 0
        var blockStart = 0

        while blockStart < vocabSize {
            let blockEnd = min(blockStart + blockSize, vocabSize)
            let blockCount = blockEnd - blockStart

            if blockIndex > 0, bestValue > -.infinity {
                let upperBound = blockMaxNorms[blockIndex] * inputNorm
                if upperBound < bestValue {
                    blockIndex += 1
                    blockStart = blockEnd
                    continue
                }
            }

            vDSP_mmul(
                classifier.advanced(by: blockStart * dim),
                1,
                input,
                1,
                logitsScratch,
                1,
                vDSP_Length(blockCount),
                1,
                vDSP_Length(dim)
            )

            var blockMaxValue: Float = 0
            var blockMaxIndex: vDSP_Length = 0
            vDSP_maxvi(logitsScratch, 1, &blockMaxValue, &blockMaxIndex, vDSP_Length(blockCount))
            if blockMaxValue > bestValue {
                bestValue = blockMaxValue
                bestIndex = blockStart + Int(blockMaxIndex)
            }

            blockIndex += 1
            blockStart = blockEnd
        }

        return bestIndex
    }


    static func evaluateGreedyClassifier(
        norm: borrowing CompiledHead,
        classifier: borrowing CompiledClassifier,
        headSpatial: Int,
        vocab: Int
    ) throws -> TokenID {
        do {
            try norm.kernel.eval()
            try classifier.kernel.eval()
            let argmax = try greedyArgmax(
                classifier: classifier,
                headSpatial: headSpatial,
                vocab: vocab
            )
            guard let token = TokenID(exactly: argmax.index) else {
                throw RealModelInferenceError.runtimeFailure(
                    "Greedy ANE classifier selected out-of-range token \(argmax.index)"
                )
            }
            return token
        } catch let error as RealModelInferenceError {
            throw error
        } catch {
            throw RealModelInferenceError.runtimeFailure("Hybrid greedy ANE head evaluation failed: \(error)")
        }
    }


    static func greedyArgmax(
        classifier: borrowing CompiledClassifier,
        headSpatial: Int,
        vocab: Int
    ) throws -> SurfaceIO.FP16ArgmaxResult {
        if let maxValueSurface = classifier.maxValueSurface {
            return try SurfaceIO.argmaxFP16SpatialSliceWithHint(
                from: classifier.outputSurface,
                channelOffset: 0,
                spatialIndex: 0,
                spatial: headSpatial,
                channels: vocab,
                hintSurface: maxValueSurface,
                hintSpatialIndex: 0,
                hintSpatial: headSpatial
            )
        }
        return try SurfaceIO.argmaxFP16SpatialSlice(
            from: classifier.outputSurface,
            channelOffset: 0,
            spatialIndex: 0,
            spatial: headSpatial,
            channels: vocab
        )
    }

}

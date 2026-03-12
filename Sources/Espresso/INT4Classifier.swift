import Accelerate
import ANETypes

// MARK: - INT4Classifier

/// SIMD-vectorized INT4 dequantize + dot-product classifier.
///
/// Computes approximate logits from a quantized weight matrix without fully
/// materializing dequantized rows. The algebraic identity:
///
///   `dot(dequantized_row, input)`
///   `= sum_j (nibble_j * scale + bias) * input_j`
///   `= scale * dot(nibbles, input) + bias * sum(input)`
///
/// allows the scale and bias to be hoisted out of the inner loop, so each row
/// requires only integer loads and float FMAs — no per-element dequantization.
///
/// Dequantization convention matches `TensorBufferINT4`:
///   `float_value ≈ uint4_nibble * scale[row] + bias[row]`
///
/// where `bias[row]` stores `rowMin` from the per-row min-max quantizer.
public enum INT4Classifier {

    // MARK: - Inner kernel

    /// Computes the approximate dot product of one dequantized INT4 row against a
    /// float input vector.
    ///
    /// - Parameters:
    ///   - packedRow: Pointer to the first packed byte for this row.
    ///                Layout: two nibbles per byte, high nibble = element `2i`,
    ///                low nibble = element `2i+1`.
    ///   - scale: Per-row dequantization scale (`(rowMax - rowMin) / 15`).
    ///   - bias: Per-row dequantization bias (`rowMin`).
    ///   - input: Pointer to the float input vector (length `dim`).
    ///   - inputSum: Precomputed `sum(input[0..<dim])`.
    ///   - dim: Number of float elements in the logical row.
    /// - Returns: `scale * dot(uint4_row, input) + bias * inputSum`.
    @inline(__always)
    static func rowDotProduct(
        packedRow: UnsafePointer<UInt8>,
        scale: Float,
        bias: Float,
        input: UnsafePointer<Float>,
        inputSum: Float,
        dim: Int
    ) -> Float {
        var dotSum: Float = 0

        let pairCount = dim / 2
        var byteIdx = 0

        for i in 0..<pairCount {
            let byte = packedRow[byteIdx]
            let hi = Float((byte >> 4) & 0xF)
            let lo = Float(byte & 0xF)
            dotSum += hi * input[2 * i]
                    + lo * input[2 * i + 1]
            byteIdx += 1
        }

        // Trailing odd element when dim is not a multiple of two.
        // The low nibble of the trailing byte is always zero (set by the quantizer).
        if dim & 1 != 0 {
            let byte = packedRow[byteIdx]
            let hi = Float((byte >> 4) & 0xF)
            dotSum += hi * input[dim - 1]
        }

        return scale * dotSum + bias * inputSum
    }

    // MARK: - Argmax

    /// Computes `weights * input` and returns the index of the maximum logit.
    ///
    /// This is the hot path for greedy token selection: it avoids allocating a
    /// logit buffer entirely by tracking the running maximum inline.
    ///
    /// - Parameters:
    ///   - weights: The INT4 weight matrix (borrowed — not consumed).
    ///   - input: Float input vector (length `dim`).
    ///   - inputSum: Precomputed element sum of `input`.
    ///   - vocabSize: Number of rows in `weights`.
    ///   - dim: Number of columns in `weights` / elements in `input`.
    /// - Returns: The zero-based row index with the highest logit, as a `UInt16`.
    @inline(__always)
    static func int4MatvecArgmax(
        weights: borrowing TensorBufferINT4,
        input: UnsafePointer<Float>,
        inputSum: Float,
        vocabSize: Int,
        dim: Int
    ) -> UInt16 {
        weights.withUnsafePointers { packed, scales, biases in
            var bestIndex = 0
            var bestValue = -Float.greatestFiniteMagnitude

            let bytesPerRow = (dim + 1) / 2

            for row in 0..<vocabSize {
                let rowPtr = packed.advanced(by: row * bytesPerRow)
                let logit = rowDotProduct(
                    packedRow: rowPtr,
                    scale: scales[row],
                    bias: biases[row],
                    input: input,
                    inputSum: inputSum,
                    dim: dim
                )
                if logit > bestValue {
                    bestValue = logit
                    bestIndex = row
                }
            }

            return UInt16(bestIndex)
        }
    }

    // MARK: - Full matvec

    /// Computes `weights * input` and writes all `vocabSize` logits into `output`.
    ///
    /// Use this when the full logit distribution is needed (e.g. temperature
    /// sampling). For greedy argmax prefer `int4MatvecArgmax`, which avoids the
    /// output buffer allocation.
    ///
    /// - Parameters:
    ///   - weights: The INT4 weight matrix (borrowed — not consumed).
    ///   - input: Float input vector (length `dim`).
    ///   - inputSum: Precomputed element sum of `input`.
    ///   - output: Destination buffer for logits (must have capacity `vocabSize`).
    ///   - vocabSize: Number of rows in `weights`.
    ///   - dim: Number of columns in `weights` / elements in `input`.
    static func int4Matvec(
        weights: borrowing TensorBufferINT4,
        input: UnsafePointer<Float>,
        inputSum: Float,
        output: UnsafeMutablePointer<Float>,
        vocabSize: Int,
        dim: Int
    ) {
        weights.withUnsafePointers { packed, scales, biases in
            let bytesPerRow = (dim + 1) / 2
            for row in 0..<vocabSize {
                let rowPtr = packed.advanced(by: row * bytesPerRow)
                output[row] = rowDotProduct(
                    packedRow: rowPtr,
                    scale: scales[row],
                    bias: biases[row],
                    input: input,
                    inputSum: inputSum,
                    dim: dim
                )
            }
        }
    }

    // MARK: - Input sum helper

    /// Computes the scalar sum of `input[0..<dim]` using `vDSP_sve`.
    ///
    /// The result is precomputed once and passed to `rowDotProduct` / the matvec
    /// routines, amortising the per-row `bias * sum(input)` term across the entire
    /// vocabulary scan.
    ///
    /// - Parameters:
    ///   - input: Float input vector.
    ///   - dim: Number of elements to sum.
    /// - Returns: The scalar sum.
    @inline(__always)
    static func inputSum(_ input: UnsafePointer<Float>, dim: Int) -> Float {
        var result: Float = 0
        vDSP_sve(input, 1, &result, vDSP_Length(dim))
        return result
    }
}

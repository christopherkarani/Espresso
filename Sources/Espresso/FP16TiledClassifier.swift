import Accelerate

/// FP16 tiled classifier: converts FP16 weights in L2-sized tiles, runs sgemm on L2-resident FP32 data.
///
/// Each tile of `tileRows × dim` FP16 values (~11.7 MB at 4000×768) fits in L2 cache.
/// The tile is converted to FP32 via vImageConvert_Planar16FtoPlanarF, then sgemm runs
/// on the warm FP32 data. This halves the DRAM bandwidth for classifier weights.
public enum FP16TiledClassifier {

    /// Default tile size: 4000 rows × 768 cols = 3.07M elements × 2 bytes = ~5.9 MB FP16
    /// The FP32 conversion buffer is 3.07M × 4 = ~11.7 MB — fits in L2 cache.
    public static let tileRows: Int = 4_000

    /// Compute FP16 tiled matmul argmax: [vocabSize × dim] FP16 × [dim × 1] FP32 → argmax token.
    ///
    /// - Parameters:
    ///   - weights: Pointer to `vocabSize * dim` packed Float16 values as UInt16.
    ///   - input: Pointer to `dim` FP32 input values (already RMSNorm'd).
    ///   - vocabSize: Number of rows in the weight matrix.
    ///   - dim: Number of columns (embedding dimension).
    ///   - tileRows: Number of rows to process per tile (default 4000).
    /// - Returns: The token index of the highest logit.
    @inline(__always)
    public static func tiledMatvecArgmax(
        weights: UnsafePointer<UInt16>,
        input: UnsafePointer<Float>,
        vocabSize: Int,
        dim: Int,
        tileRows: Int = Self.tileRows
    ) -> Int {
        precondition(vocabSize > 0)
        precondition(dim > 0)
        precondition(tileRows > 0)

        // Allocate tile conversion buffer and logits scratch
        let tileFP32 = UnsafeMutablePointer<Float>.allocate(capacity: tileRows * dim)
        let tileLogits = UnsafeMutablePointer<Float>.allocate(capacity: tileRows)
        defer {
            tileFP32.deallocate()
            tileLogits.deallocate()
        }

        var bestIndex: Int = 0
        var bestValue: Float = -.greatestFiniteMagnitude

        var rowStart = 0
        while rowStart < vocabSize {
            let rowEnd = min(rowStart + tileRows, vocabSize)
            let blockCount = rowEnd - rowStart
            let elementCount = blockCount * dim

            // Convert FP16 → FP32 for this tile using vImage
            let fp16Ptr = weights.advanced(by: rowStart * dim)
            var srcBuf = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: fp16Ptr),
                height: 1,
                width: vImagePixelCount(elementCount),
                rowBytes: elementCount * MemoryLayout<UInt16>.stride
            )
            var dstBuf = vImage_Buffer(
                data: UnsafeMutableRawPointer(tileFP32),
                height: 1,
                width: vImagePixelCount(elementCount),
                rowBytes: elementCount * MemoryLayout<Float>.stride
            )
            vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)

            // Compute logits via sgemm: [blockCount × dim] × [dim × 1]
            BLAS.sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                m: Int32(blockCount),
                n: 1,
                k: Int32(dim),
                alpha: 1.0,
                a: UnsafePointer(tileFP32),
                lda: Int32(dim),
                b: input,
                ldb: 1,
                beta: 0.0,
                c: tileLogits,
                ldc: 1
            )

            // Find best in this tile
            var tileMax: Float = 0
            var tileMaxIdx: vDSP_Length = 0
            vDSP_maxvi(tileLogits, 1, &tileMax, &tileMaxIdx, vDSP_Length(blockCount))

            if tileMax > bestValue {
                bestValue = tileMax
                bestIndex = rowStart + Int(tileMaxIdx)
            }

            rowStart = rowEnd
        }

        return bestIndex
    }
}

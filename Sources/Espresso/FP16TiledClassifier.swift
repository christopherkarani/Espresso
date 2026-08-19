import Accelerate
import ANEInterop
import Dispatch

/// FP16 tiled classifier: converts FP16 weights in L2-sized tiles, runs sgemm on L2-resident FP32 data.
///
/// Each tile of `tileRows × dim` FP16 values (~11.7 MB at 4000×768) fits in L2 cache.
/// The tile is converted to FP32 via vImageConvert_Planar16FtoPlanarF, then sgemm runs
/// on the warm FP32 data. This halves the DRAM bandwidth for classifier weights.
///
/// At dim=1536 the default 4000-row FP32 tile is ~25 MB and misses L2. Prefer
/// `streamingMatvecArgmax` (register convert, no tile store) or a reused
/// `TileScratch` sized with `l2TileRows(forDim:)`.
public enum FP16TiledClassifier {

    /// Default tile size: 4000 rows × 768 cols = 3.07M elements × 2 bytes = ~5.9 MB FP16
    /// The FP32 conversion buffer is 3.07M × 4 = ~11.7 MB — fits in L2 cache.
    public static let tileRows: Int = 4_000

    /// Target FP32 tile bytes so convert+sgemm stays in a typical P-core L2.
    public static let l2TargetFP32Bytes: Int = 4 * 1024 * 1024

    /// Reusable convert+logit scratch for the tiled path. Avoids a per-token allocate
    /// of `tileRows × dim` FP32 (~25 MB at 4000×1536).
    public final class TileScratch: @unchecked Sendable {
        public let tileFP32: UnsafeMutablePointer<Float>
        public let tileLogits: UnsafeMutablePointer<Float>
        public let tileRows: Int
        public let dim: Int

        public init(tileRows: Int, dim: Int) {
            precondition(tileRows > 0)
            precondition(dim > 0)
            self.tileRows = tileRows
            self.dim = dim
            self.tileFP32 = UnsafeMutablePointer<Float>.allocate(capacity: tileRows * dim)
            self.tileLogits = UnsafeMutablePointer<Float>.allocate(capacity: tileRows)
        }

        public func deallocate() {
            tileFP32.deallocate()
            tileLogits.deallocate()
        }
    }

    /// Tile row count whose FP32 conversion buffer is about `l2TargetFP32Bytes`.
    public static func l2TileRows(forDim dim: Int) -> Int {
        precondition(dim > 0)
        return max(1, l2TargetFP32Bytes / (dim * MemoryLayout<Float>.stride))
    }

    /// Compute FP16 tiled matmul argmax: [vocabSize × dim] FP16 × [dim × 1] FP32 → argmax token.
    ///
    /// Allocates a fresh tile buffer on every call (the shipped 1.5B path).
    @inline(__always)
    public static func tiledMatvecArgmax(
        weights: UnsafePointer<UInt16>,
        input: UnsafePointer<Float>,
        vocabSize: Int,
        dim: Int,
        tileRows: Int = Self.tileRows
    ) -> Int {
        let scratch = TileScratch(tileRows: tileRows, dim: dim)
        defer { scratch.deallocate() }
        return tiledMatvecArgmax(
            weights: weights,
            input: input,
            vocabSize: vocabSize,
            dim: dim,
            tileRows: tileRows,
            scratch: scratch
        )
    }

    /// Same convert+sgemm algorithm as `tiledMatvecArgmax`, with a reused tile buffer.
    @inline(__always)
    public static func tiledMatvecArgmax(
        weights: UnsafePointer<UInt16>,
        input: UnsafePointer<Float>,
        vocabSize: Int,
        dim: Int,
        tileRows: Int,
        scratch: TileScratch
    ) -> Int {
        precondition(vocabSize > 0)
        precondition(dim > 0)
        precondition(tileRows > 0)
        precondition(scratch.dim == dim)
        precondition(scratch.tileRows >= min(tileRows, vocabSize))

        let tileFP32 = scratch.tileFP32
        let tileLogits = scratch.tileLogits
        let step = min(tileRows, scratch.tileRows)

        var bestIndex: Int = 0
        var bestValue: Float = -.greatestFiniteMagnitude

        var rowStart = 0
        while rowStart < vocabSize {
            let rowEnd = min(rowStart + step, vocabSize)
            let blockCount = rowEnd - rowStart
            let elementCount = blockCount * dim

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

    /// Native FP16 GEMV argmax: convert weights in registers, never store an FP32 tile.
    /// Same first-max rule as the tiled path. Exact-argmax when FP16→FP32 is lossless
    /// (it is) and no two logits sit inside FMA-association noise.
    @inline(__always)
    public static func streamingMatvecArgmax(
        weights: UnsafePointer<UInt16>,
        input: UnsafePointer<Float>,
        vocabSize: Int,
        dim: Int
    ) -> Int {
        precondition(vocabSize > 0)
        precondition(dim > 0)
        return Int(
            ane_interop_fp16_gemv_argmax(
                UnsafeRawPointer(weights),
                input,
                Int32(vocabSize),
                Int32(dim)
            )
        )
    }

    /// Isolated BNNS FP16 8-shard GEMV argmax. Same first-max rule as tiled/streaming.
    public static let eightShardCount = 8

    public static func shardRowRange(vocabSize: Int, shard: Int, shardCount: Int = eightShardCount) -> Range<Int> {
        precondition(vocabSize > 0)
        precondition(shardCount > 0)
        precondition(shard >= 0 && shard < shardCount)
        let base = vocabSize / shardCount
        let rem = vocabSize % shardCount
        let start = shard * base + min(shard, rem)
        let count = base + (shard < rem ? 1 : 0)
        return start..<(start + count)
    }

    @inline(__always)
    public static func bnnsEightShardMatvecArgmax(
        weights: UnsafePointer<UInt16>,
        input: UnsafePointer<Float>,
        vocabSize: Int,
        dim: Int
    ) -> Int {
        precondition(vocabSize > 0)
        precondition(dim > 0)
        let shardCount = eightShardCount
        let shardMax = UnsafeMutablePointer<Float>.allocate(capacity: shardCount)
        let shardIdx = UnsafeMutablePointer<Int>.allocate(capacity: shardCount)
        defer {
            shardMax.deallocate()
            shardIdx.deallocate()
        }
        for shard in 0..<shardCount {
            shardMax[shard] = -.greatestFiniteMagnitude
            shardIdx[shard] = 0
        }

        nonisolated(unsafe) let weightsPtr = weights
        nonisolated(unsafe) let inputPtr = input
        nonisolated(unsafe) let shardMaxPtr = shardMax
        nonisolated(unsafe) let shardIdxPtr = shardIdx
        DispatchQueue.concurrentPerform(iterations: shardCount) { shard in
            let range = shardRowRange(vocabSize: vocabSize, shard: shard, shardCount: shardCount)
            guard !range.isEmpty else { return }
            _ = ane_interop_amx_shared_resource_hint(1, Int32(shard), 2)
            defer { _ = ane_interop_amx_shared_resource_hint(0, Int32(shard), 2) }

            let rows = range.count
            let tileFP32 = UnsafeMutablePointer<Float>.allocate(capacity: rows * dim)
            let logits = UnsafeMutablePointer<Float>.allocate(capacity: rows)
            defer {
                tileFP32.deallocate()
                logits.deallocate()
            }

            let fp16Ptr = UnsafeMutableRawPointer(mutating: weightsPtr.advanced(by: range.lowerBound * dim))
            var srcBuf = vImage_Buffer(
                data: fp16Ptr,
                height: 1,
                width: vImagePixelCount(rows * dim),
                rowBytes: rows * dim * MemoryLayout<UInt16>.stride
            )
            var dstBuf = vImage_Buffer(
                data: UnsafeMutableRawPointer(tileFP32),
                height: 1,
                width: vImagePixelCount(rows * dim),
                rowBytes: rows * dim * MemoryLayout<Float>.stride
            )
            vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)

            let bnnsRC = ane_interop_bnns_fp32_gemv(
                UnsafePointer(tileFP32),
                inputPtr,
                logits,
                Int32(rows),
                Int32(dim),
                1
            )
            if bnnsRC != 0 {
                BLAS.sgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m: Int32(rows),
                    n: 1,
                    k: Int32(dim),
                    alpha: 1.0,
                    a: UnsafePointer(tileFP32),
                    lda: Int32(dim),
                    b: inputPtr,
                    ldb: 1,
                    beta: 0.0,
                    c: logits,
                    ldc: 1
                )
            }

            var tileMax: Float = 0
            var tileMaxIdx: vDSP_Length = 0
            vDSP_maxvi(logits, 1, &tileMax, &tileMaxIdx, vDSP_Length(rows))
            shardMaxPtr[shard] = tileMax
            shardIdxPtr[shard] = range.lowerBound + Int(tileMaxIdx)
        }

        var bestValue: Float = -.greatestFiniteMagnitude
        var bestIndex = 0
        for shard in 0..<shardCount {
            if shardMax[shard] > bestValue {
                bestValue = shardMax[shard]
                bestIndex = shardIdx[shard]
            }
        }
        return bestIndex
    }
}

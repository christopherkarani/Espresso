import Darwin
import Foundation

/// A read-only buffer of packed INT4 values with per-row min-max dequantization metadata.
///
/// ## Layout
/// Each row of `cols` elements is packed as `(cols + 1) / 2` bytes, with two 4-bit
/// unsigned values per byte (high nibble first, i.e. the element at column `2i` is stored
/// in the upper nibble and column `2i+1` in the lower nibble).
///
/// ## Dequantization
/// Given the per-row `scale` and `bias` (which stores `rowMin`):
/// ```
/// float_value ≈ uint4_value * scale + rowMin
/// ```
/// where `scale = (rowMax - rowMin) / 15`. The maximum absolute reconstruction error is
/// `scale / 2` per element.
public struct TensorBufferINT4: ~Copyable {
    public let rows: Int
    public let cols: Int

    /// Bytes per packed row: `(cols + 1) / 2`.
    public var bytesPerRow: Int { (cols + 1) / 2 }

    // Packed nibble data: `rows * bytesPerRow` bytes.
    private let packedStorage: UnsafeMutableRawPointer
    private let packedBase: UnsafePointer<UInt8>

    // Per-row scale factors: `rows` Float32 values, where `scale = (rowMax - rowMin) / 15`.
    private let scaleStorage: UnsafeMutableRawPointer
    private let scaleBase: UnsafePointer<Float>

    // Per-row bias values: `rows` Float32 values storing `rowMin`.
    // Dequant formula: `float_value ≈ uint4_value * scale + bias`.
    private let biasStorage: UnsafeMutableRawPointer
    private let biasBase: UnsafePointer<Float>

    // MARK: - Initializer

    /// Creates an INT4 quantized buffer from an FP32 source using per-row min-max quantization.
    ///
    /// For each row, computes `rowMin` and `rowMax`, derives:
    /// - `scale = (rowMax - rowMin) / 15`
    /// - Each element is quantized as `clamp(round((v - rowMin) / scale), 0, 15)`
    ///
    /// Constant rows (range < 1e-10) use `scale = 1.0` and `bias = -rowMin` to avoid
    /// division by zero.
    ///
    /// - Parameters:
    ///   - source: The FP32 source buffer. Must contain exactly `rows * cols` elements.
    ///   - rows: Number of rows in the weight matrix.
    ///   - cols: Number of columns in the weight matrix.
    public init(quantizing source: borrowing TensorBuffer, rows: Int, cols: Int) {
        precondition(rows > 0 && cols > 0)
        precondition(rows * cols == source.count)

        self.rows = rows
        self.cols = cols
        let bpr = (cols + 1) / 2

        // Packed nibble buffer
        let packedByteCount = rows * bpr
        let packedRaw = UnsafeMutableRawPointer.allocate(
            byteCount: max(packedByteCount, 1),
            alignment: TensorBuffer.allocationAlignment
        )
        let packedPtr = packedRaw.bindMemory(to: UInt8.self, capacity: packedByteCount)
        // Zero-initialize so trailing odd nibbles are deterministic
        packedRaw.initializeMemory(as: UInt8.self, repeating: 0, count: packedByteCount)

        // Scale buffer
        let floatStride = MemoryLayout<Float>.stride
        let scaleRaw = UnsafeMutableRawPointer.allocate(
            byteCount: rows * floatStride,
            alignment: TensorBuffer.allocationAlignment
        )
        let scalePtr = scaleRaw.bindMemory(to: Float.self, capacity: rows)

        // Bias buffer (stores rowMin)
        let biasRaw = UnsafeMutableRawPointer.allocate(
            byteCount: rows * floatStride,
            alignment: TensorBuffer.allocationAlignment
        )
        let biasPtr = biasRaw.bindMemory(to: Float.self, capacity: rows)

        source.withUnsafePointer { srcPtr in
            for r in 0..<rows {
                let rowBase = r * cols

                // Compute row min/max
                var rowMin = srcPtr[rowBase]
                var rowMax = srcPtr[rowBase]
                for c in 1..<cols {
                    let v = srcPtr[rowBase + c]
                    if v < rowMin { rowMin = v }
                    if v > rowMax { rowMax = v }
                }

                let range = rowMax - rowMin
                let scale: Float
                if range < 1e-10 {
                    // Constant row: use scale=1 to avoid division by zero;
                    // all elements will quantize to 0 and dequantize back to rowMin.
                    scale = 1.0
                } else {
                    scale = range / 15.0
                }
                scalePtr[r] = scale
                biasPtr[r] = rowMin  // dequant bias = rowMin

                // Pack nibbles
                let packedRowBase = r * bpr
                var byteIdx = 0
                var c = 0
                while c + 1 < cols {
                    let qHi = min(15, max(0, Int(roundf((srcPtr[rowBase + c] - rowMin) / scale))))
                    let qLo = min(15, max(0, Int(roundf((srcPtr[rowBase + c + 1] - rowMin) / scale))))
                    packedPtr[packedRowBase + byteIdx] = UInt8((qHi << 4) | qLo)
                    byteIdx += 1
                    c += 2
                }
                // Handle odd trailing element (low nibble already 0 from initialisation)
                if c < cols {
                    let qHi = min(15, max(0, Int(roundf((srcPtr[rowBase + c] - rowMin) / scale))))
                    packedPtr[packedRowBase + byteIdx] = UInt8(qHi << 4)
                }
            }
        }

        self.packedStorage = packedRaw
        self.packedBase = UnsafePointer(packedPtr)
        self.scaleStorage = scaleRaw
        self.scaleBase = UnsafePointer(scalePtr)
        self.biasStorage = biasRaw
        self.biasBase = UnsafePointer(biasPtr)
    }

    // MARK: - Deinit

    deinit {
        packedStorage.deallocate()
        scaleStorage.deallocate()
        biasStorage.deallocate()
    }

    // MARK: - Accessors

    /// Calls `body` with read-only pointers to the packed nibble data, per-row scales,
    /// and per-row biases (row minimums).
    ///
    /// Dequantization: `float_value = uint4_nibble * scale[row] + bias[row]`
    @inline(__always)
    public func withUnsafePointers<R>(
        _ body: (UnsafePointer<UInt8>, UnsafePointer<Float>, UnsafePointer<Float>) throws -> R
    ) rethrows -> R {
        try body(packedBase, scaleBase, biasBase)
    }
}

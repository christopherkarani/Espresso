import Darwin
import Foundation

/// A read-only buffer of packed Float16 values (stored as UInt16 bit patterns),
/// typically used for tiled FP16 classifier weights.
///
/// Supports two storage modes:
/// - `owned`: heap-allocated memory managed by this instance
/// - `mmapped`: memory-mapped from a file; released via `munmap` on deinit
public struct TensorBufferFP16: ~Copyable {
    public let rows: Int
    public let cols: Int

    private let rawStorage: UnsafeMutableRawPointer
    private let baseAddress: UnsafePointer<UInt16>
    private let storageKind: StorageKind

    private enum StorageKind {
        case owned
        case mmapped(mappedSize: Int)
    }

    // MARK: - Initializers

    /// Creates an FP16 buffer by quantizing an FP32 source buffer.
    ///
    /// Each Float32 element is converted to IEEE 754 Float16 (round-to-nearest,
    /// flush-to-zero for subnormals). The resulting buffer owns its memory.
    ///
    /// - Parameters:
    ///   - source: The FP32 source buffer. Must contain exactly `rows * cols` elements.
    ///   - rows: Number of rows in the weight matrix.
    ///   - cols: Number of columns in the weight matrix.
    public init(quantizing source: borrowing TensorBuffer, rows: Int, cols: Int) {
        precondition(rows >= 0 && cols >= 0)
        precondition(rows * cols == source.count)

        self.rows = rows
        self.cols = cols
        let elementCount = rows * cols
        let byteCount = elementCount * MemoryLayout<UInt16>.stride

        let raw = UnsafeMutableRawPointer.allocate(
            byteCount: max(byteCount, 1),
            alignment: TensorBuffer.allocationAlignment
        )
        let fp16Ptr = raw.bindMemory(to: UInt16.self, capacity: elementCount)

        source.withUnsafePointer { srcPtr in
            for i in 0..<elementCount {
                fp16Ptr[i] = floatToFP16(srcPtr[i])
            }
        }

        self.rawStorage = raw
        self.baseAddress = UnsafePointer(fp16Ptr)
        self.storageKind = .owned
    }

    // MARK: - Deinit

    deinit {
        switch storageKind {
        case .owned:
            rawStorage.deallocate()
        case .mmapped(let mappedSize):
            _ = Darwin.munmap(rawStorage, mappedSize)
        }
    }

    // MARK: - Accessors

    /// Provides read-only access to the raw FP16 data as UInt16 bit patterns.
    @inline(__always)
    public func withUnsafePointer<R>(_ body: (UnsafePointer<UInt16>) throws -> R) rethrows -> R {
        try body(baseAddress)
    }

    /// Returns a pointer to the start of row `row`.
    ///
    /// Each row contains `cols` UInt16 elements (FP16 bit patterns).
    ///
    /// - Parameter row: Zero-based row index; must be in `0..<rows`.
    @inline(__always)
    public func rowPointer(row: Int) -> UnsafePointer<UInt16> {
        precondition(row >= 0 && row < rows)
        return baseAddress.advanced(by: row * cols)
    }
}

// MARK: - FP32 → FP16 conversion

/// Converts a single Float32 value to an IEEE 754 Float16 bit pattern (as UInt16).
///
/// Behavior matches vImageConvert_PlanarFtoPlanar16F:
/// - Values outside ±65504 saturate to ±infinity.
/// - Subnormal FP32 values with exponent < -14 flush to ±zero.
/// - No half-precision subnormal generation (flush-to-zero mode).
@inline(__always)
private func floatToFP16(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    let sign = UInt16((bits >> 16) & 0x8000)
    let exponent = Int((bits >> 23) & 0xFF) - 127

    if exponent > 15 {
        // Overflow → signed infinity
        return sign | 0x7C00
    } else if exponent < -14 {
        // Underflow → signed zero (flush-to-zero)
        return sign
    } else {
        let fp16Exp = UInt16(UInt32(exponent + 15) << 10)
        let fp16Mantissa = UInt16((bits & 0x007F_FFFF) >> 13)
        return sign | fp16Exp | fp16Mantissa
    }
}

import Foundation
import ANEInterop

public enum WeightBlob {
    @inline(__always)
    private static func writeHeader(_ raw: UnsafeMutableRawBufferPointer, payloadBytes: Int) {
        precondition(payloadBytes >= 0 && payloadBytes <= Int(UInt32.max))
        let base = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
        base[0] = 1
        base[4] = 2
        base[64] = 0xEF
        base[65] = 0xBE
        base[66] = 0xAD
        base[67] = 0xDE
        base[68] = 1
        raw.storeBytes(of: UInt32(payloadBytes).littleEndian, toByteOffset: 72, as: UInt32.self)
        raw.storeBytes(of: UInt32(128).littleEndian, toByteOffset: 80, as: UInt32.self)
    }

    public static func build(from weights: UnsafeBufferPointer<Float>, rows: Int, cols: Int) -> Data {
        precondition(rows >= 0 && cols >= 0)
        precondition(weights.count == rows * cols)

        let weightBytes = weights.count * 2
        let total = 128 + weightBytes
        var data = Data(count: total)
        data.withUnsafeMutableBytes { raw in
            writeHeader(raw, payloadBytes: weightBytes)
            if weightBytes == 0 { return }

            let payload = raw.baseAddress!.advanced(by: 128)
            guard let src = weights.baseAddress else {
                preconditionFailure("Weights base address is nil for non-empty buffer")
            }
            ane_interop_cvt_f32_to_f16(payload, src, Int32(weights.count))
        }
        return data
    }

    public static func build(from weights: [Float], rows: Int, cols: Int) -> Data {
        weights.withUnsafeBufferPointer { w in
            build(from: w, rows: rows, cols: cols)
        }
    }

    public static func buildTransposed(from weights: UnsafeBufferPointer<Float>, rows: Int, cols: Int) -> Data {
        precondition(rows >= 0 && cols >= 0)
        precondition(weights.count == rows * cols)

        let count = weights.count
        let weightBytes = count * 2
        let total = 128 + weightBytes
        var data = Data(count: total)
        data.withUnsafeMutableBytes { raw in
            writeHeader(raw, payloadBytes: weightBytes)

            let payload = raw.baseAddress!.advanced(by: 128).assumingMemoryBound(to: UInt16.self)
            for i in 0..<rows {
                for j in 0..<cols {
                    let f16 = Float16(weights[i * cols + j])
                    payload[j * rows + i] = f16.bitPattern
                }
            }
        }
        return data
    }

    public static func buildFP16(from weights: UnsafeBufferPointer<UInt16>) -> Data {
        let weightBytes = weights.count * MemoryLayout<UInt16>.stride
        let total = 128 + weightBytes
        var data = Data(count: total)
        data.withUnsafeMutableBytes { raw in
            writeHeader(raw, payloadBytes: weightBytes)
            if weightBytes == 0 { return }

            let payload = raw.baseAddress!.advanced(by: 128).assumingMemoryBound(to: UInt16.self)
            payload.update(from: weights.baseAddress!, count: weights.count)
        }
        return data
    }

    public static func buildFP16(from weights: [UInt16]) -> Data {
        weights.withUnsafeBufferPointer { w in
            buildFP16(from: w)
        }
    }

    public static func buildTransposed(from weights: [Float], rows: Int, cols: Int) -> Data {
        weights.withUnsafeBufferPointer { w in
            buildTransposed(from: w, rows: rows, cols: cols)
        }
    }
}

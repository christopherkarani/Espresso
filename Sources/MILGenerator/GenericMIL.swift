import Foundation

public enum GenericMILError: Error, Equatable {
    case invalidWeightCount(expected: Int, got: Int)
    case invalidShape
    case sizeOverflow
}

public struct GenericMILContract: Sendable, Equatable {
    public let inputBytes: Int
    public let outputByteSizes: [Int]

    public init(inputBytes: Int, outputByteSizes: [Int]) {
        self.inputBytes = inputBytes
        self.outputByteSizes = outputByteSizes
    }
}

public enum GenericMIL {
    @inline(__always)
    private static func checkedMultiply(_ lhs: Int, _ rhs: Int) throws(GenericMILError) -> Int {
        let product = lhs.multipliedReportingOverflow(by: rhs)
        guard !product.overflow else { throw .sizeOverflow }
        return product.partialValue
    }

    @inline(__always)
    private static func checkedAdd(_ lhs: Int, _ rhs: Int) throws(GenericMILError) -> Int {
        let sum = lhs.addingReportingOverflow(rhs)
        guard !sum.overflow else { throw .sizeOverflow }
        return sum.partialValue
    }

    @inline(__always)
    private static func checkedUInt32(_ value: Int) throws(GenericMILError) -> UInt32 {
        guard value >= 0, value <= Int(UInt32.max) else { throw .sizeOverflow }
        return UInt32(value)
    }

    private static func checkedSquareCount(_ dim: Int) throws(GenericMILError) -> Int {
        guard dim > 0 else { throw .invalidShape }
        return try checkedMultiply(dim, dim)
    }

    private static func checkedRectCount(rows: Int, cols: Int) throws(GenericMILError) -> Int {
        guard rows > 0, cols > 0 else { throw .invalidShape }
        return try checkedMultiply(rows, cols)
    }

    public static func fusedQKVContract(dim: Int, spatial: Int) throws(GenericMILError) -> GenericMILContract {
        guard dim > 0, spatial > 0 else { throw .invalidShape }
        let inputElems = try checkedMultiply(dim, spatial)
        let inputBytes = try checkedMultiply(inputElems, MemoryLayout<Float>.stride)
        let perOutputBytes = inputBytes
        return GenericMILContract(inputBytes: inputBytes, outputByteSizes: [perOutputBytes, perOutputBytes, perOutputBytes])
    }

    public static func fusedFFNUpContract(dim: Int, hiddenDim: Int, spatial: Int) throws(GenericMILError) -> GenericMILContract {
        guard dim > 0, hiddenDim > 0, spatial > 0 else { throw .invalidShape }
        let inputElems = try checkedMultiply(dim, spatial)
        let inputBytes = try checkedMultiply(inputElems, MemoryLayout<Float>.stride)
        let outputElems = try checkedMultiply(hiddenDim, spatial)
        let perOutputBytes = try checkedMultiply(outputElems, MemoryLayout<Float>.stride)
        return GenericMILContract(inputBytes: inputBytes, outputByteSizes: [perOutputBytes, perOutputBytes])
    }

    public static func matmul(inCh: Int, outCh: Int, spatial: Int) -> String {
        """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
            func main<ios18>(tensor<fp32, [1, \(inCh), \(spatial)]> x, tensor<fp32, [1, \(outCh), \(inCh)]> W) {
                string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
                tensor<fp16, [1, \(inCh), \(spatial)]> x16 = cast(dtype = to_fp16, x = x)[name = string("cast_x")];
                tensor<fp16, [1, \(outCh), \(inCh)]> W16 = cast(dtype = to_fp16, x = W)[name = string("cast_W")];
                bool tx = const()[name = string("tx"), val = bool(false)];
                bool ty = const()[name = string("ty"), val = bool(false)];
                tensor<fp16, [1, \(outCh), \(spatial)]> y16 = matmul(transpose_x = tx, transpose_y = ty, x = W16, y = x16)[name = string("mm")];
                string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
                tensor<fp32, [1, \(outCh), \(spatial)]> y = cast(dtype = to_fp32, x = y16)[name = string("cast_out")];
            } -> (y);
        }
        """
        + "\n"
    }

    public static func conv(inCh: Int, outCh: Int, spatial: Int) -> String {
        """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
            func main<ios18>(tensor<fp32, [1, \(inCh), 1, \(spatial)]> x) {
                string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];
                tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];
                tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
                tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];
                int32 c_groups = const()[name = string("c_groups"), val = int32(1)];
                string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
                tensor<fp16, [1, \(inCh), 1, \(spatial)]> x16 = cast(dtype = to_fp16, x = x)[name = string("cast_in")];
                tensor<fp16, [\(outCh), \(inCh), 1, 1]> W = const()[name = string("W"), val = tensor<fp16, [\(outCh), \(inCh), 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
                tensor<fp16, [1, \(outCh), 1, \(spatial)]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string("conv")];
                string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
                tensor<fp32, [1, \(outCh), 1, \(spatial)]> y = cast(dtype = to_fp32, x = y16)[name = string("cast_out")];
            } -> (y);
        }
        """
        + "\n"
    }

    public static func fusedQKV(dim: Int, spatial: Int) -> String {
        let cs = 64 + dim * dim * 2
        return """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
            func main<ios18>(tensor<fp32, [1, \(dim), 1, \(spatial)]> x) {
                string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];
                tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];
                tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
                tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];
                int32 c_groups = const()[name = string("c_groups"), val = int32(1)];
                string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
                tensor<fp16, [1, \(dim), 1, \(spatial)]> x16 = cast(dtype = to_fp16, x = x)[name = string("cast_in")];
                tensor<fp16, [\(dim), \(dim), 1, 1]> Wq = const()[name = string("Wq"), val = tensor<fp16, [\(dim), \(dim), 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
                tensor<fp16, [\(dim), \(dim), 1, 1]> Wk = const()[name = string("Wk"), val = tensor<fp16, [\(dim), \(dim), 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(\(64 + cs))))];
                tensor<fp16, [\(dim), \(dim), 1, 1]> Wv = const()[name = string("Wv"), val = tensor<fp16, [\(dim), \(dim), 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(\(64 + 2 * cs))))];
                tensor<fp16, [1, \(dim), 1, \(spatial)]> q16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wq, x = x16)[name = string("conv_q")];
                tensor<fp16, [1, \(dim), 1, \(spatial)]> k16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wk, x = x16)[name = string("conv_k")];
                tensor<fp16, [1, \(dim), 1, \(spatial)]> v16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = Wv, x = x16)[name = string("conv_v")];
                string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
                tensor<fp32, [1, \(dim), 1, \(spatial)]> q = cast(dtype = to_fp32, x = q16)[name = string("cast_q")];
                tensor<fp32, [1, \(dim), 1, \(spatial)]> k = cast(dtype = to_fp32, x = k16)[name = string("cast_k")];
                tensor<fp32, [1, \(dim), 1, \(spatial)]> v = cast(dtype = to_fp32, x = v16)[name = string("cast_v")];
            } -> (q, k, v);
        }
        """
        + "\n"
    }

    public static func fusedFFNUp(dim: Int, hiddenDim: Int, spatial: Int) -> String {
        let cs = 64 + hiddenDim * dim * 2
        return """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
            func main<ios18>(tensor<fp32, [1, \(dim), 1, \(spatial)]> x) {
                string c_pad_type = const()[name = string("c_pad_type"), val = string("valid")];
                tensor<int32, [2]> c_strides = const()[name = string("c_strides"), val = tensor<int32, [2]>([1, 1])];
                tensor<int32, [4]> c_pad = const()[name = string("c_pad"), val = tensor<int32, [4]>([0, 0, 0, 0])];
                tensor<int32, [2]> c_dilations = const()[name = string("c_dilations"), val = tensor<int32, [2]>([1, 1])];
                int32 c_groups = const()[name = string("c_groups"), val = int32(1)];
                string to_fp16 = const()[name = string("to_fp16"), val = string("fp16")];
                tensor<fp16, [1, \(dim), 1, \(spatial)]> x16 = cast(dtype = to_fp16, x = x)[name = string("cast_in")];
                tensor<fp16, [\(hiddenDim), \(dim), 1, 1]> W1 = const()[name = string("W1"), val = tensor<fp16, [\(hiddenDim), \(dim), 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
                tensor<fp16, [\(hiddenDim), \(dim), 1, 1]> W3 = const()[name = string("W3"), val = tensor<fp16, [\(hiddenDim), \(dim), 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(\(64 + cs))))];
                tensor<fp16, [1, \(hiddenDim), 1, \(spatial)]> h1 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W1, x = x16)[name = string("conv_w1")];
                tensor<fp16, [1, \(hiddenDim), 1, \(spatial)]> h3 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W3, x = x16)[name = string("conv_w3")];
                string to_fp32 = const()[name = string("to_fp32"), val = string("fp32")];
                tensor<fp32, [1, \(hiddenDim), 1, \(spatial)]> out1 = cast(dtype = to_fp32, x = h1)[name = string("cast_h1")];
                tensor<fp32, [1, \(hiddenDim), 1, \(spatial)]> out3 = cast(dtype = to_fp32, x = h3)[name = string("cast_h3")];
            } -> (out1, out3);
        }
        """
        + "\n"
    }

    public static func buildFusedQKVWeightBlob(wq: [Float], wk: [Float], wv: [Float], dim: Int) throws(GenericMILError) -> Data {
        let expected = try checkedSquareCount(dim)
        guard wq.count == expected else { throw GenericMILError.invalidWeightCount(expected: expected, got: wq.count) }
        guard wk.count == expected else { throw GenericMILError.invalidWeightCount(expected: expected, got: wk.count) }
        guard wv.count == expected else { throw GenericMILError.invalidWeightCount(expected: expected, got: wv.count) }

        let wsize = try checkedMultiply(expected, MemoryLayout<UInt16>.stride)
        let cs = try checkedAdd(64, wsize)
        let total = try checkedAdd(64, try checkedMultiply(3, cs))
        let wsizeU32 = try checkedUInt32(wsize)
        var offsetU32: [UInt32] = []
        offsetU32.reserveCapacity(3)
        for chunkIndex in 0..<3 {
            let chunkStart = try checkedAdd(64, try checkedMultiply(chunkIndex, cs))
            let dataOffset = try checkedAdd(chunkStart, 64)
            offsetU32.append(try checkedUInt32(dataOffset))
        }

        var data = Data(count: total)
        data.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            base[0] = 0x01
            base[4] = 0x02

            func writeChunk(_ chunkIndex: Int, weights: [Float]) {
                let chunkStart = 64 + chunkIndex * cs
                let chunk = base.advanced(by: chunkStart)
                chunk[0] = 0xEF
                chunk[1] = 0xBE
                chunk[2] = 0xAD
                chunk[3] = 0xDE
                chunk[4] = 0x01

                raw.storeBytes(of: wsizeU32.littleEndian, toByteOffset: chunkStart + 8, as: UInt32.self)
                raw.storeBytes(of: offsetU32[chunkIndex].littleEndian, toByteOffset: chunkStart + 16, as: UInt32.self)

                let payload = raw.baseAddress!.advanced(by: chunkStart + 64).assumingMemoryBound(to: UInt16.self)
                for i in 0..<expected {
                    payload[i] = Float16(weights[i]).bitPattern.littleEndian
                }
            }

            writeChunk(0, weights: wq)
            writeChunk(1, weights: wk)
            writeChunk(2, weights: wv)
        }
        return data
    }

    public static func buildFusedFFNUpWeightBlob(w1: [Float], w3: [Float], hiddenDim: Int, dim: Int) throws(GenericMILError) -> Data {
        let expected = try checkedRectCount(rows: hiddenDim, cols: dim)
        guard w1.count == expected else { throw GenericMILError.invalidWeightCount(expected: expected, got: w1.count) }
        guard w3.count == expected else { throw GenericMILError.invalidWeightCount(expected: expected, got: w3.count) }

        let wsize = try checkedMultiply(expected, MemoryLayout<UInt16>.stride)
        let cs = try checkedAdd(64, wsize)
        let total = try checkedAdd(64, try checkedMultiply(2, cs))
        let wsizeU32 = try checkedUInt32(wsize)
        var offsetU32: [UInt32] = []
        offsetU32.reserveCapacity(2)
        for chunkIndex in 0..<2 {
            let chunkStart = try checkedAdd(64, try checkedMultiply(chunkIndex, cs))
            let dataOffset = try checkedAdd(chunkStart, 64)
            offsetU32.append(try checkedUInt32(dataOffset))
        }

        var data = Data(count: total)
        data.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
            base[0] = 0x01
            base[4] = 0x02

            func writeChunk(_ chunkIndex: Int, weights: [Float]) {
                let chunkStart = 64 + chunkIndex * cs
                let chunk = base.advanced(by: chunkStart)
                chunk[0] = 0xEF
                chunk[1] = 0xBE
                chunk[2] = 0xAD
                chunk[3] = 0xDE
                chunk[4] = 0x01

                raw.storeBytes(of: wsizeU32.littleEndian, toByteOffset: chunkStart + 8, as: UInt32.self)
                raw.storeBytes(of: offsetU32[chunkIndex].littleEndian, toByteOffset: chunkStart + 16, as: UInt32.self)

                let payload = raw.baseAddress!.advanced(by: chunkStart + 64).assumingMemoryBound(to: UInt16.self)
                for i in 0..<expected {
                    payload[i] = Float16(weights[i]).bitPattern.littleEndian
                }
            }

            writeChunk(0, weights: w1)
            writeChunk(1, weights: w3)
        }
        return data
    }
}

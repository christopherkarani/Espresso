import XCTest
import Foundation
import IOSurface
@testable import ANEInterop

private func requireANEAvailable(file: StaticString = #filePath, line: UInt = #line) throws {
    let handle = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW)
    if handle == nil {
        throw XCTSkip("AppleNeuralEngine.framework unavailable", file: file, line: line)
    }
    dlclose(handle)

    let requiredClasses = [
        "_ANEInMemoryModelDescriptor",
        "_ANEInMemoryModel",
        "_ANERequest",
        "_ANEIOSurfaceObject",
    ]
    for c in requiredClasses where NSClassFromString(c) == nil {
        throw XCTSkip("ANE private class missing: \(c)", file: file, line: line)
    }
}

private func requireANEHardwareTestsEnabled(file: StaticString = #filePath, line: UInt = #line) throws {
    guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
        throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run ANE compile/eval tests", file: file, line: line)
    }
    try requireANEAvailable(file: file, line: line)
}

final class ANEInteropTests: XCTestCase {
    func test_init_idempotent() {
        ane_interop_init()
        ane_interop_init()
    }

    func test_init_thread_safe_under_contention() {
        DispatchQueue.concurrentPerform(iterations: 1_000) { _ in
            ane_interop_init()
        }
        ane_interop_init()
    }

    func test_create_surface_valid() throws {
        let bytes = 1024
        guard let s = ane_interop_create_surface(bytes) else {
            XCTFail("ane_interop_create_surface returned NULL")
            return
        }
        XCTAssertEqual(IOSurfaceGetAllocSize(s), bytes)
    }

    func test_get_input_output_nil_handle_returns_nil() {
        XCTAssertNil(ane_interop_get_input(nil, 0))
        XCTAssertNil(ane_interop_get_output(nil, 0))
    }

    func test_compile_count_set_get_roundtrip_without_hardware() {
        let old = ane_interop_compile_count()
        defer { ane_interop_set_compile_count(old) }

        ane_interop_set_compile_count(1234)
        XCTAssertEqual(ane_interop_compile_count(), 1234)
        ane_interop_set_compile_count(0)
        XCTAssertEqual(ane_interop_compile_count(), 0)
    }

    func test_compile_rejects_invalid_pointer_contracts() {
        let mil = Array("program(1.3)".utf8)
        let oneSize = [Int](repeating: 64, count: 1)
        let onePath = ["@model_path/weights/weight.bin"]
        let oneData = [UInt8](repeating: 0, count: 8)

        let handleWeightPointersMissing = mil.withUnsafeBytes { milBuf in
            oneSize.withUnsafeBytes { inBuf in
                oneSize.withUnsafeBytes { outBuf in
                    ane_interop_compile(
                        milBuf.bindMemory(to: UInt8.self).baseAddress!,
                        milBuf.count,
                        nil, nil, nil, 1,
                        1, inBuf.bindMemory(to: Int.self).baseAddress!,
                        1, outBuf.bindMemory(to: Int.self).baseAddress!
                    )
                }
            }
        }
        XCTAssertNil(handleWeightPointersMissing)
        XCTAssertEqual(ane_interop_last_compile_error(), ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS)

        let handleInputSizesMissing = mil.withUnsafeBytes { milBuf in
            onePath[0].withCString { cPath in
                oneData.withUnsafeBytes { dBuf in
                    var paths: [UnsafePointer<CChar>?] = [cPath]
                    var datas: [UnsafePointer<UInt8>?] = [dBuf.bindMemory(to: UInt8.self).baseAddress!]
                    let lens: [Int] = [dBuf.count]
                    return paths.withUnsafeMutableBufferPointer { p in
                        datas.withUnsafeMutableBufferPointer { d in
                            lens.withUnsafeBufferPointer { l in
                                oneSize.withUnsafeBytes { outBuf in
                                    ane_interop_compile(
                                        milBuf.bindMemory(to: UInt8.self).baseAddress!,
                                        milBuf.count,
                                        p.baseAddress, d.baseAddress, l.baseAddress, 1,
                                        1, nil,
                                        1, outBuf.bindMemory(to: Int.self).baseAddress!
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
        XCTAssertNil(handleInputSizesMissing)
        XCTAssertEqual(ane_interop_last_compile_error(), ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS)
    }

    func test_compile_rejects_duplicate_weight_paths_without_hardware() {
        let mil = Array("program(1.3)".utf8)
        let inOutSize = [Int](repeating: 64, count: 1)
        let path = "@model_path/weights/shared.bin"
        let payloadA = [UInt8](repeating: 0x11, count: 8)
        let payloadB = [UInt8](repeating: 0x22, count: 8)

        let handle = mil.withUnsafeBytes { milBuf in
            payloadA.withUnsafeBytes { bufA in
                payloadB.withUnsafeBytes { bufB in
                    path.withCString { cPath in
                        var paths: [UnsafePointer<CChar>?] = [cPath, cPath]
                        var datas: [UnsafePointer<UInt8>?] = [
                            bufA.bindMemory(to: UInt8.self).baseAddress!,
                            bufB.bindMemory(to: UInt8.self).baseAddress!,
                        ]
                        let lens: [Int] = [payloadA.count, payloadB.count]
                        return paths.withUnsafeMutableBufferPointer { pathBuf in
                            datas.withUnsafeMutableBufferPointer { dataBuf in
                                lens.withUnsafeBufferPointer { lenBuf in
                                    inOutSize.withUnsafeBytes { inBuf in
                                        inOutSize.withUnsafeBytes { outBuf in
                                            ane_interop_compile(
                                                milBuf.bindMemory(to: UInt8.self).baseAddress!,
                                                milBuf.count,
                                                pathBuf.baseAddress,
                                                dataBuf.baseAddress,
                                                lenBuf.baseAddress,
                                                2,
                                                1,
                                                inBuf.bindMemory(to: Int.self).baseAddress!,
                                                1,
                                                outBuf.bindMemory(to: Int.self).baseAddress!
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        XCTAssertNil(handle)
        XCTAssertEqual(ane_interop_last_compile_error(), ANE_INTEROP_COMPILE_ERROR_DUPLICATE_WEIGHT_PATH)
    }

    func test_compile_rejects_weight_path_traversal() throws {
        try requireANEAvailable()

        let mil = Array("program(1.3)".utf8)
        let payload = [UInt8](repeating: 0, count: 8)
        let inOutSize = [Int](repeating: 64, count: 1)
        let escapedName = "escape-\(UUID().uuidString).bin"
        let weightPath = "@model_path/../../\(escapedName)"
        let expectedEscapedPath = (NSTemporaryDirectory() as NSString).appendingPathComponent(escapedName)
        defer { try? FileManager.default.removeItem(atPath: expectedEscapedPath) }

        let handle = mil.withUnsafeBytes { milBuf in
            payload.withUnsafeBytes { payloadBuf in
                weightPath.withCString { cPath in
                    var paths: [UnsafePointer<CChar>?] = [cPath]
                    var datas: [UnsafePointer<UInt8>?] = [payloadBuf.bindMemory(to: UInt8.self).baseAddress!]
                    let lens: [Int] = [payloadBuf.count]
                    return paths.withUnsafeMutableBufferPointer { pathsBuf in
                        datas.withUnsafeMutableBufferPointer { datasBuf in
                            lens.withUnsafeBufferPointer { lensBuf in
                                inOutSize.withUnsafeBytes { inBuf in
                                    inOutSize.withUnsafeBytes { outBuf in
                                        ane_interop_compile(
                                            milBuf.bindMemory(to: UInt8.self).baseAddress!,
                                            milBuf.count,
                                            pathsBuf.baseAddress,
                                            datasBuf.baseAddress,
                                            lensBuf.baseAddress,
                                            1,
                                            1, inBuf.bindMemory(to: Int.self).baseAddress!,
                                            1, outBuf.bindMemory(to: Int.self).baseAddress!
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        XCTAssertNil(handle)
        XCTAssertFalse(FileManager.default.fileExists(atPath: expectedEscapedPath))
    }

    func test_compile_invalid_mil_returns_nil() throws {
        try requireANEHardwareTestsEnabled()

        let garbage = Array("this is not MIL".utf8)
        let inputBytes: [Int] = [64]
        let outputBytes: [Int] = [64]

        let handle = garbage.withUnsafeBytes { milBuf in
            inputBytes.withUnsafeBytes { inBuf in
                outputBytes.withUnsafeBytes { outBuf in
                    ane_interop_compile(
                        milBuf.bindMemory(to: UInt8.self).baseAddress!,
                        milBuf.count,
                        nil, nil, nil, 0,
                        1, inBuf.bindMemory(to: Int.self).baseAddress!,
                        1, outBuf.bindMemory(to: Int.self).baseAddress!
                    )
                }
            }
        }

        XCTAssertNil(handle)
    }

    func test_compile_count_unchanged_on_compile_failure() throws {
        try requireANEHardwareTestsEnabled()
        let old = ane_interop_compile_count()
        defer { ane_interop_set_compile_count(old) }

        ane_interop_set_compile_count(0)
        let garbage = Array("this is not MIL".utf8)
        let inputBytes: [Int] = [64]
        let outputBytes: [Int] = [64]

        let handle = garbage.withUnsafeBytes { milBuf in
            inputBytes.withUnsafeBytes { inBuf in
                outputBytes.withUnsafeBytes { outBuf in
                    ane_interop_compile(
                        milBuf.bindMemory(to: UInt8.self).baseAddress!,
                        milBuf.count,
                        nil, nil, nil, 0,
                        1, inBuf.bindMemory(to: Int.self).baseAddress!,
                        1, outBuf.bindMemory(to: Int.self).baseAddress!
                    )
                }
            }
        }

        XCTAssertNil(handle)
        XCTAssertEqual(ane_interop_compile_count(), 0)
    }

    func test_compile_identity_kernel() throws {
        try requireANEHardwareTestsEnabled()

        // Known-good minimal conv program with fp16 IO and identity weights.
        let CH = 4
        let SP = 4
        let mil = String(format: """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
            func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {
                string pt = const()[name=string("pt"), val=string("valid")];
                tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
                tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
                tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
                int32 gr = const()[name=string("gr"), val=int32(1)];
                tensor<fp16, [%d,%d,1,1]> W = const()[name=string("W"),
        val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string("@model_path/weights/weight.bin"), offset=uint64(64)))];
                tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)
        [name=string("conv")];
            } -> (y);
        }
        """, CH, SP, CH, CH, CH, CH, CH, SP)

        let inputCount = CH * SP
        let inputBytes = inputCount * MemoryLayout<UInt16>.stride
        let outputBytes = inputBytes

        // Weight blob for 1x1 conv: identity matrix in fp16 blob format.
        let ws = CH * CH * 2
        let total = 128 + ws
        var weightBlob = [UInt8](repeating: 0, count: total)
        weightBlob[0] = 1
        weightBlob[4] = 2
        weightBlob[64] = 0xEF
        weightBlob[65] = 0xBE
        weightBlob[66] = 0xAD
        weightBlob[67] = 0xDE
        weightBlob[68] = 1
        let wsU32 = UInt32(ws)
        weightBlob[72] = UInt8(truncatingIfNeeded: wsU32)
        weightBlob[73] = UInt8(truncatingIfNeeded: wsU32 >> 8)
        weightBlob[74] = UInt8(truncatingIfNeeded: wsU32 >> 16)
        weightBlob[75] = UInt8(truncatingIfNeeded: wsU32 >> 24)
        let offU32 = UInt32(128)
        weightBlob[80] = UInt8(truncatingIfNeeded: offU32)
        weightBlob[81] = UInt8(truncatingIfNeeded: offU32 >> 8)
        weightBlob[82] = UInt8(truncatingIfNeeded: offU32 >> 16)
        weightBlob[83] = UInt8(truncatingIfNeeded: offU32 >> 24)

        var wF32 = [Float](repeating: 0, count: CH * CH)
        for i in 0..<CH { wF32[i * CH + i] = 1 }
        weightBlob.withUnsafeMutableBytes { blobRaw in
            let payload = blobRaw.baseAddress!.advanced(by: 128)
            wF32.withUnsafeBufferPointer { wBuf in
                ane_interop_cvt_f32_to_f16(payload, wBuf.baseAddress!, Int32(CH * CH))
            }
        }

        let weightPath = "@model_path/weights/weight.bin"

        let handle = mil.data(using: .utf8)!.withUnsafeBytes { milBuf in
            weightBlob.withUnsafeBytes { weightBuf in
                weightPath.withCString { pathCStr in
                    var paths: [UnsafePointer<CChar>?] = [pathCStr]
                    var datas: [UnsafePointer<UInt8>?] = [weightBuf.bindMemory(to: UInt8.self).baseAddress!]
                    let lens: [Int] = [weightBuf.count]
                    var inSize = inputBytes
                    var outSize = outputBytes
                    return paths.withUnsafeMutableBufferPointer { pathsBuf in
                        datas.withUnsafeMutableBufferPointer { datasBuf in
                            lens.withUnsafeBufferPointer { lensBuf in
                                withUnsafeBytes(of: &inSize) { inBuf in
                                    withUnsafeBytes(of: &outSize) { outBuf in
                                        ane_interop_compile(
                                            milBuf.bindMemory(to: UInt8.self).baseAddress!,
                                            milBuf.count,
                                            pathsBuf.baseAddress,
                                            datasBuf.baseAddress,
                                            lensBuf.baseAddress,
                                            1,
                                            1, inBuf.bindMemory(to: Int.self).baseAddress!,
                                            1, outBuf.bindMemory(to: Int.self).baseAddress!
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        guard let handle else {
            XCTFail("ane_interop_compile returned NULL")
            return
        }
        defer { ane_interop_free(handle) }

        XCTAssertNil(ane_interop_get_input(handle, -1))
        XCTAssertNil(ane_interop_get_output(handle, -1))
        XCTAssertNil(ane_interop_get_input(handle, 1))
        XCTAssertNil(ane_interop_get_output(handle, 1))

        guard let inputSurface = ane_interop_get_input(handle, 0),
              let outputSurface = ane_interop_get_output(handle, 0) else {
            XCTFail("missing input/output surface")
            return
        }
        guard let retainedInput = ane_interop_copy_input(handle, 0),
              let retainedOutput = ane_interop_copy_output(handle, 0) else {
            XCTFail("copy_input/copy_output returned nil")
            return
        }
        _ = retainedInput
        _ = retainedOutput

        let input = (0..<inputCount).map { i in Float(i) * 0.1 - 0.7 }
        var output = Array(repeating: Float.nan, count: inputCount)

        input.withUnsafeBufferPointer { inBuf in
            XCTAssertTrue(
                ane_interop_io_write_fp16_at(inputSurface, 0, inBuf.baseAddress!, Int32(CH), Int32(SP))
            )
        }

        XCTAssertTrue(ane_interop_eval(handle))

        IOSurfaceLock(outputSurface, .readOnly, nil)
        output.withUnsafeMutableBytes { raw in
            ane_interop_cvt_f16_to_f32(
                raw.bindMemory(to: Float.self).baseAddress!,
                IOSurfaceGetBaseAddress(outputSurface),
                Int32(inputCount)
            )
        }
        IOSurfaceUnlock(outputSurface, .readOnly, nil)

        var maxAbsDiff: Float = 0
        for i in 0..<inputCount {
            maxAbsDiff = max(maxAbsDiff, abs(output[i] - input[i]))
        }
        XCTAssertLessThanOrEqual(maxAbsDiff, 1e-2)
    }

    func test_compile_count_increments() throws {
        try requireANEHardwareTestsEnabled()

        ane_interop_set_compile_count(0)

        let mil = """
        program(1.3)
        [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
        {
            func main<ios18>(tensor<fp32, [1, 1, 1, 1]> x) {
                string to16 = const()[name=string("to16"), val=string("fp16")];
                tensor<fp16, [1,1,1,1]> x16 = cast(dtype=to16, x=x)[name=string("x16")];
                string to32 = const()[name=string("to32"), val=string("fp32")];
                tensor<fp32, [1,1,1,1]> y = cast(dtype=to32, x=x16)[name=string("y")];
            } -> (y);
        }
        """

        let inputBytes = MemoryLayout<Float>.stride
        let outputBytes = inputBytes

        let handle = mil.data(using: .utf8)!.withUnsafeBytes { milBuf in
            var inSize = inputBytes
            var outSize = outputBytes
            return withUnsafeBytes(of: &inSize) { inBuf in
                withUnsafeBytes(of: &outSize) { outBuf in
                    ane_interop_compile(
                        milBuf.bindMemory(to: UInt8.self).baseAddress!,
                        milBuf.count,
                        nil, nil, nil, 0,
                        1, inBuf.bindMemory(to: Int.self).baseAddress!,
                        1, outBuf.bindMemory(to: Int.self).baseAddress!
                    )
                }
            }
        }
        guard let handle else {
            XCTFail("ane_interop_compile returned NULL")
            return
        }
        defer { ane_interop_free(handle) }

        XCTAssertEqual(ane_interop_compile_count(), 1)
    }

    func test_neon_f32_f16_roundtrip() {
        let count = 100_000
        let input = (0..<count).map { _ in Float.random(in: -10...10) }
        var output = Array(repeating: Float.nan, count: count)

        let buf = UnsafeMutableRawPointer.allocate(byteCount: count * 2, alignment: 2)
        defer { buf.deallocate() }

        input.withUnsafeBufferPointer { inputBuf in
            ane_interop_cvt_f32_to_f16(buf, inputBuf.baseAddress!, Int32(count))
        }
        output.withUnsafeMutableBufferPointer { outBuf in
            ane_interop_cvt_f16_to_f32(outBuf.baseAddress!, buf, Int32(count))
        }

        var maxAbsDiff: Float = 0
        for i in 0..<count {
            maxAbsDiff = max(maxAbsDiff, abs(output[i] - input[i]))
        }
        XCTAssertLessThanOrEqual(maxAbsDiff, 1e-2)
    }

    func test_io_copy_between_surfaces() throws {
        let channels = 8
        let spatial = 4
        let bytes = channels * spatial * 2

        guard let src = ane_interop_create_surface(bytes),
              let dst = ane_interop_create_surface(bytes) else {
            XCTFail("surface alloc failed")
            return
        }

        let input = (0..<(channels * spatial)).map { Float($0) * 0.25 - 1.0 }

        // Write src as fp16.
        IOSurfaceLock(src, [], nil)
        input.withUnsafeBufferPointer { inputBuf in
            ane_interop_cvt_f32_to_f16(IOSurfaceGetBaseAddress(src), inputBuf.baseAddress!, Int32(channels * spatial))
        }
        IOSurfaceUnlock(src, [], nil)

        // Zero dst.
        IOSurfaceLock(dst, [], nil)
        memset(IOSurfaceGetBaseAddress(dst), 0, bytes)
        IOSurfaceUnlock(dst, [], nil)

        // Copy src -> dst at ch offset 2 (so only last 6 channels fit).
        let srcOff = 0
        let dstOff = 2
        let copyChannels = 6
        XCTAssertTrue(ane_interop_io_copy(dst, Int32(dstOff), src, Int32(srcOff), Int32(copyChannels), Int32(spatial)))

        // Read dst back.
        var dstF32 = Array(repeating: Float.nan, count: channels * spatial)
        IOSurfaceLock(dst, .readOnly, nil)
        dstF32.withUnsafeMutableBufferPointer { outBuf in
            ane_interop_cvt_f16_to_f32(outBuf.baseAddress!, IOSurfaceGetBaseAddress(dst), Int32(channels * spatial))
        }
        IOSurfaceUnlock(dst, .readOnly, nil)

        // Verify copied region matches source (within fp16 tolerance), untouched prefix is 0.
        for ch in 0..<channels {
            for s in 0..<spatial {
                let idx = ch * spatial + s
                if ch < dstOff {
                    XCTAssertEqual(dstF32[idx], 0, accuracy: 0)
                } else {
                    let srcIdx = (ch - dstOff) * spatial + s
                    XCTAssertEqual(dstF32[idx], input[srcIdx], accuracy: 1e-2)
                }
            }
        }
    }

    func test_io_write_fp16_at_offset() throws {
        let channels = 8
        let spatial = 4
        let bytes = channels * spatial * 2

        guard let s = ane_interop_create_surface(bytes) else {
            XCTFail("surface alloc failed")
            return
        }

        IOSurfaceLock(s, [], nil)
        memset(IOSurfaceGetBaseAddress(s), 0, bytes)
        IOSurfaceUnlock(s, [], nil)

        let chOff = 3
        let writeChannels = 2
        let data = (0..<(writeChannels * spatial)).map { Float($0) * 0.5 + 2.0 }
        data.withUnsafeBufferPointer { buf in
            XCTAssertTrue(ane_interop_io_write_fp16_at(s, Int32(chOff), buf.baseAddress!, Int32(writeChannels), Int32(spatial)))
        }

        var out = Array(repeating: Float.nan, count: channels * spatial)
        IOSurfaceLock(s, .readOnly, nil)
        out.withUnsafeMutableBufferPointer { outBuf in
            ane_interop_cvt_f16_to_f32(outBuf.baseAddress!, IOSurfaceGetBaseAddress(s), Int32(channels * spatial))
        }
        IOSurfaceUnlock(s, .readOnly, nil)

        for ch in 0..<channels {
            for sp in 0..<spatial {
                let idx = ch * spatial + sp
                if chOff <= ch && ch < chOff + writeChannels {
                    let srcIdx = (ch - chOff) * spatial + sp
                    XCTAssertEqual(out[idx], data[srcIdx], accuracy: 1e-2)
                } else {
                    XCTAssertEqual(out[idx], 0, accuracy: 0)
                }
            }
        }
    }

    func test_io_copy_rejects_invalid_ranges() throws {
        let channels = 4
        let spatial = 4
        let bytes = channels * spatial * 2
        guard let src = ane_interop_create_surface(bytes),
              let dst = ane_interop_create_surface(bytes) else {
            XCTFail("surface alloc failed")
            return
        }

        XCTAssertFalse(ane_interop_io_copy(dst, -1, src, 0, 1, Int32(spatial)))
        XCTAssertFalse(ane_interop_io_copy(dst, 0, src, 0, Int32(channels + 1), Int32(spatial)))
        XCTAssertFalse(ane_interop_io_copy(nil, 0, src, 0, 1, Int32(spatial)))
    }

    func test_io_write_fp16_at_rejects_invalid_ranges() throws {
        let channels = 4
        let spatial = 4
        let bytes = channels * spatial * 2
        guard let s = ane_interop_create_surface(bytes) else {
            XCTFail("surface alloc failed")
            return
        }

        let values = [Float](repeating: 1, count: 4)
        values.withUnsafeBufferPointer { buf in
            XCTAssertFalse(ane_interop_io_write_fp16_at(s, -1, buf.baseAddress, 1, Int32(spatial)))
            XCTAssertFalse(ane_interop_io_write_fp16_at(s, Int32(channels), buf.baseAddress, 1, Int32(spatial)))
            XCTAssertFalse(ane_interop_io_write_fp16_at(nil, 0, buf.baseAddress, 1, Int32(spatial)))
        }
    }

    func test_io_copy_same_surface_overlap_uses_memmove_semantics() throws {
        let channels = 8
        let spatial = 1
        let bytes = channels * spatial * 2
        guard let s = ane_interop_create_surface(bytes) else {
            XCTFail("surface alloc failed")
            return
        }

        let input = (0..<channels).map { Float($0) }
        IOSurfaceLock(s, [], nil)
        input.withUnsafeBufferPointer { inputBuf in
            ane_interop_cvt_f32_to_f16(IOSurfaceGetBaseAddress(s), inputBuf.baseAddress!, Int32(channels))
        }
        IOSurfaceUnlock(s, [], nil)

        XCTAssertTrue(ane_interop_io_copy(s, 1, s, 0, 7, 1))

        var out = Array(repeating: Float.nan, count: channels)
        IOSurfaceLock(s, .readOnly, nil)
        out.withUnsafeMutableBufferPointer { outBuf in
            ane_interop_cvt_f16_to_f32(outBuf.baseAddress!, IOSurfaceGetBaseAddress(s), Int32(channels))
        }
        IOSurfaceUnlock(s, .readOnly, nil)

        XCTAssertEqual(out[0], 0, accuracy: 1e-2)
        XCTAssertEqual(out[1], 0, accuracy: 1e-2)
        XCTAssertEqual(out[2], 1, accuracy: 1e-2)
        XCTAssertEqual(out[3], 2, accuracy: 1e-2)
        XCTAssertEqual(out[4], 3, accuracy: 1e-2)
        XCTAssertEqual(out[5], 4, accuracy: 1e-2)
        XCTAssertEqual(out[6], 5, accuracy: 1e-2)
        XCTAssertEqual(out[7], 6, accuracy: 1e-2)
    }
}

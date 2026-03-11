import XCTest
import ANEInterop
import IOSurface

final class ANEPerfStatsTests: XCTestCase {
    private func requireANEHardwareTestsEnabled(file: StaticString = #filePath, line: UInt = #line) throws {
        guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
            throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run ANE compile/eval tests", file: file, line: line)
        }
    }

    func test_perf_stats_hw_execution_time_nonzero_smoke() throws {
        try requireANEHardwareTestsEnabled()

        // Enable perf stats collection in ANEInterop.
        setenv("ANE_PERF_STATS", "1", 1)

        // Use a moderately sized conv; small toy shapes are unstable on some hosts.
        let CH = 64
        let SP = 32

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

        // Identity weights blob (fp16).
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
        let bytes = CH * SP * MemoryLayout<UInt16>.stride
        var inSize = bytes
        var outSize = bytes

        let handle: OpaquePointer? = mil.data(using: .utf8)!.withUnsafeBytes { milBuf in
            weightBlob.withUnsafeBytes { weightBuf in
                weightPath.withCString { pathCStr in
                    var paths: [UnsafePointer<CChar>?] = [pathCStr]
                    var datas: [UnsafePointer<UInt8>?] = [weightBuf.bindMemory(to: UInt8.self).baseAddress!]
                    let lens: [Int] = [weightBuf.count]
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
            XCTFail("ane_interop_compile returned NULL (perf stats smoke)")
            return
        }
        defer { ane_interop_free(handle) }

        guard ane_interop_has_perf_stats(handle) else {
            throw XCTSkip("Perf stats factory unavailable on this host build; hwExecutionTime cannot be collected")
        }

        guard let inputSurface = ane_interop_get_input(handle, 0),
              let outputSurface = ane_interop_get_output(handle, 0) else {
            XCTFail("missing input/output surface")
            return
        }

        let inputCount = CH * SP
        let input = (0..<inputCount).map { i in Float(i) * 0.1 - 0.7 }
        input.withUnsafeBufferPointer { inBuf in
            XCTAssertTrue(ane_interop_io_write_fp16_at(inputSurface, 0, inBuf.baseAddress!, Int32(CH), Int32(SP)))
        }

        guard ane_interop_eval(handle) else {
            throw XCTSkip("ANE eval failed on this host; cannot validate perf stats")
        }

        // Exercise readback once to keep the kernel path realistic.
        var output = Array(repeating: Float.nan, count: inputCount)
        IOSurfaceLock(outputSurface, .readOnly, nil)
        output.withUnsafeMutableBytes { raw in
            ane_interop_cvt_f16_to_f32(
                raw.bindMemory(to: Float.self).baseAddress!,
                IOSurfaceGetBaseAddress(outputSurface),
                Int32(inputCount)
            )
        }
        IOSurfaceUnlock(outputSurface, .readOnly, nil)

        let hwNS = ane_interop_last_hw_execution_time_ns(handle)
        XCTAssertGreaterThan(hwNS, 0, "Expected non-zero hwExecutionTime when ANE_PERF_STATS=1")
    }
}

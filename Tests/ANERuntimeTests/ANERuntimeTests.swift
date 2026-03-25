import XCTest
import Foundation
import Darwin
@testable import ANERuntime
import ANEInterop
import MILGenerator
import ANETypes
import IOSurface

private let identityChannels = 4
private let identitySpatial = 8
private let identityElementCount = identityChannels * identitySpatial
private let identityInputBytes = identityElementCount * MemoryLayout<UInt16>.stride
private let identityOutputBytes = identityInputBytes
private let identityWeightPath = "@model_path/weights/weight.bin"

private let fwdAttnInputBytes = 393_216
private let fwdAttnOutputBytes = 2_359_296
private let fwdFFNInputBytes = 393_216
private let fwdFFNOutputBytes = 3_932_160
private let ffnBwdInputBytes = 2_490_368
private let ffnBwdOutputBytes = 2_490_368
private let sdpaBwd1InputBytes = 1_572_864
private let sdpaBwd1OutputBytes = 3_538_944
private let sdpaBwd2InputBytes = 3_932_160
private let sdpaBwd2OutputBytes = 786_432
private let qkvBwdInputBytes = 1_179_648
private let qkvBwdOutputBytes = 393_216
private let sdpaBwd2InputChannels = 2 * ModelConfig.scoreCh + 2 * ModelConfig.dim
private let sdpaBwd2OutputChannels = 2 * ModelConfig.dim
private let sdpaSpatial = ModelConfig.seqLen

private func fillTensor(_ buffer: borrowing TensorBuffer, value: Float) {
    buffer.withUnsafeMutableBufferPointer { ptr in
        guard let base = ptr.baseAddress else { return }
        for i in 0..<ptr.count {
            base[i] = value
        }
    }
}

private func fillLayerWeights(_ weights: borrowing LayerWeights, value: Float) {
    fillTensor(weights.Wq, value: value)
    fillTensor(weights.Wk, value: value)
    fillTensor(weights.Wv, value: value)
    fillTensor(weights.Wo, value: value)
    fillTensor(weights.W1, value: value)
    fillTensor(weights.W2, value: value)
    fillTensor(weights.W3, value: value)
    fillTensor(weights.rmsAtt, value: value)
    fillTensor(weights.rmsFfn, value: value)
}

private func makeTempBinaryPath(prefix: String) -> String {
    let fileName = "\(prefix)-\(UUID().uuidString).bin"
    return FileManager.default.temporaryDirectory.appendingPathComponent(fileName).path
}

private func llamaHeaderData(
    dim: Int32,
    hiddenDim: Int32,
    nLayers: Int32,
    nHeads: Int32,
    nKvHeads: Int32,
    vocabSize: Int32,
    seqLen: Int32
) -> Data {
    var fields: [Int32] = [dim, hiddenDim, nLayers, nHeads, nKvHeads, vocabSize, seqLen]
    for i in fields.indices {
        fields[i] = fields[i].littleEndian
    }
    return fields.withUnsafeBytes { raw in
        Data(raw)
    }
}

private func hasNonZeroElement(_ buffer: borrowing TensorBuffer) -> Bool {
    buffer.withUnsafeBufferPointer { ptr in
        ptr.contains(where: { $0 != 0 })
    }
}

private func storiesModelPath() -> String? {
    if let envPath = ProcessInfo.processInfo.environment["STORIES_MODEL_PATH"], !envPath.isEmpty,
       FileManager.default.fileExists(atPath: envPath) {
        return envPath
    }

    let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let candidates = [
        repoRoot.appendingPathComponent("assets/models/stories110M.bin").path,
        repoRoot.appendingPathComponent("training/../../assets/models/stories110M.bin")
            .standardizedFileURL.path,
    ]

    for candidate in candidates where FileManager.default.fileExists(atPath: candidate) {
        return candidate
    }
    return nil
}

/// Skip if ANE runtime is unavailable.
private func requireANEAvailable(file: StaticString = #filePath, line: UInt = #line) throws {
    let handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW
    )
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

    ane_interop_init()
}

/// Skip tests that require ANE hardware unless ANE_HARDWARE_TESTS=1.
private func requireANEHardwareTestsEnabled(file: StaticString = #filePath, line: UInt = #line) throws {
    guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
        throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run ANE hardware tests", file: file, line: line)
    }
    try requireANEAvailable(file: file, line: line)
}

private func requireObjCCrossValidation(file: StaticString = #filePath, line: UInt = #line) throws {
    guard ProcessInfo.processInfo.environment["OBJC_CROSS_VALIDATION"] == "1" else {
        throw XCTSkip("ObjC cross-validation test (set OBJC_CROSS_VALIDATION=1)", file: file, line: line)
    }
}

private func probeANEEvalWithInterop() -> Bool {
    ane_interop_init()

    let savedCompileCount = ane_interop_compile_count()
    defer { ane_interop_set_compile_count(savedCompileCount) }

    let mil = GenericMIL.conv(inCh: identityChannels, outCh: identityChannels, spatial: identitySpatial)
    guard let milData = mil.data(using: .utf8), !milData.isEmpty else {
        return false
    }

    let weightBlob = makeIdentityWeightBlob(channels: identityChannels)
    var inputSize = identityInputBytes
    var outputSize = identityOutputBytes

    let handle: OpaquePointer? = milData.withUnsafeBytes { milRaw in
        let milBuf = milRaw.bindMemory(to: UInt8.self)
        guard let milBase = milBuf.baseAddress else {
            return nil
        }

        return weightBlob.withUnsafeBytes { weightRaw in
                let weightBuf = weightRaw.bindMemory(to: UInt8.self)
            return identityWeightPath.withCString { cPath in
                var paths: [UnsafePointer<CChar>?] = [cPath]
                var datas: [UnsafePointer<UInt8>?] = [weightBuf.baseAddress]
                let lens: [Int] = [weightBuf.count]

                return paths.withUnsafeMutableBufferPointer { pathBuf in
                    datas.withUnsafeMutableBufferPointer { dataBuf in
                        lens.withUnsafeBufferPointer { lenBuf in
                            withUnsafePointer(to: &inputSize) { inSizePtr in
                                withUnsafePointer(to: &outputSize) { outSizePtr in
                                    ane_interop_compile(
                                        milBase,
                                        milBuf.count,
                                        pathBuf.baseAddress,
                                        dataBuf.baseAddress,
                                        lenBuf.baseAddress,
                                        1,
                                        1,
                                        inSizePtr,
                                        1,
                                        outSizePtr
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
        return false
    }
    defer { ane_interop_free(handle) }
    return ane_interop_eval(handle)
}

private enum ANEEvalProbe {
    static let isAvailable = probeANEEvalWithInterop()
}

private func makeIdentityWeightBlob(channels: Int) -> Data {
    var weights = [Float](repeating: 0, count: channels * channels)
    for i in 0..<channels {
        weights[i * channels + i] = 1
    }
    return WeightBlob.build(from: weights, rows: channels, cols: channels)
}

private func makeIdentityKernel(checkBudget: Bool = true) throws -> ANEKernel {
    let mil = GenericMIL.conv(inCh: identityChannels, outCh: identityChannels, spatial: identitySpatial)
    let weightBlob = makeIdentityWeightBlob(channels: identityChannels)
    return try ANEKernel(
        milText: mil,
        weights: [(path: identityWeightPath, data: weightBlob)],
        inputBytes: identityInputBytes,
        outputBytes: identityOutputBytes,
        checkBudget: checkBudget
    )
}

private func blobPayloadFP16(_ blob: Data, index: Int) -> Float {
    let byteOffset = 128 + index * MemoryLayout<UInt16>.stride
    precondition(byteOffset + 1 < blob.count, "FP16 payload index out of range")
    let lo = UInt16(blob[byteOffset])
    let hi = UInt16(blob[byteOffset + 1]) << 8
    let bits = lo | hi
    return Float(Float16(bitPattern: bits))
}

private func maxAbsDiff(actual: [Float], expected: [Float]) -> (index: Int, actual: Float, expected: Float, diff: Float) {
    precondition(actual.count == expected.count, "Mismatched vector lengths")
    var bestIndex = 0
    var bestActual: Float = 0
    var bestExpected: Float = 0
    var bestDiff: Float = -.infinity
    for i in actual.indices {
        let a = actual[i]
        let e = expected[i]
        let d = abs(a - e)
        if d > bestDiff {
            bestDiff = d
            bestIndex = i
            bestActual = a
            bestExpected = e
        }
    }
    return (bestIndex, bestActual, bestExpected, bestDiff)
}

private func crossValidationFixtureURL(_ name: String) -> URL {
    URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures")
        .appendingPathComponent(name)
}

private func loadFloat32LEFixture(
    _ name: String,
    expectedCount: Int,
    file: StaticString = #filePath,
    line: UInt = #line
) throws -> [Float] {
    let url = crossValidationFixtureURL(name)
    guard FileManager.default.fileExists(atPath: url.path) else {
        throw XCTSkip("Missing ObjC golden fixture: \(url.path)", file: file, line: line)
    }
    let data = try Data(contentsOf: url, options: .mappedIfSafe)
    let expectedBytes = expectedCount * MemoryLayout<UInt32>.stride
    guard data.count == expectedBytes else {
        throw XCTSkip(
            "Fixture size mismatch for \(name): expected \(expectedBytes) bytes, got \(data.count)",
            file: file,
            line: line
        )
    }

    var values = [Float](repeating: 0, count: expectedCount)
    data.withUnsafeBytes { raw in
        guard let base = raw.baseAddress else { return }
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        for i in 0..<expectedCount {
            let o = i * 4
            let bits = UInt32(bytes[o]) | (UInt32(bytes[o + 1]) << 8) | (UInt32(bytes[o + 2]) << 16) | (UInt32(bytes[o + 3]) << 24)
            values[i] = Float(bitPattern: bits)
        }
    }
    return values
}

private func set2x2Sentinel(
    _ buffer: borrowing TensorBuffer,
    cols: Int,
    v00: Float,
    v01: Float,
    v10: Float,
    v11: Float
) {
    buffer.withUnsafeMutableBufferPointer { ptr in
        guard ptr.count >= cols + 2 else { return }
        ptr[0] = v00
        ptr[1] = v01
        ptr[cols] = v10
        ptr[cols + 1] = v11
    }
}

private func multiInputAddMIL(channels: Int) -> String {
    """
    program(1.3)
    [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
    {
        func main<ios18>(tensor<fp16, [1, \(channels), 1, 1]> a, tensor<fp16, [1, \(channels), 1, 1]> b) {
            tensor<fp16, [1,\(channels),1,1]> out = add(x=a,y=b)[name=string("out")];
        } -> (out);
    }
    """
}

private func singletonQueryMatmulMIL(heads: Int, headDim: Int, seq: Int) -> String {
    """
    program(1.3)
    [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
    {
        func main<ios18>(tensor<fp16, [1, \(heads), 1, \(headDim)]> q, tensor<fp16, [1, \(heads), \(seq), \(headDim)]> k) {
            bool tx = const()[name=string("tx"), val=bool(false)];
            bool ty = const()[name=string("ty"), val=bool(true)];
            tensor<fp16, [1,\(heads),1,\(seq)]> out = matmul(transpose_x=tx,transpose_y=ty,x=q,y=k)[name=string("out")];
        } -> (out);
    }
    """
}

private func decodePassthroughProbeMIL(dim: Int, laneSpatial: Int, maxSeq: Int, inputKinds: [Character], outputCount: Int) -> String {
    precondition(!inputKinds.isEmpty)
    precondition(outputCount == 1 || outputCount == 3)
    var parts: [String] = []
    for kind in inputKinds {
        switch kind {
        case "x":
            parts.append("tensor<fp16, [1, \(dim), 1, \(laneSpatial)]> x")
        case "k":
            parts.append("tensor<fp16, [1, \(dim), 1, \(maxSeq)]> kCache")
        case "v":
            parts.append("tensor<fp16, [1, \(dim), 1, \(maxSeq)]> vCache")
        case "q":
            parts.append("tensor<fp16, [1, \(2 * dim), 1, \(maxSeq)]> kvCache")
        case "m":
            parts.append("tensor<fp16, [1, 1, 1, \(maxSeq)]> maskVec")
        case "d":
            parts.append("tensor<fp16, [1, \(dim), 1, \(maxSeq)]> maskDense")
        default:
            preconditionFailure("Unknown input kind \(kind)")
        }
    }
    let signature = parts.joined(separator: ", ")
    var lines: [String] = []
    lines.append("program(1.3)")
    lines.append("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]")
    lines.append("{")
    lines.append("    func main<ios18>(\(signature)) {")
    lines.append("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> outX = add(x=x,y=x)[name=string(\"out_x\")];")
    if outputCount == 3 {
        lines.append("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> outK = add(x=x,y=x)[name=string(\"out_k\")];")
        lines.append("        tensor<fp16, [1,\(dim),1,\(laneSpatial)]> outV = add(x=x,y=x)[name=string(\"out_v\")];")
        lines.append("    } -> (outX,outK,outV);")
    } else {
        lines.append("    } -> (outX);")
    }
    lines.append("}")
    return lines.joined(separator: "\n")
}

final class ANERuntimeTests: XCTestCase {
    func test_ane_error_conforms_to_sendable_and_error() {
        func requireErrorAndSendable<T: Error & Sendable>(_: T.Type) {}
        requireErrorAndSendable(ANEError.self)

        let errors: [ANEError] = [
            .invalidArguments("x"),
            .compilationFailed,
            .evaluationFailed,
            .compileBudgetExhausted,
            .surfaceAllocationFailed,
            .invalidSurfaceIndex(0),
            .inputSurfaceUnavailable(0),
            .outputSurfaceUnavailable(0),
        ]
        let descriptions = errors.map(\.localizedDescription)
        XCTAssertEqual(Set(descriptions).count, errors.count)
    }

    func test_compile_identity_kernel_succeeds() throws {
        try requireANEHardwareTestsEnabled()

        let kernel = try makeIdentityKernel()
        let input = try kernel.inputSurface(at: 0)
        let output = try kernel.outputSurface(at: 0)

        XCTAssertEqual(IOSurfaceGetAllocSize(input), identityInputBytes)
        XCTAssertEqual(IOSurfaceGetAllocSize(output), identityOutputBytes)
    }

    func test_compile_invalid_mil_throws_compilation_failed() throws {
        try requireANEHardwareTestsEnabled()

        do {
            _ = try ANEKernel(
                milText: "this is not valid MIL",
                weights: [],
                inputBytes: identityInputBytes,
                outputBytes: identityOutputBytes
            )
            XCTFail("Expected compilation failure")
        } catch ANEError.compilationFailed {
            return
        } catch {
            XCTFail("Expected .compilationFailed, got \(error)")
        }
    }

    func test_eval_identity_roundtrip() throws {
        try requireANEHardwareTestsEnabled()

        let kernel = try makeIdentityKernel()
        let input = (1...identityElementCount).map(Float.init)
        var output = [Float](repeating: 0, count: identityElementCount)
        let inputSurface = try kernel.inputSurface(at: 0)
        let outputSurface = try kernel.outputSurface(at: 0)

        input.withUnsafeBufferPointer { inputBuf in
            SurfaceIO.writeFP16(
                to: inputSurface,
                data: inputBuf,
                channels: identityChannels,
                spatial: identitySpatial
            )
        }

        if !ANEEvalProbe.isAvailable {
            do {
                try kernel.eval()
                throw XCTSkip("ANE baseline probe unavailable; identity eval succeeded on this host")
            } catch ANEError.evaluationFailed {
                throw XCTSkip("ANE baseline probe unavailable; identity eval failed as expected on this host")
            } catch {
                throw XCTSkip("ANE baseline probe unavailable; identity eval produced unexpected error: \(error)")
            }
        }

        try kernel.eval()

        output.withUnsafeMutableBufferPointer { outputBuf in
            SurfaceIO.readFP16(
                from: outputSurface,
                into: outputBuf,
                channelOffset: 0,
                channels: identityChannels,
                spatial: identitySpatial
            )
        }

        for i in 0..<identityElementCount {
            XCTAssertEqual(output[i], input[i], accuracy: 1e-2)
        }
    }

    func test_eval_two_input_add_roundtrip() throws {
        try requireANEHardwareTestsEnabled()
        guard ANEEvalProbe.isAvailable else {
            throw XCTSkip("ANE baseline probe unavailable on this host")
        }

        let channels = 4
        let spatial = 1
        let bytes = channels * spatial * MemoryLayout<UInt16>.stride
        let kernel = try ANEKernel(
            milText: multiInputAddMIL(channels: channels),
            weights: [],
            inputSizes: [bytes, bytes],
            outputSizes: [bytes]
        )

        let inputA = try kernel.inputSurface(at: 0)
        let inputB = try kernel.inputSurface(at: 1)
        let output = try kernel.outputSurface(at: 0)

        let a: [Float] = [1, 2, 3, 4]
        let b: [Float] = [10, 20, 30, 40]
        var y = [Float](repeating: 0, count: channels)

        a.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: inputA, data: src, channels: channels, spatial: spatial)
        }
        b.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: inputB, data: src, channels: channels, spatial: spatial)
        }

        try kernel.eval()

        y.withUnsafeMutableBufferPointer { dst in
            SurfaceIO.readFP16(from: output, into: dst, channelOffset: 0, channels: channels, spatial: spatial)
        }

        XCTAssertEqual(y[0], 11, accuracy: 1e-2)
        XCTAssertEqual(y[1], 22, accuracy: 1e-2)
        XCTAssertEqual(y[2], 33, accuracy: 1e-2)
        XCTAssertEqual(y[3], 44, accuracy: 1e-2)
    }

    func test_eval_singleton_query_batched_matmul_roundtrip() throws {
        try requireANEHardwareTestsEnabled()
        guard ANEEvalProbe.isAvailable else {
            throw XCTSkip("ANE baseline probe unavailable on this host")
        }

        let heads = 2
        let headDim = 4
        let seq = 8

        let qCount = heads * headDim
        let kCount = heads * seq * headDim
        let outCount = heads * seq
        let qBytes = qCount * MemoryLayout<UInt16>.stride
        let kBytes = kCount * MemoryLayout<UInt16>.stride
        let outBytes = outCount * MemoryLayout<UInt16>.stride

        let kernel = try ANEKernel(
            milText: singletonQueryMatmulMIL(heads: heads, headDim: headDim, seq: seq),
            weights: [],
            inputSizes: [qBytes, kBytes],
            outputSizes: [outBytes]
        )

        let qSurface = try kernel.inputSurface(at: 0)
        let kSurface = try kernel.inputSurface(at: 1)
        let outSurface = try kernel.outputSurface(at: 0)

        let q: [Float] = [1, 2, 3, 4, 2, 1, 0, -1]
        var k = [Float](repeating: 0, count: kCount)
        for h in 0..<heads {
            for s in 0..<seq {
                for d in 0..<headDim {
                    let idx = h * seq * headDim + s * headDim + d
                    k[idx] = Float((h + 1) * (s + 1) * (d + 1)) * 0.1
                }
            }
        }

        q.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: qSurface, data: src, channels: heads * headDim, spatial: 1)
        }
        k.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(to: kSurface, data: src, channels: heads * seq * headDim, spatial: 1)
        }

        try kernel.eval()

        var y = [Float](repeating: 0, count: outCount)
        y.withUnsafeMutableBufferPointer { dst in
            SurfaceIO.readFP16(from: outSurface, into: dst, channelOffset: 0, channels: heads * seq, spatial: 1)
        }

        XCTAssertTrue(y.allSatisfy(\.isFinite))
        XCTAssertTrue(y.contains(where: { abs($0) > 0 }))
    }

    func test_decode_probe_passthrough_4in_3out_eval() throws {
        try requireANEHardwareTestsEnabled()

        let dim = ModelConfig.dim
        let lane = DecodeKernelSet.defaultLaneSpatial
        let maxSeq = 32
        let xBytes = dim * lane * MemoryLayout<UInt16>.stride
        let kvBytes = dim * maxSeq * MemoryLayout<UInt16>.stride
        let maskBytes = maxSeq * MemoryLayout<UInt16>.stride

        struct Variant {
            let name: String
            let inputKinds: [Character]
            let outputCount: Int
        }
        let variants: [Variant] = [
            .init(name: "x->1", inputKinds: ["x"], outputCount: 1),
            .init(name: "x->3", inputKinds: ["x"], outputCount: 3),
            .init(name: "xk->1", inputKinds: ["x", "k"], outputCount: 1),
            .init(name: "xkv->1", inputKinds: ["x", "k", "v"], outputCount: 1),
            .init(name: "xq->1", inputKinds: ["x", "q"], outputCount: 1),
            .init(name: "xm->1", inputKinds: ["x", "m"], outputCount: 1),
            .init(name: "xkm->1", inputKinds: ["x", "k", "m"], outputCount: 1),
            .init(name: "xqm->1", inputKinds: ["x", "q", "m"], outputCount: 1),
            .init(name: "xkvm->1", inputKinds: ["x", "k", "v", "m"], outputCount: 1),
            .init(name: "xkvm->3", inputKinds: ["x", "k", "v", "m"], outputCount: 3),
            .init(name: "xkvd->1", inputKinds: ["x", "k", "v", "d"], outputCount: 1),
        ]

        var successes: [String: Bool] = [:]
        for variant in variants {
            do {
                let inputSizes = variant.inputKinds.map { kind in
                    switch kind {
                    case "x": return xBytes
                    case "k", "v": return kvBytes
                    case "q": return 2 * kvBytes
                    case "d": return kvBytes
                    case "m": return maskBytes
                    default: preconditionFailure("unknown input kind \(kind)")
                    }
                }
                let outputSizes = Array(repeating: xBytes, count: variant.outputCount)
                let kernel = try ANEKernel(
                    milText: decodePassthroughProbeMIL(
                        dim: dim,
                        laneSpatial: lane,
                        maxSeq: maxSeq,
                        inputKinds: variant.inputKinds,
                        outputCount: variant.outputCount
                    ),
                    weights: [],
                    inputSizes: inputSizes,
                    outputSizes: outputSizes
                )

                for (inputIndex, kind) in variant.inputKinds.enumerated() {
                    let surface = try kernel.inputSurface(at: inputIndex)
                    switch kind {
                    case "x":
                        Array(repeating: Float(0.25), count: dim * lane).withUnsafeBufferPointer { src in
                            SurfaceIO.writeFP16(to: surface, data: src, channels: dim, spatial: lane)
                        }
                    case "k", "v":
                        Array(repeating: Float(0), count: dim * maxSeq).withUnsafeBufferPointer { src in
                            SurfaceIO.writeFP16(to: surface, data: src, channels: dim, spatial: maxSeq)
                        }
                    case "q":
                        Array(repeating: Float(0), count: 2 * dim * maxSeq).withUnsafeBufferPointer { src in
                            SurfaceIO.writeFP16(to: surface, data: src, channels: 2 * dim, spatial: maxSeq)
                        }
                    case "m":
                        Array(repeating: Float(-1e4), count: maxSeq).withUnsafeBufferPointer { src in
                            SurfaceIO.writeFP16(to: surface, data: src, channels: 1, spatial: maxSeq)
                        }
                    case "d":
                        Array(repeating: Float(-1e4), count: dim * maxSeq).withUnsafeBufferPointer { src in
                            SurfaceIO.writeFP16(to: surface, data: src, channels: dim, spatial: maxSeq)
                        }
                    default:
                        preconditionFailure("unknown input kind \(kind)")
                    }
                }

                try kernel.eval()
                let ySurface = try kernel.outputSurface(at: 0)
                var out = [Float](repeating: 0, count: dim * lane)
                out.withUnsafeMutableBufferPointer { dst in
                    SurfaceIO.readFP16(from: ySurface, into: dst, channelOffset: 0, channels: dim, spatial: lane)
                }
                successes[variant.name] = out.allSatisfy(\.isFinite)
            } catch {
                print("decode probe variant \(variant.name) failed: \(error)")
                successes[variant.name] = false
            }
        }

        print("decode probe matrix: \(successes)")
        XCTAssertEqual(successes["x->1"], true)
    }

    func test_eval_returns_throws_on_failure() throws {
        try requireANEHardwareTestsEnabled()
        let kernel = try makeIdentityKernel()
        ane_interop_set_force_eval_failure(true)
        defer { ane_interop_set_force_eval_failure(false) }

        do {
            try kernel.eval()
            XCTFail("Expected .evaluationFailed")
        } catch ANEError.evaluationFailed {
            return
        } catch {
            XCTFail("Expected .evaluationFailed, got \(error)")
        }
    }

    func test_kernel_deinit_frees_handle() throws {
        try requireANEHardwareTestsEnabled()
        let baseline = ane_interop_live_handle_count()

        do {
            let kernel = try makeIdentityKernel()
            XCTAssertEqual(ane_interop_live_handle_count(), baseline + 1)
            XCTAssertEqual(IOSurfaceGetAllocSize(try kernel.inputSurface(at: 0)), identityInputBytes)
        }
        XCTAssertEqual(ane_interop_live_handle_count(), baseline)

        do {
            let secondKernel = try makeIdentityKernel()
            XCTAssertEqual(ane_interop_live_handle_count(), baseline + 1)
            XCTAssertEqual(IOSurfaceGetAllocSize(try secondKernel.outputSurface(at: 0)), identityOutputBytes)
        }
        XCTAssertEqual(ane_interop_live_handle_count(), baseline)
    }

    func test_compile_budget_tracks_count() throws {
        try requireANEHardwareTestsEnabled()

        let previous = CompileBudget.currentCount
        defer { try? CompileBudget.setCount(previous) }

        do {
            let kernel = try makeIdentityKernel(checkBudget: false)
            _ = try kernel.inputSurface(at: 0)
        }

        XCTAssertEqual(CompileBudget.currentCount, previous + 1)
    }

    func test_compile_budget_exhausted_blocks_compile() throws {
        let previous = CompileBudget.currentCount
        defer { try? CompileBudget.setCount(previous) }

        try CompileBudget.setCount(CompileBudget.maxCompiles)
        XCTAssertTrue(CompileBudget.isExhausted)

        do {
            _ = try makeIdentityKernel(checkBudget: true)
            XCTFail("Expected .compileBudgetExhausted")
        } catch ANEError.compileBudgetExhausted {
            XCTAssertEqual(CompileBudget.currentCount, CompileBudget.maxCompiles)
        } catch {
            XCTFail("Expected .compileBudgetExhausted, got \(error)")
        }
    }

    func test_delta_reload_initializer_rejects_empty_donor_hex_id() {
        let mil = GenericMIL.conv(inCh: identityChannels, outCh: identityChannels, spatial: identitySpatial)
        let weightBlob = makeIdentityWeightBlob(channels: identityChannels)

        do {
            _ = try ANEKernel(
                milText: mil,
                weights: [(path: identityWeightPath, data: weightBlob)],
                inputSizes: [identityInputBytes],
                outputSizes: [identityOutputBytes],
                donorHexId: ""
            )
            XCTFail("Expected .invalidArguments for empty donorHexId")
        } catch ANEError.invalidArguments {
        } catch {
            XCTFail("Expected .invalidArguments, got \(error)")
        }
    }

    func test_delta_reload_initializer_does_not_increment_compile_budget() throws {
        try requireANEHardwareTestsEnabled()

        let previous = CompileBudget.currentCount
        defer { try? CompileBudget.setCount(previous) }

        let donor = try makeIdentityKernel(checkBudget: false)
        let afterDonor = CompileBudget.currentCount

        let mil = GenericMIL.conv(inCh: identityChannels, outCh: identityChannels, spatial: identitySpatial)
        let weightBlob = makeIdentityWeightBlob(channels: identityChannels)
        let reloaded = try ANEKernel(
            milText: mil,
            weights: [(path: identityWeightPath, data: weightBlob)],
            inputSizes: [identityInputBytes],
            outputSizes: [identityOutputBytes],
            donorHexId: donor.hexId
        )

        XCTAssertFalse(reloaded.hexId.isEmpty)
        XCTAssertEqual(
            CompileBudget.currentCount,
            afterDonor,
            "Delta reload should reuse donor compilation without consuming compile budget"
        )
    }

    func test_compile_retry_policy_retries_only_generic_compiler_failures() {
        XCTAssertTrue(
            ANECompileRetryPolicy.shouldRetry(
                lastCompileError: ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE,
                attemptIndex: 0
            )
        )
        XCTAssertTrue(
            ANECompileRetryPolicy.shouldRetry(
                lastCompileError: ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE,
                attemptIndex: 1
            )
        )
        XCTAssertFalse(
            ANECompileRetryPolicy.shouldRetry(
                lastCompileError: ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE,
                attemptIndex: ANECompileRetryPolicy.maxAttempts - 1
            )
        )
        XCTAssertFalse(
            ANECompileRetryPolicy.shouldRetry(
                lastCompileError: ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS,
                attemptIndex: 0
            )
        )
        XCTAssertFalse(
            ANECompileRetryPolicy.shouldRetry(
                lastCompileError: ANE_INTEROP_COMPILE_ERROR_SURFACE_ALLOCATION_FAILED,
                attemptIndex: 0
            )
        )
    }

    func test_compile_retry_policy_backoff_is_bounded() {
        XCTAssertEqual(
            ANECompileRetryPolicy.delayMicroseconds(afterFailedAttempt: 0),
            ANECompileRetryPolicy.initialDelayMicroseconds
        )
        XCTAssertEqual(
            ANECompileRetryPolicy.delayMicroseconds(afterFailedAttempt: 1),
            200_000
        )
        XCTAssertEqual(
            ANECompileRetryPolicy.delayMicroseconds(afterFailedAttempt: 2),
            400_000
        )
        XCTAssertEqual(
            ANECompileRetryPolicy.delayMicroseconds(afterFailedAttempt: 5),
            ANECompileRetryPolicy.maxDelayMicroseconds
        )
    }

    func test_compile_retry_policy_notice_reports_next_attempt() {
        XCTAssertEqual(
            ANECompileRetryPolicy.retryNotice(afterFailedAttempt: 0),
            "ANE compile retrying (2/5) after transient compiler failure"
        )
        XCTAssertEqual(
            ANECompileRetryPolicy.retryNotice(afterFailedAttempt: 1),
            "ANE compile retrying (3/5) after transient compiler failure"
        )
        XCTAssertEqual(
            ANECompileRetryPolicy.retryNotice(afterFailedAttempt: 3),
            "ANE compile retrying (5/5) after transient compiler failure"
        )
    }

    func test_compile_with_retry_retries_until_attempt_succeeds() throws {
        var attemptCount = 0
        var sleepCalls: [Int] = []
        let handle = try ANEKernel.compileWithRetry(
            checkBudget: false,
            compileAttempt: {
                defer { attemptCount += 1 }
                if attemptCount < 2 {
                    return nil
                }
                return OpaquePointer(bitPattern: 0x1)
            },
            lastCompileError: { ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE },
            sleepAfterFailedAttempt: { sleepCalls.append($0) },
            writeRetryNotice: { _ in }
        )

        XCTAssertEqual(handle, OpaquePointer(bitPattern: 0x1))
        XCTAssertEqual(attemptCount, 3)
        XCTAssertEqual(sleepCalls, [0, 1])
    }

    func test_compile_with_retry_stops_without_sleep_for_non_retryable_error() throws {
        var attemptCount = 0
        var slept = false
        let handle = try ANEKernel.compileWithRetry(
            checkBudget: false,
            compileAttempt: {
                attemptCount += 1
                return nil
            },
            lastCompileError: { ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS },
            sleepAfterFailedAttempt: { _ in slept = true },
            writeRetryNotice: { _ in }
        )

        XCTAssertNil(handle)
        XCTAssertEqual(attemptCount, 1)
        XCTAssertFalse(slept)
    }

    func test_compile_with_retry_releases_budget_gate_before_sleeping() throws {
        final class AttemptState: @unchecked Sendable {
            private let lock = NSLock()
            private var firstAttemptCount = 0
            private(set) var secondAttemptCount = 0

            func firstCompileAttempt() -> OpaquePointer? {
                lock.lock()
                defer {
                    firstAttemptCount += 1
                    lock.unlock()
                }
                if firstAttemptCount == 0 {
                    return nil
                }
                return OpaquePointer(bitPattern: 0x11)
            }

            func secondCompileAttempt() -> OpaquePointer? {
                lock.lock()
                secondAttemptCount += 1
                lock.unlock()
                return OpaquePointer(bitPattern: 0x22)
            }
        }

        let previous = CompileBudget.currentCount
        defer { try? CompileBudget.setCount(previous) }
        try CompileBudget.setCount(0)

        let firstSleepStarted = expectation(description: "first compile entered retry sleep")
        let secondCompileEntered = expectation(description: "second compile entered while first slept")
        let allowFirstRetryToContinue = DispatchSemaphore(value: 0)
        let firstThreadDone = DispatchSemaphore(value: 0)
        let secondThreadDone = DispatchSemaphore(value: 0)
        let attemptState = AttemptState()

        DispatchQueue.global().async {
            defer { firstThreadDone.signal() }
            do {
                _ = try ANEKernel.compileWithRetry(
                    checkBudget: true,
                    compileAttempt: attemptState.firstCompileAttempt,
                    lastCompileError: { ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE },
                    sleepAfterFailedAttempt: { _ in
                        firstSleepStarted.fulfill()
                        _ = allowFirstRetryToContinue.wait(timeout: .now() + 2.0)
                    },
                    writeRetryNotice: { _ in }
                )
            } catch {
                XCTFail("unexpected first compile error: \(error)")
            }
        }

        wait(for: [firstSleepStarted], timeout: 2.0)

        DispatchQueue.global().async {
            defer {
                secondThreadDone.signal()
            }
            do {
                _ = try ANEKernel.compileWithRetry(
                    checkBudget: true,
                    compileAttempt: {
                        secondCompileEntered.fulfill()
                        return attemptState.secondCompileAttempt()
                    },
                    lastCompileError: { ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE },
                    sleepAfterFailedAttempt: { _ in XCTFail("second compile should not sleep") },
                    writeRetryNotice: { _ in }
                )
            } catch {
                XCTFail("unexpected second compile error: \(error)")
            }
        }

        wait(for: [secondCompileEntered], timeout: 2.0)
        allowFirstRetryToContinue.signal()

        XCTAssertEqual(firstThreadDone.wait(timeout: .now() + 2.0), .success)
        XCTAssertEqual(secondThreadDone.wait(timeout: .now() + 2.0), .success)
        XCTAssertEqual(attemptState.secondAttemptCount, 1)
    }

    func test_compile_budget_boundary_allows_only_one_concurrent_compile() throws {
        try requireANEHardwareTestsEnabled()

        final class CompileOutcomes: @unchecked Sendable {
            private let lock = NSLock()
            private(set) var successCount = 0
            private(set) var errors = [ANEError]()

            func recordSuccess() {
                lock.lock()
                successCount += 1
                lock.unlock()
            }

            func recordFailure(_ error: ANEError) {
                lock.lock()
                errors.append(error)
                lock.unlock()
            }
        }

        let previous = CompileBudget.currentCount
        defer { try? CompileBudget.setCount(previous) }
        try CompileBudget.setCount(CompileBudget.maxCompiles - 1)

        let outcomes = CompileOutcomes()

        DispatchQueue.concurrentPerform(iterations: 2) { _ in
            do {
                _ = try makeIdentityKernel(checkBudget: true)
                outcomes.recordSuccess()
            } catch let error as ANEError {
                outcomes.recordFailure(error)
            } catch {
                outcomes.recordFailure(.compilationFailed)
            }
        }

        XCTAssertEqual(outcomes.successCount, 1, "Exactly one compile should reserve the final budget slot")

        let exhaustedCount = outcomes.errors.reduce(into: 0) { partialResult, error in
            if case .compileBudgetExhausted = error {
                partialResult += 1
            }
        }
        XCTAssertEqual(exhaustedCount, 1, "One compile should fail with .compileBudgetExhausted")
    }

    func test_layer_kernel_set_compile_time_under_2000ms() throws {
        try requireANEHardwareTestsEnabled()

        let layerWeights = LayerWeights()
        fillLayerWeights(layerWeights, value: 0.01)

        let start = Date()
        _ = try LayerKernelSet(weights: layerWeights)
        let elapsedMs = Date().timeIntervalSince(start) * 1000.0
        XCTAssertLessThan(elapsedMs, 2000.0, "compile took \(elapsedMs)ms")
    }

    func test_surface_access_invalid_index_throws_typed_error() throws {
        try requireANEHardwareTestsEnabled()
        let kernel = try makeIdentityKernel()

        do {
            _ = try kernel.inputSurface(at: -1)
            XCTFail("Expected .invalidSurfaceIndex")
        } catch ANEError.invalidSurfaceIndex(-1) {
            // expected
        } catch {
            XCTFail("Expected .invalidSurfaceIndex(-1), got \(error)")
        }

        do {
            _ = try kernel.outputSurface(at: Int(Int32.max) + 1)
            XCTFail("Expected .invalidSurfaceIndex")
        } catch ANEError.invalidSurfaceIndex(Int(Int32.max) + 1) {
            // expected
        } catch {
            XCTFail("Expected .invalidSurfaceIndex(Int32.max + 1), got \(error)")
        }
    }

    func test_layer_kernel_set_compiles_all_five_and_surface_sizes() throws {
        try requireANEHardwareTestsEnabled()

        let layerWeights = LayerWeights()
        fillLayerWeights(layerWeights, value: 0.01)

        let kernels = try LayerKernelSet(weights: layerWeights)

        XCTAssertGreaterThanOrEqual(IOSurfaceGetAllocSize(try kernels.fwdAttn.inputSurface(at: 0)), fwdAttnInputBytes)
        XCTAssertGreaterThanOrEqual(IOSurfaceGetAllocSize(try kernels.fwdAttn.outputSurface(at: 0)), fwdAttnOutputBytes)

        XCTAssertGreaterThanOrEqual(IOSurfaceGetAllocSize(try kernels.fwdFFN.inputSurface(at: 0)), fwdFFNInputBytes)
        XCTAssertGreaterThanOrEqual(IOSurfaceGetAllocSize(try kernels.fwdFFN.outputSurface(at: 0)), fwdFFNOutputBytes)

        XCTAssertGreaterThanOrEqual(IOSurfaceGetAllocSize(try kernels.ffnBwd.inputSurface(at: 0)), ffnBwdInputBytes)
        XCTAssertGreaterThanOrEqual(IOSurfaceGetAllocSize(try kernels.ffnBwd.outputSurface(at: 0)), ffnBwdOutputBytes)

        XCTAssertGreaterThanOrEqual(IOSurfaceGetAllocSize(try kernels.sdpaBwd1.inputSurface(at: 0)), sdpaBwd1InputBytes)
        XCTAssertGreaterThanOrEqual(IOSurfaceGetAllocSize(try kernels.sdpaBwd1.outputSurface(at: 0)), sdpaBwd1OutputBytes)

        XCTAssertGreaterThanOrEqual(IOSurfaceGetAllocSize(try kernels.qkvBwd.inputSurface(at: 0)), qkvBwdInputBytes)
        XCTAssertGreaterThanOrEqual(IOSurfaceGetAllocSize(try kernels.qkvBwd.outputSurface(at: 0)), qkvBwdOutputBytes)
    }

    func test_fwd_attn_output_has_6xdim_channels() throws {
        try requireANEHardwareTestsEnabled()

        let dim = ModelConfig.dim
        let seqLen = ModelConfig.seqLen
        let layerWeights = LayerWeights()
        fillLayerWeights(layerWeights, value: 0.01)
        let kernels = try LayerKernelSet(weights: layerWeights)

        let inputSurface = try kernels.fwdAttn.inputSurface(at: 0)
        let outputSurface = try kernels.fwdAttn.outputSurface(at: 0)

        var input = [Float](repeating: 0, count: dim * seqLen)
        for i in input.indices {
            input[i] = Float(i % 64 + 1) * 0.01
        }
        input.withUnsafeBufferPointer { buf in
            SurfaceIO.writeFP16(to: inputSurface, data: buf, channels: dim, spatial: seqLen)
        }

        try kernels.fwdAttn.eval()

        let offsets = [0, dim, 2 * dim, 3 * dim, 4 * dim, 5 * dim]
        for offset in offsets {
            var region = [Float](repeating: 0, count: dim * seqLen)
            region.withUnsafeMutableBufferPointer { out in
                SurfaceIO.readFP16(
                    from: outputSurface,
                    into: out,
                    channelOffset: offset,
                    channels: dim,
                    spatial: seqLen
                )
            }

            XCTAssertTrue(region.allSatisfy(\.isFinite), "Non-finite values at offset \(offset)")
            XCTAssertTrue(region.contains(where: { $0 != 0 }), "All-zero region at offset \(offset)")
        }
    }

    func test_fwd_attn_numerical_equivalence_with_objc() throws {
        try requireANEHardwareTestsEnabled()
        try requireObjCCrossValidation()

        let dim = ModelConfig.dim
        let seqLen = ModelConfig.seqLen
        let expectedOOut = try loadFloat32LEFixture(
            "fwd_attn_oOut_seq256_f32le.bin",
            expectedCount: dim * seqLen
        )

        let layerWeights = LayerWeights()
        fillLayerWeights(layerWeights, value: 0.01)
        let kernels = try LayerKernelSet(weights: layerWeights)

        let inputSurface = try kernels.fwdAttn.inputSurface(at: 0)
        let outputSurface = try kernels.fwdAttn.outputSurface(at: 0)

        var input = [Float](repeating: 0, count: dim * seqLen)
        for i in 0..<(dim * seqLen) {
            input[i] = Float(i % 64 + 1) * 0.01
        }
        input.withUnsafeBufferPointer { buf in
            SurfaceIO.writeFP16(to: inputSurface, data: buf, channels: dim, spatial: seqLen)
        }

        try kernels.fwdAttn.eval()

        var oOut = [Float](repeating: 0, count: dim * seqLen)
        oOut.withUnsafeMutableBufferPointer { out in
            SurfaceIO.readFP16(from: outputSurface, into: out, channelOffset: 0, channels: dim, spatial: seqLen)
        }

        let worst = maxAbsDiff(actual: oOut, expected: expectedOOut)
        XCTAssertLessThan(
            worst.diff,
            1e-2,
            "max diff=\(worst.diff) at idx \(worst.index), actual=\(worst.actual), expected=\(worst.expected)"
        )
    }

    func test_fwd_ffn_numerical_equivalence_with_objc() throws {
        try requireANEHardwareTestsEnabled()
        try requireObjCCrossValidation()

        let dim = ModelConfig.dim
        let seqLen = ModelConfig.seqLen
        let expectedY = try loadFloat32LEFixture(
            "fwd_ffn_y_seq256_f32le.bin",
            expectedCount: dim * seqLen
        )

        let layerWeights = LayerWeights()
        fillLayerWeights(layerWeights, value: 0.01)
        let kernels = try LayerKernelSet(weights: layerWeights)

        let attnIn = try kernels.fwdAttn.inputSurface(at: 0)
        let attnOut = try kernels.fwdAttn.outputSurface(at: 0)
        let ffnIn = try kernels.fwdFFN.inputSurface(at: 0)
        let ffnOut = try kernels.fwdFFN.outputSurface(at: 0)

        var input = [Float](repeating: 0, count: dim * seqLen)
        for i in 0..<(dim * seqLen) {
            input[i] = Float(i % 64 + 1) * 0.01
        }
        input.withUnsafeBufferPointer { buf in
            SurfaceIO.writeFP16(to: attnIn, data: buf, channels: dim, spatial: seqLen)
        }

        try kernels.fwdAttn.eval()

        var oOut = [Float](repeating: 0, count: dim * seqLen)
        oOut.withUnsafeMutableBufferPointer { out in
            SurfaceIO.readFP16(from: attnOut, into: out, channelOffset: 0, channels: dim, spatial: seqLen)
        }
        oOut.withUnsafeBufferPointer { buf in
            SurfaceIO.writeFP16(to: ffnIn, data: buf, channels: dim, spatial: seqLen)
        }

        try kernels.fwdFFN.eval()

        var y = [Float](repeating: 0, count: dim * seqLen)
        y.withUnsafeMutableBufferPointer { out in
            SurfaceIO.readFP16(from: ffnOut, into: out, channelOffset: 0, channels: dim, spatial: seqLen)
        }

        let worst = maxAbsDiff(actual: y, expected: expectedY)
        XCTAssertLessThan(
            worst.diff,
            1e-2,
            "max diff=\(worst.diff) at idx \(worst.index), actual=\(worst.actual), expected=\(worst.expected)"
        )
    }

    func test_ffn_bwd_numerical_equivalence_with_objc() throws {
        try requireANEHardwareTestsEnabled()
        try requireObjCCrossValidation()

        let dim = ModelConfig.dim
        let hidden = ModelConfig.hidden
        let seqLen = ModelConfig.seqLen
        let expectedDX = try loadFloat32LEFixture(
            "ffn_bwd_dx_seq256_f32le.bin",
            expectedCount: dim * seqLen
        )

        let layerWeights = LayerWeights()
        fillLayerWeights(layerWeights, value: 0.01)
        let kernels = try LayerKernelSet(weights: layerWeights)

        let ffnBwdIn = try kernels.ffnBwd.inputSurface(at: 0)
        let ffnBwdOut = try kernels.ffnBwd.outputSurface(at: 0)

        var ffnBwdInput = [Float](repeating: 0, count: (dim + 2 * hidden) * seqLen)
        for i in 0..<((dim + 2 * hidden) * seqLen) {
            ffnBwdInput[i] = Float(i % 128 + 1) * 0.005
        }
        ffnBwdInput.withUnsafeBufferPointer { buf in
            SurfaceIO.writeFP16(
                to: ffnBwdIn,
                data: buf,
                channels: dim + 2 * hidden,
                spatial: seqLen
            )
        }

        try kernels.ffnBwd.eval()

        var dx = [Float](repeating: 0, count: dim * seqLen)
        dx.withUnsafeMutableBufferPointer { out in
            SurfaceIO.readFP16(from: ffnBwdOut, into: out, channelOffset: 0, channels: dim, spatial: seqLen)
        }

        let worst = maxAbsDiff(actual: dx, expected: expectedDX)
        XCTAssertLessThan(
            worst.diff,
            1e-2,
            "max diff=\(worst.diff) at idx \(worst.index), actual=\(worst.actual), expected=\(worst.expected)"
        )
    }

    func test_layer_kernel_set_compile_specs_match_paths_and_io_without_hardware() throws {
        let layerWeights = LayerWeights()
        fillLayerWeights(layerWeights, value: 0.01)

        let specs = LayerKernelSet.compileSpecs(weights: layerWeights)
        XCTAssertEqual(specs.count, 5)

        let byKind = Dictionary(uniqueKeysWithValues: specs.map { ($0.kind, $0) })

        XCTAssertEqual(
            byKind[.fwdAttn]?.weightPaths ?? [],
            [
                "@model_path/weights/rms1.bin",
                "@model_path/weights/wq.bin",
                "@model_path/weights/wk.bin",
                "@model_path/weights/wv.bin",
                "@model_path/weights/wo.bin",
                "@model_path/weights/mask.bin",
            ]
        )
        XCTAssertEqual(byKind[.fwdAttn]?.inputBytes, fwdAttnInputBytes)
        XCTAssertEqual(byKind[.fwdAttn]?.outputBytes, fwdAttnOutputBytes)

        XCTAssertEqual(
            byKind[.fwdFFN]?.weightPaths ?? [],
            [
                "@model_path/weights/rms2.bin",
                "@model_path/weights/w1.bin",
                "@model_path/weights/w3.bin",
                "@model_path/weights/w2.bin",
            ]
        )
        XCTAssertEqual(byKind[.fwdFFN]?.inputBytes, fwdFFNInputBytes)
        XCTAssertEqual(byKind[.fwdFFN]?.outputBytes, fwdFFNOutputBytes)

        XCTAssertEqual(
            byKind[.ffnBwd]?.weightPaths ?? [],
            [
                "@model_path/weights/w2t.bin",
                "@model_path/weights/w1t.bin",
                "@model_path/weights/w3t.bin",
            ]
        )
        XCTAssertEqual(byKind[.ffnBwd]?.inputBytes, ffnBwdInputBytes)
        XCTAssertEqual(byKind[.ffnBwd]?.outputBytes, ffnBwdOutputBytes)

        XCTAssertEqual(
            byKind[.sdpaBwd1]?.weightPaths ?? [],
            [
                "@model_path/weights/mask.bin",
                "@model_path/weights/wot.bin",
            ]
        )
        XCTAssertEqual(byKind[.sdpaBwd1]?.inputBytes, sdpaBwd1InputBytes)
        XCTAssertEqual(byKind[.sdpaBwd1]?.outputBytes, sdpaBwd1OutputBytes)

        XCTAssertEqual(
            byKind[.qkvBwd]?.weightPaths ?? [],
            [
                "@model_path/weights/wqt.bin",
                "@model_path/weights/wkt.bin",
                "@model_path/weights/wvt.bin",
            ]
        )
        XCTAssertEqual(byKind[.qkvBwd]?.inputBytes, qkvBwdInputBytes)
        XCTAssertEqual(byKind[.qkvBwd]?.outputBytes, qkvBwdOutputBytes)
    }

    func test_layer_kernel_set_transposed_blob_mapping_without_hardware() throws {
        let layerWeights = LayerWeights()
        fillLayerWeights(layerWeights, value: 0)

        set2x2Sentinel(layerWeights.W2, cols: ModelConfig.hidden, v00: 1, v01: 2, v10: 3, v11: 4)
        set2x2Sentinel(layerWeights.Wq, cols: ModelConfig.dim, v00: 5, v01: 6, v10: 7, v11: 8)
        set2x2Sentinel(layerWeights.Wo, cols: ModelConfig.dim, v00: 9, v01: 10, v10: 11, v11: 12)

        let specs = LayerKernelSet.compileSpecs(weights: layerWeights)
        let byKind = Dictionary(uniqueKeysWithValues: specs.map { ($0.kind, $0) })

        guard
            let fwdFFN = byKind[.fwdFFN],
            let ffnBwd = byKind[.ffnBwd],
            let fwdAttn = byKind[.fwdAttn],
            let sdpaBwd1 = byKind[.sdpaBwd1],
            let qkvBwd = byKind[.qkvBwd]
        else {
            XCTFail("Missing one or more layer compile specs")
            return
        }

        guard
            let w2 = fwdFFN.weights.first(where: { $0.path == "@model_path/weights/w2.bin" })?.data,
            let w2t = ffnBwd.weights.first(where: { $0.path == "@model_path/weights/w2t.bin" })?.data,
            let wq = fwdAttn.weights.first(where: { $0.path == "@model_path/weights/wq.bin" })?.data,
            let wqt = qkvBwd.weights.first(where: { $0.path == "@model_path/weights/wqt.bin" })?.data,
            let wot = sdpaBwd1.weights.first(where: { $0.path == "@model_path/weights/wot.bin" })?.data
        else {
            XCTFail("Missing expected weight blobs in compile specs")
            return
        }

        XCTAssertEqual(blobPayloadFP16(w2, index: 0), 1, accuracy: 1e-3)
        XCTAssertEqual(blobPayloadFP16(w2, index: 1), 2, accuracy: 1e-3)
        XCTAssertEqual(blobPayloadFP16(w2, index: ModelConfig.hidden), 3, accuracy: 1e-3)
        XCTAssertEqual(blobPayloadFP16(w2, index: ModelConfig.hidden + 1), 4, accuracy: 1e-3)

        XCTAssertEqual(blobPayloadFP16(w2t, index: 0), 1, accuracy: 1e-3)
        XCTAssertEqual(blobPayloadFP16(w2t, index: 1), 3, accuracy: 1e-3)
        XCTAssertEqual(blobPayloadFP16(w2t, index: ModelConfig.dim), 2, accuracy: 1e-3)
        XCTAssertEqual(blobPayloadFP16(w2t, index: ModelConfig.dim + 1), 4, accuracy: 1e-3)

        XCTAssertEqual(blobPayloadFP16(wq, index: 0), 5, accuracy: 1e-3)
        XCTAssertEqual(blobPayloadFP16(wq, index: 1), 6, accuracy: 1e-3)
        XCTAssertEqual(blobPayloadFP16(wqt, index: 0), 5, accuracy: 1e-3)
        XCTAssertEqual(blobPayloadFP16(wqt, index: 1), 7, accuracy: 1e-3)
        XCTAssertEqual(blobPayloadFP16(wqt, index: ModelConfig.dim), 6, accuracy: 1e-3)
        XCTAssertEqual(blobPayloadFP16(wqt, index: ModelConfig.dim + 1), 8, accuracy: 1e-3)

        XCTAssertEqual(blobPayloadFP16(wot, index: 0), 9, accuracy: 1e-3)
        XCTAssertEqual(blobPayloadFP16(wot, index: 1), 11, accuracy: 1e-3)
        XCTAssertEqual(blobPayloadFP16(wot, index: ModelConfig.dim), 10, accuracy: 1e-3)
        XCTAssertEqual(blobPayloadFP16(wot, index: ModelConfig.dim + 1), 12, accuracy: 1e-3)
    }

    func test_decode_kernel_set_compile_specs_match_paths_and_io_without_hardware() throws {
        let layerWeights = LayerWeights()
        fillLayerWeights(layerWeights, value: 0.01)

        let lane = DecodeKernelSet.defaultLaneSpatial
        let maxSeq = lane * 4
        let specs = DecodeKernelSet.compileSpecs(weights: layerWeights, maxSeq: maxSeq)
        XCTAssertEqual(specs.count, 2)

        let byKind = Dictionary(uniqueKeysWithValues: specs.map { ($0.kind, $0) })
        guard let attn = byKind[.decodeAttnQKV], let ffn = byKind[.decodeFFN] else {
            XCTFail("Missing decode compile specs")
            return
        }
        XCTAssertEqual(
            attn.weights.map(\.path),
            [
                "@model_path/weights/rms1.bin",
                "@model_path/weights/wq.bin",
                "@model_path/weights/wk.bin",
                "@model_path/weights/wv.bin",
                "@model_path/weights/wo.bin",
            ]
        )
        XCTAssertEqual(
            attn.inputSizes,
            [
                ModelConfig.dim * lane * 2,
                ModelConfig.dim * lane * 2,
                ModelConfig.dim * lane * 2,
                ModelConfig.dim * lane * 2,
            ]
        )
        XCTAssertEqual(
            attn.outputSizes,
            [ModelConfig.dim * lane * 2, ModelConfig.dim * lane * 2, ModelConfig.dim * lane * 2]
        )

        XCTAssertEqual(
            ffn.weights.map(\.path),
            [
                "@model_path/weights/rms2.bin",
                "@model_path/weights/w1.bin",
                "@model_path/weights/w3.bin",
                "@model_path/weights/w2.bin",
            ]
        )
        XCTAssertEqual(ffn.inputSizes, [ModelConfig.dim * lane * 2])
        XCTAssertEqual(ffn.outputSizes, [ModelConfig.dim * lane * 2])
    }

    func test_layer_kernel_set_partial_compile_failure_cleanup() throws {
        try requireANEHardwareTestsEnabled()

        let baselineHandles = ane_interop_live_handle_count()
        let previousCount = CompileBudget.currentCount
        defer { try? CompileBudget.setCount(previousCount) }

        // CompileBudget.maxCompiles is fixed, so reserve only two slots.
        try CompileBudget.setCount(CompileBudget.maxCompiles - 2)

        let layerWeights = LayerWeights()
        fillLayerWeights(layerWeights, value: 0.01)

        do {
            _ = try LayerKernelSet(weights: layerWeights)
            XCTFail("Expected .compileBudgetExhausted")
        } catch ANEError.compileBudgetExhausted {
            XCTAssertEqual(ane_interop_live_handle_count(), baselineHandles)
        } catch {
            XCTFail("Expected .compileBudgetExhausted, got \(error)")
        }
    }

    func test_layer_kernel_set_deinit_frees_all_handles() throws {
        try requireANEHardwareTestsEnabled()

        let baselineHandles = ane_interop_live_handle_count()

        do {
            let layerWeights = LayerWeights()
            fillLayerWeights(layerWeights, value: 0.01)
            let kernels = try LayerKernelSet(weights: layerWeights)
            _ = try kernels.fwdAttn.inputSurface(at: 0)
            XCTAssertEqual(ane_interop_live_handle_count(), baselineHandles + 5)
        }

        XCTAssertEqual(ane_interop_live_handle_count(), baselineHandles)
    }

    func test_static_kernel_compiles_without_weights() throws {
        try requireANEHardwareTestsEnabled()

        let staticKernel = try StaticKernel()
        XCTAssertGreaterThanOrEqual(IOSurfaceGetAllocSize(try staticKernel.kernel.inputSurface(at: 0)), sdpaBwd2InputBytes)
        XCTAssertGreaterThanOrEqual(IOSurfaceGetAllocSize(try staticKernel.kernel.outputSurface(at: 0)), sdpaBwd2OutputBytes)
    }

    func test_static_kernel_compile_contract_without_hardware() {
        let contract = StaticKernel.compileContract
        XCTAssertEqual(contract.weightCount, 0)
        XCTAssertEqual(contract.inputBytes, sdpaBwd2InputBytes)
        XCTAssertEqual(contract.outputBytes, sdpaBwd2OutputBytes)
    }

    func test_static_kernel_survives_layer_kernel_set_dealloc() throws {
        try requireANEHardwareTestsEnabled()

        let staticKernel = try StaticKernel()
        _ = try staticKernel.kernel.inputSurface(at: 0)
        _ = try staticKernel.kernel.outputSurface(at: 0)

        do {
            let layerWeights = LayerWeights()
            fillLayerWeights(layerWeights, value: 0.01)
            let kernels = try LayerKernelSet(weights: layerWeights)
            _ = try kernels.sdpaBwd1.outputSurface(at: 0)
        }

        _ = try staticKernel.kernel.inputSurface(at: 0)
        _ = try staticKernel.kernel.outputSurface(at: 0)
    }

    func test_model_weight_loader_config_mismatch() throws {
        let path = makeTempBinaryPath(prefix: "cfg-mismatch")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let header = llamaHeaderData(
            dim: 4,
            hiddenDim: 8,
            nLayers: 1,
            nHeads: 1,
            nKvHeads: 1,
            vocabSize: 3,
            seqLen: 2
        )
        try header.write(to: URL(fileURLWithPath: path))

        do {
            _ = try ModelWeightLoader.load(from: path)
            XCTFail("Expected .configMismatch")
        } catch let ModelLoadError.configMismatch(expected, got) {
            XCTAssertTrue(expected.contains("768"))
            XCTAssertTrue(got.contains("4"))
        } catch {
            XCTFail("Expected .configMismatch, got \(error)")
        }
    }

    func test_model_weight_loader_vocab_mismatch_fails_fast() throws {
        let path = makeTempBinaryPath(prefix: "vocab-mismatch")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let header = llamaHeaderData(
            dim: Int32(ModelConfig.dim),
            hiddenDim: Int32(ModelConfig.hidden),
            nLayers: Int32(ModelConfig.nLayers),
            nHeads: Int32(ModelConfig.heads),
            nKvHeads: Int32(ModelConfig.heads),
            vocabSize: Int32(ModelConfig.vocab - 1),
            seqLen: Int32(ModelConfig.seqLen)
        )
        try header.write(to: URL(fileURLWithPath: path))

        do {
            _ = try ModelWeightLoader.load(from: path)
            XCTFail("Expected .configMismatch for vocab mismatch")
        } catch ModelLoadError.configMismatch {
            // Expected.
        } catch {
            XCTFail("Expected .configMismatch, got \(error)")
        }
    }

    func test_model_weight_loader_file_not_found() throws {
        let path = "/nonexistent/path/model-\(UUID().uuidString).bin"

        do {
            _ = try ModelWeightLoader.load(from: path)
            XCTFail("Expected .fileNotFound")
        } catch let ModelLoadError.fileNotFound(actualPath) {
            XCTAssertEqual(actualPath, path)
        } catch {
            XCTFail("Expected .fileNotFound, got \(error)")
        }
    }

    func test_model_weight_loader_header_parsing() throws {
        let positivePath = makeTempBinaryPath(prefix: "header-positive")
        defer { try? FileManager.default.removeItem(atPath: positivePath) }
        let positiveHeader = llamaHeaderData(
            dim: 768,
            hiddenDim: 2048,
            nLayers: 12,
            nHeads: 12,
            nKvHeads: 12,
            vocabSize: 32_000,
            seqLen: 256
        )
        try positiveHeader.write(to: URL(fileURLWithPath: positivePath))

        guard let positiveFile = fopen(positivePath, "rb") else {
            XCTFail("Failed to open positive header fixture")
            return
        }
        defer { fclose(positiveFile) }

        let parsedPositive = try ModelWeightLoader.parseHeader(from: positiveFile)
        XCTAssertEqual(parsedPositive.dim, 768)
        XCTAssertEqual(parsedPositive.hiddenDim, 2048)
        XCTAssertEqual(parsedPositive.nLayers, 12)
        XCTAssertEqual(parsedPositive.nHeads, 12)
        XCTAssertEqual(parsedPositive.nKvHeads, 12)
        XCTAssertEqual(parsedPositive.vocabSize, 32_000)
        XCTAssertEqual(parsedPositive.seqLen, 256)
        XCTAssertGreaterThan(parsedPositive.vocabSize, 0)

        let negativePath = makeTempBinaryPath(prefix: "header-negative")
        defer { try? FileManager.default.removeItem(atPath: negativePath) }
        let negativeHeader = llamaHeaderData(
            dim: 768,
            hiddenDim: 2048,
            nLayers: 12,
            nHeads: 12,
            nKvHeads: 12,
            vocabSize: -32_000,
            seqLen: 256
        )
        try negativeHeader.write(to: URL(fileURLWithPath: negativePath))

        guard let negativeFile = fopen(negativePath, "rb") else {
            XCTFail("Failed to open negative header fixture")
            return
        }
        defer { fclose(negativeFile) }

        let parsedNegative = try ModelWeightLoader.parseHeader(from: negativeFile)
        XCTAssertLessThan(parsedNegative.vocabSize, 0)
        XCTAssertEqual(abs(parsedNegative.vocabSize), 32_000)
    }

    func test_model_weight_loader_truncated_file() throws {
        let path = makeTempBinaryPath(prefix: "truncated")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let header = llamaHeaderData(
            dim: Int32(ModelConfig.dim),
            hiddenDim: Int32(ModelConfig.hidden),
            nLayers: Int32(ModelConfig.nLayers),
            nHeads: Int32(ModelConfig.heads),
            nKvHeads: Int32(ModelConfig.heads),
            vocabSize: Int32(ModelConfig.vocab),
            seqLen: Int32(ModelConfig.seqLen)
        )

        var fileData = Data()
        fileData.append(header)
        fileData.append(Data(count: 100))
        try fileData.write(to: URL(fileURLWithPath: path))

        do {
            _ = try ModelWeightLoader.load(from: path)
            XCTFail("Expected .truncatedFile")
        } catch let ModelLoadError.truncatedFile(expectedBytes, actualBytes) {
            XCTAssertGreaterThan(expectedBytes, actualBytes)
            XCTAssertGreaterThan(actualBytes, 0)
        } catch {
            XCTFail("Expected .truncatedFile, got \(error)")
        }
    }

    func test_model_weight_loader_payload_layout_matches_llama2c_order_and_sizes() {
        let sharedLayout = ModelWeightLoader.payloadLayout(vocabSize: Int32(ModelConfig.vocab))
        XCTAssertEqual(
            sharedLayout.map(\.name),
            [
                "embed",
                "rms_att[all]",
                "wq[all]",
                "wk[all]",
                "wv[all]",
                "wo[all]",
                "rms_ffn[all]",
                "w1[all]",
                "w2[all]",
                "w3[all]",
                "rms_final",
            ]
        )

        let expectedCounts = [
            ModelConfig.vocab * ModelConfig.dim,
            ModelConfig.nLayers * ModelConfig.dim,
            ModelConfig.nLayers * ModelConfig.dim * ModelConfig.dim,
            ModelConfig.nLayers * ModelConfig.dim * ModelConfig.dim,
            ModelConfig.nLayers * ModelConfig.dim * ModelConfig.dim,
            ModelConfig.nLayers * ModelConfig.dim * ModelConfig.dim,
            ModelConfig.nLayers * ModelConfig.dim,
            ModelConfig.nLayers * ModelConfig.hidden * ModelConfig.dim,
            ModelConfig.nLayers * ModelConfig.dim * ModelConfig.hidden,
            ModelConfig.nLayers * ModelConfig.hidden * ModelConfig.dim,
            ModelConfig.dim,
        ]
        XCTAssertEqual(sharedLayout.map(\.floatCount), expectedCounts)

        let unsharedLayout = ModelWeightLoader.payloadLayout(vocabSize: -Int32(ModelConfig.vocab))
        XCTAssertEqual(unsharedLayout.last?.name, "wcls")
        XCTAssertEqual(unsharedLayout.last?.floatCount, ModelConfig.vocab * ModelConfig.dim)
    }

    func test_model_weight_loader_unshared_classifier_truncated_file() throws {
        let path = makeTempBinaryPath(prefix: "truncated-unshared-classifier")
        defer { try? FileManager.default.removeItem(atPath: path) }

        let header = llamaHeaderData(
            dim: Int32(ModelConfig.dim),
            hiddenDim: Int32(ModelConfig.hidden),
            nLayers: Int32(ModelConfig.nLayers),
            nHeads: Int32(ModelConfig.heads),
            nKvHeads: Int32(ModelConfig.heads),
            vocabSize: -Int32(ModelConfig.vocab),
            seqLen: Int32(ModelConfig.seqLen)
        )
        try header.write(to: URL(fileURLWithPath: path))

        let layout = ModelWeightLoader.payloadLayout(vocabSize: -Int32(ModelConfig.vocab))
        guard let classifierSegment = layout.last, classifierSegment.name == "wcls" else {
            XCTFail("Expected unshared layout to include trailing wcls segment")
            return
        }

        let bytesBeforeClassifier = layout.dropLast().reduce(0) { $0 + $1.byteCount }
        let partialClassifierBytes = 8
        let totalBytes = header.count + bytesBeforeClassifier + partialClassifierBytes

        let fd = open(path, O_RDWR)
        guard fd >= 0 else {
            XCTFail("Failed to open temp file for sparse extension")
            return
        }
        defer { close(fd) }
        XCTAssertEqual(ftruncate(fd, off_t(totalBytes)), 0)

        do {
            _ = try ModelWeightLoader.load(from: path)
            XCTFail("Expected .truncatedFile while reading unshared classifier payload")
        } catch let ModelLoadError.truncatedFile(expectedBytes, actualBytes) {
            XCTAssertEqual(expectedBytes, classifierSegment.byteCount)
            XCTAssertEqual(actualBytes, partialClassifierBytes)
        } catch {
            XCTFail("Expected .truncatedFile, got \(error)")
        }
    }

    func test_load_stories110m_weights_integration() throws {
        guard let path = storiesModelPath() else {
            throw XCTSkip("stories110M.bin not found; set STORIES_MODEL_PATH or place model in assets/models")
        }

        let loaded = try ModelWeightLoader.load(from: path)
        XCTAssertEqual(loaded.layers.count, ModelConfig.nLayers)
        XCTAssertEqual(loaded.rmsFinal.count, ModelConfig.dim)
        XCTAssertEqual(loaded.embed.count, ModelConfig.vocab * ModelConfig.dim)
        XCTAssertTrue(loaded.sharedClassifier)
        XCTAssertTrue(hasNonZeroElement(loaded.layers[0].Wq))
        XCTAssertTrue(hasNonZeroElement(loaded.rmsFinal))
    }

    func test_layer_kernel_set_recompile_with_different_weights() throws {
        try requireANEHardwareTestsEnabled()

        let weightsA = LayerWeights()
        fillLayerWeights(weightsA, value: 0.01)
        let kernelsA = try LayerKernelSet(weights: weightsA)
        _ = try kernelsA.fwdFFN.outputSurface(at: 0)

        let weightsB = LayerWeights()
        fillLayerWeights(weightsB, value: 0.02)
        let kernelsB = try LayerKernelSet(weights: weightsB)
        _ = try kernelsB.qkvBwd.outputSurface(at: 0)
    }

    func test_static_kernel_eval_produces_output() throws {
        try requireANEHardwareTestsEnabled()

        let staticKernel = try StaticKernel()
        let inputSurface = try staticKernel.kernel.inputSurface(at: 0)
        let outputSurface = try staticKernel.kernel.outputSurface(at: 0)

        var input = [Float](repeating: 0, count: sdpaBwd2InputChannels * sdpaSpatial)
        for i in input.indices {
            input[i] = Float((i % 113) + 1) * 0.001
        }
        input.withUnsafeBufferPointer { buffer in
            SurfaceIO.writeFP16(
                to: inputSurface,
                data: buffer,
                channels: sdpaBwd2InputChannels,
                spatial: sdpaSpatial
            )
        }

        if !ANEEvalProbe.isAvailable {
            do {
                try staticKernel.kernel.eval()
                throw XCTSkip("ANE baseline probe unavailable; static kernel eval succeeded on this host")
            } catch ANEError.evaluationFailed {
                throw XCTSkip("ANE eval unavailable for static kernel on this host")
            } catch {
                throw error
            }
        }

        try staticKernel.kernel.eval()

        var output = [Float](repeating: 0, count: sdpaBwd2OutputChannels * sdpaSpatial)
        output.withUnsafeMutableBufferPointer { buffer in
            SurfaceIO.readFP16(
                from: outputSurface,
                into: buffer,
                channelOffset: 0,
                channels: sdpaBwd2OutputChannels,
                spatial: sdpaSpatial
            )
        }

        XCTAssertTrue(output.allSatisfy(\.isFinite))
        XCTAssertTrue(output.contains(where: { $0 != 0 }))
    }
}

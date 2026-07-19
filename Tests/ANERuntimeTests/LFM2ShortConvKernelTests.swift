import XCTest
import Darwin
import IOSurface
import ANETypes
import MILGenerator
@testable import ANERuntime

private let lfm2ProbeChannels = 1024
private let lfm2ProbeLaneSpatial = 32
private let lfm2ProbeKernelWidth = 3
private let lfm2ProbeStateTailSpatial = lfm2ProbeKernelWidth - 1
private let lfm2ProbeStateSpatial = lfm2ProbeLaneSpatial
private let lfm2ProbeMinimumAllocationBytes = 49_152

private func makeLFM2ProbeXValues() -> [Float] {
    (0..<(lfm2ProbeChannels * lfm2ProbeLaneSpatial)).map { index in
        Float((index % 251) + 1) * 0.01
    }
}

private func makeLFM2ProbeStateValues() -> [Float] {
    (0..<(lfm2ProbeChannels * lfm2ProbeStateSpatial)).map { index in
        -Float((index % 211) + 1) * 0.01
    }
}

private func maxAbsDiff(
    _ actual: [Float],
    _ expected: [Float]
) -> (value: Float, index: Int, actual: Float, expected: Float) {
    precondition(actual.count == expected.count)
    var bestValue: Float = -1
    var bestIndex = 0
    var bestActual: Float = 0
    var bestExpected: Float = 0
    for index in actual.indices {
        let diff = abs(actual[index] - expected[index])
        if diff > bestValue {
            bestValue = diff
            bestIndex = index
            bestActual = actual[index]
            bestExpected = expected[index]
        }
    }
    return (bestValue, bestIndex, bestActual, bestExpected)
}

private func requireLFM2ProbeANEAvailable(file: StaticString = #filePath, line: UInt = #line) throws {
    let handle = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW
    )
    if handle == nil {
        throw XCTSkip("AppleNeuralEngine.framework unavailable", file: file, line: line)
    }
    dlclose(handle)
}

private func requireLFM2ProbeHardwareTests(file: StaticString = #filePath, line: UInt = #line) throws {
    guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
        throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run ANE hardware tests", file: file, line: line)
    }
    try requireLFM2ProbeANEAvailable(file: file, line: line)
}

private func requireLFM2ShortConvBenchEnabled(file: StaticString = #filePath, line: UInt = #line) throws {
    guard ProcessInfo.processInfo.environment["LFM2_SHORT_CONV_BENCH"] == "1" else {
        throw XCTSkip("Set LFM2_SHORT_CONV_BENCH=1 to run the LFM2 short-conv microbench", file: file, line: line)
    }
}

private func makeLFM2ShortConvPassthroughWeights(
    channels: Int,
    kernelWidth: Int,
    groups: Int
) -> Data {
    precondition(channels.isMultiple(of: groups))
    let inputChannelsPerGroup = channels / groups
    let colsPerRow = inputChannelsPerGroup * kernelWidth
    var weights = [Float](repeating: 0, count: channels * colsPerRow)
    for outputChannel in 0..<channels {
        let localChannel = outputChannel % inputChannelsPerGroup
        let rowStart = outputChannel * colsPerRow
        weights[rowStart + localChannel * kernelWidth + (kernelWidth - 1)] = 1
    }
    return WeightBlob.build(from: weights, rows: channels, cols: colsPerRow)
}

private func makeLFM2ShortConvKernel(
    groups: Int = lfm2ProbeChannels,
    checkBudget: Bool = false
) throws -> ANEKernel {
    let generator = LFM2ShortConvStepGenerator(
        dim: lfm2ProbeChannels,
        laneSpatial: lfm2ProbeLaneSpatial,
        kernelWidth: lfm2ProbeKernelWidth,
        groups: groups
    )
    let uniformAllocationBytes = max(
        lfm2ProbeMinimumAllocationBytes,
        (generator.inputByteSizes + generator.outputByteSizes).max() ?? 0
    )
    return try ANEKernel(
        milText: generator.milText,
        weights: [(
            path: "@model_path/weights/lfm2_short_conv.bin",
            data: makeLFM2ShortConvPassthroughWeights(
                channels: lfm2ProbeChannels,
                kernelWidth: lfm2ProbeKernelWidth,
                groups: groups
            )
        )],
        inputSizes: Array(repeating: uniformAllocationBytes, count: generator.inputByteSizes.count),
        outputSizes: Array(repeating: uniformAllocationBytes, count: generator.outputByteSizes.count),
        checkBudget: checkBudget
    )
}

private func makeLFM2ShortConvFactorizedWeights(
    channels: Int,
    kernelWidth: Int,
    groups: Int
) -> [(path: String, data: Data)] {
    precondition(channels.isMultiple(of: groups))
    let inputChannelsPerGroup = channels / groups
    let colsPerRow = inputChannelsPerGroup

    return (0..<kernelWidth).map { tapIndex in
        var weights = [Float](repeating: 0, count: channels * colsPerRow)
        if tapIndex == kernelWidth - 1 {
            for outputChannel in 0..<channels {
                let localChannel = outputChannel % inputChannelsPerGroup
                let rowStart = outputChannel * colsPerRow
                weights[rowStart + localChannel] = 1
            }
        }
        return (
            path: "@model_path/weights/lfm2_short_conv_tap\(tapIndex).bin",
            data: WeightBlob.build(from: weights, rows: channels, cols: colsPerRow)
        )
    }
}

private func makeLFM2ShortConvFactorizedKernel(
    groups: Int = lfm2ProbeChannels,
    checkBudget: Bool = false
) throws -> ANEKernel {
    let generator = LFM2ShortConvFactorizedStepGenerator(
        dim: lfm2ProbeChannels,
        laneSpatial: lfm2ProbeLaneSpatial,
        kernelWidth: lfm2ProbeKernelWidth,
        groups: groups
    )
    let uniformAllocationBytes = max(
        lfm2ProbeMinimumAllocationBytes,
        (generator.inputByteSizes + generator.outputByteSizes).max() ?? 0
    )
    return try ANEKernel(
        milText: generator.milText,
        weights: makeLFM2ShortConvFactorizedWeights(
            channels: lfm2ProbeChannels,
            kernelWidth: lfm2ProbeKernelWidth,
            groups: groups
        ),
        inputSizes: Array(repeating: uniformAllocationBytes, count: generator.inputByteSizes.count),
        outputSizes: Array(repeating: uniformAllocationBytes, count: generator.outputByteSizes.count),
        checkBudget: checkBudget
    )
}

private func makeLFM2IdentityConvKernel(
    groups: Int = lfm2ProbeChannels,
    checkBudget: Bool = false
) throws -> ANEKernel {
    precondition(lfm2ProbeChannels.isMultiple(of: groups))
    let inputChannelsPerGroup = lfm2ProbeChannels / groups
    let milText = """
    program(1.3)
    [buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
    {
        func main<ios18>(tensor<fp16, [1, \(lfm2ProbeChannels), 1, \(lfm2ProbeLaneSpatial)]> x) {
            string pt = const()[name=string("pt"), val=string("valid")];
            tensor<int32, [2]> st = const()[name=string("st"), val=tensor<int32, [2]>([1,1])];
            tensor<int32, [4]> pd = const()[name=string("pd"), val=tensor<int32, [4]>([0,0,0,0])];
            tensor<int32, [2]> dl = const()[name=string("dl"), val=tensor<int32, [2]>([1,1])];
            int32 gr = const()[name=string("gr"), val=int32(\(groups))];
            tensor<fp16, [\(lfm2ProbeChannels), \(inputChannelsPerGroup), 1, 1]> W = const()[name=string("W"), val=tensor<fp16, [\(lfm2ProbeChannels), \(inputChannelsPerGroup), 1, 1]>(BLOBFILE(path=string("@model_path/weights/lfm2_identity_conv.bin"), offset=uint64(64)))];
            tensor<fp16, [1, \(lfm2ProbeChannels), 1, \(lfm2ProbeLaneSpatial)]> xNext = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)[name=string("xNext")];
        } -> (xNext);
    }
    """
    let inputBytes = lfm2ProbeChannels * lfm2ProbeLaneSpatial * 2
    let uniformAllocationBytes = max(lfm2ProbeMinimumAllocationBytes, inputBytes)
    return try ANEKernel(
        milText: milText,
        weights: [(
            path: "@model_path/weights/lfm2_identity_conv.bin",
            data: makeLFM2ShortConvPassthroughWeights(
                channels: lfm2ProbeChannels,
                kernelWidth: 1,
                groups: groups
            )
        )],
        inputSizes: [uniformAllocationBytes],
        outputSizes: [uniformAllocationBytes],
        checkBudget: checkBudget
    )
}

final class LFM2ShortConvRuntimeTests: XCTestCase {
    func test_lfm2_short_conv_kernel_compiles_on_ane() throws {
        try requireLFM2ProbeHardwareTests()
        _ = try makeLFM2ShortConvKernel()
    }

    func test_lfm2_short_conv_kernel_evaluates_on_ane_with_uniform_padded_io() throws {
        try requireLFM2ProbeHardwareTests()
        let kernel = try makeLFM2ShortConvKernel()
        let surfaces = ANESurfaceOwner(pool: nil)
        let inputSurfaces = try surfaces.retainInputs(from: kernel, named: ["x", "convStateIn"])

        let xSurface = try XCTUnwrap(inputSurfaces["x"])
        let convStateSurface = try XCTUnwrap(inputSurfaces["convStateIn"])

        let xValues = makeLFM2ProbeXValues()
        let stateValues = makeLFM2ProbeStateValues()

        xValues.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(
                to: xSurface,
                data: src,
                channels: lfm2ProbeChannels,
                spatial: lfm2ProbeLaneSpatial
            )
        }
        stateValues.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(
                to: convStateSurface,
                data: src,
                channels: lfm2ProbeChannels,
                spatial: lfm2ProbeStateSpatial
            )
        }

        try kernel.eval()
    }

    func test_lfm2_short_conv_kernel_direct_eval_matches_passthrough_and_updates_state() throws {
        try requireLFM2ProbeHardwareTests()
        try assertLFM2ShortConvKernelPasses(groups: lfm2ProbeChannels)
    }

    func test_lfm2_grouped_identity_conv1x1_matches_passthrough() throws {
        try requireLFM2ProbeHardwareTests()
        let kernel = try makeLFM2IdentityConvKernel()
        let surfaces = ANESurfaceOwner(pool: nil)
        let inputSurface = try surfaces.retainInput(from: kernel, at: 0)
        let outputSurface = try surfaces.retainOutput(from: kernel, at: 0)

        let xValues = makeLFM2ProbeXValues()
        xValues.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(
                to: inputSurface,
                data: src,
                channels: lfm2ProbeChannels,
                spatial: lfm2ProbeLaneSpatial
            )
        }

        try kernel.eval()

        var xNext = [Float](repeating: 0, count: xValues.count)
        xNext.withUnsafeMutableBufferPointer { dst in
            SurfaceIO.readFP16(
                from: outputSurface,
                into: dst,
                channelOffset: 0,
                channels: lfm2ProbeChannels,
                spatial: lfm2ProbeLaneSpatial
            )
        }

        let diff = maxAbsDiff(xNext, xValues)
        XCTAssertLessThanOrEqual(
            diff.value,
            1e-2,
            "identity 1x1 maxDiff=\(diff.value) at index \(diff.index), actual=\(diff.actual), expected=\(diff.expected)"
        )
    }

    func test_lfm2_factorized_short_conv_kernel_eval_matches_passthrough_and_updates_state() throws {
        try requireLFM2ProbeHardwareTests()
        try assertLFM2ShortConvFactorizedKernelPasses(groups: lfm2ProbeChannels)
    }

    func test_lfm2_factorized_short_conv_kernel_microbench_on_ane() throws {
        try requireLFM2ProbeHardwareTests()
        try requireLFM2ShortConvBenchEnabled()
        let kernel = try makeLFM2ShortConvFactorizedKernel(groups: lfm2ProbeChannels)
        try benchmarkShortConvKernel(
            kernel: kernel,
            label: "lfm2_short_conv_factorized_probe"
        )
    }

    func test_lfm2_short_conv_kernel_direct_microbench_on_ane() throws {
        try requireLFM2ProbeHardwareTests()
        try requireLFM2ShortConvBenchEnabled()
        let kernel = try makeLFM2ShortConvKernel(groups: lfm2ProbeChannels)
        try benchmarkShortConvKernel(
            kernel: kernel,
            label: "lfm2_short_conv_direct_probe"
        )
    }

    private func assertLFM2ShortConvFactorizedKernelPasses(groups: Int) throws {
        let kernel = try makeLFM2ShortConvFactorizedKernel(groups: groups)
        let surfaces = ANESurfaceOwner(pool: nil)
        let inputSurfaces = try surfaces.retainInputs(from: kernel, named: ["x", "convStateIn"])
        let outputSurfaces = try surfaces.retainOutputs(from: kernel, named: ["xNext", "convStateOut"])

        let xSurface = try XCTUnwrap(inputSurfaces["x"])
        let convStateSurface = try XCTUnwrap(inputSurfaces["convStateIn"])
        let xNextSurface = try XCTUnwrap(outputSurfaces["xNext"])
        let convStateOutSurface = try XCTUnwrap(outputSurfaces["convStateOut"])

        let xValues = makeLFM2ProbeXValues()
        let stateValues = makeLFM2ProbeStateValues()

        xValues.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(
                to: xSurface,
                data: src,
                channels: lfm2ProbeChannels,
                spatial: lfm2ProbeLaneSpatial
            )
        }
        stateValues.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(
                to: convStateSurface,
                data: src,
                channels: lfm2ProbeChannels,
                spatial: lfm2ProbeStateSpatial
            )
        }

        try kernel.eval()

        var xNext = [Float](repeating: 0, count: xValues.count)
        xNext.withUnsafeMutableBufferPointer { dst in
            SurfaceIO.readFP16(
                from: xNextSurface,
                into: dst,
                channelOffset: 0,
                channels: lfm2ProbeChannels,
                spatial: lfm2ProbeLaneSpatial
            )
        }

        var convStateOut = [Float](repeating: 0, count: stateValues.count)
        convStateOut.withUnsafeMutableBufferPointer { dst in
            SurfaceIO.readFP16(
                from: convStateOutSurface,
                into: dst,
                channelOffset: 0,
                channels: lfm2ProbeChannels,
                spatial: lfm2ProbeStateSpatial
            )
        }

        let xNextDiff = maxAbsDiff(xNext, xValues)
        XCTAssertLessThanOrEqual(
            xNextDiff.value,
            1e-2,
            "factorized xNext maxDiff=\(xNextDiff.value) at index \(xNextDiff.index), actual=\(xNextDiff.actual), expected=\(xNextDiff.expected)"
        )

        let expectedConvStateOut = xValues
        let stateDiff = maxAbsDiff(convStateOut, expectedConvStateOut)
        XCTAssertLessThanOrEqual(
            stateDiff.value,
            1e-2,
            "factorized state maxDiff=\(stateDiff.value) at index \(stateDiff.index), actual=\(stateDiff.actual), expected=\(stateDiff.expected)"
        )
    }

    private func assertLFM2ShortConvKernelPasses(groups: Int) throws {
        let kernel = try makeLFM2ShortConvKernel(groups: groups)
        let surfaces = ANESurfaceOwner(pool: nil)
        let inputSurfaces = try surfaces.retainInputs(from: kernel, named: ["x", "convStateIn"])
        let outputSurfaces = try surfaces.retainOutputs(from: kernel, named: ["xNext", "convStateOut"])

        let xSurface = try XCTUnwrap(inputSurfaces["x"])
        let convStateSurface = try XCTUnwrap(inputSurfaces["convStateIn"])
        let xNextSurface = try XCTUnwrap(outputSurfaces["xNext"])
        let convStateOutSurface = try XCTUnwrap(outputSurfaces["convStateOut"])

        let xValues = makeLFM2ProbeXValues()
        let stateValues = makeLFM2ProbeStateValues()

        xValues.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(
                to: xSurface,
                data: src,
                channels: lfm2ProbeChannels,
                spatial: lfm2ProbeLaneSpatial
            )
        }
        stateValues.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(
                to: convStateSurface,
                data: src,
                channels: lfm2ProbeChannels,
                spatial: lfm2ProbeStateSpatial
            )
        }

        try kernel.eval()

        var xNext = [Float](repeating: 0, count: xValues.count)
        xNext.withUnsafeMutableBufferPointer { dst in
            SurfaceIO.readFP16(
                from: xNextSurface,
                into: dst,
                channelOffset: 0,
                channels: lfm2ProbeChannels,
                spatial: lfm2ProbeLaneSpatial
            )
        }

        var convStateOut = [Float](repeating: 0, count: stateValues.count)
        convStateOut.withUnsafeMutableBufferPointer { dst in
            SurfaceIO.readFP16(
                from: convStateOutSurface,
                into: dst,
                channelOffset: 0,
                channels: lfm2ProbeChannels,
                spatial: lfm2ProbeStateSpatial
            )
        }

        let xNextDiff = maxAbsDiff(xNext, xValues)
        XCTAssertLessThanOrEqual(
            xNextDiff.value,
            1e-2,
            "direct stateful k=3 xNext maxDiff=\(xNextDiff.value) at index \(xNextDiff.index), actual=\(xNextDiff.actual), expected=\(xNextDiff.expected)"
        )

        let expectedConvStateOut = xValues
        let stateDiff = maxAbsDiff(convStateOut, expectedConvStateOut)
        XCTAssertLessThanOrEqual(
            stateDiff.value,
            1e-2,
            "direct stateful k=3 state maxDiff=\(stateDiff.value) at index \(stateDiff.index), actual=\(stateDiff.actual), expected=\(stateDiff.expected)"
        )
    }

    private func benchmarkShortConvKernel(
        kernel: borrowing ANEKernel,
        label: String
    ) throws {
        let warmup = 20
        let iterations = 100

        let surfaces = ANESurfaceOwner(pool: nil)
        let inputSurfaces = try surfaces.retainInputs(from: kernel, named: ["x", "convStateIn"])

        let xSurface = try XCTUnwrap(inputSurfaces["x"])
        let convStateSurface = try XCTUnwrap(inputSurfaces["convStateIn"])

        let xValues = makeLFM2ProbeXValues()
        let stateValues = makeLFM2ProbeStateValues()
        xValues.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(
                to: xSurface,
                data: src,
                channels: lfm2ProbeChannels,
                spatial: lfm2ProbeLaneSpatial
            )
        }
        stateValues.withUnsafeBufferPointer { src in
            SurfaceIO.writeFP16(
                to: convStateSurface,
                data: src,
                channels: lfm2ProbeChannels,
                spatial: lfm2ProbeStateSpatial
            )
        }

        for _ in 0..<warmup {
            try kernel.eval()
        }

        var elapsedMicros: [Double] = []
        elapsedMicros.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let start = DispatchTime.now().uptimeNanoseconds
            try kernel.eval()
            let end = DispatchTime.now().uptimeNanoseconds
            elapsedMicros.append(Double(end - start) / 1_000.0)
        }

        let sorted = elapsedMicros.sorted()
        let medianMicros = sorted[sorted.count / 2]
        let p95Micros = sorted[Int(Double(sorted.count - 1) * 0.95)]
        let tokPerSecond = Double(lfm2ProbeLaneSpatial) * 1_000_000.0 / medianMicros
        let medianString = String(format: "%.2f", medianMicros)
        let p95String = String(format: "%.2f", p95Micros)
        let tokPerSecondString = String(format: "%.2f", tokPerSecond)
        print(
            "\(label) dim=\(lfm2ProbeChannels) lane=\(lfm2ProbeLaneSpatial) warmup=\(warmup) iterations=\(iterations) median_us=\(medianString) p95_us=\(p95String) tok_s=\(tokPerSecondString)"
        )
    }
}

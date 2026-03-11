import XCTest
import ANETypes
import ANERuntime
@testable import Espresso

private func requireHybridDecodeHardware(file: StaticString = #filePath, line: UInt = #line) throws {
    guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
        throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run hybrid decode hardware tests", file: file, line: line)
    }
}

private func makeHybridDecodeWeights(value: Float = 0.01) -> LayerWeights {
    let weights = LayerWeights()
    func fill(_ buf: borrowing TensorBuffer, _ value: Float) {
        buf.withUnsafeMutableBufferPointer { ptr in
            for idx in ptr.indices {
                ptr[idx] = value
            }
        }
    }

    fill(weights.Wq, value)
    fill(weights.Wk, value)
    fill(weights.Wv, value)
    fill(weights.Wo, value)
    fill(weights.W1, value)
    fill(weights.W2, value)
    fill(weights.W3, value)
    fill(weights.rmsAtt, 1.0)
    fill(weights.rmsFfn, 1.0)
    return weights
}

private struct TokenBenchmark {
    let medianMs: Double
    let latenciesMs: [Double]
    let aneQKVMedianMs: Double?
    let metalMedianMs: Double?
    let aneFFNMedianMs: Double?
    let ioMedianMs: Double?
}

final class HybridDecodeForwardPassTests: XCTestCase {
    func test_hybrid_decode_single_step_runs_on_hardware() throws {
        try requireHybridDecodeHardware()

        let weights = makeHybridDecodeWeights()
        let kernels = try LayerStorage<HybridDecodeKernelSet>(count: 1, throwingInitializer: { _ in
            try HybridDecodeKernelSet(weights: weights, maxSeq: 8)
        })
        let handles = [try HybridDecodeSurfaceHandles(kernels: kernels[0], logicalMaxSeq: 8)]
        let metal = try MetalAttentionKernel()
        let xCur = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        var decodeState = try DecodeState(maxSeq: 8)
        var timings = HybridDecodeTimingBreakdown()

        xCur.withUnsafeMutableBufferPointer { ptr in
            for idx in ptr.indices {
                ptr[idx] = Float(idx % 17) * 0.001
            }
        }

        ForwardPass.initializeHybridDecodeCaches(surfaceHandles: handles)
        try ForwardPass.runHybridDecodeTimed(
            xCur: xCur,
            kernels: kernels,
            surfaceHandles: handles,
            metalAttention: metal,
            decodeState: &decodeState,
            timings: &timings
        )

        XCTAssertEqual(decodeState.visibleTokenCount, 1)
        XCTAssertGreaterThan(timings.tAneQKV, 0)
        XCTAssertGreaterThan(timings.tMetal, 0)
        XCTAssertGreaterThan(timings.tAneFFN, 0)
    }

    func test_hybrid_decode_benchmark_reports_direct_and_hybrid_token_medians() throws {
        try requireHybridDecodeHardware()

        let direct = try benchmarkDirectDecodeTokenLatency(
            layerCount: 6,
            maxSeq: 32,
            decodeSteps: 32,
            warmup: 3,
            iterations: 20
        )
        let hybrid = try benchmarkHybridDecodeTokenLatency(
            layerCount: 6,
            maxSeq: 32,
            decodeSteps: 32,
            warmup: 3,
            iterations: 20
        )

        print(
            "direct median=\(direct.medianMs) ms/token hybrid median=\(hybrid.medianMs) ms/token"
        )
        if let aneQKV = hybrid.aneQKVMedianMs,
           let metal = hybrid.metalMedianMs,
           let aneFFN = hybrid.aneFFNMedianMs,
           let io = hybrid.ioMedianMs {
            print(
                "hybrid stages median: qkv=\(aneQKV) ms metal=\(metal) ms ffn=\(aneFFN) ms io=\(io) ms"
            )
        }
        XCTAssertGreaterThan(direct.medianMs, 0)
        XCTAssertGreaterThan(hybrid.medianMs, 0)
        XCTAssertEqual(direct.latenciesMs.count, 20 * 32)
        XCTAssertEqual(hybrid.latenciesMs.count, 20 * 32)
    }

    private func benchmarkDirectDecodeTokenLatency(
        layerCount: Int,
        maxSeq: Int,
        decodeSteps: Int,
        warmup: Int,
        iterations: Int
    ) throws -> TokenBenchmark {
        let weights = LayerStorage<LayerWeights>(count: layerCount) { _ in
            makeHybridDecodeWeights()
        }
        let kernels = try LayerStorage<DecodeKernelSet>(count: layerCount, throwingInitializer: { idx in
            try DecodeKernelSet(weights: weights[idx], maxSeq: maxSeq)
        })
        var handles: [DecodeSurfaceHandles] = []
        handles.reserveCapacity(layerCount)
        for idx in 0..<layerCount {
            handles.append(try DecodeSurfaceHandles(kernels: kernels[idx], logicalMaxSeq: maxSeq))
        }
        let xCur = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        let tokenInputs = (0..<(decodeSteps * ModelConfig.dim)).map { Float(($0 % 23) - 11) * 0.001 }

        func loadToken(step: Int) {
            let base = step * ModelConfig.dim
            xCur.withUnsafeMutableBufferPointer { ptr in
                for idx in ptr.indices {
                    ptr[idx] = tokenInputs[base + idx]
                }
            }
        }

        for _ in 0..<warmup {
            ForwardPass.initializeDecodeCachesAndMask(surfaceHandles: handles)
            var state = try DecodeState(maxSeq: maxSeq)
            for step in 0..<decodeSteps {
                loadToken(step: step)
                var timings = StepTimingBreakdown()
                try ForwardPass.runDecodeTimed(
                    xCur: xCur,
                    kernels: kernels,
                    surfaceHandles: handles,
                    decodeState: &state,
                    timings: &timings
                )
            }
        }

        var latencies: [Double] = []
        latencies.reserveCapacity(iterations * decodeSteps)
        for _ in 0..<iterations {
            ForwardPass.initializeDecodeCachesAndMask(surfaceHandles: handles)
            var state = try DecodeState(maxSeq: maxSeq)
            for step in 0..<decodeSteps {
                loadToken(step: step)
                var timings = StepTimingBreakdown()
                let start = mach_absolute_time()
                try ForwardPass.runDecodeTimed(
                    xCur: xCur,
                    kernels: kernels,
                    surfaceHandles: handles,
                    decodeState: &state,
                    timings: &timings
                )
                let end = mach_absolute_time()
                latencies.append(milliseconds(start: start, end: end))
            }
        }
        return TokenBenchmark(
            medianMs: median(latencies),
            latenciesMs: latencies,
            aneQKVMedianMs: nil,
            metalMedianMs: nil,
            aneFFNMedianMs: nil,
            ioMedianMs: nil
        )
    }

    private func benchmarkHybridDecodeTokenLatency(
        layerCount: Int,
        maxSeq: Int,
        decodeSteps: Int,
        warmup: Int,
        iterations: Int
    ) throws -> TokenBenchmark {
        let weights = LayerStorage<LayerWeights>(count: layerCount) { _ in
            makeHybridDecodeWeights()
        }
        let kernels = try LayerStorage<HybridDecodeKernelSet>(count: layerCount, throwingInitializer: { idx in
            try HybridDecodeKernelSet(weights: weights[idx], maxSeq: maxSeq)
        })
        var handles: [HybridDecodeSurfaceHandles] = []
        handles.reserveCapacity(layerCount)
        for idx in 0..<layerCount {
            handles.append(try HybridDecodeSurfaceHandles(kernels: kernels[idx], logicalMaxSeq: maxSeq))
        }
        let metal = try MetalAttentionKernel()
        let xCur = TensorBuffer(count: ModelConfig.dim, zeroed: true)
        let tokenInputs = (0..<(decodeSteps * ModelConfig.dim)).map { Float(($0 % 29) - 13) * 0.001 }

        func loadToken(step: Int) {
            let base = step * ModelConfig.dim
            xCur.withUnsafeMutableBufferPointer { ptr in
                for idx in ptr.indices {
                    ptr[idx] = tokenInputs[base + idx]
                }
            }
        }

        for _ in 0..<warmup {
            ForwardPass.initializeHybridDecodeCaches(surfaceHandles: handles)
            var state = try DecodeState(maxSeq: maxSeq)
            for step in 0..<decodeSteps {
                loadToken(step: step)
                var timings = HybridDecodeTimingBreakdown()
                try ForwardPass.runHybridDecodeTimed(
                    xCur: xCur,
                    kernels: kernels,
                    surfaceHandles: handles,
                    metalAttention: metal,
                    decodeState: &state,
                    timings: &timings
                )
            }
        }

        var latencies: [Double] = []
        latencies.reserveCapacity(iterations * decodeSteps)
        var qkvLatencies: [Double] = []
        var metalLatencies: [Double] = []
        var ffnLatencies: [Double] = []
        var ioLatencies: [Double] = []
        qkvLatencies.reserveCapacity(iterations * decodeSteps)
        metalLatencies.reserveCapacity(iterations * decodeSteps)
        ffnLatencies.reserveCapacity(iterations * decodeSteps)
        ioLatencies.reserveCapacity(iterations * decodeSteps)
        for _ in 0..<iterations {
            ForwardPass.initializeHybridDecodeCaches(surfaceHandles: handles)
            var state = try DecodeState(maxSeq: maxSeq)
            for step in 0..<decodeSteps {
                loadToken(step: step)
                var timings = HybridDecodeTimingBreakdown()
                let start = mach_absolute_time()
                try ForwardPass.runHybridDecodeTimed(
                    xCur: xCur,
                    kernels: kernels,
                    surfaceHandles: handles,
                    metalAttention: metal,
                    decodeState: &state,
                    timings: &timings
                )
                let end = mach_absolute_time()
                latencies.append(milliseconds(start: start, end: end))
                qkvLatencies.append(timings.tAneQKV)
                metalLatencies.append(timings.tMetal)
                ffnLatencies.append(timings.tAneFFN)
                ioLatencies.append(timings.tIO)
            }
        }
        return TokenBenchmark(
            medianMs: median(latencies),
            latenciesMs: latencies,
            aneQKVMedianMs: median(qkvLatencies),
            metalMedianMs: median(metalLatencies),
            aneFFNMedianMs: median(ffnLatencies),
            ioMedianMs: median(ioLatencies)
        )
    }

    private func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[mid - 1] + sorted[mid]) * 0.5
        }
        return sorted[mid]
    }

    private func milliseconds(start: UInt64, end: UInt64) -> Double {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        let elapsed = end &- start
        let nanos = elapsed &* UInt64(info.numer) / UInt64(info.denom)
        return Double(nanos) / 1_000_000.0
    }
}

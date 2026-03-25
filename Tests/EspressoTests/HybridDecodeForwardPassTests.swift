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
    func test_hybrid_decode_can_disable_fused_metal_sdpa_by_environment() {
        XCTAssertTrue(DecodeRuntimeOptions.useMetalFusedSDPA(env: [:]))
        XCTAssertFalse(
            DecodeRuntimeOptions.useMetalFusedSDPA(
                env: ["ESPRESSO_DISABLE_METAL_FUSED_SDPA": "1"]
            )
        )
    }

    func test_hybrid_decode_can_enable_cpu_attention_by_environment() {
        XCTAssertFalse(DecodeRuntimeOptions.useCPUDecodeAttention(env: [:]))
        XCTAssertTrue(
            DecodeRuntimeOptions.useCPUDecodeAttention(
                env: ["ESPRESSO_USE_CPU_DECODE_ATTENTION": "1"]
            )
        )
    }

    func test_hybrid_decode_can_enable_cpu_attention_v_dim_major_by_environment() {
        XCTAssertFalse(DecodeRuntimeOptions.cpuDecodeAttentionVDimMajor(env: [:]))
        XCTAssertTrue(
            DecodeRuntimeOptions.cpuDecodeAttentionVDimMajor(
                env: ["ESPRESSO_CPU_DECODE_ATTENTION_V_DIM_MAJOR": "1"]
            )
        )
    }

    func test_hybrid_decode_attention_window_defaults_off() {
        XCTAssertNil(DecodeRuntimeOptions.hybridAttentionWindow(env: [:]))
        XCTAssertNil(DecodeRuntimeOptions.hybridAttentionWindow(env: ["ESPRESSO_HYBRID_ATTENTION_WINDOW": "0"]))
        XCTAssertNil(DecodeRuntimeOptions.hybridAttentionWindow(env: ["ESPRESSO_HYBRID_ATTENTION_WINDOW": "bogus"]))
        XCTAssertEqual(
            DecodeRuntimeOptions.hybridAttentionWindow(
                env: ["ESPRESSO_HYBRID_ATTENTION_WINDOW": "64"]
            ),
            64
        )
    }

    func test_resolved_hybrid_attention_range_uses_suffix_window() {
        let fullContext = ForwardPass.resolvedHybridAttentionRange(tokenIndex: 0, attentionWindow: nil)
        XCTAssertEqual(fullContext.tokenBase, 0)
        XCTAssertEqual(fullContext.visibleTokens, 1)

        let withinWindow = ForwardPass.resolvedHybridAttentionRange(tokenIndex: 3, attentionWindow: 8)
        XCTAssertEqual(withinWindow.tokenBase, 0)
        XCTAssertEqual(withinWindow.visibleTokens, 4)

        let suffixWindow = ForwardPass.resolvedHybridAttentionRange(tokenIndex: 7, attentionWindow: 4)
        XCTAssertEqual(suffixWindow.tokenBase, 4)
        XCTAssertEqual(suffixWindow.visibleTokens, 4)
    }

    func test_cpu_decode_context_matches_grouped_contiguous_reference() {
        let qOut: [Float] = [1, 2, 3, 4]
        let kCache: [Float] = [
            1, 2,
            0, 1
        ]
        let vCache: [Float] = [
            10, 20,
            30, 40
        ]

        let context = ForwardPass.cpuDecodeContext(
            qOut: qOut,
            kCache: kCache,
            vCache: vCache,
            heads: 2,
            kvHeads: 1,
            headDim: 2,
            visibleTokens: 2,
            cacheStride: 2,
            useModuloKVHeadMapping: false,
            useVDimMajorInterleave: false
        )

        let scale = Float(1.0 / sqrt(2.0))
        let head0Logits = [1 * scale, 4 * scale]
        let head1Logits = [3 * scale, 10 * scale]
        let expected = softmaxWeightedContext(
            logitsByHead: [head0Logits, head1Logits],
            values: [[10, 20], [30, 40]]
        )

        XCTAssertEqual(context.count, expected.count)
        for (actual, reference) in zip(context, expected) {
            XCTAssertEqual(actual, reference, accuracy: 1e-5)
        }
    }

    func test_cpu_decode_context_respects_modulo_kv_head_mapping() {
        let qOut: [Float] = [1, 0, 0, 1]
        let kCache: [Float] = [
            1, 0,
            0, 1,
            0, 1,
            1, 0
        ]
        let vCache: [Float] = [
            10, 20,
            30, 40,
            50, 60,
            70, 80
        ]

        let grouped = ForwardPass.cpuDecodeContext(
            qOut: qOut,
            kCache: kCache,
            vCache: vCache,
            heads: 2,
            kvHeads: 2,
            headDim: 2,
            visibleTokens: 2,
            cacheStride: 2,
            useModuloKVHeadMapping: false,
            useVDimMajorInterleave: false
        )
        let modulo = ForwardPass.cpuDecodeContext(
            qOut: qOut,
            kCache: kCache,
            vCache: vCache,
            heads: 2,
            kvHeads: 2,
            headDim: 2,
            visibleTokens: 2,
            cacheStride: 2,
            useModuloKVHeadMapping: true,
            useVDimMajorInterleave: false
        )

        XCTAssertEqual(grouped, modulo)

        let qOutInterleaved: [Float] = [1, 0, 0, 1, 0, 1, 1, 0]
        let kCacheInterleaved: [Float] = [
            1, 0,
            0, 1,
            0, 1,
            1, 0
        ]
        let vCacheInterleaved: [Float] = [
            10, 20,
            30, 40,
            50, 60,
            70, 80
        ]

        let groupedInterleaved = ForwardPass.cpuDecodeContext(
            qOut: qOutInterleaved,
            kCache: kCacheInterleaved,
            vCache: vCacheInterleaved,
            heads: 4,
            kvHeads: 2,
            headDim: 2,
            visibleTokens: 2,
            cacheStride: 2,
            useModuloKVHeadMapping: false,
            useVDimMajorInterleave: false
        )
        let moduloInterleaved = ForwardPass.cpuDecodeContext(
            qOut: qOutInterleaved,
            kCache: kCacheInterleaved,
            vCache: vCacheInterleaved,
            heads: 4,
            kvHeads: 2,
            headDim: 2,
            visibleTokens: 2,
            cacheStride: 2,
            useModuloKVHeadMapping: true,
            useVDimMajorInterleave: false
        )

        XCTAssertNotEqual(groupedInterleaved, moduloInterleaved)
    }

    func test_cpu_decode_context_can_read_suffix_window() {
        let context = ForwardPass.cpuDecodeContext(
            qOut: [1],
            kCache: [0, 1, 2, 3],
            vCache: [10, 20, 30, 40],
            heads: 1,
            kvHeads: 1,
            headDim: 1,
            visibleTokens: 2,
            tokenBase: 2,
            cacheStride: 4,
            useModuloKVHeadMapping: false,
            useVDimMajorInterleave: false
        )

        let scale: Float = 1
        let logits: [Float] = [2 * scale, 3 * scale]
        let weights = softmaxWeightedContext(
            logitsByHead: [logits],
            values: [[30, 40]]
        )
        XCTAssertEqual(context.count, 1)
        XCTAssertEqual(context[0], weights[0], accuracy: 1e-5)
    }

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

private func softmaxWeightedContext(
    logitsByHead: [[Float]],
    values: [[Float]]
) -> [Float] {
    logitsByHead.flatMap { logits in
        let maxLogit = logits.max() ?? -.infinity
        let exps = logits.map { expf($0 - maxLogit) }
        let denom = exps.reduce(0, +)
        let weights = exps.map { $0 / denom }
        return values.map { channel in
            zip(weights, channel).reduce(0) { $0 + ($1.0 * $1.1) }
        }
    }
}

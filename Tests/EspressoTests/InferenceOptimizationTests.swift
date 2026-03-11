import XCTest
import Darwin
import ANETypes
import ANERuntime
@testable import Espresso

/// Inference-only correctness tests for performance optimizations.
///
/// These require ANE private frameworks and real hardware execution.
final class InferenceOptimizationTests: XCTestCase {
    // MARK: - Gating

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
    }

    private func requireANEHardwareTestsEnabled(file: StaticString = #filePath, line: UInt = #line) throws {
        guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
            throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run ANE hardware tests", file: file, line: line)
        }
        try requireANEAvailable(file: file, line: line)
    }

    // MARK: - Deterministic Fill

    private struct XorShift32 {
        var state: UInt32

        init(seed: UInt32) {
            self.state = seed == 0 ? 0x12345678 : seed
        }

        mutating func nextUInt32() -> UInt32 {
            var x = state
            x ^= x << 13
            x ^= x >> 17
            x ^= x << 5
            state = x
            return x
        }

        mutating func nextFloat(range: ClosedRange<Float>) -> Float {
            let u = nextUInt32()
            let f = Float(u) / Float(UInt32.max) // [0,1)
            return range.lowerBound + (range.upperBound - range.lowerBound) * f
        }
    }

    private func fill(_ buf: borrowing TensorBuffer, seed: UInt32, range: ClosedRange<Float>) {
        var rng = XorShift32(seed: seed)
        buf.withUnsafeMutablePointer { ptr in
            for i in 0..<buf.count {
                ptr[i] = rng.nextFloat(range: range)
            }
        }
    }

    private func fillLayerWeights(_ w: borrowing LayerWeights, seed: UInt32) {
        // Keep weights small to avoid fp16 overflow; RMS weights set to 1.0.
        fill(w.Wq, seed: seed &+ 1, range: -0.02...0.02)
        fill(w.Wk, seed: seed &+ 2, range: -0.02...0.02)
        fill(w.Wv, seed: seed &+ 3, range: -0.02...0.02)
        fill(w.Wo, seed: seed &+ 4, range: -0.02...0.02)
        fill(w.W1, seed: seed &+ 5, range: -0.02...0.02)
        fill(w.W2, seed: seed &+ 6, range: -0.02...0.02)
        fill(w.W3, seed: seed &+ 7, range: -0.02...0.02)
        w.rmsAtt.withUnsafeMutablePointer { ptr in
            for i in 0..<w.rmsAtt.count { ptr[i] = 1.0 }
        }
        w.rmsFfn.withUnsafeMutablePointer { ptr in
            for i in 0..<w.rmsFfn.count { ptr[i] = 1.0 }
        }
    }

    private func memcpy(_ dst: borrowing TensorBuffer, _ src: borrowing TensorBuffer) {
        precondition(dst.count == src.count)
        let byteCount = dst.count * MemoryLayout<Float>.stride
        dst.withUnsafeMutablePointer { d in
            src.withUnsafePointer { s in
                _ = Darwin.memcpy(d, s, byteCount)
            }
        }
    }

    private func diffStats(_ a: borrowing TensorBuffer, _ b: borrowing TensorBuffer) -> (maxAbs: Float, meanAbs: Float) {
        precondition(a.count == b.count)
        var maxAbs: Float = 0
        var sumAbs: Double = 0
        a.withUnsafePointer { ap in
            b.withUnsafePointer { bp in
                for i in 0..<a.count {
                    let d = abs(ap[i] - bp[i])
                    if d > maxAbs { maxAbs = d }
                    sumAbs += Double(d)
                }
            }
        }
        let meanAbs = Float(sumAbs / Double(a.count))
        return (maxAbs, meanAbs)
    }

    func test_inference_fp16_surface_handoff_matches_baseline_strict() throws {
        try requireANEHardwareTestsEnabled()

        // 1. Deterministic weights + input
        let layers = LayerStorage<LayerWeights>(count: 1) { _ in
            let w = LayerWeights()
            fillLayerWeights(w, seed: 0xC0FFEE)
            return w
        }

        let kernels = try LayerStorage<InferenceKernelSet>(count: 1, throwingInitializer: { i in
            try InferenceKernelSet(weights: layers[i])
        })
        let handles = [try InferenceSurfaceHandles(kernels: kernels[0])]

        let count = ModelConfig.dim * ModelConfig.seqLen
        let xInit = TensorBuffer(count: count, zeroed: false)
        fill(xInit, seed: 0xBADC0DE, range: -0.1...0.1)

        let xBaseline = TensorBuffer(count: count, zeroed: false)
        let xOpt = TensorBuffer(count: count, zeroed: false)
        memcpy(xBaseline, xInit)
        memcpy(xOpt, xInit)

        // 2. Baseline: CPU round-trip between kernels
        var t0 = StepTimingBreakdown()
        try ForwardPass.runInferenceTimed(
            xCur: xBaseline,
            kernels: kernels,
            surfaceHandles: handles,
            handoff: .cpuRoundTrip,
            timings: &t0
        )

        // 3. Optimized: FP16 surface-to-surface handoff (no intermediate fp32 conversion)
        var t1 = StepTimingBreakdown()
        try ForwardPass.runInferenceTimed(
            xCur: xOpt,
            kernels: kernels,
            surfaceHandles: handles,
            handoff: .fp16SurfaceCopy,
            timings: &t1
        )

        // 4. Strict numerical parity
        let (maxAbs, meanAbs) = diffStats(xBaseline, xOpt)
        XCTAssertLessThanOrEqual(maxAbs, 1e-3, "max_abs_err too high: \(maxAbs)")
        XCTAssertLessThanOrEqual(meanAbs, 1e-5, "mean_abs_err too high: \(meanAbs)")
    }

    func test_decode_kv_cache_updates_and_mask_progresses_on_hardware() throws {
        try requireANEHardwareTestsEnabled()

        let decodeMaxSeq = DecodeKernelSet.defaultLaneSpatial
        let decodeSteps = 4
        let dim = ModelConfig.dim
        let maskSentinel = Float(-1e4)

        let layers = LayerStorage<LayerWeights>(count: 1) { _ in
            let w = LayerWeights()
            fillLayerWeights(w, seed: 0xDEC0DE)
            return w
        }

        let decodeKernels = try LayerStorage<DecodeKernelSet>(count: 1, throwingInitializer: { i in
            try DecodeKernelSet(weights: layers[i], maxSeq: decodeMaxSeq)
        })
        let decodeHandles = [try DecodeSurfaceHandles(kernels: decodeKernels[0])]
        ForwardPass.initializeDecodeCachesAndMask(surfaceHandles: decodeHandles)

        var decodeState = try DecodeState(maxSeq: decodeMaxSeq)
        var rng = XorShift32(seed: 0xA11CE)
        var tokenInputs = [[Float]]()
        tokenInputs.reserveCapacity(decodeSteps)
        for _ in 0..<decodeSteps {
            var token = [Float](repeating: 0, count: dim)
            for i in 0..<dim {
                token[i] = rng.nextFloat(range: -0.1...0.1)
            }
            tokenInputs.append(token)
        }

        for t in 0..<decodeSteps {
            let decodeToken = TensorBuffer(count: dim, zeroed: true)
            tokenInputs[t].withUnsafeBufferPointer { src in
                decodeToken.withUnsafeMutableBufferPointer { dst in
                    for i in 0..<dim {
                        dst[i] = src[i]
                    }
                }
            }

            var decodeTimings = StepTimingBreakdown()
            try ForwardPass.runDecodeTimed(
                xCur: decodeToken,
                kernels: decodeKernels,
                surfaceHandles: decodeHandles,
                decodeState: &decodeState,
                timings: &decodeTimings,
                profiler: nil
            )

            var decodeOut = [Float](repeating: 0, count: dim)
            decodeToken.withUnsafeBufferPointer { src in
                for i in 0..<dim {
                    decodeOut[i] = src[i]
                }
            }

            let lane = decodeHandles[0].laneSpatial
            var kOut = [Float](repeating: 0, count: dim * lane)
            var vOut = [Float](repeating: 0, count: dim * lane)
            kOut.withUnsafeMutableBufferPointer { dst in
                SurfaceIO.readFP16(from: decodeHandles[0].attnKOut, into: dst, channelOffset: 0, channels: dim, spatial: lane)
            }
            vOut.withUnsafeMutableBufferPointer { dst in
                SurfaceIO.readFP16(from: decodeHandles[0].attnVOut, into: dst, channelOffset: 0, channels: dim, spatial: lane)
            }

            var kCache = [Float](repeating: 0, count: dim * decodeMaxSeq)
            var vCache = [Float](repeating: 0, count: dim * decodeMaxSeq)
            kCache.withUnsafeMutableBufferPointer { dst in
                SurfaceIO.readFP16(from: decodeHandles[0].kCacheFull, into: dst, channelOffset: 0, channels: dim, spatial: decodeMaxSeq)
            }
            vCache.withUnsafeMutableBufferPointer { dst in
                SurfaceIO.readFP16(from: decodeHandles[0].vCacheFull, into: dst, channelOffset: 0, channels: dim, spatial: decodeMaxSeq)
            }

            for d in 0..<min(32, dim) {
                let expectedK = kOut[d * lane]
                let expectedV = vOut[d * lane]
                let cachedK = kCache[d * decodeMaxSeq + t]
                let cachedV = vCache[d * decodeMaxSeq + t]
                XCTAssertEqual(cachedK, expectedK, accuracy: 2e-3, "k-cache mismatch at token \(t), channel \(d)")
                XCTAssertEqual(cachedV, expectedV, accuracy: 2e-3, "v-cache mismatch at token \(t), channel \(d)")
            }

            var maskCache = [Float](repeating: 0, count: dim * decodeMaxSeq)
            maskCache.withUnsafeMutableBufferPointer { dst in
                SurfaceIO.readFP16(from: decodeHandles[0].maskCacheFull, into: dst, channelOffset: 0, channels: dim, spatial: decodeMaxSeq)
            }
            for d in 0..<min(32, dim) {
                XCTAssertEqual(maskCache[d * decodeMaxSeq + t], 0, accuracy: 1e-3, "mask not flipped at token \(t), channel \(d)")
                if t + 1 < decodeMaxSeq {
                    XCTAssertEqual(
                        maskCache[d * decodeMaxSeq + t + 1],
                        maskSentinel,
                        accuracy: 1e-1,
                        "future mask unexpectedly visible at token \(t), channel \(d)"
                    )
                }
            }
        }
    }

    func test_decode_kv_mask_progresses_across_tile_boundaries_on_hardware() throws {
        try requireANEHardwareTestsEnabled()

        let lane = DecodeKernelSet.defaultLaneSpatial
        let decodeMaxSeq = lane * 4
        let decodeSteps = lane * 2 + 1 // crosses 31->32 and 63->64 boundaries
        let dim = ModelConfig.dim
        let maskSentinel = Float(-1e4)

        let layers = LayerStorage<LayerWeights>(count: 1) { _ in
            let w = LayerWeights()
            fillLayerWeights(w, seed: 0xD00D42)
            return w
        }

        let decodeKernels = try LayerStorage<DecodeKernelSet>(count: 1, throwingInitializer: { i in
            try DecodeKernelSet(weights: layers[i], maxSeq: decodeMaxSeq)
        })
        let decodeHandles = [try DecodeSurfaceHandles(kernels: decodeKernels[0], logicalMaxSeq: decodeMaxSeq)]
        ForwardPass.initializeDecodeCachesAndMask(surfaceHandles: decodeHandles)

        var decodeState = try DecodeState(maxSeq: decodeMaxSeq)
        var rng = XorShift32(seed: 0xFACECAFE)

        for t in 0..<decodeSteps {
            let decodeToken = TensorBuffer(count: dim, zeroed: true)
            decodeToken.withUnsafeMutableBufferPointer { dst in
                for i in 0..<dim {
                    dst[i] = rng.nextFloat(range: -0.1...0.1)
                }
            }

            var decodeTimings = StepTimingBreakdown()
            try ForwardPass.runDecodeTimed(
                xCur: decodeToken,
                kernels: decodeKernels,
                surfaceHandles: decodeHandles,
                decodeState: &decodeState,
                timings: &decodeTimings,
                profiler: nil
            )

            if t == 31 || t == 32 || t == 63 {
                var kOut = [Float](repeating: 0, count: dim * lane)
                kOut.withUnsafeMutableBufferPointer { dst in
                    SurfaceIO.readFP16(from: decodeHandles[0].attnKOut, into: dst, channelOffset: 0, channels: dim, spatial: lane)
                }
                var kCache = [Float](repeating: 0, count: dim * decodeMaxSeq)
                kCache.withUnsafeMutableBufferPointer { dst in
                    SurfaceIO.readFP16(from: decodeHandles[0].kCacheFull, into: dst, channelOffset: 0, channels: dim, spatial: decodeMaxSeq)
                }
                for d in 0..<min(32, dim) {
                    XCTAssertEqual(
                        kCache[d * decodeMaxSeq + t],
                        kOut[d * lane],
                        accuracy: 2e-3,
                        "k-cache mismatch across tile boundary at token \(t), channel \(d)"
                    )
                }
            }
        }

        XCTAssertEqual(decodeState.step, decodeSteps)

        var maskCache = [Float](repeating: 0, count: dim * decodeMaxSeq)
        maskCache.withUnsafeMutableBufferPointer { dst in
            SurfaceIO.readFP16(from: decodeHandles[0].maskCacheFull, into: dst, channelOffset: 0, channels: dim, spatial: decodeMaxSeq)
        }
        for d in 0..<min(16, dim) {
            XCTAssertEqual(maskCache[d * decodeMaxSeq + 31], 0, accuracy: 1e-3)
            XCTAssertEqual(maskCache[d * decodeMaxSeq + 32], 0, accuracy: 1e-3)
            XCTAssertEqual(maskCache[d * decodeMaxSeq + 63], 0, accuracy: 1e-3)
            XCTAssertEqual(maskCache[d * decodeMaxSeq + 64], 0, accuracy: 1e-3)
            XCTAssertEqual(maskCache[d * decodeMaxSeq + 65], maskSentinel, accuracy: 1e-1)
        }
    }
}

import XCTest
@testable import Espresso

final class MetalAttentionKernelTests: XCTestCase {
    func test_metal_attention_matches_reference_on_small_problem() throws {
        let shape = try MetalAttentionShape(heads: 2, headDim: 4, seqLen: 4)
        let q: [Float] = [
            0.25, -0.50, 0.75, 1.00,
            -1.00, 0.50, 0.25, -0.25,
        ]
        let k: [Float] = [
            0.50, 0.25, -0.75, 1.00,
            1.00, -0.50, 0.25, 0.75,
            -0.25, 1.00, 0.50, -0.50,
            0.75, 0.50, -0.25, 0.25,

            -0.50, 0.75, 0.25, -1.00,
            0.25, -0.25, 1.00, 0.50,
            1.00, 0.50, -0.50, 0.25,
            -0.75, 0.25, 0.50, 1.00,
        ]
        let v: [Float] = [
            1.00, 0.00, 0.50, -0.50,
            0.50, 1.00, -0.25, 0.25,
            -0.75, 0.25, 1.25, 0.50,
            0.25, -0.50, 0.75, 1.50,

            -0.50, 1.00, 0.25, 0.75,
            0.75, -0.25, 1.00, -0.50,
            1.25, 0.50, -0.75, 0.25,
            0.00, 0.75, 0.50, 1.25,
        ]
        let mask = [Float](repeating: 0, count: shape.heads * shape.seqLen)

        let kernel = try MetalAttentionKernel()
        let actual = try kernel.run(q: q, k: k, v: v, mask: mask, shape: shape)
        let expected = referenceAttention(q: q, k: k, v: v, mask: mask, shape: shape)

        XCTAssertEqual(actual.count, expected.count)
        for (lhs, rhs) in zip(actual, expected) {
            XCTAssertEqual(lhs, rhs, accuracy: 1e-2)
        }
    }

    func test_metal_attention_benchmark_reports_positive_latency() throws {
        try requireHardwareBenchmarks()
        let shape = try MetalAttentionShape(heads: 12, headDim: 64, seqLen: 32)
        let kernel = try MetalAttentionKernel()
        let result = try kernel.benchmark(shape: shape, warmup: 3, iterations: 20, seed: 0xA11CE)
        print("Metal attention benchmark: mean=\(result.meanMs) ms median=\(result.medianMs) ms zeroCopy=\(result.zeroCopyBindings)")

        XCTAssertTrue(result.zeroCopyBindings)
        XCTAssertGreaterThan(result.medianMs, 0)
        XCTAssertEqual(result.iterations, 20)
    }

    private func requireHardwareBenchmarks(file: StaticString = #filePath, line: UInt = #line) throws {
        guard ProcessInfo.processInfo.environment["ANE_HARDWARE_TESTS"] == "1" else {
            throw XCTSkip("Set ANE_HARDWARE_TESTS=1 to run Metal hardware benchmarks")
        }
    }

    private func referenceAttention(
        q: [Float],
        k: [Float],
        v: [Float],
        mask: [Float],
        shape: MetalAttentionShape
    ) -> [Float] {
        let scale = 1.0 / sqrt(Float(shape.headDim))
        var output = [Float](repeating: 0, count: shape.heads * shape.headDim)

        for head in 0..<shape.heads {
            var logits = [Float](repeating: 0, count: shape.seqLen)
            for token in 0..<shape.seqLen {
                var dot: Float = 0
                let qBase = head * shape.headDim
                let kBase = (head * shape.seqLen + token) * shape.headDim
                for dim in 0..<shape.headDim {
                    dot += q[qBase + dim] * k[kBase + dim]
                }
                logits[token] = dot * scale + mask[head * shape.seqLen + token]
            }

            let maxLogit = logits.max() ?? 0
            var denom: Float = 0
            var weights = [Float](repeating: 0, count: shape.seqLen)
            for token in 0..<shape.seqLen {
                let value = exp(logits[token] - maxLogit)
                weights[token] = value
                denom += value
            }

            for dim in 0..<shape.headDim {
                var accum: Float = 0
                for token in 0..<shape.seqLen {
                    let normalized = weights[token] / denom
                    let vBase = (head * shape.seqLen + token) * shape.headDim
                    accum += normalized * v[vBase + dim]
                }
                output[head * shape.headDim + dim] = accum
            }
        }

        return output
    }
}

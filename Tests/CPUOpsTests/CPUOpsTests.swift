import ANETypes
import XCTest
@testable import CPUOps

// Deterministic RNG for reproducible tests
struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        return z ^ (z >> 31)
    }
}

@inline(__always)
private func randomFloat(_ rng: inout SplitMix64, min: Float, max: Float) -> Float {
    let u = Double(rng.next()) / Double(UInt64.max)
    return min + Float(u) * (max - min)
}

@inline(__always)
private func randomInt(_ rng: inout SplitMix64, upperBound: Int) -> Int {
    Int(rng.next() % UInt64(upperBound))
}

@inline(__always)
private func channelFirstIndex(dim: Int, seq: Int, seqLen: Int) -> Int {
    dim * seqLen + seq
}

@inline(__always)
private func vocabSeqIndex(vocab: Int, seq: Int, seqLen: Int) -> Int {
    vocab * seqLen + seq
}

@inline(__always)
private func embeddingIndex(token: Int, dim: Int, modelDim: Int) -> Int {
    token * modelDim + dim
}

@inline(__always)
private func ropeIndex(seq: Int, head: Int, i: Int, nHeads: Int, headDim: Int) -> Int {
    seq * nHeads * headDim + head * headDim + i
}

@inline(__always)
private func relativeError(_ a: Float, _ b: Float, floor: Float = 1e-6) -> Float {
    abs(a - b) / max(max(abs(a), abs(b)), floor)
}

private func crossEntropyReference(
    logits: [Float],
    targets: [UInt32],
    vocabSize: Int,
    seqLen: Int
) -> (loss: Float, dlogits: [Float]) {
    var totalLoss: Float = 0
    var dlogits = [Float](repeating: 0, count: vocabSize * seqLen)
    let invS = 1.0 / Float(seqLen)

    for t in 0..<seqLen {
        var maxv = -Float.greatestFiniteMagnitude
        for v in 0..<vocabSize {
            maxv = max(maxv, logits[vocabSeqIndex(vocab: v, seq: t, seqLen: seqLen)])
        }

        var probs = [Float](repeating: 0, count: vocabSize)
        var sum: Float = 0
        for v in 0..<vocabSize {
            let idx = vocabSeqIndex(vocab: v, seq: t, seqLen: seqLen)
            let e = expf(logits[idx] - maxv)
            probs[v] = e
            sum += e
        }
        for v in 0..<vocabSize {
            probs[v] /= sum
            let idx = vocabSeqIndex(vocab: v, seq: t, seqLen: seqLen)
            dlogits[idx] = probs[v] * invS
        }

        let tgt = Int(targets[t])
        totalLoss -= logf(probs[tgt] + 1e-10)
        dlogits[vocabSeqIndex(vocab: tgt, seq: t, seqLen: seqLen)] -= invS
    }

    return (loss: totalLoss / Float(seqLen), dlogits: dlogits)
}

private func ropeApplyReference(
    q: inout [Float],
    k: inout [Float],
    seqLen: Int,
    nHeads: Int,
    headDim: Int
) {
    for t in 0..<seqLen {
        for h in 0..<nHeads {
            for i in stride(from: 0, to: headDim, by: 2) {
                let freq = 1.0 / powf(10000.0, Float(i) / Float(headDim))
                let value = Float(t) * freq
                let cosv = cosf(value)
                let sinv = sinf(value)

                let off = ropeIndex(seq: t, head: h, i: i, nHeads: nHeads, headDim: headDim)
                let q0 = q[off]
                let q1 = q[off + 1]
                q[off] = q0 * cosv - q1 * sinv
                q[off + 1] = q0 * sinv + q1 * cosv

                let k0 = k[off]
                let k1 = k[off + 1]
                k[off] = k0 * cosv - k1 * sinv
                k[off + 1] = k0 * sinv + k1 * cosv
            }
        }
    }
}

private func ropeBackwardReference(
    dq: inout [Float],
    dk: inout [Float],
    seqLen: Int,
    nHeads: Int,
    headDim: Int
) {
    for t in 0..<seqLen {
        for h in 0..<nHeads {
            for i in stride(from: 0, to: headDim, by: 2) {
                let freq = 1.0 / powf(10000.0, Float(i) / Float(headDim))
                let value = Float(t) * freq
                let cosv = cosf(value)
                let sinv = sinf(value)

                let off = ropeIndex(seq: t, head: h, i: i, nHeads: nHeads, headDim: headDim)
                let dq0 = dq[off]
                let dq1 = dq[off + 1]
                dq[off] = dq0 * cosv + dq1 * sinv
                dq[off + 1] = -dq0 * sinv + dq1 * cosv

                let dk0 = dk[off]
                let dk1 = dk[off + 1]
                dk[off] = dk0 * cosv + dk1 * sinv
                dk[off + 1] = -dk0 * sinv + dk1 * cosv
            }
        }
    }
}

final class CPUOpsTests: XCTestCase {
    func test_rmsnorm_forward_known_values() {
        let dim = 4
        let seqLen = 2
        let x: [Float] = [1, 2, 3, 4, 5, 6, 7, 8] // channel-first [DIM, SEQ]
        let weights = [Float](repeating: 1.0, count: dim)
        var output = [Float](repeating: 0, count: dim * seqLen)

        x.withUnsafeBufferPointer { xPtr in
            weights.withUnsafeBufferPointer { wPtr in
                output.withUnsafeMutableBufferPointer { outPtr in
                    RMSNorm.forward(
                        output: outPtr.baseAddress!,
                        input: xPtr.baseAddress!,
                        weights: wPtr.baseAddress!,
                        dim: dim,
                        seqLen: seqLen
                    )
                }
            }
        }

        var expected = [Float](repeating: 0, count: dim * seqLen)
        for t in 0..<seqLen {
            var ss: Float = 0
            for i in 0..<dim {
                let value = x[channelFirstIndex(dim: i, seq: t, seqLen: seqLen)]
                ss += value * value
            }
            ss = ss / Float(dim) + 1e-5
            let rrms = 1.0 / sqrtf(ss)
            for i in 0..<dim {
                let idx = channelFirstIndex(dim: i, seq: t, seqLen: seqLen)
                expected[idx] = x[idx] * rrms * weights[i]
            }
        }

        for i in 0..<(dim * seqLen) {
            XCTAssertEqual(output[i], expected[i], accuracy: 1e-5, "Mismatch at index \(i)")
        }
    }

    func test_rmsnorm_workspace_matches_default_paths() {
        let dim = 8
        let seqLen = 5
        let count = dim * seqLen
        var rng = SplitMix64(seed: 99)

        let x = (0..<count).map { _ in randomFloat(&rng, min: -2.0, max: 2.0) }
        let weights = (0..<dim).map { _ in randomFloat(&rng, min: -1.5, max: 1.5) }
        let dy = (0..<count).map { _ in randomFloat(&rng, min: -1.0, max: 1.0) }

        var outDefault = [Float](repeating: 0, count: count)
        var outWorkspace = [Float](repeating: 0, count: count)
        var dxDefault = [Float](repeating: 0, count: count)
        var dxWorkspace = [Float](repeating: 0, count: count)
        var dwDefault = [Float](repeating: 0, count: dim)
        var dwWorkspace = [Float](repeating: 0, count: dim)

        x.withUnsafeBufferPointer { xPtr in
            weights.withUnsafeBufferPointer { wPtr in
                outDefault.withUnsafeMutableBufferPointer { outPtr in
                    RMSNorm.forward(
                        output: outPtr.baseAddress!,
                        input: xPtr.baseAddress!,
                        weights: wPtr.baseAddress!,
                        dim: dim,
                        seqLen: seqLen
                    )
                }
            }
        }

        var workspace = RMSNorm.Workspace(seqLen: seqLen)
        x.withUnsafeBufferPointer { xPtr in
            weights.withUnsafeBufferPointer { wPtr in
                outWorkspace.withUnsafeMutableBufferPointer { outPtr in
                    RMSNorm.forward(
                        output: outPtr.baseAddress!,
                        input: xPtr.baseAddress!,
                        weights: wPtr.baseAddress!,
                        dim: dim,
                        seqLen: seqLen,
                        workspace: workspace
                    )
                }
            }
        }

        for i in 0..<count {
            XCTAssertEqual(outWorkspace[i], outDefault[i], accuracy: 1e-6, "forward mismatch at \(i)")
        }

        x.withUnsafeBufferPointer { xPtr in
            weights.withUnsafeBufferPointer { wPtr in
                dy.withUnsafeBufferPointer { dyPtr in
                    dxDefault.withUnsafeMutableBufferPointer { dxPtr in
                        dwDefault.withUnsafeMutableBufferPointer { dwPtr in
                            RMSNorm.backward(
                                dx: dxPtr.baseAddress!,
                                dw: dwPtr.baseAddress!,
                                dy: dyPtr.baseAddress!,
                                x: xPtr.baseAddress!,
                                weights: wPtr.baseAddress!,
                                dim: dim,
                                seqLen: seqLen
                            )
                        }
                    }
                }
            }
        }

        x.withUnsafeBufferPointer { xPtr in
            weights.withUnsafeBufferPointer { wPtr in
                dy.withUnsafeBufferPointer { dyPtr in
                    dxWorkspace.withUnsafeMutableBufferPointer { dxPtr in
                        dwWorkspace.withUnsafeMutableBufferPointer { dwPtr in
                            RMSNorm.backward(
                                dx: dxPtr.baseAddress!,
                                dw: dwPtr.baseAddress!,
                                dy: dyPtr.baseAddress!,
                                x: xPtr.baseAddress!,
                                weights: wPtr.baseAddress!,
                                dim: dim,
                                seqLen: seqLen,
                                workspace: workspace
                            )
                        }
                    }
                }
            }
        }

        for i in 0..<count {
            XCTAssertEqual(dxWorkspace[i], dxDefault[i], accuracy: 1e-6, "backward dx mismatch at \(i)")
        }
        for i in 0..<dim {
            XCTAssertEqual(dwWorkspace[i], dwDefault[i], accuracy: 1e-5, "backward dw mismatch at \(i)")
        }
    }

    func test_rmsnorm_backward_numerical_gradient_check() {
        let dim = 4
        let seqLen = 2
        let count = dim * seqLen
        let h = 1e-4
        var rng = SplitMix64(seed: 42)

	        func backwardReferenceCParity(
	            dx: inout [Float],
	            dw: inout [Float],
	            dy: [Float],
            x: [Float],
            weights: [Float],
            dim: Int,
            seqLen: Int
        ) {
            var ss = [Float](repeating: 0, count: seqLen)
            for i in 0..<dim {
                for t in 0..<seqLen {
                    let idx = channelFirstIndex(dim: i, seq: t, seqLen: seqLen)
                    let value = x[idx]
                    ss[t] += value * value
                }
            }

            let invd = 1.0 / Float(dim)
            for t in 0..<seqLen {
                ss[t] = ss[t] * invd + 1e-5
            }

            var rrms = [Float](repeating: 0, count: seqLen)
            for t in 0..<seqLen {
                rrms[t] = 1.0 / sqrtf(ss[t])
            }

            var dot = [Float](repeating: 0, count: seqLen)
            for i in 0..<dim {
                for t in 0..<seqLen {
                    let idx = channelFirstIndex(dim: i, seq: t, seqLen: seqLen)
                    dot[t] += dy[idx] * x[idx] * weights[i]
                }
            }

            for t in 0..<seqLen {
                dot[t] = dot[t] * rrms[t] * rrms[t] * invd
            }

	            for i in 0..<dim {
	                var sum: Float = 0
	                for t in 0..<seqLen {
	                    let idx = channelFirstIndex(dim: i, seq: t, seqLen: seqLen)
	                    dx[idx] = (dy[idx] * weights[i] - x[idx] * dot[t]) * rrms[t]
	                    sum += dy[idx] * x[idx] * rrms[t]
	                }
	                dw[i] += sum
	            }
	        }

        func objective(
            xValues: [Float],
            weights: [Float],
            dy: [Float],
            dim: Int,
            seqLen: Int
        ) -> Double {
            var total = Double(0)
            for t in 0..<seqLen {
                var ss = Double(0)
                for i in 0..<dim {
                    let value = Double(xValues[channelFirstIndex(dim: i, seq: t, seqLen: seqLen)])
                    ss += value * value
                }
                let rrms = 1.0 / sqrt(ss / Double(dim) + 1e-5)
                for i in 0..<dim {
                    let idx = channelFirstIndex(dim: i, seq: t, seqLen: seqLen)
                    let out = Double(xValues[idx]) * rrms * Double(weights[i])
                    total += out * Double(dy[idx])
                }
            }
            return total
        }

        for config in 0..<20 {
            let x = (0..<count).map { _ in randomFloat(&rng, min: -1.5, max: 1.5) }
            let weights = (0..<dim).map { _ in randomFloat(&rng, min: -1.5, max: 1.5) }
            let dy = (0..<count).map { _ in randomFloat(&rng, min: -1.5, max: 1.5) }

            var dxAnalytic = [Float](repeating: 0, count: count)
            var dwFresh = [Float](repeating: 0, count: dim)

            x.withUnsafeBufferPointer { xPtr in
                weights.withUnsafeBufferPointer { wPtr in
                    dy.withUnsafeBufferPointer { dyPtr in
                        dxAnalytic.withUnsafeMutableBufferPointer { dxPtr in
                            dwFresh.withUnsafeMutableBufferPointer { dwPtr in
                                RMSNorm.backward(
                                    dx: dxPtr.baseAddress!,
                                    dw: dwPtr.baseAddress!,
                                    dy: dyPtr.baseAddress!,
                                    x: xPtr.baseAddress!,
                                    weights: wPtr.baseAddress!,
                                    dim: dim,
                                    seqLen: seqLen
                                )
                            }
                        }
                    }
                }
            }

            var dxReference = [Float](repeating: 0, count: count)
            var dwReference = [Float](repeating: 0, count: dim)
            backwardReferenceCParity(
                dx: &dxReference,
                dw: &dwReference,
                dy: dy,
                x: x,
                weights: weights,
                dim: dim,
                seqLen: seqLen
            )

            for i in 0..<count {
                XCTAssertEqual(dxAnalytic[i], dxReference[i], accuracy: 1e-6, "C parity mismatch at idx \(i), config \(config)")
            }
            for i in 0..<dim {
                XCTAssertEqual(dwFresh[i], dwReference[i], accuracy: 1e-5, "dw C parity mismatch at dim \(i), config \(config)")
            }

            let initialDW = [Float](repeating: 0.75, count: dim)
            var dwAccum = initialDW
            var dxAccum = [Float](repeating: 0, count: count)

            x.withUnsafeBufferPointer { xPtr in
                weights.withUnsafeBufferPointer { wPtr in
                    dy.withUnsafeBufferPointer { dyPtr in
                        dxAccum.withUnsafeMutableBufferPointer { dxPtr in
                            dwAccum.withUnsafeMutableBufferPointer { dwPtr in
                                RMSNorm.backward(
                                    dx: dxPtr.baseAddress!,
                                    dw: dwPtr.baseAddress!,
                                    dy: dyPtr.baseAddress!,
                                    x: xPtr.baseAddress!,
                                    weights: wPtr.baseAddress!,
                                    dim: dim,
                                    seqLen: seqLen
                                )
                            }
                        }
                    }
                }
            }

            for i in 0..<dim {
                XCTAssertEqual(dwAccum[i] - initialDW[i], dwFresh[i], accuracy: 1e-5, "dw should accumulate at dim \(i), config \(config)")
            }

	            let weightsForFiniteDifference: [Float] = [0.75, -1.25, 0.5, 2.0]
	            var dxForFiniteDifference = [Float](repeating: 0, count: count)
	            var dwForFiniteDifference = [Float](repeating: 0, count: dim)
	            x.withUnsafeBufferPointer { xPtr in
                weightsForFiniteDifference.withUnsafeBufferPointer { wPtr in
                    dy.withUnsafeBufferPointer { dyPtr in
                        dxForFiniteDifference.withUnsafeMutableBufferPointer { dxPtr in
                            dwForFiniteDifference.withUnsafeMutableBufferPointer { dwPtr in
                                RMSNorm.backward(
                                    dx: dxPtr.baseAddress!,
                                    dw: dwPtr.baseAddress!,
                                    dy: dyPtr.baseAddress!,
                                    x: xPtr.baseAddress!,
                                    weights: wPtr.baseAddress!,
                                    dim: dim,
                                    seqLen: seqLen
                                )
                            }
                        }
                    }
                }
            }

            var dxNumerical = [Float](repeating: 0, count: count)
            for i in 0..<count {
                var xPlus = x
                var xMinus = x
                xPlus[i] += Float(h)
                xMinus[i] -= Float(h)

                let fPlus = objective(xValues: xPlus, weights: weightsForFiniteDifference, dy: dy, dim: dim, seqLen: seqLen)
                let fMinus = objective(xValues: xMinus, weights: weightsForFiniteDifference, dy: dy, dim: dim, seqLen: seqLen)
                dxNumerical[i] = Float((fPlus - fMinus) / (2.0 * h))
            }

            for i in 0..<count {
                let a = dxForFiniteDifference[i]
                let b = dxNumerical[i]
                let scale = max(abs(a), abs(b))
                if scale > 1e-6 {
                    XCTAssertLessThan(relativeError(a, b), 1e-3, "Relative error too high at idx \(i), config \(config), unit-weight analytic=\(a), numeric=\(b)")
                } else {
                    XCTAssertEqual(a, b, accuracy: 1e-4, "Absolute error too high at idx \(i), config \(config)")
                }
            }
        }
    }

    func test_cross_entropy_uniform_logits() {
        let vocabSize = 100
        let seqLen = 4
        let logits = [Float](repeating: 0.0, count: vocabSize * seqLen)
        let targets: [UInt32] = [0, 1, 2, 3]
        var dlogits = [Float](repeating: 0.0, count: vocabSize * seqLen)

        let loss = logits.withUnsafeBufferPointer { logitsPtr in
            targets.withUnsafeBufferPointer { targetsPtr in
                dlogits.withUnsafeMutableBufferPointer { gradPtr in
                    CrossEntropy.lossAndGradient(
                        dlogits: gradPtr.baseAddress!,
                        logits: logitsPtr.baseAddress!,
                        targets: targetsPtr.baseAddress!,
                        vocabSize: vocabSize,
                        seqLen: seqLen
                    )
                }
            }
        }

        let expected = logf(Float(vocabSize))
        XCTAssertEqual(loss, expected, accuracy: 1e-4)
    }

    func test_cross_entropy_matches_reference_and_scaling() {
        let vocabSize = 7
        let seqLen = 5
        var rng = SplitMix64(seed: 777)
        let logits = (0..<(vocabSize * seqLen)).map { _ in randomFloat(&rng, min: -4.0, max: 4.0) }
        let targets = (0..<seqLen).map { _ in UInt32(randomInt(&rng, upperBound: vocabSize)) }
        var dlogits = [Float](repeating: 0, count: vocabSize * seqLen)

        let loss = logits.withUnsafeBufferPointer { logitsPtr in
            targets.withUnsafeBufferPointer { targetsPtr in
                dlogits.withUnsafeMutableBufferPointer { gradPtr in
                    CrossEntropy.lossAndGradient(
                        dlogits: gradPtr.baseAddress!,
                        logits: logitsPtr.baseAddress!,
                        targets: targetsPtr.baseAddress!,
                        vocabSize: vocabSize,
                        seqLen: seqLen
                    )
                }
            }
        }

        let reference = crossEntropyReference(
            logits: logits,
            targets: targets,
            vocabSize: vocabSize,
            seqLen: seqLen
        )

        XCTAssertEqual(loss, reference.loss, accuracy: 1e-6)
        for i in 0..<(vocabSize * seqLen) {
            XCTAssertEqual(dlogits[i], reference.dlogits[i], accuracy: 1e-6, "Mismatch at gradient index \(i)")
        }
    }

    func test_cross_entropy_gradient_sums_to_zero() {
        let vocabSize = 50
        let seqLen = 8
        var rng = SplitMix64(seed: 42)
        let logits = (0..<(vocabSize * seqLen)).map { _ in randomFloat(&rng, min: -4.0, max: 4.0) }
        let targets = (0..<seqLen).map { _ in UInt32(randomInt(&rng, upperBound: vocabSize)) }
        var dlogits = [Float](repeating: 0.0, count: vocabSize * seqLen)

        logits.withUnsafeBufferPointer { logitsPtr in
            targets.withUnsafeBufferPointer { targetsPtr in
                dlogits.withUnsafeMutableBufferPointer { gradPtr in
                    _ = CrossEntropy.lossAndGradient(
                        dlogits: gradPtr.baseAddress!,
                        logits: logitsPtr.baseAddress!,
                        targets: targetsPtr.baseAddress!,
                        vocabSize: vocabSize,
                        seqLen: seqLen
                    )
                }
            }
        }

        for t in 0..<seqLen {
            var sum: Float = 0
            for v in 0..<vocabSize {
                sum += dlogits[vocabSeqIndex(vocab: v, seq: t, seqLen: seqLen)]
            }
            XCTAssertLessThan(abs(sum), 1e-5, "Gradient sum should be zero at position \(t)")
        }
    }

    func test_cross_entropy_workspace_matches_default_api() {
        let vocabSize = 13
        let seqLen = 6
        var rng = SplitMix64(seed: 1701)
        let logits = (0..<(vocabSize * seqLen)).map { _ in randomFloat(&rng, min: -6.0, max: 6.0) }
        let targets = (0..<seqLen).map { _ in UInt32(randomInt(&rng, upperBound: vocabSize)) }

        var gradsDefault = [Float](repeating: 0, count: vocabSize * seqLen)
        var gradsWorkspace = [Float](repeating: 0, count: vocabSize * seqLen)

        let lossDefault = logits.withUnsafeBufferPointer { logitsPtr in
            targets.withUnsafeBufferPointer { targetsPtr in
                gradsDefault.withUnsafeMutableBufferPointer { gradPtr in
                    CrossEntropy.lossAndGradient(
                        dlogits: gradPtr.baseAddress!,
                        logits: logitsPtr.baseAddress!,
                        targets: targetsPtr.baseAddress!,
                        vocabSize: vocabSize,
                        seqLen: seqLen
                    )
                }
            }
        }

        var workspace = CrossEntropy.Workspace(vocabSize: vocabSize, seqLen: seqLen)
        let lossWorkspace = logits.withUnsafeBufferPointer { logitsPtr in
            targets.withUnsafeBufferPointer { targetsPtr in
                gradsWorkspace.withUnsafeMutableBufferPointer { gradPtr in
                    CrossEntropy.lossAndGradient(
                        dlogits: gradPtr.baseAddress!,
                        logits: logitsPtr.baseAddress!,
                        targets: targetsPtr.baseAddress!,
                        vocabSize: vocabSize,
                        seqLen: seqLen,
                        workspace: workspace
                    )
                }
            }
        }

        XCTAssertEqual(lossWorkspace, lossDefault, accuracy: 1e-6)
        for i in 0..<(vocabSize * seqLen) {
            XCTAssertEqual(gradsWorkspace[i], gradsDefault[i], accuracy: 1e-6, "gradient mismatch at \(i)")
        }
    }

    func test_adam_single_step_known_values() {
        var w: [Float] = [1.0]
        let g: [Float] = [0.1]
        var m: [Float] = [0.0]
        var v: [Float] = [0.0]

        w.withUnsafeMutableBufferPointer { wPtr in
            g.withUnsafeBufferPointer { gPtr in
                m.withUnsafeMutableBufferPointer { mPtr in
                    v.withUnsafeMutableBufferPointer { vPtr in
                        AdamOptimizer.update(
                            weights: wPtr.baseAddress!,
                            gradients: gPtr.baseAddress!,
                            m: mPtr.baseAddress!,
                            v: vPtr.baseAddress!,
                            count: 1,
                            timestep: 1,
                            lr: 0.001,
                            beta1: 0.9,
                            beta2: 0.999,
                            eps: 1e-8
                        )
                    }
                }
            }
        }

        XCTAssertEqual(m[0], 0.01, accuracy: 1e-6)
        XCTAssertEqual(v[0], 0.00001, accuracy: 1e-8)
        XCTAssertEqual(w[0], 0.999, accuracy: 1e-6)
    }

    func test_adam_bias_correction() {
        var w: [Float] = [0.0]
        let g: [Float] = [1.0]
        var m: [Float] = [0.0]
        var v: [Float] = [0.0]
        let beta1: Float = 0.9
        let beta2: Float = 0.999

        for t in 1...200 {
            w.withUnsafeMutableBufferPointer { wPtr in
                g.withUnsafeBufferPointer { gPtr in
                    m.withUnsafeMutableBufferPointer { mPtr in
                        v.withUnsafeMutableBufferPointer { vPtr in
                            AdamOptimizer.update(
                                weights: wPtr.baseAddress!,
                                gradients: gPtr.baseAddress!,
                                m: mPtr.baseAddress!,
                                v: vPtr.baseAddress!,
                                count: 1,
                                timestep: t,
                                lr: 0.001,
                                beta1: beta1,
                                beta2: beta2,
                                eps: 1e-8
                            )
                        }
                    }
                }
            }
        }

        let t: Float = 200
        let bc1 = 1.0 - powf(beta1, t)
        let bc2 = 1.0 - powf(beta2, t)
        let mHat = m[0] / bc1
        let vHat = v[0] / bc2

        XCTAssertLessThan(abs(mHat - 1.0), 0.01)
        XCTAssertLessThan(abs(vHat - 1.0), 0.01)
    }

    func test_adam_multi_element_reference_parity() {
        let count = 9
        let steps = 6
        let lr: Float = 0.001
        let beta1: Float = 0.9
        let beta2: Float = 0.999
        let eps: Float = 1e-8
        var rng = SplitMix64(seed: 1337)

        var w = (0..<count).map { _ in randomFloat(&rng, min: -0.5, max: 0.5) }
        var m = (0..<count).map { _ in randomFloat(&rng, min: -0.1, max: 0.1) }
        var v = (0..<count).map { _ in randomFloat(&rng, min: 0.0, max: 0.2) }

        var wRef = w
        var mRef = m
        var vRef = v
        let gradientsPerStep = (0..<steps).map { _ in
            (0..<count).map { _ in randomFloat(&rng, min: -1.5, max: 1.5) }
        }

        for t in 1...steps {
            let gradients = gradientsPerStep[t - 1]

            w.withUnsafeMutableBufferPointer { wPtr in
                gradients.withUnsafeBufferPointer { gPtr in
                    m.withUnsafeMutableBufferPointer { mPtr in
                        v.withUnsafeMutableBufferPointer { vPtr in
                            AdamOptimizer.update(
                                weights: wPtr.baseAddress!,
                                gradients: gPtr.baseAddress!,
                                m: mPtr.baseAddress!,
                                v: vPtr.baseAddress!,
                                count: count,
                                timestep: t,
                                lr: lr,
                                beta1: beta1,
                                beta2: beta2,
                                eps: eps
                            )
                        }
                    }
                }
            }

            let bc1 = 1.0 - powf(beta1, Float(t))
            let bc2 = 1.0 - powf(beta2, Float(t))
            for i in 0..<count {
                let g = gradients[i]
                mRef[i] = beta1 * mRef[i] + (1.0 - beta1) * g
                vRef[i] = beta2 * vRef[i] + (1.0 - beta2) * g * g
                let mHat = mRef[i] / bc1
                let vHat = vRef[i] / bc2
                wRef[i] -= lr * mHat / (sqrtf(vHat) + eps)
            }
        }

        for i in 0..<count {
            XCTAssertEqual(m[i], mRef[i], accuracy: 1e-7, "m mismatch at index \(i)")
            XCTAssertEqual(v[i], vRef[i], accuracy: 1e-7, "v mismatch at index \(i)")
            XCTAssertEqual(w[i], wRef[i], accuracy: 1e-7, "w mismatch at index \(i)")
        }
    }

    func test_embedding_lookup_correct_rows() {
        let dim = 4
        let seqLen = 3
        let vocab = 5
        let tokens: [UInt32] = [3, 1, 0]
        var embedding = [Float](repeating: 0, count: vocab * dim)

        for tok in 0..<vocab {
            for d in 0..<dim {
                embedding[embeddingIndex(token: tok, dim: d, modelDim: dim)] = Float(tok * 100 + d)
            }
        }

        var output = [Float](repeating: 0, count: dim * seqLen)

        output.withUnsafeMutableBufferPointer { outPtr in
            embedding.withUnsafeBufferPointer { embPtr in
                tokens.withUnsafeBufferPointer { tokPtr in
                    Embedding.lookup(
                        output: outPtr.baseAddress!,
                        embedding: embPtr.baseAddress!,
                        tokens: tokPtr.baseAddress!,
                        vocabSize: vocab,
                        dim: dim,
                        seqLen: seqLen
                    )
                }
            }
        }

        for t in 0..<seqLen {
            for d in 0..<dim {
                let expected = embedding[embeddingIndex(token: Int(tokens[t]), dim: d, modelDim: dim)]
                let actual = output[channelFirstIndex(dim: d, seq: t, seqLen: seqLen)]
                XCTAssertEqual(actual, expected, accuracy: 0.0, "Mismatch at (d=\(d), t=\(t))")
            }
        }
    }

    func test_embedding_backward_accumulates() {
        let dim = 4
        let seqLen = 3
        let vocab = 5
        let tokens: [UInt32] = [2, 2, 1]
        var dx = [Float](repeating: 0, count: dim * seqLen)
        for d in 0..<dim {
            for t in 0..<seqLen {
                dx[channelFirstIndex(dim: d, seq: t, seqLen: seqLen)] = Float(10 * d + t + 1)
            }
        }
        var dEmbedding = [Float](repeating: 0.5, count: vocab * dim)
        var expected = [Float](repeating: 0.5, count: vocab * dim)
        for t in 0..<seqLen {
            let token = Int(tokens[t])
            for d in 0..<dim {
                expected[embeddingIndex(token: token, dim: d, modelDim: dim)] += dx[channelFirstIndex(dim: d, seq: t, seqLen: seqLen)]
            }
        }

        dEmbedding.withUnsafeMutableBufferPointer { dEmbPtr in
            dx.withUnsafeBufferPointer { dxPtr in
                tokens.withUnsafeBufferPointer { tokPtr in
                    Embedding.backward(
                        dEmbedding: dEmbPtr.baseAddress!,
                        dx: dxPtr.baseAddress!,
                        tokens: tokPtr.baseAddress!,
                        vocabSize: vocab,
                        dim: dim,
                        seqLen: seqLen
                    )
                }
            }
        }

        for i in 0..<(vocab * dim) {
            XCTAssertEqual(dEmbedding[i], expected[i], accuracy: 0.0, "Mismatch at index \(i)")
        }
    }

    func test_rope_forward_backward_matches_reference() {
        let configs: [(seqLen: Int, nHeads: Int, headDim: Int)] = [
            (seqLen: 1, nHeads: 1, headDim: 2),
            (seqLen: 3, nHeads: 2, headDim: 6),
            (seqLen: 4, nHeads: 2, headDim: 8),
        ]
        var rng = SplitMix64(seed: 2026)

        for (configIndex, cfg) in configs.enumerated() {
            let count = cfg.seqLen * cfg.nHeads * cfg.headDim
            var q = (0..<count).map { _ in randomFloat(&rng, min: -1.0, max: 1.0) }
            var k = (0..<count).map { _ in randomFloat(&rng, min: -1.0, max: 1.0) }

            var qExpected = q
            var kExpected = k
            ropeApplyReference(
                q: &qExpected,
                k: &kExpected,
                seqLen: cfg.seqLen,
                nHeads: cfg.nHeads,
                headDim: cfg.headDim
            )

            q.withUnsafeMutableBufferPointer { qPtr in
                k.withUnsafeMutableBufferPointer { kPtr in
                    RoPE.apply(
                        q: qPtr.baseAddress!,
                        k: kPtr.baseAddress!,
                        seqLen: cfg.seqLen,
                        nHeads: cfg.nHeads,
                        headDim: cfg.headDim
                    )
                }
            }

            for i in 0..<count {
                XCTAssertEqual(q[i], qExpected[i], accuracy: 1e-6, "Forward q mismatch at config \(configIndex), idx \(i)")
                XCTAssertEqual(k[i], kExpected[i], accuracy: 1e-6, "Forward k mismatch at config \(configIndex), idx \(i)")
            }

            var dq = (0..<count).map { _ in randomFloat(&rng, min: -1.0, max: 1.0) }
            var dk = (0..<count).map { _ in randomFloat(&rng, min: -1.0, max: 1.0) }
            var dqExpected = dq
            var dkExpected = dk
            ropeBackwardReference(
                dq: &dqExpected,
                dk: &dkExpected,
                seqLen: cfg.seqLen,
                nHeads: cfg.nHeads,
                headDim: cfg.headDim
            )

            dq.withUnsafeMutableBufferPointer { dqPtr in
                dk.withUnsafeMutableBufferPointer { dkPtr in
                    RoPE.backward(
                        dq: dqPtr.baseAddress!,
                        dk: dkPtr.baseAddress!,
                        seqLen: cfg.seqLen,
                        nHeads: cfg.nHeads,
                        headDim: cfg.headDim
                    )
                }
            }

            for i in 0..<count {
                XCTAssertEqual(dq[i], dqExpected[i], accuracy: 1e-6, "Backward dq mismatch at config \(configIndex), idx \(i)")
                XCTAssertEqual(dk[i], dkExpected[i], accuracy: 1e-6, "Backward dk mismatch at config \(configIndex), idx \(i)")
            }
        }
    }

    func test_rope_forward_backward_consistency() {
        let seqLen = 4
        let nHeads = 2
        let headDim = 8
        let count = seqLen * nHeads * headDim
        var rng = SplitMix64(seed: 42)

        var q = (0..<count).map { _ in randomFloat(&rng, min: -1.0, max: 1.0) }
        var k = (0..<count).map { _ in randomFloat(&rng, min: -1.0, max: 1.0) }
        let originalQ = q
        let originalK = k

        q.withUnsafeMutableBufferPointer { qPtr in
            k.withUnsafeMutableBufferPointer { kPtr in
                RoPE.apply(
                    q: qPtr.baseAddress!,
                    k: kPtr.baseAddress!,
                    seqLen: seqLen,
                    nHeads: nHeads,
                    headDim: headDim
                )
            }
        }

        q.withUnsafeMutableBufferPointer { qPtr in
            k.withUnsafeMutableBufferPointer { kPtr in
                RoPE.backward(
                    dq: qPtr.baseAddress!,
                    dk: kPtr.baseAddress!,
                    seqLen: seqLen,
                    nHeads: nHeads,
                    headDim: headDim
                )
            }
        }

        for t in 0..<seqLen {
            for h in 0..<nHeads {
                for i in 0..<headDim {
                    let idx = ropeIndex(seq: t, head: h, i: i, nHeads: nHeads, headDim: headDim)
                    XCTAssertEqual(q[idx], originalQ[idx], accuracy: 1e-6, "q mismatch at (t=\(t), h=\(h), i=\(i))")
                    XCTAssertEqual(k[idx], originalK[idx], accuracy: 1e-6, "k mismatch at (t=\(t), h=\(h), i=\(i))")
                }
            }
        }
    }

    func test_silu_forward_matches_closed_form() {
        let samples: [Float] = [-12, -5, -1, -0.25, 0, 0.25, 1, 5, 12]
        for x in samples {
            let expected = x / (1.0 + expf(-x))
            XCTAssertEqual(SiLU.forward(x), expected, accuracy: 1e-7, "Forward mismatch at x=\(x)")
        }
    }

    func test_silu_backward_matches_numerical_derivative_of_forward() {
        var rng = SplitMix64(seed: 42)
        let h = 1e-4

        for idx in 0..<100 {
            let x = randomFloat(&rng, min: -5.0, max: 5.0)
            let analytic = SiLU.backward(x)
            let xd = Double(x)
            let fp = (xd + h) / (1.0 + exp(-(xd + h)))
            let fm = (xd - h) / (1.0 + exp(-(xd - h)))
            let numerical = Float((fp - fm) / (2.0 * h))

            if abs(analytic) > 1e-6 {
                XCTAssertLessThan(relativeError(analytic, numerical), 1e-3, "Relative derivative error too high at sample \(idx), x=\(x)")
            } else {
                XCTAssertEqual(analytic, numerical, accuracy: 1e-6, "Absolute derivative error too high at sample \(idx), x=\(x)")
            }
        }
    }
}

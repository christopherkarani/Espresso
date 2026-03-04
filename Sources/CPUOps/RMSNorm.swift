import Accelerate

public enum RMSNorm {
    /// Channel-first forward: x[dim, seq], w[dim] -> out[dim, seq]
    public static func forward(
        output: UnsafeMutablePointer<Float>,
        input: UnsafePointer<Float>,
        weights: UnsafePointer<Float>,
        dim: Int,
        seqLen: Int
    ) {
        precondition(dim > 0)
        precondition(seqLen > 0)

        let n = vDSP_Length(seqLen)
        let tmp = UnsafeMutablePointer<Float>.allocate(capacity: seqLen)
        let ss = UnsafeMutablePointer<Float>.allocate(capacity: seqLen)
        ss.initialize(repeating: 0, count: seqLen)
        defer {
            tmp.deallocate()
            ss.deallocate()
        }

        for i in 0..<dim {
            let row = input + (i * seqLen)
            vDSP_vmul(row, 1, row, 1, tmp, 1, n)
            vDSP_vadd(tmp, 1, ss, 1, ss, 1, n)
        }

        var invd = 1.0 / Float(dim)
        var eps: Float = 1e-5
        vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, n)

        var count32 = Int32(seqLen)
        vvrsqrtf(ss, ss, &count32)

        for i in 0..<dim {
            let row = input + (i * seqLen)
            let outRow = output + (i * seqLen)
            vDSP_vmul(row, 1, ss, 1, outRow, 1, n)
            var wi = weights[i]
            vDSP_vsmul(outRow, 1, &wi, outRow, 1, n)
        }
    }

    /// Channel-first backward: computes dx, ACCUMULATES into dw (dw[i] += ...)
    public static func backward(
        dx: UnsafeMutablePointer<Float>,
        dw: UnsafeMutablePointer<Float>,
        dy: UnsafePointer<Float>,
        x: UnsafePointer<Float>,
        weights: UnsafePointer<Float>,
        dim: Int,
        seqLen: Int
    ) {
        precondition(dim > 0)
        precondition(seqLen > 0)

        let n = vDSP_Length(seqLen)
        let tmp = UnsafeMutablePointer<Float>.allocate(capacity: seqLen)
        let ss = UnsafeMutablePointer<Float>.allocate(capacity: seqLen)
        let rrms = UnsafeMutablePointer<Float>.allocate(capacity: seqLen)
        let dot = UnsafeMutablePointer<Float>.allocate(capacity: seqLen)
        ss.initialize(repeating: 0, count: seqLen)
        dot.initialize(repeating: 0, count: seqLen)
        defer {
            tmp.deallocate()
            ss.deallocate()
            rrms.deallocate()
            dot.deallocate()
        }

        for i in 0..<dim {
            let row = x + (i * seqLen)
            vDSP_vmul(row, 1, row, 1, tmp, 1, n)
            vDSP_vadd(tmp, 1, ss, 1, ss, 1, n)
        }

        var invd = 1.0 / Float(dim)
        var eps: Float = 1e-5
        vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, n)

        var count32 = Int32(seqLen)
        vvrsqrtf(rrms, ss, &count32)

        for i in 0..<dim {
            let dyRow = dy + (i * seqLen)
            let xRow = x + (i * seqLen)
            vDSP_vmul(dyRow, 1, xRow, 1, tmp, 1, n)
            var wi = weights[i]
            vDSP_vsma(tmp, 1, &wi, dot, 1, dot, 1, n)
        }

        vDSP_vmul(rrms, 1, rrms, 1, ss, 1, n)     // ss = rrms^2
        vDSP_vsmul(ss, 1, &invd, ss, 1, n)         // ss = rrms^2 / d
        vDSP_vmul(dot, 1, ss, 1, dot, 1, n)        // dot = dot * rrms^2 / d

	        for i in 0..<dim {
	            let xRow = x + (i * seqLen)
	            let dyRow = dy + (i * seqLen)
	            let dxRow = dx + (i * seqLen)

	            // Correct: dx = rrms * (dy*w - x*dot), where dot already includes w and rrms^2/d.
	            var wi = weights[i]
	            vDSP_vsmul(dyRow, 1, &wi, dxRow, 1, n)     // dxRow = dy*w
	            vDSP_vmul(xRow, 1, dot, 1, tmp, 1, n)      // tmp = x*dot
	            // vDSP_vsub computes second input minus first input: dxRow = dxRow - tmp
	            vDSP_vsub(tmp, 1, dxRow, 1, dxRow, 1, n)
	            vDSP_vmul(dxRow, 1, rrms, 1, dxRow, 1, n)  // dxRow *= rrms

	            vDSP_vmul(dyRow, 1, xRow, 1, tmp, 1, n)
	            vDSP_vmul(tmp, 1, rrms, 1, tmp, 1, n)
	            var s: Float = 0
            vDSP_sve(tmp, 1, &s, n)
            dw[i] += s // ACCUMULATE, do not overwrite
        }
    }
}

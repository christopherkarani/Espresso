import Accelerate

/// Minimal BLAS shims used by Espresso.
///
/// We call CBLAS via `_silgen_name` to avoid deprecation warnings on the
/// `cblas_sgemm` symbol in the macOS 13.3+ Accelerate overlay.
public enum BLAS {
    @_silgen_name("cblas_sgemm")
    private static func cblas_sgemm_shim(
        _ order: CBLAS_ORDER,
        _ transA: CBLAS_TRANSPOSE,
        _ transB: CBLAS_TRANSPOSE,
        _ m: Int32,
        _ n: Int32,
        _ k: Int32,
        _ alpha: Float,
        _ a: UnsafePointer<Float>,
        _ lda: Int32,
        _ b: UnsafePointer<Float>,
        _ ldb: Int32,
        _ beta: Float,
        _ c: UnsafeMutablePointer<Float>,
        _ ldc: Int32
    )

    @inline(__always)
    public static func sgemm(
        _ order: CBLAS_ORDER,
        _ transA: CBLAS_TRANSPOSE,
        _ transB: CBLAS_TRANSPOSE,
        m: Int32,
        n: Int32,
        k: Int32,
        alpha: Float,
        a: UnsafePointer<Float>,
        lda: Int32,
        b: UnsafePointer<Float>,
        ldb: Int32,
        beta: Float,
        c: UnsafeMutablePointer<Float>,
        ldc: Int32
    ) {
        cblas_sgemm_shim(order, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }
}


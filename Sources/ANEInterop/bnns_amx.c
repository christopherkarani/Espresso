#include "ane_interop.h"

#include <Accelerate/Accelerate.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

int ane_interop_bnns_fp32_gemv(const float *weights,
                               const float *input,
                               float *out,
                               int rows,
                               int dim,
                               int n_threads) {
    if (weights == NULL || input == NULL || out == NULL || rows <= 0 || dim <= 0) {
        return -1;
    }

    BNNSNDArrayDescriptor in_desc;
    memset(&in_desc, 0, sizeof(in_desc));
    in_desc.layout = BNNSDataLayoutVector;
    in_desc.size[0] = (size_t)dim;
    in_desc.data = (void *)(uintptr_t)input;
    in_desc.data_type = BNNSDataTypeFloat32;
    in_desc.data_scale = 1.0f;

    BNNSNDArrayDescriptor w_desc;
    memset(&w_desc, 0, sizeof(w_desc));
    w_desc.layout = BNNSDataLayoutRowMajorMatrix;
    w_desc.size[0] = (size_t)dim;
    w_desc.size[1] = (size_t)rows;
    w_desc.data = (void *)(uintptr_t)weights;
    w_desc.data_type = BNNSDataTypeFloat32;
    w_desc.data_scale = 1.0f;

    BNNSNDArrayDescriptor out_desc;
    memset(&out_desc, 0, sizeof(out_desc));
    out_desc.layout = BNNSDataLayoutVector;
    out_desc.size[0] = (size_t)rows;
    out_desc.data = out;
    out_desc.data_type = BNNSDataTypeFloat32;
    out_desc.data_scale = 1.0f;

    BNNSLayerParametersFullyConnected layer;
    memset(&layer, 0, sizeof(layer));
    layer.i_desc = in_desc;
    layer.w_desc = w_desc;
    layer.o_desc = out_desc;
    layer.activation.function = BNNSActivationFunctionIdentity;

    BNNSFilterParameters filter_params;
    memset(&filter_params, 0, sizeof(filter_params));
    filter_params.n_threads = n_threads > 0 ? (size_t)n_threads : 1;

    BNNSFilter filter = BNNSFilterCreateLayerFullyConnected(&layer, &filter_params);
    if (filter == NULL) {
        return -1;
    }
    int rc = BNNSFilterApply(filter, input, out);
    BNNSFilterDestroy(filter);
    return rc == 0 ? 0 : -1;
}

int ane_interop_bnns_fp32_gemm(const float *a,
                               const float *b,
                               float *c,
                               int m,
                               int n,
                               int k,
                               int n_threads) {
    if (a == NULL || b == NULL || c == NULL || m <= 0 || n <= 0 || k <= 0) {
        return -1;
    }
    (void)n_threads;
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        m,
        n,
        k,
        1.0f,
        a,
        k,
        b,
        n,
        0.0f,
        c,
        n
    );
    return 0;
}

int ane_interop_amx_shared_resource_hint(int enable, int worker_index, int cluster_concurrency) {
#ifdef SYS_bsdthread_ctl
    long rc = syscall(
        SYS_bsdthread_ctl,
        (void *)(uintptr_t)0x2000,
        (void *)(uintptr_t)(unsigned)enable,
        (void *)(uintptr_t)(unsigned)worker_index,
        (void *)(uintptr_t)(unsigned)cluster_concurrency
    );
    return rc == 0 ? 0 : -1;
#else
    (void)enable;
    (void)worker_index;
    (void)cluster_concurrency;
    return -1;
#endif
}

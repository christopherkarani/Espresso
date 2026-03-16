#include <limits.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dispatch/dispatch.h>

#if defined(__aarch64__) || defined(__arm64__)
#include <arm_neon.h>
#define ARGMAX_MAX_NVECS 64  /* supports up to spatial=512 (512/8=64) */
#endif

#include "ane_interop.h"

static bool mul_size_overflow(size_t a, size_t b, size_t *out) {
#if __has_builtin(__builtin_mul_overflow)
    return __builtin_mul_overflow(a, b, out);
#else
    if (a != 0 && b > SIZE_MAX / a) return true;
    *out = a * b;
    return false;
#endif
}

static bool add_size_overflow(size_t a, size_t b, size_t *out) {
#if __has_builtin(__builtin_add_overflow)
    return __builtin_add_overflow(a, b, out);
#else
    if (b > SIZE_MAX - a) return true;
    *out = a + b;
    return false;
#endif
}

bool ane_interop_io_copy(IOSurfaceRef dst, int dst_ch_off,
                         IOSurfaceRef src, int src_ch_off,
                         int channels, int spatial) {
    if (!dst || !src) return false;
    if (dst_ch_off < 0 || src_ch_off < 0 || channels < 0 || spatial < 0) return false;
    if (channels == 0 || spatial == 0) return true;

    size_t dstOffElems, srcOffElems, elemCount;
    size_t dstOffBytes, srcOffBytes, bytes;
    size_t spatialSz = (size_t)spatial;
    size_t channelsSz = (size_t)channels;
    if (mul_size_overflow((size_t)dst_ch_off, spatialSz, &dstOffElems)) return false;
    if (mul_size_overflow((size_t)src_ch_off, spatialSz, &srcOffElems)) return false;
    if (mul_size_overflow(channelsSz, spatialSz, &elemCount)) return false;
    if (mul_size_overflow(dstOffElems, sizeof(_Float16), &dstOffBytes)) return false;
    if (mul_size_overflow(srcOffElems, sizeof(_Float16), &srcOffBytes)) return false;
    if (mul_size_overflow(elemCount, sizeof(_Float16), &bytes)) return false;

    bool lockedDst = false;
    bool lockedSrc = false;
    bool ok = false;

    if (dst == src) {
        if (IOSurfaceLock(dst, 0, NULL) != kIOReturnSuccess) return false;
        lockedDst = true;

        void *base = IOSurfaceGetBaseAddress(dst);
        if (!base) goto cleanup;

        size_t allocSize = IOSurfaceGetAllocSize(dst);
        if (dstOffBytes > allocSize || srcOffBytes > allocSize) goto cleanup;
        if (bytes > allocSize - dstOffBytes || bytes > allocSize - srcOffBytes) goto cleanup;

        memmove(((_Float16 *)base) + dstOffElems, ((const _Float16 *)base) + srcOffElems, bytes);
        ok = true;
        goto cleanup;
    }

    if (IOSurfaceLock(dst, 0, NULL) != kIOReturnSuccess) return false;
    lockedDst = true;
    if (IOSurfaceLock(src, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) goto cleanup;
    lockedSrc = true;

    void *dstBase = IOSurfaceGetBaseAddress(dst);
    const void *srcBase = IOSurfaceGetBaseAddress(src);
    if (!dstBase || !srcBase) goto cleanup;

    size_t dstSize = IOSurfaceGetAllocSize(dst);
    size_t srcSize = IOSurfaceGetAllocSize(src);
    if (dstOffBytes > dstSize || srcOffBytes > srcSize) goto cleanup;
    if (bytes > dstSize - dstOffBytes || bytes > srcSize - srcOffBytes) goto cleanup;

    memmove(((_Float16 *)dstBase) + dstOffElems, ((const _Float16 *)srcBase) + srcOffElems, bytes);
    ok = true;

cleanup:
    if (lockedSrc) IOSurfaceUnlock(src, kIOSurfaceLockReadOnly, NULL);
    if (lockedDst) IOSurfaceUnlock(dst, 0, NULL);
    return ok;
}

bool ane_interop_io_copy_fp16_spatial_slice(IOSurfaceRef dst,
                                            int dst_ch_off,
                                            int dst_spatial_index,
                                            int dst_spatial,
                                            IOSurfaceRef src,
                                            int src_ch_off,
                                            int src_spatial_index,
                                            int src_spatial,
                                            int channels) {
    if (!dst || !src) return false;
    if (dst_ch_off < 0 || src_ch_off < 0) return false;
    if (dst_spatial_index < 0 || src_spatial_index < 0) return false;
    if (dst_spatial < 0 || src_spatial < 0 || channels < 0) return false;
    if (channels == 0) return true;
    if (dst_spatial == 0 || src_spatial == 0) return false;
    if (dst_spatial_index >= dst_spatial || src_spatial_index >= src_spatial) return false;

    size_t dstElems, srcElems;
    size_t dstBytes, srcBytes;
    size_t dstSpatialSz = (size_t)dst_spatial;
    size_t srcSpatialSz = (size_t)src_spatial;

    // Bounds check: ensure the highest indexed element is within alloc size.
    // maxIndex = (ch_off + channels - 1) * spatial + spatial_index
    if (channels > INT_MAX - dst_ch_off) return false;
    if (channels > INT_MAX - src_ch_off) return false;

    size_t dstMaxCh = (size_t)(dst_ch_off + channels - 1);
    size_t srcMaxCh = (size_t)(src_ch_off + channels - 1);

    size_t dstMaxIdxElems, srcMaxIdxElems;
    if (mul_size_overflow(dstMaxCh, dstSpatialSz, &dstMaxIdxElems)) return false;
    if (mul_size_overflow(srcMaxCh, srcSpatialSz, &srcMaxIdxElems)) return false;
    if (add_size_overflow(dstMaxIdxElems, (size_t)dst_spatial_index, &dstMaxIdxElems)) return false;
    if (add_size_overflow(srcMaxIdxElems, (size_t)src_spatial_index, &srcMaxIdxElems)) return false;

    if (add_size_overflow(dstMaxIdxElems, 1, &dstElems)) return false;
    if (add_size_overflow(srcMaxIdxElems, 1, &srcElems)) return false;

    if (mul_size_overflow(dstElems, sizeof(_Float16), &dstBytes)) return false;
    if (mul_size_overflow(srcElems, sizeof(_Float16), &srcBytes)) return false;

    bool lockedDst = false;
    bool lockedSrc = false;
    bool ok = false;

    if (IOSurfaceLock(dst, 0, NULL) != kIOReturnSuccess) return false;
    lockedDst = true;

    if (src != dst) {
        if (IOSurfaceLock(src, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) goto cleanup;
        lockedSrc = true;
    }

    void *dstBase = IOSurfaceGetBaseAddress(dst);
    const void *srcBase = IOSurfaceGetBaseAddress(src);
    if (!dstBase || !srcBase) goto cleanup;

    size_t dstAlloc = IOSurfaceGetAllocSize(dst);
    size_t srcAlloc = IOSurfaceGetAllocSize(src);
    if (dstBytes > dstAlloc) goto cleanup;
    if (srcBytes > srcAlloc) goto cleanup;

    const _Float16 *srcF16 = (const _Float16 *)srcBase;
    _Float16 *dstF16 = (_Float16 *)dstBase;

    for (int c = 0; c < channels; c++) {
        size_t dc = (size_t)(dst_ch_off + c);
        size_t sc = (size_t)(src_ch_off + c);
        size_t dIdx = dc * dstSpatialSz + (size_t)dst_spatial_index;
        size_t sIdx = sc * srcSpatialSz + (size_t)src_spatial_index;
        dstF16[dIdx] = srcF16[sIdx];
    }

    ok = true;

cleanup:
    if (lockedSrc) IOSurfaceUnlock(src, kIOSurfaceLockReadOnly, NULL);
    if (lockedDst) IOSurfaceUnlock(dst, 0, NULL);
    return ok;
}

bool ane_interop_io_copy_two_fp16_spatial_slices(
    IOSurfaceRef dst0,
    int dst0_ch_off,
    int dst0_spatial_index,
    int dst0_spatial,
    IOSurfaceRef src0,
    int src0_ch_off,
    int src0_spatial_index,
    int src0_spatial,
    IOSurfaceRef dst1,
    int dst1_ch_off,
    int dst1_spatial_index,
    int dst1_spatial,
    IOSurfaceRef src1,
    int src1_ch_off,
    int src1_spatial_index,
    int src1_spatial,
    int channels
) {
    if (!dst0 || !src0 || !dst1 || !src1) return false;
    if (dst0_ch_off < 0 || src0_ch_off < 0 || dst1_ch_off < 0 || src1_ch_off < 0) return false;
    if (dst0_spatial_index < 0 || src0_spatial_index < 0 || dst1_spatial_index < 0 || src1_spatial_index < 0) return false;
    if (dst0_spatial <= 0 || src0_spatial <= 0 || dst1_spatial <= 0 || src1_spatial <= 0 || channels < 0) return false;
    if (channels == 0) return true;
    if (dst0_spatial_index >= dst0_spatial || src0_spatial_index >= src0_spatial) return false;
    if (dst1_spatial_index >= dst1_spatial || src1_spatial_index >= src1_spatial) return false;
    if (channels > INT_MAX - dst0_ch_off || channels > INT_MAX - src0_ch_off) return false;
    if (channels > INT_MAX - dst1_ch_off || channels > INT_MAX - src1_ch_off) return false;

    bool lockedDst0 = false;
    bool lockedDst1 = false;
    bool lockedSrc0 = false;
    bool lockedSrc1 = false;
    bool ok = false;

    if (IOSurfaceLock(dst0, 0, NULL) != kIOReturnSuccess) return false;
    lockedDst0 = true;
    if (dst1 != dst0) {
        if (IOSurfaceLock(dst1, 0, NULL) != kIOReturnSuccess) goto cleanup;
        lockedDst1 = true;
    }
    if (src0 != dst0 && src0 != dst1) {
        if (IOSurfaceLock(src0, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) goto cleanup;
        lockedSrc0 = true;
    }
    if (src1 != dst0 && src1 != dst1 && src1 != src0) {
        if (IOSurfaceLock(src1, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) goto cleanup;
        lockedSrc1 = true;
    }

    void *dst0Base = IOSurfaceGetBaseAddress(dst0);
    void *dst1Base = IOSurfaceGetBaseAddress(dst1);
    const void *src0Base = IOSurfaceGetBaseAddress(src0);
    const void *src1Base = IOSurfaceGetBaseAddress(src1);
    if (!dst0Base || !dst1Base || !src0Base || !src1Base) goto cleanup;

    const size_t dst0SpatialSz = (size_t)dst0_spatial;
    const size_t src0SpatialSz = (size_t)src0_spatial;
    const size_t dst1SpatialSz = (size_t)dst1_spatial;
    const size_t src1SpatialSz = (size_t)src1_spatial;

    size_t dst0MaxIdxElems, src0MaxIdxElems, dst1MaxIdxElems, src1MaxIdxElems;
    size_t dst0Elems, src0Elems, dst1Elems, src1Elems;
    size_t dst0Bytes, src0Bytes, dst1Bytes, src1Bytes;

    if (mul_size_overflow((size_t)(dst0_ch_off + channels - 1), dst0SpatialSz, &dst0MaxIdxElems)) goto cleanup;
    if (mul_size_overflow((size_t)(src0_ch_off + channels - 1), src0SpatialSz, &src0MaxIdxElems)) goto cleanup;
    if (mul_size_overflow((size_t)(dst1_ch_off + channels - 1), dst1SpatialSz, &dst1MaxIdxElems)) goto cleanup;
    if (mul_size_overflow((size_t)(src1_ch_off + channels - 1), src1SpatialSz, &src1MaxIdxElems)) goto cleanup;
    if (add_size_overflow(dst0MaxIdxElems, (size_t)dst0_spatial_index, &dst0MaxIdxElems)) goto cleanup;
    if (add_size_overflow(src0MaxIdxElems, (size_t)src0_spatial_index, &src0MaxIdxElems)) goto cleanup;
    if (add_size_overflow(dst1MaxIdxElems, (size_t)dst1_spatial_index, &dst1MaxIdxElems)) goto cleanup;
    if (add_size_overflow(src1MaxIdxElems, (size_t)src1_spatial_index, &src1MaxIdxElems)) goto cleanup;
    if (add_size_overflow(dst0MaxIdxElems, 1, &dst0Elems)) goto cleanup;
    if (add_size_overflow(src0MaxIdxElems, 1, &src0Elems)) goto cleanup;
    if (add_size_overflow(dst1MaxIdxElems, 1, &dst1Elems)) goto cleanup;
    if (add_size_overflow(src1MaxIdxElems, 1, &src1Elems)) goto cleanup;
    if (mul_size_overflow(dst0Elems, sizeof(_Float16), &dst0Bytes)) goto cleanup;
    if (mul_size_overflow(src0Elems, sizeof(_Float16), &src0Bytes)) goto cleanup;
    if (mul_size_overflow(dst1Elems, sizeof(_Float16), &dst1Bytes)) goto cleanup;
    if (mul_size_overflow(src1Elems, sizeof(_Float16), &src1Bytes)) goto cleanup;

    if (dst0Bytes > IOSurfaceGetAllocSize(dst0) || src0Bytes > IOSurfaceGetAllocSize(src0)) goto cleanup;
    if (dst1Bytes > IOSurfaceGetAllocSize(dst1) || src1Bytes > IOSurfaceGetAllocSize(src1)) goto cleanup;

    _Float16 *dst0F16 = (_Float16 *)dst0Base;
    _Float16 *dst1F16 = (_Float16 *)dst1Base;
    const _Float16 *src0F16 = (const _Float16 *)src0Base;
    const _Float16 *src1F16 = (const _Float16 *)src1Base;

    for (int c = 0; c < channels; c++) {
        size_t dst0Idx = (size_t)(dst0_ch_off + c) * dst0SpatialSz + (size_t)dst0_spatial_index;
        size_t src0Idx = (size_t)(src0_ch_off + c) * src0SpatialSz + (size_t)src0_spatial_index;
        size_t dst1Idx = (size_t)(dst1_ch_off + c) * dst1SpatialSz + (size_t)dst1_spatial_index;
        size_t src1Idx = (size_t)(src1_ch_off + c) * src1SpatialSz + (size_t)src1_spatial_index;
        dst0F16[dst0Idx] = src0F16[src0Idx];
        dst1F16[dst1Idx] = src1F16[src1Idx];
    }

    ok = true;

cleanup:
    if (lockedSrc1) IOSurfaceUnlock(src1, kIOSurfaceLockReadOnly, NULL);
    if (lockedSrc0) IOSurfaceUnlock(src0, kIOSurfaceLockReadOnly, NULL);
    if (lockedDst1) IOSurfaceUnlock(dst1, 0, NULL);
    if (lockedDst0) IOSurfaceUnlock(dst0, 0, NULL);
    return ok;
}

bool ane_interop_io_write_fp16_spatial_slice(IOSurfaceRef surface,
                                             int ch_off,
                                             int spatial_index,
                                             int spatial,
                                             const float *data,
                                             int channels) {
    if (!surface || !data) return false;
    if (ch_off < 0 || spatial_index < 0 || spatial < 0 || channels < 0) return false;
    if (channels == 0) return true;
    if (spatial == 0 || spatial_index >= spatial) return false;
    if (channels > INT_MAX - ch_off) return false;

    size_t spatialSz = (size_t)spatial;
    size_t maxCh = (size_t)(ch_off + channels - 1);
    size_t maxIdxElems;
    size_t elemCount;
    if (mul_size_overflow(maxCh, spatialSz, &maxIdxElems)) return false;
    if (add_size_overflow(maxIdxElems, (size_t)spatial_index, &maxIdxElems)) return false;
    if (add_size_overflow(maxIdxElems, 1, &elemCount)) return false;

    size_t bytes;
    if (mul_size_overflow(elemCount, sizeof(_Float16), &bytes)) return false;

    if (IOSurfaceLock(surface, 0, NULL) != kIOReturnSuccess) return false;
    bool ok = false;

    void *base = IOSurfaceGetBaseAddress(surface);
    if (!base) goto cleanup;
    if (bytes > IOSurfaceGetAllocSize(surface)) goto cleanup;

    _Float16 *dstF16 = (_Float16 *)base;
    const size_t baseIdx = (size_t)ch_off * spatialSz + (size_t)spatial_index;
    // Use NEON-vectorized strided scatter: FP32 -> FP16 with stride = spatial.
    ane_interop_neon_scatter_f32_to_f16(dstF16 + baseIdx, data,
                                        channels, (int)spatialSz);
    ok = true;

cleanup:
    IOSurfaceUnlock(surface, 0, NULL);
    return ok;
}

bool ane_interop_io_read_fp16_spatial_slice(IOSurfaceRef surface,
                                            int ch_off,
                                            int spatial_index,
                                            int spatial,
                                            float *data,
                                            int channels) {
    if (!surface || !data) return false;
    if (ch_off < 0 || spatial_index < 0 || spatial < 0 || channels < 0) return false;
    if (channels == 0) return true;
    if (spatial == 0 || spatial_index >= spatial) return false;
    if (channels > INT_MAX - ch_off) return false;

    size_t spatialSz = (size_t)spatial;
    size_t maxCh = (size_t)(ch_off + channels - 1);
    size_t maxIdxElems;
    size_t elemCount;
    if (mul_size_overflow(maxCh, spatialSz, &maxIdxElems)) return false;
    if (add_size_overflow(maxIdxElems, (size_t)spatial_index, &maxIdxElems)) return false;
    if (add_size_overflow(maxIdxElems, 1, &elemCount)) return false;

    size_t bytes;
    if (mul_size_overflow(elemCount, sizeof(_Float16), &bytes)) return false;

    if (IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return false;
    bool ok = false;

    const void *base = IOSurfaceGetBaseAddress(surface);
    if (!base) goto cleanup;
    if (bytes > IOSurfaceGetAllocSize(surface)) goto cleanup;

    const _Float16 *srcF16 = (const _Float16 *)base;
    const size_t baseIdx = (size_t)ch_off * spatialSz + (size_t)spatial_index;
    // Use NEON-vectorized strided gather: FP16 -> FP32 with stride = spatial.
    ane_interop_neon_gather_f16_to_f32(data, srcF16 + baseIdx,
                                       channels, (int)spatialSz);
    ok = true;

cleanup:
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    return ok;
}

bool ane_interop_io_argmax_fp16_spatial_slice(IOSurfaceRef surface,
                                              int ch_off,
                                              int spatial_index,
                                              int spatial,
                                              int channels,
                                              int *out_index,
                                              float *out_value) {
    if (!surface) return false;
    if (!out_index || !out_value) return false;
    if (ch_off < 0 || spatial_index < 0 || spatial <= 0 || channels <= 0) return false;
    if (spatial_index >= spatial) return false;
    if (channels > INT_MAX - ch_off) return false;

    size_t spatialSz = (size_t)spatial;
    size_t maxCh = (size_t)(ch_off + channels - 1);
    size_t maxIdxElems;
    size_t elemCount;
    if (mul_size_overflow(maxCh, spatialSz, &maxIdxElems)) return false;
    if (add_size_overflow(maxIdxElems, (size_t)spatial_index, &maxIdxElems)) return false;
    if (add_size_overflow(maxIdxElems, 1, &elemCount)) return false;

    size_t bytes;
    if (mul_size_overflow(elemCount, sizeof(_Float16), &bytes)) return false;

    if (IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return false;
    bool ok = false;

    const void *base = IOSurfaceGetBaseAddress(surface);
    if (!base) goto cleanup;
    if (bytes > IOSurfaceGetAllocSize(surface)) goto cleanup;

    const _Float16 *srcF16 = (const _Float16 *)base;
    const size_t baseIdx = (size_t)ch_off * spatialSz + (size_t)spatial_index;
    const size_t stride = spatialSz;

    int bestIndex = 0;
    float bestValueF32 = 0.0f;

    // Delegate to NEON-vectorized strided argmax.
    ane_interop_neon_argmax_f16_strided(srcF16 + baseIdx, channels,
                                        (int)stride, &bestIndex, &bestValueF32);

    *out_index = bestIndex;
    *out_value = bestValueF32;
    ok = true;

cleanup:
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    return ok;
}

bool ane_interop_io_argmax_fp16_spatial_slice_with_hint(
    IOSurfaceRef surface,
    int ch_off,
    int spatial_index,
    int spatial,
    int channels,
    IOSurfaceRef hint_surface,
    int hint_spatial_index,
    int hint_spatial,
    int *out_index,
    float *out_value) {
    if (!surface || !hint_surface) return false;
    if (!out_index || !out_value) return false;
    if (ch_off < 0 || spatial_index < 0 || spatial <= 0 || channels <= 0) return false;
    if (spatial_index >= spatial) return false;
    if (channels > INT_MAX - ch_off) return false;
    if (hint_spatial_index < 0 || hint_spatial <= 0) return false;
    if (hint_spatial_index >= hint_spatial) return false;

    size_t spatialSz = (size_t)spatial;
    size_t maxCh = (size_t)(ch_off + channels - 1);
    size_t maxIdxElems;
    size_t elemCount;
    if (mul_size_overflow(maxCh, spatialSz, &maxIdxElems)) return false;
    if (add_size_overflow(maxIdxElems, (size_t)spatial_index, &maxIdxElems)) return false;
    if (add_size_overflow(maxIdxElems, 1, &elemCount)) return false;
    size_t logitsBytes;
    if (mul_size_overflow(elemCount, sizeof(_Float16), &logitsBytes)) return false;

    size_t hintBytes;
    if (mul_size_overflow((size_t)hint_spatial, sizeof(_Float16), &hintBytes)) return false;

    bool lockedHint = false;
    bool lockedLogits = false;
    bool ok = false;

    /* Read the max hint value from the reduce_max output surface. */
    if (IOSurfaceLock(hint_surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return false;
    lockedHint = true;

    const void *hintBase = IOSurfaceGetBaseAddress(hint_surface);
    if (!hintBase) goto cleanup_h;
    if (hintBytes > IOSurfaceGetAllocSize(hint_surface)) goto cleanup_h;

    _Float16 maxHint = ((const _Float16 *)hintBase)[(size_t)hint_spatial_index];

    IOSurfaceUnlock(hint_surface, kIOSurfaceLockReadOnly, NULL);
    lockedHint = false;

    /* Scan the logits surface with early exit on first match. */
    if (IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) goto cleanup_h;
    lockedLogits = true;

    {
        const void *base = IOSurfaceGetBaseAddress(surface);
        if (!base) goto cleanup_h;
        if (logitsBytes > IOSurfaceGetAllocSize(surface)) goto cleanup_h;

        const _Float16 *srcF16 = (const _Float16 *)base;
        const size_t baseIdx = (size_t)ch_off * spatialSz + (size_t)spatial_index;
        const size_t stride = spatialSz;

        for (int c = 0; c < channels; c++) {
            size_t idx = baseIdx + (size_t)c * stride;
            if (srcF16[idx] == maxHint) {
                *out_index = c;
                *out_value = (float)maxHint;
                ok = true;
                goto cleanup_h;
            }
        }

        /* Fallback: full scan if no exact fp16 match (should not happen). */
        int bestIndex = 0;
        _Float16 bestValue = srcF16[baseIdx];
        for (int c = 1; c < channels; c++) {
            size_t idx = baseIdx + (size_t)c * stride;
            if (srcF16[idx] > bestValue) {
                bestValue = srcF16[idx];
                bestIndex = c;
            }
        }
        *out_index = bestIndex;
        *out_value = (float)bestValue;
        ok = true;
    }

cleanup_h:
    if (lockedLogits) IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    if (lockedHint) IOSurfaceUnlock(hint_surface, kIOSurfaceLockReadOnly, NULL);
    return ok;
}

bool ane_interop_io_write_embedding_batch_fp16(
    IOSurfaceRef surface,
    int ch_off,
    int spatial,
    const float *embedding_table,
    int vocab_size,
    int dim,
    const uint16_t *token_ids,
    int stream_count) {

    if (!surface || !embedding_table || !token_ids) return false;
    if (ch_off < 0 || spatial <= 0 || vocab_size <= 0 || dim <= 0 || stream_count <= 0) return false;
    if (stream_count > spatial) return false;
    if (dim > INT_MAX - ch_off) return false;

    size_t spatialSz = (size_t)spatial;
    size_t maxCh = (size_t)(ch_off + dim - 1);
    size_t maxIdxElems;
    size_t elemCount;
    if (mul_size_overflow(maxCh, spatialSz, &maxIdxElems)) return false;
    if (add_size_overflow(maxIdxElems, (size_t)(spatial - 1), &maxIdxElems)) return false;
    if (add_size_overflow(maxIdxElems, 1, &elemCount)) return false;

    size_t bytes;
    if (mul_size_overflow(elemCount, sizeof(_Float16), &bytes)) return false;

    if (IOSurfaceLock(surface, 0, NULL) != kIOReturnSuccess) return false;
    bool ok = false;

    void *base = IOSurfaceGetBaseAddress(surface);
    if (!base) goto cleanup;
    if (bytes > IOSurfaceGetAllocSize(surface)) goto cleanup;

    {
        _Float16 *dstF16 = (_Float16 *)base;

#if defined(__aarch64__) || defined(__arm64__)
        /*
         * NEON channel-major embedding write.
         * For each channel c, gather embedding values from all streams and
         * convert float32→fp16 in 8-wide NEON vectors, writing a contiguous
         * row per channel. Much better cache behavior than per-stream scattered writes.
         */
        if (stream_count > 0 && (stream_count % 8) == 0 && stream_count <= 32768) {
            /* Pre-compute row pointers for each stream (avoid repeated multiply in inner loop) */
            const float *embRows[32768];
            for (int s = 0; s < stream_count; s++) {
                int token = (int)token_ids[s];
                if (token < 0 || token >= vocab_size) goto cleanup;
                embRows[s] = embedding_table + (size_t)token * (size_t)dim;
            }

            for (int c = 0; c < dim; c++) {
                _Float16 *dstRow = dstF16 + (size_t)(ch_off + c) * spatialSz;
                for (int s = 0; s < stream_count; s += 8) {
                    float32x4_t lo = {
                        embRows[s+0][c], embRows[s+1][c],
                        embRows[s+2][c], embRows[s+3][c]
                    };
                    float32x4_t hi = {
                        embRows[s+4][c], embRows[s+5][c],
                        embRows[s+6][c], embRows[s+7][c]
                    };
                    float16x4_t lo16 = vcvt_f16_f32(lo);
                    float16x4_t hi16 = vcvt_f16_f32(hi);
                    float16x8_t combined = vcombine_f16(lo16, hi16);
                    vst1q_f16(dstRow + s, combined);
                }
            }
            ok = true;
            goto cleanup;
        }
#endif

        /* Scalar fallback */
        for (int s = 0; s < stream_count; s++) {
            int token = (int)token_ids[s];
            if (token < 0 || token >= vocab_size) goto cleanup;
            const float *embRow = embedding_table + (size_t)token * (size_t)dim;
            for (int c = 0; c < dim; c++) {
                size_t idx = (size_t)(ch_off + c) * spatialSz + (size_t)s;
                dstF16[idx] = (_Float16)embRow[c];
            }
        }
    }
    ok = true;

cleanup:
    IOSurfaceUnlock(surface, 0, NULL);
    return ok;
}

bool ane_interop_io_argmax_batch_fp16_spatial(
    IOSurfaceRef surface,
    int ch_off,
    int spatial,
    int channels,
    int stream_count,
    int *out_indices,
    float *out_values) {

    if (!surface || !out_indices || !out_values) return false;
    if (ch_off < 0 || spatial <= 0 || channels <= 0 || stream_count <= 0) return false;
    if (stream_count > spatial) return false;
    if (channels > 65535) return false;  /* uint16_t index overflow guard */
    if (channels > INT_MAX - ch_off) return false;

    size_t spatialSz = (size_t)spatial;
    size_t maxCh = (size_t)(ch_off + channels - 1);
    size_t maxIdxElems;
    size_t elemCount;
    if (mul_size_overflow(maxCh, spatialSz, &maxIdxElems)) return false;
    if (add_size_overflow(maxIdxElems, (size_t)(spatial - 1), &maxIdxElems)) return false;
    if (add_size_overflow(maxIdxElems, 1, &elemCount)) return false;

    size_t bytes;
    if (mul_size_overflow(elemCount, sizeof(_Float16), &bytes)) return false;

    if (IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return false;
    bool ok = false;

    const void *base = IOSurfaceGetBaseAddress(surface);
    if (!base) goto cleanup;
    if (bytes > IOSurfaceGetAllocSize(surface)) goto cleanup;

    {
        const _Float16 *srcF16 = (const _Float16 *)base;

#if defined(__aarch64__) || defined(__arm64__)
        /*
         * NEON-vectorized channel-major argmax.
         * Process spatial lanes as N × float16x8_t per channel row.
         * Branchless: vcgtq_f16 + vbslq for masked conditional update.
         * Indices tracked as uint16_t (fits vocab ≤ 65535).
         */
        if (spatial > 0 && spatial <= 512 && (spatial % 8) == 0) {
            const int nvecs = spatial / 8;
            float16x8_t bestV[ARGMAX_MAX_NVECS];
            uint16x8_t bestI[ARGMAX_MAX_NVECS];

            const _Float16 *row0 = srcF16 + (size_t)ch_off * spatialSz;
            for (int v = 0; v < nvecs; v++) {
                bestV[v] = vld1q_f16(row0 + v * 8);
                bestI[v] = vdupq_n_u16(0);
            }

            for (int c = 1; c < channels; c++) {
                const _Float16 *row = srcF16 + (size_t)(ch_off + c) * spatialSz;
                uint16x8_t cidx = vdupq_n_u16((uint16_t)c);
                for (int v = 0; v < nvecs; v++) {
                    float16x8_t vals = vld1q_f16(row + v * 8);
                    uint16x8_t gt = vcgtq_f16(vals, bestV[v]);
                    bestV[v] = vbslq_f16(gt, vals, bestV[v]);
                    bestI[v] = vbslq_u16(gt, cidx, bestI[v]);
                }
            }

            _Float16 bvBuf[512];
            uint16_t biBuf[512];
            for (int v = 0; v < nvecs; v++) {
                vst1q_f16(bvBuf + v * 8, bestV[v]);
                vst1q_u16(biBuf + v * 8, bestI[v]);
            }
            for (int s = 0; s < stream_count; s++) {
                out_indices[s] = (int)biBuf[s];
                out_values[s] = (float)bvBuf[s];
            }

            ok = true;
            goto cleanup;
        }
#endif
        /* Scalar fallback for non-32 spatial */
        for (int s = 0; s < stream_count; s++) {
            const size_t baseIdx = (size_t)ch_off * spatialSz + (size_t)s;
            const size_t stride = spatialSz;
            int bestIndex = 0;
            _Float16 bestValue = srcF16[baseIdx];

            for (int c = 1; c < channels; c++) {
                _Float16 value = srcF16[baseIdx + (size_t)c * stride];
                if (value > bestValue) { bestValue = value; bestIndex = c; }
            }

            out_indices[s] = bestIndex;
            out_values[s] = (float)bestValue;
        }
    }
    ok = true;

cleanup:
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    return ok;
}

bool ane_interop_io_write_fp16(IOSurfaceRef surface,
                               const float *data, int channels, int spatial) {
    return ane_interop_io_write_fp16_at(surface, 0, data, channels, spatial);
}

bool ane_interop_io_read_fp16(IOSurfaceRef surface, int ch_off,
                              float *data, int channels, int spatial) {
    if (!surface) return false;
    if (ch_off < 0 || channels < 0 || spatial < 0) return false;
    if (channels == 0 || spatial == 0) return true;
    if (!data) return false;
    if (channels > INT_MAX / spatial) return false;

    size_t offElems, elemCount;
    size_t offBytes, bytes, endBytes;
    size_t spatialSz = (size_t)spatial;
    if (mul_size_overflow((size_t)ch_off, spatialSz, &offElems)) return false;
    if (mul_size_overflow((size_t)channels, spatialSz, &elemCount)) return false;
    if (mul_size_overflow(offElems, sizeof(_Float16), &offBytes)) return false;
    if (mul_size_overflow(elemCount, sizeof(_Float16), &bytes)) return false;
    if (add_size_overflow(offBytes, bytes, &endBytes)) return false;

    if (IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return false;
    bool ok = false;

    const void *base = IOSurfaceGetBaseAddress(surface);
    if (!base) goto cleanup;

    size_t allocSize = IOSurfaceGetAllocSize(surface);
    if (endBytes > allocSize) goto cleanup;

    ane_interop_cvt_f16_to_f32(data, ((const _Float16 *)base) + offElems, channels * spatial);
    ok = true;

cleanup:
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    return ok;
}

bool ane_interop_io_read_fp16_batched(IOSurfaceRef surface, int spatial,
                                      float * const *destinations,
                                      const int *channel_offsets,
                                      const int *channels,
                                      int region_count) {
    if (!surface) return false;
    if (spatial < 0 || region_count < 0) return false;
    if (region_count == 0 || spatial == 0) return true;
    if (!destinations || !channel_offsets || !channels) return false;

    if (IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return false;
    bool ok = false;

    const void *base = IOSurfaceGetBaseAddress(surface);
    if (!base) goto cleanup;

    size_t allocSize = IOSurfaceGetAllocSize(surface);
    size_t spatialSz = (size_t)spatial;

    for (int i = 0; i < region_count; i++) {
        int chOff = channel_offsets[i];
        int chCount = channels[i];
        float *dst = destinations[i];

        if (chOff < 0 || chCount < 0) goto cleanup;
        if (chCount == 0) continue;
        if (!dst) goto cleanup;
        if (chCount > INT_MAX / spatial) goto cleanup;

        size_t offElems, elemCount;
        size_t offBytes, bytes, endBytes;
        if (mul_size_overflow((size_t)chOff, spatialSz, &offElems)) goto cleanup;
        if (mul_size_overflow((size_t)chCount, spatialSz, &elemCount)) goto cleanup;
        if (mul_size_overflow(offElems, sizeof(_Float16), &offBytes)) goto cleanup;
        if (mul_size_overflow(elemCount, sizeof(_Float16), &bytes)) goto cleanup;
        if (add_size_overflow(offBytes, bytes, &endBytes)) goto cleanup;
        if (endBytes > allocSize) goto cleanup;

        ane_interop_cvt_f16_to_f32(dst, ((const _Float16 *)base) + offElems, chCount * spatial);
    }

    ok = true;

cleanup:
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    return ok;
}

bool ane_interop_io_write_fp16_at(IOSurfaceRef surface, int ch_off,
                                  const float *data, int channels, int spatial) {
    if (!surface) return false;
    if (ch_off < 0 || channels < 0 || spatial < 0) return false;
    if (channels == 0 || spatial == 0) return true;
    if (!data) return false;
    if (channels > INT_MAX / spatial) return false;

    size_t offElems, elemCount;
    size_t offBytes, bytes, endBytes;
    size_t spatialSz = (size_t)spatial;
    if (mul_size_overflow((size_t)ch_off, spatialSz, &offElems)) return false;
    if (mul_size_overflow((size_t)channels, spatialSz, &elemCount)) return false;
    if (mul_size_overflow(offElems, sizeof(_Float16), &offBytes)) return false;
    if (mul_size_overflow(elemCount, sizeof(_Float16), &bytes)) return false;
    if (add_size_overflow(offBytes, bytes, &endBytes)) return false;

    if (IOSurfaceLock(surface, 0, NULL) != kIOReturnSuccess) return false;
    bool ok = false;

    void *base = IOSurfaceGetBaseAddress(surface);
    if (!base) goto cleanup;

    size_t allocSize = IOSurfaceGetAllocSize(surface);
    if (endBytes > allocSize) goto cleanup;

    ane_interop_cvt_f32_to_f16(((_Float16 *)base) + offElems, data, channels * spatial);
    ok = true;

cleanup:
    IOSurfaceUnlock(surface, 0, NULL);
    return ok;
}

bool ane_interop_io_write_fp16_at_batched(IOSurfaceRef surface,
                                          const int *channel_offsets,
                                          const float * const *sources,
                                          const int *channels,
                                          int region_count,
                                          int spatial) {
    if (!surface) return false;
    if (region_count < 0 || spatial < 0) return false;
    if (region_count == 0 || spatial == 0) return true;
    if (!channel_offsets || !sources || !channels) return false;

    if (IOSurfaceLock(surface, 0, NULL) != kIOReturnSuccess) return false;
    bool ok = false;

    void *base = IOSurfaceGetBaseAddress(surface);
    if (!base) goto cleanup;
    size_t allocSize = IOSurfaceGetAllocSize(surface);
    size_t spatialSz = (size_t)spatial;

    for (int i = 0; i < region_count; i++) {
        int chOff = channel_offsets[i];
        int chCount = channels[i];
        const float *src = sources[i];

        if (chOff < 0 || chCount < 0) goto cleanup;
        if (chCount == 0) continue;
        if (!src) goto cleanup;
        if (chCount > INT_MAX / spatial) goto cleanup;

        size_t offElems, elemCount;
        size_t offBytes, bytes, endBytes;
        if (mul_size_overflow((size_t)chOff, spatialSz, &offElems)) goto cleanup;
        if (mul_size_overflow((size_t)chCount, spatialSz, &elemCount)) goto cleanup;
        if (mul_size_overflow(offElems, sizeof(_Float16), &offBytes)) goto cleanup;
        if (mul_size_overflow(elemCount, sizeof(_Float16), &bytes)) goto cleanup;
        if (add_size_overflow(offBytes, bytes, &endBytes)) goto cleanup;
        if (endBytes > allocSize) goto cleanup;

        ane_interop_cvt_f32_to_f16(((_Float16 *)base) + offElems, src, chCount * spatial);
    }

    ok = true;

cleanup:
    IOSurfaceUnlock(surface, 0, NULL);
    return ok;
}

bool ane_interop_io_copy_batched(IOSurfaceRef dst,
                                 IOSurfaceRef src,
                                 const int *dst_channel_offsets,
                                 const int *src_channel_offsets,
                                 const int *channels,
                                 int region_count,
                                 int spatial) {
    if (!dst || !src) return false;
    if (region_count < 0 || spatial < 0) return false;
    if (region_count == 0 || spatial == 0) return true;
    if (!dst_channel_offsets || !src_channel_offsets || !channels) return false;

    bool lockedDst = false;
    bool lockedSrc = false;
    bool ok = false;

    if (dst == src) {
        if (IOSurfaceLock(dst, 0, NULL) != kIOReturnSuccess) return false;
        lockedDst = true;
    } else {
        if (IOSurfaceLock(dst, 0, NULL) != kIOReturnSuccess) return false;
        lockedDst = true;
        if (IOSurfaceLock(src, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) goto cleanup;
        lockedSrc = true;
    }

    void *dstBase = IOSurfaceGetBaseAddress(dst);
    const void *srcBase = IOSurfaceGetBaseAddress(src);
    if (!dstBase || !srcBase) goto cleanup;

    size_t dstSize = IOSurfaceGetAllocSize(dst);
    size_t srcSize = IOSurfaceGetAllocSize(src);
    size_t spatialSz = (size_t)spatial;

    for (int i = 0; i < region_count; i++) {
        int dstOff = dst_channel_offsets[i];
        int srcOff = src_channel_offsets[i];
        int chCount = channels[i];
        if (dstOff < 0 || srcOff < 0 || chCount < 0) goto cleanup;
        if (chCount == 0) continue;
        if (chCount > INT_MAX / spatial) goto cleanup;

        size_t dstOffElems, srcOffElems, elemCount;
        size_t dstOffBytes, srcOffBytes, bytes;
        if (mul_size_overflow((size_t)dstOff, spatialSz, &dstOffElems)) goto cleanup;
        if (mul_size_overflow((size_t)srcOff, spatialSz, &srcOffElems)) goto cleanup;
        if (mul_size_overflow((size_t)chCount, spatialSz, &elemCount)) goto cleanup;
        if (mul_size_overflow(dstOffElems, sizeof(_Float16), &dstOffBytes)) goto cleanup;
        if (mul_size_overflow(srcOffElems, sizeof(_Float16), &srcOffBytes)) goto cleanup;
        if (mul_size_overflow(elemCount, sizeof(_Float16), &bytes)) goto cleanup;

        if (dstOffBytes > dstSize || srcOffBytes > srcSize) goto cleanup;
        if (bytes > dstSize - dstOffBytes || bytes > srcSize - srcOffBytes) goto cleanup;

        memmove(((_Float16 *)dstBase) + dstOffElems, ((const _Float16 *)srcBase) + srcOffElems, bytes);
    }

    ok = true;

cleanup:
    if (lockedSrc) IOSurfaceUnlock(src, kIOSurfaceLockReadOnly, NULL);
    if (lockedDst) IOSurfaceUnlock(dst, 0, NULL);
    return ok;
}

// MARK: - Lock/Unlock primitives

bool ane_interop_io_lock_write(IOSurfaceRef surface) {
    if (!surface) return false;
    return IOSurfaceLock(surface, 0, NULL) == kIOReturnSuccess;
}

bool ane_interop_io_unlock_write(IOSurfaceRef surface) {
    if (!surface) return false;
    return IOSurfaceUnlock(surface, 0, NULL) == kIOReturnSuccess;
}

bool ane_interop_io_lock_read(IOSurfaceRef surface) {
    if (!surface) return false;
    return IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) == kIOReturnSuccess;
}

bool ane_interop_io_unlock_read(IOSurfaceRef surface) {
    if (!surface) return false;
    return IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL) == kIOReturnSuccess;
}

// MARK: - Unlocked I/O (caller must hold appropriate lock)

bool ane_interop_io_write_fp16_unlocked(IOSurfaceRef surface,
                                         const float *data, int channels, int spatial) {
    if (!surface) return false;
    if (channels < 0 || spatial < 0) return false;
    if (channels == 0 || spatial == 0) return true;
    if (!data) return false;
    if (channels > INT_MAX / spatial) return false;

    size_t elemCount;
    size_t bytes;
    if (mul_size_overflow((size_t)channels, (size_t)spatial, &elemCount)) return false;
    if (mul_size_overflow(elemCount, sizeof(_Float16), &bytes)) return false;

    void *base = IOSurfaceGetBaseAddress(surface);
    if (!base) return false;

    size_t allocSize = IOSurfaceGetAllocSize(surface);
    if (bytes > allocSize) return false;

    ane_interop_cvt_f32_to_f16((_Float16 *)base, data, channels * spatial);
    return true;
}

bool ane_interop_io_read_fp16_unlocked(IOSurfaceRef surface, int ch_off,
                                        float *data, int channels, int spatial) {
    if (!surface) return false;
    if (ch_off < 0 || channels < 0 || spatial < 0) return false;
    if (channels == 0 || spatial == 0) return true;
    if (!data) return false;
    if (channels > INT_MAX / spatial) return false;

    size_t offElems, elemCount;
    size_t offBytes, bytes, endBytes;
    size_t spatialSz = (size_t)spatial;
    if (mul_size_overflow((size_t)ch_off, spatialSz, &offElems)) return false;
    if (mul_size_overflow((size_t)channels, spatialSz, &elemCount)) return false;
    if (mul_size_overflow(offElems, sizeof(_Float16), &offBytes)) return false;
    if (mul_size_overflow(elemCount, sizeof(_Float16), &bytes)) return false;
    if (add_size_overflow(offBytes, bytes, &endBytes)) return false;

    const void *base = IOSurfaceGetBaseAddress(surface);
    if (!base) return false;

    size_t allocSize = IOSurfaceGetAllocSize(surface);
    if (endBytes > allocSize) return false;

    ane_interop_cvt_f16_to_f32(data, ((const _Float16 *)base) + offElems, channels * spatial);
    return true;
}

bool ane_interop_io_copy_multi_src(IOSurfaceRef dst,
                                   IOSurfaceRef const *sources,
                                   const int *dst_channel_offsets,
                                   const int *src_channel_offsets,
                                   const int *channels,
                                   int region_count,
                                   int spatial) {
    if (!dst) return false;
    if (region_count < 0 || spatial < 0) return false;
    if (region_count == 0 || spatial == 0) return true;
    if (!sources || !dst_channel_offsets || !src_channel_offsets || !channels) return false;

    if (IOSurfaceLock(dst, 0, NULL) != kIOReturnSuccess) return false;
    bool ok = false;
    IOSurfaceRef *lockedSources = NULL;
    int lockedCount = 0;

    lockedSources = (IOSurfaceRef *)calloc((size_t)region_count, sizeof(IOSurfaceRef));
    if (!lockedSources) goto cleanup;

    for (int i = 0; i < region_count; i++) {
        IOSurfaceRef src = sources[i];
        if (!src) goto cleanup;
        if (src == dst) continue;

        bool alreadyLocked = false;
        for (int j = 0; j < lockedCount; j++) {
            if (lockedSources[j] == src) {
                alreadyLocked = true;
                break;
            }
        }
        if (alreadyLocked) continue;

        if (IOSurfaceLock(src, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) goto cleanup;
        lockedSources[lockedCount++] = src;
    }

    void *dstBase = IOSurfaceGetBaseAddress(dst);
    if (!dstBase) goto cleanup;
    size_t dstSize = IOSurfaceGetAllocSize(dst);
    size_t spatialSz = (size_t)spatial;

    for (int i = 0; i < region_count; i++) {
        IOSurfaceRef src = sources[i];
        int dstOff = dst_channel_offsets[i];
        int srcOff = src_channel_offsets[i];
        int chCount = channels[i];
        if (dstOff < 0 || srcOff < 0 || chCount < 0) goto cleanup;
        if (chCount == 0) continue;
        if (chCount > INT_MAX / spatial) goto cleanup;

        const void *srcBase = IOSurfaceGetBaseAddress(src);
        if (!srcBase) goto cleanup;
        size_t srcSize = IOSurfaceGetAllocSize(src);

        size_t dstOffElems, srcOffElems, elemCount;
        size_t dstOffBytes, srcOffBytes, bytes;
        if (mul_size_overflow((size_t)dstOff, spatialSz, &dstOffElems)) goto cleanup;
        if (mul_size_overflow((size_t)srcOff, spatialSz, &srcOffElems)) goto cleanup;
        if (mul_size_overflow((size_t)chCount, spatialSz, &elemCount)) goto cleanup;
        if (mul_size_overflow(dstOffElems, sizeof(_Float16), &dstOffBytes)) goto cleanup;
        if (mul_size_overflow(srcOffElems, sizeof(_Float16), &srcOffBytes)) goto cleanup;
        if (mul_size_overflow(elemCount, sizeof(_Float16), &bytes)) goto cleanup;

        if (dstOffBytes > dstSize || srcOffBytes > srcSize) goto cleanup;
        if (bytes > dstSize - dstOffBytes || bytes > srcSize - srcOffBytes) goto cleanup;

        memmove(((_Float16 *)dstBase) + dstOffElems, ((const _Float16 *)srcBase) + srcOffElems, bytes);
    }

    ok = true;

cleanup:
    if (lockedSources) {
        for (int i = 0; i < lockedCount; i++) {
            IOSurfaceUnlock(lockedSources[i], kIOSurfaceLockReadOnly, NULL);
        }
        free(lockedSources);
    }
    IOSurfaceUnlock(dst, 0, NULL);
    return ok;
}

/* ---------- Channel-partitioned parallel argmax ---------- */

#if defined(__aarch64__) || defined(__arm64__)

typedef struct {
    const _Float16 *base;
    int spatial;
    int ch_start;
    int ch_end;
    _Float16 *partial_values;   /* [spatial] */
    uint16_t *partial_indices;  /* [spatial] */
    bool ok;
} argmax_block_ctx;

static bool neon_partial_argmax(void *ctx_ptr) {
    argmax_block_ctx *ctx = (argmax_block_ctx *)ctx_ptr;
    const _Float16 *base = ctx->base;
    const int spatial = ctx->spatial;
    const int ch_start = ctx->ch_start;
    const int ch_end = ctx->ch_end;
    const int nvecs = spatial / 8;
    const size_t spatialSz = (size_t)spatial;
    float16x8_t stackBestV[ARGMAX_MAX_NVECS];
    uint16x8_t stackBestI[ARGMAX_MAX_NVECS];
    float16x8_t *bestV = stackBestV;
    uint16x8_t *bestI = stackBestI;
    bool usesHeap = false;

    if (nvecs <= 0 || ch_start >= ch_end) return false;

    if (nvecs > ARGMAX_MAX_NVECS) {
        bestV = (float16x8_t *)malloc((size_t)nvecs * sizeof(*bestV));
        bestI = (uint16x8_t *)malloc((size_t)nvecs * sizeof(*bestI));
        if (!bestV || !bestI) {
            free(bestV);
            free(bestI);
            return false;
        }
        usesHeap = true;
    }

    const _Float16 *row0 = base + (size_t)ch_start * spatialSz;
    for (int v = 0; v < nvecs; v++) {
        bestV[v] = vld1q_f16(row0 + v * 8);
        bestI[v] = vdupq_n_u16((uint16_t)ch_start);
    }

    for (int c = ch_start + 1; c < ch_end; c++) {
        const _Float16 *row = base + (size_t)c * spatialSz;
        uint16x8_t cidx = vdupq_n_u16((uint16_t)c);
        for (int v = 0; v < nvecs; v++) {
            float16x8_t vals = vld1q_f16(row + v * 8);
            uint16x8_t gt = vcgtq_f16(vals, bestV[v]);
            bestV[v] = vbslq_f16(gt, vals, bestV[v]);
            bestI[v] = vbslq_u16(gt, cidx, bestI[v]);
        }
    }

    for (int v = 0; v < nvecs; v++) {
        vst1q_f16(ctx->partial_values + v * 8, bestV[v]);
        vst1q_u16(ctx->partial_indices + v * 8, bestI[v]);
    }
    if (usesHeap) {
        free(bestV);
        free(bestI);
    }
    return true;
}

#endif /* __aarch64__ */

bool ane_interop_io_argmax_batch_fp16_spatial_parallel(
    IOSurfaceRef surface,
    int ch_off,
    int spatial,
    int channels,
    int stream_count,
    int *out_indices,
    float *out_values,
    int n_blocks) {

    /* Fall back to serial for small workloads or invalid n_blocks */
    if (n_blocks <= 1 || channels < n_blocks * 2) {
        return ane_interop_io_argmax_batch_fp16_spatial(
            surface, ch_off, spatial, channels, stream_count,
            out_indices, out_values);
    }

#if defined(__aarch64__) || defined(__arm64__)
    if (!surface || !out_indices || !out_values) return false;
    if (ch_off < 0 || spatial <= 0 || channels <= 0 || stream_count <= 0) return false;
    if (stream_count > spatial) return false;
    if (channels > 65535) return false;  /* uint16_t index overflow guard */
    if (spatial > 32768 || (spatial % 8) != 0) {
        return ane_interop_io_argmax_batch_fp16_spatial(
            surface, ch_off, spatial, channels, stream_count,
            out_indices, out_values);
    }
    if (n_blocks > 32) n_blocks = 32;

    size_t spatialSz = (size_t)spatial;
    size_t maxCh = (size_t)(ch_off + channels - 1);
    size_t maxIdxElems;
    if (mul_size_overflow(maxCh, spatialSz, &maxIdxElems)) return false;
    size_t elemCount;
    if (add_size_overflow(maxIdxElems, (size_t)(spatial - 1), &maxIdxElems)) return false;
    if (add_size_overflow(maxIdxElems, 1, &elemCount)) return false;
    size_t bytes;
    if (mul_size_overflow(elemCount, sizeof(_Float16), &bytes)) return false;

    if (IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) return false;
    bool ok = false;

    const void *base = IOSurfaceGetBaseAddress(surface);
    if (!base) goto cleanup;
    if (bytes > IOSurfaceGetAllocSize(surface)) goto cleanup;

    {
        const _Float16 *srcF16 = (const _Float16 *)base;
        const _Float16 *src_offset = srcF16 + (size_t)ch_off * spatialSz;

        _Float16 *all_values = (_Float16 *)malloc((size_t)n_blocks * spatialSz * sizeof(_Float16));
        uint16_t *all_indices = (uint16_t *)malloc((size_t)n_blocks * spatialSz * sizeof(uint16_t));
        argmax_block_ctx *ctxs = (argmax_block_ctx *)malloc((size_t)n_blocks * sizeof(argmax_block_ctx));
        if (!all_values || !all_indices || !ctxs) {
            free(all_values); free(all_indices); free(ctxs);
            goto cleanup;
        }

        /* Set up blocks: partition channels evenly */
        int chPerBlock = channels / n_blocks;
        for (int b = 0; b < n_blocks; b++) {
            ctxs[b].base = src_offset;
            ctxs[b].spatial = spatial;
            ctxs[b].ch_start = b * chPerBlock;
            ctxs[b].ch_end = (b == n_blocks - 1) ? channels : (b + 1) * chPerBlock;
            ctxs[b].partial_values = all_values + b * spatial;
            ctxs[b].partial_indices = all_indices + b * spatial;
            ctxs[b].ok = false;
        }

        /* Dispatch parallel partial argmax */
        dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
        dispatch_apply((size_t)n_blocks, queue, ^(size_t block_idx) {
            ctxs[block_idx].ok = neon_partial_argmax(&ctxs[block_idx]);
        });

        for (int b = 0; b < n_blocks; b++) {
            if (!ctxs[b].ok) {
                free(all_values);
                free(all_indices);
                free(ctxs);
                goto cleanup;
            }
        }

        /* Merge: for each lane, find global best across all blocks */
        for (int s = 0; s < stream_count; s++) {
            _Float16 bestVal = all_values[s];
            uint16_t bestIdx = all_indices[s];
            for (int b = 1; b < n_blocks; b++) {
                _Float16 v = all_values[b * spatial + s];
                uint16_t i = all_indices[b * spatial + s];
                if (v > bestVal) {
                    bestVal = v;
                    bestIdx = i;
                }
            }
            out_indices[s] = (int)bestIdx;
            out_values[s] = (float)bestVal;
        }

        free(all_values);
        free(all_indices);
        free(ctxs);
        ok = true;
    }

cleanup:
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    return ok;

#else
    return ane_interop_io_argmax_batch_fp16_spatial(
        surface, ch_off, spatial, channels, stream_count,
        out_indices, out_values);
#endif
}

bool ane_interop_io_argmax_batch_fp16_spatial_nolock(
    IOSurfaceRef surface,
    int ch_off,
    int spatial,
    int channels,
    int stream_count,
    int *out_indices,
    float *out_values,
    int n_blocks) {

#if defined(__aarch64__) || defined(__arm64__)
    if (!surface || !out_indices || !out_values) return false;
    if (ch_off < 0 || spatial <= 0 || channels <= 0 || stream_count <= 0) return false;
    if (stream_count > spatial) return false;
    if (channels > INT_MAX - ch_off) return false;
    if (channels > (int)UINT16_MAX + 1) return false;  /* uint16_t index overflow guard */
    if (n_blocks <= 1) n_blocks = 1;
    if (n_blocks > 32) n_blocks = 32;
    if (channels < n_blocks * 2) n_blocks = 1;
    if (spatial > 32768 || (spatial % 8) != 0) {
        return ane_interop_io_argmax_batch_fp16_spatial(
            surface, ch_off, spatial, channels, stream_count,
            out_indices, out_values);
    }

    /* No lock — caller guarantees coherency */
    const void *base = IOSurfaceGetBaseAddress(surface);
    if (!base) return false;
    size_t spatialSz = (size_t)spatial;
    size_t maxCh = (size_t)(ch_off + channels - 1);
    size_t maxIdxElems;
    if (mul_size_overflow(maxCh, spatialSz, &maxIdxElems)) return false;
    size_t elemCount;
    if (add_size_overflow(maxIdxElems, (size_t)(spatial - 1), &maxIdxElems)) return false;
    if (add_size_overflow(maxIdxElems, 1, &elemCount)) return false;
    size_t bytes;
    if (mul_size_overflow(elemCount, sizeof(_Float16), &bytes)) return false;
    if (bytes > IOSurfaceGetAllocSize(surface)) return false;

    const _Float16 *srcF16 = (const _Float16 *)base;
    const _Float16 *src_offset = srcF16 + (size_t)ch_off * spatialSz;

    if (n_blocks <= 1) {
        /* Serial path */
        _Float16 *values = (_Float16 *)malloc(spatialSz * sizeof(_Float16));
        uint16_t *indices = (uint16_t *)malloc(spatialSz * sizeof(uint16_t));
        if (!values || !indices) {
            free(values);
            free(indices);
            return false;
        }
        argmax_block_ctx ctx = {
            .base = src_offset,
            .spatial = spatial,
            .ch_start = 0,
            .ch_end = channels,
            .partial_values = values,
            .partial_indices = indices,
            .ok = false,
        };
        if (!neon_partial_argmax(&ctx)) {
            free(values);
            free(indices);
            return false;
        }
        for (int s = 0; s < stream_count; s++) {
            out_indices[s] = (int)indices[s];
            out_values[s] = (float)values[s];
        }
        free(values);
        free(indices);
        return true;
    }

    /* Parallel path — heap allocate for large spatial */
    _Float16 *all_values = (_Float16 *)malloc((size_t)n_blocks * (size_t)spatial * sizeof(_Float16));
    uint16_t *all_indices = (uint16_t *)malloc((size_t)n_blocks * (size_t)spatial * sizeof(uint16_t));
    argmax_block_ctx *ctxs = (argmax_block_ctx *)malloc((size_t)n_blocks * sizeof(argmax_block_ctx));
    if (!all_values || !all_indices || !ctxs) {
        free(all_values); free(all_indices); free(ctxs);
        return false;
    }

    int chPerBlock = channels / n_blocks;
    for (int b = 0; b < n_blocks; b++) {
        ctxs[b].base = src_offset;
        ctxs[b].spatial = spatial;
        ctxs[b].ch_start = b * chPerBlock;
        ctxs[b].ch_end = (b == n_blocks - 1) ? channels : (b + 1) * chPerBlock;
        ctxs[b].partial_values = all_values + b * spatial;
        ctxs[b].partial_indices = all_indices + b * spatial;
        ctxs[b].ok = false;
    }

    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
    dispatch_apply((size_t)n_blocks, queue, ^(size_t block_idx) {
        ctxs[block_idx].ok = neon_partial_argmax(&ctxs[block_idx]);
    });

    for (int b = 0; b < n_blocks; b++) {
        if (!ctxs[b].ok) {
            free(all_values);
            free(all_indices);
            free(ctxs);
            return false;
        }
    }

    for (int s = 0; s < stream_count; s++) {
        _Float16 bestVal = all_values[s];
        uint16_t bestIdx = all_indices[s];
        for (int b = 1; b < n_blocks; b++) {
            _Float16 v = all_values[b * spatial + s];
            uint16_t i = all_indices[b * spatial + s];
            if (v > bestVal) {
                bestVal = v;
                bestIdx = i;
            }
        }
        out_indices[s] = (int)bestIdx;
        out_values[s] = (float)bestVal;
    }
    free(all_values); free(all_indices); free(ctxs);
    return true;
#else
    return false;
#endif
}

/* ---------- Fused expansion + argmax (hybrid CPU head) ---------- */

#if defined(__aarch64__) || defined(__arm64__)

typedef struct {
    const _Float16 *proj;       /* [bottleneck][spatial] local copy */
    const _Float16 *weights;    /* [vocab_size][cols_per_group] expansion weights */
    int spatial;
    int bottleneck;
    int groups;
    int vocab_size;
    int cols_per_group;
    int ch_start;               /* first output channel for this block */
    int ch_end;                 /* one past last output channel */
    _Float16 *partial_values;   /* [spatial] best values */
    uint16_t *partial_indices;  /* [spatial] best indices */
    bool ok;
} fused_exp_argmax_ctx;

static bool neon_fused_expansion_argmax(fused_exp_argmax_ctx *ctx) {
    const int spatial = ctx->spatial;
    const int groups = ctx->groups;
    const int cols_per_group = ctx->cols_per_group;
    const int vocab_size = ctx->vocab_size;
    const int ch_per_group = vocab_size / groups;
    const int nvecs = spatial / 8;
    const _Float16 *proj = ctx->proj;
    const _Float16 *weights = ctx->weights;
    if (vocab_size > 65535) return false;
    if (nvecs <= 0 || ctx->ch_start >= ctx->ch_end) return false;

    float16x8_t stackBestV[ARGMAX_MAX_NVECS];
    uint16x8_t stackBestI[ARGMAX_MAX_NVECS];
    float16x8_t stackAccum[ARGMAX_MAX_NVECS];
    float16x8_t *bestV = stackBestV;
    uint16x8_t *bestI = stackBestI;
    float16x8_t *accum = stackAccum;
    bool usesHeap = false;

    if (nvecs > ARGMAX_MAX_NVECS) {
        bestV = (float16x8_t *)malloc((size_t)nvecs * sizeof(*bestV));
        bestI = (uint16x8_t *)malloc((size_t)nvecs * sizeof(*bestI));
        accum = (float16x8_t *)malloc((size_t)nvecs * sizeof(*accum));
        if (!bestV || !bestI || !accum) {
            free(bestV);
            free(bestI);
            free(accum);
            return false;
        }
        usesHeap = true;
    }

    /* Initialize with -inf */
    float16x8_t neg_inf = vdupq_n_f16((_Float16)(-65504.0f));
    for (int v = 0; v < nvecs; v++) {
        bestV[v] = neg_inf;
        bestI[v] = vdupq_n_u16(0);
    }

    for (int c = ctx->ch_start; c < ctx->ch_end; c++) {
        /* Determine which group this channel belongs to */
        int g = c / ch_per_group;
        int c_in_group = c % ch_per_group;
        if (g >= groups) break;

        const _Float16 *w_row = weights + (size_t)c * cols_per_group;
        int proj_ch_base = g * cols_per_group;

        /* Compute logit[s] = sum_k(proj[proj_ch_base+k][s] * w_row[k]) for all s */
        for (int v = 0; v < nvecs; v++) accum[v] = vdupq_n_f16(0);

        for (int k = 0; k < cols_per_group; k++) {
            float16x8_t w = vdupq_n_f16(w_row[k]);
            const _Float16 *proj_row = proj + (size_t)(proj_ch_base + k) * spatial;
            for (int v = 0; v < nvecs; v++) {
                accum[v] = vfmaq_f16(accum[v], w, vld1q_f16(proj_row + v * 8));
            }
        }

        /* Update running max */
        uint16x8_t cidx = vdupq_n_u16((uint16_t)c);
        for (int v = 0; v < nvecs; v++) {
            uint16x8_t gt = vcgtq_f16(accum[v], bestV[v]);
            bestV[v] = vbslq_f16(gt, accum[v], bestV[v]);
            bestI[v] = vbslq_u16(gt, cidx, bestI[v]);
        }
    }

    for (int v = 0; v < nvecs; v++) {
        vst1q_f16(ctx->partial_values + v * 8, bestV[v]);
        vst1q_u16(ctx->partial_indices + v * 8, bestI[v]);
    }
    if (usesHeap) {
        free(bestV);
        free(bestI);
        free(accum);
    }
    return true;
}

#endif /* __aarch64__ */

bool ane_interop_fused_expansion_argmax_fp16(
    IOSurfaceRef proj_surface,
    int proj_ch_off,
    int spatial,
    int bottleneck,
    int groups,
    const void *expansion_weights_fp16,
    int vocab_size,
    int stream_count,
    int *out_indices,
    float *out_values,
    int n_blocks) {

#if defined(__aarch64__) || defined(__arm64__)
    if (!proj_surface || !expansion_weights_fp16 || !out_indices || !out_values) return false;
    if (proj_ch_off < 0 || spatial <= 0 || bottleneck <= 0 || groups <= 0 || vocab_size <= 0) return false;
    if (stream_count <= 0 || stream_count > spatial) return false;
    if (vocab_size > 65535) return false;  /* uint16_t index overflow guard */
    if (spatial > 32768 || (spatial % 8) != 0) return false;
    if (bottleneck % groups != 0 || vocab_size % groups != 0) return false;
    if (vocab_size > (int)UINT16_MAX + 1) return false;
    if (bottleneck > INT_MAX - proj_ch_off) return false;
    if (n_blocks <= 1) n_blocks = 1;
    if (n_blocks > 32) n_blocks = 32;
    if (vocab_size < n_blocks * 2) n_blocks = 1;

    int cols_per_group = bottleneck / groups;

    /* Copy proj surface data to local buffer for fast cached access */
    size_t proj_elems = (size_t)bottleneck * spatial;
    _Float16 *proj_local = (_Float16 *)malloc(proj_elems * sizeof(_Float16));
    if (!proj_local) return false;

    if (IOSurfaceLock(proj_surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) {
        free(proj_local);
        return false;
    }
    const void *base = IOSurfaceGetBaseAddress(proj_surface);
    if (!base) {
        IOSurfaceUnlock(proj_surface, kIOSurfaceLockReadOnly, NULL);
        free(proj_local);
        return false;
    }
    size_t spatialSz = (size_t)spatial;
    size_t maxCh = (size_t)(proj_ch_off + bottleneck - 1);
    size_t maxIdxElems;
    if (mul_size_overflow(maxCh, spatialSz, &maxIdxElems)) {
        IOSurfaceUnlock(proj_surface, kIOSurfaceLockReadOnly, NULL);
        free(proj_local);
        return false;
    }
    size_t elemCount;
    if (add_size_overflow(maxIdxElems, (size_t)(spatial - 1), &maxIdxElems) ||
        add_size_overflow(maxIdxElems, 1, &elemCount)) {
        IOSurfaceUnlock(proj_surface, kIOSurfaceLockReadOnly, NULL);
        free(proj_local);
        return false;
    }
    size_t proj_bytes;
    if (mul_size_overflow(elemCount, sizeof(_Float16), &proj_bytes) ||
        proj_bytes > IOSurfaceGetAllocSize(proj_surface)) {
        IOSurfaceUnlock(proj_surface, kIOSurfaceLockReadOnly, NULL);
        free(proj_local);
        return false;
    }
    const _Float16 *src = (const _Float16 *)base + (size_t)proj_ch_off * spatial;
    memcpy(proj_local, src, proj_elems * sizeof(_Float16));
    IOSurfaceUnlock(proj_surface, kIOSurfaceLockReadOnly, NULL);

    bool ok = false;
    const _Float16 *exp_weights = (const _Float16 *)expansion_weights_fp16;

    if (n_blocks <= 1) {
        /* Serial path */
        fused_exp_argmax_ctx ctx;
        ctx.proj = proj_local;
        ctx.weights = exp_weights;
        ctx.spatial = spatial;
        ctx.bottleneck = bottleneck;
        ctx.groups = groups;
        ctx.vocab_size = vocab_size;
        ctx.cols_per_group = cols_per_group;
        ctx.ch_start = 0;
        ctx.ch_end = vocab_size;
        ctx.ok = false;

        _Float16 *values = (_Float16 *)malloc((size_t)spatial * sizeof(_Float16));
        uint16_t *indices = (uint16_t *)malloc((size_t)spatial * sizeof(uint16_t));
        if (!values || !indices) { free(values); free(indices); free(proj_local); return false; }
        ctx.partial_values = values;
        ctx.partial_indices = indices;

        if (!neon_fused_expansion_argmax(&ctx)) {
            free(values);
            free(indices);
            free(proj_local);
            return false;
        }

        for (int s = 0; s < stream_count; s++) {
            out_indices[s] = (int)indices[s];
            out_values[s] = (float)values[s];
        }
        free(values);
        free(indices);
        ok = true;
    } else {
        /* Parallel path */
        _Float16 *all_values = (_Float16 *)malloc((size_t)n_blocks * spatial * sizeof(_Float16));
        uint16_t *all_indices = (uint16_t *)malloc((size_t)n_blocks * spatial * sizeof(uint16_t));
        fused_exp_argmax_ctx *ctxs = (fused_exp_argmax_ctx *)malloc((size_t)n_blocks * sizeof(fused_exp_argmax_ctx));
        if (!all_values || !all_indices || !ctxs) {
            free(all_values); free(all_indices); free(ctxs); free(proj_local); return false;
        }

        int chPerBlock = vocab_size / n_blocks;
        for (int b = 0; b < n_blocks; b++) {
            ctxs[b].proj = proj_local;
            ctxs[b].weights = exp_weights;
            ctxs[b].spatial = spatial;
            ctxs[b].bottleneck = bottleneck;
            ctxs[b].groups = groups;
            ctxs[b].vocab_size = vocab_size;
            ctxs[b].cols_per_group = cols_per_group;
            ctxs[b].ch_start = b * chPerBlock;
            ctxs[b].ch_end = (b == n_blocks - 1) ? vocab_size : (b + 1) * chPerBlock;
            ctxs[b].partial_values = all_values + b * spatial;
            ctxs[b].partial_indices = all_indices + b * spatial;
            ctxs[b].ok = false;
        }

        dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
        dispatch_apply((size_t)n_blocks, queue, ^(size_t block_idx) {
            ctxs[block_idx].ok = neon_fused_expansion_argmax(&ctxs[block_idx]);
        });

        for (int b = 0; b < n_blocks; b++) {
            if (!ctxs[b].ok) {
                free(all_values);
                free(all_indices);
                free(ctxs);
                free(proj_local);
                return false;
            }
        }

        /* Merge */
        for (int s = 0; s < stream_count; s++) {
            _Float16 bestVal = all_values[s];
            uint16_t bestIdx = all_indices[s];
            for (int b = 1; b < n_blocks; b++) {
                _Float16 v = all_values[b * spatial + s];
                uint16_t i = all_indices[b * spatial + s];
                if (v > bestVal) {
                    bestVal = v;
                    bestIdx = i;
                }
            }
            out_indices[s] = (int)bestIdx;
            out_values[s] = (float)bestVal;
        }
        free(all_values);
        free(all_indices);
        free(ctxs);
        ok = true;
    }

    free(proj_local);
    return ok;
#else
    return false;
#endif
}

#include <limits.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

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
    for (int c = 0; c < channels; c++) {
        size_t idx = (size_t)(ch_off + c) * spatialSz + (size_t)spatial_index;
        dstF16[idx] = (_Float16)data[c];
    }
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
    for (int c = 0; c < channels; c++) {
        size_t idx = (size_t)(ch_off + c) * spatialSz + (size_t)spatial_index;
        data[c] = (float)srcF16[idx];
    }
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
    _Float16 bestValue = srcF16[baseIdx];
    int c = 1;
    const _Float16 *cursor = srcF16 + baseIdx + stride;

    if (channels >= 8) {
        _Float16 bestValue0 = srcF16[baseIdx];
        _Float16 bestValue1 = srcF16[baseIdx + stride];
        _Float16 bestValue2 = srcF16[baseIdx + stride * 2];
        _Float16 bestValue3 = srcF16[baseIdx + stride * 3];
        _Float16 bestValue4 = srcF16[baseIdx + stride * 4];
        _Float16 bestValue5 = srcF16[baseIdx + stride * 5];
        _Float16 bestValue6 = srcF16[baseIdx + stride * 6];
        _Float16 bestValue7 = srcF16[baseIdx + stride * 7];
        int bestIndex0 = 0;
        int bestIndex1 = 1;
        int bestIndex2 = 2;
        int bestIndex3 = 3;
        int bestIndex4 = 4;
        int bestIndex5 = 5;
        int bestIndex6 = 6;
        int bestIndex7 = 7;

        c = 8;
        cursor = srcF16 + baseIdx + stride * 8;
        for (; c + 7 < channels; c += 8) {
            _Float16 value0 = cursor[0];
            _Float16 value1 = cursor[stride];
            _Float16 value2 = cursor[stride * 2];
            _Float16 value3 = cursor[stride * 3];
            _Float16 value4 = cursor[stride * 4];
            _Float16 value5 = cursor[stride * 5];
            _Float16 value6 = cursor[stride * 6];
            _Float16 value7 = cursor[stride * 7];

            if (value0 > bestValue0) {
                bestValue0 = value0;
                bestIndex0 = c;
            }
            if (value1 > bestValue1) {
                bestValue1 = value1;
                bestIndex1 = c + 1;
            }
            if (value2 > bestValue2) {
                bestValue2 = value2;
                bestIndex2 = c + 2;
            }
            if (value3 > bestValue3) {
                bestValue3 = value3;
                bestIndex3 = c + 3;
            }
            if (value4 > bestValue4) {
                bestValue4 = value4;
                bestIndex4 = c + 4;
            }
            if (value5 > bestValue5) {
                bestValue5 = value5;
                bestIndex5 = c + 5;
            }
            if (value6 > bestValue6) {
                bestValue6 = value6;
                bestIndex6 = c + 6;
            }
            if (value7 > bestValue7) {
                bestValue7 = value7;
                bestIndex7 = c + 7;
            }

            cursor += stride * 8;
        }

        bestValue = bestValue0;
        bestIndex = bestIndex0;
        if (bestValue1 > bestValue || (bestValue1 == bestValue && bestIndex1 < bestIndex)) {
            bestValue = bestValue1;
            bestIndex = bestIndex1;
        }
        if (bestValue2 > bestValue || (bestValue2 == bestValue && bestIndex2 < bestIndex)) {
            bestValue = bestValue2;
            bestIndex = bestIndex2;
        }
        if (bestValue3 > bestValue || (bestValue3 == bestValue && bestIndex3 < bestIndex)) {
            bestValue = bestValue3;
            bestIndex = bestIndex3;
        }
        if (bestValue4 > bestValue || (bestValue4 == bestValue && bestIndex4 < bestIndex)) {
            bestValue = bestValue4;
            bestIndex = bestIndex4;
        }
        if (bestValue5 > bestValue || (bestValue5 == bestValue && bestIndex5 < bestIndex)) {
            bestValue = bestValue5;
            bestIndex = bestIndex5;
        }
        if (bestValue6 > bestValue || (bestValue6 == bestValue && bestIndex6 < bestIndex)) {
            bestValue = bestValue6;
            bestIndex = bestIndex6;
        }
        if (bestValue7 > bestValue || (bestValue7 == bestValue && bestIndex7 < bestIndex)) {
            bestValue = bestValue7;
            bestIndex = bestIndex7;
        }
    } else {
        for (; c + 3 < channels; c += 4) {
            _Float16 value0 = cursor[0];
            _Float16 value1 = cursor[stride];
            _Float16 value2 = cursor[stride * 2];
            _Float16 value3 = cursor[stride * 3];

            if (value0 > bestValue) {
                bestValue = value0;
                bestIndex = c;
            }
            if (value1 > bestValue) {
                bestValue = value1;
                bestIndex = c + 1;
            }
            if (value2 > bestValue) {
                bestValue = value2;
                bestIndex = c + 2;
            }
            if (value3 > bestValue) {
                bestValue = value3;
                bestIndex = c + 3;
            }

            cursor += stride * 4;
        }
    }

    for (; c < channels; c++, cursor += stride) {
        _Float16 value = *cursor;
        if (value > bestValue) {
            bestValue = value;
            bestIndex = c;
        }
    }

    *out_index = bestIndex;
    *out_value = (float)bestValue;
    ok = true;

cleanup:
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
    return ok;
}

bool ane_interop_io_write_embedding_batch_fp16(
    IOSurfaceRef surface,
    int ch_off,
    int spatial,
    const float *embedding_table,
    int dim,
    const uint16_t *token_ids,
    int stream_count) {

    if (!surface || !embedding_table || !token_ids) return false;
    if (ch_off < 0 || spatial <= 0 || dim <= 0 || stream_count <= 0) return false;
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
        for (int s = 0; s < stream_count; s++) {
            int token = (int)token_ids[s];
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

        for (int s = 0; s < stream_count; s++) {
            const size_t baseIdx = (size_t)ch_off * spatialSz + (size_t)s;
            const size_t stride = spatialSz;
            int bestIndex = 0;
            _Float16 bestValue = srcF16[baseIdx];

            const _Float16 *cursor = srcF16 + baseIdx + stride;
            int c = 1;

            /* 8-way unrolled scan per lane */
            if (channels >= 8) {
                _Float16 bv0 = srcF16[baseIdx];
                _Float16 bv1 = srcF16[baseIdx + stride];
                _Float16 bv2 = srcF16[baseIdx + stride * 2];
                _Float16 bv3 = srcF16[baseIdx + stride * 3];
                _Float16 bv4 = srcF16[baseIdx + stride * 4];
                _Float16 bv5 = srcF16[baseIdx + stride * 5];
                _Float16 bv6 = srcF16[baseIdx + stride * 6];
                _Float16 bv7 = srcF16[baseIdx + stride * 7];
                int bi0 = 0, bi1 = 1, bi2 = 2, bi3 = 3;
                int bi4 = 4, bi5 = 5, bi6 = 6, bi7 = 7;

                c = 8;
                cursor = srcF16 + baseIdx + stride * 8;
                for (; c + 7 < channels; c += 8) {
                    _Float16 v0 = cursor[0];
                    _Float16 v1 = cursor[stride];
                    _Float16 v2 = cursor[stride * 2];
                    _Float16 v3 = cursor[stride * 3];
                    _Float16 v4 = cursor[stride * 4];
                    _Float16 v5 = cursor[stride * 5];
                    _Float16 v6 = cursor[stride * 6];
                    _Float16 v7 = cursor[stride * 7];
                    if (v0 > bv0) { bv0 = v0; bi0 = c; }
                    if (v1 > bv1) { bv1 = v1; bi1 = c + 1; }
                    if (v2 > bv2) { bv2 = v2; bi2 = c + 2; }
                    if (v3 > bv3) { bv3 = v3; bi3 = c + 3; }
                    if (v4 > bv4) { bv4 = v4; bi4 = c + 4; }
                    if (v5 > bv5) { bv5 = v5; bi5 = c + 5; }
                    if (v6 > bv6) { bv6 = v6; bi6 = c + 6; }
                    if (v7 > bv7) { bv7 = v7; bi7 = c + 7; }
                    cursor += stride * 8;
                }

                bestValue = bv0; bestIndex = bi0;
                if (bv1 > bestValue || (bv1 == bestValue && bi1 < bestIndex)) { bestValue = bv1; bestIndex = bi1; }
                if (bv2 > bestValue || (bv2 == bestValue && bi2 < bestIndex)) { bestValue = bv2; bestIndex = bi2; }
                if (bv3 > bestValue || (bv3 == bestValue && bi3 < bestIndex)) { bestValue = bv3; bestIndex = bi3; }
                if (bv4 > bestValue || (bv4 == bestValue && bi4 < bestIndex)) { bestValue = bv4; bestIndex = bi4; }
                if (bv5 > bestValue || (bv5 == bestValue && bi5 < bestIndex)) { bestValue = bv5; bestIndex = bi5; }
                if (bv6 > bestValue || (bv6 == bestValue && bi6 < bestIndex)) { bestValue = bv6; bestIndex = bi6; }
                if (bv7 > bestValue || (bv7 == bestValue && bi7 < bestIndex)) { bestValue = bv7; bestIndex = bi7; }
            }

            for (; c < channels; c++, cursor += stride) {
                _Float16 value = *cursor;
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

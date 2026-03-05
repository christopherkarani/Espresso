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

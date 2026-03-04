#include <limits.h>
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

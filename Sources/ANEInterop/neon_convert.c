#include <arm_neon.h>
#include <float.h>

#include "ane_interop.h"

// MARK: - Bulk contiguous FP16 <-> FP32 conversion

void ane_interop_cvt_f16_to_f32(float *dst, const void *src, int count) {
    const _Float16 *src16 = (const _Float16 *)src;
    int i = 0;
    for (; i + 7 < count; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16 *)(src16 + i));
        vst1q_f32(dst + i, vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(dst + i + 4, vcvt_f32_f16(vget_high_f16(h)));
    }
    for (; i < count; i++) dst[i] = (float)src16[i];
}

void ane_interop_cvt_f32_to_f16(void *dst, const float *src, int count) {
    _Float16 *dst16 = (_Float16 *)dst;
    int i = 0;
    for (; i + 7 < count; i += 8) {
        float16x8_t h = vcombine_f16(vcvt_f16_f32(vld1q_f32(src + i)),
                                     vcvt_f16_f32(vld1q_f32(src + i + 4)));
        vst1q_f16((__fp16 *)(dst16 + i), h);
    }
    for (; i < count; i++) dst16[i] = (_Float16)src[i];
}

// MARK: - NEON-vectorized contiguous FP16 argmax
//
// Scans `count` contiguous FP16 values and returns the index and value of the
// maximum element. On ties, the lowest index wins (first-max semantics).
// Uses 8-wide vmaxq_f16 for the main loop, then reduces.

void ane_interop_neon_argmax_f16(const void *src, int count,
                                 int *out_index, float *out_value) {
    const __fp16 *s = (const __fp16 *)src;
    if (count <= 0) {
        *out_index = 0;
        *out_value = 0.0f;
        return;
    }

    int bestIndex = 0;
    __fp16 bestValue = s[0];
    int i = 1;

    // NEON 8-wide main loop: track 8 independent max lanes
    if (count >= 16) {
        // Initialize 8 best-value lanes with the first 8 elements.
        float16x8_t vBest = vld1q_f16(s);
        // Track indices as 16-bit integers for efficiency.
        uint16x8_t vIdx = (uint16x8_t){0, 1, 2, 3, 4, 5, 6, 7};
        uint16x8_t vStep = vdupq_n_u16(8);

        i = 8;
        for (; i + 7 < count; i += 8) {
            float16x8_t vCur = vld1q_f16(s + i);
            uint16x8_t vCurIdx = vaddq_u16(vIdx, vStep);
            // Mask: true where current > best (strictly greater).
            uint16x8_t mask = vcgtq_f16(vCur, vBest);
            vBest = vbslq_f16(mask, vCur, vBest);
            vIdx = vbslq_u16(mask, vCurIdx, vIdx);
            // Advance base indices.
            vIdx = vbslq_u16(mask, vCurIdx, vIdx);
            vStep = vaddq_u16(vStep, vdupq_n_u16(8));
        }

        // Reduce 8 lanes to scalar. For first-max semantics, among lanes with
        // equal max value, pick the one with the smallest index.
        __fp16 lanes[8];
        uint16_t indices[8];
        vst1q_f16(lanes, vBest);
        vst1_u16(indices, vget_low_u16(vIdx));
        vst1_u16(indices + 4, vget_high_u16(vIdx));

        bestValue = lanes[0];
        bestIndex = (int)indices[0];
        for (int k = 1; k < 8; k++) {
            if (lanes[k] > bestValue ||
                (lanes[k] == bestValue && (int)indices[k] < bestIndex)) {
                bestValue = lanes[k];
                bestIndex = (int)indices[k];
            }
        }
    }

    // Scalar tail
    for (; i < count; i++) {
        if (s[i] > bestValue) {
            bestValue = s[i];
            bestIndex = i;
        }
    }

    *out_index = bestIndex;
    *out_value = (float)bestValue;
}

// MARK: - NEON-vectorized strided FP16 argmax
//
// Scans `count` FP16 values at stride `stride` (in FP16 elements) starting
// from `src`. This is the hot path for argmax_fp16_spatial_slice where data
// is channel-first [C, S] and we scan one spatial column.

void ane_interop_neon_argmax_f16_strided(const void *src, int count,
                                          int stride,
                                          int *out_index, float *out_value) {
    const __fp16 *s = (const __fp16 *)src;
    if (count <= 0) {
        *out_index = 0;
        *out_value = 0.0f;
        return;
    }

    int bestIndex = 0;
    __fp16 bestValue = s[0];

    // For strided access, NEON gather is not directly available on ARM.
    // Use 8-way scalar unrolling with NEON comparison for the reduction.
    // The main benefit here is the 8-wide independent accumulator pattern
    // that avoids branch misprediction.
    int c = 1;
    const __fp16 *cursor = s + stride;

    if (count >= 16) {
        // 8 independent scalar accumulators
        __fp16 bestVal[8];
        int bestIdx[8];
        bestVal[0] = s[0];
        bestIdx[0] = 0;
        for (int k = 1; k < 8 && k < count; k++) {
            bestVal[k] = s[(size_t)k * stride];
            bestIdx[k] = k;
        }
        int initCount = count < 8 ? count : 8;
        (void)initCount;

        c = 8;
        cursor = s + (size_t)8 * stride;
        for (; c + 7 < count; c += 8) {
            __fp16 v0 = cursor[0];
            __fp16 v1 = cursor[(size_t)stride];
            __fp16 v2 = cursor[(size_t)stride * 2];
            __fp16 v3 = cursor[(size_t)stride * 3];
            __fp16 v4 = cursor[(size_t)stride * 4];
            __fp16 v5 = cursor[(size_t)stride * 5];
            __fp16 v6 = cursor[(size_t)stride * 6];
            __fp16 v7 = cursor[(size_t)stride * 7];

            // Use NEON to compare 8 values at once.
            float16x8_t vCur = (float16x8_t){v0, v1, v2, v3, v4, v5, v6, v7};
            float16x8_t vOld = (float16x8_t){bestVal[0], bestVal[1], bestVal[2], bestVal[3],
                                              bestVal[4], bestVal[5], bestVal[6], bestVal[7]};
            uint16x8_t mask = vcgtq_f16(vCur, vOld);
            float16x8_t vNew = vbslq_f16(mask, vCur, vOld);

            // Store back
            __fp16 newVals[8];
            vst1q_f16(newVals, vNew);

            // For indices, use the mask to conditionally update
            uint16_t maskBits[8];
            vst1q_u16(maskBits, mask);

            for (int k = 0; k < 8; k++) {
                bestVal[k] = newVals[k];
                if (maskBits[k]) {
                    bestIdx[k] = c + k;
                }
            }

            cursor += (size_t)stride * 8;
        }

        // Reduce 8 accumulators: first-max semantics
        bestValue = bestVal[0];
        bestIndex = bestIdx[0];
        for (int k = 1; k < 8; k++) {
            if (bestVal[k] > bestValue ||
                (bestVal[k] == bestValue && bestIdx[k] < bestIndex)) {
                bestValue = bestVal[k];
                bestIndex = bestIdx[k];
            }
        }
    }

    // Scalar tail
    for (; c < count; c++, cursor += stride) {
        __fp16 value = *cursor;
        if (value > bestValue) {
            bestValue = value;
            bestIndex = c;
        }
    }

    *out_index = bestIndex;
    *out_value = (float)bestValue;
}

// MARK: - NEON-vectorized strided FP16 -> FP32 gather
//
// Reads `count` FP16 values at stride `stride` and converts to FP32.
// This is the hot path for readFP16SpatialSlice: channel-first layout
// means reading one element per channel at a fixed spatial offset.

void ane_interop_neon_gather_f16_to_f32(float *dst, const void *src,
                                         int count, int stride) {
    const __fp16 *s = (const __fp16 *)src;
    int i = 0;

    // Gather 4 strided FP16 values, convert to FP32 via NEON.
    // ARM NEON lacks a true gather instruction, so we load scalars
    // into a vector register then convert as a batch.
    for (; i + 3 < count; i += 4) {
        __fp16 v0 = s[(size_t)i * stride];
        __fp16 v1 = s[(size_t)(i + 1) * stride];
        __fp16 v2 = s[(size_t)(i + 2) * stride];
        __fp16 v3 = s[(size_t)(i + 3) * stride];
        float16x4_t h = (float16x4_t){v0, v1, v2, v3};
        float32x4_t f = vcvt_f32_f16(h);
        vst1q_f32(dst + i, f);
    }
    // Scalar tail
    for (; i < count; i++) {
        dst[i] = (float)s[(size_t)i * stride];
    }
}

// MARK: - NEON-vectorized strided FP32 -> FP16 scatter
//
// Writes `count` FP32 values as FP16 at stride `stride`.
// This is the hot path for writeFP16SpatialSlice.

void ane_interop_neon_scatter_f32_to_f16(void *dst, const float *src,
                                          int count, int stride) {
    __fp16 *d = (__fp16 *)dst;
    int i = 0;

    // Convert 4 FP32 values to FP16 via NEON, then scatter-store.
    for (; i + 3 < count; i += 4) {
        float32x4_t f = vld1q_f32(src + i);
        float16x4_t h = vcvt_f16_f32(f);
        __fp16 vals[4];
        vst1_f16(vals, h);
        d[(size_t)i * stride] = vals[0];
        d[(size_t)(i + 1) * stride] = vals[1];
        d[(size_t)(i + 2) * stride] = vals[2];
        d[(size_t)(i + 3) * stride] = vals[3];
    }
    // Scalar tail
    for (; i < count; i++) {
        d[(size_t)i * stride] = (__fp16)src[i];
    }
}

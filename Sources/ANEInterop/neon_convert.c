#include <arm_neon.h>

#include "ane_interop.h"

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


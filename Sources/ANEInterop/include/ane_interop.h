#pragma once

#include <IOSurface/IOSurface.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <CoreFoundation/CoreFoundation.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ANEHandle ANEHandle;

void ane_interop_init(void);
IOSurfaceRef ane_interop_create_surface(size_t bytes) CF_RETURNS_RETAINED;

#define ANE_INTEROP_COMPILE_ERROR_NONE 0
#define ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS 1
#define ANE_INTEROP_COMPILE_ERROR_DUPLICATE_WEIGHT_PATH 2
#define ANE_INTEROP_COMPILE_ERROR_SURFACE_ALLOCATION_FAILED 3
#define ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE 4

ANEHandle *ane_interop_compile(const uint8_t *milText, size_t milLen,
                               const char **weightPaths, const uint8_t **weightDatas,
                               const size_t *weightLens, int weightCount,
                               int nInputs, const size_t *inputSizes,
                               int nOutputs, const size_t *outputSizes);

bool ane_interop_eval(ANEHandle *handle);
IOSurfaceRef ane_interop_get_input(ANEHandle *handle, int index) CF_RETURNS_NOT_RETAINED;
IOSurfaceRef ane_interop_get_output(ANEHandle *handle, int index) CF_RETURNS_NOT_RETAINED;
IOSurfaceRef ane_interop_copy_input(ANEHandle *handle, int index) CF_RETURNS_RETAINED;
IOSurfaceRef ane_interop_copy_output(ANEHandle *handle, int index) CF_RETURNS_RETAINED;
void ane_interop_free(ANEHandle *handle);

int ane_interop_compile_count(void);
void ane_interop_set_compile_count(int value);
int ane_interop_last_compile_error(void);
void ane_interop_set_force_eval_failure(bool value);
int ane_interop_live_handle_count(void);

void ane_interop_cvt_f32_to_f16(void *dst, const float *src, int count);
void ane_interop_cvt_f16_to_f32(float *dst, const void *src, int count);

bool ane_interop_io_copy(IOSurfaceRef dst, int dst_ch_off,
                         IOSurfaceRef src, int src_ch_off,
                         int channels, int spatial);
bool ane_interop_io_write_fp16_at(IOSurfaceRef surface, int ch_off,
                                  const float *data, int channels, int spatial);

#ifdef __cplusplus
} // extern "C"
#endif

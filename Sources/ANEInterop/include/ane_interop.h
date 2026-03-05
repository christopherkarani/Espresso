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
uint64_t ane_interop_last_hw_execution_time_ns(ANEHandle *handle);
bool ane_interop_has_perf_stats(ANEHandle *handle);

void ane_interop_cvt_f32_to_f16(void *dst, const float *src, int count);
void ane_interop_cvt_f16_to_f32(float *dst, const void *src, int count);

bool ane_interop_io_copy(IOSurfaceRef dst, int dst_ch_off,
                         IOSurfaceRef src, int src_ch_off,
                         int channels, int spatial);
/// Copy a single spatial index (token) across `channels` from `src` to `dst`.
///
/// Surfaces are interpreted as channel-first `[channels, spatial]` FP16 tensors.
/// This copies one element per channel at `src_spatial_index` into `dst_spatial_index`.
bool ane_interop_io_copy_fp16_spatial_slice(IOSurfaceRef dst,
                                            int dst_ch_off,
                                            int dst_spatial_index,
                                            int dst_spatial,
                                            IOSurfaceRef src,
                                            int src_ch_off,
                                            int src_spatial_index,
                                            int src_spatial,
                                            int channels);
bool ane_interop_io_write_fp16(IOSurfaceRef surface,
                               const float *data, int channels, int spatial);
bool ane_interop_io_read_fp16(IOSurfaceRef surface, int ch_off,
                              float *data, int channels, int spatial);
bool ane_interop_io_read_fp16_batched(IOSurfaceRef surface, int spatial,
                                      float * const *destinations,
                                      const int *channel_offsets,
                                      const int *channels,
                                      int region_count);
bool ane_interop_io_write_fp16_at(IOSurfaceRef surface, int ch_off,
                                  const float *data, int channels, int spatial);
bool ane_interop_io_write_fp16_at_batched(IOSurfaceRef surface,
                                          const int *channel_offsets,
                                          const float * const *sources,
                                          const int *channels,
                                          int region_count,
                                          int spatial);
bool ane_interop_io_copy_batched(IOSurfaceRef dst,
                                 IOSurfaceRef src,
                                 const int *dst_channel_offsets,
                                 const int *src_channel_offsets,
                                 const int *channels,
                                 int region_count,
                                 int spatial);
bool ane_interop_io_copy_multi_src(IOSurfaceRef dst,
                                   IOSurfaceRef const *sources,
                                   const int *dst_channel_offsets,
                                   const int *src_channel_offsets,
                                   const int *channels,
                                   int region_count,
                                   int spatial);

/// Lock/unlock surfaces independently for batched I/O sequences.
/// Use these to amortize lock overhead across write→eval→read cycles.
bool ane_interop_io_lock_write(IOSurfaceRef surface);
bool ane_interop_io_unlock_write(IOSurfaceRef surface);
bool ane_interop_io_lock_read(IOSurfaceRef surface);
bool ane_interop_io_unlock_read(IOSurfaceRef surface);

/// Write FP16 data to a surface that is already locked for write.
/// Caller must hold a write lock via `ane_interop_io_lock_write()`.
bool ane_interop_io_write_fp16_unlocked(IOSurfaceRef surface,
                                         const float *data, int channels, int spatial);

/// Read FP16 data from a surface that is already locked for read.
/// Caller must hold a read lock via `ane_interop_io_lock_read()`.
bool ane_interop_io_read_fp16_unlocked(IOSurfaceRef surface, int ch_off,
                                        float *data, int channels, int spatial);

#ifdef __cplusplus
} // extern "C"
#endif

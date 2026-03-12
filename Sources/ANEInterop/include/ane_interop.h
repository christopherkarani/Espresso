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

#define ANE_INTEROP_CHAINING_PROBE_UNAVAILABLE 0
#define ANE_INTEROP_CHAINING_PROBE_REQUEST_BUILD_FAILED 1
#define ANE_INTEROP_CHAINING_PROBE_PREPARE_FAILED 2
#define ANE_INTEROP_CHAINING_PROBE_PREPARE_SUCCEEDED 3
#define ANE_INTEROP_CHAINING_PROBE_EXCEPTION 4

typedef enum {
    ANE_INTEROP_CHAINING_STAGE_UNAVAILABLE = 0,
    ANE_INTEROP_CHAINING_STAGE_OUTPUT_SETS_BUILD_FAILED = 1,
    ANE_INTEROP_CHAINING_STAGE_REQUEST_BUILD_FAILED = 2,
    ANE_INTEROP_CHAINING_STAGE_PREPARE_FAILED = 3,
    ANE_INTEROP_CHAINING_STAGE_PREPARE_SUCCEEDED = 4,
    ANE_INTEROP_CHAINING_STAGE_EXCEPTION = 5,
    ANE_INTEROP_CHAINING_STAGE_PREPARE_SKIPPED = 6,
    ANE_INTEROP_CHAINING_STAGE_OUTPUT_SET_ENQUEUE_BUILD_FAILED = 7,
    ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_BUILD_FAILED = 8,
    ANE_INTEROP_CHAINING_STAGE_REQUEST_VALIDATE_FAILED = 9,
    ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_VALIDATE_FAILED = 10,
    ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_CALL_FAILED = 11,
    ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_CALL_SUCCEEDED = 12,
    ANE_INTEROP_CHAINING_STAGE_ENQUEUE_SETS_CALL_FAILED = 13,
    ANE_INTEROP_CHAINING_STAGE_ENQUEUE_SETS_CALL_SUCCEEDED = 14,
    ANE_INTEROP_CHAINING_STAGE_SIGNAL_EVENT_BUILD_FAILED = 15,
} ANEInteropChainingStage;

typedef struct {
    bool useRealStatsSurface;
    bool skipPrepare;
    bool validateRequest;
    bool useScalarLoopbackSymbolIndices;
    bool callEnqueueSets;
    bool callBuffersReady;
    uint32_t requestProcedureIndex;
    uint64_t requestTransactionHandle;
    uint64_t requestFWEnqueueDelay;
    uint64_t requestMemoryPoolId;
    uint32_t enqueueProcedureIndex;
    uint32_t enqueueSetIndex;
    uint64_t enqueueSignalValue;
    bool enqueueSignalNotRequired;
    bool enqueueOpenLoop;
    uint32_t readyProcedureIndex;
    uint64_t readyExecutionDelay;
    bool useSharedSignalEvent;
    uint64_t sharedSignalEventValue;
    uint32_t sharedSignalEventSymbolIndex;
    int64_t sharedSignalEventType;
} ANEInteropChainingProbeOptions;

typedef enum {
    ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE_NULL = 0,
    ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE_OUTPUT0 = 1,
    ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE_SCRATCH = 2,
} ANEInteropChainingProbeStatsSurfaceMode;

typedef struct {
    bool hasChainingRequestClass;
    bool hasPrepareSelector;
    bool hasOutputSetsClass;
    bool hasOutputSetsFactory;
    bool hasOutputSetEnqueueClass;
    bool hasInputBuffersReadyClass;
    bool hasSharedSignalEventClass;
    bool builtOutputSet;
    bool builtOutputSetEnqueue;
    bool builtInputBuffersReady;
    bool builtSharedSignalEvent;
    bool builtRequest;
    bool usedArrayLoopbackSymbolIndices;
    bool usedRealStatsSurface;
    bool requestValidated;
    bool requestValid;
    bool requestValidationFailed;
    bool inputBuffersReadyValidationFailed;
    bool calledEnqueueSets;
    bool enqueueSetsSucceeded;
    bool calledBuffersReady;
    bool buffersReadySucceeded;
    bool prepared;
    int stage;
} ANEInteropChainingProbeResult;

bool ane_interop_runtime_has_chaining_request(void);
bool ane_interop_runtime_has_prepare_chaining(void);
ANEInteropChainingProbeStatsSurfaceMode ane_interop_chaining_probe_stats_surface_mode(void);
void ane_interop_probe_chaining_with_options(ANEHandle *handle,
                                             const ANEInteropChainingProbeOptions *options,
                                             ANEInteropChainingProbeResult *result);
void ane_interop_probe_chaining(ANEHandle *handle, ANEInteropChainingProbeResult *result);
int ane_interop_probe_prepare_chaining(ANEHandle *handle);

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
bool ane_interop_io_write_fp16_spatial_slice(IOSurfaceRef surface,
                                             int ch_off,
                                             int spatial_index,
                                             int spatial,
                                             const float *data,
                                             int channels);
bool ane_interop_io_read_fp16_spatial_slice(IOSurfaceRef surface,
                                            int ch_off,
                                            int spatial_index,
                                            int spatial,
                                            float *data,
                                            int channels);
bool ane_interop_io_argmax_fp16_spatial_slice(IOSurfaceRef surface,
                                              int ch_off,
                                              int spatial_index,
                                              int spatial,
                                              int channels,
                                              int *out_index,
                                              float *out_value);
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

/// Write embeddings for multiple streams to their spatial lanes under one lock.
/// Fuses embedding lookup + FP32→FP16 conversion + strided write.
/// embedding_table is FP32 [vocabSize × dim], row-major.
bool ane_interop_io_write_embedding_batch_fp16(
    IOSurfaceRef surface,
    int ch_off,
    int spatial,
    const float *embedding_table,
    int dim,
    const uint16_t *token_ids,
    int stream_count);

/// Argmax over multiple spatial lanes under one lock.
/// Writes stream_count (index, value) pairs to out_indices and out_values.
bool ane_interop_io_argmax_batch_fp16_spatial(
    IOSurfaceRef surface,
    int ch_off,
    int spatial,
    int channels,
    int stream_count,
    int *out_indices,
    float *out_values);

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

// --- VirtualClient eval path probe ---

typedef enum {
    ANE_INTEROP_VC_STAGE_UNAVAILABLE = 0,
    ANE_INTEROP_VC_STAGE_NO_VIRTUAL_CLIENT = 1,
    ANE_INTEROP_VC_STAGE_SHARED_EVENT_BUILD_FAILED = 2,
    ANE_INTEROP_VC_STAGE_WAIT_EVENT_BUILD_FAILED = 3,
    ANE_INTEROP_VC_STAGE_SHARED_EVENTS_BUILD_FAILED = 4,
    ANE_INTEROP_VC_STAGE_REQUEST_BUILD_FAILED = 5,
    ANE_INTEROP_VC_STAGE_MAP_SURFACES_FAILED = 6,
    ANE_INTEROP_VC_STAGE_EVAL_FAILED = 7,
    ANE_INTEROP_VC_STAGE_EVAL_SUCCEEDED = 8,
    ANE_INTEROP_VC_STAGE_COMPLETION_EVENT_EVAL_FAILED = 9,
    ANE_INTEROP_VC_STAGE_COMPLETION_EVENT_EVAL_SUCCEEDED = 10,
    ANE_INTEROP_VC_STAGE_COMPLETION_HANDLER_EVAL_FAILED = 11,
    ANE_INTEROP_VC_STAGE_COMPLETION_HANDLER_EVAL_SUCCEEDED = 12,
    ANE_INTEROP_VC_STAGE_EXCEPTION = 13,
} ANEInteropVCProbeStage;

typedef struct {
    bool useCompletionEvent;
    bool useCompletionHandler;
    bool useSharedEvents;
    bool useWaitEvent;
    bool skipEval;
    bool mapSurfaces;
    bool loadOnVirtualClient;
    bool useDirectInstantiation;
    uint64_t waitEventValue;
    uint64_t waitEventType;
    uint32_t signalSymbolIndex;
    uint32_t waitSymbolIndex;
} ANEInteropVCProbeOptions;

typedef struct {
    bool hasVirtualClientClass;
    bool hasVirtualClientProperty;
    bool hasSharedEventsClass;
    bool hasSharedWaitEventClass;
    bool hasSharedSignalEventClass;
    bool hasIOSurfaceSharedEventClass;
    bool hasDoEvaluateCompletionEvent;
    bool hasStandardEvaluate;
    bool hasMapIOSurfaces;
    bool hasLoadModel;
    bool hasRequestSharedEventsFactory;
    bool hasSetSharedEvents;
    bool hasSetCompletionHandler;
    bool obtainedVirtualClient;
    bool triedPropertyOnClient;
    bool triedDirectSharedConnection;
    bool triedInitWithSingletonAccess;
    bool triedNew;
    bool directConnectSucceeded;
    bool builtIOSurfaceSharedEvent;
    bool builtWaitEvent;
    bool builtSignalEvent;
    bool builtSharedEventsContainer;
    bool builtRequest;
    bool mappedSurfaces;
    bool loadedOnVirtualClient;
    bool standardEvalSucceeded;
    bool completionEventEvalSucceeded;
    bool completionHandlerFired;
    int stage;
} ANEInteropVCProbeResult;

bool ane_interop_runtime_has_virtual_client(void);
bool ane_interop_runtime_has_shared_events_request(void);
void ane_interop_probe_virtual_client_eval(ANEHandle *handle,
                                            const ANEInteropVCProbeOptions *options,
                                            ANEInteropVCProbeResult *result);

/// Probe code signing identity on _ANEVirtualClient.
/// Calls +getCodeSigningIdentity, optionally +setCodeSigningIdentity:,
/// then retries instantiation. Returns identity string or NULL.
typedef struct {
    bool hasGetCodeSigningIdentity;
    bool hasSetCodeSigningIdentity;
    bool gotIdentityString;
    bool setIdentityBeforeInstantiation;
    bool instantiationSucceededAfterSet;
    char identityString[256];
} ANEInteropCodeSigningProbeResult;

void ane_interop_probe_code_signing(ANEInteropCodeSigningProbeResult *result);

/// Probe completionHandler on the STANDARD eval path (_ANEClient, not VirtualClient).
/// This tests whether setCompletionHandler: fires through the existing evaluateWithQoS: path.
typedef struct {
    bool requestHasCompletionHandler;
    bool completionHandlerSet;
    bool requestHasSharedEvents;
    bool metalDeviceCreated;
    bool builtMetalSharedEvent;
    bool builtSignalEvent;
    bool builtSharedEventsContainer;
    bool sharedEventsAttached;
    bool evalSucceeded;
    bool completionHandlerFired;
    bool eventValueAdvanced;
    uint64_t eventValueBefore;
    uint64_t eventValueAfter;
    double evalTimeMS;
} ANEInteropStandardCompletionProbeResult;

void ane_interop_probe_standard_completion_handler(
    ANEHandle *handle,
    bool useMetalSharedEvent,
    ANEInteropStandardCompletionProbeResult *result);

// --- Real-time eval path probe ---

/// Probes `evaluateRealTimeWithModel:options:request:error:` vs standard eval.
/// Compiles a kernel, loads via both standard and real-time paths, runs N evals
/// on each, and returns per-eval timing for comparison.
typedef struct {
    bool hasBeginRealTimeTask;
    bool hasEndRealTimeTask;
    bool hasLoadRealTimeModel;
    bool hasUnloadRealTimeModel;
    bool hasEvaluateRealTime;
    bool realtimeLoadSucceeded;
    bool realtimeEvalSucceeded;
    bool standardEvalSucceeded;
    int realtimeEvalsCompleted;
    int standardEvalsCompleted;
    double realtimeTotalMS;
    double standardTotalMS;
    double realtimePerEvalMS;
    double standardPerEvalMS;
    double savedPerEvalMS;
    double savedPercent;
} ANEInteropRealTimeProbeResult;

/// Run the real-time vs standard eval benchmark probe.
/// `handle` must be an already-compiled kernel.
/// `nIters` is the number of eval iterations per path (recommended: 20-50).
void ane_interop_probe_realtime_eval(ANEHandle *handle,
                                      int nIters,
                                      ANEInteropRealTimeProbeResult *result);

/// Check if the _ANEClient on a compiled handle supports the real-time eval selectors.
bool ane_interop_runtime_has_realtime_eval(ANEHandle *handle);

#ifdef __cplusplus
} // extern "C"
#endif

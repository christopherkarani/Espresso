#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <dispatch/dispatch.h>
#import <dlfcn.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ane_interop.h"

struct ANEHandle {
    void *model;               // CFBridgingRetain'd _ANEInMemoryModel
    void *client;              // CFBridgingRetain'd _ANEClient (optional)
    void *clientModel;         // CFBridgingRetain'd _ANEModel (optional)
    IOSurfaceRef *ioInputs;
    IOSurfaceRef *ioOutputs;
    void *request;             // CFBridgingRetain'd _ANERequest
    void *perfStats;           // CFBridgingRetain'd _ANEPerformanceStats (optional)
    bool perfStatsRequested;
    unsigned int perfStatsMask;
    void *evalOptions;         // CFBridgingRetain'd NSDictionary (optional)
    bool realtimeLoaded;
    void *tmpDir;              // CFBridgingRetain'd NSString
    int nInputs, nOutputs;
    size_t *inputBytes;
    size_t *outputBytes;
    bool liveHandleCounted;
    uint64_t lastHwExecutionTimeNS;
};

static Class g_ANEDesc = nil, g_ANEInMem = nil, g_ANEReq = nil, g_ANEIO = nil;
static bool g_ane_loaded = false;
static dispatch_once_t g_ane_once;
static int g_compile_count = 0;
static int g_last_compile_error = ANE_INTEROP_COMPILE_ERROR_NONE;
static int g_force_eval_failure = 0;
static int g_live_handle_count = 0;

typedef enum : int {
    ANE_EVAL_INMEM = 0,
    ANE_EVAL_CLIENT = 1,
    ANE_EVAL_CLIENT_DIRECT = 2,
    ANE_EVAL_REALTIME = 3,
} ANEEvalPath;

static bool ane_interop_trace_enabled(void);
static void ane_interop_set_compile_error(int value);
static NSString *ane_interop_sanitized_relative_weight_path(NSString *path);
static bool ane_interop_perf_stats_enabled(void);
static unsigned int ane_interop_perf_stats_anef_mask(void);
static bool ane_interop_env_flag(const char *key);
static bool ane_interop_strict_options_enabled(void);
static long ane_interop_env_long(const char *key, long defaultValue);
static ANEEvalPath ane_interop_eval_path(void);
static bool ane_interop_size_mul_overflow(size_t a, size_t b, size_t *out);
static void ane_interop_remove_tmpdir(NSString *td);
static bool ane_interop_write_weight_files(NSString *tmpDir,
                                           const char **weightPaths,
                                           const uint8_t **weightDatas,
                                           const size_t *weightLens,
                                           int weightCount,
                                           BOOL atomically);
static bool ane_interop_load_realtime_handle(ANEHandle *handle);

static void ane_interop_trace_methods(Class cls, const char *label) {
    if (!ane_interop_trace_enabled() || !cls) return;

    unsigned int classMethodCount = 0;
    Method *classMethods = class_copyMethodList(object_getClass(cls), &classMethodCount);
    fprintf(stderr, "ANE trace class %s class-methods=%u\n", label, classMethodCount);
    for (unsigned int i = 0; i < classMethodCount; i++) {
        SEL sel = method_getName(classMethods[i]);
        fprintf(stderr, "  + %s\n", sel_getName(sel));
    }
    free(classMethods);

    unsigned int instanceMethodCount = 0;
    Method *instanceMethods = class_copyMethodList(cls, &instanceMethodCount);
    fprintf(stderr, "ANE trace class %s instance-methods=%u\n", label, instanceMethodCount);
    for (unsigned int i = 0; i < instanceMethodCount; i++) {
        SEL sel = method_getName(instanceMethods[i]);
        fprintf(stderr, "  - %s\n", sel_getName(sel));
    }
    free(instanceMethods);
}

static bool ane_interop_copy_hex_identifier_bytes(id mdl, char *outHexId, size_t bufLen) {
    if (!mdl || !outHexId || bufLen == 0) return false;

    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    if (![hx isKindOfClass:[NSString class]]) return false;

    const char *hexCString = [((NSString *)hx) UTF8String];
    if (!hexCString) return false;

    size_t needed = strlen(hexCString) + 1;
    if (needed > bufLen) return false;

    memcpy(outHexId, hexCString, needed);
    return true;
}

static bool ane_interop_build_weight_dictionary(const char **weightPaths,
                                                const uint8_t **weightDatas,
                                                const size_t *weightLens,
                                                int weightCount,
                                                NSMutableDictionary **outWeights) {
    NSMutableDictionary *weights = [NSMutableDictionary dictionaryWithCapacity:(NSUInteger)weightCount];
    for (int i = 0; i < weightCount; i++) {
        if (!weightPaths[i]) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return false;
        }
        if (weightLens[i] > 0 && !weightDatas[i]) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return false;
        }

        NSString *path = [NSString stringWithUTF8String:weightPaths[i]];
        if (!path) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return false;
        }
        if (weights[path] != nil) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_DUPLICATE_WEIGHT_PATH);
            return false;
        }

        NSData *wd = [NSData dataWithBytesNoCopy:(void *)weightDatas[i]
                                          length:weightLens[i]
                                    freeWhenDone:NO];
        if (!wd) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return false;
        }

        weights[path] = @{@"offset": @0, @"data": wd};
    }

    if (outWeights) {
        *outWeights = weights;
    }
    return true;
}

static bool ane_interop_write_model_tree(NSString *td,
                                         NSData *milData,
                                         NSDictionary *weights,
                                         BOOL atomically) {
    NSFileManager *fm = [NSFileManager defaultManager];
    if (![fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
       withIntermediateDirectories:YES attributes:nil error:nil]) {
        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
        return false;
    }
    if (![milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES]) {
        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
        return false;
    }

    NSString *tdStd = [td stringByStandardizingPath];
    NSString *tdPrefix = [tdStd hasSuffix:@"/"] ? tdStd : [tdStd stringByAppendingString:@"/"];
    for (NSString *path in weights) {
        NSString *rel = ane_interop_sanitized_relative_weight_path(path);
        if (!rel) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return false;
        }

        NSString *full = [[td stringByAppendingPathComponent:rel] stringByStandardizingPath];
        if (![full hasPrefix:tdPrefix]) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return false;
        }
        if (![fm createDirectoryAtPath:[full stringByDeletingLastPathComponent]
           withIntermediateDirectories:YES attributes:nil error:nil]) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return false;
        }

        NSData *wd = weights[path][@"data"];
        if (![wd writeToFile:full atomically:atomically]) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return false;
        }
    }

    return true;
}

static NSString *ane_interop_unique_reload_directory(NSString *prefix) {
    return [NSTemporaryDirectory() stringByAppendingPathComponent:
        [NSString stringWithFormat:@"%@-%@", prefix, [NSUUID UUID].UUIDString]];
}

static bool ane_interop_move_directory_entries(NSString *sourceDir,
                                               NSString *destinationDir,
                                               NSSet<NSString *> *excludedRootNames) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSError *error = nil;
    NSArray<NSString *> *entries = [fm contentsOfDirectoryAtPath:sourceDir error:&error];
    if (!entries) {
        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
        return false;
    }
    if (![fm createDirectoryAtPath:destinationDir withIntermediateDirectories:YES attributes:nil error:&error]) {
        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
        return false;
    }

    for (NSString *entry in entries) {
        if ([excludedRootNames containsObject:entry]) {
            continue;
        }

        NSString *sourcePath = [sourceDir stringByAppendingPathComponent:entry];
        NSString *destinationPath = [destinationDir stringByAppendingPathComponent:entry];
        if ([fm fileExistsAtPath:destinationPath]) {
            [fm removeItemAtPath:destinationPath error:nil];
        }
        if (![fm moveItemAtPath:sourcePath toPath:destinationPath error:&error]) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return false;
        }
    }

    return true;
}

static NSDictionary *ane_interop_reload_with_fallback_options(id mdl,
                                                              NSDictionary *preferredOptions,
                                                              bool strictOptions,
                                                              NSError **outError) {
    NSError *loadError = nil;
    NSDictionary *loadOptions = preferredOptions ?: @{};
    if (((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            mdl, @selector(loadWithQoS:options:error:), 21, loadOptions, &loadError)) {
        if (outError) *outError = nil;
        return loadOptions;
    }

    if ([loadOptions count] > 0 && !strictOptions) {
        loadError = nil;
        if (((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &loadError)) {
            if (outError) *outError = nil;
            return @{};
        }
    }

    if (outError) *outError = loadError;
    return nil;
}

static NSString *ane_interop_user_caches_directory(void) {
    NSArray<NSString *> *candidates =
        NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES);
    if (candidates.count > 0) {
        return candidates.firstObject;
    }
    return [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Caches"];
}

static bool ane_interop_copy_donor_net_plist(NSString *donorHexId, NSString *td) {
    if (donorHexId.length == 0 || td.length == 0) {
        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
        return false;
    }

    NSFileManager *fm = [NSFileManager defaultManager];
    NSString *source = [[ane_interop_user_caches_directory()
        stringByAppendingPathComponent:donorHexId] stringByAppendingPathComponent:@"net.plist"];
    NSString *destination = [td stringByAppendingPathComponent:@"net.plist"];

    if (![fm fileExistsAtPath:source]) {
        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
        return false;
    }
    if ([fm fileExistsAtPath:destination]) {
        [fm removeItemAtPath:destination error:nil];
    }
    if (![fm copyItemAtPath:source toPath:destination error:nil]) {
        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
        return false;
    }
    return true;
}

static void ane_interop_persist_net_plist_to_cache(NSString *hexId, NSString *td) {
    if (hexId.length == 0 || td.length == 0) {
        return;
    }

    NSFileManager *fm = [NSFileManager defaultManager];
    NSString *source = [td stringByAppendingPathComponent:@"net.plist"];
    if (![fm fileExistsAtPath:source]) {
        return;
    }

    NSString *cacheDir = [[ane_interop_user_caches_directory() stringByAppendingPathComponent:hexId]
        stringByStandardizingPath];
    NSString *destination = [cacheDir stringByAppendingPathComponent:@"net.plist"];
    if (![fm createDirectoryAtPath:cacheDir withIntermediateDirectories:YES attributes:nil error:nil]) {
        return;
    }
    if ([fm fileExistsAtPath:destination]) {
        [fm removeItemAtPath:destination error:nil];
    }
    [fm copyItemAtPath:source toPath:destination error:nil];
}

static NSDictionary *ane_interop_prepare_load_options(id mdl,
                                                      bool wantPerfStats,
                                                      id *outPerfStats,
                                                      unsigned int *outPerfMask) {
    id perfStats = nil;
    unsigned int perfMask = 0;

    if (wantPerfStats) {
        Class perfClass = NSClassFromString(@"_ANEPerformanceStats");
        SEL makeSel = @selector(statsWithRequestPerformanceBuffer:statsBufferSize:);
        if (perfClass && [perfClass respondsToSelector:makeSel]) {
            void *buf = NULL;
            unsigned int bufSize = 0;
            perfStats = ((id(*)(Class,SEL,void **, unsigned int *))objc_msgSend)(
                perfClass, makeSel, &buf, &bufSize);
        }
    }

    NSMutableDictionary *baseOptions = nil;
    if (wantPerfStats || ane_interop_env_flag("ANE_DISABLE_POWER_SAVING") ||
        ane_interop_env_flag("ANE_KEEP_MODEL_WIRED") || ane_interop_env_flag("ANE_ENABLE_LATE_LATCH") ||
        ane_interop_env_flag("ANE_SKIP_PREPARE") || ane_interop_env_flag("ANE_ENABLE_FW_TO_FW_SIGNAL") ||
        ane_interop_env_flag("ANE_DISABLE_IO_FENCES") || getenv("ANE_MEMORY_POOL_ID") != NULL) {
        baseOptions = [NSMutableDictionary dictionary];
    }

    if (baseOptions && ane_interop_env_flag("ANE_DISABLE_POWER_SAVING")) {
        baseOptions[@"kANEFEnablePowerSavingKey"] = @NO;
    }
    if (baseOptions && ane_interop_env_flag("ANE_KEEP_MODEL_WIRED")) {
        baseOptions[@"kANEFKeepModelMemoryWiredKey"] = @YES;
    }
    if (baseOptions && ane_interop_env_flag("ANE_ENABLE_LATE_LATCH")) {
        baseOptions[@"kANEFEnableLateLatchKey"] = @YES;
    }
    if (baseOptions && ane_interop_env_flag("ANE_SKIP_PREPARE")) {
        baseOptions[@"kANEFSkipPreparePhaseKey"] = @YES;
    }
    if (baseOptions && ane_interop_env_flag("ANE_ENABLE_FW_TO_FW_SIGNAL")) {
        baseOptions[@"kANEFEnableFWToFWSignal"] = @YES;
    }
    if (baseOptions && ane_interop_env_flag("ANE_DISABLE_IO_FENCES")) {
        baseOptions[@"kANEFDisableIOFencesUseSharedEventsKey"] = @YES;
    }
    if (baseOptions && getenv("ANE_MEMORY_POOL_ID") != NULL) {
        long mp = ane_interop_env_long("ANE_MEMORY_POOL_ID", -1);
        if (mp >= 0) {
            baseOptions[@"kANEFMemoryPoolIDKey"] = @(mp);
        }
    }

    if (wantPerfStats && [mdl respondsToSelector:@selector(setPerfStatsMask:)]) {
        perfMask = ane_interop_perf_stats_anef_mask();
        Class perfClass = NSClassFromString(@"_ANEPerformanceStats");
        SEL driverSel = @selector(driverMaskForANEFMask:);
        if (perfClass && [perfClass respondsToSelector:driverSel]) {
            perfMask = ((unsigned int(*)(Class,SEL,unsigned int))objc_msgSend)(perfClass, driverSel, perfMask);
        }
        ((void(*)(id,SEL,unsigned int))objc_msgSend)(mdl, @selector(setPerfStatsMask:), perfMask);

        if (!baseOptions) {
            baseOptions = [NSMutableDictionary dictionary];
        }
        baseOptions[@"kANEFPerformanceStatsMask"] = @(perfMask);
        baseOptions[@"kANEFModelLoadPerformanceStats"] = @YES;
    }

    NSDictionary *finalOptions = baseOptions ? [baseOptions copy] : @{};
    if (ane_interop_env_flag("ANE_USE_COMPILER_OPTIONS") &&
        [mdl respondsToSelector:@selector(compilerOptionsWithOptions:isCompiledModelCached:)]) {
        id computed = ((id(*)(id,SEL,id,BOOL))objc_msgSend)(
            mdl, @selector(compilerOptionsWithOptions:isCompiledModelCached:), finalOptions, YES);
        if ([computed isKindOfClass:[NSDictionary class]]) {
            finalOptions = computed;
        }
    }

    if (outPerfStats) *outPerfStats = perfStats;
    if (outPerfMask) *outPerfMask = perfMask;
    return finalOptions;
}

static id ane_interop_make_request_for_handle(ANEHandle *handle) {
    if (!handle) return nil;

    NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:(NSUInteger)handle->nInputs];
    NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:(NSUInteger)handle->nInputs];
    for (int i = 0; i < handle->nInputs; i++) {
        id obj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), handle->ioInputs[i]);
        if (!obj) return nil;
        [wIns addObject:obj];
        [iIdx addObject:@(i)];
    }

    NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:(NSUInteger)handle->nOutputs];
    NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:(NSUInteger)handle->nOutputs];
    for (int i = 0; i < handle->nOutputs; i++) {
        id obj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), handle->ioOutputs[i]);
        if (!obj) return nil;
        [wOuts addObject:obj];
        [oIdx addObject:@(i)];
    }

    id perfStats = handle->perfStats ? (__bridge id)handle->perfStats : nil;
    id req = nil;
    SEL reqSelPerf = @selector(requestWithInputs:inputIndices:outputs:outputIndices:perfStats:procedureIndex:);
    SEL reqSelPerfWB = @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:);
    SEL reqSel = @selector(requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:);
    SEL reqSelWB = @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:);
    if (perfStats) {
        if ([g_ANEReq respondsToSelector:reqSelPerf]) {
            req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(
                g_ANEReq, reqSelPerf, wIns, iIdx, wOuts, oIdx, perfStats, @0);
        } else if ([g_ANEReq respondsToSelector:reqSelPerfWB]) {
            req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
                g_ANEReq, reqSelPerfWB, wIns, iIdx, wOuts, oIdx, nil, perfStats, @0);
        }
    } else {
        if ([g_ANEReq respondsToSelector:reqSel]) {
            req = ((id(*)(Class,SEL,id,id,id,id,id))objc_msgSend)(
                g_ANEReq, reqSel, wIns, iIdx, wOuts, oIdx, @0);
        } else if ([g_ANEReq respondsToSelector:reqSelWB]) {
            req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(
                g_ANEReq, reqSelWB, wIns, iIdx, wOuts, oIdx, nil, @0);
        }
    }
    return req;
}

static ANEHandle *ane_interop_make_loaded_handle(id mdl,
                                                 NSDictionary *finalOptions,
                                                 id perfStats,
                                                 bool perfStatsRequested,
                                                 unsigned int perfMask,
                                                 bool realtimeLoaded,
                                                 NSString *td,
                                                 int nInputs,
                                                 const size_t *inputSizes,
                                                 int nOutputs,
                                                 const size_t *outputSizes) {
    id client = nil;
    id clientModel = nil;
    if ([mdl respondsToSelector:@selector(sharedConnection)]) {
        client = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(sharedConnection));
    }
    if ([mdl respondsToSelector:@selector(model)]) {
        clientModel = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(model));
    }

    ANEHandle *h = (ANEHandle *)calloc(1, sizeof(ANEHandle));
    if (!h) {
        NSError *e = nil;
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            mdl, @selector(unloadWithQoS:error:), 21, &e);
        ane_interop_remove_tmpdir(td);
        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
        return NULL;
    }

    h->model = (void *)CFBridgingRetain(mdl);
    h->client = client ? (void *)CFBridgingRetain(client) : NULL;
    h->clientModel = clientModel ? (void *)CFBridgingRetain(clientModel) : NULL;
    h->perfStats = perfStats ? (void *)CFBridgingRetain(perfStats) : NULL;
    h->perfStatsRequested = perfStatsRequested;
    h->perfStatsMask = perfMask;
    h->evalOptions = (void *)CFBridgingRetain(finalOptions ?: @{});
    h->realtimeLoaded = realtimeLoaded;
    h->tmpDir = (void *)CFBridgingRetain(td);
    h->nInputs = nInputs;
    h->nOutputs = nOutputs;
    h->lastHwExecutionTimeNS = 0;

    if (nInputs > 0) {
        size_t inputMetaBytes = 0;
        if (ane_interop_size_mul_overflow((size_t)nInputs, sizeof(size_t), &inputMetaBytes)) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            ane_interop_free(h);
            return NULL;
        }
        h->inputBytes = (size_t *)malloc(inputMetaBytes);
        h->ioInputs = (IOSurfaceRef *)calloc((size_t)nInputs, sizeof(IOSurfaceRef));
        if (!h->inputBytes || !h->ioInputs) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            ane_interop_free(h);
            return NULL;
        }
        memcpy(h->inputBytes, inputSizes, inputMetaBytes);
        for (int i = 0; i < nInputs; i++) {
            h->ioInputs[i] = ane_interop_create_surface(inputSizes[i]);
            if (!h->ioInputs[i]) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_SURFACE_ALLOCATION_FAILED);
                ane_interop_free(h);
                return NULL;
            }
        }
    }

    if (nOutputs > 0) {
        size_t outputMetaBytes = 0;
        if (ane_interop_size_mul_overflow((size_t)nOutputs, sizeof(size_t), &outputMetaBytes)) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            ane_interop_free(h);
            return NULL;
        }
        h->outputBytes = (size_t *)malloc(outputMetaBytes);
        h->ioOutputs = (IOSurfaceRef *)calloc((size_t)nOutputs, sizeof(IOSurfaceRef));
        if (!h->outputBytes || !h->ioOutputs) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            ane_interop_free(h);
            return NULL;
        }
        memcpy(h->outputBytes, outputSizes, outputMetaBytes);
        for (int i = 0; i < nOutputs; i++) {
            h->ioOutputs[i] = ane_interop_create_surface(outputSizes[i]);
            if (!h->ioOutputs[i]) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_SURFACE_ALLOCATION_FAILED);
                ane_interop_free(h);
                return NULL;
            }
        }
    }

    id req = ane_interop_make_request_for_handle(h);
    if (!req) {
        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
        ane_interop_free(h);
        return NULL;
    }
    h->request = (void *)CFBridgingRetain(req);

    h->liveHandleCounted = true;
    __sync_fetch_and_add(&g_live_handle_count, 1);
    ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_NONE);
    return h;
}

bool ane_interop_get_hex_id(ANEHandle *handle, char *outHexId, size_t bufLen) {
    ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_NONE);
    if (!handle || !outHexId || bufLen == 0) {
        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
        return false;
    }
    if (!handle->model) {
        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
        return false;
    }
    if (!ane_interop_copy_hex_identifier_bytes((__bridge id)handle->model, outHexId, bufLen)) {
        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
        return false;
    }
    return true;
}

ANEHandle *ane_interop_compile_with_id(const uint8_t *milText, size_t milLen,
                                       const char **weightPaths,
                                       const uint8_t **weightDatas,
                                       const size_t *weightLens,
                                       int weightCount,
                                       int nInputs, const size_t *inputSizes,
                                       int nOutputs, const size_t *outputSizes,
                                       char *outHexId, size_t hexIdBufLen) {
    ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_NONE);
    if (!outHexId || hexIdBufLen == 0) {
        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
        return NULL;
    }

    ANEHandle *handle = ane_interop_compile(
        milText,
        milLen,
        weightPaths,
        weightDatas,
        weightLens,
        weightCount,
        nInputs,
        inputSizes,
        nOutputs,
        outputSizes
    );
    if (!handle) {
        return NULL;
    }
    if (!ane_interop_get_hex_id(handle, outHexId, hexIdBufLen)) {
        ane_interop_free(handle);
        return NULL;
    }
    return handle;
}

ANEHandle *ane_interop_delta_reload(const uint8_t *milText, size_t milLen,
                                    const char **weightPaths,
                                    const uint8_t **weightDatas,
                                    const size_t *weightLens,
                                    int weightCount,
                                    int nInputs, const size_t *inputSizes,
                                    int nOutputs, const size_t *outputSizes,
                                    const char *donorHexId) {
    @autoreleasepool {
        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_NONE);
        if (!milText || milLen == 0 || !donorHexId) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return NULL;
        }
        if (weightCount < 0 || nInputs < 0 || nOutputs < 0) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return NULL;
        }
        if (weightCount > 0 && (!weightPaths || !weightDatas || !weightLens)) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return NULL;
        }
        if (nInputs > 0 && !inputSizes) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return NULL;
        }
        if (nOutputs > 0 && !outputSizes) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return NULL;
        }

        NSString *donorHex = [NSString stringWithUTF8String:donorHexId];
        if (!donorHex || donorHex.length == 0) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return NULL;
        }

        ane_interop_init();
        if (!g_ANEDesc || !g_ANEInMem || !g_ANEReq || !g_ANEIO) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return NULL;
        }

        NSData *milData = [NSData dataWithBytesNoCopy:(void *)milText length:milLen freeWhenDone:NO];
        if (!milData) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return NULL;
        }

        NSMutableDictionary *weights = nil;
        if (!ane_interop_build_weight_dictionary(weightPaths, weightDatas, weightLens, weightCount, &weights)) {
            return NULL;
        }

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, (id)(weightCount ? weights : @{}), nil);
        if (!desc) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return NULL;
        }

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return NULL;
        }

        const bool wantPerfStats = ane_interop_perf_stats_enabled();
        id perfStats = nil;
        unsigned int perfMask = 0;
        NSDictionary *finalOptions = ane_interop_prepare_load_options(
            mdl, wantPerfStats, &perfStats, &perfMask
        );

        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        if (![hx isKindOfClass:[NSString class]]) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return NULL;
        }
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        if (!ane_interop_write_model_tree(td, milData, weightCount ? weights : @{}, YES)) {
            ane_interop_remove_tmpdir(td);
            return NULL;
        }
        if (!ane_interop_copy_donor_net_plist(donorHex, td)) {
            ane_interop_remove_tmpdir(td);
            return NULL;
        }

        NSError *e = nil;
        const bool strictOptions = ane_interop_strict_options_enabled();
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, finalOptions, &e)) {
            if ([finalOptions count] > 0 && !strictOptions) {
                e = nil;
                if (((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                        mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
                    finalOptions = @{};
                } else {
                    ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                    ane_interop_remove_tmpdir(td);
                    return NULL;
                }
            } else {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                ane_interop_remove_tmpdir(td);
                return NULL;
            }
        }

        long qd = ane_interop_env_long("ANE_QUEUE_DEPTH", -1);
        if (qd >= 0 && [mdl respondsToSelector:@selector(setQueueDepth:)]) {
            if (qd > 127) qd = 127;
            ((void(*)(id,SEL,char))objc_msgSend)(mdl, @selector(setQueueDepth:), (char)qd);
        }

        id client = nil;
        id clientModel = nil;
        if ([mdl respondsToSelector:@selector(sharedConnection)]) {
            client = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(sharedConnection));
        }
        if ([mdl respondsToSelector:@selector(model)]) {
            clientModel = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(model));
        }

        const bool wantsRealtime = (ane_interop_eval_path() == ANE_EVAL_REALTIME);
        BOOL realtimeLoaded = NO;
        if (wantsRealtime && client && clientModel) {
            NSError *rtErr = nil;
            if ([client respondsToSelector:@selector(beginRealTimeTask)]) {
                ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(beginRealTimeTask));
            }
            if ([client respondsToSelector:@selector(loadRealTimeModel:options:qos:error:)]) {
                realtimeLoaded = ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
                    client, @selector(loadRealTimeModel:options:qos:error:), clientModel, finalOptions, 21, &rtErr);
            }
            if (!realtimeLoaded && [client respondsToSelector:@selector(endRealTimeTask)]) {
                ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(endRealTimeTask));
            }
        }

        return ane_interop_make_loaded_handle(
            mdl,
            finalOptions,
            perfStats,
            wantPerfStats,
            perfMask,
            realtimeLoaded ? true : false,
            td,
            nInputs,
            inputSizes,
            nOutputs,
            outputSizes
        );
    }
}

bool ane_interop_fast_reload(ANEHandle *handle,
                             const char **weightPaths,
                             const uint8_t **weightDatas,
                             const size_t *weightLens,
                             int weightCount) {
    @autoreleasepool {
        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_NONE);
        if (!handle || !handle->model || !handle->tmpDir) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return false;
        }
        if (weightCount < 0) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return false;
        }
        if (weightCount > 0 && (!weightPaths || !weightDatas || !weightLens)) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return false;
        }

        id mdl = (__bridge id)handle->model;
        NSDictionary *options = handle->evalOptions ? (__bridge NSDictionary *)handle->evalOptions : @{};
        NSString *td = (__bridge NSString *)handle->tmpDir;
        const bool strictOptions = ane_interop_strict_options_enabled();
        NSSet<NSString *> *preservedRootFiles = [NSSet setWithObjects:@"model.mil", @"net.plist", nil];
        NSString *stageDir = ane_interop_unique_reload_directory(@"ane-fast-reload-stage");
        NSString *backupDir = ane_interop_unique_reload_directory(@"ane-fast-reload-backup");
        NSFileManager *fm = [NSFileManager defaultManager];
        BOOL hadRealtimeLoaded = handle->realtimeLoaded;

        if (![fm createDirectoryAtPath:stageDir withIntermediateDirectories:YES attributes:nil error:nil]) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return false;
        }
        if (![fm createDirectoryAtPath:backupDir withIntermediateDirectories:YES attributes:nil error:nil]) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            ane_interop_remove_tmpdir(stageDir);
            return false;
        }
        if (!ane_interop_write_weight_files(stageDir, weightPaths, weightDatas, weightLens, weightCount, NO)) {
            ane_interop_remove_tmpdir(stageDir);
            ane_interop_remove_tmpdir(backupDir);
            return false;
        }
        if (!ane_interop_move_directory_entries(td, backupDir, preservedRootFiles)) {
            ane_interop_move_directory_entries(backupDir, td, nil);
            ane_interop_remove_tmpdir(stageDir);
            ane_interop_remove_tmpdir(backupDir);
            return false;
        }

        if (handle->realtimeLoaded && handle->client && handle->clientModel) {
            id client = (__bridge id)handle->client;
            id modelObj = (__bridge id)handle->clientModel;
            NSError *rtErr = nil;
            if ([client respondsToSelector:@selector(unloadRealTimeModel:options:qos:error:)]) {
                ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
                    client, @selector(unloadRealTimeModel:options:qos:error:), modelObj, options, 21, &rtErr);
            }
            if ([client respondsToSelector:@selector(endRealTimeTask)]) {
                ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(endRealTimeTask));
            }
            handle->realtimeLoaded = false;
        }

        NSError *e = nil;
        if (!((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                mdl, @selector(unloadWithQoS:error:), 21, &e)) {
            ane_interop_move_directory_entries(backupDir, td, nil);
            ane_interop_remove_tmpdir(stageDir);
            ane_interop_remove_tmpdir(backupDir);
            if (hadRealtimeLoaded) {
                ane_interop_load_realtime_handle(handle);
            }
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return false;
        }
        if (!ane_interop_move_directory_entries(stageDir, td, nil)) {
            ane_interop_move_directory_entries(backupDir, td, nil);
            NSError *restoreError = nil;
            NSDictionary *restoredOptions = ane_interop_reload_with_fallback_options(
                mdl, options, strictOptions, &restoreError
            );
            if (restoredOptions && restoredOptions != options) {
                if (handle->evalOptions) {
                    CFRelease(handle->evalOptions);
                }
                handle->evalOptions = (void *)CFBridgingRetain(restoredOptions);
            }
            if (hadRealtimeLoaded) {
                ane_interop_load_realtime_handle(handle);
            }
            ane_interop_remove_tmpdir(stageDir);
            ane_interop_remove_tmpdir(backupDir);
            return false;
        }

        NSDictionary *loadOptions = ane_interop_reload_with_fallback_options(
            mdl, options, strictOptions, &e
        );
        if (!loadOptions) {
            ane_interop_move_directory_entries(td, stageDir, preservedRootFiles);
            ane_interop_move_directory_entries(backupDir, td, nil);
            NSDictionary *restoredOptions = ane_interop_reload_with_fallback_options(
                mdl, options, strictOptions, &e
            );
            if (restoredOptions && restoredOptions != options) {
                if (handle->evalOptions) {
                    CFRelease(handle->evalOptions);
                }
                handle->evalOptions = (void *)CFBridgingRetain(restoredOptions);
            }
            if (hadRealtimeLoaded) {
                ane_interop_load_realtime_handle(handle);
            }
            ane_interop_remove_tmpdir(stageDir);
            ane_interop_remove_tmpdir(backupDir);
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return false;
        }

        if (loadOptions != options) {
            if (handle->evalOptions) {
                CFRelease(handle->evalOptions);
            }
            handle->evalOptions = (void *)CFBridgingRetain(loadOptions);
        }

        if (hadRealtimeLoaded) {
            ane_interop_load_realtime_handle(handle);
        }
        handle->lastHwExecutionTimeNS = 0;
        ane_interop_remove_tmpdir(stageDir);
        ane_interop_remove_tmpdir(backupDir);
        return true;
    }
}

static bool ane_interop_trace_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        cached = (getenv("ANE_INTEROP_TRACE") != NULL) ? 1 : 0;
    }
    return cached == 1;
}

static bool ane_interop_perf_stats_enabled(void) {
    // Do not cache: tests and benchmarks may toggle this env var at runtime.
    const char *v = getenv("ANE_PERF_STATS");
    return (v && v[0] == '1');
}

static unsigned int ane_interop_perf_stats_anef_mask(void) {
    // _ANEPerformanceStats.driverMaskForANEFMask: appears to only accept a subset of bits.
    // On M3 Max, any bits outside {0x1,0x2,0x4,0x8} cause it to return 0 (disabling stats).
    // Default to 0xF (enable all supported perf-stats features) and allow override for probing.
    const char *v = getenv("ANE_PERF_STATS_MASK");
    if (!v || v[0] == '\0') return 0xFu;
    char *end = NULL;
    unsigned long parsed = strtoul(v, &end, 0);
    if (end == v) return 0xFu;
    return (unsigned int)parsed;
}

static bool ane_interop_env_flag(const char *key) {
    const char *v = getenv(key);
    return (v && v[0] == '1');
}

static bool ane_interop_strict_options_enabled(void) {
    return ane_interop_env_flag("ANE_STRICT_OPTIONS");
}

static long ane_interop_env_long(const char *key, long defaultValue) {
    const char *v = getenv(key);
    if (!v || v[0] == '\0') return defaultValue;
    char *end = NULL;
    long parsed = strtol(v, &end, 0);
    if (end == v) return defaultValue;
    return parsed;
}

typedef id (*MTLCreateSystemDefaultDeviceFn)(void);

static id ane_interop_create_metal_shared_event(bool *deviceCreated, bool *sharedEventCreated) {
    if (deviceCreated) *deviceCreated = false;
    if (sharedEventCreated) *sharedEventCreated = false;

    void *metalHandle = dlopen("/System/Library/Frameworks/Metal.framework/Metal", RTLD_NOW);
    if (!metalHandle) return nil;

    MTLCreateSystemDefaultDeviceFn createDevice =
        (MTLCreateSystemDefaultDeviceFn)dlsym(metalHandle, "MTLCreateSystemDefaultDevice");
    if (!createDevice) return nil;

    id device = createDevice();
    if (!device) return nil;
    if (deviceCreated) *deviceCreated = true;

    SEL newSharedEventSel = NSSelectorFromString(@"newSharedEvent");
    if (![device respondsToSelector:newSharedEventSel]) return nil;

    id sharedEvent = ((id(*)(id,SEL))objc_msgSend)(device, newSharedEventSel);
    if (sharedEvent && sharedEventCreated) *sharedEventCreated = true;
    return sharedEvent;
}

static uint64_t ane_interop_shared_event_value(id sharedEvent) {
    if (!sharedEvent) return 0;
    SEL signaledValueSel = NSSelectorFromString(@"signaledValue");
    if (![sharedEvent respondsToSelector:signaledValueSel]) return 0;
    return ((unsigned long long(*)(id,SEL))objc_msgSend)(sharedEvent, signaledValueSel);
}

typedef enum : int {
    ANE_COMPILE_CACHE_AUTO = 0,
    ANE_COMPILE_CACHE_PREFER_CACHED = 1,
    ANE_COMPILE_CACHE_FORCE_COLD = 2,
} ANECompileCachePolicy;

static ANECompileCachePolicy ane_interop_compile_cache_policy(void) {
    const char *v = getenv("ANE_COMPILE_CACHE_POLICY");
    if (!v || v[0] == '\0') return ANE_COMPILE_CACHE_AUTO;
    if (strcmp(v, "preferCached") == 0 || strcmp(v, "prefer_cached") == 0) return ANE_COMPILE_CACHE_PREFER_CACHED;
    if (strcmp(v, "forceCold") == 0 || strcmp(v, "force_cold") == 0) return ANE_COMPILE_CACHE_FORCE_COLD;
    return ANE_COMPILE_CACHE_AUTO;
}

static ANEEvalPath ane_interop_eval_path(void) {
    const char *v = getenv("ANE_EVAL_PATH");
    if (!v || v[0] == '\0') return ANE_EVAL_INMEM;
    if (strcmp(v, "client") == 0) return ANE_EVAL_CLIENT;
    if (strcmp(v, "clientDirect") == 0 || strcmp(v, "client_direct") == 0) return ANE_EVAL_CLIENT_DIRECT;
    if (strcmp(v, "realtime") == 0 || strcmp(v, "realTime") == 0) return ANE_EVAL_REALTIME;
    return ANE_EVAL_INMEM;
}

static void ane_interop_set_compile_error(int value) {
    __sync_lock_test_and_set(&g_last_compile_error, value);
}

static bool ane_interop_size_mul_overflow(size_t a, size_t b, size_t *out) {
#if __has_builtin(__builtin_mul_overflow)
    return __builtin_mul_overflow(a, b, out);
#else
    if (a != 0 && b > SIZE_MAX / a) return true;
    *out = a * b;
    return false;
#endif
}

void ane_interop_init(void) {
    if (g_ane_loaded) return;
    dispatch_once(&g_ane_once, ^{
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        g_ANEDesc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
        g_ANEReq = NSClassFromString(@"_ANERequest");
        g_ANEIO = NSClassFromString(@"_ANEIOSurfaceObject");
        g_ane_loaded = true;
    });
}

bool ane_interop_runtime_has_chaining_request(void) {
    ane_interop_init();
    Class chainingReq = NSClassFromString(@"_ANEChainingRequest");
    SEL factorySel = NSSelectorFromString(@"chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:");
    return chainingReq && [chainingReq respondsToSelector:factorySel];
}

bool ane_interop_runtime_has_prepare_chaining(void) {
    ane_interop_init();
    Class clientCls = NSClassFromString(@"_ANEClient");
    SEL prepareSel = @selector(prepareChainingWithModel:options:chainingReq:qos:error:);
    return clientCls && [clientCls instancesRespondToSelector:prepareSel];
}

int ane_interop_probe_prepare_chaining(ANEHandle *handle) {
    ANEInteropChainingProbeResult result;
    memset(&result, 0, sizeof(result));
    ane_interop_probe_chaining(handle, &result);

    switch (result.stage) {
    case ANE_INTEROP_CHAINING_STAGE_REQUEST_BUILD_FAILED:
        return ANE_INTEROP_CHAINING_PROBE_REQUEST_BUILD_FAILED;
    case ANE_INTEROP_CHAINING_STAGE_PREPARE_SUCCEEDED:
        return ANE_INTEROP_CHAINING_PROBE_PREPARE_SUCCEEDED;
    case ANE_INTEROP_CHAINING_STAGE_EXCEPTION:
        return ANE_INTEROP_CHAINING_PROBE_EXCEPTION;
    case ANE_INTEROP_CHAINING_STAGE_OUTPUT_SET_ENQUEUE_BUILD_FAILED:
    case ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_BUILD_FAILED:
    case ANE_INTEROP_CHAINING_STAGE_REQUEST_VALIDATE_FAILED:
    case ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_VALIDATE_FAILED:
    case ANE_INTEROP_CHAINING_STAGE_OUTPUT_SETS_BUILD_FAILED:
    case ANE_INTEROP_CHAINING_STAGE_PREPARE_FAILED:
        return ANE_INTEROP_CHAINING_PROBE_PREPARE_FAILED;
    default:
        return ANE_INTEROP_CHAINING_PROBE_UNAVAILABLE;
    }
}

static bool ane_interop_copy_hex_identifier_string(NSString *hexId,
                                                   char *outHexId,
                                                   size_t bufLen) {
    if (!hexId || !outHexId || bufLen == 0) return false;
    NSUInteger maxLength = (NSUInteger)bufLen;
    return [hexId getCString:outHexId maxLength:maxLength encoding:NSUTF8StringEncoding];
}

static NSMutableDictionary *ane_interop_make_weights_dictionary(const char **weightPaths,
                                                                const uint8_t **weightDatas,
                                                                const size_t *weightLens,
                                                                int weightCount) {
    NSMutableDictionary *weights = [NSMutableDictionary dictionaryWithCapacity:(NSUInteger)weightCount];
    for (int i = 0; i < weightCount; i++) {
        if (!weightPaths[i]) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return nil;
        }
        if (weightLens[i] > 0 && !weightDatas[i]) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return nil;
        }

        NSString *path = [NSString stringWithUTF8String:weightPaths[i]];
        if (!path) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return nil;
        }
        if (weights[path] != nil) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_DUPLICATE_WEIGHT_PATH);
            return nil;
        }

        NSData *weightData = nil;
        if (weightLens[i] == 0) {
            weightData = [NSData data];
        } else {
            weightData = [NSData dataWithBytesNoCopy:(void *)weightDatas[i]
                                              length:weightLens[i]
                                        freeWhenDone:NO];
        }
        if (!weightData) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return nil;
        }

        weights[path] = @{@"offset": @0, @"data": weightData};
    }
    return weights;
}

static bool ane_interop_write_weight_files(NSString *tmpDir,
                                           const char **weightPaths,
                                           const uint8_t **weightDatas,
                                           const size_t *weightLens,
                                           int weightCount,
                                           BOOL atomically) {
    if (!tmpDir) return false;

    NSFileManager *fm = [NSFileManager defaultManager];
    NSString *stdDir = [tmpDir stringByStandardizingPath];
    NSString *dirPrefix = [stdDir hasSuffix:@"/"] ? stdDir : [stdDir stringByAppendingString:@"/"];

    if (![fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
       withIntermediateDirectories:YES attributes:nil error:nil]) {
        return false;
    }

    for (int i = 0; i < weightCount; i++) {
        if (!weightPaths[i]) return false;
        if (weightLens[i] > 0 && !weightDatas[i]) return false;

        NSString *path = [NSString stringWithUTF8String:weightPaths[i]];
        NSString *rel = ane_interop_sanitized_relative_weight_path(path);
        if (!rel) return false;

        NSString *full = [[tmpDir stringByAppendingPathComponent:rel] stringByStandardizingPath];
        if (![full hasPrefix:dirPrefix]) return false;

        if (![fm createDirectoryAtPath:[full stringByDeletingLastPathComponent]
           withIntermediateDirectories:YES attributes:nil error:nil]) {
            return false;
        }

        NSData *data = nil;
        if (weightLens[i] == 0) {
            data = [NSData data];
        } else {
            data = [NSData dataWithBytesNoCopy:(void *)weightDatas[i]
                                        length:weightLens[i]
                                  freeWhenDone:NO];
        }
        if (!data) return false;
        if (![data writeToFile:full atomically:atomically]) return false;
    }

    return true;
}

static bool ane_interop_allocate_handle_surfaces(ANEHandle *handle,
                                                 int nInputs, const size_t *inputSizes,
                                                 int nOutputs, const size_t *outputSizes) {
    handle->nInputs = nInputs;
    handle->nOutputs = nOutputs;

    if (nInputs > 0) {
        size_t inputMetaBytes = 0;
        if (ane_interop_size_mul_overflow((size_t)nInputs, sizeof(size_t), &inputMetaBytes)) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return false;
        }

        handle->inputBytes = (size_t *)malloc(inputMetaBytes);
        handle->ioInputs = (IOSurfaceRef *)calloc((size_t)nInputs, sizeof(IOSurfaceRef));
        if (!handle->inputBytes || !handle->ioInputs) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return false;
        }

        memcpy(handle->inputBytes, inputSizes, inputMetaBytes);
        for (int i = 0; i < nInputs; i++) {
            handle->ioInputs[i] = ane_interop_create_surface(inputSizes[i]);
            if (!handle->ioInputs[i]) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_SURFACE_ALLOCATION_FAILED);
                return false;
            }
        }
    }

    if (nOutputs > 0) {
        size_t outputMetaBytes = 0;
        if (ane_interop_size_mul_overflow((size_t)nOutputs, sizeof(size_t), &outputMetaBytes)) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return false;
        }

        handle->outputBytes = (size_t *)malloc(outputMetaBytes);
        handle->ioOutputs = (IOSurfaceRef *)calloc((size_t)nOutputs, sizeof(IOSurfaceRef));
        if (!handle->outputBytes || !handle->ioOutputs) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return false;
        }

        memcpy(handle->outputBytes, outputSizes, outputMetaBytes);
        for (int i = 0; i < nOutputs; i++) {
            handle->ioOutputs[i] = ane_interop_create_surface(outputSizes[i]);
            if (!handle->ioOutputs[i]) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_SURFACE_ALLOCATION_FAILED);
                return false;
            }
        }
    }

    return true;
}

static id ane_interop_make_request_for_surfaces(IOSurfaceRef *inputs,
                                                int nInputs,
                                                IOSurfaceRef *outputs,
                                                int nOutputs,
                                                id perfStats) {
    NSMutableArray *wrappedInputs = [NSMutableArray arrayWithCapacity:(NSUInteger)nInputs];
    NSMutableArray *inputIndices = [NSMutableArray arrayWithCapacity:(NSUInteger)nInputs];
    for (int i = 0; i < nInputs; i++) {
        id obj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), inputs[i]);
        if (!obj) return nil;
        [wrappedInputs addObject:obj];
        [inputIndices addObject:@(i)];
    }

    NSMutableArray *wrappedOutputs = [NSMutableArray arrayWithCapacity:(NSUInteger)nOutputs];
    NSMutableArray *outputIndices = [NSMutableArray arrayWithCapacity:(NSUInteger)nOutputs];
    for (int i = 0; i < nOutputs; i++) {
        id obj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            g_ANEIO, @selector(objectWithIOSurface:), outputs[i]);
        if (!obj) return nil;
        [wrappedOutputs addObject:obj];
        [outputIndices addObject:@(i)];
    }

    id request = nil;
    SEL reqSelPerf = @selector(requestWithInputs:inputIndices:outputs:outputIndices:perfStats:procedureIndex:);
    SEL reqSelPerfWB = @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:);
    SEL reqSel = @selector(requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:);
    SEL reqSelWB = @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:);

    if (perfStats) {
        if ([g_ANEReq respondsToSelector:reqSelPerf]) {
            request = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(
                g_ANEReq, reqSelPerf, wrappedInputs, inputIndices, wrappedOutputs, outputIndices, perfStats, @0);
        } else if ([g_ANEReq respondsToSelector:reqSelPerfWB]) {
            request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
                g_ANEReq, reqSelPerfWB, wrappedInputs, inputIndices, wrappedOutputs, outputIndices, nil, perfStats, @0);
        }
    } else {
        if ([g_ANEReq respondsToSelector:reqSel]) {
            request = ((id(*)(Class,SEL,id,id,id,id,id))objc_msgSend)(
                g_ANEReq, reqSel, wrappedInputs, inputIndices, wrappedOutputs, outputIndices, @0);
        } else if ([g_ANEReq respondsToSelector:reqSelWB]) {
            request = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(
                g_ANEReq, reqSelWB, wrappedInputs, inputIndices, wrappedOutputs, outputIndices, nil, @0);
        }
    }

    return request;
}

static bool ane_interop_install_request_on_handle(ANEHandle *handle) {
    if (!handle || !g_ANEReq || !g_ANEIO) return false;

    id perfStats = handle->perfStats ? (__bridge id)handle->perfStats : nil;
    id request = ane_interop_make_request_for_surfaces(
        handle->ioInputs,
        handle->nInputs,
        handle->ioOutputs,
        handle->nOutputs,
        perfStats
    );
    if (!request) return false;

    if (handle->request) {
        CFRelease(handle->request);
    }
    handle->request = (void *)CFBridgingRetain(request);
    return true;
}

static void ane_interop_apply_queue_depth(id model) {
    long qd = ane_interop_env_long("ANE_QUEUE_DEPTH", -1);
    if (qd >= 0 && [model respondsToSelector:@selector(setQueueDepth:)]) {
        if (qd > 127) qd = 127;
        ((void(*)(id,SEL,char))objc_msgSend)(model, @selector(setQueueDepth:), (char)qd);
    }
}

static void ane_interop_unload_realtime_handle(ANEHandle *handle) {
    if (!handle || !handle->realtimeLoaded || !handle->client || !handle->clientModel) return;

    id client = (__bridge id)handle->client;
    id modelObj = (__bridge id)handle->clientModel;
    id options = handle->evalOptions ? (__bridge id)handle->evalOptions : @{};
    NSError *err = nil;

    if ([client respondsToSelector:@selector(unloadRealTimeModel:options:qos:error:)]) {
        ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
            client, @selector(unloadRealTimeModel:options:qos:error:), modelObj, options, 21, &err);
    }
    if ([client respondsToSelector:@selector(endRealTimeTask)]) {
        ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(endRealTimeTask));
    }
    handle->realtimeLoaded = false;
}

static bool ane_interop_load_realtime_handle(ANEHandle *handle) {
    if (!handle || !handle->client || !handle->clientModel) return false;
    if (ane_interop_eval_path() != ANE_EVAL_REALTIME) {
        handle->realtimeLoaded = false;
        return true;
    }

    id client = (__bridge id)handle->client;
    id modelObj = (__bridge id)handle->clientModel;
    id options = handle->evalOptions ? (__bridge id)handle->evalOptions : @{};
    NSError *err = nil;

    if ([client respondsToSelector:@selector(beginRealTimeTask)]) {
        ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(beginRealTimeTask));
    }

    BOOL loaded = NO;
    if ([client respondsToSelector:@selector(loadRealTimeModel:options:qos:error:)]) {
        loaded = ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
            client, @selector(loadRealTimeModel:options:qos:error:), modelObj, options, 21, &err);
    }

    if (!loaded && [client respondsToSelector:@selector(endRealTimeTask)]) {
        ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(endRealTimeTask));
    }
    handle->realtimeLoaded = loaded ? true : false;
    return true;
}

ANEInteropChainingProbeStatsSurfaceMode ane_interop_chaining_probe_stats_surface_mode(void) {
    const char *value = getenv("ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE");
    if (value == NULL) {
        return ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE_SCRATCH;
    }
    if (strcmp(value, "null") == 0) {
        return ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE_NULL;
    }
    if (strcmp(value, "output0") == 0) {
        return ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE_OUTPUT0;
    }
    if (strcmp(value, "scratch") == 0) {
        return ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE_SCRATCH;
    }
    return ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE_SCRATCH;
}

void ane_interop_probe_chaining_with_options(ANEHandle *handle,
                                             const ANEInteropChainingProbeOptions *options,
                                             ANEInteropChainingProbeResult *result) {
    @autoreleasepool {
        if (!result) return;
        memset(result, 0, sizeof(*result));
        result->stage = ANE_INTEROP_CHAINING_STAGE_UNAVAILABLE;
        ANEInteropChainingProbeOptions effectiveOptions = {0};
        if (options) {
            effectiveOptions = *options;
        }

        ane_interop_init();

        Class chainingReq = NSClassFromString(@"_ANEChainingRequest");
        Class bufferCls = NSClassFromString(@"_ANEBuffer");
        Class outputSetsCls = NSClassFromString(@"_ANEIOSurfaceOutputSets");
        Class outputSetEnqueueCls = NSClassFromString(@"_ANEOutputSetEnqueue");
        Class inputBuffersReadyCls = NSClassFromString(@"_ANEInputBuffersReady");
        Class sharedSignalEventCls = NSClassFromString(@"_ANESharedSignalEvent");
        Class ioSurfaceSharedEventCls = NSClassFromString(@"IOSurfaceSharedEvent");
        Class clientCls = NSClassFromString(@"_ANEClient");
        SEL factorySel = NSSelectorFromString(@"chainingRequestWithInputs:outputSets:lbInputSymbolId:lbOutputSymbolId:procedureIndex:signalEvents:transactionHandle:fwEnqueueDelay:memoryPoolId:");
        SEL bufferFactorySel = NSSelectorFromString(@"bufferWithIOSurfaceObject:symbolIndex:source:");
        SEL outputSetFactorySel = NSSelectorFromString(@"objectWithstatsSurRef:outputBuffer:");
        SEL outputSetEnqueueFactorySel = NSSelectorFromString(@"outputSetWithProcedureIndex:setIndex:signalValue:signalNotRequired:isOpenLoop:");
        SEL inputBuffersReadyFactorySel = NSSelectorFromString(@"inputBuffersWithProcedureIndex:inputBufferInfoIndex:inputFreeValue:executionDelay:");
        SEL sharedSignalEventFactorySel = NSSelectorFromString(@"signalEventWithValue:symbolIndex:eventType:sharedEvent:");
        SEL validateSel = @selector(validate);
        SEL prepareSel = @selector(prepareChainingWithModel:options:chainingReq:qos:error:);
        SEL enqueueSetsSel = @selector(enqueueSetsWithModel:outputSet:options:qos:error:);
        SEL buffersReadySel = @selector(buffersReadyWithModel:inputBuffers:options:qos:error:);

        result->hasChainingRequestClass = (chainingReq != Nil) && [chainingReq respondsToSelector:factorySel];
        result->hasPrepareSelector = (clientCls != Nil) && [clientCls instancesRespondToSelector:prepareSel];
        result->hasOutputSetsClass = (outputSetsCls != Nil);
        result->hasOutputSetsFactory = (outputSetsCls != Nil) && [outputSetsCls respondsToSelector:outputSetFactorySel];
        result->hasOutputSetEnqueueClass = (outputSetEnqueueCls != Nil);
        result->hasInputBuffersReadyClass = (inputBuffersReadyCls != Nil);
        result->hasSharedSignalEventClass =
            (sharedSignalEventCls != Nil) &&
            (ioSurfaceSharedEventCls != Nil) &&
            [sharedSignalEventCls respondsToSelector:sharedSignalEventFactorySel];

        if (ane_interop_trace_enabled()) {
            ane_interop_trace_methods(chainingReq, "_ANEChainingRequest");
            ane_interop_trace_methods(bufferCls, "_ANEBuffer");
            ane_interop_trace_methods(outputSetsCls, "_ANEIOSurfaceOutputSets");
            ane_interop_trace_methods(outputSetEnqueueCls, "_ANEOutputSetEnqueue");
            ane_interop_trace_methods(inputBuffersReadyCls, "_ANEInputBuffersReady");
            ane_interop_trace_methods(sharedSignalEventCls, "_ANESharedSignalEvent");
        }

        if (!handle || !handle->client || !handle->clientModel) return;
        if (!result->hasChainingRequestClass || !result->hasPrepareSelector) return;

        IOSurfaceRef statsSurRef = NULL;
        ANEInteropChainingProbeStatsSurfaceMode statsSurfaceMode =
            effectiveOptions.useRealStatsSurface
                ? ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE_OUTPUT0
                : ane_interop_chaining_probe_stats_surface_mode();
        @try {
            NSMutableArray *inputs = [NSMutableArray array];
            NSMutableArray *inputBufferInfoIndex = [NSMutableArray array];
            NSMutableArray *inputFreeValue = [NSMutableArray array];
            if (handle->ioInputs && g_ANEIO) {
                for (int i = 0; i < handle->nInputs; i++) {
                    IOSurfaceRef ioIn = handle->ioInputs[i];
                    if (!ioIn) continue;
                    id wrapped = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                        g_ANEIO, @selector(objectWithIOSurface:), ioIn);
                    if (!wrapped) continue;
                    id chainingInput = wrapped;
                    if (bufferCls && [bufferCls respondsToSelector:bufferFactorySel]) {
                        id candidate = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                            bufferCls, bufferFactorySel, wrapped, @(i), 0LL);
                        if (candidate) {
                            chainingInput = candidate;
                        }
                    }
                    [inputs addObject:chainingInput];
                    [inputBufferInfoIndex addObject:@(i)];
                    [inputFreeValue addObject:@0];
                }
            }

            NSMutableArray *outputBuffers = [NSMutableArray array];
            if (handle->ioOutputs && g_ANEIO) {
                for (int i = 0; i < handle->nOutputs; i++) {
                    IOSurfaceRef ioOut = handle->ioOutputs[i];
                    if (!ioOut) continue;
                    id wrapped = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                        g_ANEIO, @selector(objectWithIOSurface:), ioOut);
                    if (!wrapped) continue;
                    id chainingOutput = wrapped;
                    if (bufferCls && [bufferCls respondsToSelector:bufferFactorySel]) {
                        id candidate = ((id(*)(Class,SEL,id,id,long long))objc_msgSend)(
                            bufferCls, bufferFactorySel, wrapped, @(i), 0LL);
                        if (candidate) {
                            chainingOutput = candidate;
                        }
                    }
                    [outputBuffers addObject:chainingOutput];
                }
            }

            id outputSet = nil;
            if (result->hasOutputSetsFactory) {
                if (statsSurfaceMode == ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE_OUTPUT0 &&
                    handle->ioOutputs && handle->nOutputs > 0) {
                    statsSurRef = (IOSurfaceRef)CFRetain(handle->ioOutputs[0]);
                } else if (statsSurfaceMode == ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE_SCRATCH) {
                    statsSurRef = ane_interop_create_surface(256);
                }
                result->usedRealStatsSurface = (
                    statsSurfaceMode == ANE_INTEROP_CHAINING_PROBE_STATS_SURFACE_OUTPUT0 &&
                    statsSurRef != NULL
                );
                outputSet = ((id(*)(Class,SEL,IOSurfaceRef,id))objc_msgSend)(
                    outputSetsCls, outputSetFactorySel, statsSurRef, outputBuffers);
                result->builtOutputSet = (outputSet != nil);
                if (!outputSet) {
                    result->stage = ANE_INTEROP_CHAINING_STAGE_OUTPUT_SETS_BUILD_FAILED;
                    return;
                }
            }
            id outputSets = outputSet ? @[outputSet] : @[];

            id outputSetEnqueue = nil;
            if (result->hasOutputSetEnqueueClass && [outputSetEnqueueCls respondsToSelector:outputSetEnqueueFactorySel]) {
                outputSetEnqueue = ((id(*)(Class,SEL,unsigned int,unsigned int,unsigned long long,BOOL,BOOL))objc_msgSend)(
                outputSetEnqueueCls,
                outputSetEnqueueFactorySel,
                    effectiveOptions.enqueueProcedureIndex,
                    effectiveOptions.enqueueSetIndex,
                    effectiveOptions.enqueueSignalValue,
                    effectiveOptions.enqueueSignalNotRequired ? YES : NO,
                    effectiveOptions.enqueueOpenLoop ? YES : NO);
                result->builtOutputSetEnqueue = (outputSetEnqueue != nil);
                if (!outputSetEnqueue) {
                    result->stage = ANE_INTEROP_CHAINING_STAGE_OUTPUT_SET_ENQUEUE_BUILD_FAILED;
                    return;
                }
            }

            id inputBuffersReady = nil;
            if (result->hasInputBuffersReadyClass && [inputBuffersReadyCls respondsToSelector:inputBuffersReadyFactorySel]) {
                inputBuffersReady = ((id(*)(Class,SEL,unsigned int,id,id,unsigned long long))objc_msgSend)(
                    inputBuffersReadyCls,
                    inputBuffersReadyFactorySel,
                    effectiveOptions.readyProcedureIndex,
                    inputBufferInfoIndex,
                    inputFreeValue,
                    effectiveOptions.readyExecutionDelay);
                result->builtInputBuffersReady = (inputBuffersReady != nil);
                if (!inputBuffersReady) {
                    result->stage = ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_BUILD_FAILED;
                    return;
                }
                if ([inputBuffersReady respondsToSelector:validateSel]) {
                    @try {
                        BOOL validInputBuffersReady = ((BOOL(*)(id,SEL))objc_msgSend)(inputBuffersReady, validateSel);
                        result->inputBuffersReadyValidationFailed = !validInputBuffersReady;
                        if (!validInputBuffersReady) {
                            result->stage = ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_VALIDATE_FAILED;
                            return;
                        }
                    } @catch (NSException *exception) {
                        (void)exception;
                        result->inputBuffersReadyValidationFailed = true;
                        result->stage = ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_VALIDATE_FAILED;
                        return;
                    }
                }
            }

            NSMutableArray *loopbackInputSymbolIndices = [NSMutableArray array];
            NSMutableArray *loopbackOutputSymbolIndices = [NSMutableArray array];
            if ([inputs count] > 0) {
                [loopbackInputSymbolIndices addObject:@0];
            }
            if ([outputBuffers count] > 0) {
                [loopbackOutputSymbolIndices addObject:@0];
            }
            id loopbackInputSymbolIds = loopbackInputSymbolIndices;
            id loopbackOutputSymbolIds = loopbackOutputSymbolIndices;
            result->usedArrayLoopbackSymbolIndices = true;
            if (effectiveOptions.useScalarLoopbackSymbolIndices) {
                loopbackInputSymbolIds = [inputs count] > 0 ? @0 : nil;
                loopbackOutputSymbolIds = [outputBuffers count] > 0 ? @0 : nil;
                result->usedArrayLoopbackSymbolIndices = false;
            }
            id signalEvents = @[];
            if (effectiveOptions.useSharedSignalEvent) {
                if (!result->hasSharedSignalEventClass) {
                    result->stage = ANE_INTEROP_CHAINING_STAGE_SIGNAL_EVENT_BUILD_FAILED;
                    return;
                }
                id sharedEvent = ((id(*)(Class,SEL))objc_msgSend)(
                    ioSurfaceSharedEventCls,
                    @selector(new));
                id sharedSignalEvent = ((id(*)(Class,SEL,unsigned long long,unsigned int,long long,id))objc_msgSend)(
                    sharedSignalEventCls,
                    sharedSignalEventFactorySel,
                    effectiveOptions.sharedSignalEventValue,
                    effectiveOptions.sharedSignalEventSymbolIndex,
                    (long long)effectiveOptions.sharedSignalEventType,
                    sharedEvent);
                result->builtSharedSignalEvent = (sharedSignalEvent != nil);
                if (!sharedSignalEvent) {
                    result->stage = ANE_INTEROP_CHAINING_STAGE_SIGNAL_EVENT_BUILD_FAILED;
                    return;
                }
                signalEvents = @[sharedSignalEvent];
            }
            id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id,id,id))objc_msgSend)(
                chainingReq,
                factorySel,
                inputs,
                outputSets,
                loopbackInputSymbolIds,
                loopbackOutputSymbolIds,
                @(effectiveOptions.requestProcedureIndex),
                signalEvents,
                @((unsigned long long)effectiveOptions.requestTransactionHandle),
                @((unsigned long long)effectiveOptions.requestFWEnqueueDelay),
                @((unsigned long long)effectiveOptions.requestMemoryPoolId));
            if (!req) {
                result->stage = ANE_INTEROP_CHAINING_STAGE_REQUEST_BUILD_FAILED;
                return;
            }
            result->builtRequest = true;
            if (effectiveOptions.validateRequest && [req respondsToSelector:validateSel]) {
                @try {
                    BOOL validRequest = ((BOOL(*)(id,SEL))objc_msgSend)(req, validateSel);
                    result->requestValidated = true;
                    result->requestValid = validRequest ? true : false;
                    result->requestValidationFailed = !validRequest;
                    if (!validRequest) {
                        result->stage = ANE_INTEROP_CHAINING_STAGE_REQUEST_VALIDATE_FAILED;
                        return;
                    }
                } @catch (NSException *exception) {
                    (void)exception;
                    result->requestValidated = true;
                    result->requestValid = false;
                    result->requestValidationFailed = true;
                    result->stage = ANE_INTEROP_CHAINING_STAGE_REQUEST_VALIDATE_FAILED;
                    return;
                }
            }

            id client = (__bridge id)handle->client;
            id modelObj = (__bridge id)handle->clientModel;
            id options = handle->evalOptions ? (__bridge id)handle->evalOptions : @{};

            if (effectiveOptions.callEnqueueSets && outputSetEnqueue && [client respondsToSelector:enqueueSetsSel]) {
                NSError *enqueueError = nil;
                BOOL enqueueOK = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    client,
                    enqueueSetsSel,
                    modelObj,
                    outputSetEnqueue,
                    options,
                    21,
                    &enqueueError
                );
                if (!enqueueOK && ane_interop_trace_enabled()) {
                    fprintf(
                        stderr,
                        "ANE enqueueSetsWithModel failed: %s\n",
                        enqueueError ? [[enqueueError description] UTF8String] : "no error"
                    );
                }
                result->calledEnqueueSets = true;
                result->enqueueSetsSucceeded = enqueueOK ? true : false;
                result->stage = enqueueOK
                    ? ANE_INTEROP_CHAINING_STAGE_ENQUEUE_SETS_CALL_SUCCEEDED
                    : ANE_INTEROP_CHAINING_STAGE_ENQUEUE_SETS_CALL_FAILED;
                if (!enqueueOK) {
                    return;
                }
            }

            if (effectiveOptions.callBuffersReady && inputBuffersReady && [client respondsToSelector:buffersReadySel]) {
                NSError *buffersReadyError = nil;
                BOOL buffersReadyOK = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    client,
                    buffersReadySel,
                    modelObj,
                    inputBuffersReady,
                    options,
                    21,
                    &buffersReadyError
                );
                if (!buffersReadyOK && ane_interop_trace_enabled()) {
                    fprintf(
                        stderr,
                        "ANE buffersReadyWithModel failed: %s\n",
                        buffersReadyError ? [[buffersReadyError description] UTF8String] : "no error"
                    );
                }
                result->calledBuffersReady = true;
                result->buffersReadySucceeded = buffersReadyOK ? true : false;
                result->stage = buffersReadyOK
                    ? ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_CALL_SUCCEEDED
                    : ANE_INTEROP_CHAINING_STAGE_INPUT_BUFFERS_READY_CALL_FAILED;
                if (!buffersReadyOK) {
                    return;
                }
            }

            if (effectiveOptions.skipPrepare) {
                result->stage = ANE_INTEROP_CHAINING_STAGE_PREPARE_SKIPPED;
                return;
            }

            NSError *e = nil;
            BOOL ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                client, prepareSel, modelObj, options, req, 21, &e);
            if (ok && ane_interop_trace_enabled()) {
                fprintf(stderr, "ANE prepareChaining succeeded (outputSets=%lu, outputs=%lu)\n",
                        (unsigned long)[outputSets count], (unsigned long)[outputBuffers count]);
            }
            if (!ok && ane_interop_trace_enabled()) {
                fprintf(stderr, "ANE prepareChaining failed: %s\n", e ? [[e description] UTF8String] : "no error");
            }
            result->prepared = ok ? true : false;
            result->stage = ok ? ANE_INTEROP_CHAINING_STAGE_PREPARE_SUCCEEDED
                               : ANE_INTEROP_CHAINING_STAGE_PREPARE_FAILED;
        } @catch (NSException *exception) {
            if (ane_interop_trace_enabled()) {
                fprintf(stderr, "ANE prepareChaining exception: %s\n", [[exception description] UTF8String]);
            }
            result->stage = ANE_INTEROP_CHAINING_STAGE_EXCEPTION;
        } @finally {
            if (statsSurRef) {
                CFRelease(statsSurRef);
            }
        }
    }
}

void ane_interop_probe_chaining(ANEHandle *handle, ANEInteropChainingProbeResult *result) {
    ANEInteropChainingProbeOptions options;
    memset(&options, 0, sizeof(options));
    options.validateRequest = true;
    ane_interop_probe_chaining_with_options(handle, &options, result);
}

IOSurfaceRef ane_interop_create_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

static void ane_interop_remove_tmpdir(NSString *td) {
    if (!td) return;
    if (getenv("ANE_KEEP_TMPDIR") != NULL) return;
    [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
}

static NSString *ane_interop_sanitized_relative_weight_path(NSString *path) {
    static NSString * const kModelPrefix = @"@model_path/";
    if (![path hasPrefix:kModelPrefix]) return nil;

    NSString *rel = [path substringFromIndex:kModelPrefix.length];
    if (rel.length == 0 || [rel hasPrefix:@"/"]) return nil;

    NSMutableArray<NSString *> *parts = [NSMutableArray array];
    for (NSString *comp in [rel pathComponents]) {
        if (comp.length == 0 || [comp isEqualToString:@"/"]) continue;
        if ([comp isEqualToString:@"."] || [comp isEqualToString:@".."]) return nil;
        [parts addObject:comp];
    }
    if (parts.count == 0) return nil;
    return [NSString pathWithComponents:parts];
}

ANEHandle *ane_interop_compile(const uint8_t *milText, size_t milLen,
                               const char **weightPaths, const uint8_t **weightDatas,
                               const size_t *weightLens, int weightCount,
                               int nInputs, const size_t *inputSizes,
                               int nOutputs, const size_t *outputSizes) {
    @autoreleasepool {
        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_NONE);
        if (!milText || milLen == 0) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return NULL;
        }
        if (weightCount < 0 || nInputs < 0 || nOutputs < 0) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return NULL;
        }
        if (weightCount > 0 && (!weightPaths || !weightDatas || !weightLens)) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return NULL;
        }
        if (nInputs > 0 && !inputSizes) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return NULL;
        }
        if (nOutputs > 0 && !outputSizes) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
            return NULL;
        }

        ane_interop_init();
        if (!g_ANEDesc || !g_ANEInMem || !g_ANEReq || !g_ANEIO) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return NULL;
        }

        NSData *milData = [NSData dataWithBytesNoCopy:(void *)milText length:milLen freeWhenDone:NO];
        if (!milData) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return NULL;
        }

        NSMutableDictionary *weights = [NSMutableDictionary dictionaryWithCapacity:(NSUInteger)weightCount];
        for (int i = 0; i < weightCount; i++) {
            if (!weightPaths[i]) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
                return NULL;
            }
            if (weightLens[i] > 0 && !weightDatas[i]) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
                return NULL;
            }

            NSString *path = [NSString stringWithUTF8String:weightPaths[i]];
            if (!path) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
                return NULL;
            }
            if (weights[path] != nil) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_DUPLICATE_WEIGHT_PATH);
                return NULL;
            }

            NSData *wd = [NSData dataWithBytesNoCopy:(void *)weightDatas[i]
                                              length:weightLens[i]
                                        freeWhenDone:NO];
            if (!wd) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
                return NULL;
            }
            weights[path] = @{@"offset": @0, @"data": wd};
        }

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, (id)(weightCount ? weights : @{}), nil);
        if (!desc) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return NULL;
        }

        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return NULL;
        }

        const bool wantPerfStats = ane_interop_perf_stats_enabled();
        id perfStats = nil;
        if (wantPerfStats) {
            Class perfClass = NSClassFromString(@"_ANEPerformanceStats");
            SEL makeSel = @selector(statsWithRequestPerformanceBuffer:statsBufferSize:);
            if (perfClass && [perfClass respondsToSelector:makeSel]) {
                void *buf = NULL;
                unsigned int bufSize = 0;
                perfStats = ((id(*)(Class,SEL,void **, unsigned int *))objc_msgSend)(
                    perfClass, makeSel, &buf, &bufSize);
                if (ane_interop_trace_enabled()) {
                    fprintf(stderr, "ANE perfStats factory: %s (buf=%p bufSize=%u)\n", perfStats ? "OK" : "nil", buf, bufSize);
                }
            }

            if (ane_interop_trace_enabled() && !perfStats) {
                fprintf(stderr, "ANE perfStats request-buffer factory returned nil\n");
            }
        }

        ANECompileCachePolicy cachePolicy = ane_interop_compile_cache_policy();
        BOOL compiledExists = NO;
        if ([mdl respondsToSelector:@selector(compiledModelExists)]) {
            compiledExists = ((BOOL(*)(id,SEL))objc_msgSend)(mdl, @selector(compiledModelExists));
        }
        if (cachePolicy == ANE_COMPILE_CACHE_FORCE_COLD && [mdl respondsToSelector:@selector(purgeCompiledModel)]) {
            ((void(*)(id,SEL))objc_msgSend)(mdl, @selector(purgeCompiledModel));
            compiledExists = NO;
        }

        unsigned int perfMask = 0;
        NSMutableDictionary *baseOptions = nil;
        if (wantPerfStats || ane_interop_env_flag("ANE_DISABLE_POWER_SAVING") ||
            ane_interop_env_flag("ANE_KEEP_MODEL_WIRED") || ane_interop_env_flag("ANE_ENABLE_LATE_LATCH") ||
            ane_interop_env_flag("ANE_SKIP_PREPARE") || ane_interop_env_flag("ANE_ENABLE_FW_TO_FW_SIGNAL") ||
            ane_interop_env_flag("ANE_DISABLE_IO_FENCES") || getenv("ANE_MEMORY_POOL_ID") != NULL) {
            baseOptions = [NSMutableDictionary dictionary];
        }

        if (baseOptions && ane_interop_env_flag("ANE_DISABLE_POWER_SAVING")) {
            baseOptions[@"kANEFEnablePowerSavingKey"] = @NO;
        }
        if (baseOptions && ane_interop_env_flag("ANE_KEEP_MODEL_WIRED")) {
            baseOptions[@"kANEFKeepModelMemoryWiredKey"] = @YES;
        }
        if (baseOptions && ane_interop_env_flag("ANE_ENABLE_LATE_LATCH")) {
            baseOptions[@"kANEFEnableLateLatchKey"] = @YES;
        }
        if (baseOptions && ane_interop_env_flag("ANE_SKIP_PREPARE")) {
            baseOptions[@"kANEFSkipPreparePhaseKey"] = @YES;
        }
        if (baseOptions && ane_interop_env_flag("ANE_ENABLE_FW_TO_FW_SIGNAL")) {
            baseOptions[@"kANEFEnableFWToFWSignal"] = @YES;
        }
        if (baseOptions && ane_interop_env_flag("ANE_DISABLE_IO_FENCES")) {
            baseOptions[@"kANEFDisableIOFencesUseSharedEventsKey"] = @YES;
        }
        if (baseOptions && getenv("ANE_MEMORY_POOL_ID") != NULL) {
            long mp = ane_interop_env_long("ANE_MEMORY_POOL_ID", -1);
            if (mp >= 0) {
                baseOptions[@"kANEFMemoryPoolIDKey"] = @(mp);
            }
        }

        if (wantPerfStats && [mdl respondsToSelector:@selector(setPerfStatsMask:)]) {
            // Enable perf stats collection (required for hwExecutionTime to populate).
            //
            // NOTE: driverMaskForANEFMask: returns 0 if any unsupported bits are set, so we
            // must be careful to only request supported ANEF bits.
            perfMask = ane_interop_perf_stats_anef_mask();
            Class perfClass = NSClassFromString(@"_ANEPerformanceStats");
            SEL driverSel = @selector(driverMaskForANEFMask:);
            if (perfClass && [perfClass respondsToSelector:driverSel]) {
                perfMask = ((unsigned int(*)(Class,SEL,unsigned int))objc_msgSend)(perfClass, driverSel, perfMask);
            }
            if (ane_interop_trace_enabled()) {
                fprintf(stderr, "ANE perfStatsMask: 0x%08X\n", perfMask);
            }
            ((void(*)(id,SEL,unsigned int))objc_msgSend)(mdl, @selector(setPerfStatsMask:), perfMask);

            // Attempt to also enable stats via driver option keys (best-effort; ignored if unknown).
            if (!baseOptions) {
                baseOptions = [NSMutableDictionary dictionary];
            }
            baseOptions[@"kANEFPerformanceStatsMask"] = @(perfMask);
            baseOptions[@"kANEFModelLoadPerformanceStats"] = @YES;
        }

        NSDictionary *finalOptions = baseOptions ? [baseOptions copy] : @{};
        if (ane_interop_env_flag("ANE_USE_COMPILER_OPTIONS") &&
            [mdl respondsToSelector:@selector(compilerOptionsWithOptions:isCompiledModelCached:)]) {
            id computed = ((id(*)(id,SEL,id,BOOL))objc_msgSend)(
                mdl, @selector(compilerOptionsWithOptions:isCompiledModelCached:), finalOptions, compiledExists);
            if ([computed isKindOfClass:[NSDictionary class]]) {
                finalOptions = (NSDictionary *)computed;
            }
        }
        if (ane_interop_trace_enabled()) {
            fprintf(stderr, "ANE compile cachePolicy=%d compiledExists=%d options=%lu\n",
                    cachePolicy, (int)compiledExists, (unsigned long)[finalOptions count]);
        }

        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        if (![hx isKindOfClass:[NSString class]]) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return NULL;
        }
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        if (ane_interop_trace_enabled()) {
            fprintf(
                stderr,
                "ANE compile: id=%s tmpdir=%s milLen=%zu weights=%d inputs=%d outputs=%d\n",
                [((NSString *)hx) UTF8String],
                [td UTF8String],
                milLen,
                weightCount,
                nInputs,
                nOutputs
            );
            for (NSString *path in weights) {
                fprintf(stderr, "  weight: %s (%zu bytes)\n", [path UTF8String], [weights[path][@"data"] length]);
            }
        }
        NSFileManager *fm = [NSFileManager defaultManager];
        if (![fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
           withIntermediateDirectories:YES attributes:nil error:nil]) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return NULL;
        }
        if (![milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES]) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            ane_interop_remove_tmpdir(td);
            return NULL;
        }
        NSString *tdStd = [td stringByStandardizingPath];
        NSString *tdPrefix = [tdStd hasSuffix:@"/"] ? tdStd : [tdStd stringByAppendingString:@"/"];
        for (NSString *path in weights) {
            NSString *rel = ane_interop_sanitized_relative_weight_path(path);
            if (!rel) {
                fprintf(stderr, "ANE compile failed: invalid weight path '%s'\n", [path UTF8String]);
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
                ane_interop_remove_tmpdir(td);
                return NULL;
            }
            NSString *full = [[td stringByAppendingPathComponent:rel] stringByStandardizingPath];
            if (![full hasPrefix:tdPrefix]) {
                fprintf(stderr, "ANE compile failed: escaped tmp dir for weight path '%s'\n", [path UTF8String]);
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_INVALID_ARGUMENTS);
                ane_interop_remove_tmpdir(td);
                return NULL;
            }
            if (![fm createDirectoryAtPath:[full stringByDeletingLastPathComponent]
               withIntermediateDirectories:YES attributes:nil error:nil]) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                ane_interop_remove_tmpdir(td);
                return NULL;
            }
            if (![weights[path][@"data"] writeToFile:full atomically:YES]) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                ane_interop_remove_tmpdir(td);
                return NULL;
            }
        }

        NSError *e = nil;
        const bool strictOptions = ane_interop_strict_options_enabled();
        BOOL loadedFromCache = NO;
        if (cachePolicy == ANE_COMPILE_CACHE_PREFER_CACHED && compiledExists) {
            if (ane_interop_copy_donor_net_plist((NSString *)hx, td)) {
                NSDictionary *loadOptions = ane_interop_reload_with_fallback_options(
                    mdl, finalOptions, strictOptions, &e
                );
                if (loadOptions) {
                    finalOptions = loadOptions;
                    loadedFromCache = YES;
                } else if (ane_interop_trace_enabled()) {
                    fprintf(stderr, "ANE cached load failed, falling back to compile...\n");
                }
            } else if (ane_interop_trace_enabled()) {
                fprintf(stderr, "ANE cached donor net.plist missing, falling back to compile...\n");
            }
        }

        if (!loadedFromCache) {
            if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mdl, @selector(compileWithQoS:options:error:), 21, finalOptions, &e)) {
                // Retry without options (some host builds reject unknown keys).
                if ([finalOptions count] > 0) {
                    if (strictOptions) {
                        fprintf(stderr, "ANE compile failed with strict options (no fallback): %s\n",
                                e ? [[e description] UTF8String] : "no error");
                        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                        ane_interop_remove_tmpdir(td);
                        return NULL;
                    }
                    if (ane_interop_trace_enabled()) {
                        fprintf(stderr, "ANE compile retrying without options...\n");
                    }
                    e = nil;
                    if (((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                            mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
                        finalOptions = @{};
                    } else {
                        fprintf(stderr, "ANE compile failed: %s\n", e ? [[e description] UTF8String] : "no error");
                        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                        ane_interop_remove_tmpdir(td);
                        return NULL;
                    }
                } else {
                    fprintf(stderr, "ANE compile failed: %s\n", e ? [[e description] UTF8String] : "no error");
                    ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                    ane_interop_remove_tmpdir(td);
                    return NULL;
                }
            }

            NSDictionary *loadOptions = ane_interop_reload_with_fallback_options(
                mdl, finalOptions, strictOptions, &e
            );
            if (!loadOptions) {
                fprintf(stderr, "ANE load failed: %s\n", e ? [[e description] UTF8String] : "no error");
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                ane_interop_remove_tmpdir(td);
                return NULL;
            }
            finalOptions = loadOptions;
        }

        ane_interop_persist_net_plist_to_cache((NSString *)hx, td);

        long qd = ane_interop_env_long("ANE_QUEUE_DEPTH", -1);
        if (qd >= 0 && [mdl respondsToSelector:@selector(setQueueDepth:)]) {
            if (qd > 127) qd = 127;
            ((void(*)(id,SEL,char))objc_msgSend)(mdl, @selector(setQueueDepth:), (char)qd);
            if (ane_interop_trace_enabled()) {
                fprintf(stderr, "ANE queueDepth set to %ld\n", qd);
            }
        }

        id client = nil;
        id clientModel = nil;
        if ([mdl respondsToSelector:@selector(sharedConnection)]) {
            client = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(sharedConnection));
        }
        if ([mdl respondsToSelector:@selector(model)]) {
            clientModel = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(model));
        }

        const bool wantsRealtime = (ane_interop_eval_path() == ANE_EVAL_REALTIME);
        BOOL realtimeLoaded = NO;
        if (wantsRealtime && client && clientModel) {
            NSError *rtErr = nil;
            if ([client respondsToSelector:@selector(beginRealTimeTask)]) {
                ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(beginRealTimeTask));
            }
            if ([client respondsToSelector:@selector(loadRealTimeModel:options:qos:error:)]) {
                realtimeLoaded = ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
                    client, @selector(loadRealTimeModel:options:qos:error:), clientModel, finalOptions, 21, &rtErr);
            }
            if (!realtimeLoaded && ane_interop_trace_enabled()) {
                fprintf(stderr, "ANE realtime load failed: %s\n", rtErr ? [[rtErr description] UTF8String] : "no error");
            }
            if (!realtimeLoaded && [client respondsToSelector:@selector(endRealTimeTask)]) {
                ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(endRealTimeTask));
            }
        }

        ANEHandle *h = (ANEHandle *)calloc(1, sizeof(ANEHandle));
        if (!h) {
            fprintf(stderr, "ANE compile failed: OOM allocating ANEHandle\n");
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
            ane_interop_remove_tmpdir(td);
            return NULL;
        }
        h->model = (void *)CFBridgingRetain(mdl);
        h->client = client ? (void *)CFBridgingRetain(client) : NULL;
        h->clientModel = clientModel ? (void *)CFBridgingRetain(clientModel) : NULL;
        h->perfStats = perfStats ? (void *)CFBridgingRetain(perfStats) : NULL;
        h->perfStatsRequested = wantPerfStats;
        h->perfStatsMask = perfMask;
        h->evalOptions = (void *)CFBridgingRetain(finalOptions);
        h->realtimeLoaded = realtimeLoaded ? true : false;
        h->tmpDir = (void *)CFBridgingRetain(td);
        h->nInputs = nInputs;
        h->nOutputs = nOutputs;
        h->lastHwExecutionTimeNS = 0;

        if (nInputs > 0) {
            size_t inputMetaBytes = 0;
            if (ane_interop_size_mul_overflow((size_t)nInputs, sizeof(size_t), &inputMetaBytes)) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                ane_interop_free(h);
                return NULL;
            }
            h->inputBytes = (size_t *)malloc(inputMetaBytes);
            h->ioInputs = (IOSurfaceRef *)calloc((size_t)nInputs, sizeof(IOSurfaceRef));
            if (!h->inputBytes || !h->ioInputs) {
                fprintf(stderr, "ANE compile failed: OOM allocating input metadata\n");
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                ane_interop_free(h);
                return NULL;
            }
            memcpy(h->inputBytes, inputSizes, inputMetaBytes);
            for (int i = 0; i < nInputs; i++) {
                h->ioInputs[i] = ane_interop_create_surface(inputSizes[i]);
                if (!h->ioInputs[i]) {
                    fprintf(stderr, "ANE compile failed: IOSurfaceCreate returned NULL (input %d)\n", i);
                    ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_SURFACE_ALLOCATION_FAILED);
                    ane_interop_free(h);
                    return NULL;
                }
            }
        }
        if (nOutputs > 0) {
            size_t outputMetaBytes = 0;
            if (ane_interop_size_mul_overflow((size_t)nOutputs, sizeof(size_t), &outputMetaBytes)) {
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                ane_interop_free(h);
                return NULL;
            }
            h->outputBytes = (size_t *)malloc(outputMetaBytes);
            h->ioOutputs = (IOSurfaceRef *)calloc((size_t)nOutputs, sizeof(IOSurfaceRef));
            if (!h->outputBytes || !h->ioOutputs) {
                fprintf(stderr, "ANE compile failed: OOM allocating output metadata\n");
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
                ane_interop_free(h);
                return NULL;
            }
            memcpy(h->outputBytes, outputSizes, outputMetaBytes);
            for (int i = 0; i < nOutputs; i++) {
                h->ioOutputs[i] = ane_interop_create_surface(outputSizes[i]);
                if (!h->ioOutputs[i]) {
                    fprintf(stderr, "ANE compile failed: IOSurfaceCreate returned NULL (output %d)\n", i);
                    ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_SURFACE_ALLOCATION_FAILED);
                    ane_interop_free(h);
                    return NULL;
                }
            }
        }

        NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:(NSUInteger)nInputs];
        NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:(NSUInteger)nInputs];
        for (int i = 0; i < nInputs; i++) {
            id obj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), h->ioInputs[i]);
            if (!obj) {
                fprintf(stderr, "ANE compile failed: _ANEIOSurfaceObject returned nil (input %d)\n", i);
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_SURFACE_ALLOCATION_FAILED);
                ane_interop_free(h);
                return NULL;
            }
            [wIns addObject:obj];
            [iIdx addObject:@(i)];
        }
        NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:(NSUInteger)nOutputs];
        NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:(NSUInteger)nOutputs];
        for (int i = 0; i < nOutputs; i++) {
            id obj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), h->ioOutputs[i]);
            if (!obj) {
                fprintf(stderr, "ANE compile failed: _ANEIOSurfaceObject returned nil (output %d)\n", i);
                ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_SURFACE_ALLOCATION_FAILED);
                ane_interop_free(h);
                return NULL;
            }
            [wOuts addObject:obj];
            [oIdx addObject:@(i)];
        }

        id req = nil;
        SEL reqSelPerfWB = @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:);
        SEL reqSelWB = @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:);
        if (perfStats) {
            req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(
                g_ANEReq, @selector(requestWithInputs:inputIndices:outputs:outputIndices:perfStats:procedureIndex:),
                wIns, iIdx, wOuts, oIdx, perfStats, @0);
            if (!req && [g_ANEReq respondsToSelector:reqSelPerfWB]) {
                req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
                    g_ANEReq, reqSelPerfWB, wIns, iIdx, wOuts, oIdx, nil, perfStats, @0);
            }
        } else {
            // If perfStats factory is unavailable but perfStatsMask is set, the driver may attach perfStatsArray.
            req = ((id(*)(Class,SEL,id,id,id,id,id))objc_msgSend)(
                g_ANEReq, @selector(requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:),
                wIns, iIdx, wOuts, oIdx, @0);
            if (!req && [g_ANEReq respondsToSelector:reqSelWB]) {
                req = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(
                    g_ANEReq, reqSelWB, wIns, iIdx, wOuts, oIdx, nil, @0);
            }
        }
        if (!req) {
            fprintf(stderr, "ANE compile failed: _ANERequest returned nil\n");
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            ane_interop_free(h);
            return NULL;
        }
        h->request = (void *)CFBridgingRetain(req);

        h->liveHandleCounted = true;
        __sync_fetch_and_add(&g_live_handle_count, 1);
        ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_NONE);
        __sync_fetch_and_add(&g_compile_count, 1);
        return h;
    }
}

bool ane_interop_eval(ANEHandle *handle) {
    if (!handle) return false;
    if (__sync_fetch_and_add(&g_force_eval_failure, 0) != 0) {
        return false;
    }
    id mdl = (__bridge id)handle->model;
    id req = (__bridge id)handle->request;
    NSError *e = nil;
    NSDictionary *options = handle->evalOptions ? (__bridge NSDictionary *)handle->evalOptions : @{};

    BOOL ok = NO;
    ANEEvalPath evalPath = ane_interop_eval_path();
    if (evalPath == ANE_EVAL_REALTIME && !handle->realtimeLoaded) {
        // Real-time path may be unavailable on public builds; fall back to standard in-memory eval.
        evalPath = ANE_EVAL_INMEM;
    }
    const bool shouldTryClient = (evalPath != ANE_EVAL_INMEM) || handle->perfStatsRequested;
    if (shouldTryClient && handle->client && handle->clientModel) {
        id client = (__bridge id)handle->client;
        id modelObj = (__bridge id)handle->clientModel;
        if (evalPath == ANE_EVAL_REALTIME && handle->realtimeLoaded &&
            [client respondsToSelector:@selector(evaluateRealTimeWithModel:options:request:error:)]) {
            ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                client, @selector(evaluateRealTimeWithModel:options:request:error:), modelObj, options, req, &e);
        } else {
            SEL sel = @selector(evaluateWithModel:options:request:qos:error:);
            if (evalPath == ANE_EVAL_CLIENT_DIRECT || handle->perfStatsRequested) {
                SEL directSel = @selector(doEvaluateDirectWithModel:options:request:qos:error:);
                if ([client respondsToSelector:directSel]) {
                    sel = directSel;
                }
            }
            ok = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                client, sel, modelObj, options, req, 21, &e);
        }
        if (!ok && ane_interop_trace_enabled()) {
            fprintf(stderr, "ANE client eval failed (will fallback): %s\n", e ? [[e description] UTF8String] : "no error");
        }
    }
    if (!ok) {
        e = nil;
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            mdl, @selector(evaluateWithQoS:options:request:error:), 21, options, req, &e);
    }
    if (!ok) {
        fprintf(stderr, "ANE eval failed: %s\n", e ? [[e description] UTF8String] : "no error");
        handle->lastHwExecutionTimeNS = 0;
    } else if (handle->perfStatsRequested) {
        id ps = nil;
        // Some drivers appear to replace/attach perf stats on the request after eval.
        if ([req respondsToSelector:@selector(perfStats)]) {
            ps = ((id(*)(id,SEL))objc_msgSend)(req, @selector(perfStats));
        }
        if (!ps && [req respondsToSelector:@selector(perfStatsArray)]) {
            id arr = ((id(*)(id,SEL))objc_msgSend)(req, @selector(perfStatsArray));
            if ([arr isKindOfClass:[NSArray class]] && [arr count] > 0) {
                ps = [arr objectAtIndex:0];
            }
        }
        if (!ps) {
            ps = handle->perfStats ? (__bridge id)handle->perfStats : nil;
        }
        handle->lastHwExecutionTimeNS = ps ? ((uint64_t(*)(id,SEL))objc_msgSend)(ps, @selector(hwExecutionTime)) : 0;
        if (ane_interop_trace_enabled()) {
            fprintf(stderr, "ANE hwExecutionTime: %llu ns\n", (unsigned long long)handle->lastHwExecutionTimeNS);
        }
    }
    return ok;
}

IOSurfaceRef ane_interop_get_input(ANEHandle *handle, int index) {
    if (!handle) return NULL;
    if (index < 0 || index >= handle->nInputs) return NULL;
    return handle->ioInputs[index];
}

IOSurfaceRef ane_interop_get_output(ANEHandle *handle, int index) {
    if (!handle) return NULL;
    if (index < 0 || index >= handle->nOutputs) return NULL;
    return handle->ioOutputs[index];
}

IOSurfaceRef ane_interop_copy_input(ANEHandle *handle, int index) {
    IOSurfaceRef s = ane_interop_get_input(handle, index);
    if (!s) return NULL;
    CFRetain(s);
    return s;
}

IOSurfaceRef ane_interop_copy_output(ANEHandle *handle, int index) {
    IOSurfaceRef s = ane_interop_get_output(handle, index);
    if (!s) return NULL;
    CFRetain(s);
    return s;
}

void ane_interop_free(ANEHandle *handle) {
    if (!handle) return;

    if (handle->realtimeLoaded && handle->client && handle->clientModel) {
        id client = (__bridge id)handle->client;
        id modelObj = (__bridge id)handle->clientModel;
        id options = handle->evalOptions ? (__bridge id)handle->evalOptions : @{};
        NSError *rtErr = nil;
        if ([client respondsToSelector:@selector(unloadRealTimeModel:options:qos:error:)]) {
            ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
                client, @selector(unloadRealTimeModel:options:qos:error:), modelObj, options, 21, &rtErr);
        }
        if ([client respondsToSelector:@selector(endRealTimeTask)]) {
            ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(endRealTimeTask));
        }
        handle->realtimeLoaded = false;
    }

    if (handle->model) {
        id mdl = (__bridge id)handle->model;
        NSError *e = nil;
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            mdl, @selector(unloadWithQoS:error:), 21, &e);
    }

    if (handle->ioInputs) {
        for (int i = 0; i < handle->nInputs; i++) {
            if (handle->ioInputs[i]) CFRelease(handle->ioInputs[i]);
        }
    }
    if (handle->ioOutputs) {
        for (int i = 0; i < handle->nOutputs; i++) {
            if (handle->ioOutputs[i]) CFRelease(handle->ioOutputs[i]);
        }
    }
    if (handle->tmpDir) {
        ane_interop_remove_tmpdir((__bridge id)handle->tmpDir);
    }

    if (handle->model) CFRelease(handle->model);
    if (handle->client) CFRelease(handle->client);
    if (handle->clientModel) CFRelease(handle->clientModel);
    if (handle->request) CFRelease(handle->request);
    if (handle->perfStats) CFRelease(handle->perfStats);
    if (handle->evalOptions) CFRelease(handle->evalOptions);
    if (handle->tmpDir) CFRelease(handle->tmpDir);

    free(handle->ioInputs);
    free(handle->ioOutputs);
    free(handle->inputBytes);
    free(handle->outputBytes);
    if (handle->liveHandleCounted) {
        __sync_fetch_and_sub(&g_live_handle_count, 1);
    }
    free(handle);
}

int ane_interop_compile_count(void) {
    return __sync_fetch_and_add(&g_compile_count, 0);
}

void ane_interop_set_compile_count(int value) {
    __sync_lock_test_and_set(&g_compile_count, value);
}

int ane_interop_last_compile_error(void) {
    return __sync_fetch_and_add(&g_last_compile_error, 0);
}

void ane_interop_set_force_eval_failure(bool value) {
    __sync_lock_test_and_set(&g_force_eval_failure, value ? 1 : 0);
}

int ane_interop_live_handle_count(void) {
    return __sync_fetch_and_add(&g_live_handle_count, 0);
}

uint64_t ane_interop_last_hw_execution_time_ns(ANEHandle *handle) {
    if (!handle) return 0;
    return handle->lastHwExecutionTimeNS;
}

bool ane_interop_has_perf_stats(ANEHandle *handle) {
    if (!handle) return false;
    return handle->perfStatsRequested && handle->perfStats != NULL;
}

bool ane_interop_rebind_input(ANEHandle *handle, int index, IOSurfaceRef newSurface) {
    @autoreleasepool {
        if (!handle || !newSurface) return false;
        if (index < 0 || index >= handle->nInputs) return false;
        if (handle->inputBytes && handle->inputBytes[index] > 0 &&
            IOSurfaceGetAllocSize(newSurface) < (size_t)handle->inputBytes[index]) return false;

        ane_interop_init();
        if (!g_ANEReq || !g_ANEIO) return false;

        // Rebuild the _ANERequest with updated surface bindings
        NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:(NSUInteger)handle->nInputs];
        NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:(NSUInteger)handle->nInputs];
        for (int i = 0; i < handle->nInputs; i++) {
            IOSurfaceRef surface = (i == index) ? newSurface : handle->ioInputs[i];
            id obj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), surface);
            if (!obj) return false;
            [wIns addObject:obj];
            [iIdx addObject:@(i)];
        }
        NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:(NSUInteger)handle->nOutputs];
        NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:(NSUInteger)handle->nOutputs];
        for (int i = 0; i < handle->nOutputs; i++) {
            id obj = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), handle->ioOutputs[i]);
            if (!obj) return false;
            [wOuts addObject:obj];
            [oIdx addObject:@(i)];
        }

        id perfStats = handle->perfStats ? (__bridge id)handle->perfStats : nil;
        id newReq = nil;
        SEL reqSelPerf = @selector(requestWithInputs:inputIndices:outputs:outputIndices:perfStats:procedureIndex:);
        SEL reqSelPerfWB = @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:);
        SEL reqSel = @selector(requestWithInputs:inputIndices:outputs:outputIndices:procedureIndex:);
        SEL reqSelWB = @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:procedureIndex:);
        if (perfStats) {
            if ([g_ANEReq respondsToSelector:reqSelPerf]) {
                newReq = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(
                    g_ANEReq, reqSelPerf, wIns, iIdx, wOuts, oIdx, perfStats, @0);
            } else if ([g_ANEReq respondsToSelector:reqSelPerfWB]) {
                newReq = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
                    g_ANEReq, reqSelPerfWB, wIns, iIdx, wOuts, oIdx, nil, perfStats, @0);
            }
        } else {
            if ([g_ANEReq respondsToSelector:reqSel]) {
                newReq = ((id(*)(Class,SEL,id,id,id,id,id))objc_msgSend)(
                    g_ANEReq, reqSel, wIns, iIdx, wOuts, oIdx, @0);
            } else if ([g_ANEReq respondsToSelector:reqSelWB]) {
                newReq = ((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(
                    g_ANEReq, reqSelWB, wIns, iIdx, wOuts, oIdx, nil, @0);
            }
        }
        if (!newReq) return false;

        IOSurfaceRef oldSurface = handle->ioInputs[index];
        void *oldReq = handle->request;
        if (oldSurface != newSurface) {
            CFRetain(newSurface);
            handle->ioInputs[index] = newSurface;
        }
        handle->request = (void *)CFBridgingRetain(newReq);
        if (oldSurface != newSurface && oldSurface) {
            CFRelease(oldSurface);
        }
        if (oldReq) CFRelease(oldReq);

        return true;
    }
}

// --- VirtualClient eval path probe ---

bool ane_interop_runtime_has_virtual_client(void) {
    ane_interop_init();
    Class clientCls = NSClassFromString(@"_ANEClient");
    if (!clientCls) return false;
    SEL vcSel = @selector(virtualClient);
    return [clientCls instancesRespondToSelector:vcSel];
}

bool ane_interop_runtime_has_shared_events_request(void) {
    ane_interop_init();
    Class reqCls = NSClassFromString(@"_ANERequest");
    if (!reqCls) return false;
    // 8-arg factory with sharedEvents parameter
    SEL factorySel = NSSelectorFromString(
        @"requestWithInputs:inputIndices:outputs:outputIndices:"
        @"perfStats:perfStatsMask:procedureIndex:sharedEvents:");
    if ([reqCls respondsToSelector:factorySel]) return true;
    // Fallback: setSharedEvents: setter
    SEL setSel = @selector(setSharedEvents:);
    return [reqCls instancesRespondToSelector:setSel];
}

void ane_interop_probe_virtual_client_eval(ANEHandle *handle,
                                            const ANEInteropVCProbeOptions *options,
                                            ANEInteropVCProbeResult *result) {
    @autoreleasepool {
        if (!result) return;
        memset(result, 0, sizeof(*result));
        result->stage = ANE_INTEROP_VC_STAGE_UNAVAILABLE;

        ANEInteropVCProbeOptions opts = {0};
        if (options) opts = *options;

        ane_interop_init();

        // --- Class/selector discovery ---
        Class virtualClientCls = NSClassFromString(@"_ANEVirtualClient");
        Class sharedEventsCls = NSClassFromString(@"_ANESharedEvents");
        Class sharedWaitEventCls = NSClassFromString(@"_ANESharedWaitEvent");
        Class sharedSignalEventCls = NSClassFromString(@"_ANESharedSignalEvent");
        Class ioSurfaceSharedEventCls = NSClassFromString(@"IOSurfaceSharedEvent");
        Class clientCls = NSClassFromString(@"_ANEClient");

        SEL vcPropertySel = @selector(virtualClient);
        SEL sharedEventsFactorySel = NSSelectorFromString(
            @"sharedEventsWithSignalEvents:waitEvents:");
        SEL waitEventFactorySel = NSSelectorFromString(
            @"waitEventWithValue:sharedEvent:eventType:");
        SEL waitEventSimpleFactorySel = NSSelectorFromString(
            @"waitEventWithValue:sharedEvent:");
        SEL signalEventFactorySel = NSSelectorFromString(
            @"signalEventWithValue:symbolIndex:eventType:sharedEvent:");
        SEL doEvalCompletionEventSel = NSSelectorFromString(
            @"doEvaluateWithModel:options:request:qos:completionEvent:error:");
        SEL standardEvalSel = NSSelectorFromString(
            @"evaluateWithModel:options:request:qos:error:");
        SEL mapSurfacesSel = NSSelectorFromString(
            @"doMapIOSurfacesWithModel:request:cacheInference:error:");
        SEL loadModelSel = NSSelectorFromString(
            @"loadModel:options:qos:error:");

        // 8-arg request factory with sharedEvents
        SEL reqSharedEventsFactorySel = NSSelectorFromString(
            @"requestWithInputs:inputIndices:outputs:outputIndices:"
            @"perfStats:perfStatsMask:procedureIndex:sharedEvents:");
        // 9-arg request factory with sharedEvents + transactionHandle
        SEL reqSharedEvents9FactorySel = NSSelectorFromString(
            @"requestWithInputs:inputIndices:outputs:outputIndices:"
            @"perfStats:perfStatsMask:procedureIndex:sharedEvents:transactionHandle:");
        SEL setSharedEventsSel = @selector(setSharedEvents:);
        SEL setCompletionHandlerSel = @selector(setCompletionHandler:);

        // Populate capability booleans
        result->hasVirtualClientClass = (virtualClientCls != Nil);
        result->hasVirtualClientProperty = (clientCls != Nil) &&
            [clientCls instancesRespondToSelector:vcPropertySel];
        result->hasSharedEventsClass = (sharedEventsCls != Nil) &&
            [sharedEventsCls respondsToSelector:sharedEventsFactorySel];
        result->hasSharedWaitEventClass = (sharedWaitEventCls != Nil) &&
            ([sharedWaitEventCls respondsToSelector:waitEventFactorySel] ||
             [sharedWaitEventCls respondsToSelector:waitEventSimpleFactorySel]);
        result->hasSharedSignalEventClass = (sharedSignalEventCls != Nil) &&
            [sharedSignalEventCls respondsToSelector:signalEventFactorySel];
        result->hasIOSurfaceSharedEventClass = (ioSurfaceSharedEventCls != Nil);
        result->hasDoEvaluateCompletionEvent = (virtualClientCls != Nil) &&
            [virtualClientCls instancesRespondToSelector:doEvalCompletionEventSel];
        result->hasStandardEvaluate = (virtualClientCls != Nil) &&
            [virtualClientCls instancesRespondToSelector:standardEvalSel];
        result->hasMapIOSurfaces = (virtualClientCls != Nil) &&
            [virtualClientCls instancesRespondToSelector:mapSurfacesSel];
        result->hasLoadModel = (virtualClientCls != Nil) &&
            [virtualClientCls instancesRespondToSelector:loadModelSel];
        result->hasRequestSharedEventsFactory = (g_ANEReq != Nil) &&
            ([g_ANEReq respondsToSelector:reqSharedEventsFactorySel] ||
             [g_ANEReq respondsToSelector:reqSharedEvents9FactorySel]);
        result->hasSetSharedEvents = (g_ANEReq != Nil) &&
            [g_ANEReq instancesRespondToSelector:setSharedEventsSel];
        result->hasSetCompletionHandler = (g_ANEReq != Nil) &&
            [g_ANEReq instancesRespondToSelector:setCompletionHandlerSel];

        if (ane_interop_trace_enabled()) {
            ane_interop_trace_methods(virtualClientCls, "_ANEVirtualClient");
            ane_interop_trace_methods(sharedEventsCls, "_ANESharedEvents");
            ane_interop_trace_methods(sharedWaitEventCls, "_ANESharedWaitEvent");
        }

        if (!handle || !handle->client || !handle->clientModel) return;

        // --- Acquire VirtualClient ---
        id client = (__bridge id)handle->client;
        id modelObj = (__bridge id)handle->clientModel;
        id evalOptions = handle->evalOptions ? (__bridge id)handle->evalOptions : @{};

        @try {
            id virtualClient = nil;

            // Path 1: _ANEClient.virtualClient property (original path)
            if (result->hasVirtualClientProperty) {
                result->triedPropertyOnClient = true;
                virtualClient = ((id(*)(id,SEL))objc_msgSend)(client, vcPropertySel);
                if (virtualClient && ane_interop_trace_enabled()) {
                    fprintf(stderr, "ANE VC probe: obtained virtualClient via property: %s\n",
                            [NSStringFromClass([virtualClient class]) UTF8String]);
                }
            }

            // Path 2-5: Direct instantiation fallbacks (when property returns nil)
            if (!virtualClient && opts.useDirectInstantiation && virtualClientCls != Nil) {
                bool trace = ane_interop_trace_enabled();

                // Path 2: [_ANEVirtualClient sharedConnection]
                SEL sharedConnSel = @selector(sharedConnection);
                if ([virtualClientCls respondsToSelector:sharedConnSel]) {
                    result->triedDirectSharedConnection = true;
                    @try {
                        virtualClient = ((id(*)(Class,SEL))objc_msgSend)(
                            virtualClientCls, sharedConnSel);
                    } @catch (NSException *ex) {
                        if (trace) fprintf(stderr, "ANE VC probe: +sharedConnection threw: %s\n",
                                           [[ex description] UTF8String]);
                    }
                    if (trace) {
                        fprintf(stderr, "ANE VC probe: +sharedConnection → %s (class: %s)\n",
                                virtualClient ? "non-nil" : "nil",
                                virtualClient ? [NSStringFromClass([virtualClient class]) UTF8String] : "N/A");
                    }
                } else if (trace) {
                    fprintf(stderr, "ANE VC probe: +sharedConnection NOT available on _ANEVirtualClient\n");
                }

                // Path 3: alloc → initWithSingletonAccess → connect
                if (!virtualClient) {
                    SEL initSingletonSel = NSSelectorFromString(@"initWithSingletonAccess");
                    SEL connectSel = @selector(connect);
                    if ([virtualClientCls instancesRespondToSelector:initSingletonSel]) {
                        result->triedInitWithSingletonAccess = true;
                        id vc = ((id(*)(Class,SEL))objc_msgSend)(virtualClientCls, @selector(alloc));
                        if (trace) fprintf(stderr, "ANE VC probe: alloc → %s\n", vc ? "non-nil" : "nil");
                        if (vc) {
                            @try {
                                vc = ((id(*)(id,SEL))objc_msgSend)(vc, initSingletonSel);
                            } @catch (NSException *ex) {
                                if (trace) fprintf(stderr, "ANE VC probe: initWithSingletonAccess threw: %s\n",
                                                   [[ex description] UTF8String]);
                                vc = nil;
                            }
                            if (trace) fprintf(stderr, "ANE VC probe: initWithSingletonAccess → %s\n",
                                               vc ? "non-nil" : "nil");
                            if (vc) {
                                if ([vc respondsToSelector:connectSel]) {
                                    @try {
                                        ((void(*)(id,SEL))objc_msgSend)(vc, connectSel);
                                        result->directConnectSucceeded = true;
                                        if (trace) fprintf(stderr, "ANE VC probe: connect succeeded\n");
                                    } @catch (NSException *ex) {
                                        if (trace) fprintf(stderr, "ANE VC probe: connect threw: %s\n",
                                                           [[ex description] UTF8String]);
                                    }
                                }
                                // Check if VirtualClient reports ANE availability
                                SEL hasANESel = @selector(hasANE);
                                if ([vc respondsToSelector:hasANESel]) {
                                    BOOL has = ((BOOL(*)(id,SEL))objc_msgSend)(vc, hasANESel);
                                    if (trace) fprintf(stderr, "ANE VC probe: hasANE → %s\n", has ? "YES" : "NO");
                                }
                                virtualClient = vc;
                            }
                        }
                    } else if (trace) {
                        fprintf(stderr, "ANE VC probe: initWithSingletonAccess NOT available\n");
                    }
                }

                // Path 4: [_ANEVirtualClient new]
                if (!virtualClient) {
                    result->triedNew = true;
                    @try {
                        virtualClient = ((id(*)(Class,SEL))objc_msgSend)(
                            virtualClientCls, @selector(new));
                    } @catch (NSException *ex) {
                        if (trace) fprintf(stderr, "ANE VC probe: +new threw: %s\n",
                                           [[ex description] UTF8String]);
                    }
                    if (trace) {
                        fprintf(stderr, "ANE VC probe: +new → %s (class: %s)\n",
                                virtualClient ? "non-nil" : "nil",
                                virtualClient ? [NSStringFromClass([virtualClient class]) UTF8String] : "N/A");
                    }
                }

                // Path 5: alloc → init (plain)
                if (!virtualClient) {
                    @try {
                        id vc = ((id(*)(Class,SEL))objc_msgSend)(virtualClientCls, @selector(alloc));
                        if (vc) {
                            vc = ((id(*)(id,SEL))objc_msgSend)(vc, @selector(init));
                            if (trace) fprintf(stderr, "ANE VC probe: alloc/init → %s\n",
                                               vc ? "non-nil" : "nil");
                            if (vc) {
                                // Try connecting
                                SEL connectSel2 = @selector(connect);
                                if ([vc respondsToSelector:connectSel2]) {
                                    ((void(*)(id,SEL))objc_msgSend)(vc, connectSel2);
                                    if (trace) fprintf(stderr, "ANE VC probe: alloc/init + connect done\n");
                                }
                                virtualClient = vc;
                            }
                        }
                    } @catch (NSException *ex) {
                        if (trace) fprintf(stderr, "ANE VC probe: alloc/init threw: %s\n",
                                           [[ex description] UTF8String]);
                    }
                }
            }

            if (!virtualClient) {
                result->stage = ANE_INTEROP_VC_STAGE_NO_VIRTUAL_CLIENT;
                if (!opts.skipEval) return;
                // For skipEval probes, continue to report class discovery even without VC
                return;
            }
            result->obtainedVirtualClient = true;
            if (ane_interop_trace_enabled()) {
                fprintf(stderr, "ANE VC probe: virtualClient acquired (%s)\n",
                        [NSStringFromClass([virtualClient class]) UTF8String]);
            }

            // --- Optionally build IOSurfaceSharedEvent + Wait/Signal events + container ---
            id sharedEventsContainer = nil;
            id ioSharedEvent = nil;
            if (opts.useSharedEvents && result->hasIOSurfaceSharedEventClass) {
                ioSharedEvent = ((id(*)(Class,SEL))objc_msgSend)(
                    ioSurfaceSharedEventCls, @selector(new));
                result->builtIOSurfaceSharedEvent = (ioSharedEvent != nil);
                if (!ioSharedEvent) {
                    result->stage = ANE_INTEROP_VC_STAGE_SHARED_EVENT_BUILD_FAILED;
                    return;
                }

                id waitEvent = nil;
                if (opts.useWaitEvent && result->hasSharedWaitEventClass) {
                    if ([sharedWaitEventCls respondsToSelector:waitEventFactorySel]) {
                        waitEvent = ((id(*)(Class,SEL,unsigned long long,id,unsigned long long))objc_msgSend)(
                            sharedWaitEventCls, waitEventFactorySel,
                            opts.waitEventValue, ioSharedEvent, opts.waitEventType);
                    } else if ([sharedWaitEventCls respondsToSelector:waitEventSimpleFactorySel]) {
                        waitEvent = ((id(*)(Class,SEL,unsigned long long,id))objc_msgSend)(
                            sharedWaitEventCls, waitEventSimpleFactorySel,
                            opts.waitEventValue, ioSharedEvent);
                    }
                    result->builtWaitEvent = (waitEvent != nil);
                    if (!waitEvent) {
                        result->stage = ANE_INTEROP_VC_STAGE_WAIT_EVENT_BUILD_FAILED;
                        return;
                    }
                }

                id signalEvent = nil;
                if (result->hasSharedSignalEventClass) {
                    signalEvent = ((id(*)(Class,SEL,unsigned long long,unsigned int,long long,id))objc_msgSend)(
                        sharedSignalEventCls, signalEventFactorySel,
                        1ULL, opts.signalSymbolIndex, 0LL, ioSharedEvent);
                    result->builtSignalEvent = (signalEvent != nil);
                }

                if (result->hasSharedEventsClass) {
                    NSArray *signalArr = signalEvent ? @[signalEvent] : @[];
                    NSArray *waitArr = waitEvent ? @[waitEvent] : @[];
                    sharedEventsContainer = ((id(*)(Class,SEL,id,id))objc_msgSend)(
                        sharedEventsCls, sharedEventsFactorySel, signalArr, waitArr);
                    result->builtSharedEventsContainer = (sharedEventsContainer != nil);
                    if (!sharedEventsContainer) {
                        result->stage = ANE_INTEROP_VC_STAGE_SHARED_EVENTS_BUILD_FAILED;
                        return;
                    }
                }
            }

            // --- Build request (reuse handle's existing request, or build with sharedEvents) ---
            id req = (__bridge id)handle->request;
            if (sharedEventsContainer && result->hasSetSharedEvents) {
                ((void(*)(id,SEL,id))objc_msgSend)(req, setSharedEventsSel, sharedEventsContainer);
                result->builtRequest = true;
            } else {
                result->builtRequest = (req != nil);
            }
            if (!req) {
                result->stage = ANE_INTEROP_VC_STAGE_REQUEST_BUILD_FAILED;
                return;
            }

            // --- Optionally load model on virtual client ---
            if (opts.loadOnVirtualClient && result->hasLoadModel) {
                NSError *loadErr = nil;
                BOOL loadOK = ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
                    virtualClient, loadModelSel, modelObj, evalOptions, 21, &loadErr);
                result->loadedOnVirtualClient = loadOK ? true : false;
                if (!loadOK && ane_interop_trace_enabled()) {
                    fprintf(stderr, "ANE VC probe: loadModel failed: %s\n",
                            loadErr ? [[loadErr description] UTF8String] : "no error");
                }
            }

            // --- Optionally map IOSurfaces ---
            if (opts.mapSurfaces && result->hasMapIOSurfaces) {
                NSError *mapErr = nil;
                BOOL mapOK = ((BOOL(*)(id,SEL,id,id,BOOL,NSError**))objc_msgSend)(
                    virtualClient, mapSurfacesSel, modelObj, req, YES, &mapErr);
                result->mappedSurfaces = mapOK ? true : false;
                if (!mapOK) {
                    if (ane_interop_trace_enabled()) {
                        fprintf(stderr, "ANE VC probe: mapIOSurfaces failed: %s\n",
                                mapErr ? [[mapErr description] UTF8String] : "no error");
                    }
                    result->stage = ANE_INTEROP_VC_STAGE_MAP_SURFACES_FAILED;
                    return;
                }
            }

            if (opts.skipEval) {
                // Construction-only probe; report furthest stage reached
                if (result->builtSharedEventsContainer) {
                    result->stage = ANE_INTEROP_VC_STAGE_SHARED_EVENTS_BUILD_FAILED + 1;
                } else if (result->obtainedVirtualClient) {
                    result->stage = ANE_INTEROP_VC_STAGE_NO_VIRTUAL_CLIENT + 1;
                }
                return;
            }

            // --- Standard eval on VirtualClient ---
            if (result->hasStandardEvaluate) {
                NSError *evalErr = nil;
                BOOL evalOK = ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    virtualClient, standardEvalSel, modelObj, evalOptions, req, 21, &evalErr);
                result->standardEvalSucceeded = evalOK ? true : false;
                if (evalOK) {
                    result->stage = ANE_INTEROP_VC_STAGE_EVAL_SUCCEEDED;
                    if (ane_interop_trace_enabled()) {
                        fprintf(stderr, "ANE VC probe: standard eval succeeded\n");
                    }
                } else {
                    result->stage = ANE_INTEROP_VC_STAGE_EVAL_FAILED;
                    if (ane_interop_trace_enabled()) {
                        fprintf(stderr, "ANE VC probe: standard eval failed: %s\n",
                                evalErr ? [[evalErr description] UTF8String] : "no error");
                    }
                }
            } else {
                result->stage = ANE_INTEROP_VC_STAGE_EVAL_FAILED;
            }

            // --- CompletionEvent eval (doEvaluateWithModel:...completionEvent:...) ---
            if (opts.useCompletionEvent && result->hasDoEvaluateCompletionEvent) {
                dispatch_semaphore_t sem = dispatch_semaphore_create(0);
                __block BOOL ceEvalOK = NO;
                id completionEvent = ioSharedEvent; // may be nil — probe nil first
                NSError *ceErr = nil;
                ceEvalOK = ((BOOL(*)(id,SEL,id,id,id,unsigned int,id,NSError**))objc_msgSend)(
                    virtualClient, doEvalCompletionEventSel,
                    modelObj, evalOptions, req, 21, completionEvent, &ceErr);
                // If the call itself returns synchronously, check result
                if (ceEvalOK) {
                    result->completionEventEvalSucceeded = true;
                    result->stage = ANE_INTEROP_VC_STAGE_COMPLETION_EVENT_EVAL_SUCCEEDED;
                    if (ane_interop_trace_enabled()) {
                        fprintf(stderr, "ANE VC probe: completionEvent eval succeeded\n");
                    }
                } else {
                    result->stage = ANE_INTEROP_VC_STAGE_COMPLETION_EVENT_EVAL_FAILED;
                    if (ane_interop_trace_enabled()) {
                        fprintf(stderr, "ANE VC probe: completionEvent eval failed: %s\n",
                                ceErr ? [[ceErr description] UTF8String] : "no error");
                    }
                }
                (void)sem; // semaphore available if async path needed in future
            }

            // --- CompletionHandler on request ---
            if (opts.useCompletionHandler && result->hasSetCompletionHandler) {
                dispatch_semaphore_t handlerSem = dispatch_semaphore_create(0);
                __block BOOL handlerFired = NO;
                // Set block on request
                ((void(*)(id,SEL,id))objc_msgSend)(req, setCompletionHandlerSel,
                    ^{
                        handlerFired = YES;
                        dispatch_semaphore_signal(handlerSem);
                    });
                // Re-eval via standard path to trigger handler
                if (result->hasStandardEvaluate) {
                    NSError *chErr = nil;
                    ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                        virtualClient, standardEvalSel, modelObj, evalOptions, req, 21, &chErr);
                }
                long waited = dispatch_semaphore_wait(handlerSem,
                    dispatch_time(DISPATCH_TIME_NOW, 5LL * NSEC_PER_SEC));
                result->completionHandlerFired = (waited == 0 && handlerFired) ? true : false;
                if (result->completionHandlerFired) {
                    result->stage = ANE_INTEROP_VC_STAGE_COMPLETION_HANDLER_EVAL_SUCCEEDED;
                    if (ane_interop_trace_enabled()) {
                        fprintf(stderr, "ANE VC probe: completionHandler fired\n");
                    }
                } else {
                    result->stage = ANE_INTEROP_VC_STAGE_COMPLETION_HANDLER_EVAL_FAILED;
                    if (ane_interop_trace_enabled()) {
                        fprintf(stderr, "ANE VC probe: completionHandler did not fire (timeout=%s)\n",
                                waited != 0 ? "yes" : "no");
                    }
                }
            }
        } @catch (NSException *exception) {
            if (ane_interop_trace_enabled()) {
                fprintf(stderr, "ANE VC probe exception: %s\n",
                        [[exception description] UTF8String]);
            }
            result->stage = ANE_INTEROP_VC_STAGE_EXCEPTION;
        }
    }
}

// --- Code Signing Identity Probe ---

void ane_interop_probe_code_signing(ANEInteropCodeSigningProbeResult *result) {
    @autoreleasepool {
        if (!result) return;
        memset(result, 0, sizeof(*result));

        ane_interop_init();
        Class vcCls = NSClassFromString(@"_ANEVirtualClient");
        if (!vcCls) return;

        bool trace = ane_interop_trace_enabled();

        SEL getSel = NSSelectorFromString(@"getCodeSigningIdentity");
        SEL setSel = NSSelectorFromString(@"setCodeSigningIdentity:");
        result->hasGetCodeSigningIdentity = [vcCls respondsToSelector:getSel];
        result->hasSetCodeSigningIdentity = [vcCls respondsToSelector:setSel];

        if (trace) {
            fprintf(stderr, "ANE CS probe: hasGet=%d hasSet=%d\n",
                    result->hasGetCodeSigningIdentity, result->hasSetCodeSigningIdentity);
        }

        if (result->hasGetCodeSigningIdentity) {
            @try {
                id identity = ((id(*)(Class,SEL))objc_msgSend)(vcCls, getSel);
                if (identity) {
                    result->gotIdentityString = true;
                    const char *str = NULL;
                    if ([identity isKindOfClass:[NSString class]]) {
                        str = [(NSString *)identity UTF8String];
                    } else {
                        str = [[NSString stringWithFormat:@"%@", identity] UTF8String];
                    }
                    if (str) {
                        strncpy(result->identityString, str, sizeof(result->identityString) - 1);
                    }
                    if (trace) {
                        fprintf(stderr, "ANE CS probe: identity = '%s' (class: %s)\n",
                                result->identityString,
                                [NSStringFromClass([identity class]) UTF8String]);
                    }
                } else {
                    if (trace) fprintf(stderr, "ANE CS probe: getCodeSigningIdentity → nil\n");
                }
            } @catch (NSException *ex) {
                if (trace) fprintf(stderr, "ANE CS probe: get threw: %s\n",
                                   [[ex description] UTF8String]);
            }
        }

        if (result->hasSetCodeSigningIdentity) {
            // setCodeSigningIdentity: crashes with __setObject:forKey: if passed a plain
            // string — the internal implementation treats the class-level identity store
            // as a dictionary keyed by identity. Try NSData (SecCodeCopyGuestWithAttributes
            // returns kSecCodeInfoIdentifier as a string, but the setter may need a different
            // type). Wrap in @try to catch any crash.
            NSArray *identities = @[
                @"com.apple.coreml",
                @"com.apple.appleNeuralEngine",
                @"*",
            ];
            for (NSString *identity in identities) {
                @try {
                    if (trace) {
                        fprintf(stderr, "ANE CS probe: trying setCodeSigningIdentity: '%s'\n",
                                [identity UTF8String]);
                    }
                    ((void(*)(Class,SEL,id))objc_msgSend)(vcCls, setSel, identity);
                    result->setIdentityBeforeInstantiation = true;

                    SEL sharedConnSel = @selector(sharedConnection);
                    if ([vcCls respondsToSelector:sharedConnSel]) {
                        id vc = ((id(*)(Class,SEL))objc_msgSend)(vcCls, sharedConnSel);
                        if (vc) {
                            result->instantiationSucceededAfterSet = true;
                            if (trace) {
                                fprintf(stderr, "ANE CS probe: +sharedConnection after set '%s' → non-nil!\n",
                                        [identity UTF8String]);
                            }
                            break;
                        } else if (trace) {
                            fprintf(stderr, "ANE CS probe: +sharedConnection after set '%s' → nil\n",
                                    [identity UTF8String]);
                        }
                    }
                } @catch (NSException *ex) {
                    if (trace) fprintf(stderr, "ANE CS probe: set '%s' threw: %s\n",
                                       [identity UTF8String], [[ex description] UTF8String]);
                }
            }
        }
    }
}

// --- Standard Eval CompletionHandler Probe ---

void ane_interop_probe_standard_completion_handler(
    ANEHandle *handle,
    bool useMetalSharedEvent,
    ANEInteropStandardCompletionProbeResult *result)
{
    @autoreleasepool {
        if (!result) return;
        memset(result, 0, sizeof(*result));
        if (!handle) return;

        bool trace = ane_interop_trace_enabled();
        id req = (__bridge id)handle->request;
        if (!req) return;

        SEL setCompletionHandlerSel = @selector(setCompletionHandler:);
        SEL setSharedEventsSel = @selector(setSharedEvents:);
        result->requestHasCompletionHandler = [req respondsToSelector:setCompletionHandlerSel];
        result->requestHasSharedEvents = [req respondsToSelector:setSharedEventsSel];

        if (!result->requestHasCompletionHandler) {
            if (trace) fprintf(stderr, "ANE std-completion: no setCompletionHandler:\n");
            return;
        }

        id metalSharedEvent = nil;
        if (useMetalSharedEvent) {
            Class sharedEventsCls = NSClassFromString(@"_ANESharedEvents");
            Class sharedSignalEventCls = NSClassFromString(@"_ANESharedSignalEvent");
            SEL sharedEventsFactorySel = NSSelectorFromString(@"sharedEventsWithSignalEvents:waitEvents:");
            SEL signalEventFactorySel = NSSelectorFromString(@"signalEventWithValue:symbolIndex:eventType:sharedEvent:");

            metalSharedEvent = ane_interop_create_metal_shared_event(
                &result->metalDeviceCreated,
                &result->builtMetalSharedEvent
            );
            result->eventValueBefore = ane_interop_shared_event_value(metalSharedEvent);

            if (result->requestHasSharedEvents &&
                metalSharedEvent &&
                sharedEventsCls &&
                sharedSignalEventCls &&
                [sharedEventsCls respondsToSelector:sharedEventsFactorySel] &&
                [sharedSignalEventCls respondsToSelector:signalEventFactorySel]) {
                id signalEvent = ((id(*)(Class,SEL,unsigned long long,unsigned int,long long,id))objc_msgSend)(
                    sharedSignalEventCls, signalEventFactorySel, 1ULL, 0U, 0LL, metalSharedEvent);
                result->builtSignalEvent = (signalEvent != nil);
                if (signalEvent) {
                    id sharedEventsContainer = ((id(*)(Class,SEL,id,id))objc_msgSend)(
                        sharedEventsCls, sharedEventsFactorySel, @[signalEvent], @[]);
                    result->builtSharedEventsContainer = (sharedEventsContainer != nil);
                    if (sharedEventsContainer) {
                        @try {
                            ((void(*)(id,SEL,id))objc_msgSend)(req, setSharedEventsSel, sharedEventsContainer);
                            result->sharedEventsAttached = true;
                            if (trace) fprintf(stderr, "ANE std-completion: shared events attached\n");
                        } @catch (NSException *ex) {
                            if (trace) fprintf(stderr, "ANE std-completion: setSharedEvents threw: %s\n",
                                               [[ex description] UTF8String]);
                        }
                    }
                }
            }
        }

        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        __block BOOL handlerFired = NO;
        @try {
            ((void(*)(id,SEL,id))objc_msgSend)(req, setCompletionHandlerSel,
                ^{
                    handlerFired = YES;
                    dispatch_semaphore_signal(sem);
                });
            result->completionHandlerSet = true;
            if (trace) fprintf(stderr, "ANE std-completion: handler set\n");
        } @catch (NSException *ex) {
            if (trace) fprintf(stderr, "ANE std-completion: setCompletionHandler threw: %s\n",
                               [[ex description] UTF8String]);
            return;
        }

        id mdl = (__bridge id)handle->model;
        NSDictionary *options = handle->evalOptions
            ? (__bridge NSDictionary *)handle->evalOptions : @{};
        NSError *e = nil;

        uint64_t start = mach_absolute_time();
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            mdl, @selector(evaluateWithQoS:options:request:error:), 21, options, req, &e);
        uint64_t end = mach_absolute_time();

        result->evalSucceeded = ok ? true : false;
        if (trace) {
            fprintf(stderr, "ANE std-completion: eval %s\n", ok ? "succeeded" : "failed");
        }

        // Also try _ANEClient eval path
        if (!handlerFired && handle->client && handle->clientModel) {
            id client = (__bridge id)handle->client;
            id modelObj = (__bridge id)handle->clientModel;
            SEL evalSel = @selector(evaluateWithModel:options:request:qos:error:);
            if ([client respondsToSelector:evalSel]) {
                NSError *e2 = nil;
                ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(
                    client, evalSel, modelObj, options, req, 21, &e2);
                if (trace) fprintf(stderr, "ANE std-completion: tried _ANEClient eval too\n");
            }
        }

        long waited = dispatch_semaphore_wait(sem,
            dispatch_time(DISPATCH_TIME_NOW, 5LL * NSEC_PER_SEC));
        result->completionHandlerFired = (waited == 0 && handlerFired) ? true : false;
        result->eventValueAfter = ane_interop_shared_event_value(metalSharedEvent);
        result->eventValueAdvanced = result->eventValueAfter > result->eventValueBefore;

        mach_timebase_info_data_t tbi;
        mach_timebase_info(&tbi);
        double ns = (double)(end - start) * (double)tbi.numer / (double)tbi.denom;
        result->evalTimeMS = ns / 1e6;

        if (trace) {
            fprintf(stderr, "ANE std-completion: fired=%s sharedEventAdvanced=%s eval=%.3fms\n",
                    result->completionHandlerFired ? "YES" : "NO",
                    result->eventValueAdvanced ? "YES" : "NO",
                    result->evalTimeMS);
        }

        @try {
            ((void(*)(id,SEL,id))objc_msgSend)(req, setCompletionHandlerSel, (id)nil);
        } @catch (NSException *ex) {
            // ignore cleanup failure
        }
        if (result->sharedEventsAttached) {
            @try {
                ((void(*)(id,SEL,id))objc_msgSend)(req, setSharedEventsSel, (id)nil);
            } @catch (NSException *ex) {
                // ignore cleanup failure
            }
        }
    }
}

// --- Real-time eval path probe ---

bool ane_interop_runtime_has_realtime_eval(ANEHandle *handle) {
    if (!handle || !handle->client) return false;
    id client = (__bridge id)handle->client;
    return [client respondsToSelector:@selector(evaluateRealTimeWithModel:options:request:error:)]
        && [client respondsToSelector:@selector(beginRealTimeTask)]
        && [client respondsToSelector:@selector(endRealTimeTask)]
        && [client respondsToSelector:@selector(loadRealTimeModel:options:qos:error:)]
        && [client respondsToSelector:@selector(unloadRealTimeModel:options:qos:error:)];
}

void ane_interop_probe_realtime_eval(ANEHandle *handle,
                                      int nIters,
                                      ANEInteropRealTimeProbeResult *result)
{
    @autoreleasepool {
        if (!result) return;
        memset(result, 0, sizeof(*result));
        if (!handle || nIters <= 0) return;
        if (!handle->client || !handle->clientModel) return;

        bool trace = ane_interop_trace_enabled();
        id client = (__bridge id)handle->client;
        id modelObj = (__bridge id)handle->clientModel;
        id mdl = (__bridge id)handle->model;
        id req = (__bridge id)handle->request;
        NSDictionary *options = handle->evalOptions
            ? (__bridge NSDictionary *)handle->evalOptions : @{};

        // --- Selector discovery ---
        result->hasBeginRealTimeTask =
            [client respondsToSelector:@selector(beginRealTimeTask)];
        result->hasEndRealTimeTask =
            [client respondsToSelector:@selector(endRealTimeTask)];
        result->hasLoadRealTimeModel =
            [client respondsToSelector:@selector(loadRealTimeModel:options:qos:error:)];
        result->hasUnloadRealTimeModel =
            [client respondsToSelector:@selector(unloadRealTimeModel:options:qos:error:)];
        result->hasEvaluateRealTime =
            [client respondsToSelector:@selector(evaluateRealTimeWithModel:options:request:error:)];

        if (trace) {
            fprintf(stderr, "ANE rt-probe: beginRT=%d endRT=%d loadRT=%d unloadRT=%d evalRT=%d\n",
                    result->hasBeginRealTimeTask, result->hasEndRealTimeTask,
                    result->hasLoadRealTimeModel, result->hasUnloadRealTimeModel,
                    result->hasEvaluateRealTime);
        }

        mach_timebase_info_data_t tbi;
        mach_timebase_info(&tbi);

        // --- Standard eval benchmark (InMemoryModel path, warmup + timed) ---
        @try {
            // Warmup: 3 evals via _ANEInMemoryModel.evaluateWithQoS:
            for (int i = 0; i < 3; i++) {
                NSError *we = nil;
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl, @selector(evaluateWithQoS:options:request:error:),
                    21, options, req, &we);
            }

            // Timed: nIters evals via InMemoryModel (same path as decode loop)
            uint64_t stdStart = mach_absolute_time();
            int stdOK = 0;
            for (int i = 0; i < nIters; i++) {
                NSError *e = nil;
                BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    mdl, @selector(evaluateWithQoS:options:request:error:),
                    21, options, req, &e);
                if (ok) stdOK++;
                else if (trace) {
                    fprintf(stderr, "ANE rt-probe: std eval %d failed: %s\n",
                            i, e ? [[e description] UTF8String] : "no error");
                    break;
                }
            }
            uint64_t stdEnd = mach_absolute_time();

            result->standardEvalsCompleted = stdOK;
            result->standardEvalSucceeded = stdOK > 0;
            double stdNS = (double)(stdEnd - stdStart) * (double)tbi.numer / (double)tbi.denom;
            result->standardTotalMS = stdNS / 1e6;
            if (stdOK > 0) {
                result->standardPerEvalMS = result->standardTotalMS / (double)stdOK;
            }

            if (trace) {
                fprintf(stderr, "ANE rt-probe: std %d/%d evals, total=%.3fms, per=%.3fms\n",
                        stdOK, nIters, result->standardTotalMS, result->standardPerEvalMS);
            }
        } @catch (NSException *ex) {
            if (trace) fprintf(stderr, "ANE rt-probe: std eval exception: %s\n",
                               [[ex description] UTF8String]);
        }

        // --- Real-time eval benchmark ---
        if (!result->hasBeginRealTimeTask || !result->hasLoadRealTimeModel ||
            !result->hasEvaluateRealTime) {
            if (trace) fprintf(stderr, "ANE rt-probe: missing real-time selectors, skipping\n");
            return;
        }

        @try {
            // Begin real-time task
            ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(beginRealTimeTask));

            // Load real-time model
            NSError *loadErr = nil;
            BOOL loaded = ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
                client, @selector(loadRealTimeModel:options:qos:error:),
                modelObj, options, 21, &loadErr);
            result->realtimeLoadSucceeded = loaded ? true : false;

            if (!loaded) {
                if (trace) fprintf(stderr, "ANE rt-probe: loadRealTimeModel failed: %s\n",
                                   loadErr ? [[loadErr description] UTF8String] : "no error");
                // End task and return
                if (result->hasEndRealTimeTask) {
                    ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(endRealTimeTask));
                }
                return;
            }

            if (trace) fprintf(stderr, "ANE rt-probe: loadRealTimeModel succeeded\n");

            // Warmup: 3 evals
            for (int i = 0; i < 3; i++) {
                NSError *we = nil;
                ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                    client, @selector(evaluateRealTimeWithModel:options:request:error:),
                    modelObj, options, req, &we);
            }

            // Timed: nIters evals
            uint64_t rtStart = mach_absolute_time();
            int rtOK = 0;
            for (int i = 0; i < nIters; i++) {
                NSError *e = nil;
                BOOL ok = ((BOOL(*)(id,SEL,id,id,id,NSError**))objc_msgSend)(
                    client, @selector(evaluateRealTimeWithModel:options:request:error:),
                    modelObj, options, req, &e);
                if (ok) rtOK++;
                else if (trace) {
                    fprintf(stderr, "ANE rt-probe: rt eval %d failed: %s\n",
                            i, e ? [[e description] UTF8String] : "no error");
                    break;
                }
            }
            uint64_t rtEnd = mach_absolute_time();

            result->realtimeEvalsCompleted = rtOK;
            result->realtimeEvalSucceeded = rtOK > 0;
            double rtNS = (double)(rtEnd - rtStart) * (double)tbi.numer / (double)tbi.denom;
            result->realtimeTotalMS = rtNS / 1e6;
            if (rtOK > 0) {
                result->realtimePerEvalMS = result->realtimeTotalMS / (double)rtOK;
            }

            // Compute savings
            if (result->standardPerEvalMS > 0 && result->realtimePerEvalMS > 0) {
                result->savedPerEvalMS = result->standardPerEvalMS - result->realtimePerEvalMS;
                result->savedPercent =
                    (result->savedPerEvalMS / result->standardPerEvalMS) * 100.0;
            }

            if (trace) {
                fprintf(stderr, "ANE rt-probe: rt %d/%d evals, total=%.3fms, per=%.3fms\n",
                        rtOK, nIters, result->realtimeTotalMS, result->realtimePerEvalMS);
                fprintf(stderr, "ANE rt-probe: saved=%.3fms/eval (%.1f%%)\n",
                        result->savedPerEvalMS, result->savedPercent);
            }

            // Unload real-time model
            if (result->hasUnloadRealTimeModel) {
                NSError *unloadErr = nil;
                ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(
                    client, @selector(unloadRealTimeModel:options:qos:error:),
                    modelObj, options, 21, &unloadErr);
            }
            // End real-time task
            if (result->hasEndRealTimeTask) {
                ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(endRealTimeTask));
            }
        } @catch (NSException *ex) {
            if (trace) fprintf(stderr, "ANE rt-probe: exception: %s\n",
                               [[ex description] UTF8String]);
            // Best-effort cleanup
            @try {
                if (result->hasEndRealTimeTask) {
                    ((BOOL(*)(id,SEL))objc_msgSend)(client, @selector(endRealTimeTask));
                }
            } @catch (NSException *ignored) {}
        }
    }
}

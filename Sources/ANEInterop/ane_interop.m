#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <dispatch/dispatch.h>
#import <dlfcn.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <stdint.h>

#include "ane_interop.h"

struct ANEHandle {
    void *model;               // CFBridgingRetain'd _ANEInMemoryModel
    IOSurfaceRef *ioInputs;
    IOSurfaceRef *ioOutputs;
    void *request;             // CFBridgingRetain'd _ANERequest
    void *tmpDir;              // CFBridgingRetain'd NSString
    int nInputs, nOutputs;
    size_t *inputBytes;
    size_t *outputBytes;
    bool liveHandleCounted;
};

static Class g_ANEDesc = nil, g_ANEInMem = nil, g_ANEReq = nil, g_ANEIO = nil;
static bool g_ane_loaded = false;
static dispatch_once_t g_ane_once;
static int g_compile_count = 0;
static int g_last_compile_error = ANE_INTEROP_COMPILE_ERROR_NONE;
static int g_force_eval_failure = 0;
static int g_live_handle_count = 0;

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

        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        if (![hx isKindOfClass:[NSString class]]) {
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            return NULL;
        }
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
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
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            fprintf(stderr, "ANE compile failed: %s\n", e ? [[e description] UTF8String] : "no error");
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            ane_interop_remove_tmpdir(td);
            return NULL;
        }
        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
            fprintf(stderr, "ANE load failed: %s\n", e ? [[e description] UTF8String] : "no error");
            ane_interop_set_compile_error(ANE_INTEROP_COMPILE_ERROR_COMPILER_FAILURE);
            ane_interop_remove_tmpdir(td);
            return NULL;
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
        h->tmpDir = (void *)CFBridgingRetain(td);
        h->nInputs = nInputs;
        h->nOutputs = nOutputs;

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

        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ANEReq, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            wIns, iIdx, wOuts, oIdx, nil, nil, @0);
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
    BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
        mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    if (!ok) {
        fprintf(stderr, "ANE eval failed: %s\n", e ? [[e description] UTF8String] : "no error");
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
    if (handle->request) CFRelease(handle->request);
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

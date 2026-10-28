// Metal backend: NCCL stand-in. Single-device only; every collective or
// communicator call aborts with a clear message.
#pragma once
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include "cuda_runtime.h"

typedef enum { ncclSuccess = 0, ncclInvalidUsage = 5 } ncclResult_t;
typedef enum { ncclFloat = 7, ncclBfloat16 = 9 } ncclDataType_t;
typedef enum { ncclSum = 0, ncclProd = 1, ncclMax = 2, ncclMin = 3, ncclAvg = 4 } ncclRedOp_t;
typedef struct ncclComm* ncclComm_t;
typedef struct { char internal[128]; } ncclUniqueId;

[[noreturn]] static inline void puf_metal_no_nccl(const char* fn) {
    fprintf(stderr, "metal: multi-GPU (NCCL %s) is unsupported on the Metal backend; "
        "run with world_size=1\n", fn);
    abort();
}
static inline ncclResult_t ncclGetUniqueId(ncclUniqueId*) { puf_metal_no_nccl("ncclGetUniqueId"); }
static inline ncclResult_t ncclCommInitRank(ncclComm_t*, int, ncclUniqueId, int) {
    puf_metal_no_nccl("ncclCommInitRank");
}
static inline ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    if (comm != NULL) puf_metal_no_nccl("ncclCommDestroy");
    return ncclSuccess;
}
static inline ncclResult_t ncclAllReduce(const void*, void*, size_t, ncclDataType_t,
        ncclRedOp_t, ncclComm_t, cudaStream_t) {
    puf_metal_no_nccl("ncclAllReduce");
}

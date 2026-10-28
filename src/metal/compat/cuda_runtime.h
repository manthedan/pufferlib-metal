// Metal backend: host-side stand-in for <cuda_runtime.h>.
//
// Only the subset of the CUDA runtime that src/pufferl.cu, src/algo.cu and
// src/ocean.cu use from host code is declared here. Implementations live in
// src/metal/runtime.mm and map onto Apple-silicon unified memory + Metal 4
// command queues. Device code in the upstream .cu files is compiled out under
// PUF_METAL; kernels are reached through PUF_LAUNCH (src/metal/launch.h).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/types.h>

#ifndef PUF_METAL
#define PUF_METAL 1
#endif

// CUDA execution-space qualifiers. Only __host__ __device__ helpers survive
// the PUF_METAL guards, so these just need to vanish in host code.
#define __host__
#define __device__
#define __global__
#define __shared__ static
#define __forceinline__ inline
#define __launch_bounds__(...)

// glibc defines ulong in <sys/types.h>; macOS does not. It must equal uint64_t
// (unsigned long long on macOS): algo.cu passes uint64_t* where ulong* is
// expected, which is the same type on Linux.
typedef uint64_t ulong;

// nvcc ships host overloads of min/max (algo.cu uses them on long/int).
template <typename T> static inline T min(T a, T b) { return b < a ? b : a; }
template <typename T> static inline T max(T a, T b) { return a < b ? b : a; }

struct dim3 {
    unsigned int x, y, z;
    constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
};

typedef enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorNoDevice = 100,
    cudaErrorNotSupported = 801,
    cudaErrorUnknown = 999,
} cudaError_t;

typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
} cudaMemcpyKind;

typedef enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2,
} cudaStreamCaptureMode;

// library_types.h
typedef enum cudaDataType_t {
    CUDA_R_32F = 0,
    CUDA_R_16F = 2,
    CUDA_R_16BF = 14,
} cudaDataType_t;
typedef cudaDataType_t cudaDataType;

#define cudaStreamDefault 0x00
#define cudaStreamNonBlocking 0x01
#define cudaEventDefault 0x00
#define cudaEventBlockingSync 0x01
#define cudaEventDisableTiming 0x02
#define cudaHostAllocDefault 0x00
#define cudaHostAllocPortable 0x01

typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
typedef struct CUgraph_st* cudaGraph_t;
typedef struct CUgraphExec_st* cudaGraphExec_t;

// Memory. All allocations are page-aligned shared (unified) memory that is
// both CPU- and GPU-visible; cudaHostAlloc and cudaMalloc are equivalent.
cudaError_t cudaMalloc(void** ptr, size_t size);
cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags);
cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaFree(void* ptr);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n,
    cudaMemcpyKind kind, cudaStream_t stream = 0);
cudaError_t cudaMemset(void* dst, int value, size_t n);
cudaError_t cudaMemsetAsync(void* dst, int value, size_t n, cudaStream_t stream = 0);
cudaError_t cudaMemGetInfo(size_t* free_bytes, size_t* total_bytes);

template <typename T>
static inline cudaError_t cudaMalloc(T** ptr, size_t size) {
    return cudaMalloc((void**)ptr, size);
}
template <typename T>
static inline cudaError_t cudaHostAlloc(T** ptr, size_t size, unsigned int flags) {
    return cudaHostAlloc((void**)ptr, size, flags);
}

// Streams / events / device.
cudaError_t cudaStreamCreate(cudaStream_t* stream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
    unsigned int flags = 0);
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetLastError(void);
const char* cudaGetErrorString(cudaError_t err);

// CUDA graphs are not supported on Metal; PUF_METAL forces cudagraphs off.
// These abort if reached.
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode);
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph,
    unsigned long long flags = 0);
cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream);
cudaError_t cudaGraphDestroy(cudaGraph_t graph);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec);

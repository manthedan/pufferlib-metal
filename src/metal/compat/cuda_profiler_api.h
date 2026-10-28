// Metal backend: <cuda_profiler_api.h> no-ops (use Instruments / Metal capture).
#pragma once
#include "cuda_runtime.h"
static inline cudaError_t cudaProfilerStart(void) { return cudaSuccess; }
static inline cudaError_t cudaProfilerStop(void) { return cudaSuccess; }

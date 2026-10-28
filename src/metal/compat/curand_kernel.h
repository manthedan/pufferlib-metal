// Metal backend: curand_kernel.h stand-in. Device RNG functions live in the
// Metal shaders; host code only needs the state's size/layout for cudaMalloc.
#pragma once
#include <cstdint>

// Mirrors CUDA's curandStatePhilox4_32_10 (uint4 ctr, uint4 output, uint2 key,
// uint STATE, int boxmuller_flag, int boxmuller_flag_double,
// float boxmuller_extra, double boxmuller_extra_double): 64 bytes, 16-aligned.
struct alignas(16) curandStatePhilox4_32_10 {
    uint32_t ctr[4];
    uint32_t output[4];
    uint32_t key[2];
    uint32_t STATE;
    int boxmuller_flag;
    int boxmuller_flag_double;
    float boxmuller_extra;
    double boxmuller_extra_double;
};
typedef struct curandStatePhilox4_32_10 curandStatePhilox4_32_10_t;
static_assert(sizeof(curandStatePhilox4_32_10_t) == 64, "philox state layout");

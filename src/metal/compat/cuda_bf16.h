// Metal backend: host-side stand-in for <cuda_bf16.h>.
// The bf16 build (no PRECISION_FLOAT) uses this as precision_t on the host:
// envs whose obs_t is precision_t (breakout, g2048, memory, robot_arm) write
// observations through it, and host code moves raw bf16 tensors. Device
// arithmetic lives in the MSL shaders (shaders/common.h), with the same
// conversions: round to nearest even, NaN quieted as (bits >> 16) | 0x40 (CUDA
// returns 0x7FFF for every NaN; see shaders/common.h for why Metal does not).
// Like CUDA's host __nv_bfloat16, it converts implicitly from and to float.
#pragma once
#include <cstdint>
#include <cstring>

static inline uint16_t puf_float_to_bf16_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (uint16_t)((u >> 16) | 0x40);  // quiet NaN
    }
    u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest even
    return (uint16_t)(u >> 16);
}

static inline float puf_bf16_bits_to_float(uint16_t x) {
    uint32_t u = (uint32_t)x << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

struct __nv_bfloat16 {
    uint16_t x;
    __nv_bfloat16() = default;  // trivial: calloc / memcpy / cudaMemset-able
    __nv_bfloat16(float f) : x(puf_float_to_bf16_bits(f)) {}
    operator float() const { return puf_bf16_bits_to_float(x); }
    __nv_bfloat16& operator+=(float v) { return *this = (float)*this + v; }
    __nv_bfloat16& operator-=(float v) { return *this = (float)*this - v; }
    __nv_bfloat16& operator*=(float v) { return *this = (float)*this * v; }
    __nv_bfloat16& operator/=(float v) { return *this = (float)*this / v; }
    static __nv_bfloat16 from_bits(uint16_t bits) {
        __nv_bfloat16 r;
        r.x = bits;
        return r;
    }
};
static_assert(sizeof(__nv_bfloat16) == 2, "bf16 storage");

static inline float __bfloat162float(__nv_bfloat16 v) { return puf_bf16_bits_to_float(v.x); }

static inline __nv_bfloat16 __float2bfloat16(float f) {
    return __nv_bfloat16::from_bits(puf_float_to_bf16_bits(f));
}

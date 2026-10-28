// Fixed-point (2^24) int64 embedding-gradient accumulation shared by the
// craftax and nmmo3 encoders. Apple GPUs have no 64-bit atomic add, so each
// int64 is two 32-bit words updated with an explicit carry derived from the
// low word's fetch_add result. Modular addition stays associative and exact,
// so sums are bit-identical to CUDA's 64-bit integer atomics.
#pragma once
#include "common.h"

#define PUF_FXP 16777216.0f

template <typename A>
static inline void fxp_atomic_add64(A lo, A hi, ulong v) {
    uint vlo = uint(v), vhi = uint(v >> 32);
    uint old = atomic_fetch_add_explicit(lo, vlo, memory_order_relaxed);
    uint carry = (old + vlo < old) ? 1u : 0u;
    if (vhi + carry)
        atomic_fetch_add_explicit(hi, vhi + carry, memory_order_relaxed);
}

// __float2ll_rn(x * 2^24): round half to even.
static inline ulong fxp_quantize(float x) {
    return ulong(long(rint(x * PUF_FXP)));
}

// int64 -> float rounds once and the 2^-24 scale is exact, so this equals
// CUDA's (float)((double)v * 2^-24) for |v| < 2^53.
static inline float fxp_to_float(long v) {
    return float(v) * (1.0f / PUF_FXP);
}

// Threadgroup histogram helpers: N entries as (lo, hi) uint pairs.
template <int N>
static inline void fxp_tg_zero(threadgroup atomic_uint* lo, threadgroup atomic_uint* hi,
        uint tid, uint nthreads) {
    for (uint i = tid; i < N; i += nthreads) {
        atomic_store_explicit(&lo[i], 0u, memory_order_relaxed);
        atomic_store_explicit(&hi[i], 0u, memory_order_relaxed);
    }
}

// Adds every nonzero threadgroup entry to the device int64 array `dst`
// (viewed as little-endian uint pairs), like CUDA's final atomicAdd pass.
template <int N>
static inline void fxp_tg_flush(device long* dst, threadgroup atomic_uint* lo,
        threadgroup atomic_uint* hi, uint tid, uint nthreads) {
    device atomic_uint* out = (device atomic_uint*)dst;
    for (uint i = tid; i < N; i += nthreads) {
        ulong v = ulong(atomic_load_explicit(&lo[i], memory_order_relaxed))
            | (ulong(atomic_load_explicit(&hi[i], memory_order_relaxed)) << 32);
        if (v != 0) fxp_atomic_add64(&out[2 * i], &out[2 * i + 1], v);
    }
}

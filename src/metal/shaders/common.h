// Shared MSL definitions for the PufferLib Metal backend.
// Kernels mirror their CUDA counterparts in src/pufferl.cu / src/algo.cu:
// arguments arrive as one struct at [[buffer(0)]] whose layout matches the
// host-side struct in src/metal/kernels.h (device pointers are 8 bytes and
// already translated to GPU addresses), and CUDA's blockIdx/blockDim/threadIdx
// map onto threadgroup position/size/thread position.
#pragma once
#include <metal_stdlib>
using namespace metal;

// precision_t mirrors src/pufferl.cu: float with PRECISION_FLOAT, otherwise
// bf16 storage with fp32 arithmetic. The bf16 conversions are bit-exact with
// CUDA's __float2bfloat16 / __bfloat162float: round to nearest even, float
// denormals rounded (not flushed, which MSL's bfloat(float) does), NaN
// quieted as (bits >> 16) | 0x40. Only NaN payloads differ from CUDA, which
// returns the canonical 0x7FFF for every NaN (golden cast_bf16_probe). Any
// formulation whose NaN result ignores the payload (a 0x7FFF select, before or
// after rounding) is compiled by the Metal compiler into its native
// float->bfloat conversion, which flushes denormals and returns 0x7FC0 for
// NaN; tests/metal/test_bf16.mm catches that.
// Explicit bf16 <-> f32 helpers, available in both builds (GEMM kernels are
// compiled for both storage types).
static inline float puf_bf16_to_f32(bfloat v) {
    return as_type<float>(uint(as_type<ushort>(v)) << 16);
}
static inline ushort puf_f32_to_bf16_bits(float f) {
    uint u = as_type<uint>(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return ushort((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return ushort(u >> 16);
}
static inline bfloat puf_f32_to_bf16(float f) { return as_type<bfloat>(puf_f32_to_bf16_bits(f)); }
// Storage-typed element access with fp32 arithmetic (identity for float).
static inline float puf_ld(device const float* p, long i) { return p[i]; }
static inline float puf_ld(device const bfloat* p, long i) { return puf_bf16_to_f32(p[i]); }
static inline void puf_st(device float* p, long i, float v) { p[i] = v; }
static inline void puf_st(device bfloat* p, long i, float v) { p[i] = puf_f32_to_bf16(v); }
static inline float puf_round(float, float v) { return v; }
static inline float puf_round(bfloat, float v) { return puf_bf16_to_f32(puf_f32_to_bf16(v)); }

#ifdef PRECISION_FLOAT
typedef float precision_t;
#define PUF_PREC_BYTES 4
#define to_float(x) (x)
#define from_float(x) (x)
static inline float round_p(float x) { return x; }
#else
typedef bfloat precision_t;
#define PUF_PREC_BYTES 2
static inline float to_float(bfloat v) { return puf_bf16_to_f32(v); }
static inline float to_float(float v) { return v; }
static inline bfloat from_float(float f) { return puf_f32_to_bf16(f); }
// Value of x after a round trip through precision_t storage.
static inline float round_p(float x) { return to_float(from_float(x)); }
#endif

// Corrected sigmoid: sigmoid(x) with z = exp(-|x|) and the quotient n / d
// (n = 1 for x >= 0, else z; d = 1 + z) formed from the hardware
// reciprocal estimate y ~ 1/d plus one fma residual correction (Markstein):
// q = n*y, q + (n - q*d)*y. Bit equality with the original IEEE divisions
// was verified exhaustively over all 2^32 inputs only on Apple M4 with the
// macOS 26.4 and 27 compilers (tests/metal/run_test_sigmoid.sh). MSL's
// reciprocal bounds do not guarantee correct rounding: for x = 2^-23,
// z = 0x3f7ffffe, d = 0x3fffffff, a conforming y = 0.5 gives corrected
// 0x3f000000 versus IEEE 0x3f000001. exp is device/compiler dependent too.
// The gate allows <= 1 ULP versus the device's IEEE sigmoid unless
// PUF_SIG_REQUIRE_EXACT=1. -DPUF_SIG_IEEE_DIV restores IEEE division.
static inline float puf_sigmoid_exact(float x) {
    float z = exp(-fabs(x));
#ifdef PUF_SIG_IEEE_DIV
    return x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
#else
    float d = 1.0f + z, n = x >= 0.0f ? 1.0f : z;
    float y = fast::divide(1.0f, d);
    float q = n * y;
#if defined(PUF_SIG_MUTATE) && PUF_SIG_MUTATE == 1
    float p = q * d;  // test mutation: residual from a rounded product
    return fma(n - p, y, q);
#elif defined(PUF_SIG_MUTATE) && PUF_SIG_MUTATE == 2
    return q;  // test mutation: no correction step
#else
    float result = fma(fma(-q, d, n), y, q);
#if defined(PUF_SIG_MUTATE) && PUF_SIG_MUTATE == 3
    // At either signed zero the unmutated result is exactly 0.5.
    if ((as_type<uint>(x) & 0x7fffffffu) == 0)
        return as_type<float>(as_type<uint>(result) + 2u);
#endif
    return result;
#endif
#endif
}

#ifdef PUF_TEST_SIGMOID
// tests/metal/test_sigmoid.mm: sig_check_SUF compares FN(x) for every fp32
// bit pattern with the original IEEE-division formula compiled with the same
// flags in the same translation unit; bit mismatches are counted (NaN inputs
// separately), maximum ULP distance is reduced, and the first 64 mismatches
// recorded as (x, shipped, reference). NaN payloads and zero signs have no
// ULP distance; a wrong output class/range has distance UINT_MAX.
// sig_dump_SUF writes (z, GPU-rounded d, FN(x), IEEE reference) for the
// host's independent rounding/FTZ oracle. Never in the shipped libraries.
static inline float puf_sigmoid_ieee_ref(float x) {
    float z = exp(-fabs(x));
    return x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
}
static inline uint puf_sigmoid_ulp(uint got, uint want) {
    if (got == want) return 0;
    uint g = got & 0x7fffffffu, w = want & 0x7fffffffu;
    if (g > 0x7f800000u && w > 0x7f800000u) return 0;  // NaN payloads
    if (g == 0) got = 0;  // FTZ may produce either zero sign
    if (w == 0) want = 0;
    if (got > 0x3f800000u || want > 0x3f800000u) return 0xffffffffu;
    return got > want ? got - want : want - got;
}
struct PufSigArgs {
    device atomic_uint* cnt;   // [0] mismatches, [1] NaN mismatches, [2] recorded, [3] max ULP
    device uint* rec;          // 64 x (x, got, want)
    device float* dump;        // sig_dump: 4 floats per sample
    uint base, per_thread, stride;
};
#define PUF_SIG_TEST_KERNELS(SUF, FN)                                                   \
kernel void sig_check_##SUF(constant PufSigArgs& a [[buffer(0)]],                       \
        uint gid [[thread_position_in_grid]]) {                                         \
    uint bad = 0, bad_nan = 0, max_ulp = 0;                                             \
    for (uint k = 0; k < a.per_thread; ++k) {                                           \
        uint bits = a.base + gid * a.per_thread + k;                                    \
        float x = as_type<float>(bits);                                                 \
        uint got = as_type<uint>(FN(x)), want = as_type<uint>(puf_sigmoid_ieee_ref(x)); \
        max_ulp = max(max_ulp, puf_sigmoid_ulp(got, want));                              \
        if (got != want) {                                                              \
            if (isnan(x)) bad_nan++; else bad++;                                        \
            uint slot = atomic_fetch_add_explicit(&a.cnt[2], 1u, memory_order_relaxed); \
            if (slot < 64) {                                                            \
                a.rec[3 * slot] = bits; a.rec[3 * slot + 1] = got;                      \
                a.rec[3 * slot + 2] = want;                                             \
            }                                                                           \
        }                                                                               \
    }                                                                                   \
    if (bad) atomic_fetch_add_explicit(&a.cnt[0], bad, memory_order_relaxed);           \
    if (bad_nan) atomic_fetch_add_explicit(&a.cnt[1], bad_nan, memory_order_relaxed);   \
    if (max_ulp) atomic_fetch_max_explicit(&a.cnt[3], max_ulp, memory_order_relaxed);    \
}                                                                                       \
kernel void sig_dump_##SUF(constant PufSigArgs& a [[buffer(0)]],                        \
        uint gid [[thread_position_in_grid]]) {                                         \
    float x = as_type<float>(a.base + gid * a.stride);                                  \
    float z = exp(-fabs(x));                                                            \
    a.dump[4 * gid] = z;                                                               \
    a.dump[4 * gid + 1] = 1.0f + z;                                                    \
    a.dump[4 * gid + 2] = FN(x);                                                        \
    a.dump[4 * gid + 3] = puf_sigmoid_ieee_ref(x);                                      \
}
#endif

#define PUF_KERNEL_IDX \
    uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint3 blockDim [[threads_per_threadgroup]], \
    uint3 threadIdx [[thread_position_in_threadgroup]], \
    uint3 gridDim [[threadgroups_per_grid]]

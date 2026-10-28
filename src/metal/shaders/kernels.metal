// Ports of PufferLib CUDA kernels (src/pufferl.cu, src/algo.cu).
// Each kernel keeps the CUDA name (plus a type suffix for overloads) and its
// body; arguments come in a struct at [[buffer(0)]] whose host twin lives in
// src/metal/kernels.h.
#include "common.h"

// ---- casts (pufferl.cu) ----------------------------------------------------

struct CastF32Args { device precision_t* dst; device const float* src; int n; };
kernel void cast_f32_to_prec(constant CastF32Args& a [[buffer(0)]], PUF_KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < a.n) {
        a.dst[idx] = from_float(a.src[idx]);
    }
}

struct CastU8Args { device precision_t* dst; device const uchar* src; int n; };
kernel void cast_u8_to_prec(constant CastU8Args& a [[buffer(0)]], PUF_KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < a.n) {
        a.dst[idx] = from_float((float)a.src[idx]);
    }
}

struct CastToU8Args { device uchar* dst; device const precision_t* src; int n; };
kernel void cast_prec_to_u8(constant CastToU8Args& a [[buffer(0)]], PUF_KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < a.n) {
        // CUDA: float -> unsigned char conversion (truncation).
        a.dst[idx] = (uchar)to_float(a.src[idx]);
    }
}

#ifndef PRECISION_FLOAT
// bf16-only overloads (pufferl.cu): fp32 master weights <- bf16 params, and
// the identity cast that typechecks the obs -> rollout arm when obs_t is bf16.
struct CastToF32Args { device float* dst; device const precision_t* src; int n; };
kernel void cast_prec_to_f32(constant CastToF32Args& a [[buffer(0)]], PUF_KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < a.n) {
        a.dst[idx] = to_float(a.src[idx]);
    }
}

struct CastPrecArgs { device precision_t* dst; device const precision_t* src; int n; };
kernel void cast_prec_to_prec(constant CastPrecArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < a.n) {
        a.dst[idx] = a.src[idx];
    }
}
#endif

struct CastRewTermArgs {
    device precision_t* rew_dst;
    device const float* rew_src;
    device precision_t* term_dst;
    device const float* term_src;
    int n;
};
kernel void cast_rew_term(constant CastRewTermArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < a.n) {
        a.rew_dst[idx] = from_float(a.rew_src[idx]);
        a.term_dst[idx] = from_float(a.term_src[idx]);
    }
}

// ---- elementwise (pufferl.cu / algo.cu) --------------------------------------

struct Transpose102Args { device precision_t* dst; device const precision_t* src; int A, B, C; };
kernel void transpose_102(constant Transpose102Args& a [[buffer(0)]], PUF_KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int A = a.A, B = a.B, C = a.C;
    int total = A * B * C;
    if (idx >= total) {
        return;
    }
    int ai = idx / (B * C);
    int rem = idx % (B * C);
    int b = rem / C;
    int c = rem % C;
    a.dst[b * A * C + ai * C + c] = a.src[idx];
}

#ifndef PRECISION_FLOAT
// float overload (pufferl.cu, !PRECISION_FLOAT): fp32 rollout actions.
struct Transpose102F32Args { device float* dst; device const float* src; int A, B, C; };
kernel void transpose_102_f32(constant Transpose102F32Args& a [[buffer(0)]], PUF_KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int A = a.A, B = a.B, C = a.C;
    int total = A * B * C;
    if (idx >= total) {
        return;
    }
    int ai = idx / (B * C);
    int rem = idx % (B * C);
    int b = rem / C;
    int c = rem % C;
    a.dst[b * A * C + ai * C + c] = a.src[idx];
}
#endif

struct ClampArgs { device precision_t* dst; float lo, hi; int n; };
kernel void clamp_precision_kernel(constant ClampArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < a.n;
            idx += blockDim.x * gridDim.x) {
        float v = to_float(a.dst[idx]);
        a.dst[idx] = from_float(fmin(fmax(v, a.lo), a.hi));
    }
}

struct AddArgs { device precision_t* dst; device const precision_t* src; int n; };
kernel void add_kernel(constant AddArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < a.n;
            idx += blockDim.x * gridDim.x) {
        a.dst[idx] = from_float(to_float(a.dst[idx]) + to_float(a.src[idx]));
    }
}

struct UniformScaleArgs { device float* data; float bound; int n; };
kernel void uniform_scale_kernel(constant UniformScaleArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < a.n) {
        a.data[idx] = a.data[idx] * 2.0f * a.bound - a.bound;
    }
}

// CUDA reads %globaltimer; MSL has no device clock. The trainer's puf_stamp
// adapter uses an MTL4CounterHeap timestamp instead (puf_mtl::stamp); this
// kernel is kept only for direct callers and writes 0.
struct StampArgs { device ulong* dst; };
kernel void puf_stamp(constant StampArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    *a.dst = 0;
}

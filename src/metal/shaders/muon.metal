// Ports of the six Muon kernels in src/algo.cu. Keep the CUDA 256-lane
// halving tree and its grid-stride accumulation order. Wide norm kernels
// finish with explicit shuffles using the same operands and add order.
// Gradients / NS buffers are precision_t, the momentum and master weights
// fp32 (as upstream); with PRECISION_FLOAT every to_float / from_float /
// round_p is the identity.
#include "common.h"
#include <metal_simdgroup_matrix>

// Copyright © 2023-2024 Apple Inc.
// Explicit shuffle mechanism adapted from MLX reduction/ops.h at
// 56c2bf9236d929097f61e2c6e55be4e4bb6da249. Keep the existing 256-lane
// tree through stride 32, then finish its last 32 values within one SG.
static inline float muon_tree_finish(float v) {
    for (ushort offset = 16; offset; offset >>= 1)
        v += simd_shuffle_down(v, offset);
    return v;
}

struct MuonSumPrecArgs { device float* dst; device const precision_t* src; int n; };
struct MuonSumArgs { device float* dst; device const float* src; int n; };
kernel void muon_sum_sq_partials(constant MuonSumPrecArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    threadgroup float sh[256];
    uint tid = threadIdx.x;
    float sum = 0.0f;
    for (uint i = blockIdx.x * blockDim.x + tid; i < uint(a.n);
            i += blockDim.x * gridDim.x) {
        float v = to_float(a.src[i]);
        sum = fma(v, v, sum);
    }
    sh[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = blockDim.x / 2; s; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) a.dst[blockIdx.x] = sh[0];
}

kernel void muon_sum_sq_reduce(constant MuonSumArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    threadgroup float sh[256];
    uint tid = threadIdx.x;
    sh[tid] = tid < uint(a.n) ? a.src[tid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = blockDim.x / 2; s; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) *a.dst = sh[0];
}

struct MuonClipArgs {
    device float* mb;
    device precision_t* gc;
    device const float* sum_sq;
    float max_norm, eps, mu;
    int n;
};
kernel void muon_clip_nesterov(constant MuonClipArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    float coef = fmin(a.max_norm / (sqrt(*a.sum_sq) + a.eps), 1.0f);
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < uint(a.n)) {
        float raw_g = to_float(a.gc[i]);
        // nvcc contracts the clipped-gradient product into the momentum add.
        float m = fma(raw_g, coef, a.mu * a.mb[i]);
        a.mb[i] = m;
        a.gc[i] = from_float(fma(a.mu, m, raw_g * coef));
    }
}

struct MuonNormalizeArgs { device precision_t* dst; device const float* sum_sq; float eps; int n; };
kernel void muon_l2_normalize(constant MuonNormalizeArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    float inv_norm = 1.0f / fmax(sqrt(*a.sum_sq), a.eps);
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < uint(a.n)) a.dst[i] = from_float(to_float(a.dst[i]) * inv_norm);
}

struct MuonStoreArgs { device precision_t* dst; device const precision_t* src; float scale; int n; };
kernel void muon_store_update(constant MuonStoreArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < uint(a.n)) a.dst[i] = from_float(a.scale * to_float(a.src[i]));
}

struct MuonWeightArgs {
    device float* wb;
    device const precision_t* update;
    device const float* lr;
    float wd;
    int n;
};
kernel void muon_weight_update(constant MuonWeightArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    float lr = *a.lr;
    float wd_scale = fma(-lr, a.wd, 1.0f);
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    // CUDA rounds lr*update before subtracting, and contracts wb*wd_scale.
    if (i < uint(a.n)) a.wb[i] = fma(a.wb[i], wd_scale, -(lr * to_float(a.update[i])));
}

// ---------------------------------------------------------------------------
// The five Newton-Schulz steps of muon_step for one (R, C) matrix in a single
// threadgroup, replacing 25 dependent dispatches (3 GEMMs + 2 copies per step).
// Every output element repeats puf_gemm_f32's arithmetic exactly: acc starts
// at 0, fma over k = 0..K-1 in order, then the tile loop's zero-padded
// fma(0, 0, acc) up to the next multiple of 16 (only observable as -0 -> +0,
// reproduced by one `acc + 0.0f`), then v = alpha * acc and, if beta != 0,
// v = fma(beta, C_old, v). The matrix, its gram and gram_buf live in
// threadgroup memory; device buffers end exactly as the unfused sequence
// leaves them (x = step 3's result, x_buf = step 4's, gram / gram_buf of
// step 4). Host checks (muon_ns5_eligible): 1 <= min(R, C) <= 32 and
// R * C <= 4096 (<= 4 elements per thread at 1024 threads).
// bf16: the unfused GEMMs store bf16 (rounded once after alpha/beta), so
// every GEMM result is passed through round_p before it lands in threadgroup
// memory; operands read from threadgroup memory are then exactly the bf16
// values the unfused GEMMs read.
// ---------------------------------------------------------------------------
#define MUON_NS_MAX_RC 4096
#define MUON_NS_MAX_M 32
#define MUON_NS_PER_THREAD 4
struct MuonNSArgs {
    device precision_t* x;         // (R, C) normalized input
    device precision_t* x_buf;     // (R, C) receives the final result (step 4)
    device precision_t* gram;      // (M, M)
    device precision_t* gram_buf;  // (M, M)
    float coeffs[15];        // ns_coeffs[i][0..2]
    int R, C;
    int mma;                 // simdgroup-fragment stages (ns5_tg_mma)
};

static inline float ns_finish(float acc, int K, float alpha, float beta, float c_old) {
    if (K % 16) acc = acc + 0.0f;
    float v = alpha * acc;
    if (beta != 0.0f) v = fma(beta, c_old, v);
    return round_p(v);
}

// One 8x8 output fragment of a threadgroup-resident GEMM: A(m, k) =
// a[m * as_m + k * as_k], B(k, n) = b[k * bs_k + n * bs_n], K % 8 == 0.
// simdgroup_multiply_accumulate chains each element's products in ascending
// k exactly as the scalar fma loop does (muon_wide_mma above, the sg16
// GEMMs); lane fragment coordinates (fm, fn), (fm, fn + 1) as there.
static inline simdgroup_float8x8 ns_frag(threadgroup const float* a, int as_m, int as_k,
        threadgroup const float* b, int bs_k, int bs_n, int m0, int n0, int K,
        int fm, int fn) {
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k = 0; k < K; k += 8) {
        simdgroup_float8x8 av, bv;
        av.thread_elements()[0] = a[(m0 + fm) * as_m + (k + fn) * as_k];
        av.thread_elements()[1] = a[(m0 + fm) * as_m + (k + fn + 1) * as_k];
        bv.thread_elements()[0] = b[(k + fm) * bs_k + (n0 + fn) * bs_n];
        bv.thread_elements()[1] = b[(k + fm) * bs_k + (n0 + fn + 1) * bs_n];
        simdgroup_multiply_accumulate(acc, av, bv, acc);
    }
    return acc;
}

// ns5_tg's three GEMM stages on simdgroup fragments (R % 8 == 0, C % 8 == 0,
// T % 32 == 0, R * C <= 4 * T so each simdgroup owns at most two update
// fragments). Same per-element K chains and ns_finish as the scalar loops.
static void ns5_tg_mma(threadgroup float* xs, threadgroup float* gs, threadgroup float* gbs,
        int R, int C, constant float* coeffs, int t, int T,
        device precision_t* out3, device precision_t* out4) {
    const bool tall = R > C;
    const int M = tall ? C : R, MT = M / 8;
    const int sg = t / 32, nsg = T / 32, lane = t % 32;
    const int qid = lane / 4;
    const int fm = (qid & 4) + ((lane / 2) % 4);
    const int fn = (qid & 2) * 2 + (lane % 2) * 2;
    for (int i = 0; i < 5; ++i) {
        const float c0 = coeffs[3 * i], c1 = coeffs[3 * i + 1], c2 = coeffs[3 * i + 2];
        // gram = src^T src (tall, K = R) or src src^T (wide, K = C).
        for (int f = sg; f < MT * MT; f += nsg) {
            const int m0 = f / MT * 8, n0 = f % MT * 8;
            simdgroup_float8x8 acc = tall
                ? ns_frag(xs, 1, C, xs, C, 1, m0, n0, R, fm, fn)
                : ns_frag(xs, C, 1, xs, 1, C, m0, n0, C, fm, fn);
            for (int j = 0; j < 2; ++j)
                gs[(m0 + fm) * M + n0 + fn + j] =
                    ns_finish(acc.thread_elements()[j], tall ? R : C, 1.0f, 0.0f, 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // gram_buf = c2 * gram @ gram + c1 * gram.
        for (int f = sg; f < MT * MT; f += nsg) {
            const int m0 = f / MT * 8, n0 = f % MT * 8;
            simdgroup_float8x8 acc = ns_frag(gs, M, 1, gs, M, 1, m0, n0, M, fm, fn);
            for (int j = 0; j < 2; ++j) {
                int e = (m0 + fm) * M + n0 + fn + j;
                gbs[e] = ns_finish(acc.thread_elements()[j], M, c2, c1, gs[e]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // dst = src @ gram_buf + c0 * src (tall, K = C) or gram_buf @ src + c0 * src (wide, K = R).
        const int CT = C / 8, nf = (R / 8) * CT;
        float v[4];
        for (int q = 0; q < 2; ++q) {
            const int f = sg + q * nsg;
            if (f >= nf) break;
            const int m0 = f / CT * 8, n0 = f % CT * 8;
            simdgroup_float8x8 acc = tall
                ? ns_frag(xs, C, 1, gbs, C, 1, m0, n0, C, fm, fn)
                : ns_frag(gbs, R, 1, xs, C, 1, m0, n0, R, fm, fn);
            for (int j = 0; j < 2; ++j)
                v[2 * q + j] = ns_finish(acc.thread_elements()[j], M, 1.0f, c0,
                    xs[(m0 + fm) * C + n0 + fn + j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device precision_t* out = i == 3 ? out3 : i == 4 ? out4 : nullptr;
        for (int q = 0; q < 2; ++q) {
            const int f = sg + q * nsg;
            if (f >= nf) break;
            const int m0 = f / CT * 8, n0 = f % CT * 8;
            for (int j = 0; j < 2; ++j) {
                int e = (m0 + fm) * C + n0 + fn + j;
                xs[e] = v[2 * q + j];
                if (out) out[e] = from_float(v[2 * q + j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Five NS steps on the threadgroup-resident matrix xs (gs / gbs: gram and
// gram_buf). Optional out3 / out4 receive steps 3 / 4 (the unfused loop's x /
// x_buf); xs ends holding step 4's result. mma: ns5_tg_mma (host-checked
// shape: R % 8 == 0, C % 8 == 0).
static void ns5_tg(threadgroup float* xs, threadgroup float* gs, threadgroup float* gbs,
        int R, int C, constant float* coeffs, int t, int T,
        device precision_t* out3, device precision_t* out4, bool mma = false) {
    if (mma && R % 8 == 0 && C % 8 == 0 && T % 32 == 0 && R * C <= 4 * T) {
        ns5_tg_mma(xs, gs, gbs, R, C, coeffs, t, T, out3, out4);
        return;
    }
    const bool tall = R > C;
    const int M = tall ? C : R, RC = R * C, MM = M * M;
    for (int i = 0; i < 5; ++i) {
        const float c0 = coeffs[3 * i], c1 = coeffs[3 * i + 1], c2 = coeffs[3 * i + 2];
        // gram = src^T src (tall, K = R) or src src^T (wide, K = C).
        const int Kg = tall ? R : C;
        for (int e = t; e < MM; e += T) {
            int m = e / M, n = e % M;
            float acc = 0.0f;
            if (tall) {
                for (int k = 0; k < Kg; ++k) acc = fma(xs[k * C + m], xs[k * C + n], acc);
            } else {
                for (int k = 0; k < Kg; ++k) acc = fma(xs[m * C + k], xs[n * C + k], acc);
            }
            gs[e] = ns_finish(acc, Kg, 1.0f, 0.0f, 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // gram_buf = c2 * gram @ gram + c1 * gram (copy of gram as the C operand).
        for (int e = t; e < MM; e += T) {
            int m = e / M, n = e % M;
            float acc = 0.0f;
            for (int k = 0; k < M; ++k) acc = fma(gs[m * M + k], gs[k * M + n], acc);
            gbs[e] = ns_finish(acc, M, c2, c1, gs[e]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // dst = src @ gram_buf + c0 * src (tall, K = C) or gram_buf @ src + c0 * src (wide, K = R).
        float v[MUON_NS_PER_THREAD];
        for (int j = 0; j < MUON_NS_PER_THREAD; ++j) {
            int e = t + j * T;
            if (e >= RC) break;
            int m = e / C, n = e % C;
            float acc = 0.0f;
            if (tall) {
                for (int k = 0; k < C; ++k) acc = fma(xs[m * C + k], gbs[k * C + n], acc);
            } else {
                for (int k = 0; k < R; ++k) acc = fma(gbs[m * R + k], xs[k * C + n], acc);
            }
            v[j] = ns_finish(acc, M, 1.0f, c0, xs[e]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device precision_t* out = i == 3 ? out3 : i == 4 ? out4 : nullptr;
        for (int j = 0; j < MUON_NS_PER_THREAD; ++j) {
            int e = t + j * T;
            if (e >= RC) break;
            xs[e] = v[j];
            if (out) out[e] = from_float(v[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void muon_ns5(constant MuonNSArgs& a [[buffer(0)]],
        uint tid [[thread_index_in_threadgroup]], uint nthr [[threads_per_threadgroup]]) {
    threadgroup float xs[MUON_NS_MAX_RC], gs[MUON_NS_MAX_M * MUON_NS_MAX_M],
        gbs[MUON_NS_MAX_M * MUON_NS_MAX_M];
    const int R = a.R, C = a.C;
    if (R < 1 || C < 1 || min(R, C) > MUON_NS_MAX_M || R * C > MUON_NS_MAX_RC
            || R * C > MUON_NS_PER_THREAD * int(nthr)) return;
    const int M = min(R, C), RC = R * C, MM = M * M;
    const int t = int(tid), T = int(nthr);
    for (int e = t; e < RC; e += T) xs[e] = to_float(a.x[e]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ns5_tg(xs, gs, gbs, R, C, a.coeffs, t, T, a.x, a.x_buf, a.mma != 0);
    for (int e = t; e < MM; e += T) {
        a.gram[e] = from_float(gs[e]);
        a.gram_buf[e] = from_float(gbs[e]);
    }
}

// ---------------------------------------------------------------------------
// muon_step's whole per-matrix tail for up to 8 matrices, one threadgroup
// each, in one dispatch: muon_sum_sq_partials + muon_sum_sq_reduce (same
// 256-lane trees: with R * C <= 4096 every partial block's thread holds at
// most one element), muon_l2_normalize, the five NS steps (ns5_tg) and
// muon_store_update (x = scale * result). Only x (the flat gradient slice)
// is written; the unfused path's scratch (norm partials, ns_norm, x_buf,
// gram, gram_buf) is not consumed afterwards. Host checks: every matrix
// passes muon_ns5_eligible, count <= 8, 256 <= threads.
// ---------------------------------------------------------------------------
struct MuonNSMat { device precision_t* x; float scale; int R, C, mma; };
struct MuonNSBatchArgs {
    MuonNSMat mats[8];
    float coeffs[15];
    float eps;
    int count;
};

kernel void muon_ns5_batch(constant MuonNSBatchArgs& a [[buffer(0)]],
        uint group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]], uint nthr [[threads_per_threadgroup]]) {
    threadgroup float xs[MUON_NS_MAX_RC], scratch[2 * MUON_NS_MAX_M * MUON_NS_MAX_M], part[256];
    if (a.count < 1 || a.count > 8 || int(group) >= a.count || nthr < 256) return;
    constant MuonNSMat& mat = a.mats[group];
    const int R = mat.R, C = mat.C;
    if (R < 1 || C < 1 || min(R, C) > MUON_NS_MAX_M || R * C > MUON_NS_MAX_RC
            || R * C > MUON_NS_PER_THREAD * int(nthr)) return;
    const int RC = R * C, t = int(tid), T = int(nthr);
    const int nblk = min((RC + 255) / 256, 256);  // <= 16
    for (int e = t; e < RC; e += T) xs[e] = to_float(mat.x[e]);
    if (t < 256) part[t] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Partials, eight 256-lane blocks per pass through scratch.
    for (int p = 0; p * 8 < nblk; ++p) {
        for (int e = t; e < 2048; e += T) {
            int idx = p * 2048 + e;
            scratch[e] = idx < RC ? fma(xs[idx], xs[idx], 0.0f) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int s = 128; s; s >>= 1) {
            for (int e = t; e < 2048; e += T) if ((e & 255) < s) scratch[e] += scratch[e + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (t < 8 && p * 8 + t < nblk) part[p * 8 + t] = scratch[t * 256];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int s = 128; s; s >>= 1) {
        if (t < s) part[t] += part[t + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_norm = 1.0f / fmax(sqrt(part[0]), a.eps);
    for (int e = t; e < RC; e += T) xs[e] = round_p(xs[e] * inv_norm);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ns5_tg(xs, scratch, scratch + MUON_NS_MAX_M * MUON_NS_MAX_M, R, C, a.coeffs, t, T,
        nullptr, nullptr, mat.mma != 0);
    for (int e = t; e < RC; e += T) mat.x[e] = from_float(mat.scale * xs[e]);
}

// ---------------------------------------------------------------------------
// muon_step's per-matrix tail for up to 8 matrices that are too large for
// muon_ns5_batch (e.g. the (3H, H) MinGRU weights at H >= 64), batched across
// matrices: grid.z = matrix, one dispatch per unfused stage, 18 dispatches in
// all instead of 29 per matrix (gemm.h muon_ns_wide_rm). Every stage repeats
// the unfused kernel's arithmetic per element, so only the tiling and launch
// shape differ:
//   muon_wide_sumsq      = muon_sum_sq_partials (same blocks / grid stride)
//   muon_wide_normalize  = muon_sum_sq_reduce (recomputed per threadgroup,
//                          same 256-lane tree) + muon_l2_normalize
//   muon_wide_gram       = the gram GEMM: tensor_ops_gemm_tn / _nt's exact
//                          matmul2d configuration where gemm_rm takes that
//                          path (M % 64 == 0 and a 16-byte aligned source:
//                          MuonWideMat.tops bit 0 for x, bit 1 for x_buf),
//                          else puf_gemm's tiled arithmetic (ns_finish)
//   muon_wide_gram_buf   = puf_copy + tiled c2 * gram @ gram + c1 * gram
//   muon_wide_update     = puf_copy + tiled src @ gram_buf + c0 * src (tall)
//                          or gram_buf @ src + c0 * src
//   muon_wide_store      = muon_store_update
// The wide host path limits max(R, C) < 2048; larger eligible shapes use
// the separate large-Muon dispatch and kernels (see gemm.mm / wide_muon.h).
// ---------------------------------------------------------------------------
struct MuonWideMat {
    device precision_t* x;   // (R, C) flat-gradient slice, overwritten
    device precision_t* xb;  // (R, C) scratch (the unfused x_buf)
    device precision_t* g;   // (M, M) gram
    device precision_t* gb;  // (M, M) gram_buf
    device float* part;      // 256 norm partials
    float scale;
    int R, C;
    int tops;
};
struct MuonWideArgs {
    MuonWideMat mats[8];
    float c0, c1, c2, eps;
    int count, step, pad0, pad1;
};

kernel void muon_wide_sumsq(constant MuonWideArgs& a [[buffer(0)]],
        uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[256];
    if (int(tg.z) >= a.count) return;
    constant MuonWideMat& m = a.mats[tg.z];
    const int n = m.R * m.C;
    const uint nblk = uint(min((n + 255) / 256, 256));
    if (tg.x >= nblk) return;
    float sum = 0.0f;
    for (uint i = tg.x * 256 + tid; i < uint(n); i += 256 * nblk) {
        float v = to_float(m.x[i]);
        sum = fma(v, v, sum);
    }
    sh[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s >= 32; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 32) {
        float v = muon_tree_finish(sh[tid]);
        if (tid == 0) m.part[tg.x] = v;
    }
}

kernel void muon_wide_normalize(constant MuonWideArgs& a [[buffer(0)]],
        uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sh[256];
    if (int(tg.z) >= a.count) return;
    constant MuonWideMat& m = a.mats[tg.z];
    const int n = m.R * m.C;
    if (int(tg.x) * 256 >= n) return;
    const uint nblk = uint(min((n + 255) / 256, 256));
    sh[tid] = tid < nblk ? m.part[tid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s >= 32; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 32) {
        float v = muon_tree_finish(sh[tid]);
        if (tid == 0) sh[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_norm = 1.0f / fmax(sqrt(sh[0]), a.eps);
    uint i = tg.x * 256 + tid;
    if (i < uint(n)) m.x[i] = from_float(to_float(m.x[i]) * inv_norm);
}

#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

typedef tensor<device precision_t, dextents<int32_t, 2>, tensor_inline> MuonTensor2;

// One 64 x 32 tile of gram (rows tg.y * 64, columns tg.x * 32), 128 threads.
kernel void muon_wide_gram(constant MuonWideArgs& a [[buffer(0)]],
        uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    using namespace mpp::tensor_ops;
    if (int(tg.z) >= a.count) return;
    constant MuonWideMat& m = a.mats[tg.z];
    const int R = m.R, C = m.C;
    const bool tall = R > C;
    const int M = tall ? C : R, K = tall ? R : C;
    const int m0 = int(tg.y) * 64, n0 = int(tg.x) * 32;
    if (m0 >= M || n0 >= M) return;
    device precision_t* src = (a.step % 2 == 0) ? m.x : m.xb;
    if ((m.tops >> (a.step % 2)) & 1) {
        // gemm_rm's tensor_ops branch (M % 64 == 0): tall = puf_mm_tn, the
        // tensor_ops_gemm_tn kernel; otherwise puf_mm, tensor_ops_gemm_nt.
        auto A = tall ? MuonTensor2(src, dextents<int32_t, 2>(M, K))
                      : MuonTensor2(src, dextents<int32_t, 2>(K, M));
        if (tall) {
            constexpr auto desc = matmul2d_descriptor(64, 32,
                static_cast<int>(dynamic_extent), true, false, false);
            matmul2d<desc, execution_simdgroups<4>> op;
            auto mFirst = A.slice(m0, 0);
            auto mSecond = A.slice(n0, 0);
#ifdef PRECISION_FLOAT
            auto G = MuonTensor2(m.g, dextents<int32_t, 2>(M, M));
            auto mResult = G.slice(n0, m0);
            op.run(mFirst, mSecond, mResult);
#else
            auto cT = op.get_destination_cooperative_tensor<decltype(mFirst), decltype(mSecond), float>();
            #pragma unroll
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                if (cT.is_valid_element(i)) cT[i] = 0.0f;
            }
            op.run(mFirst, mSecond, cT);
            #pragma unroll
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                if (!cT.is_valid_element(i)) continue;
                auto idx = cT.get_multidimensional_index(i);
                m.g[(m0 + idx[1]) * M + n0 + idx[0]] = from_float(cT[i]);
            }
#endif
        } else {
            constexpr auto desc = matmul2d_descriptor(64, 32,
                static_cast<int>(dynamic_extent), false, true, false);
            matmul2d<desc, execution_simdgroups<4>> op;
            auto mA = A.slice(0, m0);
            auto mB = A.slice(0, n0);
#ifdef PRECISION_FLOAT
            auto G = MuonTensor2(m.g, dextents<int32_t, 2>(M, M));
            auto mC = G.slice(n0, m0);
            op.run(mA, mB, mC);
#else
            auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
            #pragma unroll
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                if (cT.is_valid_element(i)) cT[i] = 0.0f;
            }
            op.run(mA, mB, cT);
            #pragma unroll
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                if (!cT.is_valid_element(i)) continue;
                auto idx = cT.get_multidimensional_index(i);
                m.g[(m0 + idx[1]) * M + n0 + idx[0]] = from_float(cT[i]);
            }
#endif
        }
        return;
    }
    for (int e = int(tid); e < 64 * 32; e += 128) {
        int mi = m0 + e / 32, ni = n0 + e % 32;
        if (mi >= M || ni >= M) continue;
        float acc = 0.0f;
        if (tall) {
            for (int k = 0; k < K; ++k)
                acc = fma(to_float(src[k * C + mi]), to_float(src[k * C + ni]), acc);
        } else {
            for (int k = 0; k < K; ++k)
                acc = fma(to_float(src[mi * C + k]), to_float(src[ni * C + k]), acc);
        }
        m.g[mi * M + ni] = from_float(ns_finish(acc, K, 1.0f, 0.0f, 0.0f));
    }
}

// Copyright © 2024 Apple Inc.
// Four spatial 8x8 fragments per 16x16 output tile. Preserve the scalar
// kernel's ascending K chain, alpha/beta finish and BF16 rounding. The
// existing 256-thread dispatch has four inactive SGs on this path; no
// threadgroup staging or barriers are needed for these small cached operands.
// Fragment coordinates adapt MLX Steel BaseMMAFrag<T, 8, 8> (see NOTICE).
static void muon_wide_mma(constant MuonWideArgs& a, uint3 tg, uint tid, bool update) {
    constant MuonWideMat& mat = a.mats[tg.z];
    const int M = min(mat.R, mat.C), N = update ? mat.C : M;
    const int rows = update ? mat.R : M;
    const int sg = int(tid) / 32, lane = int(tid) % 32;
    if (sg >= 4) return;
    const int m0 = int(tg.x) / (N / 16) * 16 + (sg / 2) * 8;
    const int n0 = int(tg.x) % (N / 16) * 16 + (sg % 2) * 8;
    if (m0 >= rows) return;
    device const precision_t* src = a.step % 2 == 0 ? mat.x : mat.xb;
    device const precision_t* left = update ? (mat.R > mat.C ? src : mat.gb) : mat.g;
    device const precision_t* right = update ? (mat.R > mat.C ? mat.gb : src) : mat.g;
    const int lstride = update ? (mat.R > mat.C ? mat.C : M) : M;
    const int qid = lane / 4;
    const int fm = (qid & 4) + ((lane / 2) % 4);
    const int fn = (qid & 2) * 2 + (lane % 2) * 2;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k = 0; k < M; k += 8) {
        simdgroup_float8x8 av, bv;
        av.thread_elements()[0] = to_float(left[(m0 + fm) * lstride + k + fn]);
        av.thread_elements()[1] = to_float(left[(m0 + fm) * lstride + k + fn + 1]);
        bv.thread_elements()[0] = to_float(right[(k + fm) * N + n0 + fn]);
        bv.thread_elements()[1] = to_float(right[(k + fm) * N + n0 + fn + 1]);
        simdgroup_multiply_accumulate(acc, av, bv, acc);
    }
    device precision_t* dst = update ? (a.step % 2 == 0 ? mat.xb : mat.x) : mat.gb;
    device const precision_t* old = update ? src : mat.g;
    for (int i = 0; i < 2; ++i) {
        int e = (m0 + fm) * N + n0 + fn + i;
        float v = ns_finish(acc.thread_elements()[i], M, update ? 1.0f : a.c2,
            update ? a.c0 : a.c1, to_float(old[e]));
        dst[e] = from_float(v);
    }
}

kernel void muon_wide_gram_buf(constant MuonWideArgs& a [[buffer(0)]],
        uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (int(tg.z) >= a.count) return;
    constant MuonWideMat& m = a.mats[tg.z];
    if (min(m.R, m.C) == 64 && m.R % 16 == 0 && m.C % 16 == 0) {
        muon_wide_mma(a, tg, tid, false);
        return;
    }
    const int M = min(m.R, m.C);
    const int e = int(tg.x) * 256 + int(tid);
    if (e >= M * M) return;
    const int mi = e / M, ni = e % M;
    device const precision_t* g = m.g;
    float acc = 0.0f;
    for (int k = 0; k < M; ++k) acc = fma(to_float(g[mi * M + k]), to_float(g[k * M + ni]), acc);
    m.gb[e] = from_float(ns_finish(acc, M, a.c2, a.c1, to_float(g[e])));
}

kernel void muon_wide_update(constant MuonWideArgs& a [[buffer(0)]],
        uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (int(tg.z) >= a.count) return;
    constant MuonWideMat& m = a.mats[tg.z];
    if (min(m.R, m.C) == 64 && m.R % 16 == 0 && m.C % 16 == 0) {
        muon_wide_mma(a, tg, tid, true);
        return;
    }
    const int R = m.R, C = m.C;
    const int e = int(tg.x) * 256 + int(tid);
    if (e >= R * C) return;
    const bool even = a.step % 2 == 0;
    device const precision_t* src = even ? m.x : m.xb;
    device precision_t* dst = even ? m.xb : m.x;
    device const precision_t* gb = m.gb;
    const int mi = e / C, ni = e % C;
    float acc = 0.0f;
    if (R > C) {
        for (int k = 0; k < C; ++k) acc = fma(to_float(src[mi * C + k]), to_float(gb[k * C + ni]), acc);
    } else {
        for (int k = 0; k < R; ++k) acc = fma(to_float(gb[mi * R + k]), to_float(src[k * C + ni]), acc);
    }
    dst[e] = from_float(ns_finish(acc, min(R, C), 1.0f, a.c0, to_float(src[e])));
}

kernel void muon_wide_store(constant MuonWideArgs& a [[buffer(0)]],
        uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (int(tg.z) >= a.count) return;
    constant MuonWideMat& m = a.mats[tg.z];
    const int e = int(tg.x) * 256 + int(tid);
    if (e < m.R * m.C) m.x[e] = from_float(m.scale * to_float(m.xb[e]));
}

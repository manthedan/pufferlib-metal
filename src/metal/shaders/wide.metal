// Wide-policy kernels are isolated from the primary and grouped-rollout
// libraries so unused entry points do not change small-policy pipelines.
#include "common.h"
#include "wide_config.h"
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>
using namespace mpp::tensor_ops;

#ifdef PRECISION_FLOAT
kernel void puf_precision_marker_f32() {}
#else
kernel void puf_precision_marker_bf16() {}
#endif

template <typename T> struct WideGemmArgs {
    device T *A, *B, *C;
    int M, N, K;
};

// No split-K: retain the scalar fallback's single ascending-K fma chain.
// Tensor slices retain the operand bounds; only the final N tile is ragged.
template <typename T, int TILE_M, int TILE_N = PUF_WIDE_TN_N, int THREADS = PUF_WIDE_TN_THREADS>
static void wide_gemm_tn_body(constant WideGemmArgs<T>& a, uint2 tg) {
    auto A = tensor<device T, dextents<int32_t, 2>, tensor_inline>(
        a.A, dextents<int32_t, 2>(a.M, a.K));
    auto B = tensor<device T, dextents<int32_t, 2>, tensor_inline>(
        a.B, dextents<int32_t, 2>(a.N, a.K));
    constexpr auto desc = matmul2d_descriptor(TILE_M, TILE_N,
        static_cast<int>(dynamic_extent), true, false, false);
    matmul2d<desc, execution_simdgroups<THREADS / 32>> op;
    auto mA = A.slice(tg.y * TILE_M, 0);
    auto mB = B.slice(tg.x * TILE_N, 0);
    auto c = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    for (uint16_t i = 0; i < c.get_capacity(); ++i)
        if (c.is_valid_element(i)) c[i] = 0.0f;
    op.run(mA, mB, c);
    for (uint16_t i = 0; i < c.get_capacity(); ++i) {
        if (!c.is_valid_element(i)) continue;
        auto idx = c.get_multidimensional_index(i);
        int m = tg.y * TILE_M + idx[1], n = tg.x * TILE_N + idx[0];
        if (m < a.M && n < a.N) puf_st(a.C, (long)m * a.N + n, c[i]);
    }
}
#define WIDE_GEMM(SUF, T, TM) \
kernel void wide_gemm_tn_##SUF(constant WideGemmArgs<T>& a [[buffer(0)]], \
        uint2 tg [[threadgroup_position_in_grid]]) { wide_gemm_tn_body<T, TM>(a, tg); }
WIDE_GEMM(f32, float, PUF_WIDE_TN_M_F32)
WIDE_GEMM(bf16, bfloat, PUF_WIDE_TN_M_BF16)

// The H128 dW path keeps the existing split count, K chunk boundaries and
// reducer. Only the spatial tile and simdgroup count change.
struct WideTnSplitArgs {
    device void *A, *B;
    device float* partial;
    int M, N, K, k_chunk;
};
template<typename T, int TILE_M, int TILE_N = PUF_WIDE_TN_N, int THREADS = PUF_WIDE_TN_THREADS>
static void wide_tn_split_body(constant WideTnSplitArgs& a, uint3 tg) {
    int k0 = int(tg.z) * a.k_chunk, kc = min(a.k_chunk, a.K - k0);
    if (kc <= 0) return;
    auto A = tensor<device T, dextents<int32_t, 2>, tensor_inline>(
        (device T*)a.A + (long)k0 * a.M, dextents<int32_t, 2>(a.M, kc));
    auto B = tensor<device T, dextents<int32_t, 2>, tensor_inline>(
        (device T*)a.B + (long)k0 * a.N, dextents<int32_t, 2>(a.N, kc));
    auto P = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        a.partial + (long)tg.z * a.M * a.N, dextents<int32_t, 2>(a.N, a.M));
    constexpr auto desc = matmul2d_descriptor(TILE_M, TILE_N,
        static_cast<int>(dynamic_extent), true, false, false);
    matmul2d<desc, execution_simdgroups<THREADS / 32>> op;
    auto mA = A.slice(tg.y * TILE_M, 0);
    auto mB = B.slice(tg.x * TILE_N, 0);
    auto mP = P.slice(tg.x * TILE_N, tg.y * TILE_M);
    op.run(mA, mB, mP);
}
#define WIDE_TN_SPLIT(SUF, T, TM) \
kernel void wide_gemm_tn_split_##SUF(constant WideTnSplitArgs& a [[buffer(0)]], \
        uint3 tg [[threadgroup_position_in_grid]]) { wide_tn_split_body<T, TM>(a, tg); }
WIDE_TN_SPLIT(f32, float, PUF_WIDE_TN_M_F32)
WIDE_TN_SPLIT(bf16, bfloat, PUF_WIDE_TN_M_BF16)

// Separate entry points keep the H128/H256 tile table unchanged.
#define LARGE_TN(SUF, TYPE, TM, N512) \
kernel void wide_large_tn_##SUF(constant WideGemmArgs<TYPE>& a [[buffer(0)]], \
        uint2 tg [[threadgroup_position_in_grid]]) { \
    if (a.M == 3 * a.N && a.N == 512) \
        wide_gemm_tn_body<TYPE, TM, N512, PUF_LARGE_TN_THREADS>(a, tg); \
    else if (a.M == 3 * a.N && a.N == 1024) \
        wide_gemm_tn_body<TYPE, TM, PUF_LARGE_TN_N1024, PUF_LARGE_TN_THREADS>(a, tg); \
    else wide_gemm_tn_body<TYPE, TM, PUF_LARGE_TN_N, PUF_LARGE_TN_THREADS>(a, tg); \
} \
kernel void wide_large_tn_split_##SUF(constant WideTnSplitArgs& a [[buffer(0)]], \
        uint3 tg [[threadgroup_position_in_grid]]) { \
    wide_tn_split_body<TYPE, TM, PUF_LARGE_TN_N, PUF_LARGE_TN_THREADS>(a, tg); \
}
LARGE_TN(f32, float, PUF_LARGE_TN_M_F32, PUF_LARGE_TN_N512_F32)
LARGE_TN(bf16, bfloat, PUF_LARGE_TN_M_BF16, PUF_LARGE_TN_N512_BF16)

// Corrected sigmoid; common.h documents the device-specific exactness check.
static inline float mg_sigmoid(float x) { return puf_sigmoid_exact(x); }

static inline float mg_lerp(float a, float b, float w) {
    float diff = b - a;
    return (fabs(w) < 0.5f) ? fma(w, diff, a) : fma(-diff, 1.0f - w, b);
}

static inline float mg_h_tilde(float hidden) {
    return (hidden >= 0.0f) ? hidden + 0.5f : mg_sigmoid(hidden);
}

struct WideMingruFwdArgs {
    device precision_t* x;                // (B*T, H) layer input (GEMM + highway)
    device precision_t* W;                // (3H, H)
    device const precision_t* state;      // (B, H)
    device const precision_t* terminals;  // (B, T) or null
    device precision_t* scan_h;           // (B, T, H)
    device precision_t* out;              // (B, T, H)
    device precision_t* next_state;       // (B, H)
    device precision_t* combined;         // (B*T, 3H) projection, as the GEMM stores it
    int B, T, H;
};

struct WideMingruBwdArgs {
    device precision_t* x;                // (B*T, H) saved layer input
    device precision_t* W;                // (3H, H)
    device const precision_t* terminals;  // (B, T) or null
    device const precision_t* scan_h;     // (B, T, H)
    device const precision_t* grad_out;   // (B, T, H)
    device const precision_t* grad_next_state;  // (B, H)
    device precision_t* grad_combined;    // (B, T, 3H)
    device precision_t* grad_state;       // (B, H)
    device precision_t* grad_input;       // (B, T, H)
    device precision_t* grad_x;           // (B*T, H) dX = grad_combined @ W + grad_input
    device const precision_t* combined;   // (B*T, 3H) from the forward
    int B, T, H;
};

template<int H, int CHUNK, int THREADS, int TILE_N, int ROWS>
static void wide_mingru_forward_body(constant WideMingruFwdArgs& a,
        uint tile, uint tid, threadgroup precision_t* comb) {
    const int T_seq = a.T, H3 = 3 * H;
    // Uniform guard (barrier-safe): the tile must hold whole sequences and
    // lie inside the (B*T, H) input.
    if (a.H != H || T_seq <= 0 || ROWS % T_seq != 0
            || (long)(tile + 1) * ROWS > (long)a.B * T_seq) return;
    const int M = a.B * T_seq;
    // Steps of one sequence per chunk, and sequences per chunk.
    const int steps = min(T_seq, CHUNK), nseq = CHUNK / steps;
    float carry[(H + THREADS - 1) / THREADS] = {};  // T > CHUNK: one carried h per owned hidden unit

    using namespace mpp::tensor_ops;
    auto A = tensor<device precision_t, dextents<int32_t, 2>, tensor_inline>(
        a.x, dextents<int32_t, 2>(H, M));
    auto Bw = tensor<device precision_t, dextents<int32_t, 2>, tensor_inline>(
        a.W, dextents<int32_t, 2>(H, H3));
    constexpr auto desc = matmul2d_descriptor(CHUNK, TILE_N,
        static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<THREADS / 32>> op;
    for (int chunk = 0; chunk < ROWS / CHUNK; chunk++) {
        const int row0 = (int)tile * ROWS + chunk * CHUNK;
        // comb (CHUNK rows x 3H, [hidden | gate | proj]) = x[row0 ..] @ W^T.
        auto mA = A.slice(0, row0);
        for (int n0 = 0; n0 < H3; n0 += TILE_N) {
            auto mB = Bw.slice(0, n0);
            auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
            #pragma unroll
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                if (cT.is_valid_element(i)) cT[i] = 0.0f;
            }
            op.run(mA, mB, cT);
            #pragma unroll
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                if (!cT.is_valid_element(i)) continue;
                auto idx = cT.get_multidimensional_index(i);
                precision_t v = from_float(cT[i]);
                comb[idx[1] * H3 + n0 + idx[0]] = v;
                a.combined[(long)(row0 + idx[1]) * H3 + n0 + idx[0]] = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int lane = (int)tid; lane < nseq * H; lane += THREADS) {
            int seq = lane / H, h = lane % H;
            int b = row0 / T_seq + seq;
            int t_begin = row0 - b * T_seq + seq * steps;  // chunk's first step of b
            int bH = b * H, bHT = bH * T_seq;
            int out_base = bHT + h;
            device const precision_t* term_row =
                a.terminals ? a.terminals + b * T_seq : (device const precision_t*)nullptr;
            float h_t = t_begin == 0 ? to_float(a.state[bH + h]) : carry[h / THREADS];
            int r = seq * steps * H3 + h;
            for (int t = t_begin; t < t_begin + steps; t++) {
                if (term_row && to_float(term_row[t]) != 0.0f) {
                    h_t = 0.0f;
                }
                a.scan_h[out_base + t * H] = from_float(h_t);

                float hidden_val = to_float(comb[r]);
                float gate_val = to_float(comb[r + H]);
                float proj_val = to_float(comb[r + 2 * H]);
                float x_val = to_float(a.x[out_base + t * H]);

                float z = mg_sigmoid(gate_val);
                float h_tilde = mg_h_tilde(hidden_val);
                h_t = mg_lerp(h_t, h_tilde, z);

                float s = mg_sigmoid(proj_val);
                a.out[out_base + t * H] = from_float(fma(s, h_t, (1.0f - s) * x_val));
                h_t = to_float(from_float(h_t));
                r += H3;
            }
            if (t_begin + steps == T_seq) a.next_state[bH + h] = from_float(h_t);
            else carry[h / THREADS] = h_t;
        }
        // The next chunk's projection overwrites comb.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template<int H, int CHUNK, int THREADS, int TILE_N, int ROWS>
static void wide_mingru_backward_body(constant WideMingruBwdArgs& a,
        uint tile, uint tid, threadgroup precision_t* comb, threadgroup precision_t* gin) {
    const int T_seq = a.T, H3 = 3 * H;
    if (a.H != H || T_seq <= 0 || ROWS % T_seq != 0
            || (long)(tile + 1) * ROWS > (long)a.B * T_seq) return;
    // This chunk's grad_input rows, for the epilogue (device copy optional).
    const int M = a.B * T_seq;
    const int steps = min(T_seq, CHUNK), nseq = CHUNK / steps;
    float carry[(H + THREADS - 1) / THREADS] = {};  // T > CHUNK: one carried dh per owned hidden unit
    const bool store_gin = a.grad_input != nullptr;

    using namespace mpp::tensor_ops;
    auto A_cm = tensor<threadgroup precision_t, dextents<int32_t, 2>, tensor_inline>(
        comb, dextents<int32_t, 2>(H3, CHUNK));
    auto B_cm = tensor<device precision_t, dextents<int32_t, 2>, tensor_inline>(
        a.W, dextents<int32_t, 2>(H, H3));
    auto C_cm = tensor<device precision_t, dextents<int32_t, 2>, tensor_inline>(
        a.grad_x, dextents<int32_t, 2>(H, M));
    auto D_cm = tensor<threadgroup precision_t, dextents<int32_t, 2>, tensor_inline>(
        gin, dextents<int32_t, 2>(H, CHUNK));
    constexpr auto desc = matmul2d_descriptor(CHUNK, TILE_N,
        static_cast<int>(dynamic_extent), false, false, false);
    matmul2d<desc, execution_simdgroups<THREADS / 32>> op;
    for (int chunk = ROWS / CHUNK - 1; chunk >= 0; chunk--) {
        const int row0 = (int)tile * ROWS + chunk * CHUNK;
        // Reverse scan over the chunk; gradients go to grad_combined and to
        // the threadgroup tile the dX GEMM below reads.
        for (int lane = (int)tid; lane < nseq * H; lane += THREADS) {
            int seq = lane / H, h = lane % H;
            int b = row0 / T_seq + seq;
            int t_begin = row0 - b * T_seq + seq * steps;
            int bHT = b * H * T_seq;
            int cbase = 3 * bHT;
            int state_idx = b * H + h;
            int out_base = bHT + h;
            device const precision_t* term_row =
                a.terminals ? a.terminals + b * T_seq : (device const precision_t*)nullptr;

            float dh = t_begin + steps == T_seq ? to_float(a.grad_next_state[state_idx]) : carry[h / THREADS];
            // Independent device reads issued ahead of the serial dh chain;
            // t and all FP32 arithmetic keep the original reverse order.
            constexpr int PREFETCH = 4;
            for (int top = t_begin + steps - 1; top >= t_begin; top -= PREFETCH) {
                float hp[PREFETCH], xv[PREFETCH], go[PREFETCH], tv[PREFETCH],
                    hv[PREFETCH], gv[PREFETCH], pv[PREFETCH];
                for (int j = 0; j < PREFETCH; ++j) {
                    int t0 = top - j;
                    if (t0 >= t_begin) {
                        int i = out_base + t0 * H;
                        int c = cbase + t0 * H3 + h;
                        hp[j] = to_float(a.scan_h[i]);
                        xv[j] = to_float(a.x[i]);
                        go[j] = to_float(a.grad_out[i]);
                        tv[j] = term_row ? to_float(term_row[t0]) : 0.0f;
                        hv[j] = to_float(a.combined[c]);
                        gv[j] = to_float(a.combined[c + H]);
                        pv[j] = to_float(a.combined[c + 2 * H]);
                    }
                }
                for (int j = 0; j < PREFETCH && top - j >= t_begin; ++j) {
                    int t0 = top - j;
                    int c = cbase + t0 * H3 + h;
                    int input_idx = out_base + t0 * H;
                    int r = (seq * steps + t0 - t_begin) * H3 + h;
                    float h_prev = hp[j];
                    float hidden_val = hv[j];
                    float gate_val = gv[j];
                    float proj_val = pv[j];
                    float x_val = xv[j];
                    float grad_out_val = go[j];
                    float z = mg_sigmoid(gate_val);
                    float h_tilde = mg_h_tilde(hidden_val);
                    float h_t = mg_lerp(h_prev, h_tilde, z);
                    float s = mg_sigmoid(proj_val);
                    // highway: out = s*h + (1-s)*x
                    float grad_proj = grad_out_val * (h_t - x_val) * s * (1.0f - s);
                    precision_t gi = from_float(grad_out_val * (1.0f - s));
                    gin[(seq * steps + t0 - t_begin) * H + h] = gi;
                    if (store_gin) a.grad_input[input_idx] = gi;
                    a.grad_combined[c + 2 * H] = from_float(grad_proj);

                    float dh_total = fma(grad_out_val, s, dh);

                    float d_h_prev = dh_total * (1.0f - z);
                    float d_h_tilde = dh_total * z;
                    float d_z = dh_total * (h_tilde - h_prev);
                    float d_gate = d_z * z * (1.0f - z);
                    float d_hidden = (hidden_val >= 0.0f)
                        ? d_h_tilde
                        : d_h_tilde * h_tilde * (1.0f - h_tilde);

                    a.grad_combined[c] = from_float(d_hidden);
                    a.grad_combined[c + H] = from_float(d_gate);
                    comb[r] = from_float(d_hidden);
                    comb[r + H] = from_float(d_gate);
                    comb[r + 2 * H] = from_float(grad_proj);

                    dh = d_h_prev;
                    if (tv[j] != 0.0f) {
                        dh = 0.0f;
                    }
                }
            }
            if (t_begin == 0) a.grad_state[state_idx] = from_float(dh);
            else carry[h / THREADS] = dh;
        }
        // The epilogue reads this chunk's grad_input rows (gin) written by
        // other threads above and writes grad_x rows whose grad_out (possibly
        // the same buffer) every thread has finished reading; later chunks
        // read only their own (earlier) rows.
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        // grad_x = grad_combined @ W + grad_input (tensor_ops_gemm_nn_add).
        for (int j = 0; j < H / TILE_N; j++) {
            auto mSecond = B_cm.slice(j * TILE_N, 0);
            auto cT = op.template get_destination_cooperative_tensor<decltype(A_cm), decltype(mSecond), float>();
            #pragma unroll
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                if (cT.is_valid_element(i)) cT[i] = 0.0f;
            }
            op.run(A_cm, mSecond, cT);
#ifdef PRECISION_FLOAT
            auto mResult = C_cm.slice(j * TILE_N, row0);
            auto mAdd = D_cm.slice(j * TILE_N, 0);
            auto dT = op.template get_destination_cooperative_tensor<decltype(A_cm), decltype(mSecond), float>();
            dT.load(mAdd);
            #pragma unroll
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                if (cT.is_valid_element(i)) cT[i] += dT[i];
            }
            cT.store(mResult);
#else
            // Match bf16(bf16(G @ W) + grad_input): both generic stores round.
            #pragma unroll
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                if (!cT.is_valid_element(i)) continue;
                auto idx = cT.get_multidimensional_index(i);
                int col = j * TILE_N + idx[0];
                long ci = (long)(row0 + idx[1]) * H + col;
                a.grad_x[ci] = from_float(round_p(cT[i]) + to_float(gin[idx[1] * H + col]));
            }
#endif
        }
        // The next chunk's scan overwrites comb and gin.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Explicit entry points use the same template for both hidden widths.
#define WIDE_MINGRU(H, C, S, N, R) \
kernel void wide_mingru_forward_h##H(constant WideMingruFwdArgs& a [[buffer(0)]], \
        uint tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup precision_t comb[C * 3 * H]; \
    wide_mingru_forward_body<H, C, S, N, R>(a, tile, tid, comb); \
} \
kernel void wide_mingru_backward_h##H(constant WideMingruBwdArgs& a [[buffer(0)]], \
        uint tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup precision_t comb[C * 3 * H], gin[C * H]; \
    wide_mingru_backward_body<H, C, S, N, R>(a, tile, tid, comb, gin); \
}
WIDE_MINGRU(128, PUF_WIDE_128_CHUNK, PUF_WIDE_128_THREADS, PUF_WIDE_128_N, 64)
WIDE_MINGRU(256, PUF_WIDE_256_CHUNK, PUF_WIDE_256_THREADS, PUF_WIDE_256_N, 64)

#include "wide_muon.h"


#include "large_linear.h"

#include "large_muon_products.h"

#ifdef PUF_TEST_SIGMOID
PUF_SIG_TEST_KERNELS(wide, mg_sigmoid)
#endif

#pragma once
// Shared by rollout_forward.metal (the forward kernels) and rollout.metal
// (rollout_step_h32, which wraps the same forward with the step's casts and
// sampling). One simdgroup owns eight agents for the entire inference forward. Only
// recurrent state and final logits leave the threadgroup. Existing stream
// dispatch barriers order observation/state preparation and sample_logits.
//
// Storage is precision_t (float, or bf16 in the default build). bf16 operands
// are widened exactly on load and every value the generic path stores in
// precision_t is rounded at the same point here (round_p, identity in FP32):
// the encoder output and each projection (one RNE of the fp32 MMA sum, as the
// bf16 GEMMs store), the gate's next state and output, and the logits. The
// MMA accumulation order is the FP32 one, which matches the tensor_ops /
// tiled GEMMs (FP32: test_rollout; bf16 GEMM == RNE(f32 GEMM): test_bf16), so
// both precisions are bit-identical to the generic path (test_rollout).
#include "common.h"
#include <metal_simdgroup_matrix>

struct FusedRolloutArgs {
    device const precision_t* obs;
    device const precision_t* encoder;
    device const precision_t* recurrent[4];
    device const precision_t* decoder;
    device precision_t* state;
    device precision_t* out;
    device const float* terminals; // (B), null means no reset
    device precision_t* initial_state; // optional (layers, initial_agents, H) snapshot
    int initial_agents;
    int B, input_dim, hidden, layers, output_dim;
};

// 8x8 MMA operand loads from precision_t device memory into fp32 matrices.
static inline void rf_load(thread simdgroup_float8x8& d, device const float* p, ulong ld,
        bool transpose = false) {
    simdgroup_load(d, p, ld, ulong2(0), transpose);
}
static inline void rf_load(thread simdgroup_float8x8& d, device const bfloat* p, ulong ld,
        bool transpose = false) {
    simdgroup_bfloat8x8 b;
    simdgroup_load(b, p, ld, ulong2(0), transpose);
    thread auto& be = b.thread_elements();
    thread auto& de = d.thread_elements();
    de[0] = puf_bf16_to_f32(be[0]);
    de[1] = puf_bf16_to_f32(be[1]);
}

// Corrected sigmoid; common.h documents the device-specific exactness check.
static inline float rollout_sigmoid(float x) { return puf_sigmoid_exact(x); }

template <int H, bool DIRECT = false>
static void rollout_forward(constant FusedRolloutArgs& a, uint group, uint lane,
        threadgroup float* x, threadgroup float* combined,
        threadgroup float* stage_a, threadgroup float* stage_b, int obs_row_stride = 0) {
    int b0 = int(group) * 8;
    int K = a.input_dim;
    const int obs_stride = obs_row_stride ? obs_row_stride : K;
    // Apple 8x8 MMA lane layout. Ragged loads/stores are checked against
    // the original GEMM/gate path by test_rollout's exact comparisons.
    const int q = int(lane) / 4;
    const int fm = (q & 4) + (int(lane) / 2) % 4;
    const int fn = (q & 2) * 2 + (int(lane) % 2) * 2;
    for (int n = 0; n < H; n += 8) {
        simdgroup_float8x8 acc(0.0f);
        for (int k = 0; k < K; k += 8) {
            simdgroup_float8x8 lhs, rhs;
            if (b0 + 8 <= a.B && k + 8 <= K) {
                rf_load(lhs, a.obs + long(b0) * obs_stride + k, obs_stride);
            } else if constexpr (DIRECT) {
                int b = b0 + fm;
                for (int i = 0; i < 2; i++) {
                    int kk = k + fn + i;
                    lhs.thread_elements()[i] = b < a.B && kk < K ? to_float(a.obs[long(b) * obs_stride + kk]) : 0.0f;
                }
            } else {
                for (uint e = lane; e < 64; e += 32) {
                    int b = b0 + e / 8, kk = k + e % 8;
                    stage_a[e] = b < a.B && kk < K ? to_float(a.obs[long(b) * obs_stride + kk]) : 0.0f;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(lhs, stage_a, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (k + 8 <= K) {
                rf_load(rhs, a.encoder + n * K + k, K, true);
            } else if constexpr (DIRECT) {
                for (int i = 0; i < 2; i++) rhs.thread_elements()[i] =
                    k + fm < K ? to_float(a.encoder[(n + fn + i) * K + k + fm]) : 0.0f;
            } else {
                for (uint e = lane; e < 64; e += 32) {
                    int nn = n + e / 8, kk = k + e % 8;
                    stage_b[e] = kk < K ? to_float(a.encoder[nn * K + kk]) : 0.0f;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(rhs, stage_b, 8, ulong2(0), true);
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            simdgroup_multiply_accumulate(acc, lhs, rhs, acc);
        }
        simdgroup_store(acc, x + n, H);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#ifndef PRECISION_FLOAT
    for (uint e = lane; e < 8 * H; e += 32) x[e] = round_p(x[e]);  // bf16 encoder output
    simdgroup_barrier(mem_flags::mem_threadgroup);
#endif

    for (int l = 0; l < a.layers; l++) {
        for (int n = 0; n < 3 * H; n += 96) {
            simdgroup_float8x8 acc[12];
            for (int j = 0; j < 12; j++) acc[j] = simdgroup_float8x8(0.0f);
            for (int k = 0; k < H; k += 8) {
                simdgroup_float8x8 lhs;
                simdgroup_load(lhs, x + k, H);
                for (int j = 0; j < 12; j++) {
                    simdgroup_float8x8 rhs;
                    rf_load(rhs, a.recurrent[l] + (n + j * 8) * H + k, H, true);
                    simdgroup_multiply_accumulate(acc[j], lhs, rhs, acc[j]);
                }
            }
            for (int j = 0; j < 12; j++) simdgroup_store(acc[j], combined + n + j * 8, 3 * H);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = lane; e < 8 * H; e += 32) {
            int b = b0 + e / H, h = e % H;
            int c = (e / H) * 3 * H + h;
            // round_p: the projection as the bf16 GEMM stores it.
            float hidden = round_p(combined[c]), gate = round_p(combined[c + H]);
            float proj = round_p(combined[c + 2 * H]);
            long si = (long(l) * a.B + b) * H + h;
            float state = b < a.B ? to_float(a.state[si]) : 0.0f;
            if (b < a.B && a.terminals && a.terminals[b] != 0.0f) state = 0.0f;
            if (b < a.B && a.initial_state) {
                a.initial_state[(long(l) * a.initial_agents + b) * H + h] = from_float(state);
            }
            float z = rollout_sigmoid(gate);
            float ht = hidden >= 0.0f ? hidden + 0.5f : rollout_sigmoid(hidden);
            float diff = ht - state;
            float next = fabs(z) < 0.5f ? fma(z, diff, state) : fma(-diff, 1.0f - z, ht);
            if (b < a.B) a.state[si] = from_float(next);
            float s = rollout_sigmoid(proj);
            // mingru_gate: out from the unrounded next state, then stored.
            x[e] = round_p(fma(s, next, (1.0f - s) * x[e]));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int n = 0; n < a.output_dim; n += 8) {
        simdgroup_float8x8 acc(0.0f);
        for (int k = 0; k < H; k += 8) {
            simdgroup_float8x8 lhs, rhs;
            simdgroup_load(lhs, x + k, H);
            if (n + 8 <= a.output_dim) {
                rf_load(rhs, a.decoder + n * H + k, H, true);
            } else if constexpr (DIRECT) {
                for (int i = 0; i < 2; i++) rhs.thread_elements()[i] =
                    n + fn + i < a.output_dim ? to_float(a.decoder[(n + fn + i) * H + k + fm]) : 0.0f;
            } else {
                for (uint e = lane; e < 64; e += 32) {
                    int nn = n + e / 8;
                    stage_b[e] = nn < a.output_dim ? to_float(a.decoder[nn * H + k + e % 8]) : 0.0f;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(rhs, stage_b, 8, ulong2(0), true);
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            simdgroup_multiply_accumulate(acc, lhs, rhs, acc);
        }
        if constexpr (DIRECT) {
            int b = b0 + fm;
            for (int i = 0; i < 2; i++) {
                int nn = n + fn + i;
                if (b < a.B && nn < a.output_dim) a.out[long(b) * a.output_dim + nn] = from_float(acc.thread_elements()[i]);
            }
        } else {
            simdgroup_store(acc, stage_a, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                int b = b0 + e / 8, nn = n + e % 8;
                if (b < a.B && nn < a.output_dim) a.out[long(b) * a.output_dim + nn] = from_float(stage_a[e]);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}


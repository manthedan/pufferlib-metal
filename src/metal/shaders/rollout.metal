// Ports of the rollout-path CUDA kernels: rng_init, sample_logits,
// snapshot_state, zero_term_state (src/pufferl.cu) and mingru_gate
// (src/algo.cu), plus a device implementation of cuRAND's Philox4x32-10
// generator that reproduces curand_init / curand / curand_uniform /
// curand_normal on the same 64-byte state layout.
#include "common.h"

// ---- cuRAND Philox4x32-10 ---------------------------------------------------
// Layout of curandStatePhilox4_32_10_t (64 bytes): ctr@0, output@16, key@32,
// STATE@40, boxmuller_flag@44, boxmuller_flag_double@48, boxmuller_extra@52,
// boxmuller_extra_double@56 (no doubles in MSL: kept as raw bits).
struct PhiloxState {
    uint ctr[4];
    uint output[4];
    uint key[2];
    uint STATE;
    int boxmuller_flag;
    int boxmuller_flag_double;
    float boxmuller_extra;
    uint boxmuller_extra_double[2];
};
static_assert(sizeof(PhiloxState) == 64, "philox state layout");

#define PHILOX_W32_0 0x9E3779B9u
#define PHILOX_W32_1 0xBB67AE85u
#define PHILOX_M4x32_0 0xD2511F53u
#define PHILOX_M4x32_1 0xCD9E8D57u
#define CURAND_2POW32_INV 2.3283064e-10f
#define CURAND_2POW32_INV_2PI (2.3283064e-10f * 6.2831855f)

static inline uint4 philox_round(uint4 c, uint2 k) {
    uint hi0 = mulhi(PHILOX_M4x32_0, c.x), lo0 = PHILOX_M4x32_0 * c.x;
    uint hi1 = mulhi(PHILOX_M4x32_1, c.z), lo1 = PHILOX_M4x32_1 * c.z;
    return uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
}

static inline uint4 philox4x32_10(uint4 c, uint2 k) {
    c = philox_round(c, k);
    for (int i = 0; i < 9; i++) {
        k.x += PHILOX_W32_0;
        k.y += PHILOX_W32_1;
        c = philox_round(c, k);
    }
    return c;
}

// Thread-private working copy of the state.
struct Philox {
    uint4 ctr;
    uint4 output;
    uint2 key;
    uint STATE;
    int boxmuller_flag;
    float boxmuller_extra;
};

static inline Philox philox_load(device const PhiloxState& s) {
    Philox p;
    p.ctr = uint4(s.ctr[0], s.ctr[1], s.ctr[2], s.ctr[3]);
    p.output = uint4(s.output[0], s.output[1], s.output[2], s.output[3]);
    p.key = uint2(s.key[0], s.key[1]);
    p.STATE = s.STATE;
    p.boxmuller_flag = s.boxmuller_flag;
    p.boxmuller_extra = s.boxmuller_extra;
    return p;
}

// Writes back every field sample_logits / curand_init can change. The double
// Box-Muller cache is untouched (no double draws on this path).
static inline void philox_store(device PhiloxState& s, thread const Philox& p) {
    for (int i = 0; i < 4; i++) {
        s.ctr[i] = p.ctr[i];
        s.output[i] = p.output[i];
    }
    s.key[0] = p.key.x;
    s.key[1] = p.key.y;
    s.STATE = p.STATE;
    s.boxmuller_flag = p.boxmuller_flag;
    s.boxmuller_extra = p.boxmuller_extra;
}

static inline void philox_incr(thread Philox& p) {
    if (++p.ctr.x) return;
    if (++p.ctr.y) return;
    if (++p.ctr.z) return;
    ++p.ctr.w;
}

static inline uint curand_u32(thread Philox& p) {
    uint ret;
    switch (p.STATE++) {
        default: ret = p.output.x; break;
        case 1: ret = p.output.y; break;
        case 2: ret = p.output.z; break;
        case 3: ret = p.output.w; break;
    }
    if (p.STATE == 4) {
        philox_incr(p);
        p.output = philox4x32_10(p.ctr, p.key);
        p.STATE = 0;
    }
    return ret;
}

// nvcc contracts x * INV + INV/2 into one FMA.
static inline float curand_uniform(thread Philox& p) {
    return fma((float)curand_u32(p), CURAND_2POW32_INV, CURAND_2POW32_INV / 2.0f);
}

static inline float curand_normal(thread Philox& p) {
    if (p.boxmuller_flag != 1) {
        uint x = curand_u32(p);
        uint y = curand_u32(p);
        float u = fma((float)x, CURAND_2POW32_INV, CURAND_2POW32_INV / 2.0f);
        float v = fma((float)y, CURAND_2POW32_INV_2PI, CURAND_2POW32_INV_2PI / 2.0f);
        float s = sqrt(-2.0f * log(u));
        float c;
        float sn = sincos(v, c);
        p.boxmuller_extra = c * s;
        p.boxmuller_flag = 1;
        return sn * s;
    }
    p.boxmuller_flag = 0;
    return p.boxmuller_extra;
}

// curand_init(seed, subsequence, offset=0, state)
struct RngInitArgs { device PhiloxState* states; ulong seed; int n; };
kernel void rng_init(constant RngInitArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= a.n) {
        return;
    }
    Philox p;
    p.key = uint2((uint)a.seed, (uint)(a.seed >> 32));
    // skipahead_sequence(idx): subsequence lives in ctr.z/ctr.w.
    ulong sub = (ulong)idx;
    uint nlo = (uint)sub, nhi = (uint)(sub >> 32);
    p.ctr = uint4(0, 0, nlo, nhi);
    p.STATE = 0;
    p.boxmuller_flag = 0;
    p.boxmuller_extra = 0.0f;
    p.output = philox4x32_10(p.ctr, p.key);
    device PhiloxState& s = a.states[idx];
    philox_store(s, p);
    s.boxmuller_flag_double = 0;
    s.boxmuller_extra_double[0] = 0;
    s.boxmuller_extra_double[1] = 0;
}

// ---- PPO head helpers (src/algo.cu) ------------------------------------------

static inline float finite_or_clamp(float x, float lo, float hi) {
    if (isnan(x)) {
        return 0.0f;
    }
    if (isinf(x)) {
        return x > 0.0f ? hi : lo;
    }
    return fmin(hi, fmax(lo, x));
}

static inline float load_logit_masked(device const precision_t* logits, int logits_base,
        int logits_offset, int a, device const precision_t* mask, int mask_base) {
    float l = to_float(logits[logits_base + logits_offset + a]);
    float m = to_float(mask[mask_base + logits_offset + a]);
    if (m == 0.0f) {
        l = -1e4f;
    }
    return l;
}

// Same recurrence as ppo_discrete_logsumexp; the per-class cache is
// recomputed by the caller (load_logit_masked is pure), so no
// PPO_MAX_HEAD_A-sized private array is needed.
static inline float ppo_discrete_logsumexp(device const precision_t* logits, int logits_base,
        int logits_offset, int A, device const precision_t* mask, int mask_base) {
    float max_logit = -INFINITY;
    float sum = 0.0f;
    for (int a = 0; a < A; ++a) {
        float l = load_logit_masked(logits, logits_base, logits_offset, a, mask, mask_base);
        if (l > max_logit) {
            sum *= exp(max_logit - l);
            max_logit = l;
        }
        sum += exp(l - max_logit);
    }
    return max_logit + log(sum);
}

static inline void ppo_continuous_head(float mean, float log_std, float action,
        thread float* out_logp, thread float* out_entropy) {
    constexpr float HALF_LOG_2PI = 0.9189385332046727f;
    constexpr float HALF_1_PLUS_LOG_2PI = 1.4189385332046727f;
    float std = exp(log_std);
    float normalized = (action - mean) / std;
    *out_logp = -0.5f * normalized * normalized - HALF_LOG_2PI - log_std;
    *out_entropy = HALF_1_PLUS_LOG_2PI + log_std;
}

// ---- sample_logits (pufferl.cu) ----------------------------------------------
// Host adapter resolves B/fused_cols/is_continuous from the Prec views.
// NetHack uses its specialized adapters and envs_nethack_policy.metal kernels.
struct SampleLogitsArgs {
    device const precision_t* logits;   // (B, fused_cols)
    device const precision_t* logstd;   // (num_atns,) continuous only
    device const int* act_sizes;
    device float* actions;
    device float* env_actions;
    device precision_t* logprobs;
    device precision_t* value_out;
    device PhiloxState* rng_states;
    device const precision_t* action_mask;
    int B;
    int fused_cols;
    int num_atns;
    int mask_stride;
    int is_continuous;
};
// sample_logits for agent idx (< a.B); also run by rollout_step_h32.
static inline void sample_one(constant SampleLogitsArgs& a, int idx) {
    int fused_cols = a.fused_cols;
    int num_atns = a.num_atns;
    device const precision_t* logits = a.logits;

    Philox state = philox_load(a.rng_states[idx]);
    int logits_base = idx * fused_cols;
    float total_log_prob = 0.0f;

    if (a.is_continuous) {
        for (int h = 0; h < num_atns; h++) {
            float mean = finite_or_clamp(to_float(logits[logits_base + h]), -1.0e6f, 1.0e6f);
            float log_std = finite_or_clamp(to_float(a.logstd[h]), -20.0f, 2.0f);
            float std = exp(log_std);
            float action = finite_or_clamp(
                mean + std * curand_normal(state), -1.0e6f, 1.0e6f);
            precision_t stored_p = from_float(action);
            float stored = to_float(stored_p);
            float lp, ent;
            ppo_continuous_head(mean, log_std, stored, &lp, &ent);
            total_log_prob += lp;
            int aidx = idx * num_atns + h;
            a.actions[aidx] = stored;
            a.env_actions[aidx] = stored;
        }
    } else {
        int logits_offset = 0;
        int mask_base = idx * a.mask_stride;
        device const precision_t* action_mask = a.action_mask;
        for (int h = 0; h < num_atns; h++) {
            int A = a.act_sizes[h];
            float logsumexp = ppo_discrete_logsumexp(
                logits, logits_base, logits_offset, A, action_mask, mask_base);
            float rand_val = curand_uniform(state);
            float cumsum = 0.0f;
            int sampled = A - 1;
            for (int c = 0; c < A; c++) {
                float l = load_logit_masked(
                    logits, logits_base, logits_offset, c, action_mask, mask_base);
                cumsum += exp(l - logsumexp);
                if (rand_val < cumsum) {
                    sampled = c;
                    break;
                }
            }
            if (sampled == A - 1) {
                for (int c = A - 1; c >= 0; c--) {
                    if (to_float(action_mask[mask_base + logits_offset + c]) != 0.0f) {
                        sampled = c;
                        break;
                    }
                }
            }
            int aidx = idx * num_atns + h;
            float action = (float)sampled;
            a.actions[aidx] = action;
            a.env_actions[aidx] = action;
            float l_s = load_logit_masked(
                logits, logits_base, logits_offset, sampled, action_mask, mask_base);
            total_log_prob += l_s - logsumexp;
            logits_offset += A;
        }
    }

    a.logprobs[idx] = from_float(total_log_prob);
    a.value_out[idx] = logits[logits_base + fused_cols - 1];
    philox_store(a.rng_states[idx], state);
}

kernel void sample_logits(constant SampleLogitsArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= a.B) {
        return;
    }
    sample_one(a, idx);
}

// ---- recurrent state bookkeeping (pufferl.cu) --------------------------------

static inline long state_elem_idx(int layer, int agents_stride, int state_row, int h, int H) {
    return ((long)layer * agents_stride + state_row) * H + h;
}

// L/H from src.shape[0]/[2], agent strides from shape[1] of each view.
struct SnapshotStateArgs {
    device precision_t* dst;
    device const precision_t* src;
    int L, H, src_agents, dst_agents;
    int dst_start, count;
};
kernel void snapshot_state(constant SnapshotStateArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    int L = a.L, H = a.H, count = a.count;
    int total = L * count * H;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int h = idx % H;
    int rel = (idx / H) % count;
    int layer = idx / (count * H);
    long src_i = state_elem_idx(layer, a.src_agents, rel, h, H);
    long dst_i = state_elem_idx(layer, a.dst_agents, a.dst_start + rel, h, H);
    a.dst[dst_i] = a.src[src_i];
}

struct ZeroTermStateArgs {
    device precision_t* state;
    device const float* terminals;
    int L, H, agents;
    int state_start, terminal_start, count;
};
kernel void zero_term_state(constant ZeroTermStateArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    int L = a.L, H = a.H, count = a.count;
    int total = L * count * H;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int h = idx % H;
    int rel = (idx / H) % count;
    int layer = idx / (count * H);
    if (a.terminals[a.terminal_start + rel] == 0.0f) {
        return;
    }
    long i = state_elem_idx(layer, a.agents, a.state_start + rel, h, H);
    a.state[i] = from_float(0.0f);
}

// ---- mingru_gate (algo.cu) ---------------------------------------------------

// Corrected sigmoid; common.h documents the device-specific exactness check.
static inline float puf_sigmoid(float x) { return puf_sigmoid_exact(x); }

static inline float puf_lerp(float a, float b, float w) {
    float diff = b - a;
    return (fabs(w) < 0.5f) ? fma(w, diff, a) : fma(-diff, 1.0f - w, b);
}

struct MingruGateArgs {
    device precision_t* out;
    device precision_t* next_state;
    device const precision_t* combined;
    device const precision_t* state_in;
    device const precision_t* x_in;
    int H, B;
};
kernel void mingru_gate(constant MingruGateArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H = a.H;
    int N = a.B * H;
    if (idx >= N) {
        return;
    }
    int b = idx / H;
    int h = idx % H;

    int combined_base = b * 3 * H;
    float hidden = to_float(a.combined[combined_base + h]);
    float gate = to_float(a.combined[combined_base + H + h]);
    float proj = to_float(a.combined[combined_base + 2 * H + h]);
    float state = to_float(a.state_in[idx]);
    float x = to_float(a.x_in[idx]);

    float z = puf_sigmoid(gate);
    float h_tilde = (hidden >= 0.0f) ? hidden + 0.5f : puf_sigmoid(hidden);
    float h_out = puf_lerp(state, h_tilde, z);
    a.next_state[idx] = from_float(h_out);

    float s = puf_sigmoid(proj);
    a.out[idx] = from_float(fma(s, h_out, (1.0f - s) * x));
}

// ---- test hook ---------------------------------------------------------------
// tests/metal/test_kernels.mm: `draws` curand_uniform values from a private
// copy of each state (states are not modified). Not used by the trainer.
struct CurandUniformProbeArgs {
    device const PhiloxState* states;
    device float* out;  // (n, draws)
    int n, draws;
};
kernel void puf_test_curand_uniform(constant CurandUniformProbeArgs& a [[buffer(0)]],
        PUF_KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= a.n) {
        return;
    }
    Philox p = philox_load(a.states[idx]);
    for (int d = 0; d < a.draws; d++) {
        a.out[(long)idx * a.draws + d] = curand_uniform(p);
    }
}

// ---- rollout_step_h32 ---------------------------------------------------------
// One whole rollout step for the standard H=32 policy in a single dispatch
// (pufferl.cu pufferl_forward_step): under learner contention every extra
// dispatch on the rollout queue waited ~0.1-0.2 ms for a GPU slot. Each
// simdgroup owns eight agents b0..b0+7 and, for those agents only, runs
//   the env -> rollout obs copy (precision_t: bitwise, as cudaMemcpyAsync;
//   u8: cast_u8_to_prec; float into bf16: cast_f32_to_prec), cast_rew_term, the u8 action-mask cast_u8_to_prec,
//   rollout_forward<32, true> (with its terminal reset and
//   initial-state snapshot), then sample_logits (sample_one on the logits it
//   just wrote),
// with the original per-element arithmetic, so results are bit-identical.
// Every buffer row written or read belongs to the simdgroup's own agents.
// Host side: rollout.h rollout_step.
#include "rollout_forward.h"

struct RolloutStepArgs {
    FusedRolloutArgs fwd;          // fwd.obs: precision_t rollout obs rows (written here)
    SampleLogitsArgs smp;          // smp.logits == fwd.out, smp.B == fwd.B
    device const void* obs_src;    // (B, input_dim) env observations (precision_t, uchar or float)
    device precision_t* rew_dst;   // (B)
    device const float* rew_src;
    device precision_t* term_dst;
    device const float* term_src;
    device precision_t* mask_dst;  // (B, mask_size) == smp.action_mask
    device const uchar* mask_src;
    int mask_size;
    int obs_is_u8;      // 0 precision_t, 1 uchar, 2 float (BF16 builds)
    int obs_row_stride; // destination/forward row stride; 0 means input_dim
    int pad;
};

// S simdgroups per threadgroup; H=32 keeps every activation tile in
// registers / threadgroup memory without staging (DIRECT), H=64 stages
// ragged tiles as rollout_forward_h64 does.
template <int H, int S, bool DIRECT>
static void rollout_step_body(constant RolloutStepArgs& a, uint group, uint lane, uint sg,
        threadgroup float* x, threadgroup float* combined,
        threadgroup float* stage_a, threadgroup float* stage_b) {
    constant FusedRolloutArgs& f = a.fwd;
    // Uniform guards (whole grid returns together, so barriers stay safe):
    // the forward's own, plus the sampler / casts' shape assumptions.
    if (f.hidden != H || f.layers < 1 || f.layers > 4 || f.B <= 0
            || f.input_dim < 1 || f.input_dim > 256 || f.output_dim < 1 || f.output_dim > 32
            || (f.initial_state && f.initial_agents < f.B)
            || (a.obs_row_stride != 0 && a.obs_row_stride < f.input_dim)
            || a.smp.B != f.B || a.smp.fused_cols != f.output_dim || a.smp.is_continuous
            || a.mask_size < 0 || a.mask_size > 256 || a.smp.mask_stride != a.mask_size) return;
    const int B = f.B, K = f.input_dim, M = a.mask_size;
    const int obs_stride = a.obs_row_stride ? a.obs_row_stride : K;
    const int b0 = int(group * S + sg) * 8;
    device precision_t* obs = (device precision_t*)f.obs;
    if (a.obs_is_u8 == 2) {
        // float env observations into bf16 rollout rows: the generic cast kernel.
        device const float* src = (device const float*)a.obs_src;
        for (int e = int(lane); e < 8 * K; e += 32) {     // <= 64 iterations
            int b = b0 + e / K;
            if (b < B) obs[long(b) * obs_stride + e % K] = from_float(src[long(b) * K + e % K]);
        }
    } else if (a.obs_is_u8) {
        device const uchar* src = (device const uchar*)a.obs_src;
        for (int e = int(lane); e < 8 * K; e += 32) {     // <= 64 iterations
            int b = b0 + e / K;
            if (b < B) obs[long(b) * obs_stride + e % K] = from_float((float)src[long(b) * K + e % K]);
        }
    } else {
        // Same element type as the rollout buffer: a bitwise copy.
        device const precision_t* src = (device const precision_t*)a.obs_src;
        device precision_t* dst = obs;
        for (int e = int(lane); e < 8 * K; e += 32) {     // <= 64 iterations
            int b = b0 + e / K;
            if (b < B) dst[long(b) * obs_stride + e % K] = src[long(b) * K + e % K];
        }
    }
    for (int e = int(lane); e < 8 * M; e += 32) {     // <= 64 iterations
        int b = b0 + e / M;
        if (b < B) a.mask_dst[long(b) * M + e % M] = from_float((float)a.mask_src[long(b) * M + e % M]);
    }
    if (lane < 8 && b0 + int(lane) < B) {
        int b = b0 + int(lane);
        a.rew_dst[b] = from_float(a.rew_src[b]);
        a.term_dst[b] = from_float(a.term_src[b]);
    }
    threadgroup_barrier(mem_flags::mem_device);
    rollout_forward<H, DIRECT>(f, group * S + sg, lane, x, combined, stage_a, stage_b, obs_stride);
    threadgroup_barrier(mem_flags::mem_device);
    if (lane < 8 && b0 + int(lane) < B) sample_one(a.smp, b0 + int(lane));
}

kernel void rollout_step_h32(constant RolloutStepArgs& a [[buffer(0)]],
        uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float x[4][8 * 32], combined[4][8 * 3 * 32];
    rollout_step_body<32, 4, true>(a, group, lane, sg, x[sg], combined[sg], combined[sg],
        combined[sg] + 64);
}

kernel void rollout_step_h64(constant RolloutStepArgs& a [[buffer(0)]],
        uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float x[2][8 * 64], combined[2][8 * 3 * 64], stage_a[2][64], stage_b[2][64];
    rollout_step_body<64, 2, false>(a, group, lane, sg, x[sg], combined[sg], stage_a[sg],
        stage_b[sg]);
}


// ---- Packed-weight rollout step (rollout.h rollout_pack / rollout_step_packed)
//
// rollout_pack_weights widens (bf16) or copies (FP32) the policy's encoder,
// recurrent and decoder weights once per rollout into float 8x8 MMA tiles,
// already in the right-hand-operand orientation: tile element [r][c] (row-major
// 64 floats) is W[n0 + c][k0 + r], zero past the encoder's input width or the
// decoder's output width, exactly the values the staged/transposed loads of
// rollout_forward produce. The packed step loads each tile with one plain
// float simdgroup_load instead of a transposed bf16 load + per-element widen.
//
// The step itself is rollout_step_body's arithmetic, regrouped:
// - P simdgroups share one 8-agent group: each owns H/(8P) hidden tiles of
//   the encoder output and of every layer (its hidden, gate and projection
//   column tiles), so the group's MMA chain is P times shorter;
// - every output tile still accumulates k = 0, 8, ... in the same order with
//   the same operands (one MMA per k-tile), so all sums are bit-identical;
// - the gate runs on the projection accumulators in registers (the three
//   accumulators of a hidden tile hold the same (agent, h) per lane), reading
//   the layer input from one threadgroup buffer and writing the next layer's
//   input to the other.
// Per-element gate/state/snapshot/logit arithmetic and rounding points are
// rollout_forward's.

struct RolloutPackArgs {
    device const precision_t* encoder;   // (H, K)
    device const precision_t* recurrent[4]; // (3H, H) each
    device const precision_t* decoder;   // (D, H)
    device float* packed;
    int K, H, L, D;
};

kernel void rollout_pack_weights(constant RolloutPackArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
    const int KT = (a.K + 7) / 8, NT = a.H / 8, DT = (a.D + 7) / 8;
    const long enc_n = long(NT) * KT * 64, rec_n = long(3 * NT) * NT * 64;
    const long dec_n = long(DT) * NT * 64;
    if (a.H % 8 || a.L < 1 || a.L > 4 || i >= enc_n + a.L * rec_n + dec_n) return;
    const int r = int(i % 64) / 8, c = int(i % 8);
    float v;
    if (i < enc_n) {
        long tile = i / 64;
        int n = int(tile / KT) * 8 + c, k = int(tile % KT) * 8 + r;
        v = k < a.K ? to_float(a.encoder[long(n) * a.K + k]) : 0.0f;
    } else if (i < enc_n + a.L * rec_n) {
        long t = i - enc_n;
        int l = int(t / rec_n);
        long tile = (t % rec_n) / 64;
        int n = int(tile / NT) * 8 + c, k = int(tile % NT) * 8 + r;
        v = to_float(a.recurrent[l][long(n) * a.H + k]);
    } else {
        long tile = (i - enc_n - a.L * rec_n) / 64;
        int n = int(tile / NT) * 8 + c, k = int(tile % NT) * 8 + r;
        v = n < a.D ? to_float(a.decoder[long(n) * a.H + k]) : 0.0f;
    }
    a.packed[i] = v;
}

// P simdgroups per 8-agent group, G groups per threadgroup.
template <int H, int P, int G>
static void rollout_step_packed_body(constant RolloutStepArgs& a, uint tg, uint lane, uint sgi,
        threadgroup float* xa, threadgroup float* xb, threadgroup float* stage) {
    constant FusedRolloutArgs& f = a.fwd;
    if (f.hidden != H || f.layers < 1 || f.layers > 4 || f.B <= 0
            || f.input_dim < 1 || f.input_dim > 256 || f.output_dim < 1 || f.output_dim > 32
            || (f.initial_state && f.initial_agents < f.B)
            || (a.obs_row_stride != 0 && a.obs_row_stride < f.input_dim)
            || a.smp.B != f.B || a.smp.fused_cols != f.output_dim || a.smp.is_continuous
            || a.mask_size < 0 || a.mask_size > 256 || a.smp.mask_stride != a.mask_size) return;
    constexpr int NT = H / 8, NP = NT / P;
    const int B = f.B, K = f.input_dim, M = a.mask_size, D = f.output_dim;
    const int KT = (K + 7) / 8;
    const int obs_stride = a.obs_row_stride ? a.obs_row_stride : K;
    const int j = int(sgi) % P;
    const int b0 = int(tg * G + sgi / P) * 8;
    const int part = int(lane) + 32 * j;
    device precision_t* obs = (device precision_t*)f.obs;
    if (a.obs_is_u8 == 2) {
        device const float* src = (device const float*)a.obs_src;
        for (int e = part; e < 8 * K; e += 32 * P) {
            int b = b0 + e / K;
            if (b < B) obs[long(b) * obs_stride + e % K] = from_float(src[long(b) * K + e % K]);
        }
    } else if (a.obs_is_u8) {
        device const uchar* src = (device const uchar*)a.obs_src;
        for (int e = part; e < 8 * K; e += 32 * P) {
            int b = b0 + e / K;
            if (b < B) obs[long(b) * obs_stride + e % K] = from_float((float)src[long(b) * K + e % K]);
        }
    } else {
        device const precision_t* src = (device const precision_t*)a.obs_src;
        for (int e = part; e < 8 * K; e += 32 * P) {
            int b = b0 + e / K;
            if (b < B) obs[long(b) * obs_stride + e % K] = src[long(b) * K + e % K];
        }
    }
    for (int e = part; e < 8 * M; e += 32 * P) {
        int b = b0 + e / M;
        if (b < B) a.mask_dst[long(b) * M + e % M] = from_float((float)a.mask_src[long(b) * M + e % M]);
    }
    if (j == 0 && lane < 8 && b0 + int(lane) < B) {
        int b = b0 + int(lane);
        a.rew_dst[b] = from_float(a.rew_src[b]);
        a.term_dst[b] = from_float(a.term_src[b]);
    }
    threadgroup_barrier(mem_flags::mem_device);

    const int q = int(lane) / 4;
    const int fm = (q & 4) + (int(lane) / 2) % 4;
    const int fn = (q & 2) * 2 + (int(lane) % 2) * 2;
    device const float* enc = (device const float*)f.encoder;
    {
        simdgroup_float8x8 acc[NP];
        for (int t = 0; t < NP; t++) acc[t] = simdgroup_float8x8(0.0f);
        for (int kt = 0; kt < KT; kt++) {
            const int k = kt * 8;
            simdgroup_float8x8 lhs;
            if (b0 + 8 <= B && k + 8 <= K) {
                rf_load(lhs, obs + long(b0) * obs_stride + k, obs_stride);
            } else {
                for (uint e = lane; e < 64; e += 32) {
                    int b = b0 + e / 8, kk = k + e % 8;
                    stage[e] = b < B && kk < K ? to_float(obs[long(b) * obs_stride + kk]) : 0.0f;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(lhs, stage, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            for (int t = 0; t < NP; t++) {
                simdgroup_float8x8 rhs;
                simdgroup_load(rhs, enc + (long(j * NP + t) * KT + kt) * 64, 8);
                simdgroup_multiply_accumulate(acc[t], lhs, rhs, acc[t]);
            }
        }
        for (int t = 0; t < NP; t++) {
            thread auto& e = acc[t].thread_elements();
            e[0] = round_p(e[0]);  // bf16 encoder output
            e[1] = round_p(e[1]);
            simdgroup_store(acc[t], xa + (j * NP + t) * 8, H);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* xin = xa;
    threadgroup float* xout = xb;
    const int b = b0 + fm;
    for (int l = 0; l < f.layers; l++) {
        device const float* W = (device const float*)f.recurrent[l];
        simdgroup_float8x8 ah[NP], ag[NP], ap[NP];
        for (int t = 0; t < NP; t++) {
            ah[t] = simdgroup_float8x8(0.0f);
            ag[t] = simdgroup_float8x8(0.0f);
            ap[t] = simdgroup_float8x8(0.0f);
        }
        for (int kt = 0; kt < NT; kt++) {
            simdgroup_float8x8 lhs, rhs;
            simdgroup_load(lhs, xin + kt * 8, H);
            for (int t = 0; t < NP; t++) {
                const int nt = j * NP + t;
                simdgroup_load(rhs, W + (long(nt) * NT + kt) * 64, 8);
                simdgroup_multiply_accumulate(ah[t], lhs, rhs, ah[t]);
                simdgroup_load(rhs, W + (long(NT + nt) * NT + kt) * 64, 8);
                simdgroup_multiply_accumulate(ag[t], lhs, rhs, ag[t]);
                simdgroup_load(rhs, W + (long(2 * NT + nt) * NT + kt) * 64, 8);
                simdgroup_multiply_accumulate(ap[t], lhs, rhs, ap[t]);
            }
        }
        for (int t = 0; t < NP; t++) {
            for (int i = 0; i < 2; i++) {
                const int h = (j * NP + t) * 8 + fn + i;
                // round_p: the projection as the bf16 GEMM stores it.
                float hidden = round_p(ah[t].thread_elements()[i]);
                float gate = round_p(ag[t].thread_elements()[i]);
                float proj = round_p(ap[t].thread_elements()[i]);
                long si = (long(l) * B + b) * H + h;
                float state = b < B ? to_float(f.state[si]) : 0.0f;
                if (b < B && f.terminals && f.terminals[b] != 0.0f) state = 0.0f;
                if (b < B && f.initial_state) {
                    f.initial_state[(long(l) * f.initial_agents + b) * H + h] = from_float(state);
                }
                float z = rollout_sigmoid(gate);
                float ht = hidden >= 0.0f ? hidden + 0.5f : rollout_sigmoid(hidden);
                float diff = ht - state;
                float next = fabs(z) < 0.5f ? fma(z, diff, state) : fma(-diff, 1.0f - z, ht);
                if (b < B) f.state[si] = from_float(next);
                float s = rollout_sigmoid(proj);
                // mingru_gate: out from the unrounded next state, then stored.
                xout[fm * H + h] = round_p(fma(s, next, (1.0f - s) * xin[fm * H + h]));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float* tmp = xin;
        xin = xout;
        xout = tmp;
    }
    device const float* dec = (device const float*)f.decoder;
    for (int dt = j; dt * 8 < D; dt += P) {
        simdgroup_float8x8 acc(0.0f);
        for (int kt = 0; kt < NT; kt++) {
            simdgroup_float8x8 lhs, rhs;
            simdgroup_load(lhs, xin + kt * 8, H);
            simdgroup_load(rhs, dec + (long(dt) * NT + kt) * 64, 8);
            simdgroup_multiply_accumulate(acc, lhs, rhs, acc);
        }
        for (int i = 0; i < 2; i++) {
            int nn = dt * 8 + fn + i;
            if (b < B && nn < D) f.out[long(b) * D + nn] = from_float(acc.thread_elements()[i]);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (j == 0 && lane < 8 && b0 + int(lane) < B) sample_one(a.smp, b0 + int(lane));
}

#define ROLLOUT_STEP_PACKED(H, P, G) \
kernel void rollout_step_packed_h##H##_p##P(constant RolloutStepArgs& a [[buffer(0)]], \
        uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], \
        uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup float xa[G][8 * H], xb[G][8 * H], stage[G * P][64]; \
    rollout_step_packed_body<H, P, G>(a, tg, lane, sg, xa[sg / P], xb[sg / P], stage[sg]); \
}
ROLLOUT_STEP_PACKED(64, 1, 2)
ROLLOUT_STEP_PACKED(64, 2, 2)
ROLLOUT_STEP_PACKED(64, 4, 1)
ROLLOUT_STEP_PACKED(32, 1, 4)
ROLLOUT_STEP_PACKED(32, 2, 2)

#ifdef PUF_TEST_SIGMOID
PUF_SIG_TEST_KERNELS(rollout, puf_sigmoid)
#endif

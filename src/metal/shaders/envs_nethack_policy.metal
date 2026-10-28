// NetHack variants of sampling/importance/PPO. Core shaders stay independent.
// Ports of the rollout-path CUDA kernels: rng_init, sample_logits,
// snapshot_state, zero_term_state (src/pufferl.cu) and mingru_gate
// (src/algo.cu), plus a device implementation of cuRAND's Philox4x32-10
// generator that reproduces curand_init / curand / curand_uniform /
// curand_normal on the same 64-byte state layout.
// precision_t storage, with FP32 arithmetic and explicit CUDA-compatible rounding.
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


static inline bool nh_head_used(device const char* hc, int stride, int verb, int h) {
    return h == 0 || !hc || hc[verb * stride + h] != 0;
}
static inline float nh_eps(device const float* veps, device const precision_t* mask,
        int A, thread float& inv_K) {
    inv_K = 0.0f;
    float eps = veps ? *veps : 0.0f;
    if (eps <= 0.0f) return 0.0f;
    int K = 0;
    for (int j=0; j<A; j++) K += to_float(mask[j]) != 0.0f;
    inv_K = 1.0f / float(K ? K : A);
    return eps;
}
static inline float nh_train_logp(device const float* veps, device const precision_t* mask,
        int A, float lp, thread float& scale) {
    float inv_K;
    float eps = nh_eps(veps, mask, A, inv_K);
    if (eps <= 0.0f) return lp;
    float p_act = exp(lp);
    float mixed = (1.0f - eps) * p_act + eps * inv_K;
    scale = (1.0f - eps) * p_act / mixed;
    return log(mixed);
}

// ---- sample_logits (pufferl.cu) ----------------------------------------------
// Host adapter resolves B/fused_cols/is_continuous from the Prec views.
// Verb-conditional heads and epsilon mixture mirror the CUDA NetHack policy.
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
    device const char* hc;
    device const float* veps;
    int hc_stride;
};
kernel void nh_sample_logits(constant SampleLogitsArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= a.B) {
        return;
    }
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
            float inv_K = 0.0f;
            float eps = h == 0 ? nh_eps(a.veps, action_mask + mask_base + logits_offset, A, inv_K) : 0.0f;
            float rand_val = curand_uniform(state);
            float cumsum = 0.0f;
            int sampled = A - 1;
            for (int c = 0; c < A; c++) {
                float l = load_logit_masked(
                    logits, logits_base, logits_offset, c, action_mask, mask_base);
                float prob = exp(l - logsumexp);
                cumsum += eps > 0.0f ? (1.0f-eps)*prob + eps*(action_mask[mask_base+logits_offset+c] != 0.0f ? 1.0f : 0.0f)*inv_K : prob;
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
            int verb = int(a.actions[idx*num_atns]);
            if (nh_head_used(a.hc, a.hc_stride, verb, h))
                total_log_prob += eps > 0.0f ? log((1.0f-eps)*exp(l_s-logsumexp)+eps*inv_K) : l_s-logsumexp;
            logits_offset += A;
        }
    }

    a.logprobs[idx] = from_float(total_log_prob);
    a.value_out[idx] = logits[logits_base + fused_cols - 1];
    philox_store(a.rng_states[idx], state);
}

static inline float ppo_finite_or_clamp(float x, float lo, float hi) {
    if (isnan(x)) return 0.0f;
    return fmin(fmax(x, lo), hi);
}
static inline float ppo_masked(device const precision_t* logits, device const precision_t* mask, int l, int m) {
    return to_float(mask[m]) == 0.0f ? -1.0e4f : to_float(logits[l]);
}
struct CacheImpArgs {
    device const precision_t* logits;
    device const float* actions;
    device const precision_t* old_logprobs;
    device const precision_t* mask;
    device const precision_t* logstd;
    device const int* act_sizes;
    device precision_t* imp;
    device precision_t* values;
    device float* logps;
    device float* new_lp;
    int NT, fused_cols, num_atns;
    device const char* hc;
    device const float* veps;
    int hc_stride;
};
kernel void nh_cache_imp_and_v(constant CacheImpArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= uint(a.NT)) return;
    int lb = idx * a.fused_cols;
    int ab = idx * (a.fused_cols - 1);
    a.values[idx] = from_float(to_float(a.logits[lb + a.fused_cols - 1]));
    float new_lp = 0.0f;
    if (a.logstd) {
        for (int h = 0; h < a.num_atns; h++) {
            float mean = ppo_finite_or_clamp(to_float(a.logits[lb + h]), -1e6f, 1e6f);
            float ls = ppo_finite_or_clamp(to_float(a.logstd[h]), -20.0f, 2.0f);
            // Upstream cache does NOT clamp actions (the loss kernel does).
            float norm = (a.actions[idx * a.num_atns + h] - mean) / exp(ls);
            new_lp += -0.5f * norm * norm - 0.9189385332046727f - ls;
        }
    } else {
        int offset = 0;
        for (int h = 0; h < a.num_atns; h++) {
            int A = a.act_sizes[h];
            float mx = -INFINITY, sum = 0.0f;
            for (int j = 0; j < A; j++) {
                float l = ppo_masked(a.logits, a.mask, lb + offset + j, ab + offset + j);
                if (l > mx) { sum *= exp(mx - l); mx = l; }
                sum += exp(l - mx);
            }
            float lse = mx + log(sum);
            for (int j = 0; j < A; j++)
                a.logps[ab + offset + j] = ppo_masked(a.logits, a.mask, lb + offset + j, ab + offset + j) - lse;
            int act = int(a.actions[idx * a.num_atns + h]);
            int verb = int(a.actions[idx*a.num_atns]);
            if (nh_head_used(a.hc, a.hc_stride, verb, h)) {
                float lp = ppo_masked(a.logits, a.mask, lb + offset + act, ab + offset + act) - lse;
                float scale = 1.0f;
                new_lp += h == 0 ? nh_train_logp(a.veps, a.mask+ab+offset, A, lp, scale) : lp;
            }
            offset += A;
        }
    }
    a.new_lp[idx] = new_lp;
    a.imp[idx] = from_float(exp(new_lp - to_float(a.old_logprobs[idx])));
}

struct PPOLossArgs {
    device float* partials;
    device float* grad_logits;
    device float* grad_logstd;
    device float* grad_values;
    device const precision_t* logits;
    device const precision_t* logstd;
    device const precision_t* values_pred;
    device const int* act_sizes;
    device const float* ent_coef;
    device const float* actions;
    device const precision_t* old_logprobs;
    device const precision_t* advantages;
    device const precision_t* values;
    device const precision_t* returns;
    int num_atns;
    float clip_coef, vf_clip_coef, vf_coef;
    int T_seq, A_total, N, is_continuous;
    device const char* hc;
    device const float* veps;
    int hc_stride;
    device const precision_t* mask;
};
kernel void nh_ppo_loss_compute(constant PPOLossArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x, tid = threadIdx.x;
    int total = a.N * a.T_seq;
    float inv_NT = 1.0f / float(total);
    threadgroup float sh[8][256];
    for (int c = 0; c < 8; c++) sh[c][tid] = 0.0f;
    if (idx < uint(total)) {
        int lb = idx * (a.A_total + 1), ab = idx * a.A_total;
        float adv = to_float(a.advantages[idx]), val = to_float(a.values[idx]), ret = to_float(a.returns[idx]);
        float pred = to_float(a.values_pred[lb]), ec = *a.ent_coef;
        float de = inv_NT * -ec;
        float logratio = a.grad_values[idx] - to_float(a.old_logprobs[idx]);
        float ratio = exp(logratio);
        float vc = val + fmax(-a.vf_clip_coef, fmin(a.vf_clip_coef, pred - val));
        float vu = (pred - ret) * (pred - ret), vlc = (vc - ret) * (vc - ret);
        float vl = 0.5f * fmax(vu, vlc);
        a.grad_values[idx] = inv_NT * a.vf_coef * (vlc > vu ? 0.0f : pred - ret);
        float lo = 1.0f - a.clip_coef, hi = 1.0f + a.clip_coef;
        float rc = fmax(lo, fmin(hi, ratio));
        float wa = -adv, pg1 = wa * ratio, pg2 = wa * rc;
        float pg = fmax(pg1, pg2), dr = wa * inv_NT;
        if (pg2 > pg1 && (ratio <= lo || ratio >= hi)) dr = 0.0f;
        float dlp = dr * ratio, entropy = 0.0f;
        if (a.is_continuous) {
            for (int h = 0; h < a.num_atns; h++) {
                float mean = ppo_finite_or_clamp(to_float(a.logits[lb + h]), -1e6f, 1e6f);
                float ls = ppo_finite_or_clamp(to_float(a.logstd[h]), -20.0f, 2.0f);
                float act = ppo_finite_or_clamp(a.actions[idx * a.num_atns + h], -1e6f, 1e6f);
                entropy += 1.4189385332046727f + ls;
                float std = exp(ls), var = std * std, diff = act - mean;
                a.grad_logits[ab + h] = dlp * diff / var;
                a.grad_logstd[idx * a.num_atns + h] = fma(dlp, diff * diff / var - 1.0f, de);
            }
        } else {
            int offset = 0;
            for (int h = 0; h < a.num_atns; h++) {
                int A = a.act_sizes[h], act = int(a.actions[idx * a.num_atns + h]);
                int verb = int(a.actions[idx*a.num_atns]);
                if (!nh_head_used(a.hc, a.hc_stride, verb, h)) {
                    for (int j=0; j<A; j++) a.grad_logits[ab+offset+j]=0.0f;
                    offset += A;
                    continue;
                }
                float d_logp = dlp;
                if (h == 0) {
                    float scale = 1.0f;
                    nh_train_logp(a.veps, a.mask+ab+offset, A, a.grad_logits[ab+act], scale);
                    d_logp *= scale;
                }
                float ent = 0.0f;
                for (int j = 0; j < A; j++) {
                    float lp = a.grad_logits[ab + offset + j];
                    ent = fma(-exp(lp), lp, ent);
                }
                entropy += ent;
                for (int j = 0; j < A; j++) {
                    float lp = a.grad_logits[ab + offset + j], p = exp(lp);
                    a.grad_logits[ab + offset + j] =
                        fma(de * p, -ent - lp, ((j == act ? 1.0f : 0.0f) - p) * d_logp);
                }
                offset += A;
            }
        }
        sh[0][tid] = pg * inv_NT;
        sh[1][tid] = vl * inv_NT;
        sh[2][tid] = entropy * inv_NT;
        sh[3][tid] = fma(-ec, entropy, fma(a.vf_coef, vl, pg)) * inv_NT;
        sh[4][tid] = -logratio * inv_NT;
        sh[5][tid] = ((ratio - 1.0f) - logratio) * inv_NT;
        sh[6][tid] = (fabs(ratio - 1.0f) > a.clip_coef ? 1.0f : 0.0f) * inv_NT;
        sh[7][tid] = ratio * inv_NT;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s; s >>= 1) {
        if (tid < s) for (int c = 0; c < 8; c++) sh[c][tid] += sh[c][tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) for (int c = 0; c < 8; c++) a.partials[blockIdx.x * 8 + c] = sh[c][0];
}

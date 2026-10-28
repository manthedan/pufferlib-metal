// PufferLib 5.0 advantage and PPO kernels (src/algo.cu). precision_t
// buffers as upstream (rollout / train tensors), fp32 PPO gradients, actions
// and loss partials.
#include "common.h"

struct AdvantageArgs {
    device const precision_t* values;
    device const precision_t* rewards;
    device const precision_t* dones;
    device const precision_t* importance;
    device precision_t* advantages;
    device precision_t* returns;
    float gamma, lambda, rho_clip, c_clip;
    int num_steps, horizon;
};
kernel void puff_advantage(constant AdvantageArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    uint row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= uint(a.num_steps)) return;
    int off = row * a.horizon;
    float lastlam = 0.0f;
#ifndef PRECISION_FLOAT
    // bf16: ADV_VEC_WIDTH = 16 / sizeof(bf16) = 8 (H % 8 == 0). Per-element
    // loads of each whole segment before its stores; same recurrence and
    // contractions as the FP32 branch below, rounded at the two stores.
    constexpr int W = 8;
    float next_v = to_float(a.values[off + a.horizon - 1]);
    float next_d = to_float(a.dones[off + a.horizon - 1]);
    float next_r = to_float(a.rewards[off + a.horizon - 1]);
    for (int seg = a.horizon / W - 1; seg >= 0; seg--) {
        int base = off + seg * W;
        float v[W], r[W], d[W], imp[W], adv[W];
        for (int i = 0; i < W; i++) {
            v[i] = to_float(a.values[base + i]);
            r[i] = to_float(a.rewards[base + i]);
            d[i] = to_float(a.dones[base + i]);
            imp[i] = a.importance ? to_float(a.importance[base + i]) : 1.0f;
            adv[i] = 0.0f;
        }
        int i0 = seg + 1 == a.horizon / W ? W - 2 : W - 1;
        for (int i = i0; i >= 0; i--) {
            float nnt = 1.0f - next_d;
            float rho = fmin(imp[i], a.rho_clip);
            float c = fmin(imp[i], a.c_clip);
            float delta_inner = fma(a.gamma * next_v, nnt, next_r) - v[i];
            float trace = a.gamma * a.lambda * c * lastlam * nnt;
            lastlam = fma(rho, delta_inner, trace);
            adv[i] = lastlam;
            next_v = v[i]; next_d = d[i]; next_r = r[i];
        }
        for (int i = 0; i < W; i++) {
            a.advantages[base + i] = from_float(adv[i]);
            a.returns[base + i] = from_float(v[i] + adv[i]);
        }
    }
#else
    float next_v = a.values[off + a.horizon - 1];
    float next_d = a.dones[off + a.horizon - 1];
    float next_r = a.rewards[off + a.horizon - 1];
    // Load each entire segment before stores: returns aliases live values in
    // the learner. H is a multiple of ADV_VEC_WIDTH=4, as in CUDA FP32.
    for (int seg = a.horizon / 4 - 1; seg >= 0; seg--) {
        int base = off + seg * 4;
        float4 v = *reinterpret_cast<device const float4*>(a.values + base);
        float4 r = *reinterpret_cast<device const float4*>(a.rewards + base);
        float4 d = *reinterpret_cast<device const float4*>(a.dones + base);
        float4 imp = a.importance ? *reinterpret_cast<device const float4*>(a.importance + base) : float4(1.0f);
        float4 adv = 0.0f;
        int i0 = seg + 1 == a.horizon / 4 ? 2 : 3;
        for (int i = i0; i >= 0; i--) {
            float nnt = 1.0f - next_d;
            float rho = fmin(imp[i], a.rho_clip);
            float c = fmin(imp[i], a.c_clip);
            float delta_inner = fma(a.gamma * next_v, nnt, next_r) - v[i];
            // nvcc contracts rho * delta_inner into the recurrence's final
            // add. Contracting the other product gives different V-trace bits.
            float trace = a.gamma * a.lambda * c * lastlam * nnt;
            lastlam = fma(rho, delta_inner, trace);
            adv[i] = lastlam;
            next_v = v[i]; next_d = d[i]; next_r = r[i];
        }
        *reinterpret_cast<device float4*>(a.advantages + base) = adv;
        *reinterpret_cast<device float4*>(a.returns + base) = v + adv;
    }
#endif
}

static inline float ppo_finite_or_clamp(float x, float lo, float hi) {
    if (isnan(x)) return 0.0f;
    return fmin(fmax(x, lo), hi);
}
static inline float ppo_masked(device const precision_t* logits, device const precision_t* mask,
        int l, int m) {
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
};
kernel void cache_imp_and_v(constant CacheImpArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= uint(a.NT)) return;
    int lb = idx * a.fused_cols;
    int ab = idx * (a.fused_cols - 1);
    a.values[idx] = a.logits[lb + a.fused_cols - 1];
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
            new_lp += ppo_masked(a.logits, a.mask, lb + offset + act, ab + offset + act) - lse;
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
};
kernel void ppo_loss_compute(constant PPOLossArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x, tid = threadIdx.x;
    int total = a.N * a.T_seq;
    float inv_NT = 1.0f / float(total);
    threadgroup float sh[8][256];
    for (int c = 0; c < 8; c++) sh[c][tid] = 0.0f;
    if (idx < uint(total)) {
        int lb = idx * (a.A_total + 1), ab = idx * a.A_total;
        float adv = to_float(a.advantages[idx]), val = to_float(a.values[idx]);
        float ret = to_float(a.returns[idx]);
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
                float ent = 0.0f;
                for (int j = 0; j < A; j++) {
                    float lp = a.grad_logits[ab + offset + j];
                    ent = fma(-exp(lp), lp, ent);
                }
                entropy += ent;
                for (int j = 0; j < A; j++) {
                    float lp = a.grad_logits[ab + offset + j], p = exp(lp);
                    a.grad_logits[ab + offset + j] =
                        fma(de * p, -ent - lp, ((j == act ? 1.0f : 0.0f) - p) * dlp);
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

struct PPOReduceArgs { device float* losses; device const float* partials; int num_blocks; };
kernel void ppo_loss_reduce(constant PPOReduceArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    uint tid = threadIdx.x;
    float sum = 0.0f;
    for (int b = 0; b < a.num_blocks; b++) sum += a.partials[b * 8 + tid];
    a.losses[tid] += sum;
    if (tid == 0) a.losses[8] += 1.0f;
}

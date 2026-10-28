// Learner adapters. Included last by kernels.h; shaders in ppo/muon/log.metal.
#pragma once
#include <cstddef>

#define PUF_ADAPTER(kname, ...) \
    static inline void puf_mtl_kernel(puf_kid_##kname*, const puf_mtl::LaunchCfg& cfg, __VA_ARGS__)

struct MtlAdvantageArgs {
    uint64_t values, rewards, dones, importance, advantages, returns;
    float gamma, lambda, rho_clip, c_clip;
    int32_t num_steps, horizon;
};
static_assert(sizeof(MtlAdvantageArgs) == 72, "AdvantageArgs layout");
PUF_ADAPTER(puff_advantage, const precision_t* values, const precision_t* rewards,
        const precision_t* dones, const precision_t* importance, precision_t* advantages,
        precision_t* returns, float gamma, float lambda, float rho_clip, float c_clip,
        int num_steps, int horizon) {
    // shaders/ppo.metal segments rows by ADV_VEC_WIDTH = 16 / sizeof(precision_t).
    if (horizon % (16 / (int)sizeof(precision_t)) != 0) {
        puf_mtl::fatal("puff_advantage: horizon %d is not a multiple of %d", horizon,
            16 / (int)sizeof(precision_t));
    }
    puf_mtl::launch(cfg, "puff_advantage", MtlAdvantageArgs{puf_mtl::dptr(values),
        puf_mtl::dptr(rewards), puf_mtl::dptr(dones), puf_mtl::dptr(importance),
        puf_mtl::dptr(advantages), puf_mtl::dptr(returns), gamma, lambda, rho_clip,
        c_clip, num_steps, horizon});
}

#ifndef PUFFER_NETHACK  // NetHack variants live in kernels_nethack.h.
struct MtlCacheImpArgs {
    uint64_t logits, actions, old_logprobs, mask, logstd, act_sizes, imp, values, logps, new_lp;
    int32_t NT, fused_cols, num_atns;
};
static_assert(sizeof(MtlCacheImpArgs) == 96, "CacheImpArgs layout");
PUF_ADAPTER(cache_imp_and_v, Prec dec_out, const float* actions,
        const precision_t* old_logprobs, const precision_t* action_mask, Prec logstd,
        const int* act_sizes, precision_t* imp_out, precision_t* value_out,
        float* logps, float* new_lp_out) {
    puf_mtl::launch(cfg, "cache_imp_and_v", MtlCacheImpArgs{puf_mtl::dptr(dec_out.data),
        puf_mtl::dptr(actions), puf_mtl::dptr(old_logprobs), puf_mtl::dptr(action_mask),
        puf_mtl::dptr(logstd.data), puf_mtl::dptr(act_sizes), puf_mtl::dptr(imp_out),
        puf_mtl::dptr(value_out), puf_mtl::dptr(logps), puf_mtl::dptr(new_lp_out),
        (int32_t)(dec_out.shape[0] * dec_out.shape[1]), (int32_t)dec_out.shape[2], NUM_ATNS});
}

struct MtlPPOLossArgs {
    uint64_t partials, grad_logits, grad_logstd, grad_values, logits, logstd,
        values_pred, act_sizes, ent_coef, actions, old_logprobs, advantages, values, returns;
    int32_t num_atns;
    float clip_coef, vf_clip_coef, vf_coef;
    int32_t T_seq, A_total, N, is_continuous;
};
static_assert(sizeof(MtlPPOLossArgs) == 144, "PPOLossArgs layout");
// Templates keep this header usable by the existing minimal-type Metal tests.
// Production instantiates these with upstream PPOKernelArgs/PPOGraphArgs.
template <typename A, typename G>
PUF_ADAPTER(ppo_loss_compute, float* partials, A a, G g) {
    puf_mtl::launch(cfg, "ppo_loss_compute", MtlPPOLossArgs{puf_mtl::dptr(partials),
        puf_mtl::dptr(a.grad_logits), puf_mtl::dptr(a.grad_logstd),
        puf_mtl::dptr(a.grad_values_pred), puf_mtl::dptr(a.logits), puf_mtl::dptr(a.logstd),
        puf_mtl::dptr(a.values_pred), puf_mtl::dptr(a.act_sizes), puf_mtl::dptr(a.ent_coef),
        puf_mtl::dptr(g.actions), puf_mtl::dptr(g.old_logprobs), puf_mtl::dptr(g.advantages),
        puf_mtl::dptr(g.values), puf_mtl::dptr(g.returns), a.num_atns, a.clip_coef,
        a.vf_clip_coef, a.vf_coef, a.T_seq, a.A_total, a.N, a.is_continuous ? 1 : 0});
}
#endif

struct MtlPPOReduceArgs { uint64_t losses, partials; int32_t num_blocks; };
static_assert(sizeof(MtlPPOReduceArgs) == 24, "PPOReduceArgs layout");
PUF_ADAPTER(ppo_loss_reduce, float* losses, const float* partials, int num_blocks) {
    puf_mtl::launch(cfg, "ppo_loss_reduce", MtlPPOReduceArgs{
        puf_mtl::dptr(losses), puf_mtl::dptr(partials), num_blocks});
}

struct MtlMuonSumArgs { uint64_t dst, src; int32_t n; };
static_assert(sizeof(MtlMuonSumArgs) == 24, "MuonSumArgs layout");
PUF_ADAPTER(muon_sum_sq_partials, float* partials, const precision_t* src, int n) {
    puf_mtl::launch(cfg, "muon_sum_sq_partials", MtlMuonSumArgs{
        puf_mtl::dptr(partials), puf_mtl::dptr(src), n});
}
PUF_ADAPTER(muon_sum_sq_reduce, float* out, const float* partials, int num_blocks) {
    puf_mtl::launch(cfg, "muon_sum_sq_reduce", MtlMuonSumArgs{
        puf_mtl::dptr(out), puf_mtl::dptr(partials), num_blocks});
}
struct MtlMuonClipArgs { uint64_t mb, gc, sum_sq; float max_norm, eps, mu; int32_t n; };
static_assert(sizeof(MtlMuonClipArgs) == 40, "MuonClipArgs layout");
PUF_ADAPTER(muon_clip_nesterov, float* mb, precision_t* gc, const float* sum_sq,
        float max_norm, float eps, float mu, int n) {
    puf_mtl::launch(cfg, "muon_clip_nesterov", MtlMuonClipArgs{puf_mtl::dptr(mb),
        puf_mtl::dptr(gc), puf_mtl::dptr(sum_sq), max_norm, eps, mu, n});
}
struct MtlMuonNormalizeArgs { uint64_t dst, sum_sq; float eps; int32_t n; };
static_assert(sizeof(MtlMuonNormalizeArgs) == 24, "MuonNormalizeArgs layout");
PUF_ADAPTER(muon_l2_normalize, precision_t* dst, const float* sum_sq, float eps, int n) {
    puf_mtl::launch(cfg, "muon_l2_normalize", MtlMuonNormalizeArgs{
        puf_mtl::dptr(dst), puf_mtl::dptr(sum_sq), eps, n});
}
struct MtlMuonStoreArgs { uint64_t dst, src; float scale; int32_t n; };
static_assert(sizeof(MtlMuonStoreArgs) == 24, "MuonStoreArgs layout");
PUF_ADAPTER(muon_store_update, precision_t* dst, const precision_t* src, float scale, int n) {
    puf_mtl::launch(cfg, "muon_store_update", MtlMuonStoreArgs{
        puf_mtl::dptr(dst), puf_mtl::dptr(src), scale, n});
}
struct MtlMuonWeightArgs { uint64_t wb, update, lr; float wd; int32_t n; };
static_assert(sizeof(MtlMuonWeightArgs) == 32, "MuonWeightArgs layout");
PUF_ADAPTER(muon_weight_update, float* wb, const precision_t* update, const float* lr, float wd, int n) {
    puf_mtl::launch(cfg, "muon_weight_update", MtlMuonWeightArgs{puf_mtl::dptr(wb),
        puf_mtl::dptr(update), puf_mtl::dptr(lr), wd, n});
}

struct MtlLogReduceArgs {
    uint64_t envs, out;
    int32_t total_agents, clear, env_stride, log_offset, num_agents_offset, n_offset, nf;
};
static_assert(sizeof(MtlLogReduceArgs) == 48, "LogReduceArgs layout");
template <typename E>
PUF_ADAPTER(log_reduce, E* envs, float* out, int total_agents, int clear) {
    using L = decltype(E::log);
    puf_mtl::launch(cfg, "log_reduce", MtlLogReduceArgs{puf_mtl::dptr(envs),
        puf_mtl::dptr(out), total_agents, clear, (int32_t)sizeof(E), (int32_t)offsetof(E, log),
        (int32_t)offsetof(E, num_agents), (int32_t)offsetof(L, n), (int32_t)(sizeof(L) / sizeof(float))});
}

#undef PUF_ADAPTER

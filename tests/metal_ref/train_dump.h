#pragma once
#include "common.h"
// Included only under PUF_GOLDEN_DUMP, after cosine_annealing and all learner types.
static float *golden_mb_losses = nullptr;
static golden::Case *golden_train_case = nullptr;
#ifndef PUF_METAL
__global__ void golden_record_losses(float *dst, const float *partials, int blocks) {
    int c = threadIdx.x;
    if (c < LOSS_N) {
        float s = 0;
        for (int b = 0; b < blocks; b++)
            s += partials[b * LOSS_N + c];
        dst[c] = s;
    }
    if (c == 0)
        dst[LOSS_N] = 1;
}
#endif
static int golden_nmb(PuffeRL *p) {
    return p->hypers.replay_ratio * (p->hypers.total_agents * p->hypers.horizon) /
           p->hypers.minibatch_size;
}
static void golden_minibatch(PuffeRL *p, int mb, cudaStream_t s) {
    int blocks = (p->hypers.minibatch_size + PPO_THREADS - 1) / PPO_THREADS;
#ifdef PUF_METAL
    // Unified memory: same sum, same order, on the host once the stream drains.
    cudaStreamSynchronize(s);
    float *dst = golden_mb_losses + mb * NUM_LOSSES;
    const float *partials = p->ppo_bufs.ppo_partials.data;
    for (int c = 0; c < LOSS_N; c++) {
        float acc = 0;
        for (int b = 0; b < blocks; b++)
            acc += partials[b * LOSS_N + c];
        dst[c] = acc;
    }
    dst[LOSS_N] = 1;
#else
    golden_record_losses<<<1, LOSS_N, 0, s>>>(golden_mb_losses + mb * NUM_LOSSES,
                                              p->ppo_bufs.ppo_partials.data, blocks);
#endif
}
static void golden_rollout(golden::Case &c, RolloutBuf r) {
    c.tensor("rollout_obs", r.observations, "input");
    c.tensor("rollout_actions", r.actions, "input");
    c.tensor("rollout_logprobs", r.logprobs, "input");
    c.tensor("rollout_values", r.values, "input");
    c.tensor("rollout_rewards", r.rewards, "input");
    c.tensor("rollout_terminals", r.terminals, "input");
    c.tensor("rollout_masks", r.action_mask, "input");
    if (r.initial_states.data)
        c.tensor("initial_states_all_slots", r.initial_states, "state");
}
static void golden_parameters(golden::Case &c, Policy *p, std::string suffix, std::string role) {
    // param and its per-layer views are Prec (bf16 in BF16 builds). The fp32 master that
    // muon_step updates aliases param under PRECISION_FLOAT, so it is dumped only in BF16.
    c.tensor("params_" + suffix, p->param, role);
    if (USE_BF16)
        c.tensor("master_" + suffix, p->master_weights, role);
    long off = 0;
    for (int i = 0; i < p->params_alloc.num_regs; i++) {
        auto &e = p->params_alloc.regs[i];
        std::string n = i == 0   ? "encoder_weight"
                        : i == 1 ? "decoder_weight"
                                 : "mingru_" + std::to_string(i - 2) + "_weight";
        c.device(n + "_" + suffix, *e.data_ptr, golden::shape(e.shape),
                 golden::dtype_of_size(e.elem_size), role);
        if (suffix == "in")
            c.scalar(n + "_flat_offset", off);
        off += numel(e.shape);
    }
}
static void golden_before_train(PuffeRL *p, RolloutBuf src, int slot) {
    if (!golden_mb_losses)
        golden::check(cudaMalloc(&golden_mb_losses, golden_nmb(p) * NUM_LOSSES * sizeof(float)));
    const char *root = getenv("PUF_GOLDEN_TRAIN");
    if (!root || p->epoch != 3)
        return;
    // Finish the already-started actor horizon so snapshots of RNG/state are stable.
    // This affects overlap timing only; the ready training slot and actor weights are immutable.
    if (p->hypers.async)
        rollout_finish(p, wall_clock());
    golden::check(cudaDeviceSynchronize());
    auto &c = *(golden_train_case = new golden::Case(root));
    auto &h = p->hypers;
    c.note("operation", "train_impl");
    c.note("loss_order", "PG,VF,ENT,TOTAL,OLD_APPROX_KL,APPROX_KL,CLIPFRAC,IMP,count");
    c.note("rng_note",
           "train consumes no RNG; captured after completing the prefetched actor horizon");
    c.scalar("epoch", p->epoch);
    c.scalar("global_step", p->global_step);
    c.scalar("slot", slot);
    c.scalar("total_agents", h.total_agents);
    c.scalar("obs_size", OBS_SIZE);
    c.scalar("num_atns", NUM_ATNS);
    c.scalar("num_buffers", h.num_buffers);
    c.scalar("slots", p->async_num_slots);
    c.scalar("horizon", h.horizon);
    c.scalar("hidden", h.hidden_size);
    c.scalar("layers", h.num_layers);
    c.scalar("minibatch_size", h.minibatch_size);
    c.scalar("total_minibatches", golden_nmb(p));
    c.scalar("total_timesteps", h.total_timesteps);
    c.scalar("replay_ratio", h.replay_ratio);
    c.scalar("gamma", h.gamma);
    c.scalar("gae_lambda", h.gae_lambda);
    c.scalar("vtrace", h.vtrace);
    c.scalar("rho_clip", h.vtrace_rho_clip);
    c.scalar("c_clip", h.vtrace_c_clip);
    c.scalar("clip_coef", h.clip_coef);
    c.scalar("vf_clip_coef", h.vf_clip_coef);
    c.scalar("vf_coef", h.vf_coef);
    c.scalar("momentum", h.momentum);
    c.scalar("max_grad_norm", h.max_grad_norm);
    c.scalar("reset_every_horizon", h.reset_every_horizon);
    c.scalar("async", h.async);
    c.scalar("cudagraphs", h.cudagraphs);
    c.scalar("seed", p->seed);
    c.scalar("world_size", h.world_size);
    c.scalar("weight_decay", 0);
    c.scalar("clip_eps", 1e-6);
    c.scalar("ns_eps", 1e-7);
    c.scalar("anneal_lr", h.anneal_lr);
    c.scalar("anneal_ent_coef", h.anneal_ent_coef);
    c.scalar("base_lr", h.lr);
    c.scalar("min_lr_ratio", h.min_lr_ratio);
    c.scalar("base_ent_coef", h.ent_coef);
    c.scalar("min_ent_coef_ratio", h.min_ent_coef_ratio);
    float lr, ent;
    golden::check(cudaMemcpy(&lr, p->muon.lr, 4, cudaMemcpyDeviceToHost));
    golden::check(cudaMemcpy(&ent, p->ppo_bufs.ent_coef, 4, cudaMemcpyDeviceToHost));
    c.scalar("lr", lr);
    c.scalar("ent_coef", ent);
    golden_parameters(c, &p->policies[0], "in", "param");
    c.tensor("momentum_in", p->muon.mb, "state");
    c.tensor("train_state_in", p->train_state, "state");
    c.tensor("grad_next_state", ((MinGRUActivations *)p->train_activs.network)->grad_next_state,
             "input");
    c.device("losses_in", p->losses, {NUM_LOSSES}, "f32", "state");
    golden_rollout(c, src);
    c.device("act_sizes", p->act_sizes, {NUM_ATNS}, "i32", "input");
    c.device("rng_offsets_in", p->rng_offset, {h.num_buffers + 1}, "u64", "state");
    c.scalar("philox_state_bytes", sizeof(curandStatePhilox4_32_10_t));
    for (int b = 0; b < h.num_buffers; b++) {
        c.device("rng_" + std::to_string(b) + "_in", p->rng_states[b],
                 {h.total_agents / h.num_buffers, (int64_t)sizeof(curandStatePhilox4_32_10_t)},
                 "u8", "state");
        c.tensor("actor_state_" + std::to_string(b) + "_in", p->policies[0].buffer_states[b],
                 "state");
    }
    if (h.async)
        c.tensor("actor_params_in", p->actor_param, "state");
    std::vector<int32_t> order(golden_nmb(p));
    for (int i = 0; i < golden_nmb(p); i++)
        order[i] = (i * (h.minibatch_size / h.horizon)) % h.total_agents;
    c.host("minibatch_row_offsets", order.data(), {golden_nmb(p)}, "i32", "input");
}
static void golden_after_train(PuffeRL *p) {
    if (!golden_train_case)
        return;
    auto &c = *golden_train_case;
    golden_parameters(c, &p->policies[0], "out", "output");
    c.tensor("momentum_out", p->muon.mb);
    c.tensor("final_update", p->grad);
    c.tensor("train_state_out", p->train_state);
    c.device("losses_out", p->losses, {NUM_LOSSES});
    c.device("minibatch_losses", golden_mb_losses, {golden_nmb(p), NUM_LOSSES});
    c.device("rng_offsets_out", p->rng_offset, {p->hypers.num_buffers + 1}, "u64");
    for (int b = 0; b < p->hypers.num_buffers; b++)
        c.device("rng_" + std::to_string(b) + "_out", p->rng_states[b],
                 {p->hypers.total_agents / p->hypers.num_buffers,
                  (int64_t)sizeof(curandStatePhilox4_32_10_t)},
                 "u8");
    c.finish();
    delete golden_train_case;
    golden_train_case = nullptr;
    if (getenv("PUF_GOLDEN_STOP"))
        exit(0);
}

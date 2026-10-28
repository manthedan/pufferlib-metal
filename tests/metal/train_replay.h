// Metal replay of the captured CUDA train() call (golden case train_epoch3).
// Included by src/metal/pufferl_metal.cpp when built with
//   NVCC_EXTRA="-DPUF_GOLDEN_DUMP -DPUF_METAL_TRAIN_REPLAY" ./build.sh breakout puffer_replay --metal
// Usage: ./puffer_replay replay GOLDEN_CASE_DIR OUT_DIR
// Precision follows the build: FP32 (--float) replays a PRECISION_FLOAT
// capture, the default BF16 build a BF16 one (bf16 Prec tensors plus the fp32
// master weights; tensor byte sizes come from the buffers' element types).
// Mirrors replay() in tests/metal_ref/generate.cu so both backends load the
// same state and dump through the same tests/metal_ref/train_dump.h hooks.
#pragma once

#include <string>
#include <vector>

static void metal_replay_upload(void* dst, const std::string& path, size_t n) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "replay: cannot open %s\n", path.c_str());
        exit(1);
    }
    std::vector<char> b(n);
    if (fread(b.data(), 1, n, f) != n) {
        fprintf(stderr, "replay: short read %s (want %zu bytes)\n", path.c_str(), n);
        exit(1);
    }
    if (fgetc(f) != EOF) {
        fprintf(stderr, "replay: oversized input %s (expected %zu bytes)\n", path.c_str(), n);
        exit(1);
    }
    fclose(f);
    cudaMemcpy(dst, b.data(), n, cudaMemcpyHostToDevice);
}

static int metal_train_replay(const std::string& input, const std::string& output, bool wide = false) {
    Ini ini = {};
    puf_ini_load_env(&ini, wide ? PUFFER_ENV_NAME : "breakout", 0, nullptr);
    if (wide) {
        int horizon = puf_ini_get(&ini, "train", "horizon");
        dict_set(puf_ini_section(&ini, "vec", 0), "total_agents", 128);
        dict_set(puf_ini_section(&ini, "train", 0), "minibatch_size", 128 * horizon);
    }
    TrainContext ctx = {.world_size = 1};
    PuffeRL* p = create_pufferl(&ini, &ctx);
    auto load = [&](const std::string& n, auto t) {
        metal_replay_upload(t.data, input + "/" + n + ".bin", numel(t.shape) * sizeof(*t.data));
    };
    load("params_in", p->policies[0].param);
    if (USE_BF16) load("master_in", p->policies[0].master_weights);
    // PUF_REPLAY_PERTURB_SEED=N nudges every parameter by +-1 ulp: replaying
    // twice with different seeds measures how much a train() call amplifies
    // rounding noise, the yardstick for judging Metal-vs-CUDA differences.
    // BF16: the bf16 parameter copy the forward/backward reads moves by +-1
    // bf16 ulp; the fp32 master Muon updates is left as captured.
    if (const char* seed = getenv("PUF_REPLAY_PERTURB_SEED")) {
        cudaDeviceSynchronize();
        precision_t* w = p->policies[0].param.data;
        uint32_t s = (uint32_t)atoi(seed) * 2654435761u + 1;
        for (int64_t i = 0; i < numel(p->policies[0].param.shape); i++) {
            s = s * 1664525u + 1013904223u;
            bool up = (s >> 31) != 0;
#ifdef PRECISION_FLOAT
            w[i] = nextafterf(w[i], up ? INFINITY : -INFINITY);
#else
            // Finite bf16 only (as Breakout weights are): step the magnitude
            // away from or toward zero; +-0 steps to the smallest denormal.
            uint16_t b = w[i].x, mag = b & 0x7fff;
            bool neg = (b & 0x8000) != 0;
            if (mag == 0) b = up ? 0x0001 : 0x8001;
            else if (up != neg) b = (uint16_t)(b + 1);
            else b = (uint16_t)(b - 1);
            w[i].x = b;
#endif
        }
    }
    load("momentum_in", p->muon.mb);
    load("train_state_in", p->train_state);
    load("grad_next_state", ((MinGRUActivations*)p->train_activs.network)->grad_next_state);
    p->epoch = 3;
    p->global_step = 4L * p->hypers.total_agents * p->hypers.horizon;
    p->async_ready_slot = 1;
    RolloutBuf src = rollout_time_view(&p->rollouts, p->hypers.horizon, p->hypers.horizon);
    load("rollout_obs", src.observations);
    load("rollout_actions", src.actions);
    load("rollout_logprobs", src.logprobs);
    load("rollout_values", src.values);
    load("rollout_rewards", src.rewards);
    load("rollout_terminals", src.terminals);
    load("rollout_masks", src.action_mask);
    load("initial_states_all_slots", src.initial_states);
    load("actor_params_in", p->actor_param);
    for (int b = 0; b < p->hypers.num_buffers; b++) {
        load("actor_state_" + std::to_string(b) + "_in", p->policies[0].buffer_states[b]);
        metal_replay_upload(p->rng_states[b], input + "/rng_" + std::to_string(b) + "_in.bin",
                            (p->hypers.total_agents / p->hypers.num_buffers) * sizeof(curandStatePhilox4_32_10_t));
    }
    metal_replay_upload(p->rng_offset, input + "/rng_offsets_in.bin", (p->hypers.num_buffers + 1) * sizeof(uint64_t));
    metal_replay_upload(p->losses, input + "/losses_in.bin", NUM_LOSSES * 4);
    setenv("PUF_GOLDEN_TRAIN", output.c_str(), 1);
    train_impl(p, &src);
    cudaDeviceSynchronize();
    printf("replay completed\n");
    return 0;
}

int main(int argc, char** argv) {
    setbuf(stdout, nullptr);
    if (argc == 4 && std::string(argv[1]) == "replay-wide") {
        return metal_train_replay(argv[2], argv[3], true);
    }
    if (argc == 4 && std::string(argv[1]) == "replay") {
        return metal_train_replay(argv[2], argv[3]);
    }
    return puf_upstream_main(argc, argv);
}

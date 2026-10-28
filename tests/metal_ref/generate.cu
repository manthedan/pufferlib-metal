#include "common.h"
#include "pufferl.cu"
using golden::Case;
using golden::Shape;
static std::string root;
static std::vector<void *> loose;
static float *data(Shape s, int seed = 1, float scale = 1) {
    int64_t n = golden::count(s);
    float *p;
    golden::check(cudaMalloc(&p, n * 4));
    loose.push_back(p);
    std::vector<float> v(n);
    for (int64_t i = 0; i < n; i++)
        v[i] = golden::pattern(i, seed, scale);
    golden::check(cudaMemcpy(p, v.data(), n * 4, cudaMemcpyHostToDevice));
    return p;
}
static Prec prec(Shape s, int seed = 1, float scale = 1) {
    Prec t = {};
    std::copy(s.begin(), s.end(), t.shape);
    t.data = data(s, seed, scale);
    return t;
}
static Float flt(Shape s, int seed = 1, float scale = 1) {
    Prec p = prec(s, seed, scale);
    Float f = {};
    f.data = p.data;
    memcpy(f.shape, p.shape, sizeof(f.shape));
    return f;
}
static void fill(float *p, const std::vector<float> &v) {
    golden::check(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
}
static void clean() {
    golden::check(cudaDeviceSynchronize());
    for (void *p : loose)
        golden::check(cudaFree(p));
    loose.clear();
}
static void dims(Case &c, int B, int T, int H) {
    c.scalar("B", B);
    c.scalar("T", T);
    c.scalar("H", H);
}
static void gemm(std::string name, int op, int M, int N, int K, float alpha = 1, float beta = 0,
                 int T = 1) {
    Case c(root + "/" + name);
    c.note("operation", op == 0 ? "puf_mm" : op == 1 ? "puf_mm_tn" : "puf_mm_nn");
    c.scalar("M", M);
    c.scalar("N", N);
    c.scalar("K", K);
    c.scalar("alpha", alpha);
    c.scalar("beta", beta);
    Shape as = op == 1 ? Shape{K, M} : Shape{M, K};
    Shape bs = op == 0 ? Shape{N, K} : op == 1 ? Shape{K, N} : Shape{K, N};
    if (T > 1) {
        if (op == 1)
            as = {K / T, T, M};
        else
            as = {M / T, T, K};
        if (op == 1)
            bs = {K / T, T, N};
    }
    Prec a = prec(as, 2, .2), b = prec(bs, 3, .2), out = prec({M, N}, 4, .2);
    c.tensor("a", a, "input");
    c.tensor("b", b, "input");
    c.tensor("c_in", out, "input");
    if (op == 0)
        puf_mm(&a, &b, &out, 0, alpha, beta);
    else if (op == 1)
        puf_mm_tn(&a, &b, &out, 0, alpha, beta);
    else
        puf_mm_nn(&a, &b, &out, 0, alpha, beta);
    c.tensor("c_out", out);
    c.finish();
    clean();
}
static void gemms(bool real) {
    int H = real ? 32 : 5, I = real ? OBS_SIZE : 7, NT = real ? 65536 : 16, T = real ? 32 : 4;
    std::string p = real ? "gemm_breakout_" : "gemm_tiny_";
    gemm(p + "rollout_encoder_nt", 0, real ? 4096 : 3, H, I);
    gemm(p + "rollout_decoder_nt", 0, real ? 4096 : 3, 4, H);
    gemm(p + "rollout_mingru_nt", 0, real ? 4096 : 3, 3 * H, H);
    gemm(p + "encoder_nt", 0, NT, H, I, 1, 0, T);
    gemm(p + "encoder_tn", 1, H, I, NT, 1, 0, T);
    gemm(p + "decoder_nt", 0, NT, 4, H, 1, 0, T);
    gemm(p + "decoder_tn", 1, 4, H, NT, 1, 0, T);
    gemm(p + "decoder_nn", 2, NT, H, 4, 1, 0, T);
    gemm(p + "mingru_nt", 0, NT, 3 * H, H, 1, 0, T);
    gemm(p + "mingru_tn", 1, 3 * H, H, NT, 1, 0, T);
    gemm(p + "mingru_nn", 2, NT, H, 3 * H, 1, 0, T);
    int rs[] = {H, 4, 3 * H}, cs[] = {I, H, H};
    for (int q = 0; q < 3; q++) {
        int R = rs[q], C = cs[q], n = std::min(R, C);
        std::string tag = p + "muon_param" + std::to_string(q);
        gemm(tag + "_gram", R > C ? 1 : 0, n, n, std::max(R, C));
        for (int i = 0; i < 5; i++) {
            gemm(tag + "_poly" + std::to_string(i), 2, n, n, n, ns_coeffs[i][2], ns_coeffs[i][1]);
            if (R > C)
                gemm(tag + "_iterate" + std::to_string(i), 2, R, C, C, 1, ns_coeffs[i][0]);
            else
                gemm(tag + "_iterate" + std::to_string(i), 2, R, C, R, 1, ns_coeffs[i][0]);
        }
    }
}
static void gate(bool real) {
    int B = real ? 4096 : 3, H = real ? 32 : 5;
    Case c(root + (real ? "/gate_breakout" : "/gate_tiny"));
    c.note("operation", "mingru_gate");
    dims(c, B, 1, H);
    auto combined = prec({B, 3 * H}, 3, 4), state = prec({B, H}, 4), x = prec({B, H}, 5),
         out = prec({B, H}), next = prec({B, H});
    c.tensor("combined", combined, "input");
    c.tensor("state_in", state, "state");
    c.tensor("x", x, "input");
    mingru_gate<<<grid_size(B * H), BLOCK_SIZE>>>(out.data, next.data, combined.data, state.data,
                                                  x.data, H, B);
    c.tensor("out", out);
    c.tensor("state_out", next);
    c.finish();
    clean();
}
static void scan_case(std::string name, int B, int T, int H, bool row) {
    Case c(root + "/" + name);
    c.note("operation",
           row ? "mingru_scan_forward_row+backward_row" : "mingru_scan_forward+backward");
    dims(c, B, T, H);
    auto combined = prec({B, T, 3 * H}, 2, 4), state = prec({B, H}, 3), x = prec({B, T, H}, 4),
         term = prec({B, T}), go = prec({B, T, H}, 6, .2), gn = prec({B, H}, 7, .2);
    std::vector<float> flags(B * T);
    for (int b = 0; b < B; b++) {
        if (b % 2 == 0)
            flags[b * T] = 1;
        flags[b * T + T / 2] = 1;
        if (b % 3 == 0)
            flags[b * T + T - 1] = 1;
        if (b % 5 == 0)
            flags[b * T + T / 3] = .25f;
    }
    fill(term.data, flags);
    PrefixScan s = {};
    s.B = B;
    s.T = T;
    s.H = H;
    s.combined_ptr = combined.data;
    s.state_ptr = state.data;
    s.input_ptr = x.data;
    s.terminals_ptr = term.data;
    s.scan_h = prec({B, T, H});
    s.out = prec({B, T, H});
    s.next_state = prec({B, H});
    s.grad_combined = prec({B, T, 3 * H});
    s.grad_input = prec({B, T, H});
    s.grad_state = prec({B, H});
    c.tensor("combined", combined, "input");
    c.tensor("state_in", state, "state");
    c.tensor("x", x, "input");
    c.tensor("terminals", term, "input");
    c.tensor("grad_out", go, "input");
    c.tensor("grad_next_state", gn, "input");
    dim3 block(MINGRU_TILE, MINGRU_TILE), grid(B, (H + 31) / 32);
    if (row)
        mingru_scan_forward_row<<<grid, block>>>(s);
    else
        mingru_scan_forward<<<grid_size(B * H), BLOCK_SIZE>>>(s);
    if (row)
        mingru_scan_backward_row<<<grid, block>>>(s, go.data, gn.data);
    else
        mingru_scan_backward<<<grid_size(B * H), BLOCK_SIZE>>>(s, go.data, gn.data);
    c.tensor("out", s.out);
    c.tensor("next_state", s.next_state);
    c.tensor("scan_h", s.scan_h);
    c.tensor("grad_combined", s.grad_combined);
    c.tensor("grad_input", s.grad_input);
    c.tensor("grad_state", s.grad_state);
    c.finish();
    clean();
}
static int *action_sizes() {
    int sizes[] = ACT_SIZES;
    int *p;
    golden::check(cudaMalloc(&p, sizeof(sizes)));
    loose.push_back(p);
    golden::check(cudaMemcpy(p, sizes, sizeof(sizes), cudaMemcpyHostToDevice));
    return p;
}
static Prec masks(int N, int A, bool allmasked = false) {
    auto m = prec({N, A});
    std::vector<float> v(N * A, 1);
    for (int n = 0; n < N; n++) {
        if (n % 4 == 0)
            v[n * A + A - 1] = 0;
        if (n % 4 == 1) {
            for (int a = 0; a < A; a++)
                v[n * A + a] = a == 1;
        }
        if (allmasked && n % 13 == 0)
            for (int a = 0; a < A; a++)
                v[n * A + a] = 0;
    }
    fill(m.data, v);
    return m;
}
__global__ void warm_rng(curandStatePhilox4_32_10_t *states, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B) {
        auto s = states[i];
        for (int j = 0; j < i % 7; j++)
            curand_uniform(&s);
        states[i] = s;
    }
}
__global__ void uniform_probe(curandStatePhilox4_32_10_t *states, float *draws, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B) {
        auto s = states[i];
        for (int h = 0; h < NUM_ATNS; h++)
            draws[i * NUM_ATNS + h] = curand_uniform(&s);
    }
}
__global__ void rng_fields(curandStatePhilox4_32_10_t *states, unsigned int *fields, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < B) {
        auto s = states[i];
        unsigned int *p = fields + i * 12;
        p[0] = s.ctr.x;
        p[1] = s.ctr.y;
        p[2] = s.ctr.z;
        p[3] = s.ctr.w;
        p[4] = s.key.x;
        p[5] = s.key.y;
        p[6] = s.output.x;
        p[7] = s.output.y;
        p[8] = s.output.z;
        p[9] = s.output.w;
        p[10] = s.STATE;
        p[11] = s.boxmuller_flag;
    }
}
static void sample_case(bool real) {
    int B = real ? 4096 : 17, A = 3;
    Case c(root + (real ? "/sample_breakout" : "/sample_tiny"));
    c.note("operation", "sample_logits");
    dims(c, B, 1, 0);
    c.scalar("seed", 73);
    c.scalar("philox_state_bytes", sizeof(curandStatePhilox4_32_10_t));
    c.note("rng_fields", "ctr.x,ctr.y,ctr.z,ctr.w,key.x,key.y,output.x,output.y,output.z,output.w,"
                         "STATE,boxmuller_flag; i32 stores raw uint32 bits");
    auto dec = prec({B, A + 1}, 9, 5), mask = masks(B, A, true);
    auto actions = flt({B, NUM_ATNS}), env = flt({B, NUM_ATNS}), draw = flt({B, NUM_ATNS});
    auto lp = prec({B}), val = prec({B});
    int *sizes = action_sizes();
    curandStatePhilox4_32_10_t *rng;
    golden::check(cudaMalloc(&rng, B * sizeof(*rng)));
    loose.push_back(rng);
    golden::check(cudaMemset(rng, 0, B * sizeof(*rng)));
    rng_init<<<grid_size(B), BLOCK_SIZE>>>(rng, 73, B);
    warm_rng<<<grid_size(B), BLOCK_SIZE>>>(rng, B);
    c.note("rng_initialization",
           "curand_init(73,agent,0); discard agent%7 uniforms before input snapshot");
    unsigned int *fields;
    golden::check(cudaMalloc(&fields, B * 12 * 4));
    loose.push_back(fields);
    rng_fields<<<grid_size(B), BLOCK_SIZE>>>(rng, fields, B);
    c.device("rng_fields_in", fields, {B, 12}, "i32", "state");
    c.device("rng_in", rng, {B, (int64_t)sizeof(*rng)}, "u8", "state");
    c.tensor("decoder", dec, "input");
    c.tensor("masks", mask, "input");
    c.device("act_sizes", sizes, {NUM_ATNS}, "i32", "input");
    uniform_probe<<<grid_size(B), BLOCK_SIZE>>>(rng, draw.data, B);
    sample_logits<<<grid_size(B), BLOCK_SIZE>>>(dec, {}, sizes, actions.data, env.data, lp.data,
                                                val.data, rng, mask.data, A);
    rng_fields<<<grid_size(B), BLOCK_SIZE>>>(rng, fields, B);
    c.device("rng_fields_out", fields, {B, 12}, "i32");
    c.device("rng_out", rng, {B, (int64_t)sizeof(*rng)}, "u8");
    c.tensor("uniform_draws", draw);
    c.tensor("actions", actions);
    c.tensor("env_actions", env);
    c.tensor("logprobs", lp);
    c.tensor("values", val);
    c.finish();
    clean();
}
static void advantage(bool real, bool vt) {
    int B = real ? 2048 : 3, T = real ? 32 : 8;
    Case c(root + "/adv_" + (real ? "breakout_" : "tiny_") + (vt ? "vtrace" : "gae"));
    c.note("operation", "puff_advantage");
    dims(c, B, T, 0);
    float gamma = .96761483f, lambda = .982877254f, rho = vt ? .8f : 1, cclip = vt ? .6f : 1;
    c.scalar("gamma", gamma);
    c.scalar("lambda", lambda);
    c.scalar("rho_clip", rho);
    c.scalar("c_clip", cclip);
    c.scalar("importance_is_null", !vt);
    auto v = prec({B, T}, 3), r = prec({B, T}, 4, 1), d = prec({B, T}), imp = prec({B, T}),
         adv = prec({B, T});
    std::vector<float> flags(B * T), imps(B * T);
    for (int i = 0; i < B * T; i++) {
        flags[i] = (i % T == T / 2 || i % T == T - 1) ? 1 : 0;
        imps[i] = 1 + golden::pattern(i, 8, .95);
    }
    fill(d.data, flags);
    fill(imp.data, imps);
    c.tensor("values", v, "input");
    c.tensor("rewards", r, "input");
    c.tensor("terminals", d, "input");
    if (vt)
        c.tensor("importance", imp, "input");
    puff_advantage<<<(B + 63) / 64, 64>>>(v.data, r.data, d.data, vt ? imp.data : nullptr, adv.data,
                                          v.data, gamma, lambda, rho, cclip, B, T);
    c.tensor("advantages", adv);
    c.tensor("returns", v);
    c.finish();
    clean();
}
static void loss_case(bool real) {
    int B = real ? 2048 : 3, T = real ? 32 : 8, A = 3, N = B * T;
    Case c(root + (real ? "/ppo_breakout" : "/ppo_tiny"));
    c.note("operation", "cache_imp_and_v+ppo_loss_compute+ppo_loss_reduce");
    dims(c, B, T, 0);
    c.note("loss_order", "PG,VF,ENT,TOTAL,OLD_APPROX_KL,APPROX_KL,CLIPFRAC,IMP,count");
    auto dec = prec({B, T, A + 1}, 4, 8), oldlp = prec({B, T}), mask = masks(N, A),
         values = prec({B, T}, 5, 8), returns = prec({B, T}, 6, 12), adv = prec({B, T}, 7, 4),
         imp = prec({B, T}), live = prec({B, T});
    auto act = flt({B, T, NUM_ATNS});
    int *sizes = action_sizes();
    std::vector<float> ac(N), ol(N);
    for (int i = 0; i < N; i++) {
        ac[i] = i % 4 == 1 ? 1 : 0;
        ol[i] = -1 + golden::pattern(i, 11, 2);
    }
    fill(act.data, ac);
    fill(oldlp.data, ol);
    Allocator arena = {};
    PPOBufs ppo;
    register_ppo_buffers(ppo, &arena, B, T, A, false);
    alloc_create(&arena);
    float ent = .00489364704f;
    golden::check(cudaMemcpy(ppo.ent_coef, &ent, 4, cudaMemcpyHostToDevice));
    auto losses = flt({NUM_LOSSES}, 1, 0);
    c.tensor("decoder", dec, "input");
    c.tensor("actions", act, "input");
    c.tensor("old_logprobs", oldlp, "input");
    c.tensor("mask", mask, "input");
    c.tensor("old_values", values, "input");
    c.tensor("returns", returns, "input");
    c.tensor("advantages", adv, "input");
    c.device("act_sizes", sizes, {NUM_ATNS}, "i32", "input");
    c.tensor("losses_in", losses, "state");
    c.scalar("clip_coef", .81228894f);
    c.scalar("vf_clip_coef", 4.97819614f);
    c.scalar("vf_coef", 3.46657705f);
    c.scalar("ent_coef", ent);
    cache_imp_and_v<<<grid_size(N), BLOCK_SIZE>>>(dec, act.data, oldlp.data, mask.data, {}, sizes,
                                                  imp.data, live.data, ppo.grad_logits.data,
                                                  ppo.grad_values.data);
    c.tensor("cached_logps", ppo.grad_logits);
    c.tensor("cached_new_logprobs", ppo.grad_values);
    c.tensor("importance", imp);
    c.tensor("live_values", live);
    TrainGraph g = {};
    g.mb_actions = act;
    g.mb_logprobs = oldlp;
    g.mb_action_mask = mask;
    g.mb_imp = imp;
    g.mb_values = values;
    g.mb_returns = returns;
    g.mb_advantages = adv;
    Prec logstd = {};
    ppo_loss_fwd_bwd(dec, logstd, g, sizes, losses.data, .81228894f, 4.97819614f, 3.46657705f,
                     ppo.ent_coef, ppo, false, 0);
    c.tensor("dlogits", ppo.grad_logits);
    c.tensor("dvalue", ppo.grad_values);
    c.tensor("loss_partials", ppo.ppo_partials);
    c.tensor("losses", losses);
    c.finish();
    golden::check(cudaFree(arena.mem));
    golden::check(cudaFree(ppo.ent_coef));
    clean();
}
static void muon_case(bool real, bool full, bool wide = false) {
    int H = real ? 32 : (full ? 8 : 5);
    Case c(root + "/muon_" + (real ? "breakout_" : "tiny_") +
           (full   ? "full"
            : wide ? "wide"
                   : "matrix"));
    c.note("operation", "muon_step");
    Allocator params = {}, acts = {};
    Arch arch;
    Prec single = {};
    if (full) {
        arch = build_arch(real ? OBS_SIZE : 12, H, real ? 4 : 2, 3, false, real ? 32 : 8);
        weights_create(&arch, &params);
    } else {
        single = {.shape = {wide ? 4 : 3 * H, H}};
        alloc_register(&params, &single);
    }
    alloc_create(&params);
    long n = params.total_elems;
    std::vector<float> pv(n), gv(n), mv(n);
    for (long i = 0; i < n; i++) {
        pv[i] = golden::pattern(i, 12, .1);
        gv[i] = golden::pattern(i, 13, .2);
        mv[i] = golden::pattern(i, 14, .1);
    }
    fill((float *)params.mem, pv);
    auto grads = prec({n});
    fill(grads.data, gv);
    Muon m = {};
    muon_init(&m, &params, .869982362f, &acts);
    alloc_create(&acts);
    fill(m.mb.data, mv);
    float lr = .0627130643f;
    golden::check(cudaMemcpy(m.lr, &lr, 4, cudaMemcpyHostToDevice));
    Float weights = {.data = (float *)params.mem, .shape = {n}};
    c.scalar("lr", lr);
    c.scalar("momentum", m.momentum);
    c.scalar("max_grad_norm", 1.10759413f);
    c.scalar("clip_eps", 1e-6f);
    c.scalar("ns_eps", 1e-7f);
    c.scalar("weight_decay", 0);
    c.scalar("ns_iterations", 5);
    c.tensor("params_in", weights, "param");
    c.tensor("grads", grads, "input");
    c.tensor("momentum_in", m.mb, "state");
    long off = 0;
    for (int i = 0; i < params.num_regs; i++) {
        auto &e = params.regs[i];
        c.device("param_" + std::to_string(i) + "_in", *e.data_ptr, golden::shape(e.shape), "f32",
                 "param");
        c.scalar("param_" + std::to_string(i) + "_offset", off);
        off += numel(e.shape);
    }
    std::vector<float> coeff;
    for (auto &row : ns_coeffs)
        for (double v : row)
            coeff.push_back((float)v);
    c.host("ns_coefficients", coeff.data(), {5, 3}, "f32", "input");
    muon_step(&m, weights, grads, 1.10759413f);
    c.tensor("params_out", weights);
    c.tensor("momentum_out", m.mb);
    c.tensor("update", grads);
    c.device("grad_sum_sq", m.grad_norm, {1});
    for (int i = 0; i < params.num_regs; i++) {
        auto &e = params.regs[i];
        c.device("param_" + std::to_string(i) + "_out", *e.data_ptr, golden::shape(e.shape));
    }
    c.finish();
    golden::check(cudaFree(params.mem));
    golden::check(cudaFree(acts.mem));
    for (float *p : {m.lr, m.grad_norm, m.ns_norm, m.norm_partials})
        golden::check(cudaFree(p));
    clean();
}
static void policy_case(bool real, bool with_loss = false) {
    int B = real ? 2048 : 2, T = real ? 32 : 8, H = real ? 32 : 8, L = real ? 4 : 2,
        I = real ? OBS_SIZE : 12, A = 3;
    Case c(root + (with_loss ? (real ? "/policy_ppo_breakout" : "/policy_ppo_tiny")
                             : (real ? "/policy_breakout" : "/policy_tiny")));
    c.note("operation", with_loss ? "arch_forward_train+GAE+PPO+arch_backward"
                                  : "arch_forward_train+arch_backward");
    dims(c, B, T, H);
    c.scalar("layers", L);
    c.scalar("input_dim", I);
    int agent_off = real ? 2048 : 1;
    c.scalar("agent_off", agent_off);
    Arch arch = build_arch(I, H, L, A, false, T);
    Allocator params = {}, acts = {}, grads = {};
    Weights w = weights_create(&arch, &params);
    Activations a = arch_reg_train(&arch, w, &acts, &grads, B * T);
    TrainGraph g;
    register_train_buffers(g, &acts, B, T);
    alloc_create(&params);
    alloc_create(&acts);
    alloc_create(&grads);
    std::vector<float> pv(params.total_elems);
    for (size_t i = 0; i < pv.size(); i++)
        pv[i] = golden::pattern(i, 19, .15);
    fill((float *)params.mem, pv);
    auto obs = prec({B, T, I}, 20), state = prec({L, real ? 4096 : B + 2, H}, 21, .3),
         term = prec({B, T}), mask = masks(B * T, A), oldlp = prec({B, T}, 25);
    auto act = flt({B, T, NUM_ATNS}), dl = flt({B, T, A}, 26, .01), dv = flt({B, T, 1}, 27, .01),
         cached = flt({B, T, A}), newlp = flt({B, T, 1});
    std::vector<float> tv(B * T), av(B * T);
    for (int i = 0; i < B * T; i++) {
        tv[i] = (i % T == T / 2) ? 1 : 0;
        av[i] = i % 4 == 1 ? 1 : 0;
    }
    fill(term.data, tv);
    fill(act.data, av);
    g.mb_actions = act;
    g.mb_logprobs = oldlp;
    g.mb_action_mask = mask;
    int *sizes = action_sizes();
    c.tensor("obs", obs, "input");
    c.tensor("initial_state", state, "state");
    c.tensor("terminals", term, "input");
    c.tensor("actions", act, "input");
    c.tensor("mask", mask, "input");
    c.tensor("old_logprobs", oldlp, "input");
    if (!with_loss) {
        c.tensor("dlogits", dl, "input");
        c.tensor("dvalue", dv, "input");
    }
    c.device("act_sizes", sizes, {NUM_ATNS}, "i32", "input");
    for (int i = 0; i < params.num_regs; i++) {
        auto &e = params.regs[i];
        c.device("param_" + std::to_string(i), *e.data_ptr, golden::shape(e.shape), "f32", "param");
    }
    Prec out = arch_forward_train(&arch, w, a, obs, state, term, agent_off, g, {}, sizes,
                                  cached.data, newlp.data, 0);
    c.tensor("decoder", out);
    c.tensor("importance", g.mb_imp);
    c.tensor("live_values", g.mb_gae_v);
    c.tensor("cached_logps", cached);
    c.tensor("new_logprobs", newlp);
    auto *ma = (MinGRUActivations *)a.network;
    for (int i = 0; i < L; i++) {
        c.tensor("layer_" + std::to_string(i) + "_out", ma->scan_bufs[i].out);
        c.tensor("layer_" + std::to_string(i) + "_next_state", ma->scan_bufs[i].next_state);
    }
    Allocator ppoarena = {};
    PPOBufs ppob = {};
    if (with_loss) {
        auto rewards = prec({B, T}, 31), oldvalues = prec({B, T}, 32);
        c.tensor("rewards", rewards, "input");
        c.tensor("old_values", oldvalues, "input");
        g.mb_rewards = rewards;
        g.mb_values = oldvalues;
        c.scalar("gamma", .96761483f);
        c.scalar("lambda", .982877254f);
        c.scalar("vtrace", 0);
        c.scalar("clip_coef", .81228894f);
        c.scalar("vf_clip_coef", 4.97819614f);
        c.scalar("vf_coef", 3.46657705f);
        c.scalar("ent_coef", .00489364704f);
        puff_advantage<<<(B + 63) / 64, 64>>>(g.mb_gae_v.data, rewards.data, term.data, nullptr,
                                              g.mb_advantages.data, g.mb_gae_v.data, .96761483f,
                                              .982877254f, 1, 1, B, T);
        g.mb_returns = g.mb_gae_v;
        c.tensor("advantages", g.mb_advantages);
        c.tensor("returns", g.mb_returns);
        register_ppo_buffers(ppob, &ppoarena, B, T, A, false);
        alloc_create(&ppoarena);
        float ent = .00489364704f;
        golden::check(cudaMemcpy(ppob.ent_coef, &ent, 4, cudaMemcpyHostToDevice));
        golden::check(cudaMemcpy(ppob.grad_logits.data, cached.data, B * T * A * 4,
                                 cudaMemcpyDeviceToDevice));
        golden::check(
            cudaMemcpy(ppob.grad_values.data, newlp.data, B * T * 4, cudaMemcpyDeviceToDevice));
        auto losses = flt({NUM_LOSSES}, 1, 0);
        Prec empty = {};
        ppo_loss_fwd_bwd(out, empty, g, sizes, losses.data, .81228894f, 4.97819614f, 3.46657705f,
                         ppob.ent_coef, ppob, false, 0);
        dl = ppob.grad_logits;
        dv = ppob.grad_values;
        c.tensor("dlogits", dl);
        c.tensor("dvalue", dv);
        c.tensor("losses", losses);
        c.tensor("loss_partials", ppob.ppo_partials);
    }
    c.tensor("grad_next_state", ma->grad_next_state, "input");
    arch_backward(&arch, w, a, dl, {}, dv, 0);
    for (int i = 0; i < grads.num_regs; i++) {
        auto &e = grads.regs[i];
        c.device("param_" + std::to_string(i) + "_grad", *e.data_ptr, golden::shape(e.shape));
    }
    for (int i = 0; i < L; i++)
        c.tensor("layer_" + std::to_string(i) + "_grad_state", ma->scan_bufs[i].grad_state);
    c.tensor("encoder_grad", ma->grad_input_buf);
    c.finish();
    if (with_loss) {
        golden::check(cudaFree(ppoarena.mem));
        golden::check(cudaFree(ppob.ent_coef));
    }
    golden::check(cudaFree(params.mem));
    golden::check(cudaFree(acts.mem));
    golden::check(cudaFree(grads.mem));
    clean();
}
static void replay(std::string input, std::string output) {
    Ini ini = {};
    puf_ini_load_env(&ini, "breakout", 0, nullptr);
    TrainContext ctx = {.world_size = 1};
    PuffeRL *p = create_pufferl(&ini, &ctx);
    auto load = [&](std::string n, auto t) {
        golden::upload(t.data, input + "/" + n + ".bin", numel(t.shape) * 4);
    };
    load("params_in", p->policies[0].param);
    load("momentum_in", p->muon.mb);
    load("train_state_in", p->train_state);
    load("grad_next_state", ((MinGRUActivations *)p->train_activs.network)->grad_next_state);
    p->epoch = 3;
    p->global_step = 4 * 4096 * 32;
    p->async_ready_slot = 1;
    RolloutBuf src = rollout_time_view(&p->rollouts, 32, 32);
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
        golden::upload(p->rng_states[b], input + "/rng_" + std::to_string(b) + "_in.bin",
                       4096 * sizeof(curandStatePhilox4_32_10_t));
    }
    golden::upload(p->rng_offset, input + "/rng_offsets_in.bin", 16);
    golden::upload(p->losses, input + "/losses_in.bin", NUM_LOSSES * 4);
    setenv("PUF_GOLDEN_TRAIN", output.c_str(), 1);
    train_impl(p, &src);
    printf("replay completed\n");
}
int main(int argc, char **argv) {
    setbuf(stdout, nullptr);
    if (argc == 4 && std::string(argv[1]) == "replay") {
        replay(argv[2], argv[3]);
        return 0;
    }
    if (argc != 2) {
        fprintf(stderr, "usage: generate OUT | generate replay INPUT OUTPUT\n");
        return 2;
    }
    root = argv[1];
    cublas_init_handle();
    for (bool real : {false, true}) {
        gemms(real);
        gate(real);
        scan_case(real ? "scan_breakout" : "scan_tiny", real ? 2048 : 2, real ? 32 : 8,
                  real ? 32 : 5, false);
        scan_case(real ? "scan_row_breakout" : "scan_row_tiny", real ? 2048 : 2, real ? 32 : 35,
                  real ? 32 : 5, true);
        sample_case(real);
        for (bool vt : {false, true})
            advantage(real, vt);
        loss_case(real);
        muon_case(real, false);
        muon_case(real, false, true);
        muon_case(real, true);
        policy_case(real);
        policy_case(real, true);
    }
    scan_case("scan_row_long", 2, 129, 32, true);
    scan_case("scan_plain_long", 2, 129, 32, false);
    scan_case("scan_plain_partial", 2, 35, 5, false);
    golden::check(cudaGetLastError());
    return 0;
}

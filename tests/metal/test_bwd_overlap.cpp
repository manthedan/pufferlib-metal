// Exercise the actual learner backward and per-epoch policy in one translation
// unit. No production hooks: this wrapper only observes deferred GEMM results.
#include "launch.h"
#include "runtime.h"
#include "gemm.h"
#include "rollout.h"
#include <vector>

struct WorkspaceStats {
    int exhausted = 0, retries = 0;
    size_t high_water = 0;
    bool waiting_retry = false, in_bounds = true;
};
static WorkspaceStats workspace_stats;

namespace puf_mtl {
template<class T>
static GemmPath observed_gemm_rm_deferred(cudaStream_t s, bool ta, bool tb,
        int M, int N, int K, float alpha, const T* A, int lda, const T* B, int ldb,
        float beta, T* C, int ldc, void* ws, size_t bytes, size_t off,
        GemmDeferred* d, const GemmOpts* opts = nullptr) {
    GemmPath path = gemm_rm_deferred(s, ta, tb, M, N, K, alpha, A, lda, B, ldb,
        beta, C, ldc, ws, bytes, off, d, opts);
    if (path == GEMM_NONE) {
        workspace_stats.exhausted++;
        workspace_stats.waiting_retry = true;
    } else {
        if (workspace_stats.waiting_retry && off == 0) workspace_stats.retries++;
        workspace_stats.waiting_retry = false;
        workspace_stats.in_bounds &= off <= bytes && d->ws_used <= bytes - off;
        if (off + d->ws_used > workspace_stats.high_water)
            workspace_stats.high_water = off + d->ws_used;
    }
    return path;
}
}

#define gemm_rm_deferred observed_gemm_rm_deferred
#define main puf_upstream_main
#include "../../src/metal/pufferl_metal.cpp"
#undef main
#undef gemm_rm_deferred

static int failures;
static void check(bool ok, const char* name) {
    printf("  %s: %s\n", name, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

// Shared-memory inputs change each minibatch; binary fractions make host
// generation identical in FP32/BF16 and keep the recurrent chain finite.
template<class Tensor>
static void fill(Tensor t, int salt, float scale) {
    for (int64_t i = 0; i < numel(t.shape); i++)
        t.data[i] = from_float(float((i * 17 + salt * 13) % 61 - 30) * scale);
}

struct BackwardCase {
    int B, T = 64, H = 64, L;
    Allocator params = {}, acts = {}, grads = {};
    Arch arch;
    Weights weights;
    Activations activations;
    Prec input, state, terminals;
    Float logits, value;
    cudaStream_t stream;

    BackwardCase(int batch, int layers) : B(batch), L(layers) {
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cublas_init_stream(stream);
        arch = build_arch(52, H, L, 7, false, T);
        weights = weights_create(&arch, &params);
        activations = arch_reg_train(&arch, weights, &acts, &grads, B * T);
        input = {.shape = {B * T, 52}};
        state = {.shape = {L, B, H}};
        terminals = {.shape = {B, T}};
        logits = {.shape = {B, T, 7}};
        value = {.shape = {B, T}};
        alloc_register(&acts, &input);
        alloc_register(&acts, &state);
        alloc_register(&acts, &terminals);
        alloc_register(&acts, &logits);
        alloc_register(&acts, &value);
        alloc_create(&params);
        alloc_create(&acts);
        alloc_create(&grads);
        // Flat registration order includes every encoder/recurrent/decoder weight.
        Prec flat = {(precision_t*)params.mem, {params.total_elems}};
        fill(flat, 3, 1.0f / 512);
    }

    void forward(int mb) {
        fill(input, mb + 1, 1.0f / 32);
        fill(state, mb + 2, 1.0f / 64);
        fill(logits, mb + 3, 1.0f / 128);
        fill(value, mb + 4, 1.0f / 128);
        for (int i = 0; i < B * T; i++)
            terminals.data[i] = from_float((i + mb * 11) % 97 == 0 ? 1.0f : 0.0f);
        auto* ma = (MinGRUActivations*)activations.network;
        fill(ma->grad_next_state, mb + 5, 1.0f / 256);
        Prec h = encoder_forward(weights.encoder, activations.encoder, input, stream);
        h = *puf_unsqueeze(&h, 0, B, T);
        h = mingru_forward_train(weights.network, h, state, terminals,
            activations.network, 0, stream);
        h = *puf_squeeze(&h, 0);
        decoder_forward(weights.decoder, activations.decoder, h, stream);
        cudaStreamSynchronize(stream);
        bool fused = true;
        for (int i = 0; i < L; i++) fused &= ma->fused[i];
        check(fused, "all H64 layers use the fused path");
        // Poison gradients, but do not clear the workspace between minibatches.
        memset(grads.mem, 0xff, grads.total_bytes);
    }

    puf_mtl::Stats backward() {
        auto before = puf_mtl::stream_stats(stream);
        arch_backward(&arch, weights, activations, logits, {}, value, stream);
        auto after = puf_mtl::stream_stats(stream);
        cudaStreamSynchronize(stream);
        check(!g_ov.active && !g_ov.open && g_ov.ws_off == 0
            && g_ov.queued.empty() && g_ov.ready.empty() && g_ov.fresh.empty(),
            "backward drains and resets workspace for the next minibatch");
        after.dispatches -= before.dispatches;
        after.barriers -= before.barriers;
        return after;
    }

    std::vector<unsigned char> gradients() {
        std::vector<unsigned char> result;
        auto append = [&](Prec t) {
            auto* p = (const unsigned char*)t.data;
            result.insert(result.end(), p, p + numel(t.shape) * sizeof(precision_t));
            for (int64_t i = 0; i < numel(t.shape); i++)
                if (!std::isfinite(to_float(t.data[i]))) {
                    check(false, "all gradients finite (no poison)");
                    break;
                }
        };
        append(((EncoderActivations*)activations.encoder)->wgrad_scratch);
        append(((DecoderActivations*)activations.decoder)->wgrad_scratch);
        auto* ma = (MinGRUActivations*)activations.network;
        append(ma->grad_input_buf);
        for (int i = 0; i < L; i++) {
            append(ma->wgrad_scratch[i]);
            append(ma->scan_bufs[i].grad_combined);
            append(ma->scan_bufs[i].grad_state);
        }
        return result;
    }

    ~BackwardCase() {
        cudaDeviceSynchronize();
        auto* ma = (MinGRUActivations*)activations.network;
        free(ma->saved_inputs); free(ma->scan_bufs); free(ma->combined_bufs);
        free(ma->wgrad_scratch); free(ma->fused);
        free(((MinGRUWeights*)weights.network)->weights);
        free(weights.encoder); free(weights.decoder); free(weights.network);
        free(activations.encoder); free(activations.decoder); free(activations.network);
        for (Allocator* a : {&params, &acts, &grads}) {
            cudaFree(a->mem);
            free(a->regs);
        }
        // Retire this fixture's cuBLAS registration (always the last one).
        auto& cs = g_cublas_streams[--g_num_cublas_streams];
        cublasDestroy(cs.handle);
        cudaFree(cs.workspace);
        cudaStreamDestroy(stream);
    }
};

static void test_exhaustion() {
    printf("== integrated H64 workspace exhaustion, three minibatches\n");
    // K=32768, four H64 dWs plus encoder/decoder dW exceed the real 4 MiB
    // workspace. Keep the full capacity: split counts match overlap disabled.
    BackwardCase c(512, 4);
    std::vector<std::vector<unsigned char>> reference;
    std::vector<uint64_t> dispatches;
    for (int mb = 0; mb < 3; mb++) {
        c.forward(mb);
        g_metal_bwd_overlap = false;
        auto stats = c.backward();
        reference.push_back(c.gradients());
        dispatches.push_back(stats.dispatches);
    }
    void* ws = g_cublas_streams[g_num_cublas_streams - 1].workspace;
    memset(ws, 0xff, kCublasWsBytes);
    for (int mb = 0; mb < 3; mb++) {
        c.forward(mb);
        g_metal_bwd_overlap = true;
        workspace_stats = {};
        auto stats = c.backward();
        check(c.gradients() == reference[mb], "every gradient bit-exact vs overlap disabled");
        check(stats.dispatches == dispatches[mb], "exhaustion preserves dispatch count");
        check(g_ov.ws == ws && g_ov.ws_bytes == kCublasWsBytes
            && workspace_stats.in_bounds && workspace_stats.exhausted > 0
            && workspace_stats.retries == workspace_stats.exhausted
            && !workspace_stats.waiting_retry,
            "same 4 MiB workspace exhausted, flushed and reused each minibatch");
        printf("  minibatch %d: exhausted=%d retries=%d high_water=%zu/%zu\n", mb,
            workspace_stats.exhausted, workspace_stats.retries,
            workspace_stats.high_water, kCublasWsBytes);
    }
}

static void test_policy() {
    printf("== automatic overlap policy threshold and schedule transitions\n");
    BackwardCase c(64, 2);
    puf_mtl::Stats expected[2];
    std::vector<unsigned char> reference;
    for (int on = 0; on < 2; on++) {
        c.forward(0);
        g_metal_bwd_overlap = on;
        expected[on] = c.backward();
        if (on == 0) reference = c.gradients();
        else check(c.gradients() == reference, "both schedules have identical gradients");
    }
    check(expected[0].dispatches == expected[1].dispatches
        && expected[1].barriers < expected[0].barriers,
        "on/off schedule baselines have equal dispatches and fewer overlap barriers");
    // Invoke the actual policy, including exact and adjacent-double boundaries.
    // Two buffers keep the independent decoder side-stream policy off.
    PuffeRL p = {};
    p.hypers.num_buffers = 2;
    struct Case { double slack, epoch; bool on; };
    const Case cases[] = {
        {0, 100, true},
        {std::nextafter(5.0, INFINITY), 100, false}, // on -> off
        {5, 100, true},                            // off -> on, inclusive threshold
        {std::nextafter(5.0, 0.0), 100, true},
        {20, 100, false},
        {0, 1, true},                              // sync-training inputs
        {0.051, 1, false},
        {0.05, 1, true},                           // scaled threshold
    };
    for (const auto& t : cases) {
        metal_dw_policy(&p, t.slack, t.epoch);
        check(g_metal_bwd_overlap == t.on && !g_metal_dw_side,
            "policy selects expected overlap flag at the 5% boundary");
        c.forward(0);
        auto stats = c.backward();
        check(stats.dispatches == expected[t.on].dispatches
            && stats.barriers == expected[t.on].barriers,
            "next backward uses the selected schedule after policy transition");
        check(c.gradients() == reference, "policy transitions preserve every gradient bit");
        printf("  slack=%.17g epoch=%g on=%d dispatches=%llu barriers=%llu\n",
            t.slack, t.epoch, t.on, (unsigned long long)stats.dispatches,
            (unsigned long long)stats.barriers);
    }
}

static void test_rollout_group_policy() {
    printf("== rollout group policy: epochs 3-8, median > 5%%, one decision\n");
    const float threshold = 0.05f;
    struct Case { const char* name; float samples[6]; int group; };
    const Case cases[] = {
        {"below threshold", {0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.90f}, 2},
        {"exact threshold", {threshold, threshold, threshold, threshold, threshold, threshold}, 2},
        {"adjacent float below", {std::nextafter(threshold, 0.0f), std::nextafter(threshold, 0.0f),
            std::nextafter(threshold, 0.0f), std::nextafter(threshold, 0.0f),
            std::nextafter(threshold, 0.0f), std::nextafter(threshold, 0.0f)}, 2},
        {"adjacent float above", {std::nextafter(threshold, 1.0f), std::nextafter(threshold, 1.0f),
            std::nextafter(threshold, 1.0f), std::nextafter(threshold, 1.0f),
            std::nextafter(threshold, 1.0f), std::nextafter(threshold, 1.0f)}, 4},
        // Unsorted and skewed: median is 6%, mean < 5%, final sample < 5%.
        {"median, not mean or last sample", {0.06f, 0.0f, 0.07f, 0.06f, 0.08f, 0.01f}, 4},
    };
    for (const auto& c : cases) {
        printf("  %s\n", c.name);
        PuffeRL p = {};
        p.hypers.num_buffers = 4;
        p.hypers.async = true;
        // Epochs 0-2 must not contribute even with opposing warm-up slack.
        for (p.epoch = 0; p.epoch <= 2; p.epoch++) {
            metal_dw_policy(&p, c.group == 4 ? 0.0 : 1.0, 1.0);
            check(p.metal_slack_n == 0 && p.metal_group_pref == 0,
                "warm-up epochs are excluded and preference stays undecided");
        }
        for (int i = 0; i < 6; i++) {
            p.epoch = i + 3;
            // Non-unit epoch duration also checks slack normalization.
            metal_dw_policy(&p, 8.0 * c.samples[i], 8.0);
            check(p.metal_slack_n == i + 1 && p.metal_slack[i] == c.samples[i],
                "each epoch 3-8 records exactly its normalized slack sample");
            check(p.metal_group_pref == (i == 5 ? c.group : 0),
                "preference is chosen only at epoch 8 with the expected threshold outcome");
        }
        float samples[8];
        memcpy(samples, p.metal_slack, sizeof(samples));
        for (p.epoch = 9; p.epoch < 25; p.epoch++) {
            metal_dw_policy(&p, c.group == 4 ? 0.0 : 1.0, 1.0);
            check(p.metal_group_pref == c.group && p.metal_slack_n == 6
                && memcmp(samples, p.metal_slack, sizeof(samples)) == 0,
                "later opposing slack cannot change the decision or sampling window");
        }
    }
    PuffeRL p = {};
    p.epoch = 3;
    metal_dw_policy(&p, 1.0, 1.0);
    check(p.metal_slack_n == 0 && p.metal_group_pref == 0,
        "synchronous epochs do not sample the rollout group policy");
    p.hypers.async = true;
    metal_dw_policy(&p, 1.0, 0.0);
    check(p.metal_slack_n == 0 && p.metal_group_pref == 0,
        "nonpositive epoch duration does not sample the rollout group policy");
}

int main() {
    setbuf(stdout, nullptr);
    cublas_init_handle();
    test_exhaustion();
    test_policy();
    test_rollout_group_policy();
    printf(failures ? "FAILED: %d checks\n" : "ALL PASS\n", failures);
    return failures ? 1 : 0;
}

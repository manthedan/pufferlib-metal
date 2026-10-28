// Host-gated commits (runtime.h gate_arm / gate_commit / gate_open /
// gate_ran), as pufferl.cu metal_gated_horizon uses them for rollout steps.
// Each gated step is one puf_copy_bytes dispatch copying a host-written
// "env output" word into its own result slot; the host writes the word only
// after the previous step completed, then opens the gate.
//   pipeline: 400 pipelined steps (step i+1 committed behind the gate while
//             step i runs) on two streams from two threads: every result is
//             the value written right before its gate opened, gate_ran is
//             true for every step, and no gated step starts before its gate.
//   early:    the gate stays closed 4.6 s, longer than Metal holds a queue
//             wait: the dispatch must have run empty (result untouched,
//             gate_ran false); an ungated redo then lands.
//   cancel:   gate_open(run=false) releases an empty dispatch.
//   strict:   with a gated commit pending, a strict host call (cudaMemcpy ->
//             join_all_feedback) on the gate's own thread returns at once;
//             the feedback of the commit before the gated one is held back
//             by MTL4 meanwhile (reported, not required).
//   spin:     stream_wait_spin polls before blocking: waits return only
//             after completion, both when the poll sees it and when work is
//             held past the budget; gated steps work on a polling stream.
//   misuse:   double arm, two dispatches in a gated buffer, gate_open without
//             a pending commit, gate_ran on an unknown value -> fatal.
// Build/run: tests/metal/run_test_gate.sh [case...]
#import <Foundation/Foundation.h>

#include <mach/mach_time.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include "compat/cuda_runtime.h"
#include "runtime.h"

static int g_fail = 0;

static void check(const char* name, bool ok) {
    if (!ok) g_fail++;
    printf("  %-60s %s\n", name, ok ? "ok" : "FAIL");
    fflush(stdout);
}

static double now_s() {
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

struct FatalError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
static void throw_fatal(const char* msg) { throw FatalError(msg); }

struct CopyArgs {  // runtime.mm puf_copy_bytes
    uint64_t dst, src, n;
    uint32_t aligned, pad;
};

static puf_mtl::Pipeline* copy_pipe() {
    static puf_mtl::Pipeline* p = puf_mtl::pipeline("puf_copy_bytes");
    return p;
}

// One dispatch: dst[0..3] = src[0..3] (16 bytes).
static void copy16(cudaStream_t s, uint32_t* dst, const uint32_t* src) {
    CopyArgs a = {puf_mtl::gpu_addr(dst), puf_mtl::gpu_addr(src), 16, 1, 0};
    puf_mtl::dispatch(s, copy_pipe(), dim3(1), dim3(256), a);
}

static cudaStream_t gated_stream() {
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    puf_mtl::stream_host_wait_only(s);
    return s;
}

static uint32_t* ualloc(size_t n) {
    void* p = nullptr;
    cudaMalloc(&p, n * sizeof(uint32_t));
    return (uint32_t*)p;
}

// Worker-style pipelined loop (metal_gated_horizon without the env).
struct PipeResult {
    int wrong = 0, not_ran = 0;
};

static PipeResult pipeline_run(int steps) {
    cudaStream_t s = gated_stream();
    uint32_t* env = ualloc(4);      // "env output" the step reads
    uint32_t* out = ualloc(4 * (size_t)steps);
    for (int i = 0; i < 4 * steps; i++) out[i] = 0xffffffffu;
    cudaEvent_t done[3];
    for (auto& e : done) cudaEventCreate(&e);
    PipeResult r;
    env[0] = 1;  // step 0's input
    copy16(s, out, env);
    cudaEventRecord(done[0], s);
    uint64_t cur = 0;  // gated commit of step t (0: step 0, ungated)
    for (int t = 0; t < steps; t++) {
        uint64_t next = 0;
        if (t + 1 < steps) {
            puf_mtl::gate_arm(s);
            copy16(s, out + 4 * (t + 1), env);
            next = puf_mtl::gate_commit(s);
            cudaEventRecord(done[(t + 1) % 3], s);
        }
        cudaEventSynchronize(done[t % 3]);
        if (cur && !puf_mtl::gate_ran(s, cur)) r.not_ran++;
        // "env step": the next step must read this value, not the old one.
        usleep(t % 7 == 0 ? 300 : 20);
        if (next) {
            // Written after the previous step completed and only now
            // released: a step that started before gate_open reads t + 1.
            env[0] = (uint32_t)(t + 2);
            puf_mtl::gate_open(s);
        }
        cur = next;
    }
    cudaStreamSynchronize(s);
    for (int t = 0; t < steps; t++) {
        if (out[4 * t] != (uint32_t)(t + 1)) {
            if (r.wrong < 5) printf("    step %d read %u, want %d\n", t, out[4 * t], t + 1);
            r.wrong++;
        }
    }
    for (auto& e : done) cudaEventDestroy(e);
    return r;
}

static void case_pipeline() {
    printf("pipeline\n");
    uint64_t gated0 = puf_mtl::stats().gated;
    PipeResult a, b;
    std::thread th([&] { b = pipeline_run(400); });
    a = pipeline_run(400);
    th.join();
    check("every gated step read its released input (stream A)", a.wrong == 0);
    check("every gated step read its released input (stream B)", b.wrong == 0);
    check("gate_ran true for every gated step (A, B)", a.not_ran == 0 && b.not_ran == 0);
    puf_mtl::Stats st = puf_mtl::stats();
    check("gated commits counted (798)", st.gated - gated0 == 798);
}

static void case_early() {
    printf("early (takes ~5 s)\n");
    cudaStream_t s = gated_stream();
    uint32_t* env = ualloc(4);
    uint32_t* out = ualloc(4);
    env[0] = 7;
    out[0] = 0xffffffffu;
    puf_mtl::gate_arm(s);
    copy16(s, out, env);
    uint64_t v = puf_mtl::gate_commit(s);
    cudaEvent_t e;
    cudaEventCreate(&e);
    cudaEventRecord(e, s);
    usleep(4600000);  // Metal releases the queue wait ~4.0 s after submission
    env[0] = 8;
    puf_mtl::gate_open(s);
    cudaEventSynchronize(e);
    check("released early: gate_ran false", !puf_mtl::gate_ran(s, v));
    check("released early: the dispatch did not run", out[0] == 0xffffffffu);
    copy16(s, out, env);  // the caller's redo
    cudaStreamSynchronize(s);
    check("ungated redo reads the final input", out[0] == 8);
    // The stream keeps working gated afterwards.
    puf_mtl::gate_arm(s);
    copy16(s, out, env);
    v = puf_mtl::gate_commit(s);
    env[0] = 9;
    puf_mtl::gate_open(s);
    cudaStreamSynchronize(s);
    check("next gated step runs", puf_mtl::gate_ran(s, v) && out[0] == 9);
    cudaEventDestroy(e);
}

static void case_cancel() {
    printf("cancel\n");
    cudaStream_t s = gated_stream();
    uint32_t* env = ualloc(4);
    uint32_t* out = ualloc(4);
    env[0] = 3;
    out[0] = 0xffffffffu;
    puf_mtl::gate_arm(s);
    copy16(s, out, env);
    uint64_t v = puf_mtl::gate_commit(s);
    puf_mtl::gate_open(s, false);
    cudaStreamSynchronize(s);
    check("cancelled: gate_ran false", !puf_mtl::gate_ran(s, v));
    check("cancelled: the dispatch did not run", out[0] == 0xffffffffu);
}

static void case_strict() {
    printf("strict\n");
    cudaStream_t s = gated_stream();
    uint32_t* env = ualloc(4);
    uint32_t* out = ualloc(4);
    uint32_t* host = ualloc(4);
    env[0] = 5;
    copy16(s, out, env);
    cudaEvent_t e;
    cudaEventCreate(&e);
    cudaEventRecord(e, s);
    puf_mtl::gate_arm(s);
    copy16(s, out, env);
    uint64_t v = puf_mtl::gate_commit(s);
    cudaEventSynchronize(e);  // the commit before the gated one completed
    usleep(50000);
    uint64_t held = puf_mtl::unconfirmed_commits();
    printf("    feedback held behind the closed gate: %llu commit(s)\n",
        (unsigned long long)held);
    double t0 = now_s();
    uint32_t tmp[4];
    cudaMemcpy(tmp, host, sizeof(tmp), cudaMemcpyDeviceToHost);  // strict join
    double dt = now_s() - t0;
    printf("    strict join with the gate closed: %.1f ms\n", dt * 1e3);
    check("strict join does not wait for the closed gate", dt < 1.0);
    env[0] = 6;
    puf_mtl::gate_open(s);
    cudaStreamSynchronize(s);
    check("gated step ran after the strict join", puf_mtl::gate_ran(s, v) && out[0] == 6);
    cudaDeviceSynchronize();
    check("all feedback confirmed after the gate opened", puf_mtl::unconfirmed_commits() == 0);
    cudaEventDestroy(e);
}

// stream_wait_spin: a bounded poll before the blocking wait. Waits must
// still return only after completion, on hits (short work) and misses (work
// held past the budget), and gated steps must still work on such a stream.
static void case_spin() {
    printf("spin\n");
    cudaStream_t s = gated_stream();
    puf_mtl::stream_wait_spin(s, 2000);
    uint32_t* env = ualloc(4);
    uint32_t* out = ualloc(4 * 64);
    cudaEvent_t e;
    cudaEventCreate(&e);
    puf_mtl::Stats st0 = puf_mtl::stream_stats(s);
    int wrong = 0;
    for (int i = 0; i < 64; i++) {
        env[0] = (uint32_t)(100 + i);
        out[4 * i] = 0xffffffffu;
        copy16(s, out + 4 * i, env);
        cudaEventRecord(e, s);
        cudaEventSynchronize(e);
        if (out[4 * i] != (uint32_t)(100 + i)) wrong++;
    }
    puf_mtl::Stats st1 = puf_mtl::stream_stats(s);
    printf("    short waits: %llu hits, %llu misses\n",
        (unsigned long long)(st1.spin_hit - st0.spin_hit),
        (unsigned long long)(st1.spin_miss - st0.spin_miss));
    check("short waits return after completion", wrong == 0);
    // Start with work definitely pending, then complete it well inside a
    // generous 200 ms budget. A us-as-ns budget is only 200 us and must miss.
    // The release thread starts its delay only when the host is ready to wait.
    puf_mtl::stream_wait_spin(s, 200000);
    void* hit_hold = puf_mtl::test_hold(s);
    env[0] = 11;
    out[0] = 0xffffffffu;
    copy16(s, out, env);
    cudaEventRecord(e, s);
    check("controlled hit: held work has not run", out[0] == 0xffffffffu);
    std::atomic<bool> waiting{false};
    std::thread hit_rel([&] {
        while (!waiting.load(std::memory_order_acquire)) std::this_thread::yield();
        usleep(20000);
        puf_mtl::test_release(hit_hold);
    });
    waiting.store(true, std::memory_order_release);
    cudaEventSynchronize(e);
    hit_rel.join();
    puf_mtl::Stats st_hit = puf_mtl::stream_stats(s);
    check("controlled hit: result visible", out[0] == 11);
    check("controlled hit: completion observed within spin budget", st_hit.spin_hit > st1.spin_hit);
    // Held 30 ms (> 2 ms budget): the poll misses, the block finds completion.
    puf_mtl::stream_wait_spin(s, 2000);
    void* h = puf_mtl::test_hold(s);
    env[0] = 7;
    out[0] = 0xffffffffu;
    copy16(s, out, env);
    cudaEventRecord(e, s);
    std::thread rel([&] { usleep(30000); puf_mtl::test_release(h); });
    double t0 = now_s();
    cudaEventSynchronize(e);
    double dt = now_s() - t0;
    rel.join();
    puf_mtl::Stats st2 = puf_mtl::stream_stats(s);
    check("held work: wait returned after the release (>= 25 ms)", dt >= 0.025);
    check("held work: result visible", out[0] == 7);
    check("held work: poll ran out and blocked (miss)", st2.spin_miss > st_hit.spin_miss);
    // Gated steps on a polling stream.
    for (int i = 0; i < 16; i++) {
        puf_mtl::gate_arm(s);
        copy16(s, out + 4 * i, env);
        uint64_t v = puf_mtl::gate_commit(s);
        cudaEventRecord(e, s);
        env[0] = (uint32_t)(200 + i);
        puf_mtl::gate_open(s);
        cudaEventSynchronize(e);
        if (!puf_mtl::gate_ran(s, v) || out[4 * i] != (uint32_t)(200 + i)) wrong++;
    }
    check("gated steps on a polling stream", wrong == 0);
    cudaEventDestroy(e);
}

template <typename F>
static bool fatals(F f) {
    try {
        f();
    } catch (const FatalError&) {
        return true;
    }
    return false;
}

static void case_misuse() {
    printf("misuse\n");
    puf_mtl::set_fatal_handler(throw_fatal);
    uint32_t* env = ualloc(4);
    uint32_t* out = ualloc(4);
    {
        cudaStream_t s = gated_stream();
        puf_mtl::gate_arm(s);
        check("double arm -> fatal", fatals([&] { puf_mtl::gate_arm(s); }));
    }
    {
        cudaStream_t s = gated_stream();
        puf_mtl::gate_arm(s);
        copy16(s, out, env);
        check("second dispatch in a gated buffer -> fatal",
            fatals([&] { copy16(s, out, env); }));
    }
    {
        cudaStream_t s = gated_stream();
        check("gate_open without a gated commit -> fatal",
            fatals([&] { puf_mtl::gate_open(s); }));
        check("gate_ran on an unknown commit -> fatal",
            fatals([&] { puf_mtl::gate_ran(s, 12345); }));
        check("gate_commit without a gated dispatch -> fatal",
            fatals([&] { puf_mtl::gate_commit(s); }));
    }
    {
        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        check("gate on a stream with progress words -> fatal",
            fatals([&] { puf_mtl::gate_arm(s); }));
    }
    puf_mtl::set_fatal_handler(nullptr);
}

int main(int argc, char** argv) {
    @autoreleasepool {
        std::vector<std::string> cases;
        for (int i = 1; i < argc; i++) cases.push_back(argv[i]);
        if (cases.empty()) cases = {"pipeline", "cancel", "strict", "spin", "early", "misuse"};
        for (const std::string& c : cases) {
            if (c == "pipeline") case_pipeline();
            else if (c == "early") case_early();
            else if (c == "cancel") case_cancel();
            else if (c == "strict") case_strict();
            else if (c == "misuse") case_misuse();
            else if (c == "spin") case_spin();
            else {
                printf("unknown case %s\n", c.c_str());
                return 2;
            }
        }
    }
    printf(g_fail ? "gate: %d FAILED\n" : "gate: ALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}

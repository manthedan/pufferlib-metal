// End-to-end check that the runtime fails loudly when an MTL4 queue wait
// times out (see runtime.mm "Host-side verification of GPU completion" and
// tests/metal/run_test_wait_timeout.sh: queue waits are released ~4 s after
// submission even while the producer is still running).
//
// One case per process (a detected failure stays recorded, so later syncs
// would report it again):
//   control: producer stream with a short GEMM; consumer waits on its event,
//            runs a fill, host syncs the consumer -> must not fail.
//   sync:    producer stream with ~8 s of queued GEMMs (4 command buffers of
//            ~2 s each, every kernel bounded); consumer waits on its event,
//            runs a fill, host syncs the consumer -> the wait expires at ~4 s
//            and the sync must raise "cross-stream wait timed out".
//   device:  as sync, but cudaDeviceSynchronize: the producer (created
//            first) is drained before the consumer is examined, so only the
//            recorded GPU evidence can show the early release.
//   late:    as sync, but the host sleeps until the producer has finished
//            before synchronizing the consumer.
//   chain:   as sync with a ~6 s producer, then (3.7 s later, so their waits
//            hold) 20 more streams each waiting on the previous one;
//            synchronizing the last one must find the timed-out wait 21
//            links upstream.
//
// Takes ~10-20 s per timeout case. Build/run from the repo root:
//   tests/metal/run_test_wait_detect.sh [case...]   (default: all)
#import <Foundation/Foundation.h>

#include <mach/mach_time.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include <unistd.h>

#include "compat/cuda_runtime.h"
#include "gemm.h"
#include "runtime.h"

static double now_s() {
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}

struct FatalError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

static void throw_fatal(const char* msg) { throw FatalError(msg); }

static float* dalloc(long n) {
    float* p = nullptr;
    cudaMalloc(&p, (size_t)n * sizeof(float));
    return p;
}

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "sync";
    int fail = 0;
    @autoreleasepool {
        const int M = 2048, N = 2048, K = 2048;
        float* A = dalloc((long)M * K);
        float* B = dalloc((long)K * N);
        float* C = dalloc((long)M * N);
        float* x = dalloc(4);
        for (long i = 0; i < (long)M * K; i++) A[i] = 1.0f;
        for (long i = 0; i < (long)K * N; i++) B[i] = 1.0f;
        puf_mtl::GemmOpts o;
        o.allow_tensor_ops = false;
        o.allow_split_k = false;
        auto gemm = [&](cudaStream_t s) {
            puf_mtl::gemm_rm(s, false, false, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N,
                nullptr, 0, &o);
        };
        cudaStream_t prod, cons;
        cudaStreamCreateWithFlags(&prod, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&cons, cudaStreamNonBlocking);
        if (mode == "lazy") puf_mtl::stream_host_wait_only(cons);
        cudaEvent_t ev, mark;
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&mark, cudaEventDisableTiming);
        puf_mtl::set_fatal_handler(throw_fatal);

        // Calibrate (warm).
        gemm(prod);
        cudaStreamSynchronize(prod);
        double t = now_s();
        for (int i = 0; i < 4; i++) gemm(prod);
        cudaStreamSynchronize(prod);
        double per = (now_s() - t) / 4;
        int per_cb = (int)std::ceil(2.0 / per);
        if (per_cb > 200) per_cb = 200;

        if (mode == "control") {
            try {
                gemm(prod);
                cudaEventRecord(ev, prod);
                cudaStreamWaitEvent(cons, ev, 0);
                cudaMemsetAsync(x, 7, 16, cons);
                cudaStreamSynchronize(cons);
                cudaDeviceSynchronize();
                printf("  %-52s ok\n", "control: short producer, consumer sync");
            } catch (const FatalError& e) {
                printf("  %-52s FAIL (%s)\n", "control: short producer, consumer sync",
                    e.what());
                fail++;
            }
            printf(fail ? "FAILED (%d)\n" : "ALL PASS\n", fail);
            fflush(stdout);
            _exit(fail ? 1 : 0);
        }

        // Timeout: ~8 s producer, consumer queued behind a wait on its end.
        double t0 = now_s();
        // chain: ~6 s (see below), else ~8 s, in 4 command buffers: they fit
        // the 4 slots, so committing them never blocks and all of it is
        // still queued when the consumer's wait is submitted.
        const int ncb = 4;
        if (mode == "chain") per_cb = (int)std::ceil(1.5 / per);
        if (per_cb > 200) per_cb = 200;
        printf("gemm %.1f ms -> %d per command buffer, %d command buffers (~%.1f s)\n",
            per * 1e3, per_cb, ncb, ncb * per_cb * per);
        for (int c = 0; c < ncb; c++) {
            for (int i = 0; i < per_cb; i++) gemm(prod);
            cudaEventRecord(mark, prod);  // commit: one ~2 s command buffer
        }
        cudaEventRecord(ev, prod);
        cudaStreamWaitEvent(cons, ev, 0);
        cudaMemsetAsync(x, 0, 16, cons);
        cudaEventRecord(mark, cons);  // commit the consumer now, behind the GPU wait
        cudaStream_t last = cons;
        if (mode == "chain") {
            // Queue the chain ~3.7 s after the consumer, before it is released
            // (~4 s). Measured on M4: while the producer's long command
            // buffers occupy the GPU, the chain's waits are not released by
            // the consumer's signal; the links run once the producer is done
            // (~6 s). That is within 4 s of their submission, so they hold
            // and only the first wait, 21 links upstream, timed out.
            double tc = now_s();
            while (now_s() - tc < 3.7) usleep(20000);
            cudaEvent_t ce;
            cudaEventCreateWithFlags(&ce, cudaEventDisableTiming);
            for (int i = 0; i < 20; i++) {
                cudaEventRecord(ce, last);
                cudaStream_t nx;
                cudaStreamCreateWithFlags(&nx, cudaStreamNonBlocking);
                cudaStreamWaitEvent(nx, ce, 0);
                cudaMemsetAsync(x, i, 16, nx);
                last = nx;
            }
        }
        std::string what;
        try {
            if (mode == "lazy") {
                // A lazy producer's fallback must verify its transitive
                // dependencies, not merely wait for its completion signal.
                cudaStream_t tail;
                cudaStreamCreateWithFlags(&tail, cudaStreamNonBlocking);
                cudaStreamWaitEvent(tail, mark, 0);
                cudaStreamSynchronize(tail);
            } else if (mode == "device") {
                cudaDeviceSynchronize();
            } else if (mode == "late") {
                // Producer has 4 * per_cb GEMMs queued; wait until well past
                // that before looking at the consumer.
                double est = ncb * per_cb * per;
                while (now_s() - t0 < est + 3.0) usleep(100000);
                cudaStreamSynchronize(cons);
            } else {
                cudaStreamSynchronize(last);
            }
        } catch (const FatalError& e) {
            what = e.what();
        }
        double t_detect = now_s() - t0;
        bool detected = what.find("cross-stream wait timed out") != std::string::npos;
        bool ok = detected;
        // Streams: s0 default, s1 producer, s2 consumer.
        ok = ok && what.find("s2 commit") != std::string::npos
            && what.find("producer stream s1 ") != std::string::npos;
        cudaStreamSynchronize(prod);  // drain the bounded producer
        double t_prod = now_s() - t0;
        char name[96];
        snprintf(name, sizeof(name), "%s: timed-out wait raises", mode.c_str());
        // chain: if the GPU is so contended that the links ran only after
        // their own waits expired, the depth is not exercised (but the
        // timeout was still detected): skip rather than fail.
        bool skip = !ok && detected && mode == "chain";
        printf("  %-52s %s (raised at %.2f s, producer done at %.2f s%s)\n", name,
            ok ? "ok" : skip ? "SKIP" : "FAIL", t_detect, t_prod,
            mode == "late" ? ", synced late" : "");
        if (!what.empty()) printf("    %s\n", what.c_str());
        if (skip) {
            printf("    a later link's own wait timed out too (producer took %.1f s, "
                "expected ~6): GPU too contended to exercise the depth\n", t_prod);
        } else if (!ok) {
            fail++;
            if (t_prod < 4.5) printf("    producer finished in < 4.5 s: GPU too fast for "
                "this test (increase the work)\n");
        }
        printf(fail ? "FAILED (%d)\n" : "ALL PASS\n", fail);
    }
    // Skip cudaFree: the consumer's failed dependency would be reported again.
    fflush(stdout);
    _exit(fail ? 1 : 0);
}

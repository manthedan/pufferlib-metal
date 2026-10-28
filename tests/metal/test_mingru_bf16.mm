// BF16 fused MinGRU train vs the generic BF16 GEMM + plain scans.
// Projection, recurrent state, scan gradients and dX must be bit-exact.
// dW changes FP32 reduction order: apply the BF16 policy gate (4 ulp or
// 2^-5 normalized max error), and require repeat/serial-reduce determinism.
// Run on a non-default stream and cover every supported T, resets at tile
// edges, multiple tiles, in-place dX, and rejected shapes/alignments.
#import <Foundation/Foundation.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "compat/cuda_bf16.h"
#include "compat/curand_kernel.h"
#include "launch.h"
#include "gemm.h"
#include "runtime.h"

#ifdef PRECISION_FLOAT
#error "This test requires the BF16 metallib"
#endif
using bf16 = __nv_bfloat16;
using precision_t = bf16;
#define PUF_MAX_DIMS 8
#define BLOCK_SIZE 256
struct Float { float* data; int64_t shape[PUF_MAX_DIMS]; };
struct Prec { bf16* data; int64_t shape[PUF_MAX_DIMS]; };
static int64_t numel(const int64_t* shape) {
    int64_t n = 1;
    for (int i = 0; i < PUF_MAX_DIMS && shape[i]; ++i) n *= shape[i];
    return n;
}
#define NUM_ATNS 1
struct PrefixScan {
    bf16 *combined_ptr = nullptr, *state_ptr = nullptr, *input_ptr = nullptr,
        *terminals_ptr = nullptr;
    int B = 0, T = 0, H = 32;
    Prec scan_h, out, next_state, grad_combined, grad_state, grad_input;
};
#include "kernels.h"

static int failures = 0;
static float widen(bf16 x) { return __bfloat162float(x); }
static void check(bool ok, const char* name) {
    printf("  %-40s %s\n", name, ok ? "ok" : "FAIL");
    failures += !ok;
}
template <typename T> static T* alloc(long n) {
    T* p = nullptr;
    if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess) exit(2);
    return p;
}
// NaN poison plus aligned guards detect missed writes and tile overruns.
struct Buffer {
    bf16 *base, *p;
    long n;
    Buffer(long size) : n(size) {
        base = alloc<bf16>(n + 32); p = base + 16;
        for (long i = 0; i < n + 32; ++i) base[i].x = 0x7fc1;
    }
    ~Buffer() { cudaFree(base); }
    // Re-poison between entrypoints so a skipped store can't reuse an earlier answer.
    void poison() { for (long i = 0; i < n; ++i) p[i].x = 0x7fc1; }
    bool guarded() const {
        for (int i = 0; i < 16; ++i)
            if (base[i].x != 0x7fc1 || p[n + i].x != 0x7fc1) return false;
        return true;
    }
};
struct Scan {
    Buffer sh, out, ns, gc, gs, gi;
    PrefixScan s;
    Scan(int B, int T, int H) : sh((long)B*T*H), out(sh.n), ns((long)B*H),
            gc(3*sh.n), gs(ns.n), gi(sh.n) {
        s.B = B; s.T = T; s.H = H;
        s.scan_h.data = sh.p; s.out.data = out.p; s.next_state.data = ns.p;
        s.grad_combined.data = gc.p; s.grad_state.data = gs.p; s.grad_input.data = gi.p;
    }
    void poison() { for (Buffer* b : {&sh, &out, &ns, &gc, &gs, &gi}) b->poison(); }
    bool guarded() const {
        return sh.guarded() && out.guarded() && ns.guarded() && gc.guarded()
            && gs.guarded() && gi.guarded();
    }
};
static void exact(const char* name, const bf16* got, const bf16* ref, long n) {
    long bad = 0;
    for (long i = 0; i < n; ++i) {
        if (got[i].x != ref[i].x || !std::isfinite(widen(got[i]))) {
            if (bad < 2) printf("    %s[%ld] got=%04x want=%04x\n", name, i, got[i].x, ref[i].x);
            ++bad;
        }
    }
    check(!bad, name);
}
static long ordered(uint16_t b) { return (b & 0x8000) ? -(long)(b & 0x7fff) : b; }
static void weight_error(const bf16* got, const bf16* ref, long n) {
    double mx = 0, mr = 0;
    long bad = 0, different = 0, max_ulp = 0;
    for (long i = 0; i < n; ++i) mr = fmax(mr, fabs(widen(ref[i])));
    for (long i = 0; i < n; ++i) {
        float a = widen(got[i]), b = widen(ref[i]);
        double d = fabs((double)a - b);
        long ulp = labs(ordered(got[i].x) - ordered(ref[i].x));
        // Identical per-element OR rule to test_bf16_golden's policy gate.
        bad += !std::isfinite(a) || !std::isfinite(b) || (ulp > 4 && d > mr/32);
        different += got[i].x != ref[i].x;
        mx = fmax(mx, d); max_ulp = std::max(max_ulp, ulp);
    }
    printf("    dW differing=%ld/%ld norm=%.3g max_ulp=%ld\n", different, n,
        mx / fmax(mr, 1e-30), max_ulp);
    check(!bad, "dW BF16 policy tolerance");
}
static void fill(Buffer& b, std::mt19937& rng, float scale) {
    std::uniform_real_distribution<float> d(-scale, scale);
    for (long i = 0; i < b.n; ++i) b.p[i] = __float2bfloat16(d(rng));
}
// H=32 fuses dW (FP32 partials, reordered sum). H=64 writes grad_combined
// for the generic dW GEMM and rejects grad_weight, so all of it is exact.
static void run_case(cudaStream_t stream, int B, int T, bool terminals, int repeats,
        int H = 32) {
    printf("B=%d T=%d H=%d terminals=%d repeats=%d\n", B, T, H, terminals, repeats);
    const int M = B*T;
    const long N = (long)M*H, S = (long)B*H;
    std::mt19937 rng(23 + B + T);
    Buffer x(N), W(3*H*H), state(S), term(M), go(N), gn(S), comb(3*N),
        gx(N), gx_ref(N), io(N), dw(3*H*H), dw_ref(3*H*H), dw2(3*H*H), fcomb(3*N);
    // H=64's forward also stores the projection, which its backward reads.
    bf16* fc = H == 64 ? fcomb.p : nullptr;
    for (Buffer* b : {&x, &state, &go, &gn}) fill(*b, rng, 2);
    fill(W, rng, 0.6f);
    for (int i = 0; i < M; ++i) term.p[i] = __float2bfloat16(
        (rng()%7 == 0 || i%T == 0 || i%T == T-1) ? 1.0f : 0.0f);
    bf16* tp = terminals ? term.p : nullptr;
    Scan r(B,T,H), f(B,T,H);
    void* ws = alloc<char>(4<<20);
    auto path = puf_mtl::gemm_rm(stream, false, true, M, 3*H, H, 1, x.p, H, W.p, H,
        0, comb.p, 3*H);
    check(path == puf_mtl::GEMM_TENSOR_OPS, "reference tensor_ops projection selected");
    r.s.combined_ptr = comb.p; r.s.input_ptr = x.p; r.s.state_ptr = state.p; r.s.terminals_ptr = tp;
    for (int it = 0; it < repeats; ++it) {
        cudaStreamSynchronize(stream);
        r.poison(); f.poison();
        for (Buffer* b : {&gx, &gx_ref, &io, &dw, &dw_ref, &dw2, &fcomb}) b->poison();
        if (it) comb.poison();
        if (it) puf_mtl::gemm_rm(stream, false, true, M, 3*H, H, 1, x.p, H, W.p, H,
            0, comb.p, 3*H);
        PUF_LAUNCH(mingru_scan_forward, (B*H+255)/256, 256, 0, stream)(r.s);
        bool ok = puf_mtl::mingru_fused_forward_rm(stream, B, T, H, x.p, W.p, state.p, tp,
            f.sh.p, f.out.p, f.ns.p, fc);
        check(ok, "fused forward accepted");
        if (!ok) break;
        PUF_LAUNCH(mingru_scan_backward, (B*H+255)/256, 256, 0, stream)(r.s, go.p, gn.p);
        check(puf_mtl::gemm_nn_add_rm(stream, M, H, 3*H, r.gc.p, W.p, r.gi.p, gx_ref.p),
            "reference dX selected");
        check(puf_mtl::mingru_fused_backward_rm(stream, B, T, H, x.p, W.p, tp, f.sh.p,
            go.p, gn.p, f.gc.p, f.gs.p, f.gi.p, gx.p, nullptr, fc), "fused backward accepted");
        cudaStreamSynchronize(stream);
        if (H == 64) exact("combined", fcomb.p, comb.p, 3*N);
        else check(fcomb.guarded() && fcomb.p[0].x == 0x7fc1, "H=32 leaves combined alone");
        exact("out", f.out.p, r.out.p, N); exact("scan_h", f.sh.p, r.sh.p, N);
        exact("next_state", f.ns.p, r.ns.p, S); exact("grad_combined", f.gc.p, r.gc.p, 3*N);
        exact("grad_input", f.gi.p, r.gi.p, N); exact("grad_state", f.gs.p, r.gs.p, S);
        exact("grad_x", gx.p, gx_ref.p, N);
        if (H == 64) {
            // In place (algo.cu below the top layer), then dW is rejected.
            io.poison(); f.gc.poison(); f.gs.poison(); f.gi.poison();
            cudaMemcpyAsync(io.p, go.p, N*sizeof(bf16), cudaMemcpyDeviceToDevice, stream);
            check(puf_mtl::mingru_fused_backward_rm(stream, B, T, H, x.p, W.p, tp, f.sh.p,
                io.p, gn.p, f.gc.p, f.gs.p, f.gi.p, io.p, nullptr, fc), "in-place dX accepted");
            cudaStreamSynchronize(stream);
            exact("in-place grad_x", io.p, gx_ref.p, N);
            exact("in-place grad_state", f.gs.p, r.gs.p, S);
            exact("in-place grad_input", f.gi.p, r.gi.p, N);
            exact("in-place grad_combined", f.gc.p, r.gc.p, 3*N);
            // grad_input == nullptr (algo.cu): addend on chip, device copy
            // untouched, every other output unchanged.
            io.poison(); f.gc.poison(); f.gs.poison(); f.gi.poison();
            cudaMemcpyAsync(io.p, go.p, N*sizeof(bf16), cudaMemcpyDeviceToDevice, stream);
            check(puf_mtl::mingru_fused_backward_rm(stream, B, T, H, x.p, W.p, tp, f.sh.p,
                io.p, gn.p, f.gc.p, f.gs.p, nullptr, io.p, nullptr, fc), "null grad_input accepted");
            cudaStreamSynchronize(stream);
            exact("null-gi grad_x", io.p, gx_ref.p, N);
            exact("null-gi grad_state", f.gs.p, r.gs.p, S);
            exact("null-gi grad_combined", f.gc.p, r.gc.p, 3*N);
            {
                bool untouched = f.gi.guarded();
                for (long k = 0; k < N; ++k) untouched &= f.gi.p[k].x == 0x7fc1;
                check(untouched, "null grad_input leaves the device copy alone");
            }
            auto before = puf_mtl::stats();
            check(!puf_mtl::mingru_fused_backward_rm(stream, B, T, H, x.p, W.p, tp, f.sh.p,
                go.p, gn.p, f.gc.p, f.gs.p, f.gi.p, gx.p, dw.p, fc), "H=64 fused dW rejected");
            check(!puf_mtl::mingru_fused_backward_rm(stream, B, T, H, x.p, W.p, tp, f.sh.p,
                go.p, gn.p, f.gc.p, f.gs.p, f.gi.p, gx.p), "H=64 null combined rejected");
            check(!puf_mtl::mingru_fused_backward_rm(stream, B, T, H, x.p, W.p, tp, f.sh.p,
                go.p, gn.p, f.gc.p, f.gs.p, f.gi.p, gx.p, nullptr, f.gc.p),
                "H=64 combined == grad_combined rejected");
            check(!puf_mtl::mingru_fused_forward_rm(stream, B, T, H, x.p, W.p, state.p, tp,
                f.sh.p, f.out.p, f.ns.p), "H=64 forward without combined rejected");
            check(before.dispatches == puf_mtl::stats().dispatches, "rejected dW encodes nothing");
            cudaMemcpyAsync(state.p, r.ns.p, S*sizeof(bf16), cudaMemcpyDeviceToDevice, stream);
            continue;
        }
        // The partials occupy all of BF16 grad_combined, interpreted as FP32.
        puf_mtl::gemm_rm(stream, true, false, 3*H, H, M, 1, r.gc.p, 3*H, x.p, H,
            0, dw_ref.p, H, ws, 4<<20);
        for (bool with_dw : {false, true}) {
            io.poison(); f.gc.poison(); f.gs.poison(); f.gi.poison();
            cudaMemcpyAsync(io.p, go.p, N*sizeof(bf16), cudaMemcpyDeviceToDevice, stream);
            check(puf_mtl::mingru_fused_backward_rm(stream, B, T, H, x.p, W.p, tp, f.sh.p,
                io.p, gn.p, f.gc.p, f.gs.p, f.gi.p, io.p, with_dw ? dw.p : nullptr),
                with_dw ? "fused dW + in-place dX accepted" : "in-place dX accepted");
            cudaStreamSynchronize(stream);
            exact("in-place grad_x", io.p, gx_ref.p, N);
            exact("in-place grad_state", f.gs.p, r.gs.p, S);
            exact("in-place grad_input", f.gi.p, r.gi.p, N);
            if (!with_dw) exact("in-place grad_combined", f.gc.p, r.gc.p, 3*N);
        }
        weight_error(dw.p, dw_ref.p, 3*H*H);
        dw2.poison(); gx.poison(); f.gc.poison(); f.gs.poison(); f.gi.poison();
        check(puf_mtl::mingru_fused_backward_rm(stream, B, T, H, x.p, W.p, tp, f.sh.p,
            go.p, gn.p, f.gc.p, f.gs.p, f.gi.p, gx.p, dw2.p), "repeat fused dW accepted");
        cudaStreamSynchronize(stream);
        exact("repeat dW", dw2.p, dw.p, 3*H*H);
        exact("repeat grad_x", gx.p, gx_ref.p, N);
        exact("repeat grad_state", f.gs.p, r.gs.p, S);
        exact("repeat grad_input", f.gi.p, r.gi.p, N);
        dw2.poison();
        // This API's BF16 overload rounds only after the FP32 tile sum.
        puf_mtl::mingru_dw_reduce_rm(stream, (const float*)f.gc.p, dw2.p, M/64, true);
        cudaStreamSynchronize(stream);
        exact("serial reduce dW", dw2.p, dw.p, 3*H*H);
        // Carry the state across horizons, with independent generic/fused inputs.
        cudaMemcpyAsync(state.p, r.ns.p, S*sizeof(bf16), cudaMemcpyDeviceToDevice, stream);
    }
    check(r.guarded() && f.guarded(), "scan allocation guards");
    bool guards = true;
    for (Buffer* b : {&x,&W,&state,&term,&go,&gn,&comb,&gx,&gx_ref,&io,&dw,&dw_ref,&dw2,&fcomb})
        guards &= b->guarded();
    check(guards, "input/output allocation guards");
    cudaFree(ws);
}
// --bench64: per-launch wall time (back-to-back dispatches / N) of the H=64
// fused kernels vs the generic sequence they replace, Pong's minibatch shape.
template <typename F> static double time_ms(cudaStream_t s, F&& launch, int iters = 200) {
    for (int i = 0; i < 10; i++) launch();
    cudaStreamSynchronize(s);
    NSDate* t0 = [NSDate date];
    for (int i = 0; i < iters; i++) launch();
    cudaStreamSynchronize(s);
    return -[t0 timeIntervalSinceNow] * 1e3 / iters;
}
static void bench64(cudaStream_t stream, int B, int T) {
    const int H = 64, M = B*T;
    const long N = (long)M*H, S = (long)B*H;
    std::mt19937 rng(5);
    Buffer x(N), W(3*H*H), state(S), term(M), go(N), gn(S), comb(3*N), gx(N), fcomb(3*N);
    for (Buffer* b : {&x, &state, &go, &gn}) fill(*b, rng, 2);
    fill(W, rng, 0.6f);
    for (int i = 0; i < M; ++i) term.p[i] = __float2bfloat16(rng()%50 == 0 ? 1.0f : 0.0f);
    Scan r(B,T,H), f(B,T,H);
    r.s.combined_ptr = comb.p; r.s.input_ptr = x.p; r.s.state_ptr = state.p; r.s.terminals_ptr = term.p;
    double g = time_ms(stream, [&] { puf_mtl::gemm_rm(stream, false, true, M, 3*H, H, 1, x.p, H,
        W.p, H, 0, comb.p, 3*H); });
    double sf = time_ms(stream, [&] {
        PUF_LAUNCH(mingru_scan_forward, (B*H+255)/256, 256, 0, stream)(r.s); });
    double sb = time_ms(stream, [&] {
        PUF_LAUNCH(mingru_scan_backward, (B*H+255)/256, 256, 0, stream)(r.s, go.p, gn.p); });
    double na = time_ms(stream, [&] { puf_mtl::gemm_nn_add_rm(stream, M, H, 3*H, r.gc.p, W.p,
        r.gi.p, gx.p); });
    double ff = time_ms(stream, [&] { puf_mtl::mingru_fused_forward_rm(stream, B, T, H, x.p, W.p,
        state.p, term.p, f.sh.p, f.out.p, f.ns.p, fcomb.p); });
    double fb = time_ms(stream, [&] { puf_mtl::mingru_fused_backward_rm(stream, B, T, H, x.p, W.p,
        term.p, f.sh.p, go.p, gn.p, f.gc.p, f.gs.p, f.gi.p, gx.p, nullptr, fcomb.p); });
    // algo.cu's call: grad_input stays on chip.
    double fbn = time_ms(stream, [&] { puf_mtl::mingru_fused_backward_rm(stream, B, T, H, x.p,
        W.p, term.p, f.sh.p, go.p, gn.p, f.gc.p, f.gs.p, nullptr, gx.p, nullptr, fcomb.p); });
    printf("B=%d T=%d H=64 (ms/launch): generic fwd %.4f (nt %.4f + scan %.4f) fused %.4f | "
        "generic bwd %.4f (scan %.4f + nn_add %.4f) fused %.4f (no grad_input %.4f)\n",
        B, T, g + sf, g, sf, ff, sb + na, sb, na, fb, fbn);
}
int main(int argc, char** argv) {
    @autoreleasepool {
        puf_mtl::init();
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        if (argc > 1 && strcmp(argv[1], "--bench64") == 0) {
            bench64(stream, 256, 64);
            bench64(stream, 256, 64);
            cudaStreamDestroy(stream);
            return 0;
        }
        // --no64: PUF_METAL_FUSED_MINGRU64=0 must reject every H=64 shape.
        const bool no64 = argc > 1 && strcmp(argv[1], "--no64") == 0;
        if (argc > 1 && strcmp(argv[1], "--disabled") == 0) {
            Buffer d(64*32*32*3);
            auto before = puf_mtl::stats();
            check(!puf_mtl::mingru_fused_eligible(64,32,32,d.p,d.p), "opt-out eligibility");
            check(!puf_mtl::mingru_fused_forward_rm(stream,64,32,32,d.p,d.p,d.p,nullptr,
                d.p,d.p,d.p), "opt-out forward");
            check(!puf_mtl::mingru_fused_backward_rm(stream,64,32,32,d.p,d.p,nullptr,
                d.p,d.p,d.p,d.p,d.p,d.p,d.p+16), "opt-out backward");
            check(!puf_mtl::mingru_fused_eligible(4,64,64,d.p,d.p), "opt-out H=64 eligibility");
            check(!puf_mtl::mingru_fused_forward_rm(stream,4,64,64,d.p,d.p,d.p,nullptr,
                d.p,d.p,d.p,d.p+64), "opt-out H=64 forward");
            check(!puf_mtl::mingru_fused_backward_rm(stream,4,64,64,d.p,d.p,nullptr,
                d.p,d.p,d.p,d.p,d.p,d.p,d.p+16,nullptr,d.p+64), "opt-out H=64 backward");
            check(before.dispatches == puf_mtl::stats().dispatches, "opt-out encodes nothing");
            cudaStreamDestroy(stream);
            return failures ? 1 : 0;
        }
        // Every divisor of 64; 1/3/9-tile cases cover reduction tails.
        for (int T : {1,2,4,8,16,32,64}) run_case(stream, 192/T, T, true, 1);
        run_case(stream, 2, 32, false, 4);
        run_case(stream, 18, 32, true, 2);
        run_case(stream, 2048, 32, false, 1);
        run_case(stream, 2048, 32, true, 1);
        if (!no64) {
            // H=64: 64-row tiles in two 32-row chunks (whole sequences for
            // T <= 32, one sequence carried across both for T = 64);
            // Pong / Flappy minibatch B=256 T=64.
            for (int T : {1,2,4,8,16,32,64}) run_case(stream, 192/T, T, true, 1, 64);
            run_case(stream, 2, 64, false, 4, 64);
            run_case(stream, 9, 64, true, 2, 64);
            run_case(stream, 256, 64, false, 1, 64);
            run_case(stream, 256, 64, true, 2, 64);
        }
        Buffer dummy(64*32*32*3);
        long wrong = 0;
        for (int T : {0,3,48,128}) wrong += puf_mtl::mingru_fused_eligible(64,T,32,dummy.p,dummy.p);
        wrong += puf_mtl::mingru_fused_eligible(3,32,32,dummy.p,dummy.p);
        wrong += puf_mtl::mingru_fused_eligible(64,32,48,dummy.p,dummy.p);
        wrong += puf_mtl::mingru_fused_eligible(64,32,512,dummy.p,dummy.p);
        for (int T : {0,3,48,128}) wrong += puf_mtl::mingru_fused_eligible(64,T,64,dummy.p,dummy.p);
        wrong += puf_mtl::mingru_fused_eligible(3,32,64,dummy.p,dummy.p);
        wrong += puf_mtl::mingru_fused_eligible(1,32,64,dummy.p,dummy.p);
        wrong += puf_mtl::mingru_fused_eligible(64,64,64,dummy.p+1,dummy.p);
        wrong += puf_mtl::mingru_fused_eligible(64,64,64,dummy.p,dummy.p+1);
        wrong += puf_mtl::mingru_fused_eligible(0x7fffffff/64,64,64,dummy.p,dummy.p);
        wrong += no64 ? puf_mtl::mingru_fused_eligible(64,64,64,dummy.p,dummy.p) : 0;
        if (!no64) check(puf_mtl::mingru_fused_eligible(1,64,64,dummy.p,dummy.p)
            && puf_mtl::mingru_fused_eligible(2,32,64,dummy.p,dummy.p), "H=64 single tile eligible");
        wrong += puf_mtl::mingru_fused_eligible(0,32,32,dummy.p,dummy.p);
        wrong += puf_mtl::mingru_fused_eligible(64,32,32,dummy.p+1,dummy.p);
        wrong += puf_mtl::mingru_fused_eligible(64,32,32,dummy.p,dummy.p+1);
        wrong += puf_mtl::mingru_fused_eligible(0x7fffffff,32,32,dummy.p,dummy.p);
        check(!wrong, "invalid shapes/alignment rejected");
        check(!puf_mtl::mingru_fused_backward_rm(stream, 64,32,32,dummy.p,dummy.p,nullptr,
            dummy.p,dummy.p,dummy.p,dummy.p,dummy.p,dummy.p,dummy.p), "grad_x == grad_input rejected");
        auto before = puf_mtl::stats();
        check(!puf_mtl::mingru_fused_backward_rm(stream,64,32,32,dummy.p,dummy.p,nullptr,
            dummy.p,dummy.p,dummy.p,dummy.p+1,dummy.p,dummy.p+32,dummy.p+16,dummy.p+64),
            "unaligned FP32 dW scratch rejected");
        check(before.dispatches == puf_mtl::stats().dispatches, "bad scratch encodes nothing");
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        printf(failures ? "FAILURES: %d\n" : "ALL PASS\n", failures);
        return failures ? 1 : 0;
    }
}

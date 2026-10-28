// Wide fused MinGRU vs independent generic GEMM + plain scans, both precisions.
// Projection, recurrent state, scan gradients and dX must be bit-exact.
// dW remains the original generic GEMM; no sum or storage precision changes.
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
using Scalar = float;
#else
using Scalar = __nv_bfloat16;
#endif
static Scalar narrow(float x) {
#ifdef PRECISION_FLOAT
    return x;
#else
    return __float2bfloat16(x);
#endif
}
static uint32_t bits(Scalar v) {
    uint32_t b=0; memcpy(&b,&v,sizeof(v)); return b;
}
static Scalar poison_value() { return narrow(NAN); }

using precision_t = Scalar;
#define PUF_MAX_DIMS 8
#define BLOCK_SIZE 256
struct Float { float* data; int64_t shape[PUF_MAX_DIMS]; };
struct Prec { Scalar* data; int64_t shape[PUF_MAX_DIMS]; };
static int64_t numel(const int64_t* shape) {
    int64_t n = 1;
    for (int i = 0; i < PUF_MAX_DIMS && shape[i]; ++i) n *= shape[i];
    return n;
}
#define NUM_ATNS 1
struct PrefixScan {
    Scalar *combined_ptr = nullptr, *state_ptr = nullptr, *input_ptr = nullptr,
        *terminals_ptr = nullptr;
    int B = 0, T = 0, H = 32;
    Prec scan_h, out, next_state, grad_combined, grad_state, grad_input;
};
#include "kernels.h"

static int failures = 0;
static float widen(Scalar x) {
#ifdef PRECISION_FLOAT
    return x;
#else
    return __bfloat162float(x);
#endif
}
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
    Scalar *base, *p;
    long n;
    Buffer(long size) : n(size) {
        base = alloc<Scalar>(n + 32); p = base + 16;
        for (long i = 0; i < n + 32; ++i) base[i] = poison_value();
    }
    ~Buffer() { cudaFree(base); }
    // Re-poison between entrypoints so a skipped store can't reuse an earlier answer.
    void poison() { for (long i = 0; i < n; ++i) p[i] = poison_value(); }
    bool guarded() const {
        for (int i = 0; i < 16; ++i)
            if (bits(base[i]) != bits(poison_value()) || bits(p[n + i]) != bits(poison_value())) return false;
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
static void exact(const char* name, const Scalar* got, const Scalar* ref, long n) {
    long bad = 0;
    for (long i = 0; i < n; ++i) {
        if (bits(got[i]) != bits(ref[i]) || !std::isfinite(widen(got[i]))) {
            if (bad < 2) printf("    %s[%ld] got=%04x want=%04x\n", name, i, bits(got[i]), bits(ref[i]));
            ++bad;
        }
    }
    check(!bad, name);
}
static void fill(Buffer& b, std::mt19937& rng, float scale) {
    std::uniform_real_distribution<float> d(-scale, scale);
    for (long i = 0; i < b.n; ++i) b.p[i] = narrow(d(rng));
}
// Wide backward writes grad_combined for the unchanged generic dW GEMM.
static void run_case(cudaStream_t stream, int B, int T, bool terminals, int repeats,
        int H) {
    printf("B=%d T=%d H=%d terminals=%d repeats=%d\n", B, T, H, terminals, repeats);
    const int M = B*T;
    const long N = (long)M*H, S = (long)B*H;
    std::mt19937 rng(23 + B + T);
    Buffer x(N), W(3*H*H), state(S), term(M), go(N), gn(S), comb(3*N),
        gx(N), gx_ref(N), io(N), dw(3*H*H), fcomb(3*N);
    // The wide forward also stores the projection, which its backward reads.
    Scalar* fc = fcomb.p;
    for (Buffer* b : {&x, &state, &go, &gn}) fill(*b, rng, 2);
    fill(W, rng, 0.6f);
    for (int i = 0; i < M; ++i) term.p[i] = narrow(
        (rng()%7 == 0 || i%T == 0 || i%T == T-1) ? 1.0f : 0.0f);
    Scalar* tp = terminals ? term.p : nullptr;
    Scan r(B,T,H), f(B,T,H);
    auto path = puf_mtl::gemm_rm(stream, false, true, M, 3*H, H, 1, x.p, H, W.p, H,
        0, comb.p, 3*H);
    check(path == puf_mtl::GEMM_TENSOR_OPS, "reference tensor_ops projection selected");
    r.s.combined_ptr = comb.p; r.s.input_ptr = x.p; r.s.state_ptr = state.p; r.s.terminals_ptr = tp;
    for (int it = 0; it < repeats; ++it) {
        cudaStreamSynchronize(stream);
        r.poison(); f.poison();
        for (Buffer* b : {&gx, &gx_ref, &io, &dw, &fcomb}) b->poison();
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
        exact("combined", fcomb.p, comb.p, 3*N);
        exact("out", f.out.p, r.out.p, N); exact("scan_h", f.sh.p, r.sh.p, N);
        exact("next_state", f.ns.p, r.ns.p, S); exact("grad_combined", f.gc.p, r.gc.p, 3*N);
        exact("grad_input", f.gi.p, r.gi.p, N); exact("grad_state", f.gs.p, r.gs.p, S);
        exact("grad_x", gx.p, gx_ref.p, N);
        {
            // In place (algo.cu below the top layer), then dW is rejected.
            io.poison(); f.gc.poison(); f.gs.poison(); f.gi.poison();
            cudaMemcpyAsync(io.p, go.p, N*sizeof(Scalar), cudaMemcpyDeviceToDevice, stream);
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
            cudaMemcpyAsync(io.p, go.p, N*sizeof(Scalar), cudaMemcpyDeviceToDevice, stream);
            check(puf_mtl::mingru_fused_backward_rm(stream, B, T, H, x.p, W.p, tp, f.sh.p,
                io.p, gn.p, f.gc.p, f.gs.p, nullptr, io.p, nullptr, fc), "null grad_input accepted");
            cudaStreamSynchronize(stream);
            exact("null-gi grad_x", io.p, gx_ref.p, N);
            exact("null-gi grad_state", f.gs.p, r.gs.p, S);
            exact("null-gi grad_combined", f.gc.p, r.gc.p, 3*N);
            {
                bool untouched = f.gi.guarded();
                for (long k = 0; k < N; ++k) untouched &= bits(f.gi.p[k]) == bits(poison_value());
                check(untouched, "null grad_input leaves the device copy alone");
            }
            auto before = puf_mtl::stats();
            check(!puf_mtl::mingru_fused_backward_rm(stream, B, T, H, x.p, W.p, tp, f.sh.p,
                go.p, gn.p, f.gc.p, f.gs.p, f.gi.p, gx.p, dw.p, fc), "wide fused dW rejected");
            check(!puf_mtl::mingru_fused_backward_rm(stream, B, T, H, x.p, W.p, tp, f.sh.p,
                go.p, gn.p, f.gc.p, f.gs.p, f.gi.p, gx.p), "wide null combined rejected");
            check(!puf_mtl::mingru_fused_backward_rm(stream, B, T, H, x.p, W.p, tp, f.sh.p,
                go.p, gn.p, f.gc.p, f.gs.p, f.gi.p, gx.p, nullptr, f.gc.p),
                "wide combined == grad_combined rejected");
            check(!puf_mtl::mingru_fused_forward_rm(stream, B, T, H, x.p, W.p, state.p, tp,
                f.sh.p, f.out.p, f.ns.p), "wide forward without combined rejected");
            check(before.dispatches == puf_mtl::stats().dispatches, "rejected dW encodes nothing");
            cudaMemcpyAsync(state.p, r.ns.p, S*sizeof(Scalar), cudaMemcpyDeviceToDevice, stream);
            continue;
        }
    }
    check(r.guarded() && f.guarded(), "scan allocation guards");
    bool guards = true;
    for (Buffer* b : {&x,&W,&state,&term,&go,&gn,&comb,&gx,&gx_ref,&io,&dw,&fcomb})
        guards &= b->guarded();
    check(guards, "input/output allocation guards");
}
template <typename F> static double time_ms(cudaStream_t s, F&& launch, int iters = 30) {
    for (int i = 0; i < 10; i++) launch();
    cudaStreamSynchronize(s);
    NSDate* t0 = [NSDate date];
    for (int i = 0; i < iters; i++) launch();
    cudaStreamSynchronize(s);
    return -[t0 timeIntervalSinceNow] * 1e3 / iters;
}
static void bench_wide(cudaStream_t stream, int B, int T, int H) {
    const int M = B*T;
    const long N = (long)M*H, S = (long)B*H;
    std::mt19937 rng(5);
    Buffer x(N), W(3*H*H), state(S), term(M), go(N), gn(S), comb(3*N), gx(N), fcomb(3*N);
    for (Buffer* b : {&x, &state, &go, &gn}) fill(*b, rng, 2);
    fill(W, rng, 0.6f);
    for (int i = 0; i < M; ++i) term.p[i] = narrow(rng()%50 == 0 ? 1.0f : 0.0f);
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
    printf("B=%d T=%d H=%d (ms/launch): generic fwd %.4f (nt %.4f + scan %.4f) fused %.4f | "
        "generic bwd %.4f (scan %.4f + nn_add %.4f) fused %.4f (no grad_input %.4f)\n",
        B, T, H, g + sf, g, sf, ff, sb + na, sb, na, fb, fbn);
}

int main(int argc,char**argv) {
    if (argc > 1 && !strcmp(argv[1], "--disabled")) {
        Buffer dummy(512);
        for (int H : {128, 256})
            check(!puf_mtl::mingru_fused_eligible(64, 64, H, dummy.p, dummy.p), "wide opt-out");
        return failures ? 1 : 0;
    }
    cudaStream_t stream; cudaStreamCreate(&stream);
    if(argc>1 && !strcmp(argv[1],"--bench")) {
        for(int H:{128,256}) if(argc<3 || H==atoi(argv[2])) bench_wide(stream,H==128?256:128,64,H);
    } else {
        for(int H:{128,256}) {
            if(argc>2 && H!=atoi(argv[2])) continue;
            for(int T:{1,2,4,8,16,32,64}) run_case(stream,192/T,T,true,1,H);
            run_case(stream,9,64,true,2,H);
            run_case(stream,2,64,false,3,H);
            run_case(stream,H==128?256:128,64,true,1,H);
            Buffer d(256);
            for(int T:{0,3,48,128}) check(!puf_mtl::mingru_fused_eligible(64,T,H,d.p,d.p),"reject bad T");
            check(!puf_mtl::mingru_fused_eligible(3,32,H,d.p,d.p),"reject incomplete tile");
            check(!puf_mtl::mingru_fused_eligible(64,64,H,d.p+1,d.p),"reject unaligned input");
            check(!puf_mtl::mingru_fused_eligible(0x7fffffff/64,64,H,d.p,d.p),"reject index overflow");
        }
    }
    cudaStreamSynchronize(stream); cudaStreamDestroy(stream);
    printf("%s (%d failures)\n",failures?"FAIL":"ALL PASS",failures); return failures?1:0;
}

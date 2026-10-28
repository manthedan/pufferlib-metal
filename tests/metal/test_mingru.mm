// MinGRU train-scan / policy-backward test for the Metal backend: the real
// PUF_LAUNCH adapters (src/metal/kernels_mingru.h) dispatching the MSL ports in
// src/metal/shaders/mingru.metal.
//
// Golden (CUDA FP32 dumps, see ../golden/REPORT.md):
//   scan_{tiny,breakout,plain_partial,plain_long}  mingru_scan_forward + _backward
//   scan_row_{tiny,breakout,long}                  mingru_scan_forward_row + _backward_row
//       out, next_state, scan_h, grad_combined, grad_input, grad_state
//   scan_* (cross)  the other variant on the same inputs, vs the same reference
//       (CUDA's own plain-vs-row difference is <= 4.77e-7)
//   policy_{tiny,breakout}, policy_ppo_{tiny,breakout}
//       algo.cu arch_forward_train (minus cache_imp_and_v) + arch_backward,
//       replayed host-side with the same buffers, launch configs and
//       cublasGemmEx mapping as algo.cu: encoder / MinGRU layers / decoder
//       forward, assemble_decoder_grad, decoder dW/dX, per-layer scan backward,
//       dW, dX, add_kernel, encoder dW. Compared: decoder output, per-layer
//       out / next_state / grad_state, every parameter gradient, encoder grad.
//       policy_ppo_* feeds the CUDA-emitted dlogits/dvalue (PPO kernels are
//       ported separately), so its backward half is checked the same way.
//       importance / live_values / cached_logps / new_logprobs come from
//       cache_imp_and_v (PPO port) and are not checked here.
// Synthetic (host reference): sum_rows_to_precision_kernel (exact sequential
// sum), assemble_decoder_grad, terminals == NULL vs all-zero terminals.
//
// Errors: max_abs, max_rel = max |x-ref| / max(|x|,|ref|) (golden.py
// definition), norm = max_abs / max|ref|. Pass: norm <= tol.
// L1/H1024 additionally validates a derivative-kink branch correction against
// independent float64 math; raw CUDA errors are still printed. See GOLDWIDE.md.
//
// Build/run: tests/metal/run_test_mingru.sh [golden_run_dir ...] [--bench]
#import <Foundation/Foundation.h>
#include "policy_kink_reference.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "compat/cublas_v2.h"
#include "compat/cuda_runtime.h"
#include "compat/curand_kernel.h"
#include "launch.h"
#include "gemm.h"
#include "runtime.h"

// Minimal stand-ins for the upstream types kernels.h adapts (src/pufferl.cu,
// src/algo.cu).
typedef float precision_t;
#define PUF_MAX_DIMS 8
#define BLOCK_SIZE 256
typedef struct { float* data; int64_t shape[PUF_MAX_DIMS]; } Float;
typedef struct { precision_t* data; int64_t shape[PUF_MAX_DIMS]; } Prec;
static int64_t numel(const int64_t* shape) {
    int64_t n = 1;
    for (int i = 0; i < PUF_MAX_DIMS && shape[i] != 0; i++) n *= shape[i];
    return n;
}
static int g_num_atns = 1;
#define NUM_ATNS g_num_atns

// Same field names as algo.cu's PrefixScan.
struct PrefixScan {
    precision_t* combined_ptr = NULL;
    precision_t* state_ptr = NULL;
    precision_t* input_ptr = NULL;
    precision_t* terminals_ptr = NULL;
    int B = 0, T = 0, H = 0;
    Prec scan_h;
    Prec out, next_state;
    Prec grad_combined, grad_state;
    Prec grad_input;
};
#define MINGRU_TILE 32

#include "kernels.h"

static int g_fail = 0;

template <typename T>
static T* dalloc(long n) {
    T* p = nullptr;
    if (cudaMalloc(&p, (size_t)(n > 0 ? n : 1) * sizeof(T)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed\n");
        exit(2);
    }
    return p;
}

static int grid_size(int n) { return (n + BLOCK_SIZE - 1) / BLOCK_SIZE; }

struct Err {
    double max_abs = 0, max_rel = 0, max_ref = 0;
    long mismatches = 0, n = 0;
    double norm() const { return max_ref > 0 ? max_abs / max_ref : max_abs; }
};

static Err compare_f32(const float* x, const float* ref, long n) {
    Err e;
    e.n = n;
    for (long i = 0; i < n; i++) {
        if (memcmp(&x[i], &ref[i], 4) != 0) e.mismatches++;
        double a = x[i], b = ref[i];
        if (std::isnan(a) && std::isnan(b)) continue;
        double d = fabs(a - b);
        if (!std::isfinite(d)) d = INFINITY;
        double den = fmax(fmax(fabs(a), fabs(b)), 2.2250738585072014e-308);
        e.max_abs = fmax(e.max_abs, d);
        e.max_rel = fmax(e.max_rel, d / den);
        e.max_ref = fmax(e.max_ref, fabs(b));
    }
    return e;
}

static void report(const char* cs, const char* what, const Err& e, double tol_norm) {
    bool ok = e.norm() <= tol_norm;
    if (!ok) g_fail++;
    printf("  %-20s %-20s n=%-8ld bit-exact=%-8ld max_abs=%.3e max_rel=%.3e norm=%.2e%s\n",
        cs, what, e.n, e.n - e.mismatches, e.max_abs, e.max_rel, e.norm(), ok ? "" : "  FAIL");
}

// ---------------------------------------------------------------------------
// Golden loading
// ---------------------------------------------------------------------------

struct Case {
    NSString* dir;
    NSDictionary* tensors;
    NSDictionary* scalars;
};

static Case load_case(NSString* dir) {
    NSData* mj = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"manifest.json"]];
    NSDictionary* man = mj ? [NSJSONSerialization JSONObjectWithData:mj options:0 error:nil] : nil;
    if (!man) {
        fprintf(stderr, "bad manifest in %s\n", dir.UTF8String);
        exit(2);
    }
    return {dir, man[@"tensors"], man[@"scalars"]};
}

static bool has(const Case& c, NSString* name) { return c.tensors[name] != nil; }

template <typename T>
static std::vector<T> read_vec(const Case& c, NSString* name) {
    NSDictionary* spec = c.tensors[name];
    if (!spec) {
        fprintf(stderr, "%s: no tensor %s\n", c.dir.UTF8String, name.UTF8String);
        exit(2);
    }
    NSData* d = [NSData dataWithContentsOfFile:[c.dir stringByAppendingPathComponent:spec[@"file"]]];
    if (!d) {
        fprintf(stderr, "cannot read %s/%s\n", c.dir.UTF8String, name.UTF8String);
        exit(2);
    }
    std::vector<T> v(d.length / sizeof(T));
    memcpy(v.data(), d.bytes, d.length);
    return v;
}

template <typename T>
static T* upload(const std::vector<T>& v) {
    T* p = dalloc<T>((long)v.size());
    memcpy(p, v.data(), v.size() * sizeof(T));
    return p;
}

static void check(const Case& c, const char* cs, NSString* name, const float* x, long n,
        double tol) {
    std::vector<float> ref = read_vec<float>(c, name);
    if ((long)ref.size() != n) {
        fprintf(stderr, "%s/%s: size %zu != %ld\n", cs, name.UTF8String, ref.size(), n);
        exit(2);
    }
    report(cs, name.UTF8String, compare_f32(x, ref.data(), n), tol);
}

// ---------------------------------------------------------------------------
// Scan launches (algo.cu mingru_forward_train / mingru_backward)
// ---------------------------------------------------------------------------

static bool use_row(const PrefixScan& s) {
    return (s.T >= 256 && s.B * s.H <= 32768) || (s.T >= 128 && s.H <= 32);
}

static void scan_forward(PrefixScan& scan, bool row, cudaStream_t stream) {
    if (row) {
        dim3 block(MINGRU_TILE, MINGRU_TILE);
        dim3 grid(scan.B, (scan.H + MINGRU_TILE - 1) / MINGRU_TILE);
        PUF_LAUNCH(mingru_scan_forward_row, grid, block, 0, stream)(scan);
    } else {
        PUF_LAUNCH(mingru_scan_forward, grid_size(scan.B * scan.H), BLOCK_SIZE, 0, stream)(scan);
    }
}

static void scan_backward(PrefixScan& scan, bool row, precision_t* grad,
        precision_t* grad_next, cudaStream_t stream) {
    if (row) {
        dim3 block(MINGRU_TILE, MINGRU_TILE);
        dim3 grid(scan.B, (scan.H + MINGRU_TILE - 1) / MINGRU_TILE);
        PUF_LAUNCH(mingru_scan_backward_row, grid, block, 0, stream)(scan, grad, grad_next);
    } else {
        PUF_LAUNCH(mingru_scan_backward, grid_size(scan.B * scan.H), BLOCK_SIZE, 0,
            stream)(scan, grad, grad_next);
    }
}

static PrefixScan alloc_scan(int B, int T, int H) {
    PrefixScan s;
    s.B = B; s.T = T; s.H = H;
    s.scan_h = {dalloc<float>((long)B * T * H), {B, T, H}};
    s.out = {dalloc<float>((long)B * T * H), {B, T, H}};
    s.next_state = {dalloc<float>((long)B * H), {B, 1, H}};
    s.grad_combined = {dalloc<float>((long)B * T * 3 * H), {B, T, 3 * H}};
    s.grad_state = {dalloc<float>((long)B * H), {B, 1, H}};
    s.grad_input = {dalloc<float>((long)B * T * H), {B, T, H}};
    return s;
}

static void free_scan(PrefixScan& s) {
    cudaFree(s.scan_h.data); cudaFree(s.out.data); cudaFree(s.next_state.data);
    cudaFree(s.grad_combined.data); cudaFree(s.grad_state.data); cudaFree(s.grad_input.data);
}

static void poison(float* p, long n) {
    for (long i = 0; i < n; i++) p[i] = NAN;
}

static const double SCAN_TOL = 1e-5;

static void run_scan(const Case& c, const char* cs, bool row, const char* tag) {
    int B = [c.scalars[@"B"] intValue], T = [c.scalars[@"T"] intValue];
    int H = [c.scalars[@"H"] intValue];
    float* comb = upload(read_vec<float>(c, @"combined"));
    float* st = upload(read_vec<float>(c, @"state_in"));
    float* x = upload(read_vec<float>(c, @"x"));
    float* term = upload(read_vec<float>(c, @"terminals"));
    float* gout = upload(read_vec<float>(c, @"grad_out"));
    float* gnext = upload(read_vec<float>(c, @"grad_next_state"));
    PrefixScan s = alloc_scan(B, T, H);
    long BTH = (long)B * T * H;
    poison(s.scan_h.data, BTH); poison(s.out.data, BTH); poison(s.next_state.data, (long)B * H);
    poison(s.grad_combined.data, 3 * BTH); poison(s.grad_state.data, (long)B * H);
    poison(s.grad_input.data, BTH);
    s.combined_ptr = comb; s.state_ptr = st; s.input_ptr = x; s.terminals_ptr = term;
    scan_forward(s, row, 0);
    scan_backward(s, row, gout, gnext, 0);
    cudaDeviceSynchronize();
    std::string name = std::string(cs) + tag;
    const char* n = name.c_str();
    check(c, n, @"out", s.out.data, BTH, SCAN_TOL);
    check(c, n, @"next_state", s.next_state.data, (long)B * H, SCAN_TOL);
    check(c, n, @"scan_h", s.scan_h.data, BTH, SCAN_TOL);
    check(c, n, @"grad_combined", s.grad_combined.data, 3 * BTH, SCAN_TOL);
    check(c, n, @"grad_input", s.grad_input.data, BTH, SCAN_TOL);
    check(c, n, @"grad_state", s.grad_state.data, (long)B * H, SCAN_TOL);
    free_scan(s);
    cudaFree(comb); cudaFree(st); cudaFree(x); cudaFree(term); cudaFree(gout); cudaFree(gnext);
}

// ---------------------------------------------------------------------------
// Policy replay (algo.cu arch_forward_train + arch_backward)
// ---------------------------------------------------------------------------

static cublasHandle_t g_cublas;

// algo.cu cublasGemmExDense: C = alpha * op_a(A) @ op_b(B) + beta * C, row-major.
static void gemm_dense(cublasOperation_t op_a, cublasOperation_t op_b, int M, int N, int K,
        const float* A, const float* B, float* C) {
    float alpha = 1.0f, beta = 0.0f;
    int lda = (op_a == CUBLAS_OP_N) ? K : M;
    int ldb = (op_b == CUBLAS_OP_N) ? N : K;
    cublasGemmEx(g_cublas, op_b, op_a, N, M, K, &alpha, B, CUDA_R_32F, ldb, A, CUDA_R_32F,
        lda, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}
// out(M,N) = a(M,K) @ b(N,K)^T
static void mm_nt(const float* a, const float* b, float* out, int M, int N, int K) {
    gemm_dense(CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, a, b, out);
}
// out(M,N) = a(K,M)^T @ b(K,N)
static void mm_tn(const float* a, const float* b, float* out, int M, int N, int K) {
    gemm_dense(CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, a, b, out);
}
// out(M,N) = a(M,K) @ b(K,N)
static void mm_nn(const float* a, const float* b, float* out, int M, int N, int K) {
    gemm_dense(CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, a, b, out);
}

static const double POLICY_TOL = 1e-4;

// CUDA policy fixtures do not dump combined/grad_combined. Validate against
// float64 from their inputs before allowing a branch-aware comparison. This
// is deliberately limited to the diagnosed one-layer H1024 FP32 fixture.
static bool policy_kink_reference(const Case& c, const char* cs, int B, int T, int H, int I,
        int O, const float* obs, const float* we, const float* wd, const float* w,
        const float* state, const float* term, const float* dl, const float* dv,
        const float* gn, const float* x, const float* comb, const PrefixScan& scan,
        const float* g, const float* dw, const float* dx, const float* ew,
        std::vector<float>& adjusted_dw, std::vector<float>& adjusted_dx,
        std::vector<float>& adjusted_ew) {
    using namespace policy_ref;
    const int R=B*T;
    D dg((size_t)R*O);
    for (int j=0;j<R;j++) {
        for (int k=0;k<O-1;k++) dg[j*O+k]=dl[j*(O-1)+k];
        dg[j*O+O-1]=dv[j];
    }
    D ob=widen(obs,(size_t)R*I), wt=widen(w,(size_t)3*H*H);
    Reference r=one_layer(B,T,H,I,O,ob,widen(we,(size_t)H*I),widen(wd,(size_t)O*H),
        wt,widen(state,(size_t)B*H),widen(term,R),dg,widen(gn,(size_t)B*H));
    D delta(r.gc.size());
    long flips=0;
    for (int j=0;j<R;j++) for (int h=0;h<H;h++) {
        size_t k=(size_t)j*3*H+h;
        if ((r.combined[k]>=0)==(comb[k]>=0)) continue;
        // Standard dot-product roundoff bound, including propagated encoder
        // error. No sign change outside this bound is excused.
        double products=0, propagated=0;
        for (int n=0;n<H;n++) {
            products+=fabs((double)x[j*H+n]*w[h*H+n]);
            propagated+=fabs((x[j*H+n]-r.x[j*H+n])*w[h*H+n]);
        }
        double u=0x1p-24, gamma=H*u/(1-H*u), bound=gamma*products+propagated;
        if (!std::isfinite(comb[k]) || fabs(comb[k]-r.combined[k])>bound) {
            fprintf(stderr,"%s: hidden sign change exceeds FP32 dot-product bound\n",cs);
            g_fail++; return false;
        }
        delta[k]=branch_delta(r.combined[k],comb[k],r.gc[k]);
        printf("  %s derivative kink b=%d t=%d h=%d f64=%+.12e Metal=%+.12e delta_dc=%+.12e\n",
            cs,j/T,j%T,h,r.combined[k],comb[k],delta[k]);
        flips++;
    }
    if (!flips) return false;
    bool valid=true;
    auto verify=[&](const char* label,const float* a,const D& b) {
        std::vector<float> ref(b.begin(),b.end());
        Err e=compare_f32(a,ref.data(),(long)b.size());
        // Tighter than the existing policy bound (1e-4), for every element.
        report(cs,label,e,1e-5);
        valid &= e.norm()<=1e-5;
    };
    D alt_gc=r.gc;
    for (size_t k=0;k<delta.size();k++) alt_gc[k]+=delta[k];
    verify("f64 encoder",x,r.x); verify("f64 projection",comb,r.combined);
    verify("f64 scan cache",scan.scan_h.data,r.cache); verify("f64 scan output",scan.out.data,r.out);
    verify("f64 decoder dX",g,r.grad); verify("f64 scan dC (branch)",scan.grad_combined.data,alt_gc);
    verify("f64 scan direct dX",scan.grad_input.data,r.gi); verify("f64 scan dState",scan.grad_state.data,r.gs);
    // Separately validate the GEMMs on their actual Metal operands.
    D mx=widen(x,(size_t)R*H), mgc=widen(scan.grad_combined.data,(size_t)R*3*H);
    verify("f64 same-input dW",dw,mm(mgc,mx,3*H,H,R,true));
    D mdx=mm(mgc,wt,R,H,3*H);
    for(size_t j=0;j<mdx.size();j++) mdx[j]+=scan.grad_input.data[j];
    verify("f64 same-input dX",dx,mdx);
    verify("f64 same-input encdW",ew,mm(widen(dx,(size_t)R*H),ob,H,I,R,true));
    // The CUDA golden must itself agree with the canonical float64 branch.
    auto cdw=read_vec<float>(c,@"param_2_grad"), cdx=read_vec<float>(c,@"encoder_grad");
    auto cew=read_vec<float>(c,@"param_0_grad");
    verify("CUDA vs f64 dW",cdw.data(),r.dw); verify("CUDA vs f64 dX",cdx.data(),r.dx);
    verify("CUDA vs f64 encdW",cew.data(),r.ew);
    if (!valid) return false;
    D ddw=mm(delta,r.x,3*H,H,R,true), ddx=mm(delta,wt,R,H,3*H);
    D dew=mm(ddx,ob,H,I,R,true);
    auto adjust=[&](const std::vector<float>& cuda,const D& d,std::vector<float>& result) {
        result.resize(cuda.size());
        for(size_t j=0;j<cuda.size();j++) result[j]=(float)(cuda[j]+d[j]);
    };
    adjust(cdw,ddw,adjusted_dw); adjust(cdx,ddx,adjusted_dx); adjust(cew,dew,adjusted_ew);
    printf("  %s: %ld roundoff-bounded derivative branch change(s); analytic correction only\n",cs,flips);
    return true;
}

static void check_policy_gradient(const Case& c,const char* cs,NSString* name,const float* x,
        long n,const std::vector<float>& adjusted) {
    if (adjusted.empty()) { check(c,cs,name,x,n,POLICY_TOL); return; }
    auto raw=read_vec<float>(c,name);
    Err e=compare_f32(x,raw.data(),n);
    printf("  %s %s raw CUDA norm=%.3e (derivative kink accounted below)\n",cs,name.UTF8String,e.norm());
    std::string label=std::string(name.UTF8String)+" [kink accounted]";
    report(cs,label.c_str(),compare_f32(x,adjusted.data(),n),POLICY_TOL);
}

static void run_policy(const Case& c, const char* cs) {
    int B = [c.scalars[@"B"] intValue], TT = [c.scalars[@"T"] intValue];
    int H = [c.scalars[@"H"] intValue], L = [c.scalars[@"layers"] intValue];
    int in_dim = [c.scalars[@"input_dim"] intValue];
    int agent_off = [c.scalars[@"agent_off"] intValue];
    int B_TT = B * TT;
    std::vector<float> enc_w = read_vec<float>(c, @"param_0");
    std::vector<float> dec_w = read_vec<float>(c, @"param_1");
    int od1 = (int)(dec_w.size() / H), od = od1 - 1;
    NSArray* st_shape = c.tensors[@"initial_state"][@"shape"];
    int A = [st_shape[1] intValue];

    float* obs = upload(read_vec<float>(c, @"obs"));
    float* state = upload(read_vec<float>(c, @"initial_state"));
    float* term = upload(read_vec<float>(c, @"terminals"));
    float* dlogits = upload(read_vec<float>(c, @"dlogits"));
    float* dvalue = upload(read_vec<float>(c, @"dvalue"));
    float* gnext = upload(read_vec<float>(c, @"grad_next_state"));
    float* W_enc = upload(enc_w);
    float* W_dec = upload(dec_w);
    std::vector<float*> W(L);
    for (int i = 0; i < L; i++) {
        W[i] = upload(read_vec<float>(c, [NSString stringWithFormat:@"param_%d", 2 + i]));
    }

    // Activations (algo.cu *_reg_train).
    float* enc_out = dalloc<float>((long)B_TT * H);
    float* enc_saved = dalloc<float>((long)B_TT * in_dim);
    float* enc_wgrad = dalloc<float>((long)H * in_dim);
    float* dec_out = dalloc<float>((long)B_TT * od1);
    float* dec_grad_out = dalloc<float>((long)B_TT * od1);
    float* dec_saved = dalloc<float>((long)B_TT * H);
    float* dec_grad_in = dalloc<float>((long)B_TT * H);
    float* dec_wgrad = dalloc<float>((long)od1 * H);
    float* grad_input_buf = dalloc<float>((long)B_TT * H);
    std::vector<float*> saved(L), combined(L), wgrad(L);
    std::vector<PrefixScan> scans(L);
    for (int i = 0; i < L; i++) {
        saved[i] = dalloc<float>((long)B_TT * H);
        combined[i] = dalloc<float>((long)B_TT * 3 * H);
        wgrad[i] = dalloc<float>((long)3 * H * H);
        scans[i] = alloc_scan(B, TT, H);
    }

    bool want_fused = getenv("PUF_GOLDWIDE_FUSED") && atoi(getenv("PUF_GOLDWIDE_FUSED"));
    bool fused = want_fused && puf_mtl::mingru_fused_eligible(B, TT, H, enc_out, W[0]);
    if (want_fused && (H == 128 || H == 256) && TT == 64 && !fused) {
        fprintf(stderr, "required H128/H256 golden fused path unavailable\n"); exit(2);
    }
    printf("  %s H=%d T=%d policy path=%s\n", cs, H, TT, fused ? "fused" : "generic");
    // Forward: encoder_forward, mingru_forward_train, decoder_forward.
    cudaStream_t stream = 0;
    cudaMemcpyAsync(enc_saved, obs, (size_t)B_TT * in_dim * 4, cudaMemcpyDeviceToDevice, stream);
    mm_nt(obs, W_enc, enc_out, B_TT, H, in_dim);
    float* x = enc_out;
    for (int i = 0; i < L; i++) {
        cudaMemcpyAsync(saved[i], x, (size_t)B_TT * H * 4, cudaMemcpyDeviceToDevice, stream);
        float* state_i = state + ((long)i * A + agent_off) * H;  // mingru_state_layer
        PrefixScan& s = scans[i];
        s.combined_ptr = combined[i];
        s.state_ptr = state_i;
        s.input_ptr = saved[i];
        s.terminals_ptr = term;
        if (fused) {
            if (!puf_mtl::mingru_fused_forward_rm(stream, B, TT, H, saved[i], W[i], state_i,
                    term, s.scan_h.data, s.out.data, s.next_state.data, combined[i])) {
                fprintf(stderr, "eligible golden fused forward rejected\n"); exit(2);
            }
        } else {
            mm_nt(saved[i], W[i], combined[i], B_TT, 3 * H, H);
            scan_forward(s, use_row(s), stream);
        }
        x = s.out.data;
    }
    cudaMemcpyAsync(dec_saved, x, (size_t)B_TT * H * 4, cudaMemcpyDeviceToDevice, stream);
    mm_nt(x, W_dec, dec_out, B_TT, od1, H);

    // Backward: decoder_backward, mingru_backward, encoder_backward.
    PUF_LAUNCH(assemble_decoder_grad, grid_size(B_TT * od1), BLOCK_SIZE, 0, stream)(
        dec_grad_out, dlogits, dvalue, B_TT, od, od1);
    mm_tn(dec_grad_out, dec_saved, dec_wgrad, od1, H, B_TT);
    mm_nn(dec_grad_out, W_dec, dec_grad_in, B_TT, H, od1);
    float* grad = dec_grad_in;
    for (int i = L - 1; i >= 0; i--) {
        PrefixScan& s = scans[i];
        if (fused) {
            if (!puf_mtl::mingru_fused_backward_rm(stream, B, TT, H, saved[i], W[i], term,
                    s.scan_h.data, grad, gnext, s.grad_combined.data, s.grad_state.data,
                    s.grad_input.data, grad_input_buf, nullptr, combined[i])) {
                fprintf(stderr, "eligible golden fused backward rejected\n"); exit(2);
            }
            mm_tn(s.grad_combined.data, saved[i], wgrad[i], 3 * H, H, B_TT);
        } else {
            scan_backward(s, use_row(s), grad, gnext, stream);
            mm_tn(s.grad_combined.data, saved[i], wgrad[i], 3 * H, H, B_TT);
            mm_nn(s.grad_combined.data, W[i], grad_input_buf, B_TT, H, 3 * H);
            int n = (int)numel(s.grad_input.shape);
            PUF_LAUNCH(add_kernel, grid_size(n), BLOCK_SIZE, 0, stream)(
                grad_input_buf, s.grad_input.data, n);
        }
        grad = grad_input_buf;
    }
    mm_tn(grad, enc_saved, enc_wgrad, H, in_dim, B_TT);
    cudaDeviceSynchronize();

    std::vector<float> adjusted_dw, adjusted_dx, adjusted_ew;
    if (H == 1024 && L == 1) {
        policy_kink_reference(c,cs,B,TT,H,in_dim,od1,obs,W_enc,W_dec,W[0],
            state + agent_off*H,term,dlogits,dvalue,gnext,saved[0],combined[0],scans[0],
            dec_grad_in,wgrad[0],grad,enc_wgrad,adjusted_dw,adjusted_dx,adjusted_ew);
    }
    if (const char* dir = getenv("PUF_POLICY_DUMP_DIR")) {
        [[NSFileManager defaultManager] createDirectoryAtPath:[NSString stringWithUTF8String:dir]
            withIntermediateDirectories:YES attributes:nil error:nil];
        auto dump = [&](const char* name, const float* data, long n) {
            std::string path=std::string(dir)+"/"+cs+"_"+name+".bin";
            FILE* fp=fopen(path.c_str(),"wb");
            if(!fp || fwrite(data,sizeof(float),n,fp)!=(size_t)n) exit(2);
            fclose(fp);
        };
        dump("enc_out",enc_out,(long)B_TT*H);
        dump("dec_grad_in",dec_grad_in,(long)B_TT*H);
        dump("encoder_grad",grad,(long)B_TT*H);
        dump("enc_dw",enc_wgrad,(long)H*in_dim);
        for(int i=0;i<L;i++) {
            std::string prefix=L==1 ? "" : "layer_"+std::to_string(i)+"_";
            auto layer_dump=[&](const char* name,const float* data,long n) {
                dump((prefix+name).c_str(),data,n);
            };
            layer_dump("combined",combined[i],(long)B_TT*3*H);
            layer_dump("scan_h",scans[i].scan_h.data,(long)B_TT*H);
            layer_dump("grad_combined",scans[i].grad_combined.data,(long)B_TT*3*H);
            layer_dump("grad_input",scans[i].grad_input.data,(long)B_TT*H);
            layer_dump("grad_state",scans[i].grad_state.data,(long)B*H);
            layer_dump("dw",wgrad[i],(long)3*H*H);
        }
    }
    check(c, cs, @"decoder", dec_out, (long)B_TT * od1, POLICY_TOL);
    for (int i = 0; i < L; i++) {
        check(c, cs, [NSString stringWithFormat:@"layer_%d_out", i], scans[i].out.data,
            (long)B_TT * H, POLICY_TOL);
        check(c, cs, [NSString stringWithFormat:@"layer_%d_next_state", i],
            scans[i].next_state.data, (long)B * H, POLICY_TOL);
    }
    check_policy_gradient(c, cs, @"param_0_grad", enc_wgrad, (long)H * in_dim, adjusted_ew);
    check(c, cs, @"param_1_grad", dec_wgrad, (long)od1 * H, POLICY_TOL);
    for (int i = 0; i < L; i++) {
        check_policy_gradient(c, cs, [NSString stringWithFormat:@"param_%d_grad", 2 + i], wgrad[i],
            (long)3 * H * H, adjusted_dw);
        check(c, cs, [NSString stringWithFormat:@"layer_%d_grad_state", i],
            scans[i].grad_state.data, (long)B * H, POLICY_TOL);
    }
    check_policy_gradient(c, cs, @"encoder_grad", grad, (long)B_TT * H, adjusted_dx);

    for (int i = 0; i < L; i++) {
        cudaFree(saved[i]); cudaFree(combined[i]); cudaFree(wgrad[i]); cudaFree(W[i]);
        free_scan(scans[i]);
    }
    for (float* p : {obs, state, term, dlogits, dvalue, gnext, W_enc, W_dec, enc_out, enc_saved,
             enc_wgrad, dec_out, dec_grad_out, dec_saved, dec_grad_in, dec_wgrad, grad_input_buf}) {
        cudaFree(p);
    }
}

static void run_golden(const char* run_dir) {
    NSString* root = [NSString stringWithUTF8String:run_dir];
    NSArray* names = [[[NSFileManager defaultManager] contentsOfDirectoryAtPath:root error:nil]
        sortedArrayUsingSelector:@selector(compare:)];
    printf("== golden (%s)\n", run_dir);
    int count = 0;
    for (NSString* name in names) {
        NSString* dir = [root stringByAppendingPathComponent:name];
        const char* cs = name.UTF8String;
        if ([name hasPrefix:@"scan_"]) {
            Case c = load_case(dir);
            bool row = [c.scalars[@"operation"] hasPrefix:@"mingru_scan_forward_row"];
            run_scan(c, cs, row, "");
            count++;
        } else if ([name hasPrefix:@"policy_"]) {
            Case c = load_case(dir);
            if (!has(c, @"dlogits") || !has(c, @"dvalue")) {
                printf("  %-20s skipped (no dlogits/dvalue)\n", cs);
                continue;
            }
            run_policy(c, cs);
            count++;
        }
    }
    // Cross-variant: the other scan implementation against each reference.
    printf("== cross-variant (plain on row references and vice versa)\n");
    for (NSString* name in names) {
        if (![name hasPrefix:@"scan_"]) continue;
        Case c = load_case([root stringByAppendingPathComponent:name]);
        bool row = [c.scalars[@"operation"] hasPrefix:@"mingru_scan_forward_row"];
        run_scan(c, name.UTF8String, !row, row ? " [plain]" : " [row]");
    }
    printf("golden cases: %d\n", count);
    if (count == 0) {
        fprintf(stderr, "no scan_/policy_ cases under %s\n", run_dir);
        g_fail++;
    }
}

// ---------------------------------------------------------------------------
// Synthetic
// ---------------------------------------------------------------------------

static void run_synthetic() {
    printf("== synthetic\n");
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> U(-2.0f, 2.0f);

    // sum_rows_to_precision_kernel: column sums in row order (host float loop
    // with the same order is exact).
    for (int R : {1, 7, 65536}) {
        for (int C : {1, 3, 300}) {
            float* src = dalloc<float>((long)R * C);
            float* dst = dalloc<float>(C);
            for (long i = 0; i < (long)R * C; i++) src[i] = U(rng);
            PUF_LAUNCH(sum_rows_to_precision_kernel, grid_size(C), BLOCK_SIZE, 0, 0)(
                dst, src, R, C);
            cudaDeviceSynchronize();
            std::vector<float> ref(C);
            for (int col = 0; col < C; col++) {
                float s = 0.0f;
                for (int r = 0; r < R; r++) s += src[(long)r * C + col];
                ref[col] = s;
            }
            char nm[64];
            snprintf(nm, sizeof nm, "sum_rows R=%d C=%d", R, C);
            Err e = compare_f32(dst, ref.data(), C);
            if (e.mismatches) g_fail++;
            printf("  %-20s %-20s n=%-8d mismatches=%ld%s\n", "synthetic", nm, C, e.mismatches,
                e.mismatches ? "  FAIL" : "");
            cudaFree(src); cudaFree(dst);
        }
    }

    // assemble_decoder_grad.
    for (int od : {1, 3, 17}) {
        int B_TT = 1000, od1 = od + 1;
        float* gl = dalloc<float>((long)B_TT * od);
        float* gv = dalloc<float>(B_TT);
        float* dst = dalloc<float>((long)B_TT * od1);
        for (long i = 0; i < (long)B_TT * od; i++) gl[i] = U(rng);
        for (int i = 0; i < B_TT; i++) gv[i] = U(rng);
        PUF_LAUNCH(assemble_decoder_grad, grid_size(B_TT * od1), BLOCK_SIZE, 0, 0)(
            dst, gl, gv, B_TT, od, od1);
        cudaDeviceSynchronize();
        long bad = 0;
        for (int r = 0; r < B_TT; r++) {
            for (int col = 0; col < od1; col++) {
                float want = col < od ? gl[(long)r * od + col] : gv[r];
                if (memcmp(&dst[(long)r * od1 + col], &want, 4) != 0) bad++;
            }
        }
        if (bad) g_fail++;
        char nm[64];
        snprintf(nm, sizeof nm, "assemble od=%d", od);
        printf("  %-20s %-20s n=%-8d mismatches=%ld%s\n", "synthetic", nm, B_TT * od1, bad,
            bad ? "  FAIL" : "");
        cudaFree(gl); cudaFree(gv); cudaFree(dst);
    }

    // terminals == NULL must equal all-zero terminals, both variants, with H
    // not a multiple of the tile (row kernel partial h-tile) and T spanning
    // several T tiles.
    for (bool row : {false, true}) {
        int B = 3, T = 70, H = 37;
        long BTH = (long)B * T * H;
        float* comb = dalloc<float>(3 * BTH);
        float* x = dalloc<float>(BTH);
        float* st = dalloc<float>((long)B * H);
        float* gout = dalloc<float>(BTH);
        float* gnext = dalloc<float>((long)B * H);
        float* zeros = dalloc<float>((long)B * T);
        for (long i = 0; i < 3 * BTH; i++) comb[i] = U(rng);
        for (long i = 0; i < BTH; i++) { x[i] = U(rng); gout[i] = U(rng); }
        for (long i = 0; i < (long)B * H; i++) { st[i] = U(rng); gnext[i] = U(rng); }
        PrefixScan a = alloc_scan(B, T, H), b = alloc_scan(B, T, H);
        for (PrefixScan* s : {&a, &b}) {
            s->combined_ptr = comb; s->state_ptr = st; s->input_ptr = x;
        }
        a.terminals_ptr = NULL;
        b.terminals_ptr = zeros;
        for (PrefixScan* s : {&a, &b}) {
            scan_forward(*s, row, 0);
            scan_backward(*s, row, gout, gnext, 0);
        }
        cudaDeviceSynchronize();
        long bad = 0;
        bad += memcmp(a.out.data, b.out.data, BTH * 4) != 0;
        bad += memcmp(a.scan_h.data, b.scan_h.data, BTH * 4) != 0;
        bad += memcmp(a.next_state.data, b.next_state.data, B * H * 4) != 0;
        bad += memcmp(a.grad_combined.data, b.grad_combined.data, 3 * BTH * 4) != 0;
        bad += memcmp(a.grad_input.data, b.grad_input.data, BTH * 4) != 0;
        bad += memcmp(a.grad_state.data, b.grad_state.data, B * H * 4) != 0;
        if (bad) g_fail++;
        printf("  %-20s %-20s tensors_differing=%ld%s\n", "synthetic",
            row ? "null_terminals row" : "null_terminals plain", bad, bad ? "  FAIL" : "");
        free_scan(a); free_scan(b);
        for (float* p : {comb, x, st, gout, gnext, zeros}) cudaFree(p);
    }
}

// ---------------------------------------------------------------------------
// Fused projection + scan (gemm.mm mingru_fused_*_rm) vs the unfused
// algo.cu sequence (tensor_ops nt GEMM into combined, then the plain scans):
// every output must be bit-identical.
// ---------------------------------------------------------------------------

static void run_fused() {
    printf("== fused projection + scan (+ dX) vs GEMM + scan (+ GEMM/add), bit-exact\n");
    std::mt19937 rng(23);
    std::uniform_real_distribution<float> U(-2.0f, 2.0f);
    struct Shape { int B, T; bool term; };
    for (Shape sh : {Shape{2048, 32, true}, Shape{64, 32, false}, Shape{96, 16, true},
             Shape{40, 64, true}, Shape{512, 8, true}, Shape{256, 1, true}, Shape{160, 2, true}}) {
        int B = sh.B, T = sh.T, H = 32, M = B * T;
        long BTH = (long)M * H;
        float* x = dalloc<float>(BTH);
        float* W = dalloc<float>(3 * H * H);
        float* st = dalloc<float>((long)B * H);
        float* term = dalloc<float>(M);
        float* gout = dalloc<float>(BTH);
        float* gnext = dalloc<float>((long)B * H);
        float* comb = dalloc<float>(3 * BTH);
        for (long i = 0; i < BTH; i++) { x[i] = U(rng); gout[i] = U(rng); }
        for (int i = 0; i < 3 * H * H; i++) W[i] = 0.3f * U(rng);
        for (long i = 0; i < (long)B * H; i++) { st[i] = U(rng); gnext[i] = U(rng); }
        for (int i = 0; i < M; i++) term[i] = (rng() % 7) == 0;
        float* tp = sh.term ? term : nullptr;
        PrefixScan r = alloc_scan(B, T, H), f = alloc_scan(B, T, H);
        poison(f.scan_h.data, BTH); poison(f.out.data, BTH); poison(f.next_state.data, (long)B * H);
        poison(f.grad_combined.data, 3 * BTH); poison(f.grad_state.data, (long)B * H);
        poison(f.grad_input.data, BTH);
        mm_nt(x, W, comb, M, 3 * H, H);
        r.combined_ptr = comb; r.state_ptr = st; r.input_ptr = x; r.terminals_ptr = tp;
        scan_forward(r, false, 0);
        scan_backward(r, false, gout, gnext, 0);
        bool okf = puf_mtl::mingru_fused_forward_rm(0, B, T, H, x, W, st, tp,
            f.scan_h.data, f.out.data, f.next_state.data);
        // dX: tensor_ops nn GEMM + add epilogue (algo.cu's unfused path).
        float* gx_ref = dalloc<float>(BTH);
        float* gx = dalloc<float>(BTH);
        poison(gx, BTH);
        bool okr = puf_mtl::gemm_nn_add_rm(0, M, H, 3 * H, r.grad_combined.data, W,
            r.grad_input.data, gx_ref);
        bool okb = okr && puf_mtl::mingru_fused_backward_rm(0, B, T, H, x, W, tp,
            f.scan_h.data, gout, gnext, f.grad_combined.data, f.grad_state.data,
            f.grad_input.data, gx);
        // grad_x must not alias grad_input (the epilogue's addend).
        okb = okb && !puf_mtl::mingru_fused_backward_rm(0, B, T, H, x, W, tp, f.scan_h.data,
            gout, gnext, f.grad_combined.data, f.grad_state.data, gx, gx);
        // H=32 always writes grad_input: a null one is rejected.
        okb = okb && !puf_mtl::mingru_fused_backward_rm(0, B, T, H, x, W, tp, f.scan_h.data,
            gout, gnext, f.grad_combined.data, f.grad_state.data, nullptr, gx);
        cudaDeviceSynchronize();
        long bad = 0;
        if (okf && okb) bad += memcmp(gx_ref, gx, BTH * 4) != 0;
        // In place, as algo.cu runs it below the top layer: grad_out is the
        // previous layer's grad_x buffer and receives this layer's grad_x.
        float* gio = dalloc<float>(BTH);
        memcpy(gio, gout, BTH * 4);
        PrefixScan d2 = alloc_scan(B, T, H);
        poison(d2.grad_combined.data, 3 * BTH);
        bool oki = puf_mtl::mingru_fused_backward_rm(0, B, T, H, x, W, tp, f.scan_h.data,
            gio, gnext, d2.grad_combined.data, d2.grad_state.data, d2.grad_input.data, gio);
        cudaDeviceSynchronize();
        if (oki) {
            bad += memcmp(gx_ref, gio, BTH * 4) != 0;
            bad += memcmp(r.grad_combined.data, d2.grad_combined.data, 3 * BTH * 4) != 0;
            bad += memcmp(r.grad_state.data, d2.grad_state.data, B * H * 4) != 0;
        } else {
            okb = false;
        }
        // Fused dW emits one partial per 64 rows, then reduces in tile order.
        // Its different summation order has a tight FP32 relative error gate;
        // dX, highway/state gradients and repeated output must remain exact.
        float* dw_ref = dalloc<float>(3 * H * H);
        float* dw = dalloc<float>(3 * H * H);
        float* dw2 = dalloc<float>(3 * H * H);
        mm_tn(r.grad_combined.data, x, dw_ref, 3 * H, H, M);
        bool okdw = puf_mtl::mingru_fused_backward_rm(0, B, T, H, x, W, tp, f.scan_h.data,
            gout, gnext, d2.grad_combined.data, d2.grad_state.data, d2.grad_input.data, gx, dw);
        cudaDeviceSynchronize();
        Err de = compare_f32(dw, dw_ref, 3 * H * H);
        bad += !okdw || de.norm() > 2e-5;
        bad += memcmp(gx_ref, gx, BTH * 4) != 0;
        bad += memcmp(r.grad_input.data, d2.grad_input.data, BTH * 4) != 0;
        bad += memcmp(r.grad_state.data, d2.grad_state.data, B * H * 4) != 0;
        // Also exercise grad_out == grad_x with dW enabled.
        memcpy(gio, gout, BTH * 4);
        okdw = puf_mtl::mingru_fused_backward_rm(0, B, T, H, x, W, tp, f.scan_h.data,
            gio, gnext, d2.grad_combined.data, d2.grad_state.data, d2.grad_input.data, gio, dw2);
        cudaDeviceSynchronize();
        bad += !okdw || memcmp(dw, dw2, 3 * H * H * 4) != 0;
        bad += memcmp(gx_ref, gio, BTH * 4) != 0;
        // The batched-load reduction must equal the plain serial loop exactly.
        float* dw_serial = dalloc<float>(3 * H * H);
        poison(dw_serial, 3 * H * H);
        puf_mtl::mingru_dw_reduce_rm(0, d2.grad_combined.data, dw_serial, M / 64, true);
        cudaDeviceSynchronize();
        bad += memcmp(dw, dw_serial, 3 * H * H * 4) != 0;
        cudaFree(dw_serial);
        printf("  fused dW B=%d T=%d norm_err=%.3g repeat_exact=%d\n", B, T, de.norm(),
            (int)(memcmp(dw, dw2, 3 * H * H * 4) == 0));
        cudaFree(dw_ref); cudaFree(dw); cudaFree(dw2);
        free_scan(d2); cudaFree(gx_ref); cudaFree(gx); cudaFree(gio);
        if (okf && okb) {
            bad += memcmp(r.out.data, f.out.data, BTH * 4) != 0;
            bad += memcmp(r.scan_h.data, f.scan_h.data, BTH * 4) != 0;
            bad += memcmp(r.next_state.data, f.next_state.data, B * H * 4) != 0;
            bad += memcmp(r.grad_combined.data, f.grad_combined.data, 3 * BTH * 4) != 0;
            bad += memcmp(r.grad_input.data, f.grad_input.data, BTH * 4) != 0;
            bad += memcmp(r.grad_state.data, f.grad_state.data, B * H * 4) != 0;
        }
        bool fail = !okf || !okb || bad;
        if (fail) g_fail++;
        char nm[64];
        snprintf(nm, sizeof nm, "B=%d T=%d term=%d", B, T, (int)sh.term);
        printf("  %-20s %-22s eligible=%d tensors_differing=%ld%s\n", "fused", nm,
            (int)(okf && okb), bad, fail ? "  FAIL" : "");
        free_scan(r); free_scan(f);
        for (float* p : {x, W, st, term, gout, gnext, comb}) cudaFree(p);
    }
    // Ineligible shapes must encode nothing and report false.
    float* dummy = dalloc<float>(4096);
    struct Bad { int B, T, H; };
    long wrong = 0;
    for (Bad b : {Bad{64, 32, 48}, Bad{64, 32, 512}, Bad{64, 48, 32}, Bad{3, 32, 32}, Bad{2, 128, 32}, Bad{0, 32, 32}}) {
        wrong += puf_mtl::mingru_fused_eligible(b.B, b.T, b.H, dummy, dummy);
    }
    wrong += puf_mtl::mingru_fused_eligible(64, 32, 32, dummy + 1, dummy);
    if (wrong) g_fail++;
    printf("  %-20s %-22s accepted=%ld%s\n", "fused", "ineligible shapes", wrong,
        wrong ? "  FAIL" : "");
    cudaFree(dummy);
}

// ---------------------------------------------------------------------------
// H=64 fused projection + scan (+ dX) vs the unfused algo.cu sequence. The
// H=64 forward also stores combined, the backward reads it and writes
// grad_combined (no fused dW), so every output must be bit-identical. Every output is poisoned with 0x7fc1 halfwords (NaN as f32
// and bf16) immediately before the call that must write it.
// ---------------------------------------------------------------------------

static void poison_nan(float* p, long n) {
    uint16_t* q = (uint16_t*)p;
    for (long i = 0; i < 2 * n; i++) q[i] = 0x7fc1;
}

static long bits_differ(const float* a, const float* b, long n) {
    long bad = 0;
    for (long i = 0; i < n; i++) {
        bad += memcmp(&a[i], &b[i], 4) != 0 || !std::isfinite(a[i]);
    }
    return bad;
}

static void run_fused64() {
    printf("== fused H=64 projection + scan (+ dX) vs GEMM + scan (+ GEMM/add), bit-exact\n");
    std::mt19937 rng(29);
    std::uniform_real_distribution<float> U(-2.0f, 2.0f);
    struct Shape { int B, T; bool term; int repeats; };
    for (Shape sh : {Shape{192, 1, true, 1}, Shape{96, 2, true, 1}, Shape{48, 4, true, 1},
             Shape{24, 8, true, 1}, Shape{12, 16, true, 1}, Shape{6, 32, true, 1},
             Shape{3, 64, true, 1}, Shape{2, 64, false, 3}, Shape{9, 64, true, 2},
             Shape{256, 64, true, 2}, Shape{256, 64, false, 1}}) {
        const int B = sh.B, T = sh.T, H = 64, M = B * T;
        const long BTH = (long)M * H, BH = (long)B * H;
        float* x = dalloc<float>(BTH);
        float* W = dalloc<float>(3 * H * H);
        float* st = dalloc<float>(BH);
        float* term = dalloc<float>(M);
        float* gout = dalloc<float>(BTH);
        float* gnext = dalloc<float>(BH);
        float* comb = dalloc<float>(3 * BTH);
        float* gx_ref = dalloc<float>(BTH);
        float* gx = dalloc<float>(BTH);
        float* gio = dalloc<float>(BTH);
        float* dw = dalloc<float>(3 * H * H);
        float* fcomb = dalloc<float>(3 * BTH);
        for (long i = 0; i < BTH; i++) { x[i] = U(rng); gout[i] = U(rng); }
        for (int i = 0; i < 3 * H * H; i++) W[i] = 0.3f * U(rng);
        for (long i = 0; i < BH; i++) { st[i] = U(rng); gnext[i] = U(rng); }
        for (int i = 0; i < M; i++) term[i] = (rng() % 7) == 0 || i % T == 0 || i % T == T - 1;
        float* tp = sh.term ? term : nullptr;
        PrefixScan r = alloc_scan(B, T, H), f = alloc_scan(B, T, H);
        long bad = 0;
        bool ok = true;
        for (int it = 0; it < sh.repeats && ok; it++) {
            for (PrefixScan* s : {&r, &f}) {
                poison_nan(s->scan_h.data, BTH); poison_nan(s->out.data, BTH);
                poison_nan(s->next_state.data, BH); poison_nan(s->grad_combined.data, 3 * BTH);
                poison_nan(s->grad_state.data, BH); poison_nan(s->grad_input.data, BTH);
            }
            poison_nan(comb, 3 * BTH); poison_nan(gx_ref, BTH); poison_nan(gx, BTH);
            poison_nan(fcomb, 3 * BTH);
            mm_nt(x, W, comb, M, 3 * H, H);
            ok &= puf_mtl::gemm_last_path() == puf_mtl::GEMM_TENSOR_OPS;
            r.combined_ptr = comb; r.state_ptr = st; r.input_ptr = x; r.terminals_ptr = tp;
            scan_forward(r, false, 0);
            scan_backward(r, false, gout, gnext, 0);
            ok &= puf_mtl::gemm_nn_add_rm(0, M, H, 3 * H, r.grad_combined.data, W,
                r.grad_input.data, gx_ref);
            ok &= puf_mtl::mingru_fused_forward_rm(0, B, T, H, x, W, st, tp,
                f.scan_h.data, f.out.data, f.next_state.data, fcomb);
            ok &= puf_mtl::mingru_fused_backward_rm(0, B, T, H, x, W, tp, f.scan_h.data,
                gout, gnext, f.grad_combined.data, f.grad_state.data, f.grad_input.data, gx,
                nullptr, fcomb);
            cudaDeviceSynchronize();
            if (!ok) break;
            bad += bits_differ(fcomb, comb, 3 * BTH);
            bad += bits_differ(f.out.data, r.out.data, BTH);
            bad += bits_differ(f.scan_h.data, r.scan_h.data, BTH);
            bad += bits_differ(f.next_state.data, r.next_state.data, BH);
            bad += bits_differ(f.grad_combined.data, r.grad_combined.data, 3 * BTH);
            bad += bits_differ(f.grad_input.data, r.grad_input.data, BTH);
            bad += bits_differ(f.grad_state.data, r.grad_state.data, BH);
            bad += bits_differ(gx, gx_ref, BTH);
            // In place (grad_out == grad_x), as algo.cu runs it below the top layer.
            poison_nan(f.grad_combined.data, 3 * BTH); poison_nan(f.grad_state.data, BH);
            poison_nan(f.grad_input.data, BTH);
            memcpy(gio, gout, BTH * 4);
            ok &= puf_mtl::mingru_fused_backward_rm(0, B, T, H, x, W, tp, f.scan_h.data,
                gio, gnext, f.grad_combined.data, f.grad_state.data, f.grad_input.data, gio,
                nullptr, fcomb);
            cudaDeviceSynchronize();
            bad += bits_differ(gio, gx_ref, BTH);
            bad += bits_differ(f.grad_combined.data, r.grad_combined.data, 3 * BTH);
            bad += bits_differ(f.grad_input.data, r.grad_input.data, BTH);
            bad += bits_differ(f.grad_state.data, r.grad_state.data, BH);
            // grad_input == nullptr (algo.cu): the epilogue's addend stays on
            // chip; grad_x and the other outputs are unchanged and the device
            // grad_input is left untouched.
            poison_nan(f.grad_combined.data, 3 * BTH); poison_nan(f.grad_state.data, BH);
            poison_nan(f.grad_input.data, BTH);
            memcpy(gio, gout, BTH * 4);
            ok &= puf_mtl::mingru_fused_backward_rm(0, B, T, H, x, W, tp, f.scan_h.data,
                gio, gnext, f.grad_combined.data, f.grad_state.data, nullptr, gio,
                nullptr, fcomb);
            cudaDeviceSynchronize();
            bad += bits_differ(gio, gx_ref, BTH);
            bad += bits_differ(f.grad_combined.data, r.grad_combined.data, 3 * BTH);
            bad += bits_differ(f.grad_state.data, r.grad_state.data, BH);
            for (long k = 0; k < BTH; k++) bad += !std::isnan(f.grad_input.data[k]);
            // No fused dW at H=64: rejected without encoding anything.
            auto before = puf_mtl::stats();
            ok &= !puf_mtl::mingru_fused_backward_rm(0, B, T, H, x, W, tp, f.scan_h.data,
                gout, gnext, f.grad_combined.data, f.grad_state.data, f.grad_input.data, gx, dw,
                fcomb);
            // combined is required (and must not be grad_combined).
            ok &= !puf_mtl::mingru_fused_backward_rm(0, B, T, H, x, W, tp, f.scan_h.data,
                gout, gnext, f.grad_combined.data, f.grad_state.data, f.grad_input.data, gx);
            ok &= !puf_mtl::mingru_fused_backward_rm(0, B, T, H, x, W, tp, f.scan_h.data,
                gout, gnext, f.grad_combined.data, f.grad_state.data, f.grad_input.data, gx,
                nullptr, f.grad_combined.data);
            ok &= !puf_mtl::mingru_fused_forward_rm(0, B, T, H, x, W, st, tp,
                f.scan_h.data, f.out.data, f.next_state.data);
            ok &= before.dispatches == puf_mtl::stats().dispatches;
            // Carry the state across horizons.
            memcpy(st, r.next_state.data, BH * 4);
        }
        bool fail = !ok || bad;
        if (fail) g_fail++;
        char nm[64];
        snprintf(nm, sizeof nm, "B=%d T=%d term=%d x%d", B, T, (int)sh.term, sh.repeats);
        printf("  %-20s %-22s eligible=%d values_differing=%ld%s\n", "fused H=64", nm,
            (int)ok, bad, fail ? "  FAIL" : "");
        free_scan(r); free_scan(f);
        for (float* p : {x, W, st, term, gout, gnext, comb, gx_ref, gx, gio, dw, fcomb}) cudaFree(p);
    }
    float* dummy = dalloc<float>(4096);
    long wrong = 0;
    struct Bad { int B, T; };
    for (Bad b : {Bad{64, 48}, Bad{3, 32}, Bad{1, 32}, Bad{2, 128}, Bad{0, 64}, Bad{64, 0}}) {
        wrong += puf_mtl::mingru_fused_eligible(b.B, b.T, 64, dummy, dummy);
    }
    wrong += puf_mtl::mingru_fused_eligible(64, 64, 64, dummy + 1, dummy);
    wrong += puf_mtl::mingru_fused_eligible(64, 64, 64, dummy, dummy + 1);
    wrong += puf_mtl::mingru_fused_eligible(0x7fffffff / 64, 64, 64, dummy, dummy);
    if (wrong) g_fail++;
    printf("  %-20s %-22s accepted=%ld%s\n", "fused H=64", "ineligible shapes", wrong,
        wrong ? "  FAIL" : "");
    cudaFree(dummy);
}

// ---------------------------------------------------------------------------
// Bench: Breakout train shape (4096 agents, horizon 32, minibatch 65536 =
// 2048 sequences, hidden 32). Wall time over N back-to-back dispatches on one
// stream (dispatch barriers between them) / N, after a warmup: an upper bound
// on per-kernel GPU time.
// ---------------------------------------------------------------------------

template <typename F>
static double time_ms(F&& launch, int iters = 200) {
    for (int i = 0; i < 10; i++) launch();
    cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) launch();
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

static void run_bench() {
    printf("== bench (Breakout minibatch: B=2048 seqs, T=32, H=32; per-launch ms)\n");
    int B = 2048, T = 32, H = 32;
    long BTH = (long)B * T * H;
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> U(-2.0f, 2.0f);
    float* comb = dalloc<float>(3 * BTH);
    float* x = dalloc<float>(BTH);
    float* st = dalloc<float>((long)B * H);
    float* gout = dalloc<float>(BTH);
    float* gnext = dalloc<float>((long)B * H);
    float* term = dalloc<float>((long)B * T);
    for (long i = 0; i < 3 * BTH; i++) comb[i] = U(rng);
    for (long i = 0; i < BTH; i++) { x[i] = U(rng); gout[i] = U(rng); }
    for (long i = 0; i < (long)B * H; i++) { st[i] = U(rng); gnext[i] = U(rng); }
    for (long i = 0; i < (long)B * T; i++) term[i] = (rng() % 50) == 0;
    PrefixScan s = alloc_scan(B, T, H);
    s.combined_ptr = comb; s.state_ptr = st; s.input_ptr = x; s.terminals_ptr = term;
    double f = time_ms([&] { scan_forward(s, false, 0); });
    double b = time_ms([&] { scan_backward(s, false, gout, gnext, 0); });
    double fr = time_ms([&] { scan_forward(s, true, 0); });
    double br = time_ms([&] { scan_backward(s, true, gout, gnext, 0); });
    printf("  mingru_scan_forward       %.4f ms  (x4 layers = %.3f ms)\n", f, 4 * f);
    printf("  mingru_scan_backward      %.4f ms  (x4 layers = %.3f ms)\n", b, 4 * b);
    printf("  mingru_scan_forward_row   %.4f ms  (not selected at T=32)\n", fr);
    {
        float* W = dalloc<float>(3 * H * H);
        for (int i = 0; i < 3 * H * H; i++) W[i] = 0.1f * U(rng);
        PrefixScan fs = alloc_scan(B, T, H);
        double g = time_ms([&] { mm_nt(x, W, comb, B * T, 3 * H, H); });
        double ff = time_ms([&] { puf_mtl::mingru_fused_forward_rm(0, B, T, H, x, W, st, term,
            fs.scan_h.data, fs.out.data, fs.next_state.data); });
        printf("  gemm nt x@W^T (65536x96)  %.4f ms\n", g);
        float* gx = dalloc<float>(BTH);
        float* gx2 = dalloc<float>(BTH);
        double bdx = time_ms([&] { puf_mtl::mingru_fused_backward_rm(0, B, T, H, x, W, term,
            fs.scan_h.data, gout, gnext, fs.grad_combined.data, fs.grad_state.data,
            fs.grad_input.data, gx); });
        float* dw = dalloc<float>(3 * H * H);
        double bdw = time_ms([&] { puf_mtl::mingru_fused_backward_rm(0, B, T, H, x, W, term,
            fs.scan_h.data, gout, gnext, fs.grad_combined.data, fs.grad_state.data,
            fs.grad_input.data, gx, dw); });
        {
            // dW reduction alone: shipped batched-load kernel vs the serial oracle.
            cudaDeviceSynchronize();
            float* ref = dalloc<float>(3 * H * H);
            puf_mtl::mingru_dw_reduce_rm(0, fs.grad_combined.data, ref, B * T / 64, true);
            cudaDeviceSynchronize();
            if (memcmp(ref, dw, 3 * H * H * 4) != 0) g_fail++;
            double rs = time_ms([&] { puf_mtl::mingru_dw_reduce_rm(0, fs.grad_combined.data,
                ref, B * T / 64, true); });
            double rf = time_ms([&] { puf_mtl::mingru_dw_reduce_rm(0, fs.grad_combined.data,
                ref, B * T / 64); });
            printf("  dW reduce (1024 tiles)    %.4f ms  (serial oracle %.4f ms)\n", rf, rs);
            cudaFree(ref);
        }
        // Restore full grad_combined before timing the standalone dX.
        puf_mtl::mingru_fused_backward_rm(0, B, T, H, x, W, term,
            fs.scan_h.data, gout, gnext, fs.grad_combined.data, fs.grad_state.data,
            fs.grad_input.data, gx);
        double nadd = time_ms([&] { puf_mtl::gemm_nn_add_rm(0, B * T, H, 3 * H,
            fs.grad_combined.data, W, fs.grad_input.data, gx2); });
        printf("  fused backward (+ dX)     %.4f ms  (vs scan bwd + nn_add %.4f ms)\n", bdx, b + nadd);
        printf("  fused backward + dW/dX     %.4f ms\n", bdw);
        cudaFree(dw); cudaFree(gx); cudaFree(gx2);
        printf("  fused forward             %.4f ms  (vs gemm + scan %.4f ms)\n", ff, g + f);
        free_scan(fs);
        cudaFree(W);
    }
    printf("  mingru_scan_backward_row  %.4f ms  (not selected at T=32)\n", br);
    int B_TT = B * T, od = 3, od1 = 4;
    float* gl = dalloc<float>((long)B_TT * od);
    float* gv = dalloc<float>(B_TT);
    float* dst = dalloc<float>((long)B_TT * od1);
    double ad = time_ms([&] {
        PUF_LAUNCH(assemble_decoder_grad, grid_size(B_TT * od1), BLOCK_SIZE, 0, 0)(
            dst, gl, gv, B_TT, od, od1);
    });
    double sr = time_ms([&] {
        PUF_LAUNCH(sum_rows_to_precision_kernel, grid_size(od), BLOCK_SIZE, 0, 0)(
            dst, gl, B_TT, od);
    });
    float* gi = dalloc<float>(BTH);
    double ak = time_ms([&] {
        PUF_LAUNCH(add_kernel, grid_size((int)BTH), BLOCK_SIZE, 0, 0)(gi, x, (int)BTH);
    });
    double empty = time_ms([&] {
        PUF_LAUNCH(assemble_decoder_grad, 1, 1, 0, 0)(dst, gl, gv, 0, od, od1);
    });
    printf("  assemble_decoder_grad     %.4f ms  (B_TT=65536, od+1=4)\n", ad);
    printf("  sum_rows_to_precision     %.4f ms  (R=65536, C=3; continuous envs only)\n", sr);
    printf("  add_kernel                %.4f ms  (n=2.1M, per layer)\n", ak);
    printf("  [empty dispatch overhead  %.4f ms]\n", empty);
    free_scan(s);
    for (float* p : {comb, x, st, gout, gnext, term, gl, gv, dst, gi}) cudaFree(p);
}

int main(int argc, char** argv) {
    @autoreleasepool {
        puf_mtl::init();
        // algo.cu cublas_init_stream: stream first, then a 4 MiB workspace.
        cublasCreate(&g_cublas);
        cublasSetStream(g_cublas, 0);
        size_t ws_bytes = 4 * 1024 * 1024;
        void* ws = nullptr;
        cudaMalloc(&ws, ws_bytes);
        cublasSetWorkspace(g_cublas, ws, ws_bytes);
        bool bench = false;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--bench") == 0) {
                bench = true;
            } else {
                run_golden(argv[i]);
            }
        }
        run_synthetic();
        run_fused();
        run_fused64();
        if (bench) run_bench();
        printf(g_fail ? "FAILURES: %d\n" : "ALL PASS\n", g_fail);
        return g_fail ? 1 : 0;
    }
}

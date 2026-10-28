// Learner kernel parity: real PUF_LAUNCH adapters, CUDA FP32 goldens, and
// independent host references for branches not present in the golden set.
// Run tests/metal/run_test_ppo_muon.sh [GOLDEN_RUN_DIR ...] [--timings]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mach/mach_time.h>
#include <type_traits>
#include <string>
#include <vector>
#include "compat/cuda_runtime.h"
#include "compat/curand_kernel.h"
#include "compat/cublas_v2.h"
#include "launch.h"
#include "runtime.h"
#include "gemm.h"

// Host types used by kernels.h, matching src/pufferl.cu and src/algo.cu.
using precision_t = float;
struct Float { float* data; int64_t shape[8]; };
struct Prec { float* data; int64_t shape[8]; };
static int64_t numel(int64_t* s) { int64_t n = 1; for (int i = 0; i < 8 && s[i]; i++) n *= s[i]; return n; }
static int g_num_atns = 1;
#define NUM_ATNS g_num_atns
struct PPOGraphArgs {
    float* imp;
    const float *actions, *old_logprobs, *advantages, *values, *returns;
};
struct PPOKernelArgs {
    float *grad_logits, *grad_logstd, *grad_values_pred;
    const float *logits, *logstd, *values_pred;
    const int* act_sizes;
    const float* action_mask;
    int num_atns;
    float clip_coef, vf_clip_coef, vf_coef;
    const float* ent_coef;
    int T_seq, A_total, N;
    bool is_continuous;
};
#include "kernels.h"

static int failures = 0;
static NSMutableDictionary* results;
static void require(bool ok, const char* msg) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", msg); failures++; }
}
static int grid(int n) { return (n + 255) / 256; }
// Scoped GPU allocations; free only after all streams have drained (runtime).
struct Buffers {
    struct Allocation { unsigned char* base; size_t bytes; };
    std::vector<Allocation> ptrs;
    template<class T = float> T* alloc(size_t n) {
        unsigned char* base = nullptr;
        size_t bytes = std::max(n, size_t(1)) * sizeof(T);
        if (cudaMalloc(&base, bytes + 128) != cudaSuccess) abort();
        memset(base, 0xa5, 64); memset(base + 64 + bytes, 0xa5, 64);
        ptrs.push_back({base, bytes}); return reinterpret_cast<T*>(base + 64);
    }
    template<class T> T* put(const std::vector<T>& v) {
        T* p = alloc<T>(v.size()); memcpy(p, v.data(), v.size() * sizeof(T)); return p;
    }
    ~Buffers() {
        cudaDeviceSynchronize();
        for (auto a : ptrs) {
            bool intact = true;
            for (int j=0;j<64;j++) intact &= a.base[j]==0xa5 && a.base[64+a.bytes+j]==0xa5;
            require(intact, "GPU allocation guard overwritten");
            cudaFree(a.base);
        }
    }
};
struct Case {
    NSString* dir;
    NSDictionary *tensors, *scalars;
    int I(NSString* k) const { return [scalars[k] intValue]; }
    float F(NSString* k) const { return [scalars[k] floatValue]; }
    NSArray* shape(NSString* k) const { return tensors[k][@"shape"]; }
    template<class T = float> std::vector<T> read(NSString* k) const {
        NSDictionary* spec = tensors[k];
        NSString* file = spec[@"file"];
        if (!spec || ![file.lastPathComponent isEqualToString:file]) abort();
        NSString* dtype = sizeof(T) == 4 && std::is_same<T, int>::value ? @"i32" : @"f32";
        if (![spec[@"dtype"] isEqual:dtype]) abort();
        size_t n = 1;
        for (NSNumber* d in spec[@"shape"]) { if (d.longLongValue <= 0) abort(); n *= d.unsignedLongLongValue; }
        NSData* data = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:file]];
        if (!data || data.length != n * sizeof(T)) { fprintf(stderr, "bad tensor %s/%s\n", dir.UTF8String, k.UTF8String); exit(2); }
        std::vector<T> v(n); memcpy(v.data(), data.bytes, data.length); return v;
    }
};
static Case load(NSString* root, NSString* name) {
    NSString* dir = [root stringByAppendingPathComponent:name];
    NSData* data = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"manifest.json"]];
    if (!data) { fprintf(stderr, "missing case %s\n", dir.UTF8String); exit(2); }
    NSDictionary* m = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    if ([m[@"schema_version"] intValue] != 1 || ![m[@"byte_order"] isEqual:@"little"] ||
        ![m[@"precision"] isEqual:@"PRECISION_FLOAT"]) abort();
    return {dir, m[@"tensors"], m[@"scalars"]};
}
static void compare(NSString* cs, NSString* tensor, const float* got, const std::vector<float>& ref,
        double atol = 2e-6, double rtol = 2e-6) {
    double ma = 0, mr = 0;
    size_t bad = 0, bits = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        double a = got[i], b = ref[i];
        bits += memcmp(got + i, ref.data() + i, 4) != 0;
        if (a == b || (std::isnan(a) && std::isnan(b))) continue;
        double d = (std::isfinite(a) && std::isfinite(b)) ? fabs(a - b) : INFINITY;
        double rel = std::isfinite(a) && std::isfinite(b) ? d / std::max(std::max(fabs(a), fabs(b)), 2.2250738585072014e-308) : INFINITY;
        ma = std::max(ma, d); mr = std::max(mr, rel);
        if (!std::isfinite(d) || d > atol + rtol * fabs(b)) bad++;
    }
    if (bad) failures++;
    printf("%-25s %-23s abs=%.9g rel=%.9g bits=%zu/%zu%s\n", cs.UTF8String,
        tensor.UTF8String, ma, mr, ref.size() - bits, ref.size(), bad ? " FAIL" : "");
    if (!results[cs]) results[cs] = [NSMutableDictionary dictionary];
    results[cs][tensor] = @{@"max_abs": @(ma), @"max_rel": @(mr), @"mismatch_count": @(bad),
        @"bitwise_equal": @(bits == 0), @"elements": @(ref.size()), @"atol": @(atol), @"rtol": @(rtol)};
}
static void output(const Case& c, NSString* key, const float* p, double atol=2e-6, double rtol=2e-6) {
    compare(c.dir.lastPathComponent, key, p, c.read(key), atol, rtol);
}
static void check_outputs(const Case& c) {
    for (NSString* k in c.tensors) {
        if ([c.tensors[k][@"role"] isEqual:@"output"])
            require(results[c.dir.lastPathComponent][k] != nil, "unvalidated golden output");
    }
}

static void advantage_case(const Case& c) {
    Buffers b;
    int B = c.I(@"B"), T = c.I(@"T");
    float *v = b.put(c.read(@"values")), *r = b.put(c.read(@"rewards")), *d = b.put(c.read(@"terminals"));
    float *imp = c.I(@"importance_is_null") ? nullptr : b.put(c.read(@"importance"));
    float *adv = b.alloc(B*T), *ret = b.alloc(B*T);
    auto run = [&](float* returns) {
        PUF_LAUNCH(puff_advantage, (B+63)/64, 64, 0, 0)(v, r, d, imp, adv, returns,
            c.F(@"gamma"), c.F(@"lambda"), c.F(@"rho_clip"), c.F(@"c_clip"), B, T);
        cudaDeviceSynchronize();
    };
    run(ret); output(c, @"advantages", adv,0,0); output(c, @"returns", ret,0,0);
    // The native train call writes returns over values, including at segment boundaries.
    run(v); compare(c.dir.lastPathComponent,@"returns_in_place",v,c.read(@"returns"),0,0);
    compare(c.dir.lastPathComponent,@"advantages_in_place",adv,c.read(@"advantages"),0,0);
    check_outputs(c);
}

static void ppo_case(const Case& c) {
    Buffers b;
    auto dec = c.read(@"decoder"), actions = c.read(@"actions");
    auto sizes = c.read<int>(@"act_sizes");
    int B = c.I(@"B"), T = c.I(@"T"), NT = B*T, A = [c.shape(@"decoder")[2] intValue]-1;
    g_num_atns = (int)sizes.size();
    float *logits = b.put(dec), *act = b.put(actions), *oldlp = b.put(c.read(@"old_logprobs"));
    float *mask = b.put(c.read(@"mask")), *lp = b.alloc(NT*A), *newlp = b.alloc(NT), *imp = b.alloc(NT), *v = b.alloc(NT);
    int* sz = b.put(sizes);
    Prec decoder{logits, {B,T,A+1}}, empty{};
    PUF_LAUNCH(cache_imp_and_v, grid(NT), 256, 0, 0)(decoder, act, oldlp, mask, empty, sz, imp, v, lp, newlp);
    cudaDeviceSynchronize();
    output(c, @"cached_logps", lp,1e-6,2e-7); output(c, @"cached_new_logprobs", newlp,1e-6,4e-7);
    output(c, @"importance", imp); output(c, @"live_values", v, 0, 0);
    float *parts = b.alloc(grid(NT)*8), *loss = b.put(c.read(@"losses_in"));
    float* ec = b.put(std::vector<float>{c.F(@"ent_coef")});
    PPOKernelArgs a{lp, nullptr, newlp, logits, nullptr, logits+A, sz, mask,
        g_num_atns, c.F(@"clip_coef"), c.F(@"vf_clip_coef"), c.F(@"vf_coef"), ec, T, A, B, false};
    PPOGraphArgs g{imp, act, oldlp, b.put(c.read(@"advantages")), b.put(c.read(@"old_values")), b.put(c.read(@"returns"))};
    PUF_LAUNCH(ppo_loss_compute, grid(NT), 256, 0, 0)(parts, a, g);
    PUF_LAUNCH(ppo_loss_reduce, 1, 8, 0, 0)(loss, parts, grid(NT));
    cudaDeviceSynchronize();
    output(c, @"dlogits", lp,NT>256?1e-9:8e-7,2e-6); output(c, @"dvalue", newlp,0,0);
    output(c, @"loss_partials", parts,2e-7,5e-7); output(c, @"losses", loss,2e-7,5e-7);
    // Verify accumulation, including count, from a nonzero accumulator twice.
    PUF_LAUNCH(ppo_loss_reduce, 1, 8, 0, 0)(loss, parts, grid(NT));
    cudaDeviceSynchronize();
    auto expected = c.read(@"losses_in");
    for (int j = 0; j < 8; j++) { float sum=0; for (int k=0; k<grid(NT); k++) sum += parts[k*8+j]; expected[j] += sum; expected[j] += sum; }
    expected[8] += 2;
    compare(c.dir.lastPathComponent, @"losses_twice_host", loss, expected, 0, 0);
    check_outputs(c);
}

// Exercise the cuBLAS shim with exactly the row-major op/alpha/beta calls of
// muon_step. No alternate transpose-to-wide optimizer path.
static void mm(cublasHandle_t h, bool ta, bool tb, int M, int N, int K,
        float* A, float* B, float* C, float alpha=1, float beta=0) {
    auto status = cublasGemmEx(h, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N,
        N, M, K, &alpha, B, CUDA_R_32F, tb ? K : N, A, CUDA_R_32F, ta ? M : K,
        &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    require(status == CUBLAS_STATUS_SUCCESS, "Muon GEMM failed");
}
// algo.cu muon_step's unfused per-matrix tail: norm, normalize, 5 NS steps, store.
static void ref_ns_tail(Buffers& b, cublasHandle_t h, float* x, int R, int C,
        const float* coeff, float eps, float* partials, float* nsnorm) {
    int ne=R*C, M=std::min(R,C);
    float *xb=b.alloc(ne), *gram=b.alloc(M*M), *gb=b.alloc(M*M);
    int nblk=std::min(grid(ne),256);
    PUF_LAUNCH(muon_sum_sq_partials, nblk, 256, 0, 0)(partials, x, ne);
    PUF_LAUNCH(muon_sum_sq_reduce, 1, 256, 0, 0)(nsnorm, partials, nblk);
    PUF_LAUNCH(muon_l2_normalize, grid(ne), 256, 0, 0)(x, nsnorm, eps, ne);
    for (int i=0; i<5; i++) {
        float *src=i%2 ? xb : x, *dst=i%2 ? x : xb;
        if (R>C) mm(h,true,false,C,C,R,src,src,gram);
        else mm(h,false,true,R,R,C,src,src,gram);
        cudaMemcpyAsync(gb,gram,M*M*4,cudaMemcpyDeviceToDevice,0);
        mm(h,false,false,M,M,M,gram,gram,gb,coeff[i*3+2],coeff[i*3+1]);
        cudaMemcpyAsync(dst,src,ne*4,cudaMemcpyDeviceToDevice,0);
        if (R>C) mm(h,false,false,R,C,C,src,gb,dst,1,coeff[i*3]);
        else mm(h,false,false,R,C,R,gb,src,dst,1,coeff[i*3]);
    }
    float scale=sqrtf(fmaxf(1,(float)R/(float)C));
    PUF_LAUNCH(muon_store_update, grid(ne), 256, 0, 0)(x, xb, scale, ne);
}

// gemm.h muon_ns5_batch_rm on a copy of the flat gradient must reproduce
// ref_ns_tail on every matrix bit for bit (and leave the rest untouched).
static void check_batch(const char* what, const float* ref, float* batch, size_t n,
        const std::vector<puf_mtl::MuonNSMatrix>& mats, const float* coeff, float eps) {
    float cf[15]; for (int i=0;i<15;i++) cf[i]=coeff[i];
    bool ran=puf_mtl::muon_ns5_batch_rm(nullptr, mats.data(), (int)mats.size(), cf, eps);
    cudaDeviceSynchronize();
    size_t bad=0; for (size_t i=0;i<n;i++) bad += memcmp(ref+i, batch+i, 4) != 0;
    printf("muon batch %-22s matrices=%zu ran=%d differing=%zu\n", what, mats.size(), (int)ran, bad);
    require(ran && bad==0, "muon_ns5_batch_rm differs from the unfused tail");
}

static void muon_case(const Case& c) {
    Buffers b;
    auto grads = c.read(@"grads"); int n = (int)grads.size(), nb = std::min(grid(n),256);
    float *gc=b.put(grads), *wb=b.put(c.read(@"params_in")), *mb=b.put(c.read(@"momentum_in"));
    float *partials=b.alloc(256), *norm=b.alloc(1), *nsnorm=b.alloc(1);
    float* lr=b.put(std::vector<float>{c.F(@"lr")});
    // Isolate the weight kernel from NS GEMM drift with CUDA's saved update.
    float *isolated_w=b.put(c.read(@"params_in")), *cuda_update=b.put(c.read(@"update"));
    PUF_LAUNCH(muon_weight_update,grid(n),256,0,0)(isolated_w,cuda_update,lr,c.F(@"weight_decay"),n);
    cudaDeviceSynchronize();
    compare(c.dir.lastPathComponent,@"weights_from_cuda_update",isolated_w,c.read(@"params_out"),0,0);
    PUF_LAUNCH(muon_sum_sq_partials, nb, 256, 0, 0)(partials, gc, n);
    PUF_LAUNCH(muon_sum_sq_reduce, 1, 256, 0, 0)(norm, partials, nb);
    PUF_LAUNCH(muon_clip_nesterov, grid(n), 256, 0, 0)(mb, gc, norm,
        c.F(@"max_grad_norm"), c.F(@"clip_eps"), c.F(@"momentum"), n);
    auto coeff = c.read(@"ns_coefficients");
    cublasHandle_t h; cublasCreate(&h);
    void* workspace = b.alloc<unsigned char>(4*1024*1024);
    cublasSetStream(h, nullptr); cublasSetWorkspace(h, workspace, 4*1024*1024);
    cudaDeviceSynchronize();
    float* gc_batch=b.put(std::vector<float>(gc, gc+n));
    std::vector<puf_mtl::MuonNSMatrix> mats;
    bool all_eligible=true;
    for (int p=0; ; p++) {
        NSString* name=[NSString stringWithFormat:@"param_%d_in",p];
        if (!c.tensors[name]) break;
        NSArray* shape=c.shape(name);
        if (shape.count < 2) continue;
        int R=[shape[0] intValue], C=1;
        for (NSUInteger d=1; d<shape.count; d++) C *= [shape[d] intValue];
        int off=c.I([NSString stringWithFormat:@"param_%d_offset",p]);
        ref_ns_tail(b, h, gc+off, R, C, coeff.data(), c.F(@"ns_eps"), partials, nsnorm);
        all_eligible = all_eligible && puf_mtl::muon_ns5_eligible(R, C);
        mats.push_back({gc_batch+off, R, C, sqrtf(fmaxf(1,(float)R/(float)C))});
    }
    PUF_LAUNCH(muon_weight_update, grid(n), 256, 0, 0)(wb, gc, lr, c.F(@"weight_decay"), n);
    cudaDeviceSynchronize();
    if (all_eligible && mats.size() <= 8)
        check_batch(c.dir.lastPathComponent.UTF8String, gc, gc_batch, n, mats, coeff.data(), c.F(@"ns_eps"));
    bool all_wide = !all_eligible && !mats.empty() && mats.size() <= 8; size_t wide_bytes=0;
    for (const auto& mat : mats) {
        all_wide = all_wide && puf_mtl::muon_ns_wide_eligible(mat.R, mat.C);
        wide_bytes += puf_mtl::muon_ns_wide_scratch_bytes(mat.R, mat.C);
    }
    if (all_wide) {
        void* scratch=b.alloc<unsigned char>(wide_bytes);
        bool ran=puf_mtl::muon_ns_wide_rm(nullptr,mats.data(),(int)mats.size(),coeff.data(),
            c.F(@"ns_eps"),scratch,wide_bytes);
        cudaDeviceSynchronize();
        require(ran,"eligible golden wide Muon rejected");
        size_t bad=0; for (int i=0;i<n;i++) bad += memcmp(gc+i,gc_batch+i,4)!=0;
        require(bad==0,"golden wide Muon differs from generic tail");
        compare(c.dir.lastPathComponent,@"wide_update_cuda",gc_batch,c.read(@"update"),2e-6,2e-6);
    }
    output(c,@"grad_sum_sq",norm,0,0); output(c,@"momentum_out",mb,0,0);
    // Five NS iterations amplify differences from CUDA/cuBLAS accumulation.
    output(c,@"update",gc,2e-6,2e-6); output(c,@"params_out",wb,1e-7,2e-6);
    for (int p=0; ; p++) {
        NSString* key=[NSString stringWithFormat:@"param_%d_out",p]; if (!c.tensors[key]) break;
        int off=c.I([NSString stringWithFormat:@"param_%d_offset",p]); output(c,key,wb+off,1e-7,2e-6);
    }
    check_outputs(c); cublasDestroy(h);
}

// CUDA's exact reduction tree, used for isolated synthetic norm/log tests.
static float tree(std::vector<float> x) {
    for (size_t s=x.size()/2;s;s>>=1) for (size_t i=0;i<s;i++) x[i]+=x[i+s];
    return x[0];
}
// Odd eligible shapes (tall/wide, 1-row/col, ragged 256-lane partials, up
// to 16 blocks, 1D gaps), batched vs unfused; and ineligible batches.
static void muon_batch_synthetic() {
    static const float coeff[15] = {4.0848f, -6.8946f, 2.9270f, 3.9505f, -6.3029f, 2.6377f,
        3.7418f, -5.5913f, 2.3037f, 2.8769f, -3.1427f, 1.2046f, 2.8366f, -3.0525f, 1.2012f};
    struct { int R, C; } shapes[] = {{32,118},{96,32},{5,32},{1,5},{7,1},{31,129},{2047,2},{64,17}};
    Buffers b;
    cublasHandle_t h; cublasCreate(&h);
    void* workspace = b.alloc<unsigned char>(4*1024*1024);
    cublasSetStream(h, nullptr); cublasSetWorkspace(h, workspace, 4*1024*1024);
    float *partials=b.alloc(256), *nsnorm=b.alloc(1);
    long n=0; std::vector<long> offs;
    for (auto& sh : shapes) { offs.push_back(n); n += (long)sh.R*sh.C + 3; }  // 3-float 1D gap
    std::vector<float> init(n);
    uint32_t r=12345; for (auto& v : init) { r=r*1664525u+1013904223u; v=((int)(r>>8)%2001-1000)*1e-3f; }
    init[offs[3]]=0.0f;  // includes an exact zero
    float *ref=b.put(init), *bat=b.put(init);
    std::vector<puf_mtl::MuonNSMatrix> mats;
    for (size_t i=0;i<sizeof(shapes)/sizeof(shapes[0]);i++) {
        int R=shapes[i].R, C=shapes[i].C;
        ref_ns_tail(b, h, ref+offs[i], R, C, coeff, 1e-7f, partials, nsnorm);
        mats.push_back({bat+offs[i], R, C, sqrtf(fmaxf(1,(float)R/(float)C))});
    }
    cudaDeviceSynchronize();
    check_batch("synthetic_8_shapes", ref, bat, n, mats, coeff, 1e-7f);
    // Ineligible: 9 matrices, or one too-large matrix -> nothing encoded.
    std::vector<puf_mtl::MuonNSMatrix> nine(9, mats[0]), big = {mats[0], {bat, 33, 64, 1.0f}};
    std::vector<float> before(bat, bat+n);
    bool r9=puf_mtl::muon_ns5_batch_rm(nullptr, nine.data(), 9, coeff, 1e-7f);
    bool rb=puf_mtl::muon_ns5_batch_rm(nullptr, big.data(), 2, coeff, 1e-7f);
    cudaDeviceSynchronize();
    require(!r9 && !rb && memcmp(before.data(), bat, n*4)==0, "ineligible muon batch encoded work");
    printf("muon batch ineligible (9 matrices / 33x64) not encoded: %s\n", !r9 && !rb ? "ok" : "FAIL");
    cublasDestroy(h);
}

// gemm.h muon_ns_wide_rm (matrices too large for muon_ns5_batch): tensor_ops
// grams (tall tn / wide nt / square, 16-byte aligned or not, which flips
// the unfused path's gram to tiled on even steps), tiled grams with K % 16
// != 0, a 2047-row matrix; batched vs ref_ns_tail bit for bit; then
// ineligible / short-scratch calls encode nothing.
static void muon_wide_synthetic() {
    static const float coeff[15] = {4.0848f, -6.8946f, 2.9270f, 3.9505f, -6.3029f, 2.6377f,
        3.7418f, -5.5913f, 2.3037f, 2.8769f, -3.1427f, 1.2046f, 2.8366f, -3.0525f, 1.2012f};
    struct { int R, C, misalign; } shapes[] = {{192,64,0},{64,192,0},{128,128,0},{100,40,0},
        {40,100,1},{2047,3,0},{192,64,1},{33,64,0}};
    Buffers b;
    cublasHandle_t h; cublasCreate(&h);
    void* workspace = b.alloc<unsigned char>(4*1024*1024);
    cublasSetStream(h, nullptr); cublasSetWorkspace(h, workspace, 4*1024*1024);
    float *partials=b.alloc(256), *nsnorm=b.alloc(1);
    long n=0; std::vector<long> offs;
    size_t ws_bytes=0;
    for (auto& sh : shapes) {
        n = (n + 3) / 4 * 4 + sh.misalign;  // 16-byte aligned unless misalign
        offs.push_back(n); n += (long)sh.R*sh.C + 3;
        ws_bytes += puf_mtl::muon_ns_wide_scratch_bytes(sh.R, sh.C);
    }
    std::vector<float> init(n);
    uint32_t r=777; for (auto& v : init) { r=r*1664525u+1013904223u; v=((int)(r>>8)%2001-1000)*1e-3f; }
    float *ref=b.put(init), *bat=b.put(init);
    unsigned char* ws=b.alloc<unsigned char>(ws_bytes);
    std::vector<puf_mtl::MuonNSMatrix> mats;
    for (size_t i=0;i<sizeof(shapes)/sizeof(shapes[0]);i++) {
        int R=shapes[i].R, C=shapes[i].C;
        require(puf_mtl::muon_ns_wide_eligible(R, C), "muon wide shape ineligible");
        ref_ns_tail(b, h, ref+offs[i], R, C, coeff, 1e-7f, partials, nsnorm);
        mats.push_back({bat+offs[i], R, C, sqrtf(fmaxf(1,(float)R/(float)C))});
    }
    cudaDeviceSynchronize();
    std::vector<float> before(bat, bat+n);
    bool short_ws=puf_mtl::muon_ns_wide_rm(nullptr, mats.data(), (int)mats.size(), coeff, 1e-7f,
        ws, ws_bytes-16);
    std::vector<puf_mtl::MuonNSMatrix> big = {mats[0], {bat, 2048, 2, 1.0f}};
    bool too_big=puf_mtl::muon_ns_wide_rm(nullptr, big.data(), 2, coeff, 1e-7f, ws, ws_bytes);
    cudaDeviceSynchronize();
    require(!short_ws && !too_big && memcmp(before.data(), bat, n*4)==0,
        "ineligible muon wide batch encoded work");
    printf("muon wide ineligible (short scratch / 2048x2) not encoded: ok\n");
    bool ran=puf_mtl::muon_ns_wide_rm(nullptr, mats.data(), (int)mats.size(), coeff, 1e-7f,
        ws, ws_bytes);
    cudaDeviceSynchronize();
    size_t bad=0; for (long i=0;i<n;i++) bad += memcmp(ref+i, bat+i, 4) != 0;
    printf("muon wide synthetic_8_shapes      matrices=%zu ran=%d differing=%zu\n",
        mats.size(), (int)ran, bad);
    require(ran && bad==0, "muon_ns_wide_rm differs from the unfused tail");
    cublasDestroy(h);
}

static void muon_edges() {
    Buffers b;
    int n=200003, nb=256;
    std::vector<float> in(n); for (int i=0;i<n;i++) in[i]=(i%29-14)*0.125f;
    float *src=b.put(in), *p=b.alloc(nb), *sum=b.alloc(1);
    std::vector<float> ref(nb);
    for (int block=0;block<nb;block++) {
        std::vector<float> sh(256);
        for(int t=0;t<256;t++) for(int i=block*256+t;i<n;i+=256*nb) sh[t]=fmaf(in[i],in[i],sh[t]);
        ref[block]=tree(sh);
    }
    PUF_LAUNCH(muon_sum_sq_partials,nb,256,0,0)(p,src,n);
    PUF_LAUNCH(muon_sum_sq_reduce,1,256,0,0)(sum,p,nb);
    cudaDeviceSynchronize();
    compare(@"synthetic_muon",@"partials_grid_stride",p,ref,0,0);
    compare(@"synthetic_muon",@"sum_sq_256_partials",sum,{tree(ref)},0,0);
    // Epsilon, clipping/no clipping, nonzero decay, guards on partial blocks.
    for (float ss : {0.0f, 0.01f, 400.0f}) {
        std::vector<float> grad(259), mom(259), weight(259), clipped(259), mout(259), wout(259);
        for(int i=0;i<259;i++) { grad[i]=(i%11-5)*.01f; mom[i]=(i%7-3)*.1f; weight[i]=i*.01f; }
        float *g=b.put(grad), *m=b.put(mom), *w=b.put(weight), *s=b.put(std::vector<float>{ss}), *lr=b.put(std::vector<float>{.03f});
        float coef=fminf(.7f/(sqrtf(ss)+1e-6f),1.0f);
        for(int i=0;i<259;i++) { float gg=grad[i]*coef; mout[i]=fmaf(grad[i],coef,.87f*mom[i]); clipped[i]=fmaf(.87f,mout[i],gg); }
        PUF_LAUNCH(muon_clip_nesterov,2,256,0,0)(m,g,s,.7f,1e-6f,.87f,259);
        PUF_LAUNCH(muon_weight_update,2,256,0,0)(w,g,lr,.2f,259);
        cudaDeviceSynchronize();
        for(int i=0;i<259;i++) wout[i]=fmaf(weight[i],fmaf(-.03f,.2f,1.0f),-(.03f*clipped[i]));
        compare(@"synthetic_muon",@"clipped_nesterov",g,clipped,1e-7,1e-6);
        compare(@"synthetic_muon",@"momentum",m,mout,1e-7,1e-6);
        compare(@"synthetic_muon",@"weight_decay",w,wout,1e-7,1e-6);
        float inv=1/fmaxf(sqrtf(ss),1e-7f); for(float& x:clipped)x*=inv;
        PUF_LAUNCH(muon_l2_normalize,2,256,0,0)(g,s,1e-7f,259);
        cudaDeviceSynchronize(); compare(@"synthetic_muon",@"normalize",g,clipped,1e-7,2e-6);
    }
}

struct TestLog { float score, n, reward, length, perf; }; // n intentionally in the middle
struct TestEnv { uint64_t sentinel; TestLog log; int num_agents; float tail[7]; };
static void log_edges() {
    Buffers b; int ne=1027;
    std::vector<TestEnv> envs(ne);
    for(int i=0;i<ne;i++) {
        envs[i].sentinel=0x12345678abcdef01ull; envs[i].num_agents=3;
        envs[i].log={i*.125f,i%7==0?0.0f:2.0f,float(i%17),float(i%31),-.25f*i};
        for(float& x:envs[i].tail)x=987;
    }
    TestEnv* e=b.put(envs); float* out=b.alloc(5);
    std::vector<float> ref(5);
    for(int c=0;c<5;c++) {
        std::vector<float> sh(256);
        for(int t=0;t<256;t++) for(int i=t;i<ne;i+=256) if(envs[i].log.n!=0) sh[t]+=reinterpret_cast<float*>(&envs[i].log)[c];
        ref[c]=tree(sh);
    }
    for(int clear : {0,1,1}) {
        PUF_LAUNCH(log_reduce,1,256,5*256*sizeof(float),0)(e,out,ne*3+2,clear);
        cudaDeviceSynchronize(); compare(@"synthetic_log",clear?@"clear":@"retain",out,ref,0,0);
        if(clear) { for(auto& v:envs) memset(&v.log,0,sizeof(v.log)); std::fill(ref.begin(),ref.end(),0); }
        require(memcmp(e,envs.data(),envs.size()*sizeof(TestEnv))==0,"log clear or adjacent Env fields changed");
    }
}

// Host PPO equations use doubles for an independent reference. Deliberately
// cover VF clipping winner/tie, both PG clip directions, multihead masks and
// continuous clamps. The cache's raw-action behavior is checked separately.
static double finite_clamp(double x,double lo,double hi) { return std::isnan(x)?0:std::min(hi,std::max(lo,x)); }
static void ppo_edges(bool continuous) {
    Buffers b; int NT=259, heads=2, A=continuous?2:5, fused=A+1;
    g_num_atns=heads;
    std::vector<float> logits(NT*fused), actions(NT*heads), mask(NT*A,1), oldlp(NT), oldv(NT), ret(NT), adv(NT);
    std::vector<int> sizes{2,3}; std::vector<float> ls{-.3f,.4f};
    std::vector<float> cached(NT*A,123), newlp(NT), importance(NT), live(NT), dl(NT*A,123), dv(NT), dls(NT*heads), losses(9);
    std::vector<double> logps(NT*A);
    float ec=.03f, clip=.2f,vclip=.5f,vcoef=.7f;
    for(int i=0;i<NT;i++) {
        for(int j=0;j<A;j++) logits[i*fused+j]=.1f*(j+1)-.02f*(i%7);
        logits[i*fused+A]=(i%5-2)*.7f;
        oldv[i]=0; ret[i]=(i%3-1)*1.2f; adv[i]=(i%2?1:-1)*(.3f+i%4);
        actions[i*heads]=continuous?.12f:float(i%2); actions[i*heads+1]=continuous?-.42f:float(i%3);
        // Keep selected actions valid except a deliberate all-masked head.
        if(!continuous) { mask[i*A+(int(actions[i*heads])+1)%2]=0; if(i%9==0) for(int j=2;j<5;j++)mask[i*A+j]=0; }
        double lp=0;
        if(continuous) {
            if(i==1) logits[i*fused]=NAN;
            if(i==2) logits[i*fused+1]=INFINITY;
            for(int h=0;h<heads;h++) {
                double mean=finite_clamp(logits[i*fused+h],-1e6,1e6), norm=(actions[i*heads+h]-mean)/exp(ls[h]);
                lp+=-.5*norm*norm-.9189385332046727-double(ls[h]);
            }
        } else {
            int off=0;
            for(int h=0;h<heads;h++) {
                double mx=-INFINITY, sum=0;
                for(int j=0;j<sizes[h];j++)mx=std::max(mx,double(mask[i*A+off+j]==0?-1e4f:logits[i*fused+off+j]));
                for(int j=0;j<sizes[h];j++)sum+=exp((mask[i*A+off+j]==0?-1e4f:logits[i*fused+off+j])-mx);
                double lse=mx+log(sum);
                for(int j=0;j<sizes[h];j++) { double v=(mask[i*A+off+j]==0?-1e4f:logits[i*fused+off+j])-lse; logps[i*A+off+j]=v; cached[i*A+off+j]=v; }
                lp+=logps[i*A+off+int(actions[i*heads+h])]; off+=sizes[h];
            }
        }
        newlp[i]=lp; live[i]=logits[i*fused+A];
        oldlp[i]=float(lp)-logf(i%3==0?.5f:i%3==1?1.0f:2.0f);
    }
    float *dec=b.put(logits), *act=b.put(actions), *old=b.put(oldlp), *msk=b.put(mask), *lstd=continuous?b.put(ls):nullptr;
    float *cache=b.put(std::vector<float>(NT*A,123)), *nlp=b.alloc(NT), *imp=b.alloc(NT), *v=b.alloc(NT);
    int* sz=b.put(sizes); Prec decoder{dec,{NT,1,fused}}, logstd{lstd,{heads}}, empty{};
    PUF_LAUNCH(cache_imp_and_v,grid(NT),256,0,0)(decoder,act,old,msk,continuous?logstd:empty,sz,imp,v,cache,nlp);
    cudaDeviceSynchronize();
    NSString* cs=continuous?@"synthetic_ppo_continuous":@"synthetic_ppo_discrete";
    if(!continuous) compare(cs,@"cache",cache,cached,6e-4,2e-6);
    compare(cs,@"new_lp",nlp,newlp,3e-5,3e-6); compare(cs,@"live_values",v,live,0,0);
    // Use the actual cached FP32 logps/lp as inputs to the independent loss reference.
    // This isolates loss branch semantics from CUDA/Metal transcendental error.
    for(int i=0;i<NT;i++) {
        double lr=double(nlp[i])-oldlp[i], ratio=exp(lr), wa=-double(adv[i]);
        double pg1=wa*ratio, pg2=wa*std::max(1.0-clip,std::min(1.0+clip,ratio)), pg=std::max(pg1,pg2);
        double dr=wa/NT; if(pg2>pg1&&(ratio<=1.0-clip||ratio>=1.0+clip))dr=0;
        double dlp=dr*ratio, pred=live[i], vc=std::max(-double(vclip),std::min(double(vclip),pred));
        double vu=(pred-ret[i])*(pred-ret[i]), vlc=(vc-ret[i])*(vc-ret[i]);
        double vl=.5*std::max(vu,vlc), ent=0;
        dv[i]=vcoef/NT*(vlc>vu?0:pred-ret[i]);
        if(continuous) {
            for(int h=0;h<heads;h++) {
                double mean=finite_clamp(logits[i*fused+h],-1e6,1e6), diff=actions[i*heads+h]-mean, var=exp(2*double(ls[h]));
                ent+=1.4189385332046727+ls[h]; dl[i*A+h]=dlp*diff/var;
                dls[i*heads+h]=dlp*(diff*diff/var-1)-double(ec)/NT;
            }
        } else {
            int off=0;
            for(int h=0;h<heads;h++) {
                double he=0;for(int j=0;j<sizes[h];j++) {double lp=cache[i*A+off+j];he-=exp(lp)*lp;}
                ent+=he;
                for(int j=0;j<sizes[h];j++) {double lp=cache[i*A+off+j],p=exp(lp);dl[i*A+off+j]=((j==int(actions[i*heads+h])?1:0)-p)*dlp-double(ec)/NT*p*(-he-lp);}
                off+=sizes[h];
            }
        }
        double terms[8]={pg,vl,ent,pg+vcoef*vl-ec*ent,-lr,(ratio-1)-lr,fabs(ratio-1)>clip?1.:0.,ratio};
        for(int j=0;j<8;j++)losses[j]+=terms[j]/NT;
    }
    losses[8]=1;
    float *parts=b.alloc(grid(NT)*8), *loss=b.alloc(9), *gls=continuous?b.alloc(NT*heads):nullptr;
    PPOKernelArgs a{cache,gls,nlp,dec,lstd,dec+A,sz,msk,heads,clip,vclip,vcoef,b.put(std::vector<float>{ec}),1,A,NT,continuous};
    PPOGraphArgs g{imp,act,old,b.put(adv),b.put(oldv),b.put(ret)};
    PUF_LAUNCH(ppo_loss_compute,grid(NT),256,0,0)(parts,a,g);
    PUF_LAUNCH(ppo_loss_reduce,1,8,0,0)(loss,parts,grid(NT));
    cudaDeviceSynchronize();
    compare(cs,@"dlogits",cache,dl,2e-6,5e-6); compare(cs,@"dvalue",nlp,dv,2e-6,5e-6);
    if(continuous)compare(cs,@"dlogstd",gls,dls,2e-6,5e-6);
    compare(cs,@"losses",loss,losses,3e-5,1e-5);
}

// Cache intentionally propagates raw nonfinite actions; the continuous loss
// clamps them. Isolate the loss by resetting the cached log ratio to zero.
static void continuous_guards() {
    int which=0;
    for(float raw_ls : {NAN,INFINITY,-INFINITY,3.0f,-21.0f,.5f}) {
        Buffers b; g_num_atns=1; constexpr int n=6;
        std::vector<float> means{NAN,INFINITY,-INFINITY,2e6f,-2e6f,.75f};
        std::vector<float> actions{NAN,INFINITY,-INFINITY,2e6f,-2e6f,.25f};
        std::vector<float> dec(n*2), cache_ref(n), dl(n), dls(n), loss_ref(9), zero(n,0), one(n,1);
        float ls=finite_clamp(raw_ls,-20,2), std=expf(ls), var=std*std, ec=.03f;
        for(int i=0;i<n;i++) {
            dec[i*2]=means[i];
            float mean=finite_clamp(means[i],-1e6,1e6), norm=(actions[i]-mean)/std;
            cache_ref[i]=-.5f*norm*norm-.9189385332046727f-ls;
            float diff=float(finite_clamp(actions[i],-1e6,1e6))-mean;
            dl[i]=(-1.0f/n)*diff/var;
            dls[i]=fmaf(-1.0f/n,diff*diff/var-1,-ec/n);
        }
        float *decoder=b.put(dec),*act=b.put(actions),*old=b.put(zero),*logstd=b.put(std::vector<float>{raw_ls});
        float *gl=b.alloc(n),*gv=b.alloc(n),*gs=b.alloc(n),*imp=b.alloc(n),*v=b.alloc(n),*parts=b.alloc(8),*loss=b.alloc(9);
        Prec d{decoder,{n,1,2}}, l{logstd,{1}}; int* sizes=b.put(std::vector<int>{1});
        PUF_LAUNCH(cache_imp_and_v,1,256,0,0)(d,act,old,(float*)nullptr,l,sizes,imp,v,gl,gv);
        cudaDeviceSynchronize();
        NSString* cs=[NSString stringWithFormat:@"continuous_guards_%d",which++];
        compare(cs,@"raw_action_newlp",gv,cache_ref,1e-6,2e-6);
        for(int i=0;i<n;i++)gv[i]=0;
        PPOKernelArgs a{gl,gs,gv,decoder,logstd,decoder+1,sizes,nullptr,1,.2f,.5f,.7f,b.put(std::vector<float>{ec}),1,1,n,true};
        PPOGraphArgs g{imp,act,old,b.put(one),b.put(zero),b.put(zero)};
        PUF_LAUNCH(ppo_loss_compute,1,256,0,0)(parts,a,g);
        PUF_LAUNCH(ppo_loss_reduce,1,8,0,0)(loss,parts,1);
        cudaDeviceSynchronize();
        compare(cs,@"clamped_dmean",gl,dl,2e-7,5e-6);
        compare(cs,@"clamped_dlogstd",gs,dls,2e-7,5e-6);
        float ent=1.4189385332046727f+ls;
        loss_ref={-1,0,ent,fmaf(-ec,ent,-1),0,0,0,1,1};
        compare(cs,@"losses",loss,loss_ref,2e-7,3e-6);
    }
}

static void timings(NSString* root);
int main(int argc,char** argv) { @autoreleasepool {
    results=[NSMutableDictionary dictionary];
    std::vector<NSString*> roots; bool timing=false;
    for(int i=1;i<argc;i++) { if(strcmp(argv[i],"--timings")==0)timing=true; else roots.push_back([NSString stringWithUTF8String:argv[i]]); }
    if(roots.empty())roots.push_back(@"../golden/run_a");
    for(NSString* root:roots) {
        printf("Golden root: %s\n",root.UTF8String);
        NSArray* names=[[[NSFileManager defaultManager] contentsOfDirectoryAtPath:root error:nil]
            sortedArrayUsingSelector:@selector(compare:)];
        int cases=0;
        for (NSString* name in names) {
            if ([name hasPrefix:@"adv_"]) { advantage_case(load(root,name)); cases++; }
            else if ([name hasPrefix:@"ppo_"]) { ppo_case(load(root,name)); cases++; }
            else if ([name hasPrefix:@"muon_"]) { muon_case(load(root,name)); cases++; }
        }
        require(cases>0,"no advantage/PPO/Muon golden cases found");
    }
    muon_edges(); muon_batch_synthetic(); muon_wide_synthetic(); log_edges(); ppo_edges(false); ppo_edges(true); continuous_guards();
    if(timing)timings(roots[0]);
    NSData* json=[NSJSONSerialization dataWithJSONObject:results options:NSJSONWritingPrettyPrinted|NSJSONWritingSortedKeys error:nil];
    require(json!=nil,"report JSON serialization failed");
    require([json writeToFile:@"build/metal/test_ppo_muon/report.json" atomically:YES],"report write failed");
    printf("%s (%d failures)\n",failures?"FAIL":"PASS",failures);
    return failures?1:0;
} }

// Independent Metal 4 timestamp harness: no runtime changes and no CPU encode,
// submit or wait time in the measured interval. Each isolated dispatch restores
// its mutable inputs on the CPU after the preceding sample has completed.
struct GpuTimer {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLLibrary> lib;
    id<MTL4CommandQueue> queue;
    id<MTL4CommandAllocator> allocator;
    id<MTL4ArgumentTable> table;
    id<MTLSharedEvent> event;
    id<MTLResidencySet> residency;
    id<MTL4CounterHeap> counters;
    std::vector<id<MTLBuffer>> buffers;
    uint64_t serial=0;
    GpuTimer() {
        NSError* error=nil;
        lib=[dev newLibraryWithURL:[NSURL fileURLWithPath:@"build/metal/test_ppo_muon/puffer.metallib"] error:&error];
        queue=[dev newMTL4CommandQueue]; allocator=[dev newCommandAllocator]; event=[dev newSharedEvent];
        MTL4ArgumentTableDescriptor* td=[MTL4ArgumentTableDescriptor new]; td.maxBufferBindCount=1;
        table=[dev newArgumentTableWithDescriptor:td error:&error];
        MTLResidencySetDescriptor* rd=[MTLResidencySetDescriptor new]; rd.initialCapacity=64;
        residency=[dev newResidencySetWithDescriptor:rd error:&error];
        [residency requestResidency]; [queue addResidencySet:residency];
        MTL4CounterHeapDescriptor* cd=[MTL4CounterHeapDescriptor new]; cd.type=MTL4CounterHeapTypeTimestamp; cd.count=2;
        counters=[dev newCounterHeapWithDescriptor:cd error:&error];
        if(!lib||!queue||!allocator||!table||!residency||!counters) { fprintf(stderr,"GPU timer setup failed: %s\n",error.description.UTF8String); exit(2); }
        printf("GPU timestamp device: %s; median of 40 isolated dispatches after 5 warmups\n",dev.name.UTF8String);
    }
    id<MTLBuffer> alloc(size_t bytes) {
        id<MTLBuffer> buf=[dev newBufferWithLength:std::max(bytes,size_t(16)) options:MTLResourceStorageModeShared];
        if(!buf)abort(); memset(buf.contents,0,buf.length);
        buffers.push_back(buf); [residency addAllocation:buf]; [residency commit]; return buf;
    }
    template<class T> id<MTLBuffer> put(const std::vector<T>& v) { auto buf=alloc(v.size()*sizeof(T)); memcpy(buf.contents,v.data(),v.size()*sizeof(T));return buf; }
    template<class A> void measure(NSString* name,const A& args,int groups,int threads=256,size_t smem=0,
            std::function<void()> reset={}) {
        NSError* error=nil;
        auto fn=[lib newFunctionWithName:name];
        auto pipeline=[dev newComputePipelineStateWithFunction:fn error:&error];
        if(!pipeline)abort();
        auto ab=alloc(sizeof(args)); memcpy(ab.contents,&args,sizeof(args));
        std::vector<double> ticks;

        for(int i=0;i<45;i++) {
            if(reset)reset();
            uint64_t begin_tick=mach_absolute_time();
            [allocator reset];
            auto cmd=[dev newCommandBuffer]; [cmd beginCommandBufferWithAllocator:allocator];
            auto enc=[cmd computeCommandEncoder];
            [enc setArgumentTable:table]; [table setAddress:ab.gpuAddress atIndex:0];
            [enc setComputePipelineState:pipeline];
            if(smem)[enc setThreadgroupMemoryLength:smem atIndex:0];
            [enc barrierAfterQueueStages:MTLStageDispatch beforeStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
            [enc writeTimestampWithGranularity:MTL4TimestampGranularityPrecise intoHeap:counters atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
            [enc writeTimestampWithGranularity:MTL4TimestampGranularityPrecise intoHeap:counters atIndex:1];
            [enc endEncoding]; [cmd endCommandBuffer];
            const id<MTL4CommandBuffer> cmds[]={cmd}; [queue commit:cmds count:1];
            [queue signalEvent:event value:++serial];
            if(![event waitUntilSignaledValue:serial timeoutMS:10000])abort();
            NSData* data=[counters resolveCounterRange:NSMakeRange(0,2)];
            if(data.length!=2*sizeof(MTL4TimestampHeapEntry))abort();
            auto ts=static_cast<const MTL4TimestampHeapEntry*>(data.bytes);
            // MTL4 resolved timestamps on Apple silicon are CPU mach clock ticks;
            // sampleTimestamps returns nanoseconds and must not calibrate them.
            if(ts[1].timestamp<=ts[0].timestamp || ts[0].timestamp<begin_tick || ts[1].timestamp>mach_absolute_time())abort();
            if(i>=5)ticks.push_back(double(ts[1].timestamp-ts[0].timestamp));
        }
        mach_timebase_info_data_t tb; mach_timebase_info(&tb);
        double ns_per_tick=double(tb.numer)/tb.denom;
        std::sort(ticks.begin(),ticks.end());
        double us=(ticks[19]+ticks[20])*.5*ns_per_tick/1000;
        printf("TIMING %-24s %9.3f us (min %.3f, max %.3f) grid=%d block=%d\n",name.UTF8String,
            us,ticks.front()*ns_per_tick/1000,ticks.back()*ns_per_tick/1000,groups,threads);
        if(!results[@"gpu_timings"])results[@"gpu_timings"]=[NSMutableDictionary dictionary];
        results[@"gpu_timings"][name]=@{@"median_us":@(us),@"min_us":@(ticks.front()*ns_per_tick/1000),
            @"max_us":@(ticks.back()*ns_per_tick/1000),@"grid":@(groups),@"block":@(threads)};
    }
};
static void timings(NSString* root) {
    cudaDeviceSynchronize();
    GpuTimer t;
    auto c=load(root,@"ppo_breakout"); int B=c.I(@"B"),T=c.I(@"T"),NT=B*T,A=3;
    auto decoder=t.put(c.read(@"decoder")), act=t.put(c.read(@"actions")),old=t.put(c.read(@"old_logprobs"));
    auto mask=t.put(c.read(@"mask")), sizes=t.put(c.read<int>(@"act_sizes"));
    auto lp=t.put(c.read(@"cached_logps")), nlp=t.put(c.read(@"cached_new_logprobs")), imp=t.put(c.read(@"importance")),v=t.put(c.read(@"live_values"));
    auto adv=t.put(c.read(@"advantages")), ret=t.put(c.read(@"returns")), oldv=t.put(c.read(@"old_values"));
    auto ec=t.put(std::vector<float>{c.F(@"ent_coef")}), parts=t.put(c.read(@"loss_partials")), losses=t.put(c.read(@"losses_in"));
    t.measure(@"cache_imp_and_v",MtlCacheImpArgs{decoder.gpuAddress,act.gpuAddress,old.gpuAddress,mask.gpuAddress,0,sizes.gpuAddress,
        imp.gpuAddress,v.gpuAddress,lp.gpuAddress,nlp.gpuAddress,NT,A+1,1},grid(NT));
    auto cached=c.read(@"cached_logps"), newlp=c.read(@"cached_new_logprobs");
    t.measure(@"ppo_loss_compute",MtlPPOLossArgs{parts.gpuAddress,lp.gpuAddress,0,nlp.gpuAddress,decoder.gpuAddress,0,
        decoder.gpuAddress+A*4,sizes.gpuAddress,ec.gpuAddress,act.gpuAddress,old.gpuAddress,adv.gpuAddress,oldv.gpuAddress,ret.gpuAddress,
        1,c.F(@"clip_coef"),c.F(@"vf_clip_coef"),c.F(@"vf_coef"),T,A,B,0},grid(NT),256,0,[&]{
            memcpy(lp.contents,cached.data(),cached.size()*4);memcpy(nlp.contents,newlp.data(),newlp.size()*4);
        });
    t.measure(@"ppo_loss_reduce",MtlPPOReduceArgs{losses.gpuAddress,parts.gpuAddress,grid(NT)},1,8);
    auto ac=load(root,@"adv_breakout_vtrace");
    auto av=t.put(ac.read(@"values")),rew=t.put(ac.read(@"rewards")),done=t.put(ac.read(@"terminals")),importance=t.put(ac.read(@"importance"));
    t.measure(@"puff_advantage",MtlAdvantageArgs{av.gpuAddress,rew.gpuAddress,done.gpuAddress,importance.gpuAddress,adv.gpuAddress,ret.gpuAddress,
        ac.F(@"gamma"),ac.F(@"lambda"),ac.F(@"rho_clip"),ac.F(@"c_clip"),B,T},(B+63)/64,64);
    auto mc=load(root,@"muon_breakout_full");auto grad=mc.read(@"grads"),mom=mc.read(@"momentum_in");int n=(int)grad.size(),nb=std::min(grid(n),256);
    auto gc=t.put(grad),mb=t.put(mom),wb=t.put(mc.read(@"params_in")),norm=t.put(mc.read(@"grad_sum_sq"));
    auto partials=t.alloc(256*4),lr=t.put(std::vector<float>{mc.F(@"lr")});
    t.measure(@"muon_sum_sq_partials",MtlMuonSumArgs{partials.gpuAddress,gc.gpuAddress,n},nb);
    t.measure(@"muon_sum_sq_reduce",MtlMuonSumArgs{norm.gpuAddress,partials.gpuAddress,nb},1);
    t.measure(@"muon_clip_nesterov",MtlMuonClipArgs{mb.gpuAddress,gc.gpuAddress,norm.gpuAddress,
        mc.F(@"max_grad_norm"),mc.F(@"clip_eps"),mc.F(@"momentum"),n},grid(n),256,0,[&]{
            memcpy(gc.contents,grad.data(),grad.size()*4);memcpy(mb.contents,mom.data(),mom.size()*4);
        });
    // Largest Breakout matrix is encoder (32,118); per-matrix normalize/store.
    int ne=32*118;
    t.measure(@"muon_l2_normalize",MtlMuonNormalizeArgs{gc.gpuAddress,norm.gpuAddress,1e-7f,ne},grid(ne),256,0,[&]{memcpy(gc.contents,grad.data(),grad.size()*4);});
    t.measure(@"muon_store_update",MtlMuonStoreArgs{gc.gpuAddress,mb.gpuAddress,1.0f,ne},grid(ne));
    t.measure(@"muon_weight_update",MtlMuonWeightArgs{wb.gpuAddress,gc.gpuAddress,lr.gpuAddress,0.0f,n},grid(n));
    // GPU Breakout layout from ocean/breakout/breakout.cu and src/pufferenv.h.
    struct AgentLayout { uint64_t obs, actions, rewards, terminals, mask; int policy; };
    struct BreakoutLog { float perf, score, reward, length, n; };
    struct BreakoutGpuEnv {
        BreakoutLog log; AgentLayout agents[1]; int num_agents, tag, boundary;
        float paddle_x,paddle_y,ball_x,ball_y,ball_vx,ball_vy,paddle_width,ball_speed;
        int score,balls_fired,hits,num_balls,tick; unsigned rng,brick_mask[4];
    };
    static_assert(sizeof(BreakoutGpuEnv)==160 && offsetof(BreakoutGpuEnv,num_agents)==72,"Breakout GPU layout");
    std::vector<BreakoutGpuEnv> envs(4096);
    for(auto& e:envs){e.num_agents=1;e.log={1,2,3,4,1};}
    auto env=t.put(envs),out=t.alloc(5*4);
    t.measure(@"log_reduce",MtlLogReduceArgs{env.gpuAddress,out.gpuAddress,4096,0,sizeof(BreakoutGpuEnv),offsetof(BreakoutGpuEnv,log),
        offsetof(BreakoutGpuEnv,num_agents),offsetof(BreakoutLog,n),5},1,256,5*256*4);
}

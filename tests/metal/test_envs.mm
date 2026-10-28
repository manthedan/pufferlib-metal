// CPU references for shared minimal/asteroids encoders. No CUDA goldens exist.
#import <Foundation/Foundation.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "compat/cuda_runtime.h"
#include "compat/curand_kernel.h"
#include "launch.h"
#include "runtime.h"
using precision_t = float;
struct Prec { float* data; int64_t shape[8]; };
using Float = Prec;
static int64_t numel(int64_t* s) { int64_t n=1; for (int i=0;i<8 && s[i];i++) n*=s[i]; return n; }
#define NUM_ATNS 1
#include "kernels.h"

static int failures = 0;
template<class T> struct Buffer {
    T* raw; T* data; int n;
    explicit Buffer(int count): n(count) {
        if (cudaMalloc(&raw, (n+2)*sizeof(T)) != cudaSuccess) std::abort();
        data = raw+1; std::fill(raw, raw+n+2, T(12345));
    }
    ~Buffer() { cudaFree(raw); }
    void guards() { if (raw[0]!=T(12345) || raw[n+1]!=T(12345)) { puts("FAIL guard"); failures++; } }
};
static void compare(const char* what, const float* got, const std::vector<double>& ref) {
    double error=0;
    for (size_t i=0;i<ref.size();i++) {
        double e=std::abs(got[i]-ref[i]); error=std::max(error,e);
        if (!std::isfinite(got[i]) || e>2e-5*(1+std::abs(ref[i]))) { failures++; break; }
    }
    printf("  %s max_abs=%.3g\n",what,error);
}
template<class Material, class Pool, class WG, class IG>
static void check(int S, int F, int P, int B, int H, int mode) {
    int O=S+F*P+3;  // Padded observation stride must be respected.
    Buffer<float> obs(B*O), points(B*P*(S+F)), x(B*P*16), w(H*16), dy(B*H);
    Buffer<float> y(B*H), dw(H*16), dx(B*P*16);
    Buffer<int> arg(B*H);
    std::mt19937 rng(73+B+H); std::uniform_real_distribution<float> dist(-1,1);
    for (int i=0;i<obs.n;i++) obs.data[i]=dist(rng);
    for (int i=0;i<x.n;i++) x.data[i]=mode==0 ? -1.0f : mode==2 ? float(i%16)*0.05f : dist(rng);
    for (int i=0;i<w.n;i++) w.data[i]=dist(rng);
    for (int i=0;i<dy.n;i++) dy.data[i]=dist(rng);
    // Include exact ReLU zeros in the random case.
    if (mode==1) for (int i=0;i<x.n;i+=7) x.data[i]=0;
    std::vector<double> rp(points.n), ry(y.n), rw(dw.n,0), rx(dx.n,0);
    std::vector<int> ra(arg.n);
    for (int b=0;b<B;b++) for (int p=0;p<P;p++) for (int d=0;d<S+F;d++)
        rp[(b*P+p)*(S+F)+d]=obs.data[b*O+(d<S ? d : S+p*F+d-S)];
    for (int b=0;b<B;b++) for (int h=0;h<H;h++) {
        double best=-INFINITY; int which=0;
        for (int p=0;p<P;p++) {
            double v=0;
            for (int d=0;d<16;d++) v+=double(w.data[h*16+d])*std::max(0.0f,x.data[(b*P+p)*16+d]);
            if (v>best) {best=v;which=p;}
        }
        ry[b*H+h]=best; ra[b*H+h]=which;
        for (int d=0;d<16;d++) {
            int i=(b*P+which)*16+d;
            rw[h*16+d]+=double(dy.data[b*H+h])*std::max(0.0f,x.data[i]);
            if (x.data[i]>0) rx[i]+=double(dy.data[b*H+h])*w.data[h*16+d];
        }
    }
    puf_mtl::LaunchCfg c={"test_entity",dim3(1),dim3(1),0,nullptr};
    puf_mtl::Launch<Material>{c}(points.data,obs.data,B,O);
    puf_mtl::Launch<Pool>{c}(y.data,arg.data,x.data,w.data,B,H);
    puf_mtl::Launch<WG>{c}(dw.data,dy.data,x.data,arg.data,B,H);
    puf_mtl::Launch<IG>{c}(dx.data,dy.data,w.data,x.data,arg.data,B,H);
    cudaDeviceSynchronize();
    printf("S=%d F=%d P=%d B=%d H=%d mode=%d\n",S,F,P,B,H,mode);
    compare("materialize",points.data,rp); compare("max_pool",y.data,ry);
    compare("weight_gradient",dw.data,rw); compare("input_gradient",dx.data,rx);
    for (int i=0;i<arg.n;i++) if (arg.data[i]!=ra[i]) { puts("FAIL argmax/tie"); failures++; break; }
    puf_mtl::Launch<Pool>{c}(y.data,(int*)nullptr,x.data,w.data,B,H);
    cudaDeviceSynchronize(); compare("nullable_argmax",y.data,ry);
    obs.guards(); points.guards(); x.guards(); w.guards(); dy.guards();
    y.guards(); dw.guards(); dx.guards(); arg.guards();
}
#define CHECK(prefix,S,F,P,B,H,M) check<puf_kid_##prefix##_materialize_points_kernel, \
    puf_kid_##prefix##_linear_max_kernel,puf_kid_##prefix##_output_wgrad_kernel, \
    puf_kid_##prefix##_grad_entity_kernel>(S,F,P,B,H,M)
int main() {
    @autoreleasepool {
        CHECK(me,2,4,16,1,1,0); CHECK(ae,4,5,26,1,1,0);
        CHECK(me,2,4,16,9,33,1); CHECK(ae,4,5,26,9,33,1);
        CHECK(me,2,4,16,7,65,2); CHECK(ae,4,5,26,7,65,2);
    }
    printf("entity CPU-reference tests: %s (%d failures)\n",failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}

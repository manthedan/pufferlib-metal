// Unsplit ragged-N GEMM: exact against the original scalar kernel and an
// independent CPU fma chain. Both storage types, guard zones and NaN outputs.
#include "gemm.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <chrono>
using namespace puf_mtl;
static int failures;
template<class T> static T narrow(float f) { return T(f); }
template<> __nv_bfloat16 narrow(float f) { return __float2bfloat16(f); }
template<class T> static float wide(T f) { return float(f); }
template<> float wide(__nv_bfloat16 f) { return __bfloat162float(f); }
template<class T> struct Buf {
    T *base,*p; int n;
    Buf(int n):n(n) { cudaMalloc(&base,(n+32)*sizeof(T));p=base+16;poison(); }
    void poison() { for(int i=0;i<n+32;i++) base[i]=narrow<T>(NAN); }
    bool guarded() { for(int i=0;i<16;i++) if(!std::isnan(wide(base[i])) || !std::isnan(wide(p[n+i]))) return false; return true; }
    ~Buf(){cudaFree(base);}
};
template<class T> static void run(int M,int N,int K,bool bench) {
    Buf<T> a(K*M),b(K*N),c(M*N),d(M*N);
    std::mt19937 r(7+N);std::uniform_real_distribution<float> u(-1,1);
    for(int i=0;i<a.n;i++)a.p[i]=narrow<T>(u(r));
    for(int i=0;i<b.n;i++)b.p[i]=narrow<T>(u(r));
    GemmOpts ref;ref.allow_tensor_ops=false;ref.allow_split_k=false;
    auto p=gemm_rm(0,true,false,M,N,K,1,a.p,M,b.p,N,0,c.p,N);
    gemm_rm(0,true,false,M,N,K,1,a.p,M,b.p,N,0,d.p,N,nullptr,0,&ref);
    cudaDeviceSynchronize();
    int bad=0;
    for(int i=0;i<c.n;i++)bad+=memcmp(c.p+i,d.p+i,sizeof(T))!=0 || !std::isfinite(wide(c.p[i]));
    for(int m:{0,M/2,M-1}) for(int n:{0,31,N-1}) {
        float sum=0;for(int k=0;k<K;k++)sum=std::fma(wide(a.p[k*M+m]),wide(b.p[k*N+n]),sum);
        T v=narrow<T>(sum);bad+=memcmp(&v,c.p+m*N+n,sizeof(T))!=0;
    }
    const char* knob = N%32 ? "PUF_METAL_GEMM_RAGGED" : "PUF_METAL_GEMM_WIDE_TN";
    bool disabled=getenv(knob) && !strcmp(getenv(knob),"0");
    bad+=p!=(disabled?(N%32?GEMM_TILED:GEMM_TENSOR_OPS):GEMM_WIDE_TN);
    bad+=!a.guarded()||!b.guarded()||!c.guarded()||!d.guarded();
    printf("M%d N%d K%d bytes%zu path%d differing%d\n",M,N,K,sizeof(T),p,bad);failures+=bad!=0;
    if(bench)for(int mode=0;mode<2;mode++) {
        auto st=std::chrono::steady_clock::now();
        for(int i=0;i<30;i++)gemm_rm(0,true,false,M,N,K,1,a.p,M,b.p,N,0,c.p,N,nullptr,0,mode?nullptr:&ref);
        cudaDeviceSynchronize();
        printf("mode%d %.3fms\n",mode,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-st).count()/30);
    }
}

// Preserve the original split selection, every FP32 partial, the deferred
// workspace contract and alpha/beta reduction, independently of final rounding.
template<class T> static void split_case(int K) {
    const int M=384,N=128, count=M*N;
    Buf<T> a(K*M),b(K*N),c(count),d(count);
    Buf<float> ws(1<<20),refws(1<<20);
    std::mt19937 rng(17+K);std::uniform_real_distribution<float> u(-1,1);
    for(int i=0;i<a.n;i++)a.p[i]=narrow<T>(u(rng));
    for(int i=0;i<b.n;i++)b.p[i]=narrow<T>(u(rng));
    for(int i=0;i<count;i++)c.p[i]=d.p[i]=narrow<T>(u(rng));
    GemmOpts ref;ref.allow_wide_tn=false;
    GemmDeferred f,r;
    auto fp=gemm_rm_deferred(0,true,false,M,N,K,0.75,a.p,M,b.p,N,-0.25,c.p,N,
        ws.p,ws.n*sizeof(float),16,&f);
    auto rp=gemm_rm_deferred(0,true,false,M,N,K,0.75,a.p,M,b.p,N,-0.25,d.p,N,
        refws.p,refws.n*sizeof(float),16,&r,&ref);
    cudaDeviceSynchronize();
    int bad=fp!=GEMM_TOPS_TN_SPLIT || rp!=GEMM_TOPS_TN_SPLIT || !f.pipe || !r.pipe || f.ws_used!=r.ws_used;
    if(bad){printf("FAIL split eligibility K%d\n",K);failures++;return;}
    const int used=f.ws_used/sizeof(float);
    bad+=memcmp(ws.p+4,refws.p+4,f.ws_used)!=0;
    for(int i=0;i<used;i++)bad+=!std::isfinite(ws.p[4+i]);
    for(int i=0;i<4;i++)bad+=!std::isnan(ws.p[i]);
    for(int i=4+used;i<ws.n;i++)bad+=!std::isnan(ws.p[i]);
    gemm_finish(0,&f);gemm_finish(0,&r);cudaDeviceSynchronize();
    bad+=f.pipe!=nullptr || r.pipe!=nullptr;
    for(int i=0;i<count;i++)bad+=memcmp(c.p+i,d.p+i,sizeof(T))!=0 || !std::isfinite(wide(c.p[i]));
    bad+=!ws.guarded()||!refws.guarded()||!c.guarded()||!d.guarded();
    printf("split M%d N%d K%d bytes%zu partials%d differing%d\n",M,N,K,sizeof(T),used,bad);
    failures+=bad!=0;
}
int main(int argc,char**) {
    bool bench=argc>1;
    for(int M:{128,256}) for(int N:{1007,1008,1009}) {
        run<float>(M,N,bench?8192:64,bench);run<__nv_bfloat16>(M,N,bench?8192:64,bench);
    }
    if(!bench) {
        for(int K:{2048,2064,16384}) {split_case<float>(K);split_case<__nv_bfloat16>(K);}
        run<float>(768,256,8192,false);run<__nv_bfloat16>(768,256,8192,false);
        run<float>(384,128,64,false);run<__nv_bfloat16>(384,128,64,false);
        run<float>(256,1008,8192,false);run<__nv_bfloat16>(256,1008,8192,false);
        run<float>(128,1009,2064,false);run<__nv_bfloat16>(128,1009,2064,false);
    }
    printf("%s\n",failures?"FAIL":"ALL PASS");return failures?1:0;
}

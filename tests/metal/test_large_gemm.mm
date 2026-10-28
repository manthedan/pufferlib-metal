// H512/H1024 TN tiles against the unchanged production dispatcher, every
// split FP32 partial, sampled CPU ascending-fma sums, NaNs and canaries.
#include "gemm.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
using namespace puf_mtl;
static int failures;
template<class T> static T narrow(float f) { return T(f); }
template<> __nv_bfloat16 narrow(float f) { return __float2bfloat16(f); }
template<class T> static float wide(T f) { return float(f); }
template<> float wide(__nv_bfloat16 f) { return __bfloat162float(f); }
template<class T> struct Buf {
    static constexpr size_t guard_n=16*256;
    T *base,*p; size_t n;
    Buf(size_t n):n(n) { cudaMalloc(&base,(n+2*guard_n)*sizeof(T));p=base+guard_n;
        memset(base,0xa5,guard_n*sizeof(T));memset(p+n,0xa5,guard_n*sizeof(T));poison(); }
    void poison() { for(size_t i=0;i<n;i++) p[i]=narrow<T>(NAN); }
    bool guarded() const {
        const auto* head=reinterpret_cast<const unsigned char*>(base);
        const auto* tail=reinterpret_cast<const unsigned char*>(p+n);
        for(size_t i=0;i<guard_n*sizeof(T);i++)if(head[i]!=0xa5||tail[i]!=0xa5)return false;
        return true;
    }
    ~Buf(){cudaFree(base);}
};
template<class T> static void run(int M,int N,int K,bool split,bool bench=false) {
    Buf<T> a((size_t)K*M),b((size_t)K*N),c(M*N),d(M*N);
    Buf<float> ws(split?4194304:16),rw(split?4194304:16);
    std::mt19937 rng(71+N);std::uniform_real_distribution<float> u(-1,1);
    for(size_t i=0;i<a.n;i++)a.p[i]=narrow<T>(u(rng));
    for(size_t i=0;i<b.n;i++)b.p[i]=narrow<T>(u(rng));
    const float alpha=split?0.75f:1, beta=split?-0.25f:0;
    if(split) for(size_t i=0;i<c.n;i++)c.p[i]=d.p[i]=narrow<T>(u(rng));
    GemmOpts ref;ref.allow_large_tn=false;
    GemmDeferred f,r;
    auto call=[&](bool oracle,GemmDeferred* finish){return gemm_rm_deferred(0,true,false,M,N,K,alpha,
        a.p,M,b.p,N,beta,oracle?d.p:c.p,N,split?(oracle?rw.p:ws.p):nullptr,
        split?ws.n*sizeof(float):0,split?16:0,finish,oracle?&ref:nullptr);};
    auto fp=call(false,&f),rp=call(true,&r);cudaDeviceSynchronize();
    int bad=0;
    bool off=getenv("PUF_METAL_GEMM_LARGE_TN")&&!strcmp(getenv("PUF_METAL_GEMM_LARGE_TN"),"0");
    bad+=fp!=(split?GEMM_SPLIT_K:((off || K%16)?rp:GEMM_WIDE_TN));
    bad+=f.ws_used!=r.ws_used || bool(f.pipe)!=bool(r.pipe);
    if(split) {
        bad+=!f.pipe || !f.ws_used;
        bad+=memcmp(ws.p+4,rw.p+4,f.ws_used)!=0;
        for(size_t i=0;i<f.ws_used/4;i++)bad+=!std::isfinite(ws.p[4+i]);
        for(size_t i=0;i<ws.n;i++)if(i<4 || i>=4+f.ws_used/4)bad+=!std::isnan(ws.p[i]);
        int splits=int(f.ws_used/4/(M*N));
        int chunk=((K+splits-1)/splits+15)/16*16;
        for(int z:{0,splits-1}) for(int m:{0,M-1}) for(int n:{0,N-1}) {
            float acc=0;for(int k=z*chunk;k<std::min((z+1)*chunk,K);k++)
                acc=std::fma(wide(a.p[(size_t)k*M+m]),wide(b.p[(size_t)k*N+n]),acc);
            bad+=memcmp(&acc,ws.p+4+(size_t)z*M*N+m*N+n,4)!=0;
        }
    } else {
        for(int m:{0,M/2,M-1})for(int n:{0,N/2,N-1}) {
            float acc=0;for(int k=0;k<K;k++)acc=std::fma(wide(a.p[(size_t)k*M+m]),wide(b.p[(size_t)k*N+n]),acc);
            T v=narrow<T>(acc);bad+=memcmp(&v,c.p+(size_t)m*N+n,sizeof(T))!=0;
        }
    }
    gemm_finish(0,&f);gemm_finish(0,&r);cudaDeviceSynchronize();
    for(size_t i=0;i<c.n;i++)bad+=memcmp(c.p+i,d.p+i,sizeof(T))!=0 || !std::isfinite(wide(c.p[i]));
    bad+=!a.guarded()||!b.guarded()||!c.guarded()||!d.guarded()||!ws.guarded()||!rw.guarded();
    printf("M%d N%d K%d bytes%zu split%d path%d reference%d differing%d\n",M,N,K,sizeof(T),split,fp,rp,bad);failures+=bad!=0;
    if(bench) for(int mode=0;mode<2;mode++) {
        for(int i=0;i<2;i++){GemmDeferred q;call(mode==0,&q);gemm_finish(0,&q);}cudaDeviceSynchronize();
        auto st=std::chrono::steady_clock::now();
        for(int i=0;i<5;i++){GemmDeferred q;call(mode==0,&q);gemm_finish(0,&q);}cudaDeviceSynchronize();
        printf("bytes%zu mode%d %.4fms\n",sizeof(T),mode,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-st).count()/5);
    }
}
int main(int argc,char** argv) {
    if(argc==4 && !strcmp(argv[1],"--bench")) {
        int H=atoi(argv[2]),K=atoi(argv[3]);run<float>(3*H,H,K,false,true);run<__nv_bfloat16>(3*H,H,K,false,true);
    } else if(argc==1) {
        for(int H:{512,1024})for(int K:{64,65,2064}) {
            run<float>(3*H,H,K,false);run<__nv_bfloat16>(3*H,H,K,false);
        }
        for(int H:{512,1024})for(int N:{H==512?233:109,H==512?234:110,H==512?235:111}) {
            run<float>(H,N,2064,true);run<__nv_bfloat16>(H,N,2064,true);
            run<float>(H,N,64,false);run<__nv_bfloat16>(H,N,64,false);
        }
        run<float>(512,234,65536,true);run<__nv_bfloat16>(512,234,65536,true);
        run<float>(3072,1024,4096,false);run<__nv_bfloat16>(3072,1024,4096,false);
    } else return 2;
    printf("%s\n",failures?"FAIL":"ALL PASS");return failures?1:0;
}

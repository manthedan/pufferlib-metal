// Exact norm-tree and Muon matrix checks against independent CPU arithmetic.
// Every pure output is NaN-poisoned before each call; in-place outputs receive
// fresh inputs. No reference and candidate share output or workspace storage.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "gemm.h"
using namespace puf_mtl;
struct Buffer {
    unsigned char* base;
    size_t bytes;
    Buffer(size_t n): bytes(n) {
        cudaMalloc(&base,n+128);
        memset(base,0xa5,n+128);
        poison();
    }
    void* data() { return base+64; }
    uint64_t addr() { return gpu_addr(data()); }
    void poison() {
        auto p=static_cast<uint16_t*>(data());
        for(size_t i=0;i<bytes/2;++i) p[i]=0x7fc1;
    }
    bool guard() {
        for(int i=0;i<64;++i) if(base[i]!=0xa5 || base[64+bytes+i]!=0xa5) return false;
        return true;
    }
    ~Buffer() { cudaFree(base); }
};
static float tree(std::vector<float> v) {
    v.resize(256,0.0f);
    for(int s=128;s;s>>=1) for(int i=0;i<s;++i) v[i]+=v[i+s];
    return v[0];
}
struct Sum { uint64_t dst,src; int32_t n; };
struct Normalize { uint64_t dst,sum; float eps; int32_t n; };
struct WideMat { uint64_t x,xb,g,gb,part; float scale; int32_t R,C,tops; };
struct Wide { WideMat mats[8]; float c0,c1,c2,eps; int32_t count,step,pad0,pad1; };
static_assert(sizeof(Wide)==480,"MuonWideArgs ABI");
static const float coeffs[15] = {
    4.0848f,-6.8946f,2.9270f, 3.9505f,-6.3029f,2.6377f,
    3.7418f,-5.5913f,2.3037f, 2.8769f,-3.1427f,1.2046f, 2.8366f,-3.0525f,1.2012f
};
static std::vector<prec_t> host_mm(const std::vector<prec_t>& A, const std::vector<prec_t>& B,
        const std::vector<prec_t>& old, int M,int N,int K,bool ta,bool tb,float alpha,float beta) {
    std::vector<prec_t> out(M*N);
    for(int m=0;m<M;++m) for(int n=0;n<N;++n) {
        float acc=0.0f;
        for(int k=0;k<K;++k) acc=fmaf(float(A[ta?k*M+m:m*K+k]),float(B[tb?n*K+k:k*N+n]),acc);
        float v=alpha*acc;
        if(beta!=0.0f) v=fmaf(beta,float(old[m*N+n]),v);
        out[m*N+n]=prec_t(v);
    }
    return out;
}
static bool mma_checks() {
    bool ok=true;
    for(auto shape:{std::vector<int>{96,32},{32,96},{32,32},{128,32},{32,128}}) {
        int R=shape[0],C=shape[1],M=std::min(R,C),ne=R*C,mm=M*M;
        Buffer x(ne*sizeof(prec_t)),xb(ne*sizeof(prec_t)),g(mm*sizeof(prec_t)),gb(mm*sizeof(prec_t));
        for(int trial=0;trial<2;++trial) {
            std::mt19937 rng(14100+trial);
            std::uniform_real_distribution<float> u(-0.01f,0.01f);
            std::vector<prec_t> input(ne);
            for(auto& v:input) v=prec_t(u(rng));
            auto state=input;
            std::vector<prec_t> gram,grambuf,out3;
            for(int step=0;step<5;++step) {
                gram=host_mm(state,state,{},M,M,std::max(R,C),R>C,R<=C,1,0);
                grambuf=host_mm(gram,gram,gram,M,M,M,false,false,coeffs[step*3+2],coeffs[step*3+1]);
                state=R>C ? host_mm(state,grambuf,state,R,C,C,false,false,1,coeffs[step*3])
                    : host_mm(grambuf,state,state,R,C,R,false,false,1,coeffs[step*3]);
                if(step==3) out3=state;
            }
            for(int repeat=0;repeat<2;++repeat) {
                memcpy(x.data(),input.data(),ne*sizeof(prec_t));
                xb.poison(); g.poison(); gb.poison();
                bool ran=muon_ns5_rm(0,static_cast<prec_t*>(x.data()),static_cast<prec_t*>(xb.data()),
                    static_cast<prec_t*>(g.data()),static_cast<prec_t*>(gb.data()),R,C,coeffs);
                cudaDeviceSynchronize();
                bool pass=ran && !memcmp(x.data(),out3.data(),ne*sizeof(prec_t)) &&
                    !memcmp(xb.data(),state.data(),ne*sizeof(prec_t)) &&
                    !memcmp(g.data(),gram.data(),mm*sizeof(prec_t)) &&
                    !memcmp(gb.data(),grambuf.data(),mm*sizeof(prec_t)) &&
                    x.guard() && xb.guard() && g.guard() && gb.guard();
                printf("NS MMA %dx%d trial=%d repeat=%d %s\n",R,C,trial,repeat,pass?"PASS":"FAIL");
                ok &= pass;
            }
        }
    }
    for(auto shape:{std::vector<int>{192,64},{64,192},{64,64},{128,128}}) {
        int R=shape[0],C=shape[1],M=std::min(R,C),ne=R*C,mm=M*M;
        Buffer x(ne*sizeof(prec_t)),xb(ne*sizeof(prec_t)),g(mm*sizeof(prec_t)),gb(mm*sizeof(prec_t));
        Wide args={}; args.count=1;args.c0=coeffs[0];args.c1=coeffs[1];args.c2=coeffs[2];
        args.mats[0]={x.addr(),xb.addr(),g.addr(),gb.addr(),0,1,R,C,0};
        for(int trial=0;trial<2;++trial) {
            std::mt19937 rng(14200+trial);
            std::uniform_real_distribution<float> u(-0.1f,0.1f);
            std::vector<prec_t> input(ne),gram(mm),grambuf(mm);
            for(auto& v:input) v=prec_t(u(rng));
            for(auto& v:gram) v=prec_t(u(rng));
            for(auto& v:grambuf) v=prec_t(u(rng));
            memcpy(g.data(),gram.data(),mm*sizeof(prec_t));
            auto expected=host_mm(gram,gram,gram,M,M,M,false,false,args.c2,args.c1);
            for(int repeat=0;repeat<2;++repeat) {
                gb.poison();
                dispatch(0,pipeline("muon_wide_gram_buf"),dim3((mm+255)/256,1,1),dim3(256),args);
                cudaDeviceSynchronize();
                bool pass=!memcmp(gb.data(),expected.data(),mm*sizeof(prec_t)) && gb.guard();
                printf("wide gram MMA %dx%d trial=%d repeat=%d %s\n",R,C,trial,repeat,pass?"PASS":"FAIL");
                ok &= pass;
            }
            // Fresh independent gram_buf input, never the preceding output.
            memcpy(gb.data(),grambuf.data(),mm*sizeof(prec_t));
            auto update=R>C ? host_mm(input,grambuf,input,R,C,M,false,false,1,args.c0)
                : host_mm(grambuf,input,input,R,C,M,false,false,1,args.c0);
            for(int step=0;step<2;++step) {
                args.step=step;
                Buffer& src=step==0?x:xb;
                Buffer& dst=step==0?xb:x;
                memcpy(src.data(),input.data(),ne*sizeof(prec_t));dst.poison();
                dispatch(0,pipeline("muon_wide_update"),dim3((ne+255)/256,1,1),dim3(256),args);
                cudaDeviceSynchronize();
                bool pass=!memcmp(dst.data(),update.data(),ne*sizeof(prec_t)) && x.guard() && xb.guard();
                printf("wide update MMA %dx%d trial=%d step=%d %s\n",R,C,trial,step,pass?"PASS":"FAIL");
                ok &= pass;
            }
        }
    }
    return ok;
}

int main() {
    bool ok=true;
    for(int n:{1,31,255,256,257,3072,12288,200003}) for(int trial=0;trial<2;++trial) {
        int blocks=std::min((n+255)/256,256);
        Buffer x(size_t(n)*sizeof(prec_t)), ref(size_t(n)*sizeof(prec_t));
        Buffer partial(256*4), widepartial(256*4), sum(4);
        std::mt19937 rng(14000+trial);
        std::uniform_real_distribution<float> u(-1.0f,1.0f);
        auto xp=static_cast<prec_t*>(x.data());
        for(int i=0;i<n;++i) xp[i]=prec_t(u(rng)*std::ldexp(1.0f,(i%21)-10));
        // Include signed zero and tiny values without changing tree topology.
        xp[0]=prec_t(trial?-0.0f:1e-38f);
        std::vector<float> expected(blocks);
        for(int b=0;b<blocks;++b) {
            std::vector<float> sh(256,0.0f);
            for(int t=0;t<256;++t) for(int i=b*256+t;i<n;i+=blocks*256) {
                float v=float(xp[i]); sh[t]=fmaf(v,v,sh[t]);
            }
            expected[b]=tree(sh);
        }
        partial.poison();
        dispatch(0,pipeline("muon_sum_sq_partials"),dim3(blocks),dim3(256),Sum{partial.addr(),x.addr(),n});
        sum.poison();
        dispatch(0,pipeline("muon_sum_sq_reduce"),dim3(1),dim3(256),Sum{sum.addr(),partial.addr(),blocks});
        cudaDeviceSynchronize();
        float total=tree(expected);
        long bad=memcmp(partial.data(),expected.data(),blocks*4)!=0;
        bad+=memcmp(sum.data(),&total,4)!=0;
        Wide args={};
        args.count=1; args.eps=1e-7f;
        args.mats[0]={x.addr(),0,0,0,widepartial.addr(),1.0f,n,1,0};
        widepartial.poison();
        dispatch(0,pipeline("muon_wide_sumsq"),dim3(blocks,1,1),dim3(256),args);
        cudaDeviceSynchronize();
        bad+=memcmp(widepartial.data(),expected.data(),blocks*4)!=0;
        // Compare wide normalization against the unchanged primitive using
        // separate fresh in-place inputs and the independently checked norm.
        memcpy(ref.data(),x.data(),size_t(n)*sizeof(prec_t));
        dispatch(0,pipeline("muon_l2_normalize"),dim3((n+255)/256),dim3(256),
            Normalize{ref.addr(),sum.addr(),1e-7f,n});
        dispatch(0,pipeline("muon_wide_normalize"),dim3((n+255)/256,1,1),dim3(256),args);
        cudaDeviceSynchronize();
        bad+=memcmp(x.data(),ref.data(),size_t(n)*sizeof(prec_t))!=0;
        bool guards=x.guard() && ref.guard() && partial.guard() && widepartial.guard() && sum.guard();
        printf("norm tree n=%d trial=%d mismatches=%ld guards=%d %s\n",n,trial,bad,guards,!bad&&guards?"PASS":"FAIL");
        ok &= !bad && guards;
    }
    ok &= mma_checks();
    return ok?0:1;
}

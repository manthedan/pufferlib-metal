// Large recurrent NT/dX against original dispatches, sampled CPU FMA
// references, BF16 double rounding, fresh NaNs and allocation canaries.
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
struct AddArgs{uint64_t A,B,C,D;int32_t M,N,K,pad;};
template<class T> static void run(int M,int H,bool add,bool bench=false){
 int N=add?H:3*H,K=add?3*H:H;
 Buf<T> a((size_t)M*K),b((size_t)N*K),c((size_t)M*N),d((size_t)M*N),plus((size_t)M*N);
 std::mt19937 rng(57+M+H);std::uniform_real_distribution<float> u(-1,1);
 for(auto buf:{&a,&b,&plus})for(size_t i=0;i<buf->n;i++)buf->p[i]=narrow<T>(u(rng));
 GemmOpts ref;ref.allow_large_linear=false;
 auto oracle=[&]{
  if(add){
   AddArgs args={gpu_addr(a.p),gpu_addr(b.p),gpu_addr(d.p),gpu_addr(plus.p),M,N,K,0};
   dispatch(0,pipeline(sizeof(T)==4?"tensor_ops_gemm_nn_add_f32":"tensor_ops_gemm_nn_add_bf16"),dim3(N/32,M/64),dim3(128),args);
  }else gemm_rm(0,false,true,M,N,K,1,a.p,K,b.p,K,0,d.p,N,nullptr,0,&ref);
 };
 bool off=getenv("PUF_METAL_GEMM_LARGE_LINEAR")&&!strcmp(getenv("PUF_METAL_GEMM_LARGE_LINEAR"),"0");
 auto candidate=[&]{
  if(add){if(!gemm_nn_add_rm(0,M,N,K,a.p,b.p,plus.p,c.p))failures++;}
  else {auto path=gemm_rm(0,false,true,M,N,K,1,a.p,K,b.p,K,0,c.p,N);
   if(path!=(M%64 ? GEMM_TILED : ((M>=4096&&!off)?GEMM_LARGE_LINEAR:GEMM_TENSOR_OPS)))failures++;}
 };
 oracle();candidate();cudaDeviceSynchronize();int bad=0;
 for(size_t i=0;i<c.n;i++)bad+=memcmp(c.p+i,d.p+i,sizeof(T))!=0||!std::isfinite(wide(c.p[i]));
 for(int m:{0,M/2,M-1})for(int n:{0,N/2,N-1}){
  float acc=0;for(int k=0;k<K;k++)acc=std::fma(wide(a.p[(size_t)m*K+k]),wide(b.p[add?(size_t)k*N+n:(size_t)n*K+k]),acc);
  if(add)acc=wide(narrow<T>(acc))+wide(plus.p[(size_t)m*N+n]);
  T want=narrow<T>(acc);bad+=memcmp(&want,c.p+(size_t)m*N+n,sizeof(T))!=0;
 }
 bad+=!a.guarded()||!b.guarded()||!c.guarded()||!d.guarded()||!plus.guarded();
 printf("%s M%d H%d bytes%zu differing%d\n",add?"nn_add":"nt",M,H,sizeof(T),bad);failures+=bad!=0;
 if(bench)for(int mode=0;mode<2;mode++){
  for(int i=0;i<2;i++){if(mode)candidate();else oracle();}cudaDeviceSynchronize();
  auto start=std::chrono::steady_clock::now();
  for(int i=0;i<8;i++){if(mode)candidate();else oracle();}cudaDeviceSynchronize();
  printf("%s H%d bytes%zu mode%d %.4fms\n",add?"nn_add":"nt",H,sizeof(T),mode,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/8);
 }
}
int main(int argc,char** argv){
 if(argc==4&&!strcmp(argv[1],"--bench")){
  int H=atoi(argv[2]),M=atoi(argv[3]);for(bool add:{false,true}){run<float>(M,H,add,true);run<__nv_bfloat16>(M,H,add,true);}
 }else if(argc==1){
  for(int H:{512,1024})for(int M:{64,4096,4160})for(bool add:{false,true}){
   run<float>(M,H,add);run<__nv_bfloat16>(M,H,add);
  }
  for(int H:{512,1024}){run<float>(4097,H,false);run<__nv_bfloat16>(4097,H,false);}
  Buf<float> a(64),b(64),c(64),d(64);
  if(gemm_nn_add_rm(0,4097,512,1536,a.p,b.p,c.p,d.p))failures++;
 }else return 2;
 printf("%s\n",failures?"FAIL":"ALL PASS");return failures?1:0;
}

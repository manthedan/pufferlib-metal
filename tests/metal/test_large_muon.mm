// Every large Muon stage and production wrapper against the original generic
// norm/GEMM/copy/store sequence. Covers unaligned gradients, both orientations,
// a mixed ragged fallback, finite outputs and allocation guards.
#include "gemm.h"
#include "shaders/wide_config.h"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using namespace puf_mtl;
static prec_t narrow(float x) {
#ifdef PRECISION_FLOAT
 return x;
#else
 return __float2bfloat16(x);
#endif
}
static float wide(prec_t x) {
#ifdef PRECISION_FLOAT
 return x;
#else
 return __bfloat162float(x);
#endif
}
static int failures;
struct Buf {
 static constexpr size_t guard_n=16*256;
 prec_t *base,*p; size_t n; int off;
 Buf(size_t n,int off=0):n(n),off(off){cudaMalloc(&base,(n+2*guard_n+off)*sizeof(prec_t));p=base+guard_n+off;
  memset(base,0xa5,(guard_n+off)*sizeof(prec_t));memset(p+n,0xa5,guard_n*sizeof(prec_t));poison();}
 ~Buf(){cudaFree(base);}
 void poison(){for(size_t i=0;i<n;i++)p[i]=narrow(NAN);}
 bool guarded() const {
  const auto* head=reinterpret_cast<const unsigned char*>(base);
  const auto* tail=reinterpret_cast<const unsigned char*>(p+n);
  for(size_t i=0;i<(guard_n+off)*sizeof(prec_t);i++)if(head[i]!=0xa5)return false;
  for(size_t i=0;i<guard_n*sizeof(prec_t);i++)if(tail[i]!=0xa5)return false;
  return true;
 }
};
struct Mat {uint64_t x,xb,g,gb,part;float scale;int32_t R,C,tops;};
struct Args {Mat mats[8];float c0,c1,c2,eps;int32_t count,step,pad0,pad1;};
struct SumArgs {uint64_t dst,src;int32_t n;};
struct NormArgs {uint64_t dst,sum;float eps;int32_t n;};
struct StoreArgs {uint64_t dst,src;float scale;int32_t n;};
static const float coeffs[15]={4.0848f,-6.8946f,2.9270f,3.9505f,-6.3029f,2.6377f,
 3.7418f,-5.5913f,2.3037f,2.8769f,-3.1427f,1.2046f,2.8366f,-3.0525f,1.2012f};
static void exact(const char* name,const Buf& a,const Buf& b){
 size_t bad=0;for(size_t i=0;i<a.n;i++)bad+=memcmp(a.p+i,b.p+i,sizeof(prec_t))!=0||!std::isfinite(wide(a.p[i]));
 if(bad)printf("FAIL %s size%zu differences%zu\n",name,a.n,bad);failures+=bad!=0;
}
struct Case {
 int R,C,M;float scale;Buf x,xb,g,gb,ref,rb,rg,rgb,wrapped;float *part,*norm;
 Case(int R,int C,int off):R(R),C(C),M(std::min(R,C)),scale(sqrtf(std::max(1.f,float(R)/C))),
 x(R*C,off),xb(R*C),g(M*M),gb(M*M),ref(R*C,off),rb(R*C),rg(M*M),rgb(M*M),wrapped(R*C,off){
  cudaMalloc(&part,256*4);cudaMalloc(&norm,4);
  std::mt19937 rng(91+R+C);std::uniform_real_distribution<float> u(-0.01,0.01);
  for(size_t i=0;i<x.n;i++)x.p[i]=ref.p[i]=wrapped.p[i]=narrow(u(rng));
 }
 ~Case(){cudaFree(part);cudaFree(norm);}
 bool guarded(){bool ok=true;for(Buf* b:{&x,&xb,&g,&gb,&ref,&rb,&rg,&rgb,&wrapped})ok&=b->guarded();return ok;}
};
static void run(bool unaligned, bool ragged=false){
 const std::vector<std::pair<int,int>> shapes=ragged
  ? std::vector<std::pair<int,int>>{{528,512},{1024,1040},{512,234},{1024,110}}
  : std::vector<std::pair<int,int>>{{1536,512},{3072,1024},{1024,3072},{513,512},{33,64},{384,128},{129,128},{1024,1024}};
 Args a={};a.count=shapes.size();a.eps=1e-7f;
 std::vector<Case*> cases;std::vector<MuonNSMatrix> mats;
 size_t actual_ws=0;int rcmax=0,mmmax=0,mmax=0;std::vector<float*> partials;
 for(int i=0;i<a.count;i++){
  auto c=new Case(shapes[i].first,shapes[i].second,unaligned?1:0);cases.push_back(c);
  actual_ws+=muon_ns_wide_scratch_bytes(c->R,c->C,!unaligned);rcmax=std::max(rcmax,c->R*c->C);mmax=std::max(mmax,c->M);mmmax=std::max(mmmax,c->M*c->M);
  float* part;cudaMalloc(&part,256*4);partials.push_back(part);
  // tops bits0/1 select the old aligned Gram. Bit2 permits large fragments.
  int tops=(c->M%64==0?((unaligned?0:1)|2):0)|4;
  a.mats[i]={gpu_addr(c->x.p),gpu_addr(c->xb.p),gpu_addr(c->g.p),gpu_addr(c->gb.p),gpu_addr(part),c->scale,c->R,c->C,tops};
  mats.push_back({c->wrapped.p,c->R,c->C,c->scale});
 }
 Buf scratch(actual_ws/sizeof(prec_t));
 auto before=stats();
 if(muon_ns_wide_rm(0,mats.data(),mats.size(),coeffs,a.eps,scratch.p,actual_ws-1)
         || stats().dispatches!=before.dispatches){printf("FAIL short scratch encoded work\n");failures++;}
 // Production reaches Muon with preceding work queued. Exercise GPU copies,
 // rather than the runtime's idle-stream CPU memcpy shortcut.
 dispatch(0,pipeline("muon_sum_sq_partials"),dim3(1),dim3(256),
     SumArgs{gpu_addr(cases[0]->part),gpu_addr(cases[0]->x.p),cases[0]->R*cases[0]->C});
 bool accepted=muon_ns_wide_rm(0,mats.data(),mats.size(),coeffs,a.eps,scratch.p,actual_ws);
 if(!accepted){printf("FAIL wrapper rejected large batch\n");failures++;}
 const int blocks=(rcmax+255)/256;
 int gram_groups=0,gb_groups=0,update_groups=0;
 for(auto c:cases){
  bool large=(c->M==512||c->M==1024)&&c->R%16==0&&c->C%16==0;
  auto tiles=[](int r,int n,int tm,int tn){return ((r+tm-1)/tm)*((n+tn-1)/tn);};
  gram_groups=std::max(gram_groups,large?tiles(c->M,c->M,PUF_LARGE_MUON_GRAM_M,PUF_LARGE_MUON_GRAM_N):(c->M*c->M+PUF_LARGE_MUON_GRAM_THREADS-1)/PUF_LARGE_MUON_GRAM_THREADS);
  gb_groups=std::max(gb_groups,large?tiles(c->M,c->M,PUF_LARGE_MUON_PRODUCT_M,PUF_LARGE_MUON_PRODUCT_N):(c->M*c->M+PUF_LARGE_MUON_PRODUCT_THREADS-1)/PUF_LARGE_MUON_PRODUCT_THREADS);
  update_groups=std::max(update_groups,large?tiles(c->R,c->C,PUF_LARGE_MUON_PRODUCT_M,PUF_LARGE_MUON_PRODUCT_N):(c->R*c->C+PUF_LARGE_MUON_PRODUCT_THREADS-1)/PUF_LARGE_MUON_PRODUCT_THREADS);
 }
 dispatch(0,pipeline("muon_wide_sumsq"),dim3(std::min(blocks,256),1,a.count),dim3(256),a);
 dispatch(0,pipeline("muon_wide_normalize"),dim3(blocks,1,a.count),dim3(256),a);
 for(auto c:cases){
  int n=c->R*c->C,nb=std::min((n+255)/256,256);
  dispatch(0,pipeline("muon_sum_sq_partials"),dim3(nb),dim3(256),SumArgs{gpu_addr(c->part),gpu_addr(c->ref.p),n});
  dispatch(0,pipeline("muon_sum_sq_reduce"),dim3(1),dim3(256),SumArgs{gpu_addr(c->norm),gpu_addr(c->part),nb});
  dispatch(0,pipeline("muon_l2_normalize"),dim3((n+255)/256),dim3(256),NormArgs{gpu_addr(c->ref.p),gpu_addr(c->norm),a.eps,n});
 }
 cudaDeviceSynchronize();for(auto c:cases)exact("normalize",c->x,c->ref);
 GemmOpts oracle;oracle.allow_large_tn=false;oracle.allow_wide_tn=false;
 for(int step=0;step<5;step++){
  a.step=step;a.c0=coeffs[3*step];a.c1=coeffs[3*step+1];a.c2=coeffs[3*step+2];
  dispatch(0,pipeline("wide_large_muon_gram"),dim3(gram_groups,1,a.count),dim3(PUF_LARGE_MUON_GRAM_THREADS),a);
  for(auto c:cases){
   prec_t* src=step%2?c->rb.p:c->ref.p;int M=c->M;
   if(c->R>c->C)gemm_rm(0,true,false,M,M,c->R,1,src,c->C,src,c->C,0,c->rg.p,M,nullptr,0,&oracle);
   else gemm_rm(0,false,true,M,M,c->C,1,src,c->C,src,c->C,0,c->rg.p,M,nullptr,0,&oracle);
  }
  cudaDeviceSynchronize();for(auto c:cases)exact("gram",c->g,c->rg);
  dispatch(0,pipeline("wide_large_muon_gram_buf"),dim3(gb_groups,1,a.count),dim3(PUF_LARGE_MUON_PRODUCT_THREADS),a);
  for(auto c:cases){int M=c->M;
   cudaMemcpyAsync(c->rgb.p,c->rg.p,M*M*sizeof(prec_t),cudaMemcpyDeviceToDevice,0);
   gemm_rm(0,false,false,M,M,M,a.c2,c->rg.p,M,c->rg.p,M,a.c1,c->rgb.p,M,nullptr,0,&oracle);
  }
  cudaDeviceSynchronize();for(auto c:cases)exact("gram_buf",c->gb,c->rgb);
  dispatch(0,pipeline("wide_large_muon_update"),dim3(update_groups,1,a.count),dim3(PUF_LARGE_MUON_PRODUCT_THREADS),a);
  for(auto c:cases){int R=c->R,C=c->C,M=c->M;prec_t* src=step%2?c->rb.p:c->ref.p;prec_t* dst=step%2?c->ref.p:c->rb.p;
   cudaMemcpyAsync(dst,src,R*C*sizeof(prec_t),cudaMemcpyDeviceToDevice,0);
   if(R>C)gemm_rm(0,false,false,R,C,M,1,src,C,c->rgb.p,M,a.c0,dst,C,nullptr,0,&oracle);
   else gemm_rm(0,false,false,R,C,M,1,c->rgb.p,M,src,C,a.c0,dst,C,nullptr,0,&oracle);
  }
  cudaDeviceSynchronize();for(auto c:cases){exact("update",step%2?c->x:c->xb,step%2?c->ref:c->rb);failures+=!c->guarded();}
  printf("unaligned%d step%d cumulative_failures%d\n",unaligned,step,failures);
  // Fresh NaNs for the next stage's destinations; the current source survives.
  if(step<4)for(auto c:cases){(step%2?c->xb:c->x).poison();(step%2?c->rb:c->ref).poison();c->g.poison();c->rg.poison();c->gb.poison();c->rgb.poison();}
 }
 for(auto c:cases){int n=c->R*c->C;
  dispatch(0,pipeline("muon_store_update"),dim3((n+255)/256),dim3(256),StoreArgs{gpu_addr(c->ref.p),gpu_addr(c->rb.p),c->scale,n});
 }
 cudaDeviceSynchronize();for(auto c:cases){if(accepted)exact("production wrapper",c->wrapped,c->ref);failures+=!c->guarded();}
 failures+=!scratch.guarded();
 for(auto c:cases)delete c;for(auto p:partials)cudaFree(p);
}
// Timing batches match the two native workloads. Numerical coverage remains
// the independent per-stage tests above; benchmark mode measures only wrapper
// dispatches, without mixing direct-reference timings into the profile.
static int bench(int H) {
 if(H!=512 && H!=1024)return 2;
 std::vector<std::pair<int,int>> shapes={{H,H==512?234:110}};
 if(H==1024)shapes.push_back({5,H});
 for(int i=0;i<(H==512?2:4);++i)shapes.push_back({3*H,H});
 std::vector<Case*> cases;std::vector<MuonNSMatrix> mats;size_t ws=0;
 for(auto sh:shapes){
  int off=0;
#ifndef PRECISION_FLOAT
  if(H==1024 && sh.first==3*H)off=4; // Native WEF logstd offset: 8 bytes.
#endif
  auto c=new Case(sh.first,sh.second,off);cases.push_back(c);
  mats.push_back({c->wrapped.p,c->R,c->C,c->scale});
  ws+=muon_ns_wide_scratch_bytes(c->R,c->C,off==0);
 }
 Buf scratch((ws+sizeof(prec_t)-1)/sizeof(prec_t));
 auto start=std::chrono::steady_clock::now();
 for(int i=0;i<10;++i){
  if(i==2){cudaDeviceSynchronize();start=std::chrono::steady_clock::now();}
  if(!muon_ns_wide_rm(0,mats.data(),mats.size(),coeffs,1e-7f,scratch.p,ws))return 1;
 }
 cudaDeviceSynchronize();
 printf("muon H%d wrapper %.4fms\n",H,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/8);
 for(auto c:cases){for(size_t i=0;i<c->wrapped.n;++i)failures+=!std::isfinite(wide(c->wrapped.p[i]));failures+=!c->guarded();delete c;}
 failures+=!scratch.guarded();return failures?1:0;
}

int main(int argc,char** argv){
 if(argc==3&&!strcmp(argv[1],"--bench"))return bench(atoi(argv[2]));
 if(argc==2 && !strcmp(argv[1],"--disabled")) {
  bool ok=!muon_ns_wide_eligible(3072,1024)&&muon_ns_wide_eligible(1536,512);
  printf("large opt-out %s\n",ok?"PASS":"FAIL");return ok?0:1;
 }

 run(false);run(true);run(true,true);
 // Width and size whitelist must preserve split-K cases outside this tier.
 for(auto shape:{std::pair<int,int>{3073,1024},{2048,2},{3072,1023}})
  failures+=muon_ns_wide_eligible(shape.first,shape.second);
 printf("%s (%d failures)\n",failures?"FAIL":"ALL PASS",failures);return failures?1:0;
}

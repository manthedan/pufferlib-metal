// Every wide Muon stage vs the independent pre-leg20 scalar kernel. Check
// every intermediate, both ping-pong sides, mixed batches and odd fallbacks.
#include "gemm.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace puf_mtl;
struct Mat { uint64_t x,xb,g,gb,part; float scale; int R,C,tops; };
struct Args {Mat mats[8];float c0,c1,c2,eps;int count,step,pad0,pad1;};
static_assert(sizeof(Args)==480,"Metal ABI");
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
// Back the logical allocation with enough canary space to contain a complete
// erroneous fragment tile at a ragged edge. Compare bytes: a NaN guard would
// miss an out-of-bounds store whose result is also NaN.
struct Buf {
 static constexpr int guard_n=16*256;
 prec_t *base,*p;int n;
 Buf(int n):n(n){cudaMalloc(&base,(n+2*guard_n)*sizeof(prec_t));p=base+guard_n;
  memset(base,0xa5,guard_n*sizeof(prec_t));memset(p+n,0xa5,guard_n*sizeof(prec_t));poison();}
 void poison(){for(int i=0;i<n;i++)p[i]=narrow(NAN);}
 bool guard() const {
  const auto* head=reinterpret_cast<const unsigned char*>(base);
  const auto* tail=reinterpret_cast<const unsigned char*>(p+n);
  for(size_t i=0;i<guard_n*sizeof(prec_t);i++)if(head[i]!=0xa5||tail[i]!=0xa5)return false;
  return true;
 }
 ~Buf(){cudaFree(base);}
};
static int bad;
static void exact(Buf& a,Buf& b,const char* stage,int R,int C){
 int differences=0;
 for(int i=0;i<a.n;i++)differences+=memcmp(a.p+i,b.p+i,sizeof(prec_t))!=0||!std::isfinite(wide(a.p[i]));
 if(differences)printf("stage=%s shape=%dx%d differences=%d\n",stage,R,C,differences);
 bad+=differences;
}
static void check_guards(const std::vector<Buf*>& buf,const char* stage){
 for(size_t i=0;i<buf.size();i++)if(!buf[i]->guard()){
  printf("stage=%s buffer=%zu guard_corruption\n",stage,i);++bad;
 }
}
static void stages(const int shapes[][2],int count){
 Args a={},r={};a.count=r.count=count;
 std::vector<Buf*> buf;std::mt19937 rng(91);std::uniform_real_distribution<float> u(-0.01,0.01);
 int blocks=0,mm=0;
 for(int i=0;i<count;i++){
  int R=shapes[i][0],C=shapes[i][1],M=std::min(R,C),rc=R*C;blocks=std::max(blocks,(rc+255)/256);mm=std::max(mm,M*M);
  for(int j=0;j<4;j++){
   auto x=new Buf(j<2?rc:M*M),y=new Buf(j<2?rc:M*M);buf.push_back(x);buf.push_back(y);
   if(j==0)for(int k=0;k<rc;k++)x->p[k]=y->p[k]=narrow(u(rng));
   switch(j) {
   case 0: a.mats[i].x=gpu_addr(x->p);r.mats[i].x=gpu_addr(y->p);break;
   case 1: a.mats[i].xb=gpu_addr(x->p);r.mats[i].xb=gpu_addr(y->p);break;
   case 2: a.mats[i].g=gpu_addr(x->p);r.mats[i].g=gpu_addr(y->p);break;
   case 3: a.mats[i].gb=gpu_addr(x->p);r.mats[i].gb=gpu_addr(y->p);break;
   }
  }
  a.mats[i].R=r.mats[i].R=R;a.mats[i].C=r.mats[i].C=C;
  // Force the same existing Gram kernel (tensor_ops where aligned).
  a.mats[i].tops=r.mats[i].tops=M%64==0?3:0;
 }
 cudaStream_t s;cudaStreamCreate(&s);
 for(int step=0;step<5;step++){
  a.step=r.step=step;a.c0=r.c0=1.25;a.c1=r.c1=-1.3;a.c2=r.c2=0.6;
  for(Args* p:{&a,&r})dispatch(s,pipeline("muon_wide_gram"),dim3(8,4,count),dim3(128),*p);
  dispatch(s,pipeline("wide_muon_gram_buf"),dim3((mm+255)/256,1,count),dim3(256),a);
  dispatch(s,pipeline("muon_wide_gram_buf"),dim3((mm+255)/256,1,count),dim3(256),r);
  cudaStreamSynchronize(s);
  for(int i=0;i<count;i++){exact(*buf[i*8+4],*buf[i*8+5],"gram",shapes[i][0],shapes[i][1]);exact(*buf[i*8+6],*buf[i*8+7],"gram_buf",shapes[i][0],shapes[i][1]);}
  check_guards(buf,"gram_buf");
  dispatch(s,pipeline("wide_muon_update"),dim3(blocks,1,count),dim3(256),a);
  dispatch(s,pipeline("muon_wide_update"),dim3(blocks,1,count),dim3(256),r);
  cudaStreamSynchronize(s);
  for(int i=0;i<count;i++)exact(*buf[i*8+(step%2?0:2)],*buf[i*8+(step%2?1:3)],"update",shapes[i][0],shapes[i][1]);
  check_guards(buf,"update");
  printf("step%d cumulative_differences=%d\n",step,bad);
  // Fresh poison for next iteration's outputs, after all readers finish.
  for(int i=0;i<count;i++){buf[i*8+(step%2?2:0)]->poison();buf[i*8+(step%2?3:1)]->poison();for(int j=4;j<8;j++)buf[i*8+j]->poison();}
 }
 cudaStreamDestroy(s);for(auto b:buf)delete b;
}
// Exercise the production selector, normalization, scratch layout and store.
// The reference encodes the pre-leg20 stages directly and never calls the
// wrapper. The driver checks KPROF counts, including these 18 oracle dispatches.
static void wrapper(int H){
 const int shapes[][2]={{2*H,H},{H,2*H},{H==128?129:256,H==128?128:257},{33,64}};
 constexpr int count=4;
 const float coeffs[15]={4.0848f,-6.8946f,2.9270f,3.9505f,-6.3029f,2.6377f,
  3.7418f,-5.5913f,2.3037f,2.8769f,-3.1427f,1.2046f,2.8366f,-3.0525f,1.2012f};
 Args r={};r.count=count;r.eps=1e-7f;
 std::vector<Buf*> buf;
 std::vector<MuonNSMatrix> mats;
 std::mt19937 rng(92);std::uniform_real_distribution<float> u(-0.01,0.01);
 size_t scratch_bytes=0;int max_rc=0,max_m=0;
 for(int i=0;i<count;i++){
  int R=shapes[i][0],C=shapes[i][1],M=std::min(R,C),rc=R*C;
  max_rc=std::max(max_rc,rc);max_m=std::max(max_m,M);
  auto x=new Buf(rc),ref=new Buf(rc),xb=new Buf(rc),g=new Buf(M*M),gb=new Buf(M*M);
  auto part=new Buf(256*sizeof(float)/sizeof(prec_t));
  for(auto b:{x,ref,xb,g,gb,part})buf.push_back(b);
  for(int j=0;j<rc;j++)x->p[j]=ref->p[j]=narrow(u(rng));
  float scale=1.125f+0.25f*i;
  mats.push_back({x->p,R,C,scale});
  r.mats[i]={gpu_addr(ref->p),gpu_addr(xb->p),gpu_addr(g->p),gpu_addr(gb->p),
   gpu_addr(part->p),scale,R,C,M%64==0?3:0};
  scratch_bytes+=muon_ns_wide_scratch_bytes(R,C);
 }
 auto scratch=new Buf(scratch_bytes/sizeof(prec_t));buf.push_back(scratch);
 cudaStream_t s;cudaStreamCreate(&s);
 bool ran=muon_ns_wide_rm(s,mats.data(),count,coeffs,r.eps,scratch->p,scratch_bytes);
 bad+=!ran;
 const int blocks=(max_rc+255)/256;
 dispatch(s,pipeline("muon_wide_sumsq"),dim3(std::min(blocks,256),1,count),dim3(256),r);
 dispatch(s,pipeline("muon_wide_normalize"),dim3(blocks,1,count),dim3(256),r);
 for(int step=0;step<5;step++){
  r.step=step;r.c0=coeffs[3*step];r.c1=coeffs[3*step+1];r.c2=coeffs[3*step+2];
  dispatch(s,pipeline("muon_wide_gram"),dim3((max_m+31)/32,(max_m+63)/64,count),dim3(128),r);
  dispatch(s,pipeline("muon_wide_gram_buf"),dim3((max_m*max_m+255)/256,1,count),dim3(256),r);
  dispatch(s,pipeline("muon_wide_update"),dim3(blocks,1,count),dim3(256),r);
 }
 dispatch(s,pipeline("muon_wide_store"),dim3(blocks,1,count),dim3(256),r);
 cudaStreamSynchronize(s);
 for(int i=0;i<count;i++)exact(*buf[i*6],*buf[i*6+1],"wrapper",shapes[i][0],shapes[i][1]);
 check_guards(buf,"wrapper");
 printf("wrapper H=%d ran=%d cumulative_differences=%d\n",H,ran,bad);
 cudaStreamDestroy(s);for(auto b:buf)delete b;
}
int main(int argc,char** argv){
 if(argc==3&&strcmp(argv[1],"--wrapper")==0&&(atoi(argv[2])==128||atoi(argv[2])==256)){
  wrapper(atoi(argv[2]));
 }else if(argc==1){
  const int shapes[][2]={{256,1008},{768,256},{256,768},{384,128},{128,384},{128,128},{192,64},{33,64}};
  stages(shapes,8);
  // These ragged matrices pass the width whitelist. A mixed batch selects the
  // wide kernels, so their per-matrix divisibility guards must select fallback.
  const int ragged[][2]={{384,128},{256,768},{129,128},{256,257}};
  stages(ragged,4);
 }else{
  fprintf(stderr,"usage: %s [--wrapper 128|256]\n",argv[0]);return 2;
 }
 printf("%s\n",bad?"FAIL":"ALL PASS");return bad?1:0;
}

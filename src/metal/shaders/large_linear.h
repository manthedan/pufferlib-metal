// Spatially retiled recurrent projection and dX. No split-K or reduction change.
template<class T> struct LargeAddArgs { device T *A, *B, *C, *D; int M,N,K; };
template<class T,int TM,int TN,int THREADS>
static void large_nt_body(constant WideGemmArgs<T>& a,uint2 tg) {
    auto A=tensor<device T,dextents<int32_t,2>,tensor_inline>(a.A,dextents<int32_t,2>(a.K,a.M));
    auto B=tensor<device T,dextents<int32_t,2>,tensor_inline>(a.B,dextents<int32_t,2>(a.K,a.N));
    constexpr auto desc=matmul2d_descriptor(TM,TN,static_cast<int>(dynamic_extent),false,true,false);
    matmul2d<desc,execution_simdgroups<THREADS/32>> op;
    auto left=A.slice(0,tg.y*TM),right=B.slice(0,tg.x*TN);
    auto c=op.template get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    for(uint16_t i=0;i<c.get_capacity();i++)if(c.is_valid_element(i))c[i]=0.0f;
    op.run(left,right,c);
    for(uint16_t i=0;i<c.get_capacity();i++)if(c.is_valid_element(i)) {
        auto ix=c.get_multidimensional_index(i);int row=tg.y*TM+ix[1],col=tg.x*TN+ix[0];
        if(row<a.M&&col<a.N)puf_st(a.C,(long)row*a.N+col,c[i]);
    }
}
template<class T,int TM,int TN,int THREADS>
static void large_nn_add_body(constant LargeAddArgs<T>& a,uint2 tg) {
    auto A=tensor<device T,dextents<int32_t,2>,tensor_inline>(a.A,dextents<int32_t,2>(a.K,a.M));
    auto B=tensor<device T,dextents<int32_t,2>,tensor_inline>(a.B,dextents<int32_t,2>(a.N,a.K));
    constexpr auto desc=matmul2d_descriptor(TM,TN,static_cast<int>(dynamic_extent),false,false,false);
    matmul2d<desc,execution_simdgroups<THREADS/32>> op;
    auto left=A.slice(0,tg.y*TM),right=B.slice(tg.x*TN,0);
    auto c=op.template get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
    for(uint16_t i=0;i<c.get_capacity();i++)if(c.is_valid_element(i))c[i]=0.0f;
    op.run(left,right,c);
    for(uint16_t i=0;i<c.get_capacity();i++)if(c.is_valid_element(i)) {
        auto ix=c.get_multidimensional_index(i);int row=tg.y*TM+ix[1],col=tg.x*TN+ix[0];
        if(row>=a.M||col>=a.N)continue;
        long at=(long)row*a.N+col;float prod=c[i];
        // Preserve bf16(bf16(A@B)+D), not a single rounded epilogue.
        if constexpr(sizeof(T)==2)prod=puf_bf16_to_f32(puf_f32_to_bf16(prod));
        puf_st(a.C,at,prod+puf_ld(a.D,at));
    }
}
#define LARGE_LINEAR(SUF,TYPE) \
kernel void wide_large_nt_##SUF(constant WideGemmArgs<TYPE>& a [[buffer(0)]], \
        uint2 tg [[threadgroup_position_in_grid]]) { \
    large_nt_body<TYPE,PUF_LARGE_LINEAR_M,PUF_LARGE_LINEAR_N,PUF_LARGE_LINEAR_THREADS>(a,tg); \
} \
kernel void wide_large_nn_add_##SUF(constant LargeAddArgs<TYPE>& a [[buffer(0)]], \
        uint2 tg [[threadgroup_position_in_grid]]) { \
    large_nn_add_body<TYPE,PUF_LARGE_LINEAR_M,PUF_LARGE_LINEAR_N,PUF_LARGE_LINEAR_THREADS>(a,tg); \
}
LARGE_LINEAR(f32,float)
LARGE_LINEAR(bf16,bfloat)

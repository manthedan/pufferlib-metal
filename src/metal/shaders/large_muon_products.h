// Full ascending-K products with the original NS alpha/beta epilogue.
// A group's interpretation is per matrix: a spatial tile for large matrices,
// or one output per thread for scalar fallbacks. No split-K is introduced.
template<bool UPDATE>
static void large_muon_product(constant MuonWideArgs& a, uint3 tg, uint tid) {
    if (int(tg.z) >= a.count) return;
    constant MuonWideMat& m = a.mats[tg.z];
    constexpr int TM = PUF_LARGE_MUON_PRODUCT_M, TN = PUF_LARGE_MUON_PRODUCT_N;
    constexpr int THREADS = PUF_LARGE_MUON_PRODUCT_THREADS;
    const int M = min(m.R, m.C), rows = UPDATE ? m.R : M, N = UPDATE ? m.C : M;
    device precision_t* src = a.step % 2 == 0 ? m.x : m.xb;
    device precision_t* left = UPDATE ? (m.R > m.C ? src : m.gb) : m.g;
    device precision_t* right = UPDATE ? (m.R > m.C ? m.gb : src) : m.g;
    device precision_t* dst = UPDATE ? (a.step % 2 == 0 ? m.xb : m.x) : m.gb;
    device precision_t* old = UPDATE ? src : m.g;
    const float alpha = UPDATE ? 1.0f : a.c2, beta = UPDATE ? a.c0 : a.c1;
    if (!large_muon_matrix(m)) {
        int e = int(tg.x) * THREADS + int(tid);
        if (e >= rows * N) return;
        int row = e / N, col = e % N;
        float acc = 0.0f;
        for (int k = 0; k < M; ++k)
            acc = fma(to_float(left[row * M + k]), to_float(right[k * N + col]), acc);
        dst[e] = from_float(ns_finish(acc, M, alpha, beta, to_float(old[e])));
        return;
    }
    int cols = (N + TN - 1) / TN;
    int m0 = int(tg.x) / cols * TM, n0 = int(tg.x) % cols * TN;
    if (m0 >= rows || n0 >= N) return;
    if (!UPDATE || ((m.tops >> (a.step % 2)) & 1)) {
        using Tensor = tensor<device precision_t, dextents<int32_t,2>, tensor_inline>;
        auto A = Tensor(left, dextents<int32_t,2>(M, rows));
        auto B = Tensor(right, dextents<int32_t,2>(N, M));
        constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, false, false);
        matmul2d<desc, execution_simdgroups<THREADS/32>> op;
        auto l = A.slice(0, m0), r = B.slice(n0, 0);
        auto c = op.template get_destination_cooperative_tensor<decltype(l),decltype(r),float>();
        for (uint16_t i=0; i<c.get_capacity(); ++i) if(c.is_valid_element(i)) c[i]=0.0f;
        op.run(l,r,c);
        for (uint16_t i=0; i<c.get_capacity(); ++i) if(c.is_valid_element(i)) {
            auto ix=c.get_multidimensional_index(i); int row=m0+ix[1],col=n0+ix[0];
            if(row<rows && col<N) {
                int e=row*N+col;
                dst[e]=from_float(ns_finish(c[i],M,alpha,beta,to_float(old[e])));
            }
        }
        return;
    }
    // Independent unit tests also exercise unaligned direct stage inputs.
    // The production wrapper stages these matrices once for the whole chain.
    int sg=int(tid)/32, lane=int(tid)%32, qid=lane/4;
    int fm=(qid&4)+((lane/2)%4), fn=(qid&2)*2+(lane%2)*2;
    for(int tile=sg; tile<(TM/8)*(TN/8); tile+=THREADS/32) {
        int row=m0+tile/(TN/8)*8,col=n0+tile%(TN/8)*8;
        if(row>=rows || col>=N) continue;
        simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8,8>(0.0f);
        for(int k=0;k<M;k+=8) {
            simdgroup_float8x8 av,bv;
            for(int e=0;e<2;++e) {
                av.thread_elements()[e]=to_float(left[(row+fm)*M+k+fn+e]);
                bv.thread_elements()[e]=to_float(right[(k+fm)*N+col+fn+e]);
            }
            simdgroup_multiply_accumulate(acc,av,bv,acc);
        }
        for(int i=0;i<2;++i) {
            int e=(row+fm)*N+col+fn+i;
            dst[e]=from_float(ns_finish(acc.thread_elements()[i],M,alpha,beta,to_float(old[e])));
        }
    }
}
kernel void wide_large_muon_gram_buf(constant MuonWideArgs& a [[buffer(0)]],
        uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    large_muon_product<false>(a,tg,tid);
}
kernel void wide_large_muon_update(constant MuonWideArgs& a [[buffer(0)]],
        uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    large_muon_product<true>(a,tg,tid);
}

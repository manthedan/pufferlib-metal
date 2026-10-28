// Wide-policy Muon stages; the original scalar kernels remain the oracle.
static inline float ns_finish(float acc, int K, float alpha, float beta, float c_old) {
    if (K % 16) acc = acc + 0.0f;
    float v = alpha * acc;
    if (beta != 0.0f) v = fma(beta, c_old, v);
    return round_p(v);
}

struct MuonWideMat {
    device precision_t* x;   // (R, C) flat-gradient slice, overwritten
    device precision_t* xb;  // (R, C) scratch (the unfused x_buf)
    device precision_t* g;   // (M, M) gram
    device precision_t* gb;  // (M, M) gram_buf
    device float* part;      // 256 norm partials
    float scale;
    int R, C;
    int tops;
};
struct MuonWideArgs {
    MuonWideMat mats[8];
    float c0, c1, c2, eps;
    int count, step, pad0, pad1;
};

// Copyright © 2024 Apple Inc.
// Four spatial 8x8 fragments per 16x16 output tile. Preserve the scalar
// kernel's ascending K chain, alpha/beta finish and BF16 rounding. The
// existing 256-thread dispatch has four inactive SGs on this path; no
// threadgroup staging or barriers are needed for these small cached operands.
// Fragment coordinates adapt MLX Steel BaseMMAFrag<T, 8, 8> (see NOTICE).
static void muon_wide_mma(constant MuonWideArgs& a, uint3 tg, uint tid, bool update) {
    constant MuonWideMat& mat = a.mats[tg.z];
    const int M = min(mat.R, mat.C), N = update ? mat.C : M;
    const int rows = update ? mat.R : M;
    const int sg = int(tid) / 32, lane = int(tid) % 32;
    if (sg >= 4) return;
    const int m0 = int(tg.x) / (N / 16) * 16 + (sg / 2) * 8;
    const int n0 = int(tg.x) % (N / 16) * 16 + (sg % 2) * 8;
    if (m0 >= rows) return;
    device const precision_t* src = a.step % 2 == 0 ? mat.x : mat.xb;
    device const precision_t* left = update ? (mat.R > mat.C ? src : mat.gb) : mat.g;
    device const precision_t* right = update ? (mat.R > mat.C ? mat.gb : src) : mat.g;
    const int lstride = update ? (mat.R > mat.C ? mat.C : M) : M;
    const int qid = lane / 4;
    const int fm = (qid & 4) + ((lane / 2) % 4);
    const int fn = (qid & 2) * 2 + (lane % 2) * 2;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int k = 0; k < M; k += 8) {
        simdgroup_float8x8 av, bv;
        av.thread_elements()[0] = to_float(left[(m0 + fm) * lstride + k + fn]);
        av.thread_elements()[1] = to_float(left[(m0 + fm) * lstride + k + fn + 1]);
        bv.thread_elements()[0] = to_float(right[(k + fm) * N + n0 + fn]);
        bv.thread_elements()[1] = to_float(right[(k + fm) * N + n0 + fn + 1]);
        simdgroup_multiply_accumulate(acc, av, bv, acc);
    }
    device precision_t* dst = update ? (a.step % 2 == 0 ? mat.xb : mat.x) : mat.gb;
    device const precision_t* old = update ? src : mat.g;
    for (int i = 0; i < 2; ++i) {
        int e = (m0 + fm) * N + n0 + fn + i;
        float v = ns_finish(acc.thread_elements()[i], M, update ? 1.0f : a.c2,
            update ? a.c0 : a.c1, to_float(old[e]));
        dst[e] = from_float(v);
    }
}

kernel void wide_muon_gram_buf(constant MuonWideArgs& a [[buffer(0)]],
        uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (int(tg.z) >= a.count) return;
    constant MuonWideMat& m = a.mats[tg.z];
    if ((min(m.R, m.C) == 64 || min(m.R, m.C) == 128 || min(m.R, m.C) == 256) && m.R % 16 == 0 && m.C % 16 == 0) {
        muon_wide_mma(a, tg, tid, false);
        return;
    }
    const int M = min(m.R, m.C);
    const int e = int(tg.x) * 256 + int(tid);
    if (e >= M * M) return;
    const int mi = e / M, ni = e % M;
    device const precision_t* g = m.g;
    float acc = 0.0f;
    for (int k = 0; k < M; ++k) acc = fma(to_float(g[mi * M + k]), to_float(g[k * M + ni]), acc);
    m.gb[e] = from_float(ns_finish(acc, M, a.c2, a.c1, to_float(g[e])));
}

kernel void wide_muon_update(constant MuonWideArgs& a [[buffer(0)]],
        uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (int(tg.z) >= a.count) return;
    constant MuonWideMat& m = a.mats[tg.z];
    if ((min(m.R, m.C) == 64 || min(m.R, m.C) == 128 || min(m.R, m.C) == 256) && m.R % 16 == 0 && m.C % 16 == 0) {
        muon_wide_mma(a, tg, tid, true);
        return;
    }
    const int R = m.R, C = m.C;
    const int e = int(tg.x) * 256 + int(tid);
    if (e >= R * C) return;
    const bool even = a.step % 2 == 0;
    device const precision_t* src = even ? m.x : m.xb;
    device precision_t* dst = even ? m.xb : m.x;
    device const precision_t* gb = m.gb;
    const int mi = e / C, ni = e % C;
    float acc = 0.0f;
    if (R > C) {
        for (int k = 0; k < C; ++k) acc = fma(to_float(src[mi * C + k]), to_float(gb[k * C + ni]), acc);
    } else {
        for (int k = 0; k < R; ++k) acc = fma(to_float(gb[mi * R + k]), to_float(src[k * C + ni]), acc);
    }
    dst[e] = from_float(ns_finish(acc, min(R, C), 1.0f, a.c0, to_float(src[e])));
}


static bool large_muon_matrix(constant MuonWideMat& m) {
    int M = min(m.R, m.C);
    return (m.tops & 4) && (M == 512 || M == 1024) && m.R % 16 == 0 && m.C % 16 == 0;
}

// Retile the independent Gram outputs. Aligned inputs retain tensor_ops;
// large unaligned inputs use scalar fragment loads with the same ascending-K
// MMA chain. Other matrices retain the original aligned/scalar choice.
kernel void wide_large_muon_gram(constant MuonWideArgs& a [[buffer(0)]],
        uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    if (int(tg.z) >= a.count) return;
    constant MuonWideMat& m = a.mats[tg.z];
    constexpr int TM = PUF_LARGE_MUON_GRAM_M, TN = PUF_LARGE_MUON_GRAM_N;
    constexpr int THREADS = PUF_LARGE_MUON_GRAM_THREADS;
    const int M = min(m.R, m.C), K = max(m.R, m.C), C = m.C;
    const bool tall = m.R > m.C;
    device precision_t* src = a.step % 2 == 0 ? m.x : m.xb;
    const bool large = large_muon_matrix(m);
    const bool aligned = ((m.tops >> (a.step % 2)) & 1) != 0;
    if (!large && !aligned) {
        // One scalar output per thread retains the original fallback's
        // concurrency even when a much larger matrix shares this batch.
        int e = int(tg.x) * THREADS + int(tid);
        if (e >= M * M) return;
        int row = e / M, col = e % M;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            float left = to_float(src[tall ? k * C + row : row * C + k]);
            float right = to_float(src[tall ? k * C + col : col * C + k]);
            acc = fma(left, right, acc);
        }
        m.g[e] = from_float(ns_finish(acc, K, 1.0f, 0.0f, 0.0f));
        return;
    }
    const int cols = (M + TN - 1) / TN;
    const int m0 = int(tg.x) / cols * TM, n0 = int(tg.x) % cols * TN;
    if (m0 >= M || n0 >= M) return;
    using Tensor = tensor<device precision_t, dextents<int32_t, 2>, tensor_inline>;
    if (aligned) {
        auto A = tall ? Tensor(src, dextents<int32_t, 2>(M, K))
                      : Tensor(src, dextents<int32_t, 2>(K, M));
        if (tall) {
            constexpr auto desc = matmul2d_descriptor(TM, TN,
                static_cast<int>(dynamic_extent), true, false, false);
            matmul2d<desc, execution_simdgroups<THREADS / 32>> op;
            auto left = A.slice(m0, 0), right = A.slice(n0, 0);
            auto c = op.get_destination_cooperative_tensor<decltype(left), decltype(right), float>();
            for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0.0f;
            op.run(left, right, c);
            for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) {
                auto ix = c.get_multidimensional_index(i);
                if (m0 + ix[1] < M && n0 + ix[0] < M)
                    m.g[(m0 + ix[1]) * M + n0 + ix[0]] = from_float(c[i]);
            }
        } else {
            constexpr auto desc = matmul2d_descriptor(TM, TN,
                static_cast<int>(dynamic_extent), false, true, false);
            matmul2d<desc, execution_simdgroups<THREADS / 32>> op;
            auto left = A.slice(0, m0), right = A.slice(0, n0);
            auto c = op.get_destination_cooperative_tensor<decltype(left), decltype(right), float>();
            for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0.0f;
            op.run(left, right, c);
            for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) {
                auto ix = c.get_multidimensional_index(i);
                if (m0 + ix[1] < M && n0 + ix[0] < M)
                    m.g[(m0 + ix[1]) * M + n0 + ix[0]] = from_float(c[i]);
            }
        }
        return;
    }
    if (large) {
        const int sg = int(tid) / 32, lane = int(tid) % 32;
        const int qid = lane / 4;
        const int fm = (qid & 4) + ((lane / 2) % 4);
        const int fn = (qid & 2) * 2 + (lane % 2) * 2;
        for (int tile = sg; tile < (TM / 8) * (TN / 8); tile += THREADS / 32) {
            const int row = m0 + tile / (TN / 8) * 8, col = n0 + tile % (TN / 8) * 8;
            if (row >= M || col >= M) continue;
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int k = 0; k < K; k += 8) {
                simdgroup_float8x8 av, bv;
                for (int e = 0; e < 2; ++e) {
                    av.thread_elements()[e] = to_float(src[tall ? (k + fn + e) * C + row + fm : (row + fm) * C + k + fn + e]);
                    bv.thread_elements()[e] = to_float(src[tall ? (k + fm) * C + col + fn + e : (col + fn + e) * C + k + fm]);
                }
                simdgroup_multiply_accumulate(acc, av, bv, acc);
            }
            for (int e = 0; e < 2; ++e)
                m.g[(row + fm) * M + col + fn + e] = from_float(acc.thread_elements()[e]);
        }
        return;
    }
}

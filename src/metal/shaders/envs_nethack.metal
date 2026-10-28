// NetHack default encoder and pointer decoder. CUDA: ocean/nethack/nethack.cu.
// Each output follows CUDA's iteration order; scatters use paired 32-bit
// atomics for exactly the same associative 2^24 fixed-point sums.
// precision_t storage, with FP32 arithmetic and explicit CUDA-compatible rounding.
#include "envs_nethack.h"
#include "envs_fxp.h"
static inline float log1p(float x) { return log(1.0f + x); }
static inline void nh_fxp_atomic_add(device long* addr, float v) {
    device atomic_uint* p = (device atomic_uint*)addr;
    fxp_atomic_add64(p, p + 1, fxp_quantize(v));
}
static inline float nh_fxp_to_float(long v) { return fxp_to_float(v); }

static inline int nh_bl_read_i32(device const precision_t* p) {
    return (int)((unsigned int)(int)to_float(p[0])
               | ((unsigned int)(int)to_float(p[1]) << 8)
               | ((unsigned int)(int)to_float(p[2]) << 16)
               | ((unsigned int)(int)to_float(p[3]) << 24));

}

static inline int nh_acc_slot(device const float*  vid,
    device const precision_t*  sfeat, int b, int k) {
    int g = (int)vid[(long)b * NH_INV + k];
    if (g == NH_PAD_GLYPH) return -1;
    if (to_float(sfeat[((long)b * NH_INV + k) * NH_SFEAT + 9]) <= 0.5f) return -1;
    int ot = g - NH_GLYPH_OBJ_OFF, lo, hi, base, cap;
    if (ot >= 178 && ot <= 188) { lo = 178; hi = 188; base = 0; cap = 1; }
    else if (ot >= 207 && ot <= 209) { lo = 207; hi = 209; base = 3; cap = 1; }
    else if (ot >= 150 && ot <= 177) { lo = 150; hi = 177; base = 1; cap = 2; }
    else return -1;
    int n = 0; // ordinal among earlier worn slots of the same kind (first owner wins)
    for (int j = 0; j < k; j++) {
        int gj = (int)vid[(long)b * NH_INV + j];
        if (gj == NH_PAD_GLYPH) continue;
        int oj = gj - NH_GLYPH_OBJ_OFF;
        if (oj >= lo && oj <= hi
            && to_float(sfeat[((long)b * NH_INV + j) * NH_SFEAT + 9]) > 0.5f) n++;
    }
    return n < cap ? base + n : -1;

}

static inline int nh_msg_lc(int c) {
    return (c >= 'A' && c <= 'Z') ? c + 32 : c; // lowercase; keep spaces/punct

}

static inline int nh_msg_hash(int c0, int c1, int c2) {
    unsigned key = ((unsigned)c0 << 16) | ((unsigned)c1 << 8) | (unsigned)c2;
    return (int)((key * 2654435761u) >> (32 - NH_MSG_LOG2V)); // top log2V bits

}

struct nh_bias_relu_kernel_args {
    device precision_t* data;
    device const precision_t* bias;
    int total;
    int dim;
};
kernel void nh_bias_relu_kernel(constant nh_bias_relu_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* data = args_.data;
    device const precision_t* bias = args_.bias;
    int total = args_.total;
    int dim = args_.dim;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    data[idx] = from_float(fmax(0.0f, to_float(data[idx]) + to_float(bias[idx % dim])));

}

struct nh_bias_kernel_args {
    device precision_t* data;
    device const precision_t* bias;
    int total;
    int dim;
};
kernel void nh_bias_kernel(constant nh_bias_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* data = args_.data;
    device const precision_t* bias = args_.bias;
    int total = args_.total;
    int dim = args_.dim;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    data[idx] = from_float(to_float(data[idx]) + to_float(bias[idx % dim]));

}

struct nh_f32_to_precision_kernel_args {
    device precision_t* dst;
    device const float* src;
    int n;
};
kernel void nh_f32_to_precision_kernel(constant nh_f32_to_precision_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* dst = args_.dst;
    device const float* src = args_.src;
    int n = args_.n;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = from_float(src[i]);

}

struct nh_fxp_to_precision_kernel_args {
    device precision_t* dst;
    device const long* src;
    int n;
};
kernel void nh_fxp_to_precision_kernel(constant nh_fxp_to_precision_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* dst = args_.dst;
    device const long* src = args_.src;
    int n = args_.n;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = from_float(nh_fxp_to_float(src[idx]));

}

struct nh_secband_init_kernel_args {
    device uchar* lut;
};
kernel void nh_secband_init_kernel(constant nh_secband_init_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device uchar* lut = args_.lut;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 157 * 41) return;
    int dy = idx / 157 - 20, dx = idx % 157 - 78;
    int ady = dy < 0 ? -dy : dy, adx = dx < 0 ? -dx : dx;
    int cheb = adx > ady ? adx : ady;
    float a = atan2((float)dy, (float)dx) + 3.14159265358979f;
    int sct = ((int)(a / 0.78539816339745f)) & 7;
    int band = cheb < 3 ? 0 : cheb < 7 ? 1 : cheb < 15 ? 2 : 3;
    lut[idx] = (uchar)(sct * 4 + band);

}

struct nh_loc3_gather_kernel_args {
    device precision_t* x;
    device const precision_t* locc_w;
    device const float* crop_glyph;
    device const uchar* lut;
    int B;
};
kernel void nh_loc3_gather_kernel(constant nh_loc3_gather_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* x = args_.x;
    device const precision_t* locc_w = args_.locc_w;
    device const float* crop_glyph = args_.crop_glyph;
    device const uchar* lut = args_.lut;
    int B = args_.B;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_LOC_IN) return;
    int b = t / NH_LOC_IN, j = t % NH_LOC_IN;
    int p = j / NH_LOCC_DIM, d = j % NH_LOCC_DIM;
    int g = (int)crop_glyph[(long)b * NH_CGRID + p];
    int cls = p == (NH_CGRID / 2) ? 7 : (int)lut[g];
    x[t] = locc_w[cls * NH_LOCC_DIM + d];

}

struct nh_loc3_scatter_kernel_args {
    device long* acc;
    device const precision_t* dx;
    device const float* crop_glyph;
    device const uchar* lut;
    int B;
};
kernel void nh_loc3_scatter_kernel(constant nh_loc3_scatter_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device long* acc = args_.acc;
    device const precision_t* dx = args_.dx;
    device const float* crop_glyph = args_.crop_glyph;
    device const uchar* lut = args_.lut;
    int B = args_.B;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_LOC_IN) return;
    int b = t / NH_LOC_IN, j = t % NH_LOC_IN;
    int p = j / NH_LOCC_DIM, d = j % NH_LOCC_DIM;
    int g = (int)crop_glyph[(long)b * NH_CGRID + p];
    int cls = p == (NH_CGRID / 2) ? 7 : (int)lut[g];
    float v = to_float(dx[t]);
    if (v != 0.0f) nh_fxp_atomic_add(&acc[cls * NH_LOCC_DIM + d], v);

}

struct nh_terr_feat_kernel_args {
    device precision_t* tf;
    device const float* gidx;
    device const precision_t* obs;
    device const uchar* lut;
    device const uchar* sblut;
    int B;
};
kernel void nh_terr_feat_kernel(constant nh_terr_feat_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* tf = args_.tf;
    device const float* gidx = args_.gidx;
    device const precision_t* obs = args_.obs;
    device const uchar* lut = args_.lut;
    device const uchar* sblut = args_.sblut;
    int B = args_.B;

    int b = blockIdx.x;
    if (b >= B) return;
    threadgroup atomic_int sec[8 * 4 * 17];
    threadgroup atomic_uint lmkey[12]; // (cheb << 11) | cell: min == serial first-hit
    for (int i = threadIdx.x; i < 8 * 4 * 17; i += blockDim.x) atomic_store_explicit(&sec[i], 0, memory_order_relaxed);
    if (threadIdx.x < 12) atomic_store_explicit(&lmkey[threadIdx.x], 0xFFFFFFFFu, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const precision_t* bl = obs + (long)b * NH_OBS_SIZE + NH_BL_OFF;
    int hx = nh_bl_read_i32(bl), hy = nh_bl_read_i32(bl + 4);
    int hcell = hy * NH_MAPW + hx;
    for (int cell = threadIdx.x; cell < NH_MGRID; cell += blockDim.x) {
        int g = (int)gidx[(long)b * NH_MGRID + cell];
        int tc = cell == hcell ? 13 : (int)lut[g];
        if (tc == 255) continue;
        int dy = cell / NH_MAPW - hy, dx = cell % NH_MAPW - hx;
        if (tc < 12) {
            int ady = dy < 0 ? -dy : dy, adx = dx < 0 ? -dx : dx;
            int cheb = adx > ady ? adx : ady;
            atomic_fetch_min_explicit(&lmkey[tc], ((unsigned int)cheb << 11) | (unsigned int)cell, memory_order_relaxed);
        }
        atomic_fetch_add_explicit(&sec[(int)sblut[(dy + 20) * 157 + (dx + 78)] * 17 + tc], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device precision_t* o = tf + (long)b * NH_TERRF;
    if (threadIdx.x < 12) {
        int t = threadIdx.x;
        unsigned int key = atomic_load_explicit(&lmkey[t], memory_order_relaxed);
        if (key == 0xFFFFFFFFu) {
            o[t * 4 + 0] = from_float(0.0f); o[t * 4 + 1] = from_float(0.0f);
            o[t * 4 + 2] = from_float(0.0f); o[t * 4 + 3] = from_float(0.0f);
        } else {
            int cell = (int)(key & 2047u);
            int cheb = (int)(key >> 11);
            int dy = cell / NH_MAPW - hy, dx = cell % NH_MAPW - hx;
            o[t * 4 + 0] = from_float(1.0f);
            o[t * 4 + 1] = from_float((float)dx * (1.0f / 78.0f));
            o[t * 4 + 2] = from_float((float)dy * (1.0f / 20.0f));
            o[t * 4 + 3] = from_float((float)(cheb < 30 ? cheb : 30) * (1.0f / 30.0f));
        }
    }
    float inv_log = 1.0f / log(1660.0f);
    for (int i = threadIdx.x; i < 8 * 4 * 17; i += blockDim.x)
        o[48 + i] = from_float(log1p((float)atomic_load_explicit(&sec[i], memory_order_relaxed)) * inv_log);

}

struct nh_lab_tok_kernel_args {
    device precision_t* tok;
    device float* gid;
    device const precision_t* obs;
    device const precision_t* e_eff;
    device const uchar* haz;
    int list_off;
    int is_mon;
    int B;
};
kernel void nh_lab_tok_kernel(constant nh_lab_tok_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* tok = args_.tok;
    device float* gid = args_.gid;
    device const precision_t* obs = args_.obs;
    device const precision_t* e_eff = args_.e_eff;
    device const uchar* haz = args_.haz;
    int list_off = args_.list_off;
    int is_mon = args_.is_mon;
    int B = args_.B;

    // thread per (b, token, out dim): pure independent writes -> bit-identical
    // to the serial per-token version by construction.
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_LABK * NH_LAB_IN) return;
    int d = t % NH_LAB_IN;
    int bk = t / NH_LAB_IN;
    int k = bk % NH_LABK;
    device const precision_t* e = obs + (long)(bk / NH_LABK) * NH_OBS_SIZE + list_off + k * NH_V3_MONF;
    int row = (int)to_float(e[0]) | ((int)to_float(e[1]) << 8);
    int g = row <= 0 ? -1
          : is_mon ? row - 1
          : (row < 454 ? NH_OBJ_LO + row - 1 : NH_BODY_OFF + row - 454);
    if (d == 0) gid[bk] = (float)g;
    device precision_t* o = tok + (long)bk * NH_LAB_IN;
    if (d < NH_EMBED_DIM) {
        o[d] = g >= 0 ? e_eff[(long)g * NH_EMBED_DIM + d] : from_float(0.0f);
        return;
    }
    if (d >= 46) { o[d] = from_float(0.0f); return; }
    int dx = (int)to_float(e[2]); if (dx >= 128) dx -= 256;
    int dy = (int)to_float(e[3]); if (dy >= 128) dy -= 256;
    int f4 = (int)to_float(e[4]), f5 = (int)to_float(e[5]), f6 = (int)to_float(e[6]);
    int cheb = abs(dx) > abs(dy) ? abs(dx) : abs(dy);
    float v = 0.0f;
    switch (d) {
    case 32: v = g >= 0 ? fmax(fmin((float)dx * (1.0f / 40.0f), 1.0f), -1.0f) : 0.0f; break;
    case 33: v = g >= 0 ? fmax(fmin((float)dy * (1.0f / 11.0f), 1.0f), -1.0f) : 0.0f; break;
    case 34: v = g >= 0 ? fmin((float)cheb, 15.0f) * (1.0f / 15.0f) : 0.0f; break;
    case 35: v = g >= 0 ? (float)k * (1.0f / 15.0f) : 0.0f; break;
    case 36: v = is_mon ? (g >= 0 && (f4 & 1) ? 1.0f : 0.0f)
                        : (g >= 0 && (f5 & 1) ? 1.0f : 0.0f); break;
    case 37: v = is_mon ? (g >= 0 && (f4 & 8) ? 1.0f : 0.0f)
                        : (g >= 0 && (f5 & 2) ? 1.0f : 0.0f); break;
    case 38: v = is_mon && g >= 0 && (f4 & 4) ? 1.0f : 0.0f; break;
    case 39: v = is_mon && g >= 0 && cheb <= 1 ? 1.0f : 0.0f; break;
    case 40: v = is_mon && g >= 0 ? fmin((float)f5 * 0.04f, 1.0f) : 0.0f; break;
    case 41: v = is_mon && g >= 0 ? fmin((float)f6 * (1.0f / 24.0f), 1.0f) : 0.0f; break;
    case 42: case 43: case 44: case 45: {
        int hb = is_mon && row > 0 && haz != nullptr ? (int)haz[(row - 1) % 381] : 0;
        v = (hb & (1 << (d - 42))) ? 1.0f : 0.0f; break;
    }
    }
    o[d] = from_float(v);

}

struct nh_lab_dE_scatter_kernel_args {
    device long* dE_i;
    device const precision_t* dtok;
    device const float* gid;
    int B;
};
kernel void nh_lab_dE_scatter_kernel(constant nh_lab_dE_scatter_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device long* dE_i = args_.dE_i;
    device const precision_t* dtok = args_.dtok;
    device const float* gid = args_.gid;
    int B = args_.B;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_LABK * NH_EMBED_DIM) return;
    int bk = t / NH_EMBED_DIM, d = t % NH_EMBED_DIM;
    int g = (int)gid[bk];
    if (g < 0) return;
    float v = to_float(dtok[(long)bk * NH_LAB_IN + d]);
    if (v != 0.0f) nh_fxp_atomic_add(&dE_i[(long)g * NH_EMBED_DIM + d], v);

}

struct nh_decode_kernel_args {
    device float* idx;
    device const precision_t* obs;
    int B;
};
kernel void nh_decode_kernel(constant nh_decode_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device float* idx = args_.idx;
    device const precision_t* obs = args_.obs;
    int B = args_.B;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_MGRID) return;
    int b = t / NH_MGRID, cell = t % NH_MGRID;
    device const precision_t* src = obs + (long)b * NH_OBS_SIZE + 2 * cell;
    int g = (int)to_float(src[0]) | ((int)to_float(src[1]) << 8);
    idx[t] = (float)max(0, min(g, NH_GLYPH_VOCAB - 1));

}

struct nh_crop_kernel_args {
    device float* crop;
    device const float* idx;
    device const precision_t* obs;
    int B;
};
kernel void nh_crop_kernel(constant nh_crop_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device float* crop = args_.crop;
    device const float* idx = args_.idx;
    device const precision_t* obs = args_.obs;
    int B = args_.B;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_CGRID) return;
    int b = t / NH_CGRID, p = t % NH_CGRID;
    device const precision_t* bl = obs + (long)b * NH_OBS_SIZE + NH_BL_OFF;
    int r = nh_bl_read_i32(bl + 4) - NH_CHALF + p / NH_CROP; // blstats[1] = y
    int c = nh_bl_read_i32(bl)     - NH_CHALF + p % NH_CROP; // blstats[0] = x
    crop[t] = (r < 0 || r >= NH_MAPH || c < 0 || c >= NH_MAPW)
        ? (float)NH_PAD_GLYPH : idx[b * NH_MGRID + r * NH_MAPW + c];

}

struct nh_eff_embed_kernel_args {
    device precision_t* out;
    device const precision_t* E;
    device const precision_t* K;
    device const precision_t* S;
};
kernel void nh_eff_embed_kernel(constant nh_eff_embed_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* out = args_.out;
    device const precision_t* E = args_.E;
    device const precision_t* K = args_.K;
    device const precision_t* S = args_.S;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NH_GLYPH_VOCAB * NH_EMBED_DIM) return;
    int g = i / NH_EMBED_DIM, d = i % NH_EMBED_DIM;
    out[i] = from_float(to_float(E[i])
        + to_float(K[nh_glyph_kind[g] * NH_EMBED_DIM + d])
        + to_float(S[nh_glyph_sub[g] * NH_EMBED_DIM + d]));

}

struct nh_ekind_grad_kernel_args {
    device precision_t* dK;
    device const precision_t* dE;
};
kernel void nh_ekind_grad_kernel(constant nh_ekind_grad_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* dK = args_.dK;
    device const precision_t* dE = args_.dE;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NH_NKIND * NH_EMBED_DIM) return;
    int r = i / NH_EMBED_DIM, d = i % NH_EMBED_DIM;
    float acc = 0.0f;
    for (int j = nh_kind_csr_off[r]; j < nh_kind_csr_off[r + 1]; j++)
        acc += to_float(dE[(long)nh_kind_csr_glyph[j] * NH_EMBED_DIM + d]);
    dK[i] = from_float(acc);

}

struct nh_esub_grad_kernel_args {
    device precision_t* dS;
    device const precision_t* dE;
};
kernel void nh_esub_grad_kernel(constant nh_esub_grad_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* dS = args_.dS;
    device const precision_t* dE = args_.dE;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NH_NSUB * NH_EMBED_DIM) return;
    int r = i / NH_EMBED_DIM, d = i % NH_EMBED_DIM;
    float acc = 0.0f;
    for (int j = nh_sub_csr_off[r]; j < nh_sub_csr_off[r + 1]; j++)
        acc += to_float(dE[(long)nh_sub_csr_glyph[j] * NH_EMBED_DIM + d]);
    dS[i] = from_float(acc);

}

struct nh_min_summax_kernel_args {
    device precision_t* concat;
    device float* amax;
    device const precision_t* v;
    device const float* vid;
    float eps;
    int n;
    int padv;
    int off;
    int B;
};
kernel void nh_min_summax_kernel(constant nh_min_summax_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* concat = args_.concat;
    device float* amax = args_.amax;
    device const precision_t* v = args_.v;
    device const float* vid = args_.vid;
    float eps = args_.eps;
    int n = args_.n;
    int padv = args_.padv;
    int off = args_.off;
    int B = args_.B;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_MV) return;
    int b = t / NH_MV, d = t % NH_MV;
    float sm = 0.0f, mx = -1e30f;
    int bm = -1;
    for (int k = 0; k < n; k++) {
        float id = vid[(long)b * n + k];
        int bad = padv >= 0 ? ((int)id == padv) : (id < 0.0f);
        if (bad) continue;
        float vv = to_float(v[((long)b * n + k) * NH_MV + d]);
        sm += vv;
        if (vv > mx) { mx = vv; bm = k; }
    }
    amax[t] = (float)bm;
    device precision_t* dst = concat + (long)b * NH_CONCAT + off;
    dst[d] = from_float(sm * eps);
    dst[NH_MV + d] = from_float(bm >= 0 ? mx : 0.0f);

}

struct nh_min_dv_kernel_args {
    device precision_t* dv;
    device const precision_t* grad_concat;
    device const float* amax;
    device const float* vid;
    float eps;
    int n;
    int padv;
    int off;
    int B;
};
kernel void nh_min_dv_kernel(constant nh_min_dv_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* dv = args_.dv;
    device const precision_t* grad_concat = args_.grad_concat;
    device const float* amax = args_.amax;
    device const float* vid = args_.vid;
    float eps = args_.eps;
    int n = args_.n;
    int padv = args_.padv;
    int off = args_.off;
    int B = args_.B;

    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)B * n * NH_MV) return;
    int d = t % NH_MV;
    int k = (t / NH_MV) % n;
    int b = t / ((long)n * NH_MV);
    float id = vid[(long)b * n + k];
    int bad = padv >= 0 ? ((int)id == padv) : (id < 0.0f);
    device const precision_t* gc = grad_concat + (long)b * NH_CONCAT + off;
    float g = 0.0f;
    if (!bad) {
        g = eps * to_float(gc[d]);
        if ((int)amax[(long)b * NH_MV + d] == k) g += to_float(gc[NH_MV + d]);
    }
    dv[t] = from_float(g);

}

struct nh_min_sgate_kernel_args {
    device precision_t* concat;
    device const precision_t* rp;
    device const precision_t* tok;
    device const float* gid;
    int underfoot;
    int off;
    int B;
};
kernel void nh_min_sgate_kernel(constant nh_min_sgate_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* concat = args_.concat;
    device const precision_t* rp = args_.rp;
    device const precision_t* tok = args_.tok;
    device const float* gid = args_.gid;
    int underfoot = args_.underfoot;
    int off = args_.off;
    int B = args_.B;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_INV_HID) return;
    int b = t / NH_INV_HID, d = t % NH_INV_HID;
    device precision_t* dst = concat + (long)b * NH_CONCAT + off;
    float g0 = gid[(long)b * NH_LABK] >= 0.0f
        ? to_float(rp[((long)b * NH_LABK) * NH_INV_HID + d]) : 0.0f;
    dst[d] = from_float(g0);
    if (underfoot) {
        float uf = 0.0f;
        for (int k = 0; k < NH_LABK; k++) {
            if (gid[(long)b * NH_LABK + k] < 0.0f) continue;
            if (to_float(tok[((long)b * NH_LABK + k) * NH_LAB_IN + 36]) > 0.5f)
                uf += to_float(rp[((long)b * NH_LABK + k) * NH_INV_HID + d]);
        }
        dst[NH_INV_HID + d] = from_float(uf);
    }

}

struct nh_min_sgate_bwd_kernel_args {
    device precision_t* drp;
    device const precision_t* grad_concat;
    device const precision_t* tok;
    device const float* gid;
    int underfoot;
    int off;
    int B;
};
kernel void nh_min_sgate_bwd_kernel(constant nh_min_sgate_bwd_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* drp = args_.drp;
    device const precision_t* grad_concat = args_.grad_concat;
    device const precision_t* tok = args_.tok;
    device const float* gid = args_.gid;
    int underfoot = args_.underfoot;
    int off = args_.off;
    int B = args_.B;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_INV_HID) return;
    int b = t / NH_INV_HID, d = t % NH_INV_HID;
    device const precision_t* gc = grad_concat + (long)b * NH_CONCAT + off;
    if (gid[(long)b * NH_LABK] >= 0.0f) {
        device precision_t* dr = drp + ((long)b * NH_LABK) * NH_INV_HID + d;
        *dr = from_float(to_float(*dr) + to_float(gc[d]));
    }
    if (underfoot) {
        for (int k = 0; k < NH_LABK; k++) {
            if (gid[(long)b * NH_LABK + k] < 0.0f) continue;
            if (to_float(tok[((long)b * NH_LABK + k) * NH_LAB_IN + 36]) > 0.5f) {
                device precision_t* dr = drp + ((long)b * NH_LABK + k) * NH_INV_HID + d;
                *dr = from_float(to_float(*dr) + to_float(gc[NH_INV_HID + d]));
            }
        }
    }

}

struct nh_pass_kernel_args {
    device precision_t* concat;
    device const precision_t* inv_out;
    device const precision_t* sfeat;
    device const float* vid;
    int nopass;
    int passoff;
    int B;
};
kernel void nh_pass_kernel(constant nh_pass_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* concat = args_.concat;
    device const precision_t* inv_out = args_.inv_out;
    device const precision_t* sfeat = args_.sfeat;
    device const float* vid = args_.vid;
    int nopass = args_.nopass;
    int passoff = args_.passoff;
    int B = args_.B;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_PASS_DIM) return;
    int b = t / NH_PASS_DIM, d = t % NH_PASS_DIM;
    device precision_t* dst = concat + (long)b * NH_CONCAT + passoff + d;
    *dst = from_float(0.0f);
    if (nopass == 7) return;
    if (d >= 80 + NH_PWORN_DIM) { // accessory panel: first slot owning index a
        if (nopass & 4) return;
        int j3 = d - 80 - NH_PWORN_DIM;
        int a = j3 / (NH_INV_HID + NH_SFEAT), e = j3 % (NH_INV_HID + NH_SFEAT);
        for (int k = 0; k < NH_INV; k++) {
            if (nh_acc_slot(vid, sfeat, b, k) != a) continue;
            device const precision_t* f = sfeat + ((long)b * NH_INV + k) * NH_SFEAT;
            device const precision_t* r = inv_out + ((long)b * NH_INV + k) * NH_INV_HID;
            *dst = e < NH_INV_HID ? r[e] : f[e - NH_INV_HID];
            return;
        }
        return;
    }
    int region = d < 40 ? 0 : d < 80 ? 1 : 2;   // wield | quiver | worn
    int j = region == 0 ? d : region == 1 ? d - 40 : d - 80;
    if (region == 0 && (nopass & 1)) return;
    if (region == 1 && (nopass & 2)) return;
    if (region == 2 && (nopass & 4)) return;
    int fbit = region == 0 ? 10 : region == 1 ? 12 : 9;
    float wsum = 0.0f; int nworn = 0;
    for (int k = 0; k < NH_INV; k++) {
        if ((int)vid[(long)b * NH_INV + k] == NH_PAD_GLYPH) continue;
        device const precision_t* f = sfeat + ((long)b * NH_INV + k) * NH_SFEAT;
        if (to_float(f[fbit]) <= 0.5f) continue;
        device const precision_t* r = inv_out + ((long)b * NH_INV + k) * NH_INV_HID;
        if (region == 2) { nworn++; wsum += to_float(r[j]); }
        else {
            float v = j < NH_INV_HID ? to_float(r[j]) : to_float(f[j - NH_INV_HID]);
            *dst = from_float(to_float(*dst) + v);
        }
    }
    if (region == 2 && nworn > 0) *dst = from_float(wsum / (float)nworn);

}

struct nh_passbwd_kernel_args {
    device precision_t* inv_grad;
    device const precision_t* grad_concat;
    device const precision_t* sfeat;
    device const float* vid;
    int nopass;
    int passoff;
    int B;
};
kernel void nh_passbwd_kernel(constant nh_passbwd_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* inv_grad = args_.inv_grad;
    device const precision_t* grad_concat = args_.grad_concat;
    device const precision_t* sfeat = args_.sfeat;
    device const float* vid = args_.vid;
    int nopass = args_.nopass;
    int passoff = args_.passoff;
    int B = args_.B;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_INV * NH_INV_HID) return;
    int d = t % NH_INV_HID;
    int k = (t / NH_INV_HID) % NH_INV;
    int b = t / (NH_INV * NH_INV_HID);
    if ((int)vid[(long)b * NH_INV + k] == NH_PAD_GLYPH) return;
    device const precision_t* gp = grad_concat + (long)b * NH_CONCAT + passoff;
    device const precision_t* f = sfeat + ((long)b * NH_INV + k) * NH_SFEAT;
    device precision_t* dr = inv_grad + ((long)b * NH_INV + k) * NH_INV_HID + d;
    if (!(nopass & 1) && to_float(f[10]) > 0.5f)
        *dr = from_float(to_float(*dr) + to_float(gp[d]));
    if (!(nopass & 2) && to_float(f[12]) > 0.5f)
        *dr = from_float(to_float(*dr) + to_float(gp[40 + d]));
    if (!(nopass & 4) && to_float(f[9]) > 0.5f) {
        int nworn = 0;
        for (int j = 0; j < NH_INV; j++) {
            if ((int)vid[(long)b * NH_INV + j] == NH_PAD_GLYPH) continue;
            if (to_float(sfeat[((long)b * NH_INV + j) * NH_SFEAT + 9]) > 0.5f) nworn++;
        }
        if (nworn > 0)
            *dr = from_float(to_float(*dr) + to_float(gp[80 + d]) / (float)nworn);
    }
    if (!(nopass & 4)) {
        int a = nh_acc_slot(vid, sfeat, b, k);
        if (a >= 0)
            *dr = from_float(to_float(*dr)
                + to_float(gp[80 + NH_PWORN_DIM + a * (NH_INV_HID + NH_SFEAT) + d]));
    }

}

struct nh_blstats_kernel_args {
    device precision_t* out;
    device const precision_t* obs;
    int B;
};
kernel void nh_blstats_kernel(constant nh_blstats_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* out = args_.out;
    device const precision_t* obs = args_.obs;
    int B = args_.B;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int b = tid / 32, lane = tid % 32;
    if (b >= B) return;
    device const precision_t* src = obs + (long)b * NH_OBS_SIZE + NH_BL_OFF;
    device const precision_t* ex = src + 4 * NH_BL_RAW;
    device precision_t* dst = out + (long)b * NH_BL_FEAT;
    for (int j = lane; j < NH_BL_FEAT; j += 32) {
        float f;
        if (j < NH_F_HUNGER) {
            // 25 scaled scalars: blstats minus hunger(21) and condition(25)
            int i = j + (j >= 21) + (j >= 24);
            int v = nh_bl_read_i32(src + 4*i);
            f = NH_BL_ISLOG[i] ? log1p(fmax((float)v, 0.0f)) * NH_BL_SCALE[i]
                               : (float)v * NH_BL_SCALE[i];
        } else if (j < NH_F_COND) {
            int v = nh_bl_read_i32(src + 4*NH_BL_HUNGER);
            f = (j - NH_F_HUNGER == max(0, min(v, 6))) ? 1.0f : 0.0f;
        } else if (j < NH_F_PREV) {
            unsigned int cond = (unsigned int)nh_bl_read_i32(src + 4*NH_BL_CONDITION);
            f = (float)((cond >> (j - NH_F_COND)) & 1u);
        } else if (j < NH_F_INV) {
            f = (j - NH_F_PREV == nh_bl_read_i32(ex + 4)) ? 1.0f : 0.0f;
        } else if (j < NH_F_FRAC) {
            f = (float)nh_bl_read_i32(ex + 4*(2 + j - NH_F_INV)) * 0.125f;
        } else if (j < NH_F_DNUM) {
            // hp_frac / ene_frac in [0,1]: the "how close to death/empty" ratio
            int base = (j == NH_F_FRAC) ? NH_BL_HP : NH_BL_ENE;
            int cur = nh_bl_read_i32(src + 4*base);
            int mx = nh_bl_read_i32(src + 4*(base + 1));
            f = fmin(fmax((float)cur / (float)(mx > 1 ? mx : 1), 0.0f), 1.0f);
        } else if (j < NH_F_ENGR) {
            int v = nh_bl_read_i32(src + 4*NH_BL_DNUM);
            f = (j - NH_F_DNUM == max(0, min(v, 7))) ? 1.0f : 0.0f;
        } else if (j < NH_F_SHOP) {
            // underfoot engraving from ex[0]: any engraving, active Elbereth
            int v = nh_bl_read_i32(ex);
            f = (j == NH_F_ENGR) ? (v >= 1 ? 1.0f : 0.0f) : (v >= 2 ? 1.0f : 0.0f);
        } else if (j < NH_F_SPELL) {
            // standing on shop goods, and gold/price capped at 1
            int v = nh_bl_read_i32(ex + 4*(NH_EXTRA_SHOP + (j - NH_F_SHOP)));
            f = (j == NH_F_SHOP) ? (float)v : (float)v * 0.01f;
        } else if (j < NH_F_WEIGHT) {
            // known-spell count/8; per-slot content rides the spell-key path
            int v = nh_bl_read_i32(ex + 4*(NH_EXTRA_SHOP + 2));
            f = (float)v * 0.125f;
        } else if (j < NH_F_ROLE) {
            // encumbrance: softsign(ratio-1) is 0 at the wall, keeps gradient
            // through Overloaded (~3x); capacity /1000 (engine caps there)
            int v = nh_bl_read_i32(ex + 4*(NH_EXTRA_SHOP + 2 + 1 + 4 * NH_SPELL_SLOTS
                                           + (j - NH_F_WEIGHT)));
            if (j == NH_F_WEIGHT) {
                float d = (float)v * 0.01f - 1.0f;
                f = d / (1.0f + fabs(d));
            } else
                f = (float)v * 0.001f;
        } else if (j < NH_F_INTRINS) {
            // role/race/gender one-hots, already 0/1
            // dead under NH_ID_EMBED: identity flows via the embed channel
            f = 0.0f; // identity flows via the embed channel
        } else {
            f = (float)((nh_bl_read_i32(ex + 4*NH_EX_INTRINS) >> (j - NH_F_INTRINS)) & 1);
        }
        // strict [-1,1]: bounds deep-play excursions (AC -15 -> -1.5, hp 300 ->
        // 1.5, stacked inv counts) — validated neutral-now, deep-safe (n=4)
        dst[j] = from_float(fmin(fmax(f, -1.0f), 1.0f));
    }

}

struct nh_msg_ids_kernel_args {
    device float* ids;
    device const precision_t* obs;
    int B;
};
kernel void nh_msg_ids_kernel(constant nh_msg_ids_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device float* ids = args_.ids;
    device const precision_t* obs = args_.obs;
    int B = args_.B;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * NH_MSG_LEN) return;
    int b = i / NH_MSG_LEN, t = i % NH_MSG_LEN;
    if (t > NH_MSG_LEN - 3) {
        ids[i] = -1.0f;
        return;
    }
    device const precision_t* m = obs + (long)b * NH_OBS_SIZE + NH_MSG_OFF;
    int c0 = (int)to_float(m[t]), c1 = (int)to_float(m[t + 1]), c2 = (int)to_float(m[t + 2]);
    if (c0 == 0 || c1 == 0 || c2 == 0) {
        ids[i] = -1.0f;
        return;
    }
    ids[i] = (float)nh_msg_hash(nh_msg_lc(c0), nh_msg_lc(c1), nh_msg_lc(c2));

}

struct nh_msg_pool_kernel_args {
    device precision_t* out;
    device const precision_t* msg_w;
    device const float* ids;
    int B;
};
kernel void nh_msg_pool_kernel(constant nh_msg_pool_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* out = args_.out;
    device const precision_t* msg_w = args_.msg_w;
    device const float* ids = args_.ids;
    int B = args_.B;

    int b = blockIdx.x;
    if (b >= B) return;
    int d = threadIdx.x;
    device const float* mi = ids + (long)b * NH_MSG_LEN;
    int count = 0; float acc = 0.0f;
    for (int t = 0; t < NH_MSG_LEN; t++) {
        int id = (int)mi[t];
        if (id < 0) break;
        count++;
        acc += to_float(msg_w[(long)id * NH_MSG_HID + d]);
    }
    out[(long)b * NH_MSG_HID + d] = from_float(acc * rsqrt((float)count + 1.0f));

}

struct nh_msg_bwd_kernel_args {
    device long* dmsg_acc;
    device const precision_t* grad_concat;
    device const float* ids;
    int B;
};
kernel void nh_msg_bwd_kernel(constant nh_msg_bwd_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device long* dmsg_acc = args_.dmsg_acc;
    device const precision_t* grad_concat = args_.grad_concat;
    device const float* ids = args_.ids;
    int B = args_.B;

    int b = blockIdx.x;
    if (b >= B) return;
    int d = threadIdx.x;
    device const float* mi = ids + (long)b * NH_MSG_LEN;
    int count = 0;
    for (int t = 0; t < NH_MSG_LEN; t++) {
        if ((int)mi[t] < 0) break;
        count++;
    }
    float g = to_float(grad_concat[(long)b * NH_CONCAT + NH_MSG_CONCAT_OFF + d])
              * rsqrt((float)count + 1.0f);
    if (g == 0.0f) return;
    for (int t = 0; t < count; t++)
        nh_fxp_atomic_add(&dmsg_acc[(long)(int)mi[t] * NH_MSG_HID + d], g);

}

struct nh_spkey_kernel_args {
    device precision_t* keys;
    device precision_t* sp_in;
    device float* sp_idx;
    device const precision_t* spk_w;
    device const precision_t* e_eff;
    device const precision_t* obs;
    int B;
};
kernel void nh_spkey_kernel(constant nh_spkey_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* keys = args_.keys;
    device precision_t* sp_in = args_.sp_in;
    device float* sp_idx = args_.sp_idx;
    device const precision_t* spk_w = args_.spk_w;
    device const precision_t* e_eff = args_.e_eff;
    device const precision_t* obs = args_.obs;
    int B = args_.B;

    // thread per (b, slot, key row): rebuilds the float input (cached loads),
    // computes one matvec row in the original MAC order -> bit-identical.
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_SPELL_SLOTS * NH_SPKEY) return;
    int r = t % NH_SPKEY;
    int bs = t / NH_SPKEY;
    int b = bs / NH_SPELL_SLOTS, s = bs % NH_SPELL_SLOTS;
    device const precision_t* src = obs + (long)b * NH_OBS_SIZE + NH_BL_OFF
                           + 4 * (NH_BL_RAW + NH_EXTRA_SHOP + 2 + 1 + 4 * s);
    int id = (int)to_float(src[0]) | ((int)to_float(src[1]) << 8);
    int g = id > 0 ? min(id + 1906, NH_GLYPH_VOCAB - 1) : -1;
    float in[NH_SPIN];
    for (int d = 0; d < NH_EMBED_DIM; d++)
        in[d] = g >= 0 ? to_float(e_eff[(long)g * NH_EMBED_DIM + d]) : 0.0f;
    int lev = (int)to_float(src[4]) | ((int)to_float(src[5]) << 8);
    int fail = (int)to_float(src[8]) | ((int)to_float(src[9]) << 8);
    int know = (int)to_float(src[12]) | ((int)to_float(src[13]) << 8)
             | ((int)to_float(src[14]) << 16);
    in[NH_EMBED_DIM + 0] = id > 0 ? 1.0f : 0.0f;
    in[NH_EMBED_DIM + 1] = fmin((float)lev * 0.142857f, 1.0f);
    in[NH_EMBED_DIM + 2] = fmin((float)fail * 0.01f, 1.0f);
    in[NH_EMBED_DIM + 3] = fmin((float)know * 0.00005f, 1.0f);
    if (r == 0) {
        sp_idx[(long)bs] = (float)g;
        device precision_t* inb = sp_in + (long)bs * NH_SPIN;
        for (int c = 0; c < NH_SPIN; c++) inb[c] = from_float(in[c]);
    }
    float acc = 0.0f;
    for (int c = 0; c < NH_SPIN; c++)
        acc += to_float(spk_w[r * NH_SPIN + c]) * in[c];
    keys[(long)bs * NH_SPKEY + r] = from_float(fmax(acc, 0.0f));

}

struct nh_sp_doorstep_kernel_args {
    device precision_t* concat;
    device const precision_t* obs;
    int B;
};
kernel void nh_sp_doorstep_kernel(constant nh_sp_doorstep_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* concat = args_.concat;
    device const precision_t* obs = args_.obs;
    int B = args_.B;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * 4) return;
    int b = t / 4, d = 2 * NH_SPM + t % 4;
    float v;
    {
        device const precision_t* ex = obs + (long)b * NH_OBS_SIZE + NH_BL_OFF + NH_BL_RAW * 4;
        int mf = 100, ml = 0, n = 0, mr = 20000;
        for (int s = 0; s < NH_SPELL_SLOTS; s++) {
            device const precision_t* q = ex + 4 * (NH_EXTRA_SHOP + 2 + 1 + 4 * s);
            int sid = nh_bl_read_i32(q);
            if (sid <= 0) continue;
            n++;
            int lv = nh_bl_read_i32(q + 4), fl = nh_bl_read_i32(q + 8), kn = nh_bl_read_i32(q + 12);
            if (fl < mf) mf = fl;
            if (lv > ml) ml = lv;
            if (kn < mr) mr = kn;
        }
        int j = d - 2 * NH_SPM;
        v = j == 0 ? (float)mf * 0.01f
          : j == 1 ? (float)ml * (1.0f / 7.0f)
          : j == 2 ? (float)(n > 8 ? 8 : n) * 0.125f
                   : (float)mr * 0.00005f;
        v = fmin(fmax(v, 0.0f), 1.0f);
    }
    concat[(long)b * NH_CONCAT + NH_SPELL_CONCAT_OFF + d] = from_float(v);

}

struct nh_sp2_dk_kernel_args {
    device precision_t* dkeys;
    device const precision_t* ptr_dkeys;
    device const precision_t* keys;
    int B;
};
kernel void nh_sp2_dk_kernel(constant nh_sp2_dk_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* dkeys = args_.dkeys;
    device const precision_t* ptr_dkeys = args_.ptr_dkeys;
    device const precision_t* keys = args_.keys;
    int B = args_.B;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_SPELL_SLOTS * NH_SPKEY) return;
    float v = to_float(dkeys[t]) + (ptr_dkeys ? to_float(ptr_dkeys[t]) : 0.0f);
    if (to_float(keys[t]) <= 0.0f) v = 0.0f;
    dkeys[t] = from_float(v);

}

struct nh_spkey_dE_kernel_args {
    device long* dE_i;
    device const precision_t* dkeys;
    device const precision_t* spk_w;
    device const float* sp_idx;
    int B;
};
kernel void nh_spkey_dE_kernel(constant nh_spkey_dE_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device long* dE_i = args_.dE_i;
    device const precision_t* dkeys = args_.dkeys;
    device const precision_t* spk_w = args_.spk_w;
    device const float* sp_idx = args_.sp_idx;
    int B = args_.B;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_SPELL_SLOTS * NH_EMBED_DIM) return;
    int bs = t / NH_EMBED_DIM;
    int d = t % NH_EMBED_DIM;
    int g = (int)sp_idx[bs];
    if (g < 0) return;
    float acc = 0.0f;
    for (int r = 0; r < NH_SPKEY; r++)
        acc += to_float(spk_w[r * NH_SPIN + d]) * to_float(dkeys[(long)bs * NH_SPKEY + r]);
    if (acc != 0.0f) nh_fxp_atomic_add(&dE_i[(long)g * NH_EMBED_DIM + d], acc);

}

struct nh_concat_kernel_args {
    device precision_t* out;
    device const precision_t* loc;
    device const precision_t* glb;
    device const precision_t* inv;
    device const precision_t* bl_out;
    device const precision_t* bl_feats;
    device const precision_t* msg;
    int B;
};
kernel void nh_concat_kernel(constant nh_concat_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* out = args_.out;
    device const precision_t* loc = args_.loc;
    device const precision_t* glb = args_.glb;
    device const precision_t* inv = args_.inv;
    device const precision_t* bl_out = args_.bl_out;
    device const precision_t* bl_feats = args_.bl_feats;
    device const precision_t* msg = args_.msg;
    int B = args_.B;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH_CONCAT) return;
    int b = idx / NH_CONCAT, c = idx % NH_CONCAT;
    precision_t val;
    if (c < NH_LOC_HID)
        val = loc[(long)b * NH_LOC_HID + c];
    else if (c < NH_LOC_HID + NH_GLB_HID)
        val = glb[(long)b * NH_GLB_HID + (c - NH_LOC_HID)];
    else if (c < NH_LOC_HID + NH_GLB_HID + NH_INVP_DIM)
        val = inv[(long)b * NH_INV_POOL + (c - NH_LOC_HID - NH_GLB_HID)];
    else if (c < NH_LOC_HID + NH_GLB_HID + NH_INVP_DIM + NH_BL_HID)
        val = bl_out[(long)b * NH_BL_HID + (c - NH_LOC_HID - NH_GLB_HID - NH_INVP_DIM)];
    else if (c < NH_MSG_CONCAT_OFF)
        val = bl_feats[(long)b * NH_BL_FEAT + (c - NH_LOC_HID - NH_GLB_HID - NH_INVP_DIM - NH_BL_HID)];
    else if (c < NH_SPELL_CONCAT_OFF)
        val = msg[(long)b * NH_MSG_HID + (c - NH_MSG_CONCAT_OFF)];
    else
        return; // spell slice: nh_spell_gather_kernel fills it afterwards
    out[idx] = val;

}

struct nh_slice_kernel_args {
    device precision_t* dst;
    device const precision_t* src;
    int B;
    int stride;
    int offset;
    int n;
};
kernel void nh_slice_kernel(constant nh_slice_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* dst = args_.dst;
    device const precision_t* src = args_.src;
    int B = args_.B;
    int stride = args_.stride;
    int offset = args_.offset;
    int n = args_.n;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * n) return;
    dst[idx] = src[(idx / n) * stride + offset + idx % n];

}

struct nh_inv_decode_kernel_args {
    device float* idx;
    device const precision_t* obs;
    int B;
    int off;
};
kernel void nh_inv_decode_kernel(constant nh_inv_decode_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device float* idx = args_.idx;
    device const precision_t* obs = args_.obs;
    int B = args_.B;
    int off = args_.off;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_INV) return;
    int b = t / NH_INV, s = t % NH_INV;
    device const precision_t* src = obs + (long)b * NH_OBS_SIZE + off + 2 * s;
    int g = (int)to_float(src[0]) | ((int)to_float(src[1]) << 8);
    idx[t] = (float)max(0, min(g, NH_GLYPH_VOCAB - 1));

}

struct nh_inv_table_kernel_args {
    device precision_t* T;
    device const precision_t* E;
    device const precision_t* w1;
};
kernel void nh_inv_table_kernel(constant nh_inv_table_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* T = args_.T;
    device const precision_t* E = args_.E;
    device const precision_t* w1 = args_.w1;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NH_GLYPH_VOCAB * NH_INV_HID) return;
    int g = i / NH_INV_HID, k = i % NH_INV_HID;
    float acc = 0.0f;
    for (int d = 0; d < NH_EMBED_DIM; d++)
        acc += to_float(E[g * NH_EMBED_DIM + d]) * to_float(w1[k * NH_EMBED_DIM + d]);
    T[i] = from_float(acc);

}

struct nh_inv_sfeat_kernel_args {
    device precision_t* out;
    device const precision_t* obs;
    int B;
};
kernel void nh_inv_sfeat_kernel(constant nh_inv_sfeat_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* out = args_.out;
    device const precision_t* obs = args_.obs;
    int B = args_.B;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_INV) return;
    int b = t / NH_INV, s = t % NH_INV;
    device const precision_t* src = obs + (long)b * NH_OBS_SIZE + NH_INVST_OFF + s * NH_ST_RAW;
    int st[NH_ST_RAW];
    for (int j = 0; j < NH_ST_RAW; j++) {
        int v = (int)to_float(src[j]);
        st[j] = v >= 128 ? v - 256 : v; // bytes -> int8
    }
    device precision_t* f = out + (long)t * NH_SFEAT;
    for (int c = 0; c < 4; c++) f[c] = from_float(st[0] == c ? 1.0f : 0.0f);
    int spe_known = st[1] != -128;
    f[4] = from_float((float)spe_known);
    // strict [-1,1]: spe clamp/7; quantity LINEAR capped (log-squash
    // regressed ammo-count .88->.76 across two seeds — counts want scale)
    f[5] = from_float(spe_known ? fmax(fmin((float)st[1], 7.0f), -7.0f) * (1.0f / 7.0f) : 0.0f);
    f[6] = from_float(fmin(fmax((float)st[2], 0.0f), 30.0f) * (1.0f / 30.0f));
    f[7] = from_float((float)st[3] * (1.0f / 3.0f));
    f[8] = from_float((float)st[4] * (1.0f / 3.0f));
    for (int c = 0; c < 7; c++)
        f[9 + c] = from_float((float)((st[5] >> c) & 1));
    f[16] = from_float((float)st[6]);
    // armor slot category one-hot (suit/shield/helm/gloves/boots/cloak/shirt)
    // from the slot glyph via the engine's baked otyp->ARM_* table
    device const precision_t* gsrc = obs + (long)b * NH_OBS_SIZE + NH_INV_OFF + 2 * s;
    int g = (int)to_float(gsrc[0]) + ((int)to_float(gsrc[1]) << 8);
    int ot = g - NH_GLYPH_OBJ_OFF;
    int cat = (ot >= 0 && ot < NH_NUM_OBJECTS) ? nh_obj_armcat_dev[ot] : -1;
    for (int c = 0; c < 7; c++)
        f[17 + c] = from_float(cat == c ? 1.0f : 0.0f);

}

struct nh_inv_gather_kernel_args {
    device precision_t* out;
    device const precision_t* T;
    device const precision_t* b1;
    device const precision_t* ws;
    device const precision_t* sfeat;
    device const float* idx;
    device const precision_t* Tt;
    device const float* idxt;
    int B;
};
kernel void nh_inv_gather_kernel(constant nh_inv_gather_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* out = args_.out;
    device const precision_t* T = args_.T;
    device const precision_t* b1 = args_.b1;
    device const precision_t* ws = args_.ws;
    device const precision_t* sfeat = args_.sfeat;
    device const float* idx = args_.idx;
    device const precision_t* Tt = args_.Tt;
    device const float* idxt = args_.idxt;
    int B = args_.B;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_INV_FLAT) return;
    int b = t / NH_INV_FLAT, r = t % NH_INV_FLAT;
    int s = r / NH_INV_HID, k = r % NH_INV_HID;
    int g = (int)idx[b * NH_INV + s];
    float v = to_float(T[g * NH_INV_HID + k]) + to_float(b1[k]);
    // discovered-type channel: pad = unknown identity, hard zero contribution
    int gt = (int)idxt[b * NH_INV + s];
    if (gt != NH_GLYPH_VOCAB - 1)
        v += to_float(Tt[gt * NH_INV_HID + k]);
    device const precision_t* f = sfeat + ((long)b * NH_INV + s) * NH_SFEAT;
    for (int j = 0; j < NH_SFEAT; j++)
        v += to_float(ws[k * NH_SFEAT + j]) * to_float(f[j]);
    out[t] = from_float(v > 0.0f ? v : 0.0f);

}

struct nh_dTinv_scatter_kernel_args {
    device long* dT;
    device const precision_t* dflat;
    device const float* idx;
    long n;
    int skip_pad;
};
kernel void nh_dTinv_scatter_kernel(constant nh_dTinv_scatter_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device long* dT = args_.dT;
    device const precision_t* dflat = args_.dflat;
    device const float* idx = args_.idx;
    long n = args_.n;
    int skip_pad = args_.skip_pad;

    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    float v = to_float(dflat[t]);
    if (v == 0.0f) return;
    int g = (int)idx[t / NH_INV_HID];
    // true-glyph channel: pad slots contributed hard zero in the forward
    if (skip_pad && g == NH_GLYPH_VOCAB - 1) return;
    nh_fxp_atomic_add(&dT[(long)g * NH_INV_HID + t % NH_INV_HID], v);

}

struct nh_add_inplace_kernel_args {
    device precision_t* dst;
    device const precision_t* src;
    int n;
};
kernel void nh_add_inplace_kernel(constant nh_add_inplace_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* dst = args_.dst;
    device const precision_t* src = args_.src;
    int n = args_.n;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = from_float(to_float(dst[i]) + to_float(src[i]));

}

struct nh_relu_bias_bwd_kernel_args {
    device precision_t* grad;
    device const precision_t* out;
    device long* bias_acc;
    long total;
    int dim;
};
kernel void nh_relu_bias_bwd_kernel(constant nh_relu_bias_bwd_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* grad = args_.grad;
    device const precision_t* out = args_.out;
    device long* bias_acc = args_.bias_acc;
    long total = args_.total;
    int dim = args_.dim;

    long i0 = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (long i = i0; i < total; i += stride) {
        float g = to_float(out[i]) > 0.0f ? to_float(grad[i]) : 0.0f;
        grad[i] = from_float(g);
        acc += g;
    }
    if (acc != 0.0f) nh_fxp_atomic_add(&bias_acc[i0 % dim], acc);

}

struct nh_bias_bwd_kernel_args {
    device precision_t* grad;
    device const precision_t* out;
    device long* bias_acc;
    long total;
    int dim;
};
kernel void nh_bias_bwd_kernel(constant nh_bias_bwd_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* grad = args_.grad;
    device long* bias_acc = args_.bias_acc;
    long total = args_.total;
    int dim = args_.dim;

    long i0 = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (long i = i0; i < total; i += stride) {
        float g = to_float(grad[i]);
        grad[i] = from_float(g);
        acc += g;
    }
    if (acc != 0.0f) nh_fxp_atomic_add(&bias_acc[i0 % dim], acc);

}

struct nh_bias_flush_kernel_args {
    device const long* acc;
    device precision_t* d0;
    int n0;
    device precision_t* d1;
    int n1;
    device precision_t* d2;
    int n2;
    device precision_t* d3;
    int n3;
    device precision_t* d4;
    int n4;
    device precision_t* d5;
    int n5;
    device precision_t* d6;
    int n6;
};
kernel void nh_bias_flush_kernel(constant nh_bias_flush_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device const long* acc = args_.acc;
    device precision_t* d0 = args_.d0;
    int n0 = args_.n0;
    device precision_t* d1 = args_.d1;
    int n1 = args_.n1;
    device precision_t* d2 = args_.d2;
    int n2 = args_.n2;
    device precision_t* d3 = args_.d3;
    int n3 = args_.n3;
    device precision_t* d4 = args_.d4;
    int n4 = args_.n4;
    device precision_t* d5 = args_.d5;
    int n5 = args_.n5;
    device precision_t* d6 = args_.d6;
    int n6 = args_.n6;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = i < n0 + n1 + n2 + n3 + n4 + n5 + n6 ? nh_fxp_to_float(acc[i]) : 0.0f;
    if (i < n0) d0[i] = from_float(v);
    else if ((i -= n0) < n1) d1[i] = from_float(v);
    else if ((i -= n1) < n2) d2[i] = from_float(v);
    else if ((i -= n2) < n3) d3[i] = from_float(v);
    else if ((i -= n3) < n4) d4[i] = from_float(v);
    else if ((i -= n4) < n5) d5[i] = from_float(v);
    else if ((i -= n5) < n6) d6[i] = from_float(v);

}

struct nh_idemb_kernel_args {
    device precision_t* concat;
    device float* idx_out;
    device const precision_t* obs;
    device const precision_t* role_w;
    device const precision_t* race_w;
    device const precision_t* gend_w;
    device const precision_t* algn_w;
    int B;
};
kernel void nh_idemb_kernel(constant nh_idemb_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* concat = args_.concat;
    device float* idx_out = args_.idx_out;
    device const precision_t* obs = args_.obs;
    device const precision_t* role_w = args_.role_w;
    device const precision_t* race_w = args_.race_w;
    device const precision_t* gend_w = args_.gend_w;
    device const precision_t* algn_w = args_.algn_w;
    int B = args_.B;

    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    device const precision_t* bl = obs + (long)b * NH_OBS_SIZE + NH_BL_OFF;
    device const precision_t* ex = bl + 4 * NH_BL_RAW;
    int r = 0, rc = 0, g = 0;
    for (int i = 0; i < 13; i++) if (nh_bl_read_i32(ex + 4 * (NH_EX_ROLEOH + i))) r = i;
    for (int i = 0; i < 5; i++) if (nh_bl_read_i32(ex + 4 * (NH_EX_ROLEOH + 13 + i))) rc = i;
    for (int i = 0; i < 2; i++) if (nh_bl_read_i32(ex + 4 * (NH_EX_ROLEOH + 18 + i))) g = i;
    int al = 1 - nh_bl_read_i32(bl + 4 * 26); // align: lawful 1, neutral 0, chaotic -1
    al = al < 0 ? 0 : al > 2 ? 2 : al;
    idx_out[b * 4 + 0] = (float)r;
    idx_out[b * 4 + 1] = (float)rc;
    idx_out[b * 4 + 2] = (float)g;
    idx_out[b * 4 + 3] = (float)al;
    device precision_t* c = concat + (long)b * NH_CONCAT + NH_IDE_CONCAT_OFF;
    for (int d = 0; d < NH_IDE_ROLE; d++) c[d] = role_w[r * NH_IDE_ROLE + d];
    c += NH_IDE_ROLE;
    for (int d = 0; d < NH_IDE_RACE; d++) c[d] = race_w[rc * NH_IDE_RACE + d];
    c += NH_IDE_RACE;
    for (int d = 0; d < NH_IDE_GEND; d++) c[d] = gend_w[g * NH_IDE_GEND + d];
    c += NH_IDE_GEND;
    for (int d = 0; d < NH_IDE_ALGN; d++) c[d] = algn_w[al * NH_IDE_ALGN + d];

}

struct nh_idemb_grad_kernel_args {
    device precision_t* role_g;
    device precision_t* race_g;
    device precision_t* gend_g;
    device precision_t* algn_g;
    device const precision_t* grad_concat;
    device const float* idx;
    int B;
};
kernel void nh_idemb_grad_kernel(constant nh_idemb_grad_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* role_g = args_.role_g;
    device precision_t* race_g = args_.race_g;
    device precision_t* gend_g = args_.gend_g;
    device precision_t* algn_g = args_.algn_g;
    device const precision_t* grad_concat = args_.grad_concat;
    device const float* idx = args_.idx;
    int B = args_.B;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_role = 13 * NH_IDE_ROLE, n_race = 5 * NH_IDE_RACE;
    const int n_gend = 2 * NH_IDE_GEND, n_algn = 3 * NH_IDE_ALGN;
    if (t >= n_role + n_race + n_gend + n_algn) return;
    int comp, wdt, off;
    device precision_t* out;
    int u = t;
    if (u < n_role) { comp = 0; wdt = NH_IDE_ROLE; off = 0; out = role_g; }
    else if ((u -= n_role) < n_race) { comp = 1; wdt = NH_IDE_RACE; off = NH_IDE_ROLE; out = race_g; }
    else if ((u -= n_race) < n_gend) { comp = 2; wdt = NH_IDE_GEND; off = NH_IDE_ROLE + NH_IDE_RACE; out = gend_g; }
    else { u -= n_gend; comp = 3; wdt = NH_IDE_ALGN; off = NH_IDE_ROLE + NH_IDE_RACE + NH_IDE_GEND; out = algn_g; }
    int row = u / wdt, d = u % wdt;
    float acc = 0.0f;
    for (int b = 0; b < B; b++) {
        if ((int)idx[b * 4 + comp] != row) continue;
        acc += to_float(grad_concat[(long)b * NH_CONCAT + NH_IDE_CONCAT_OFF + off + d]);
    }
    out[row * wdt + d] = from_float(acc);

}

struct nh_dec_assemble_kernel_args {
    device precision_t* out;
    device const precision_t* tmp;
    device const precision_t* slot_logits;
    device const precision_t* q;
    device const precision_t* spkeys;
    int B;
};
kernel void nh_dec_assemble_kernel(constant nh_dec_assemble_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* out = args_.out;
    device const precision_t* tmp = args_.tmp;
    device const precision_t* slot_logits = args_.slot_logits;
    device const precision_t* q = args_.q;
    device const precision_t* spkeys = args_.spkeys;
    int B = args_.B;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int od1 = NH_DEC_OD + 1;
    if (idx >= B * od1) return;
    int b = idx / od1, c = idx % od1;
    float v;
    if (c < NH_ACTIONS) v = to_float(tmp[(long)b * NH_DEC_PAD + c]);
    else if (c < NH_ACTIONS + NH_SLOT_OD)
        v = to_float(slot_logits[(long)b * NH_SLOT_OD + (c - NH_ACTIONS)]);
    else if (c < NH_SPELL_BASE) // per-verb dir rows from lin
        v = to_float(tmp[(long)b * NH_DEC_PAD + NH_ACTIONS + (c - NH_ACTIONS - NH_SLOT_OD)]);
    else if (c < NH_SPELL_BASE + NH_SPELL_SLOTS) {
        // spell head: dot(q_spell, key_s) / sqrt(keydim)
        int s = c - NH_SPELL_BASE;
        device const precision_t* qs = q + ((long)b * NH_QHEADS + NH_HEADS) * NH_INV_HID;
        device const precision_t* ks = spkeys + ((long)b * NH_SPELL_SLOTS + s) * NH_SPKEY;
        float dot = 0.0f;
        for (int k = 0; k < NH_SPKEY; k++) dot += to_float(qs[k]) * to_float(ks[k]);
        v = dot * 0.25f;
    } else // value
        v = to_float(tmp[(long)b * NH_DEC_PAD + NH_DEC_LIN - 1]);
    out[idx] = from_float(v);

}

struct nh_ptr_rownorm_kernel_args {
    device precision_t* n;
    device const precision_t* rows;
    int total;
};
kernel void nh_ptr_rownorm_kernel(constant nh_ptr_rownorm_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* n = args_.n;
    device const precision_t* rows = args_.rows;
    int total = args_.total;

    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= total) return;
    float acc = 0.0f;
    for (int k = 0; k < NH_INV_HID; k++) {
        float v = to_float(rows[(long)r * NH_INV_HID + k]);
        acc += v * v;
    }
    n[r] = from_float(sqrt(acc) + 1e-6f);

}

struct nh_ptr3_cos_kernel_args {
    device precision_t* slot_logits;
    device const precision_t* q;
    device const precision_t* qn;
    device const precision_t* kmat;
    device const precision_t* tau;
    int B;
};
kernel void nh_ptr3_cos_kernel(constant nh_ptr3_cos_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* slot_logits = args_.slot_logits;
    device const precision_t* q = args_.q;
    device const precision_t* qn = args_.qn;
    device const precision_t* kmat = args_.kmat;
    device const precision_t* tau = args_.tau;
    int B = args_.B;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH_SLOT_OD) return;
    int b = idx / NH_SLOT_OD, hi = idx % NH_SLOT_OD;
    int h = hi / NH_INV, i = hi % NH_INV;
    device const precision_t* qb = q + ((long)b * NH_QHEADS + h) * NH_INV_HID;
    device const precision_t* ki = kmat + ((long)b * NH_INV + i) * NH_INV_HID;
    float dot = 0.0f;
    for (int k = 0; k < NH_INV_HID; k++)
        dot += to_float(qb[k]) * to_float(ki[k]);
    slot_logits[idx] = from_float(exp(to_float(tau[h])) * dot /
        to_float(qn[(long)b * NH_QHEADS + h]));

}

struct nh_dec_dtmp_kernel_args {
    device precision_t* dtmp;
    device const precision_t* g;
    int B;
};
kernel void nh_dec_dtmp_kernel(constant nh_dec_dtmp_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* dtmp = args_.dtmp;
    device const precision_t* g = args_.g;
    int B = args_.B;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * NH_DEC_PAD) return;
    int b = idx / NH_DEC_PAD, c = idx % NH_DEC_PAD;
    if (c >= NH_DEC_LIN) { // pad rows
        dtmp[idx] = from_float(0.0f);
        return;
    }
    // lin rows: verbs, dirs, then value (spell logits are pointer-derived)
    int src = c < NH_ACTIONS ? c
            : c < NH_DEC_LIN - 1 ? NH_ACTIONS + NH_SLOT_OD + (c - NH_ACTIONS)
            : NH_DEC_OD;
    dtmp[idx] = g[(long)b * (NH_DEC_OD + 1) + src];

}

struct nh_ptr3_dq_kernel_args {
    device precision_t* dq;
    device long* tau_acc;
    device const precision_t* g;
    device const precision_t* out;
    device const precision_t* q;
    device const precision_t* qn;
    device const precision_t* kmat;
    device const precision_t* tau;
    int B;
};
kernel void nh_ptr3_dq_kernel(constant nh_ptr3_dq_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* dq = args_.dq;
    device long* tau_acc = args_.tau_acc;
    device const precision_t* g = args_.g;
    device const precision_t* out = args_.out;
    device const precision_t* q = args_.q;
    device const precision_t* qn = args_.qn;
    device const precision_t* kmat = args_.kmat;
    device const precision_t* tau = args_.tau;
    int B = args_.B;

    int bh = blockIdx.x * blockDim.x + threadIdx.x;
    if (bh >= B * NH_HEADS) return;
    int b = bh / NH_HEADS, h = bh % NH_HEADS;
    long qrow = (long)b * NH_QHEADS + h;
    float tauv = exp(to_float(tau[h]));
    float qnv = to_float(qn[qrow]);
    float vhat[NH_INV_HID], dv[NH_INV_HID];
    for (int k = 0; k < NH_INV_HID; k++) {
        vhat[k] = to_float(q[qrow * NH_INV_HID + k]) / qnv;
        dv[k] = 0.0f;
    }
    float dtau = 0.0f;
    const long gbase = (long)b * (NH_DEC_OD + 1) + NH_ACTIONS + h * NH_INV;
    for (int i = 0; i < NH_INV; i++) {
        float gi = to_float(g[gbase + i]);
        if (gi == 0.0f) continue;
        float cosv = to_float(out[gbase + i]) / tauv;
        dtau += gi * cosv;
        device const precision_t* ki = kmat + ((long)b * NH_INV + i) * NH_INV_HID;
        for (int k = 0; k < NH_INV_HID; k++)
            dv[k] += tauv * gi * to_float(ki[k]);
    }
    float vdv = 0.0f;
    for (int k = 0; k < NH_INV_HID; k++) vdv += vhat[k] * dv[k];
    for (int k = 0; k < NH_INV_HID; k++)
        dq[qrow * NH_INV_HID + k] = from_float((dv[k] - vhat[k] * vdv) / qnv);
    if (dtau != 0.0f) nh_fxp_atomic_add(&tau_acc[h], dtau * tauv);

}

struct nh_spq_bwd_kernel_args {
    device precision_t* dq;
    device precision_t* spdk;
    device const precision_t* g;
    device const precision_t* q;
    device const precision_t* spkeys;
    int B;
};
kernel void nh_spq_bwd_kernel(constant nh_spq_bwd_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* dq = args_.dq;
    device precision_t* spdk = args_.spdk;
    device const precision_t* g = args_.g;
    device const precision_t* q = args_.q;
    device const precision_t* spkeys = args_.spkeys;
    int B = args_.B;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= B * NH_SPKEY) return;
    int b = t / NH_SPKEY, k = t % NH_SPKEY;
    long qrow = ((long)b * NH_QHEADS + NH_HEADS) * NH_INV_HID;
    float qk = to_float(q[qrow + k]);
    float dqk = 0.0f;
    for (int s = 0; s < NH_SPELL_SLOTS; s++) {
        float gs = to_float(g[(long)b * (NH_DEC_OD + 1) + NH_SPELL_BASE + s]);
        dqk += 0.25f * gs * to_float(spkeys[((long)b * NH_SPELL_SLOTS + s) * NH_SPKEY + k]);
        spdk[((long)b * NH_SPELL_SLOTS + s) * NH_SPKEY + k] = from_float(0.25f * gs * qk);
    }
    dq[qrow + k] = from_float(dqk);

}

struct nh_ptr3_dkmat_kernel_args {
    device precision_t* dkmat;
    device const precision_t* g;
    device const precision_t* out;
    device const precision_t* q;
    device const precision_t* qn;
    device const precision_t* kmat;
    device const precision_t* tau;
    int B;
};
kernel void nh_ptr3_dkmat_kernel(constant nh_ptr3_dkmat_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* dkmat = args_.dkmat;
    device const precision_t* g = args_.g;
    device const precision_t* q = args_.q;
    device const precision_t* qn = args_.qn;
    device const precision_t* tau = args_.tau;
    int B = args_.B;

    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)B * NH_INV_FLAT) return;
    long bi = idx / NH_INV_HID;
    long b = bi / NH_INV;
    int i = (int)(bi % NH_INV);
    int k = (int)(idx % NH_INV_HID);
    float acc = 0.0f;
    for (int h = 0; h < NH_HEADS; h++) {
        long gi_idx = b * (NH_DEC_OD + 1) + NH_ACTIONS + h * NH_INV + i;
        float gi = to_float(g[gi_idx]);
        if (gi == 0.0f) continue;
        float tauv = exp(to_float(tau[h]));
        float vk = to_float(q[((long)b * NH_QHEADS + h) * NH_INV_HID + k])
                 / to_float(qn[(long)b * NH_QHEADS + h]);
        acc += tauv * gi * vk;
    }
    dkmat[idx] = from_float(acc);

}

struct nh_fill_kernel_args {
    device precision_t* p;
    float v;
    int n;
};
kernel void nh_fill_kernel(constant nh_fill_kernel_args& args_ [[buffer(0)]], PUF_KERNEL_IDX) {
    device precision_t* p = args_.p;
    float v = args_.v;
    int n = args_.n;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = from_float(v);

}

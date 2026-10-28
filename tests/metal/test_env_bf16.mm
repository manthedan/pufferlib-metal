// BF16 custom encoder arithmetic against fp64, plus exact copies/scatters.
// Each call poisons its destinations and both trials change every input. These
// fixtures isolate rounding from network error propagation in env_golden.h.
#import <Foundation/Foundation.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "compat/cuda_bf16.h"
#include "runtime.h"
#include "launch.h"

using P = __nv_bfloat16;
static int failures = 0, checks = 0;
template <class T> struct Buf {
    T *p;
    long n;
    explicit Buf(long count) : n(count) { cudaMalloc((void **)&p, n * sizeof(T)); }
    ~Buf() { cudaFree(p); }
    // 0x7fc1 halfwords: NaN as bf16/f32, a huge positive int32, so a skipped
    // store can't pass as a value within tolerance.
    void poison() {
        std::vector<uint16_t> v((n * sizeof(T) + 1) / 2, 0x7fc1);
        cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    }
    uint64_t addr() const { return puf_mtl::gpu_addr(p); }
    T &operator[](long i) { return p[i]; }
};
static float value(long i, int seed) { return float((i * 37 + seed * 17) % 127 - 63) / 256; }
static void fill(Buf<P> &b, int seed) {
    for (long i = 0; i < b.n; i++)
        b[i] = P(value(i, seed));
}
static void check(bool ok, const char *name) {
    checks++;
    failures += !ok;
    printf("%-46s %s\n", name, ok ? "PASS" : "FAIL");
}
template <class A> static void launch(const char *name, const A &a, dim3 g, dim3 b = dim3(256)) {
    puf_mtl::dispatch(0, puf_mtl::pipeline(name), g, b, a);
    cudaDeviceSynchronize();
}
static dim3 grid(long n) { return dim3((unsigned)((n + 255) / 256)); }
static void expect(Buf<P> &out, const std::vector<double> &ref, const char *name, int ulps = 0) {
    bool ok = long(ref.size()) == out.n;
    double maxerr = 0;
    for (long i = 0; i < out.n; i++) {
        P rounded((float)ref[i]);
        double spacing = std::ldexp(1.0, std::max(-126, std::ilogb(std::abs(float(rounded)))) - 7);
        if (!std::isfinite(float(out[i])) ||
            std::abs(double(float(out[i])) - float(rounded)) > ulps * spacing)
            ok = false;
        maxerr = std::max(maxerr, std::abs(double(float(out[i])) - float(rounded)));
    }
    if (!ok)
        printf("  max error from RNE(fp64) %.9g\n", maxerr);
    check(ok, name);
}
struct EntityArgs {
    uint64_t dst, argmax, input, weight, grad;
    int B, H, O, S, F, R;
};
static void entity(int S, int F, int R, int seed) {
    int B = 13, H = 40, O = S + R * F;
    Buf<P> obs(B * O), flat(B * R * (S + F)), x(B * R * 16), w(H * 16), g(B * H), out(B * H),
        wg(H * 16), dx(B * R * 16);
    Buf<int> arg(B * H);
    fill(obs, seed);
    fill(x, seed + 1);
    fill(w, seed + 2);
    fill(g, seed + 3);
    flat.poison();
    launch("entity_materialize", EntityArgs{flat.addr(), 0, obs.addr(), 0, 0, B, 0, O, S, F, R},
           grid(flat.n));
    std::vector<double> ref(flat.n);
    for (long i = 0; i < flat.n; i++) {
        int d = i % (S + F), r = (i / (S + F)) % R, b = i / ((S + F) * R);
        ref[i] = float(obs[b * O + (d < S ? d : S + r * F + d - S)]);
    }
    expect(flat, ref, "entity materialize exact");
    out.poison();
    arg.poison();
    launch("entity_linear_max",
           EntityArgs{out.addr(), arg.addr(), x.addr(), w.addr(), 0, B, H, 0, S, F, R},
           grid(out.n));
    ref.resize(out.n);
    std::vector<int> best(out.n);
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++) {
            double mx = -INFINITY;
            int am = 0;
            for (int r = 0; r < R; r++) {
                double s = 0;
                for (int d = 0; d < 16; d++)
                    s += double(float(w[h * 16 + d])) *
                         std::max(0.f, float(x[(b * R + r) * 16 + d]));
                if (s > mx) {
                    mx = s;
                    am = r;
                }
            }
            ref[b * H + h] = mx;
            best[b * H + h] = am;
        }
    expect(out, ref, "entity pool RNE(fp64) exact");
    check(!memcmp(arg.p, best.data(), arg.n * 4), "entity first-wins argmax exact");
    wg.poison();
    launch("entity_output_wgrad",
           EntityArgs{wg.addr(), arg.addr(), x.addr(), 0, g.addr(), B, H, 0, S, F, R}, grid(wg.n));
    ref.assign(wg.n, 0);
    for (int h = 0; h < H; h++)
        for (int d = 0; d < 16; d++)
            for (int b = 0; b < B; b++)
                ref[h * 16 + d] += double(float(g[b * H + h])) *
                                   std::max(0.f, float(x[(b * R + best[b * H + h]) * 16 + d]));
    expect(wg, ref, "entity wgrad RNE(fp64) exact");
    dx.poison();
    launch("entity_grad_input",
           EntityArgs{dx.addr(), arg.addr(), x.addr(), w.addr(), g.addr(), B, H, 0, S, F, R},
           grid(dx.n));
    ref.assign(dx.n, 0);
    for (int b = 0; b < B; b++)
        for (int r = 0; r < R; r++)
            for (int d = 0; d < 16; d++)
                if (float(x[(b * R + r) * 16 + d]) > 0)
                    for (int h = 0; h < H; h++)
                        if (best[b * H + h] == r)
                            ref[(b * R + r) * 16 + d] +=
                                double(float(g[b * H + h])) * float(w[h * 16 + d]);
    expect(dx, ref, "entity input grad RNE(fp64) exact");
}
struct OsrsPool {
    uint64_t out, arg, h1, z1, flat, w;
    int B, H, R, F, A;
};
struct OsrsBwd {
    uint64_t dst, grad, src, z1, arg;
    int B, H, R;
};
static double gelu(double x) {
    return .5 * x * (1 + std::tanh(.7978845608028654 * (x + .044715 * x * x * x)));
}
static double dgelu(double x) {
    double t = std::tanh(.7978845608028654 * (x + .044715 * x * x * x));
    return .5 * (1 + t) + .5 * x * (1 - t * t) * .7978845608028654 * (1 + 3 * .044715 * x * x);
}
static void osrs(int seed, int H) {
    int B = 13, R = 5, F = 6;
    Buf<P> z(B * R * 16), flat(B * R * F), w(H * 16), out(B * H), h1(B * R * 16), g(B * H),
        wg(H * 16), dz(B * R * 16);
    Buf<int> arg(B * H);
    fill(z, seed);
    fill(w, seed + 1);
    fill(g, seed + 2);
    for (long i = 0; i < flat.n; i++)
        flat[i] = P(i % F == 0 ? 1.f : 0.f);
    // Pool accumulates into an existing projection. Poison, then initialize
    // that required input explicitly; saved h1 and argmax are pure outputs.
    out.poison();
    for (long i = 0; i < out.n; i++)
        out[i] = P(value(i, seed + 4));
    h1.poison();
    arg.poison();
    std::vector<double> ref(out.n), hr(h1.n);
    std::vector<int> best(out.n);
    for (long i = 0; i < h1.n; i++)
        hr[i] = gelu(float(z[i]));
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++) {
            double mx = -INFINITY;
            int am = -1;
            for (int r = 0; r < R; r++) {
                double s = 0;
                for (int d = 0; d < 16; d++)
                    s += float(w[h * 16 + d]) * hr[(b * R + r) * 16 + d];
                if (s > mx) {
                    mx = s;
                    am = r;
                }
            }
            ref[b * H + h] = float(out[b * H + h]) + mx;
            best[b * H + h] = am;
        }
    launch(
        "osrs_entity_fused_pool_fwd",
        OsrsPool{out.addr(), arg.addr(), h1.addr(), z.addr(), flat.addr(), w.addr(), B, H, R, F, 1},
        dim3((B + 7) / 8, (H + 31) / 32), dim3(32, 8));
    expect(out, ref, "OSRS pool unrounded GELU tile vs fp64", 1);
    expect(h1, hr, "OSRS saved GELU vs fp64", 1);
    check(!memcmp(arg.p, best.data(), arg.n * 4), "OSRS argmax exact");
    // Independent, dyadic inputs give exact fp32 accumulation here.
    fill(h1, seed + 6);
    wg.poison();
    launch("osrs_entity_fused_l2_wgrad",
           OsrsBwd{wg.addr(), g.addr(), h1.addr(), 0, arg.addr(), B, H, R}, dim3(H));
    ref.assign(wg.n, 0);
    for (int h = 0; h < H; h++)
        for (int d = 0; d < 16; d++)
            for (int b = 0; b < B; b++)
                ref[h * 16 + d] +=
                    double(float(g[b * H + h])) * float(h1[(b * R + best[b * H + h]) * 16 + d]);
    expect(wg, ref, "OSRS L2 wgrad RNE(fp64) exact");
    dz.poison();
    launch("osrs_entity_fused_grad_z1",
           OsrsBwd{dz.addr(), g.addr(), w.addr(), z.addr(), arg.addr(), B, H, R}, dim3(B));
    ref.assign(dz.n, 0);
    for (int b = 0; b < B; b++)
        for (int r = 0; r < R; r++)
            for (int d = 0; d < 16; d++) {
                double v = 0;
                for (int h = 0; h < H; h++)
                    if (best[b * H + h] == r)
                        v += double(float(g[b * H + h])) * float(w[h * 16 + d]);
                ref[(b * R + r) * 16 + d] = v * dgelu(float(z[(b * R + r) * 16 + d]));
            }
    expect(dz, ref, "OSRS GELU grad vs fp64", 1);
}
struct Craftax {
    uint64_t dst, src, w;
    int B, O, n;
};
static void craftax(int seed) {
    int B = 13, O = 843, C = 1635, offsets[] = {0, 64, 72, 74, 90, 106, 122, 138},
        vocab[] = {64, 8, 2, 16, 16, 16, 16, 16};
    Buf<P> obs(B * O), w(154 * 16), out(B * C), g(B * C), wg(154 * 16);
    Buf<long> acc(154 * 16);
    fill(obs, seed);
    fill(w, seed + 1);
    fill(g, seed + 2);
    for (int b = 0; b < B; b++)
        for (int c = 0; c < 99; c++)
            for (int ch = 0; ch < 8; ch++)
                obs[b * O + c * 8 + ch] = P(float((b + c * 17 + ch + seed) % vocab[ch]));
    out.poison();
    launch("craftax_embed_bag", Craftax{out.addr(), obs.addr(), w.addr(), B, O, 0},
           grid(B * 99 * 16));
    launch("craftax_copy_scalars", Craftax{out.addr(), obs.addr(), 0, B, O, 0}, grid(B * 51));
    std::vector<double> ref(out.n);
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < 99; c++)
            for (int d = 0; d < 16; d++) {
                double sum = 0;
                for (int ch = 0; ch < 8; ch++)
                    sum += float(w[(offsets[ch] + int(float(obs[b * O + c * 8 + ch]))) * 16 + d]);
                ref[b * C + c * 16 + d] = sum;
            }
        for (int d = 0; d < 51; d++)
            ref[b * C + 1584 + d] = float(obs[b * O + 792 + d]);
    }
    expect(out, ref, "craftax embedding/scalars exact");
    acc.poison();
    cudaMemset(acc.p, 0, acc.n * 8);
    wg.poison();
    launch("craftax_embed_bag_backward", Craftax{acc.addr(), obs.addr(), g.addr(), B, O, 0},
           dim3(3));
    std::vector<long> ir(acc.n, 0);
    for (int b = 0; b < B; b++)
        for (int c = 0; c < 99; c++)
            for (int d = 0; d < 16; d++)
                for (int ch = 0; ch < 8; ch++)
                    ir[(offsets[ch] + int(float(obs[b * O + c * 8 + ch]))) * 16 + d] +=
                        long(std::llrint(double(float(g[b * C + c * 16 + d])) * 16777216));
    check(!memcmp(acc.p, ir.data(), acc.n * 8), "craftax fixed-point scatter exact");
    launch("craftax_fxp_to_precision", Craftax{wg.addr(), acc.addr(), 0, 0, 0, int(acc.n)},
           grid(acc.n));
    ref.resize(acc.n);
    for (long i = 0; i < acc.n; i++)
        ref[i] = double(ir[i]) / 16777216;
    expect(wg, ref, "craftax fixed-point flush exact");
}
struct N3 {
    uint64_t dst, src, src2, src3;
    int B, O, OC, S, flag, n;
};
static void nmmo(int seed) {
    int B = 2, O = 1707, C = 1817;
    Buf<P> obs(B * O), col1(B * 12 * 1475), x(B * 128 * 12), col2(B * 2 * 1152), dx(B * 128 * 12),
        emb(B * 47 * 32), w(128 * 32), out(B * C), g(B * C), wg(128 * 32);
    Buf<long> acc(128 * 32);
    fill(obs, seed);
    fill(x, seed + 1);
    fill(w, seed + 2);
    fill(g, seed + 3);
    int offsets[] = {0, 4, 8, 25, 30, 33, 38, 43, 48, 55},
        vocab[] = {4, 4, 17, 5, 3, 5, 5, 5, 7, 4};
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < 165; c++)
            for (int f = 0; f < 10; f++)
                obs[b * O + c * 10 + f] = P(float((b + c + f + seed) % vocab[f]));
        for (int f = 0; f < 47; f++)
            obs[b * O + 1650 + f] = P(float((b + f * 17 + seed) % 128));
    }
    col1.poison();
    launch("n3_c1_im2col_obs", N3{col1.addr(), obs.addr(), 0, 0, B, O, 0, 0, 0, 0}, grid(col1.n));
    std::vector<double> ref(col1.n);
    for (long i = 0; i < col1.n; i++) {
        int row = i / 1475, c = i % 1475, b = row / 12, oh = row % 12 / 4, ow = row % 4,
            ic = c / 25, kh = c % 25 / 5, kw = c % 5, f = 0;
        for (int j = 1; j < 10; j++)
            if (ic >= offsets[j])
                f = j;
        ref[i] =
            int(float(obs[b * O + ((oh * 3 + kh) * 15 + ow * 3 + kw) * 10 + f])) == ic - offsets[f];
    }
    expect(col1, ref, "nmmo3 categorical im2col exact");
    col2.poison();
    launch("n3_c2_im2col", N3{col2.addr(), x.addr(), 0, 0, B, 0, 0, 0, 0, 0}, grid(col2.n));
    ref.resize(col2.n);
    for (long i = 0; i < col2.n; i++) {
        int row = i / 1152, c = i % 1152, b = row / 2, ow = row % 2, ic = c / 9, kh = c % 9 / 3,
            kw = c % 3;
        ref[i] = float(x[(b * 128 + ic) * 12 + kh * 4 + ow + kw]);
    }
    expect(col2, ref, "nmmo3 im2col copy exact");
    fill(col2, seed + 4);
    dx.poison();
    launch("n3_c2_col2im", N3{dx.addr(), col2.addr(), 0, 0, B, 0, 0, 0, 0, 0}, grid(dx.n));
    ref.assign(dx.n, 0);
    for (long i = 0; i < dx.n; i++) {
        int iw = i % 4, ih = i / 4 % 3, ic = i / 12 % 128, b = i / (128 * 12);
        for (int kw = 0; kw < 3; kw++) {
            int ow = iw - kw;
            if (ow >= 0 && ow < 2)
                ref[i] += float(col2[b * 2 * 1152 + ic * 9 + ih * 3 + ow * 1152 + kw]);
        }
    }
    expect(dx, ref, "nmmo3 col2im RNE(fp64) exact");
    emb.poison();
    launch("n3_embedding", N3{emb.addr(), obs.addr(), w.addr(), 0, B, O, 0, 0, 0, 0}, grid(B * 47));
    ref.resize(emb.n);
    for (long i = 0; i < emb.n; i++)
        ref[i] =
            float(w[int(float(obs[(i / (47 * 32)) * O + 1650 + (i / 32) % 47])) * 32 + i % 32]);
    expect(emb, ref, "nmmo3 BF16 embedding row copy exact");
    out.poison();
    launch("n3_concat", N3{out.addr(), x.addr(), emb.addr(), obs.addr(), B, O, 0, 0, 0, 0},
           grid(out.n));
    ref.resize(out.n);
    for (long i = 0; i < out.n; i++) {
        int b = i / C, c = i % C;
        ref[i] = c < 256    ? float(x[b * 256 + c])
                 : c < 1760 ? float(emb[b * 1504 + c - 256])
                 : c < 1807 ? float(obs[b * O + 1650 + c - 1760])
                            : float(obs[b * O + O - 10 + c - 1807]);
    }
    expect(out, ref, "nmmo3 concat exact");
    acc.poison();
    cudaMemset(acc.p, 0, acc.n * 8);
    wg.poison();
    launch("n3_embedding_backward", N3{acc.addr(), g.addr(), obs.addr(), 0, B, O, 0, 0, 0, 0},
           dim3(3));
    std::vector<long> ir(acc.n, 0);
    for (int b = 0; b < B; b++)
        for (int f = 0; f < 47; f++)
            for (int d = 0; d < 32; d++)
                ir[int(float(obs[b * O + 1650 + f])) * 32 + d] +=
                    long(std::llrint(double(float(g[b * C + 256 + f * 32 + d])) * 16777216));
    check(!memcmp(acc.p, ir.data(), acc.n * 8), "nmmo3 fixed-point scatter exact");
    launch("n3_fxp_to_precision", N3{wg.addr(), acc.addr(), 0, 0, 0, 0, 0, 0, 0, int(acc.n)},
           grid(acc.n));
    ref.resize(acc.n);
    for (long i = 0; i < acc.n; i++)
        ref[i] = double(ir[i]) / 16777216;
    expect(wg, ref, "nmmo3 fixed-point flush exact");
}
struct NhBias {
    uint64_t data, bias;
    int total, dim;
};
struct NhNorm {
    uint64_t out, rows;
    int total;
};
static void nethack(int seed) {
    Buf<P> x(13 * 16), bias(16), out(13);
    fill(x, seed);
    fill(bias, seed + 1);
    std::vector<double> ref(x.n);
    for (long i = 0; i < x.n; i++)
        ref[i] = std::max(0.0, double(float(x[i])) + float(bias[i % 16]));
    launch("nh_bias_relu_kernel", NhBias{x.addr(), bias.addr(), int(x.n), 16}, grid(x.n));
    expect(x, ref, "NetHack BF16 bias/relu exact");
    fill(x, seed + 2);
    out.poison();
    ref.resize(13);
    for (int r = 0; r < 13; r++) {
        double s = 0;
        for (int k = 0; k < 16; k++)
            s += double(float(x[r * 16 + k])) * float(x[r * 16 + k]);
        ref[r] = std::sqrt(s) + 1e-6;
    }
    launch("nh_ptr_rownorm_kernel", NhNorm{out.addr(), x.addr(), 13}, grid(13));
    expect(out, ref, "NetHack pointer norms vs fp64", 1);
}
int main() {
    for (int seed : {1, 19}) {
        entity(2, 4, 16, seed);
        entity(4, 5, 26, seed);
        osrs(seed, 40);
        osrs(seed, 289);
        craftax(seed);
        nmmo(seed);
        nethack(seed);
    }
    printf("BF16 env arithmetic: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}

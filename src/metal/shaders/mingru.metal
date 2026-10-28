// MinGRU train scans and decoder-backward helpers (src/algo.cu):
// mingru_scan_forward / mingru_scan_backward (one thread per (b, h) lane),
// mingru_scan_forward_row / mingru_scan_backward_row (MINGRU_TILE x MINGRU_TILE
// threadgroups: coalesced T-tile load into threadgroup memory, then a
// sequential apply by the t_local == 0 row), sum_rows_to_precision_kernel and
// assemble_decoder_grad. Host twins of the argument structs live in
// src/metal/kernels_mingru.h.
//
// Semantics follow 5.0 exactly: linear scan h = lerp(h, h_tilde, z) with exact
// sigmoid, terminal[t] != 0 zeroes h before step t (and cuts dh after the
// backward step t), scan_h caches the post-reset recurrent input of every step
// (layout (B, T, H)), and the recurrent state is rounded through precision_t
// after each step (identity for FP32, kept for structure). Layouts:
// combined/grad_combined (B, T, 3H) with [hidden | gate | proj] per step,
// input/out/scan_h/grad_out/grad_input (B, T, H), terminals (B, T),
// state/next_state/grad_state/grad_next_state (B, H).
//
// Floating-point contraction: nvcc (default -fmad=true) fuses a*b + c into
// FMA; MSL is compiled without implicit contraction guarantees, so every
// multiply-add nvcc fuses is written as an explicit fma() here, mirroring
// mingru_gate in rollout.metal. Remaining differences vs the CUDA goldens are
// ulp-level exp() differences inside the exact sigmoid (CUDA expf uses the
// MUFU.EX2 approximation; not reproducible bit-for-bit).
#include "common.h"

#define MINGRU_TILE 32

// Corrected sigmoid; common.h documents the device-specific exactness check.
static inline float mg_sigmoid(float x) { return puf_sigmoid_exact(x); }

static inline float mg_lerp(float a, float b, float w) {
    float diff = b - a;
    return (fabs(w) < 0.5f) ? fma(w, diff, a) : fma(-diff, 1.0f - w, b);
}

static inline float mg_h_tilde(float hidden) {
    return (hidden >= 0.0f) ? hidden + 0.5f : mg_sigmoid(hidden);
}

// ---- plain scans (one thread per (b, h)) -------------------------------------

struct MingruScanFwdArgs {
    device const precision_t* combined;   // (B, T, 3H)
    device const precision_t* state;      // (B, H) h_0 (layer slice, agent_off applied)
    device const precision_t* input;      // (B, T, H) highway input
    device const precision_t* terminals;  // (B, T) or null
    device precision_t* scan_h;           // (B, T, H)
    device precision_t* out;              // (B, T, H)
    device precision_t* next_state;       // (B, H)
    int B, T, H;
};

kernel void mingru_scan_forward(constant MingruScanFwdArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    int T_seq = a.T, H = a.H, B = a.B;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) {
        return;
    }
    int b = idx / H;
    int h = idx % H;

    int bH = b * H;
    int H3 = 3 * H;
    int bHT = bH * T_seq;
    int out_base = bHT + h;
    int cbase = 3 * bHT;
    int h_base = b * T_seq * H + h;
    device const precision_t* term_row =
        a.terminals ? a.terminals + b * T_seq : (device const precision_t*)nullptr;

    float h_t = to_float(a.state[bH + h]);
    int t_offset = cbase + h;
    for (int t = 0; t < T_seq; t++) {
        if (term_row && to_float(term_row[t]) != 0.0f) {
            h_t = 0.0f;
        }
        a.scan_h[h_base + t * H] = from_float(h_t);

        float hidden_val = to_float(a.combined[t_offset]);
        float gate_val = to_float(a.combined[t_offset + H]);
        float proj_val = to_float(a.combined[t_offset + 2 * H]);
        float x_val = to_float(a.input[out_base + t * H]);

        float z = mg_sigmoid(gate_val);
        float h_tilde = mg_h_tilde(hidden_val);
        h_t = mg_lerp(h_t, h_tilde, z);

        float s = mg_sigmoid(proj_val);
        a.out[out_base + t * H] = from_float(fma(s, h_t, (1.0f - s) * x_val));
        h_t = to_float(from_float(h_t));

        t_offset += H3;
    }
    a.next_state[bH + h] = from_float(h_t);
}

struct MingruScanBwdArgs {
    device const precision_t* combined;         // (B, T, 3H)
    device const precision_t* input;            // (B, T, H)
    device const precision_t* terminals;        // (B, T) or null
    device const precision_t* scan_h;           // (B, T, H)
    device const precision_t* grad_out;         // (B, T, H)
    device const precision_t* grad_next_state;  // (B, H)
    device precision_t* grad_combined;          // (B, T, 3H)
    device precision_t* grad_state;             // (B, H)
    device precision_t* grad_input;             // (B, T, H)
    int B, T, H;
};

kernel void mingru_scan_backward(constant MingruScanBwdArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    int T_seq = a.T, H = a.H, B = a.B;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) {
        return;
    }
    int b = idx / H;
    int h = idx % H;

    int bHT = b * H * T_seq;
    int cbase = 3 * bHT;
    int H3 = 3 * H;
    int state_idx = b * H + h;
    int out_base = bHT + h;
    int h_base = b * T_seq * H + h;
    device const precision_t* term_row =
        a.terminals ? a.terminals + b * T_seq : (device const precision_t*)nullptr;

    // dh flowing into h_t from the future (grad_next_state at t = T).
    float dh = to_float(a.grad_next_state[state_idx]);
    for (int t0 = T_seq - 1; t0 >= 0; --t0) {
        int c = cbase + t0 * H3 + h;
        int input_idx = out_base + t0 * H;

        float h_prev = to_float(a.scan_h[h_base + t0 * H]);
        float hidden_val = to_float(a.combined[c]);
        float gate_val = to_float(a.combined[c + H]);
        float proj_val = to_float(a.combined[c + 2 * H]);
        float x_val = to_float(a.input[input_idx]);
        float grad_out_val = to_float(a.grad_out[input_idx]);

        float z = mg_sigmoid(gate_val);
        float h_tilde = mg_h_tilde(hidden_val);
        float h_t = mg_lerp(h_prev, h_tilde, z);
        float s = mg_sigmoid(proj_val);

        // highway: out = s*h + (1-s)*x
        float grad_proj = grad_out_val * (h_t - x_val) * s * (1.0f - s);
        a.grad_input[input_idx] = from_float(grad_out_val * (1.0f - s));
        a.grad_combined[c + 2 * H] = from_float(grad_proj);

        // dh + grad_out * s (nvcc fuses into one FMA)
        float dh_total = fma(grad_out_val, s, dh);

        // h = (1-z)*h_prev + z*h_tilde
        float d_h_prev = dh_total * (1.0f - z);
        float d_h_tilde = dh_total * z;
        float d_z = dh_total * (h_tilde - h_prev);
        float d_gate = d_z * z * (1.0f - z);
        float d_hidden = (hidden_val >= 0.0f)
            ? d_h_tilde
            : d_h_tilde * h_tilde * (1.0f - h_tilde);

        a.grad_combined[c] = from_float(d_hidden);
        a.grad_combined[c + H] = from_float(d_gate);

        // The reset before step t0 zeroed h_prev's contribution: cut dh.
        dh = d_h_prev;
        if (term_row && to_float(term_row[t0]) != 0.0f) {
            dh = 0.0f;
        }
    }
    a.grad_state[state_idx] = from_float(dh);
}

// ---- row (tiled) scans: grid (B, ceil(H/32)), threadgroup (32, 32) ----------
// threadIdx.x = h lane within the tile, threadIdx.y = t within the T tile.

kernel void mingru_scan_forward_row(constant MingruScanFwdArgs& a [[buffer(0)]],
        PUF_KERNEL_IDX) {
    int T_seq = a.T, H = a.H;
    int hx = threadIdx.x, t_local = threadIdx.y;
    int h = blockIdx.y * MINGRU_TILE + hx;
    bool h_ok = h < H;
    int b = blockIdx.x, H3 = 3 * H, bHT = b * H * T_seq, cbase = 3 * bHT;
    float h_t = h_ok ? to_float(a.state[b * H + h]) : 0.0f;
    threadgroup float sm_z[MINGRU_TILE][MINGRU_TILE];
    threadgroup float sm_ht[MINGRU_TILE][MINGRU_TILE];
    threadgroup float sm_s[MINGRU_TILE][MINGRU_TILE];
    threadgroup float sm_x[MINGRU_TILE][MINGRU_TILE];
    threadgroup float sm_term[MINGRU_TILE][MINGRU_TILE];
    int nchunks = (T_seq + MINGRU_TILE - 1) / MINGRU_TILE;
    for (int c = 0; c < nchunks; c++) {
        int t = c * MINGRU_TILE + t_local;
        float z = 0.0f, h_tilde = 0.0f, s = 0.0f, x = 0.0f, term = 0.0f;
        if (t < T_seq && h_ok) {
            int off = t * H3 + h;
            float hidden = to_float(a.combined[cbase + off]);
            float gate = to_float(a.combined[cbase + H + off]);
            float proj = to_float(a.combined[cbase + 2 * H + off]);
            x = to_float(a.input[bHT + t * H + h]);
            z = mg_sigmoid(gate);
            h_tilde = mg_h_tilde(hidden);
            s = mg_sigmoid(proj);
            term = (a.terminals && to_float(a.terminals[b * T_seq + t]) != 0.0f) ? 1.0f : 0.0f;
        }
        sm_z[t_local][hx] = z;
        sm_ht[t_local][hx] = h_tilde;
        sm_s[t_local][hx] = s;
        sm_x[t_local][hx] = x;
        sm_term[t_local][hx] = term;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t_local == 0 && h_ok) {
            int nloc = min(MINGRU_TILE, T_seq - c * MINGRU_TILE);
            for (int i = 0; i < nloc; i++) {
                if (sm_term[i][hx] != 0.0f) {
                    h_t = 0.0f;
                }
                int tt = c * MINGRU_TILE + i;
                a.scan_h[bHT + tt * H + h] = from_float(h_t);
                h_t = mg_lerp(h_t, sm_ht[i][hx], sm_z[i][hx]);
                float sv = sm_s[i][hx];
                // nvcc fuses the other product here than in the plain kernel
                // (the source of CUDA's plain-vs-row out difference; measured
                // against the scan_row_* goldens).
                a.out[bHT + tt * H + h] = from_float(fma(1.0f - sv, sm_x[i][hx], sv * h_t));
                h_t = to_float(from_float(h_t));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t_local == 0 && h_ok) {
        a.next_state[b * H + h] = from_float(h_t);
    }
}

kernel void mingru_scan_backward_row(constant MingruScanBwdArgs& a [[buffer(0)]],
        PUF_KERNEL_IDX) {
    int T_seq = a.T, H = a.H;
    int hx = threadIdx.x, t_local = threadIdx.y;
    int h = blockIdx.y * MINGRU_TILE + hx;
    bool h_ok = h < H;
    int b = blockIdx.x, H3 = 3 * H, bHT = b * H * T_seq, cbase = 3 * bHT;
    float dh = h_ok ? to_float(a.grad_next_state[b * H + h]) : 0.0f;
    // 8 x 4 KiB = 32 KiB: exactly the Apple GPU threadgroup-memory limit.
    threadgroup float sm_z[MINGRU_TILE][MINGRU_TILE];
    threadgroup float sm_ht[MINGRU_TILE][MINGRU_TILE];
    threadgroup float sm_s[MINGRU_TILE][MINGRU_TILE];
    threadgroup float sm_x[MINGRU_TILE][MINGRU_TILE];
    threadgroup float sm_hp[MINGRU_TILE][MINGRU_TILE];
    threadgroup float sm_go[MINGRU_TILE][MINGRU_TILE];
    threadgroup float sm_hid[MINGRU_TILE][MINGRU_TILE];
    threadgroup float sm_term[MINGRU_TILE][MINGRU_TILE];
    for (int c = (T_seq + MINGRU_TILE - 1) / MINGRU_TILE - 1; c >= 0; c--) {
        int t = c * MINGRU_TILE + t_local;
        float z = 0.0f, h_tilde = 0.0f, s = 0.0f, x = 0.0f;
        float h_prev = 0.0f, gout = 0.0f, hidden = 0.0f, term = 0.0f;
        if (t < T_seq && h_ok) {
            int off = t * H3 + h;
            hidden = to_float(a.combined[cbase + off]);
            float gate = to_float(a.combined[cbase + H + off]);
            float proj = to_float(a.combined[cbase + 2 * H + off]);
            x = to_float(a.input[bHT + t * H + h]);
            h_prev = to_float(a.scan_h[bHT + t * H + h]);
            gout = to_float(a.grad_out[bHT + t * H + h]);
            z = mg_sigmoid(gate);
            h_tilde = mg_h_tilde(hidden);
            s = mg_sigmoid(proj);
            term = (a.terminals && to_float(a.terminals[b * T_seq + t]) != 0.0f) ? 1.0f : 0.0f;
        }
        sm_z[t_local][hx] = z;
        sm_ht[t_local][hx] = h_tilde;
        sm_s[t_local][hx] = s;
        sm_x[t_local][hx] = x;
        sm_hp[t_local][hx] = h_prev;
        sm_go[t_local][hx] = gout;
        sm_hid[t_local][hx] = hidden;
        sm_term[t_local][hx] = term;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t_local == 0 && h_ok) {
            for (int i = min(MINGRU_TILE, T_seq - c * MINGRU_TILE) - 1; i >= 0; i--) {
                int tt = c * MINGRU_TILE + i;
                float zi = sm_z[i][hx];
                float si = sm_s[i][hx];
                float go = sm_go[i][hx];
                float hti = sm_ht[i][hx];
                float hpi = sm_hp[i][hx];
                float h_t = mg_lerp(hpi, hti, zi);
                float dh_total = fma(go, si, dh);
                float d_h_tilde = dh_total * zi;
                int off = tt * H3 + h;
                a.grad_input[bHT + tt * H + h] = from_float(go * (1.0f - si));
                a.grad_combined[cbase + off] = from_float(
                    sm_hid[i][hx] >= 0.0f ? d_h_tilde
                    : d_h_tilde * hti * (1.0f - hti));
                a.grad_combined[cbase + H + off] =
                    from_float(dh_total * (hti - hpi) * zi * (1.0f - zi));
                a.grad_combined[cbase + 2 * H + off] =
                    from_float(go * (h_t - sm_x[i][hx]) * si * (1.0f - si));
                dh = sm_term[i][hx] != 0.0f ? 0.0f : dh_total * (1.0f - zi);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t_local == 0 && h_ok) {
        a.grad_state[b * H + h] = from_float(dh);
    }
}

// ---- decoder backward helpers ------------------------------------------------

struct SumRowsArgs { device precision_t* dst; device const float* src; int R, C; };
kernel void sum_rows_to_precision_kernel(constant SumRowsArgs& a [[buffer(0)]], PUF_KERNEL_IDX) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= a.C) {
        return;
    }
    float sum = 0.0f;
    for (int r = 0; r < a.R; r++) {
        sum += a.src[r * a.C + col];
    }
    a.dst[col] = from_float(sum);
}

struct AssembleDecoderGradArgs {
    device precision_t* dst;
    device const float* grad_logits;
    device const float* grad_value;
    int B_TT, od, od_plus_1;
};
kernel void assemble_decoder_grad(constant AssembleDecoderGradArgs& a [[buffer(0)]],
        PUF_KERNEL_IDX) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= a.B_TT * a.od_plus_1) {
        return;
    }
    int row = idx / a.od_plus_1, col = idx % a.od_plus_1;
    a.dst[idx] = from_float((col < a.od) ? a.grad_logits[row * a.od + col] : a.grad_value[row]);
}

// ---- fused projection + scan (H = 32, T divides 64) ---------------------------
// BF16 T=32 forward has a separate 32-row/two-simdgroup entrypoint below.
// One threadgroup (4 simdgroups) owns 64 consecutive rows of the (B*T, H)
// layer input, i.e. 64 / T whole sequences. The 64 x 96 projection tile
// combined = x @ W^T is computed with the same matmul2d descriptor, scope and
// operand views as tensor_ops_gemm_nt_f32 (one 64 x 32 column block per run),
// so it is bit-identical to the standalone GEMM, but it lives only in
// threadgroup memory: the forward never writes `combined` to device memory and
// the backward recomputes it from the saved layer input instead of reading it.
// The per-(sequence, h) scan bodies are mingru_scan_forward/backward verbatim
// (one thread per chain; a variant that moved the chain-independent math and
// the device traffic into separate all-thread phases was ~25% slower on M4).
// The backward then runs the dX GEMM grad_x = grad_combined @ W + grad_input
// from the threadgroup grad_combined tile with tensor_ops_gemm_nn_add_f32's
// exact matmul2d configuration (bit-identical to that kernel). With dW
// enabled, it also multiplies the tile's gradient transpose by x and writes
// a (96,32) partial instead of materializing grad_combined. A separate fixed
// tile-order reduction replaces the side-stream GEMM. This changes dW's FP32
// summation order, while all scan and dX arithmetic remains unchanged.
// Host preconditions (gemm.mm mingru_fused_*_rm): H == 32, 64 % T == 0,
// B*T % 64 == 0, 16-byte aligned x / W / grad_x / grad_input, grad_x !=
// grad_input. Grid (B*T/64), 128 threads, 24 KiB FP32 / 12 KiB BF16 TG memory.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

// BF16 keeps the generic path's rounding at projection/scan/GEMM stores.
// dW partials stay FP32, and only the final reduction rounds to precision_t.
#define MGF_ROWS 64
#define MGF_H 32
// Forward scan steps per unrolled block (T % 8 == 0; other T keep the
// per-step loop): the block's x / terminal loads issue together and the
// chain-independent sigmoid work of later steps can overlap the serial h
// chain. Per-step arithmetic and order are unchanged.
#define MGF_FWD_UNROLL 8

struct MingruFusedFwdArgs {
    device precision_t* x;                // (B*T, H) layer input (GEMM + highway)
    device precision_t* W;                // (3H, H)
    device const precision_t* state;      // (B, H)
    device const precision_t* terminals;  // (B, T) or null
    device precision_t* scan_h;           // (B, T, H)
    device precision_t* out;              // (B, T, H)
    device precision_t* next_state;       // (B, H)
    int B, T, H;
};

struct MingruFusedBwdArgs {
    device precision_t* x;                // (B*T, H) saved layer input
    device precision_t* W;                // (3H, H)
    device const precision_t* terminals;  // (B, T) or null
    device const precision_t* scan_h;     // (B, T, H)
    device const precision_t* grad_out;   // (B, T, H)
    device const precision_t* grad_next_state;  // (B, H)
    device precision_t* grad_combined;    // (B, T, 3H), or dW partial scratch
    device precision_t* grad_state;       // (B, H)
    device precision_t* grad_input;       // (B, T, H)
    device precision_t* grad_x;           // (B*T, H) dX = grad_combined @ W + grad_input
    int B, T, H;
};

// Serial forward chain of (sequence seq of tile-local block b, unit h) over
// the projection in comb (row-major [hidden | gate | proj], 3H per step).
static inline void mgf_forward_chain(constant MingruFusedFwdArgs& a,
        threadgroup const precision_t* comb, int T_seq, int seq, int h, int b) {
    const int H = MGF_H, H3 = 3 * MGF_H;
    int bH = b * H, bHT = bH * T_seq;
    int out_base = bHT + h;
    device const precision_t* term_row =
        a.terminals ? a.terminals + b * T_seq : (device const precision_t*)nullptr;
    float h_t = to_float(a.state[bH + h]);
    int r = seq * T_seq * H3 + h;
    int t = 0;
    for (; T_seq % MGF_FWD_UNROLL == 0 && t < T_seq; t += MGF_FWD_UNROLL) {
        float xv[MGF_FWD_UNROLL], tv[MGF_FWD_UNROLL];
        #pragma unroll
        for (int j = 0; j < MGF_FWD_UNROLL; ++j) {
            xv[j] = to_float(a.x[out_base + (t + j) * H]);
            tv[j] = term_row ? to_float(term_row[t + j]) : 0.0f;
        }
        #pragma unroll
        for (int j = 0; j < MGF_FWD_UNROLL; ++j) {
            const int rj = r + j * H3, tj = t + j;
            if (tv[j] != 0.0f) h_t = 0.0f;
            a.scan_h[out_base + tj * H] = from_float(h_t);
            float hidden_val = to_float(comb[rj]);
            float gate_val = to_float(comb[rj + H]);
            float proj_val = to_float(comb[rj + 2 * H]);
            float z = mg_sigmoid(gate_val);
            float h_tilde = mg_h_tilde(hidden_val);
            h_t = mg_lerp(h_t, h_tilde, z);
            float s = mg_sigmoid(proj_val);
            a.out[out_base + tj * H] = from_float(fma(s, h_t, (1.0f - s) * xv[j]));
            h_t = to_float(from_float(h_t));
        }
        r += MGF_FWD_UNROLL * H3;
    }
    for (; t < T_seq; t++) {
        if (term_row && to_float(term_row[t]) != 0.0f) {
            h_t = 0.0f;
        }
        a.scan_h[out_base + t * H] = from_float(h_t);

        float hidden_val = to_float(comb[r]);
        float gate_val = to_float(comb[r + H]);
        float proj_val = to_float(comb[r + 2 * H]);
        float x_val = to_float(a.x[out_base + t * H]);

        float z = mg_sigmoid(gate_val);
        float h_tilde = mg_h_tilde(hidden_val);
        h_t = mg_lerp(h_t, h_tilde, z);

        float s = mg_sigmoid(proj_val);
        a.out[out_base + t * H] = from_float(fma(s, h_t, (1.0f - s) * x_val));
        h_t = to_float(from_float(h_t));
        r += H3;
    }
    a.next_state[bH + h] = from_float(h_t);
}

// comb (64 rows x 96, row-major [hidden | gate | proj]) = x[row0 .. row0+63] @ W^T.
static inline void mgf_project(device precision_t* xp, device precision_t* Wp, int M, uint tile,
        threadgroup precision_t* comb) {
    using namespace mpp::tensor_ops;
    auto A = tensor<device precision_t, dextents<int32_t, 2>, tensor_inline>(
        xp, dextents<int32_t, 2>(MGF_H, M));
    auto B = tensor<device precision_t, dextents<int32_t, 2>, tensor_inline>(
        Wp, dextents<int32_t, 2>(MGF_H, 3 * MGF_H));
    auto C = tensor<threadgroup precision_t, dextents<int32_t, 2>, tensor_inline>(
        comb, dextents<int32_t, 2>(3 * MGF_H, MGF_ROWS));
    // One 96-wide tile (Breakout-shape FP32 forward 443 -> 408 us, backward
    // 820 -> 785 us). The matmul stores straight into the threadgroup tile;
    // in BF16 its fp32 -> bf16 store conversion is round-to-nearest-even,
    // bit-identical to from_float() of the fp32 cooperative-tensor result the
    // GEMM stores (exhaustively checked: tests/metal/run_test_proj_store.sh),
    // and ~2x faster than the per-element cooperative-tensor store (Breakout
    // BF16 backward ~735 -> ~620 us). Bit-identical for any tile width.
    constexpr auto desc = matmul2d_descriptor(64, 3 * MGF_H,
        static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = A.slice(0, tile * MGF_ROWS);
    auto mB = B.slice(0, 0);
    auto mC = C.slice(0, 0);
    op.run(mA, mB, mC);
}

kernel void mingru_fused_forward(constant MingruFusedFwdArgs& a [[buffer(0)]],
        uint tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int T_seq = a.T, H = MGF_H;
    // Uniform guard (barrier-safe): the tile must hold whole sequences and
    // lie inside the (B*T, H) input.
    if (a.H != MGF_H || T_seq <= 0 || MGF_ROWS % T_seq != 0
            || (long)(tile + 1) * MGF_ROWS > (long)a.B * T_seq) return;
    threadgroup precision_t comb[MGF_ROWS * 3 * MGF_H];
    mgf_project(a.x, a.W, a.B * T_seq, tile, comb);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int nseq = MGF_ROWS / T_seq;
    for (int lane = (int)tid; lane < nseq * H; lane += 128) {
        int seq = lane / H, h = lane % H;
        mgf_forward_chain(a, comb, T_seq, seq, h, (int)tile * nseq + seq);
    }
}

#ifndef PRECISION_FLOAT
// BF16 T=32 forward: 6 KiB tile / two simdgroups. Same K reduction and
// rounding as the 64-row projection; other horizons retain the original tile.
static inline void mgf_project32(device precision_t* xp, device precision_t* Wp, int M, uint tile,
        threadgroup precision_t* comb) {
    using namespace mpp::tensor_ops;
    auto A = tensor<device precision_t, dextents<int32_t, 2>, tensor_inline>(
        xp, dextents<int32_t, 2>(MGF_H, M));
    auto B = tensor<device precision_t, dextents<int32_t, 2>, tensor_inline>(
        Wp, dextents<int32_t, 2>(MGF_H, 3 * MGF_H));
    constexpr auto desc = matmul2d_descriptor(32, 32,
        static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<2>> op;
    auto mA = A.slice(0, tile * 32);
    // Direct threadgroup store, as in mgf_project.
    auto C = tensor<threadgroup precision_t, dextents<int32_t, 2>, tensor_inline>(
        comb, dextents<int32_t, 2>(3 * MGF_H, 32));
    for (int j = 0; j < 3; j++) {
        auto mB = B.slice(0, j * MGF_H);
        auto mC = C.slice(j * MGF_H, 0);
        op.run(mA, mB, mC);
    }
}

kernel void mingru_fused_forward32(constant MingruFusedFwdArgs& a [[buffer(0)]],
        uint tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int T_seq = a.T, H = MGF_H;
    // Uniform guard (barrier-safe): the tile must hold whole sequences and
    // lie inside the (B*T, H) input.
    if (a.H != MGF_H || T_seq <= 0 || 32 % T_seq != 0
            || (long)(tile + 1) * 32 > (long)a.B * T_seq) return;
    threadgroup precision_t comb[32 * 3 * MGF_H];
    mgf_project32(a.x, a.W, a.B * T_seq, tile, comb);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int nseq = 32 / T_seq;
    for (int lane = (int)tid; lane < nseq * H; lane += 64) {
        int seq = lane / H, h = lane % H;
        mgf_forward_chain(a, comb, T_seq, seq, h, (int)tile * nseq + seq);
    }
}

#endif

template <bool DW>
static void mgf_backward(constant MingruFusedBwdArgs& a, uint tile, uint tid,
        threadgroup precision_t* comb) {
    const int T_seq = a.T, H = MGF_H, H3 = 3 * MGF_H;
    if (a.H != MGF_H || T_seq <= 0 || MGF_ROWS % T_seq != 0
            || (long)(tile + 1) * MGF_ROWS > (long)a.B * T_seq) return;
    const int M = a.B * T_seq;
    mgf_project(a.x, a.W, M, tile, comb);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reverse scan. Each chain thread overwrites the projection values it
    // owns with their gradients, so the tile becomes grad_combined.
    const int nseq = MGF_ROWS / T_seq;
    for (int lane = (int)tid; lane < nseq * H; lane += 128) {
        int seq = lane / H, h = lane % H;
        int b = (int)tile * nseq + seq;
        int bHT = b * H * T_seq;
        int cbase = 3 * bHT;
        int state_idx = b * H + h;
        int out_base = bHT + h;
        device const precision_t* term_row =
            a.terminals ? a.terminals + b * T_seq : (device const precision_t*)nullptr;

        float dh = to_float(a.grad_next_state[state_idx]);
        // Issue several independent device reads before consuming the serial
        // dh chain. Keep t and all FP32 arithmetic in the original reverse
        // order; partial chunks handle T=1 and T=2 without speculative loads.
        constexpr int PREFETCH = 4;
        for (int top = T_seq - 1; top >= 0; top -= PREFETCH) {
            float hp[PREFETCH], xv[PREFETCH], go[PREFETCH], tv[PREFETCH];
            for (int j = 0; j < PREFETCH; ++j) {
                int t0 = top - j;
                if (t0 >= 0) {
                    int i = out_base + t0 * H;
                    hp[j] = to_float(a.scan_h[i]);
                    xv[j] = to_float(a.x[i]);
                    go[j] = to_float(a.grad_out[i]);
                    tv[j] = term_row ? to_float(term_row[t0]) : 0.0f;
                }
            }
            for (int j = 0; j < PREFETCH && top - j >= 0; ++j) {
                int t0 = top - j;
                int c = cbase + t0 * H3 + h;
                int input_idx = out_base + t0 * H;
                int r = (seq * T_seq + t0) * H3 + h;
                float h_prev = hp[j];
                float x_val = xv[j];
                float grad_out_val = go[j];
                float hidden_val = to_float(comb[r]);
                float gate_val = to_float(comb[r + H]);
                float proj_val = to_float(comb[r + 2 * H]);
                float z = mg_sigmoid(gate_val);
                float h_tilde = mg_h_tilde(hidden_val);
                float h_t = mg_lerp(h_prev, h_tilde, z);
                float s = mg_sigmoid(proj_val);
                // highway: out = s*h + (1-s)*x
                float grad_proj = grad_out_val * (h_t - x_val) * s * (1.0f - s);
                a.grad_input[input_idx] = from_float(grad_out_val * (1.0f - s));
                if constexpr (!DW) a.grad_combined[c + 2 * H] = from_float(grad_proj);

                float dh_total = fma(grad_out_val, s, dh);

                float d_h_prev = dh_total * (1.0f - z);
                float d_h_tilde = dh_total * z;
                float d_z = dh_total * (h_tilde - h_prev);
                float d_gate = d_z * z * (1.0f - z);
                float d_hidden = (hidden_val >= 0.0f)
                    ? d_h_tilde
                    : d_h_tilde * h_tilde * (1.0f - h_tilde);

                if constexpr (!DW) {
                    a.grad_combined[c] = from_float(d_hidden);
                    a.grad_combined[c + H] = from_float(d_gate);
                }
                comb[r] = from_float(d_hidden);
                comb[r + H] = from_float(d_gate);
                comb[r + 2 * H] = from_float(grad_proj);

                dh = d_h_prev;
                if (tv[j] != 0.0f) {
                    dh = 0.0f;
                }
            }
        }
        a.grad_state[state_idx] = from_float(dh);
    }
    // The epilogue reads grad_input rows written by other threads above, and
    // writes grad_x rows whose grad_out (possibly the same buffer) every
    // thread of this tile has finished reading.
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    if constexpr (DW) {
        // One (96,32) partial per tile, stored in the first half of the
        // grad_combined allocation (all of it in BF16). No global gradient tile.
        using namespace mpp::tensor_ops;
        auto G = tensor<threadgroup precision_t, dextents<int32_t, 2>, tensor_inline>(
            comb, dextents<int32_t, 2>(H3, MGF_ROWS));
        auto X = tensor<device precision_t, dextents<int32_t, 2>, tensor_inline>(
            a.x + (long)tile * MGF_ROWS * H, dextents<int32_t, 2>(H, MGF_ROWS));
        auto P = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
            (device float*)a.grad_combined + (long)tile * H3 * H, dextents<int32_t, 2>(H, H3));
        constexpr auto dw_desc = matmul2d_descriptor(32, 32,
            static_cast<int>(dynamic_extent), true, false, false);
        matmul2d<dw_desc, execution_simdgroups<4>> dw_op;
        auto mX = X.slice(0, 0);
        for (int j = 0; j < 3; ++j) {
            auto mG = G.slice(j * H, 0);
            auto mP = P.slice(0, j * H);
            dw_op.run(mG, mX, mP);
        }
    }

    // grad_x = grad_combined @ W + grad_input (tensor_ops_gemm_nn_add_f32).
    using namespace mpp::tensor_ops;
    auto A_cm = tensor<threadgroup precision_t, dextents<int32_t, 2>, tensor_inline>(
        comb, dextents<int32_t, 2>(H3, MGF_ROWS));
    auto B_cm = tensor<device precision_t, dextents<int32_t, 2>, tensor_inline>(
        a.W, dextents<int32_t, 2>(H, H3));
    auto C_cm = tensor<device precision_t, dextents<int32_t, 2>, tensor_inline>(
        a.grad_x, dextents<int32_t, 2>(H, M));
    auto D_cm = tensor<device precision_t, dextents<int32_t, 2>, tensor_inline>(
        a.grad_input, dextents<int32_t, 2>(H, M));
    constexpr auto desc = matmul2d_descriptor(64, 32,
        static_cast<int>(dynamic_extent), false, false, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mSecond = B_cm.slice(0, 0);
    auto mResult = C_cm.slice(0, tile * MGF_ROWS);
    auto mAdd = D_cm.slice(0, tile * MGF_ROWS);
    auto cT = op.get_destination_cooperative_tensor<decltype(A_cm), decltype(mSecond), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }
    op.run(A_cm, mSecond, cT);
#ifdef PRECISION_FLOAT
    auto dT = op.get_destination_cooperative_tensor<decltype(A_cm), decltype(mSecond), float>();
    dT.load(mAdd);
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] += dT[i];
    }
    cT.store(mResult);
#else
    // Match bf16(bf16(G @ W) + grad_input): both generic stores round.
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (!cT.is_valid_element(i)) continue;
        auto idx = cT.get_multidimensional_index(i);
        long ci = (long)(tile * MGF_ROWS + idx[1]) * H + idx[0];
        a.grad_x[ci] = from_float(round_p(cT[i]) + to_float(a.grad_input[ci]));
    }
#endif
}

#define MGF_BWD(NAME, DW) \
kernel void NAME(constant MingruFusedBwdArgs& a [[buffer(0)]], \
        uint tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup precision_t comb[MGF_ROWS * 3 * MGF_H]; \
    mgf_backward<DW>(a, tile, tid, comb); \
}
MGF_BWD(mingru_fused_backward, false)
MGF_BWD(mingru_fused_backward_dw, true)

struct MingruDWReduceArgs {
    device const float* partial;
    device precision_t* out;
    int tiles;
};
// One output weight per thread, summed in ascending tile order from 0.0f.
// The shipped kernel issues 8 independent loads before adding them in that
// same order (memory-level parallelism; bit-identical to the serial loop,
// which is kept as the test oracle). Any grid of >= 3072 threads is valid.
kernel void mingru_dw_reduce_serial(constant MingruDWReduceArgs& a [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
    if (i >= 3 * MGF_H * MGF_H) return;
    float sum = 0.0f;
    for (int p = 0; p < a.tiles; ++p) sum += a.partial[(long)p * 3 * MGF_H * MGF_H + i];
    a.out[i] = from_float(sum);
}
kernel void mingru_dw_reduce(constant MingruDWReduceArgs& a [[buffer(0)]],
        uint i [[thread_position_in_grid]]) {
    constexpr int W = 3 * MGF_H * MGF_H, NB = 8;
    if (i >= (uint)W) return;
    const int tiles = a.tiles;
    device const float* src = a.partial + i;
    float sum = 0.0f;
    int p = 0;
    for (; p + NB <= tiles; p += NB) {
        float v[NB];
        for (int k = 0; k < NB; ++k) v[k] = src[(long)(p + k) * W];
        for (int k = 0; k < NB; ++k) sum += v[k];
    }
    for (; p < tiles; ++p) sum += src[(long)p * W];
    a.out[i] = from_float(sum);
}

// ---- fused train layer, H=64 -------------------------------------------------
// At H = 64 the projection GEMM is compute-bound (~0.4 GFLOP per Pong
// minibatch layer, near the M4's matmul rate), so recomputing it in the
// backward (the H=32 design) costs more than the `combined` traffic it saves.
// Instead:
// - forward: per 16-row chunk, project into threadgroup memory with
//   matmul2d (the tensor_ops nt GEMM's per-element K reduction), store the
//   chunk to `combined` exactly as the GEMM stores it (one BF16 rounding),
//   and run the forward scan from the threadgroup copy;
// - backward: the reverse scan reads `combined`, writes grad_combined (for
//   the generic dW GEMM; no fused dW) and keeps it in threadgroup memory for
//   grad_x = grad_combined @ W + grad_input (tensor_ops_gemm_nn_add's
//   reduction and rounding), per chunk. The chunk's rounded grad_input also
//   stays in threadgroup memory for that epilogue; its device copy is
//   optional (grad_input == nullptr skips it; algo.cu needs only grad_x).
// Chunks are 16 rows (8 KiB BF16 / 16 KiB FP32 of threadgroup memory: larger
// chunks measured slower, occupancy). For T <= 16 a chunk holds whole
// sequences; longer sequences carry h (forward) or dh (backward) across the
// chunks of their 64-row tile in a register (the forward runs chunks in
// order, the backward in reverse). Scan, gradient and rounding arithmetic is
// unchanged, so every output is bit-identical to the generic path
// (test_mingru_bf16 / test_mingru). Grid (B*T/64), 64 threads.
#define MGF64_H 64
#ifndef MGF64_CHUNK
#define MGF64_CHUNK 16
#endif
// Projection (forward) and dX (backward) matmul2d tile widths: wider tiles
// reload each chunk's operand tile fewer times (Pong forward 295 -> 223 us
// BF16, 369 -> 277 us FP32; backward 280 -> 263 / 489 -> 458 us). The
// per-element K reduction does not depend on the tile width (bit-identical,
// test_mingru / test_mingru_bf16).
#ifndef MGF64_FWD_N
#ifdef PRECISION_FLOAT
#define MGF64_FWD_N 64
#else
#define MGF64_FWD_N 96
#endif
#endif
#ifndef MGF64_BWD_N
#define MGF64_BWD_N 64
#endif
#ifndef MGF64_BWD_PREFETCH
#define MGF64_BWD_PREFETCH 4
#endif

struct MingruFused64FwdArgs {
    device precision_t* x;                // (B*T, H) layer input (GEMM + highway)
    device precision_t* W;                // (3H, H)
    device const precision_t* state;      // (B, H)
    device const precision_t* terminals;  // (B, T) or null
    device precision_t* scan_h;           // (B, T, H)
    device precision_t* out;              // (B, T, H)
    device precision_t* next_state;       // (B, H)
    device precision_t* combined;         // (B*T, 3H) projection, as the GEMM stores it
    int B, T, H;
};

struct MingruFused64BwdArgs {
    device precision_t* x;                // (B*T, H) saved layer input
    device precision_t* W;                // (3H, H)
    device const precision_t* terminals;  // (B, T) or null
    device const precision_t* scan_h;     // (B, T, H)
    device const precision_t* grad_out;   // (B, T, H)
    device const precision_t* grad_next_state;  // (B, H)
    device precision_t* grad_combined;    // (B, T, 3H)
    device precision_t* grad_state;       // (B, H)
    device precision_t* grad_input;       // (B, T, H)
    device precision_t* grad_x;           // (B*T, H) dX = grad_combined @ W + grad_input
    device const precision_t* combined;   // (B*T, 3H) from the forward
    int B, T, H;
};

#ifndef PRECISION_FLOAT
// BF16 H64 forward projection of one chunk: comb (16 rows x 192, [hidden |
// gate | proj]) = x[row0 .. row0+15] @ W^T, stored by the matmul straight into
// the threadgroup tile (RNE, same bits as from_float of the fp32 result; see
// mgf_project and tests/metal/run_test_proj_store.sh). Two simdgroups.
static inline void mgf64_project_tg(device precision_t* xp, device precision_t* Wp, int M,
        int row0, threadgroup precision_t* comb) {
    using namespace mpp::tensor_ops;
    constexpr int H3 = 3 * MGF64_H;
    auto A = tensor<device precision_t, dextents<int32_t, 2>, tensor_inline>(
        xp, dextents<int32_t, 2>(MGF64_H, M));
    auto Bw = tensor<device precision_t, dextents<int32_t, 2>, tensor_inline>(
        Wp, dextents<int32_t, 2>(MGF64_H, H3));
    auto Ct = tensor<threadgroup precision_t, dextents<int32_t, 2>, tensor_inline>(
        comb, dextents<int32_t, 2>(H3, MGF64_CHUNK));
    constexpr auto desc = matmul2d_descriptor(MGF64_CHUNK, MGF64_FWD_N,
        static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<2>> op;
    auto mA = A.slice(0, row0);
    for (int n0 = 0; n0 < H3; n0 += MGF64_FWD_N) {
        auto mB = Bw.slice(0, n0);
        auto mC = Ct.slice(n0, 0);
        op.run(mA, mB, mC);
    }
}
#endif

kernel void mingru_fused_forward_h64(constant MingruFused64FwdArgs& a [[buffer(0)]],
        uint tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int T_seq = a.T, H = MGF64_H, H3 = 3 * MGF64_H;
    // Uniform guard (barrier-safe): the tile must hold whole sequences and
    // lie inside the (B*T, H) input.
    if (a.H != MGF64_H || T_seq <= 0 || MGF_ROWS % T_seq != 0
            || (long)(tile + 1) * MGF_ROWS > (long)a.B * T_seq) return;
    threadgroup precision_t comb[MGF64_CHUNK * 3 * MGF64_H];
    const int M = a.B * T_seq;
    // Steps of one sequence per chunk, and sequences per chunk.
    const int steps = min(T_seq, MGF64_CHUNK), nseq = MGF64_CHUNK / steps;
    float carry = 0.0f;  // T > 16: lane tid's h_t after the previous chunk

#ifdef PRECISION_FLOAT
    using namespace mpp::tensor_ops;
    auto A = tensor<device precision_t, dextents<int32_t, 2>, tensor_inline>(
        a.x, dextents<int32_t, 2>(H, M));
    auto Bw = tensor<device precision_t, dextents<int32_t, 2>, tensor_inline>(
        a.W, dextents<int32_t, 2>(H, H3));
    constexpr auto desc = matmul2d_descriptor(MGF64_CHUNK, MGF64_FWD_N,
        static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<2>> op;
#endif
    for (int chunk = 0; chunk < MGF_ROWS / MGF64_CHUNK; chunk++) {
        const int row0 = (int)tile * MGF_ROWS + chunk * MGF64_CHUNK;
        // comb (16 rows x 192, [hidden | gate | proj]) = x[row0 ..] @ W^T.
#ifdef PRECISION_FLOAT
        auto mA = A.slice(0, row0);
        for (int n0 = 0; n0 < H3; n0 += MGF64_FWD_N) {
            auto mB = Bw.slice(0, n0);
            auto cT = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
            #pragma unroll
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                if (cT.is_valid_element(i)) cT[i] = 0.0f;
            }
            op.run(mA, mB, cT);
            #pragma unroll
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                if (!cT.is_valid_element(i)) continue;
                auto idx = cT.get_multidimensional_index(i);
                precision_t v = from_float(cT[i]);
                comb[idx[1] * H3 + n0 + idx[0]] = v;
                a.combined[(long)(row0 + idx[1]) * H3 + n0 + idx[0]] = v;
            }
        }
#else
        {
            mgf64_project_tg(a.x, a.W, M, row0, comb);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // BF16: the matmul stores straight into comb (the same RNE
            // conversion as from_float, see mgf_project); the chunk's rows are
            // contiguous in `combined`, so copy them out in 16-byte words.
            // ~5% faster than per-element stores (Pong forward). FP32 keeps
            // the per-element path (no gain measured).
            device uint4* dst = (device uint4*)(a.combined + (long)row0 * H3);
            threadgroup const uint4* src = (threadgroup const uint4*)comb;
            constexpr int n16 = MGF64_CHUNK * 3 * MGF64_H * (int)sizeof(precision_t) / 16;
            for (int i = (int)tid; i < n16; i += 64) dst[i] = src[i];
        }
#endif
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int lane = (int)tid; lane < nseq * H; lane += 64) {
            int seq = lane / H, h = lane % H;
            int b = row0 / T_seq + seq;
            int t_begin = row0 - b * T_seq + seq * steps;  // chunk's first step of b
            int bH = b * H, bHT = bH * T_seq;
            int out_base = bHT + h;
            device const precision_t* term_row =
                a.terminals ? a.terminals + b * T_seq : (device const precision_t*)nullptr;
            float h_t = t_begin == 0 ? to_float(a.state[bH + h]) : carry;
            int r = seq * steps * H3 + h;
            int t = t_begin;
            for (; t < t_begin + steps; t++) {
                if (term_row && to_float(term_row[t]) != 0.0f) {
                    h_t = 0.0f;
                }
                a.scan_h[out_base + t * H] = from_float(h_t);

                float hidden_val = to_float(comb[r]);
                float gate_val = to_float(comb[r + H]);
                float proj_val = to_float(comb[r + 2 * H]);
                float x_val = to_float(a.x[out_base + t * H]);

                float z = mg_sigmoid(gate_val);
                float h_tilde = mg_h_tilde(hidden_val);
                h_t = mg_lerp(h_t, h_tilde, z);

                float s = mg_sigmoid(proj_val);
                a.out[out_base + t * H] = from_float(fma(s, h_t, (1.0f - s) * x_val));
                h_t = to_float(from_float(h_t));
                r += H3;
            }
            if (t_begin + steps == T_seq) a.next_state[bH + h] = from_float(h_t);
            else carry = h_t;
        }
        // The next chunk's projection overwrites comb.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void mingru_fused_backward_h64(constant MingruFused64BwdArgs& a [[buffer(0)]],
        uint tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const int T_seq = a.T, H = MGF64_H, H3 = 3 * MGF64_H;
    if (a.H != MGF64_H || T_seq <= 0 || MGF_ROWS % T_seq != 0
            || (long)(tile + 1) * MGF_ROWS > (long)a.B * T_seq) return;
    threadgroup precision_t comb[MGF64_CHUNK * 3 * MGF64_H];
    // This chunk's grad_input rows, for the epilogue (device copy optional).
    threadgroup precision_t gin[MGF64_CHUNK * MGF64_H];
    const int M = a.B * T_seq;
    const int steps = min(T_seq, MGF64_CHUNK), nseq = MGF64_CHUNK / steps;
    float carry = 0.0f;  // T > 16: lane tid's dh after the next chunk
    const bool store_gin = a.grad_input != nullptr;

    using namespace mpp::tensor_ops;
    auto A_cm = tensor<threadgroup precision_t, dextents<int32_t, 2>, tensor_inline>(
        comb, dextents<int32_t, 2>(H3, MGF64_CHUNK));
    auto B_cm = tensor<device precision_t, dextents<int32_t, 2>, tensor_inline>(
        a.W, dextents<int32_t, 2>(H, H3));
    auto C_cm = tensor<device precision_t, dextents<int32_t, 2>, tensor_inline>(
        a.grad_x, dextents<int32_t, 2>(H, M));
    auto D_cm = tensor<threadgroup precision_t, dextents<int32_t, 2>, tensor_inline>(
        gin, dextents<int32_t, 2>(H, MGF64_CHUNK));
    constexpr auto desc = matmul2d_descriptor(MGF64_CHUNK, MGF64_BWD_N,
        static_cast<int>(dynamic_extent), false, false, false);
    matmul2d<desc, execution_simdgroups<2>> op;
    for (int chunk = MGF_ROWS / MGF64_CHUNK - 1; chunk >= 0; chunk--) {
        const int row0 = (int)tile * MGF_ROWS + chunk * MGF64_CHUNK;
        // Reverse scan over the chunk; gradients go to grad_combined and to
        // the threadgroup tile the dX GEMM below reads.
        for (int lane = (int)tid; lane < nseq * H; lane += 64) {
            int seq = lane / H, h = lane % H;
            int b = row0 / T_seq + seq;
            int t_begin = row0 - b * T_seq + seq * steps;
            int bHT = b * H * T_seq;
            int cbase = 3 * bHT;
            int state_idx = b * H + h;
            int out_base = bHT + h;
            device const precision_t* term_row =
                a.terminals ? a.terminals + b * T_seq : (device const precision_t*)nullptr;

            float dh = t_begin + steps == T_seq ? to_float(a.grad_next_state[state_idx]) : carry;
            // Independent device reads issued ahead of the serial dh chain;
            // t and all FP32 arithmetic keep the original reverse order.
            constexpr int PREFETCH = MGF64_BWD_PREFETCH;
            for (int top = t_begin + steps - 1; top >= t_begin; top -= PREFETCH) {
                float hp[PREFETCH], xv[PREFETCH], go[PREFETCH], tv[PREFETCH],
                    hv[PREFETCH], gv[PREFETCH], pv[PREFETCH];
                for (int j = 0; j < PREFETCH; ++j) {
                    int t0 = top - j;
                    if (t0 >= t_begin) {
                        int i = out_base + t0 * H;
                        int c = cbase + t0 * H3 + h;
                        hp[j] = to_float(a.scan_h[i]);
                        xv[j] = to_float(a.x[i]);
                        go[j] = to_float(a.grad_out[i]);
                        tv[j] = term_row ? to_float(term_row[t0]) : 0.0f;
                        hv[j] = to_float(a.combined[c]);
                        gv[j] = to_float(a.combined[c + H]);
                        pv[j] = to_float(a.combined[c + 2 * H]);
                    }
                }
                for (int j = 0; j < PREFETCH && top - j >= t_begin; ++j) {
                    int t0 = top - j;
                    int c = cbase + t0 * H3 + h;
                    int input_idx = out_base + t0 * H;
                    int r = (seq * steps + t0 - t_begin) * H3 + h;
                    float h_prev = hp[j];
                    float hidden_val = hv[j];
                    float gate_val = gv[j];
                    float proj_val = pv[j];
                    float x_val = xv[j];
                    float grad_out_val = go[j];
                    float z = mg_sigmoid(gate_val);
                    float h_tilde = mg_h_tilde(hidden_val);
                    float h_t = mg_lerp(h_prev, h_tilde, z);
                    float s = mg_sigmoid(proj_val);
                    // highway: out = s*h + (1-s)*x
                    float grad_proj = grad_out_val * (h_t - x_val) * s * (1.0f - s);
                    precision_t gi = from_float(grad_out_val * (1.0f - s));
                    gin[(seq * steps + t0 - t_begin) * H + h] = gi;
                    if (store_gin) a.grad_input[input_idx] = gi;
                    a.grad_combined[c + 2 * H] = from_float(grad_proj);

                    float dh_total = fma(grad_out_val, s, dh);

                    float d_h_prev = dh_total * (1.0f - z);
                    float d_h_tilde = dh_total * z;
                    float d_z = dh_total * (h_tilde - h_prev);
                    float d_gate = d_z * z * (1.0f - z);
                    float d_hidden = (hidden_val >= 0.0f)
                        ? d_h_tilde
                        : d_h_tilde * h_tilde * (1.0f - h_tilde);

                    a.grad_combined[c] = from_float(d_hidden);
                    a.grad_combined[c + H] = from_float(d_gate);
                    comb[r] = from_float(d_hidden);
                    comb[r + H] = from_float(d_gate);
                    comb[r + 2 * H] = from_float(grad_proj);

                    dh = d_h_prev;
                    if (tv[j] != 0.0f) {
                        dh = 0.0f;
                    }
                }
            }
            if (t_begin == 0) a.grad_state[state_idx] = from_float(dh);
            else carry = dh;
        }
        // The epilogue reads this chunk's grad_input rows (gin) written by
        // other threads above and writes grad_x rows whose grad_out (possibly
        // the same buffer) every thread has finished reading; later chunks
        // read only their own (earlier) rows.
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        // grad_x = grad_combined @ W + grad_input (tensor_ops_gemm_nn_add).
        for (int j = 0; j < 64 / MGF64_BWD_N; j++) {
            auto mSecond = B_cm.slice(j * MGF64_BWD_N, 0);
            auto cT = op.get_destination_cooperative_tensor<decltype(A_cm), decltype(mSecond), float>();
            #pragma unroll
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                if (cT.is_valid_element(i)) cT[i] = 0.0f;
            }
            op.run(A_cm, mSecond, cT);
#ifdef PRECISION_FLOAT
            auto mResult = C_cm.slice(j * MGF64_BWD_N, row0);
            auto mAdd = D_cm.slice(j * MGF64_BWD_N, 0);
            auto dT = op.get_destination_cooperative_tensor<decltype(A_cm), decltype(mSecond), float>();
            dT.load(mAdd);
            #pragma unroll
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                if (cT.is_valid_element(i)) cT[i] += dT[i];
            }
            cT.store(mResult);
#else
            // Match bf16(bf16(G @ W) + grad_input): both generic stores round.
            #pragma unroll
            for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
                if (!cT.is_valid_element(i)) continue;
                auto idx = cT.get_multidimensional_index(i);
                int col = j * MGF64_BWD_N + idx[0];
                long ci = (long)(row0 + idx[1]) * H + col;
                a.grad_x[ci] = from_float(round_p(cT[i]) + to_float(gin[idx[1] * H + col]));
            }
#endif
        }
        // The next chunk's scan overwrites comb and gin.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#if defined(PUF_TEST_PROJ_STORE) && !defined(PRECISION_FLOAT)
#include "../../../tests/metal/proj_store_counters.h"
// Test-only kernels (tests/metal/run_test_proj_store.sh; never in the shipped
// library): the BF16 fused projections store the matmul result straight into a
// bfloat threadgroup tile. Check that store against the original reference,
// from_float() of the fp32 cooperative-tensor result, over every fp32 pattern.
// x rows are built so output column n (n < slots) accumulates three exact
// bf16 parts hi + mid + lo of a chosen fp32 target (W[n][3n..3n+2] = w, other
// W entries 0). Series 0: targets with biased exponent 24..254 (w = 1);
// series 1: exponent 0..23, parts of target * 2^64 with w = 2^-64; series 2:
// x[0] = each bf16 inf/NaN pattern (w = 1). Series 3 is a short slice/column
// probe: x[0] = +/-1, W[n][0] = n+1, so every one of the 3K output columns
// has a distinct, nonzero result, including every gate/projection tile.
struct PstArgs {
    device bfloat* x;          // (rows, K)
    device bfloat* W;          // (3K, K)
    device atomic_uint* cnt;   // PstCounter (shared with the host)
    uint base, total, series, K, slots, rows;
};

static inline bool pst_target(constant PstArgs& a, uint p, thread uint& bits) {
    if (p >= a.total) return false;
    if (a.series == 3) {
        bits = as_type<uint>(float(p % (3 * a.K) + 1)) | (((p / (3 * a.K)) & 1u) << 31);
        return true;
    }
    if (a.series == 2) { bits = (((p >> 7) & 1u) << 31) | 0x7f800000u | ((p & 127u) << 16); return true; }
    const uint nexp = a.series == 0 ? 231u : 24u, e0 = a.series == 0 ? 24u : 0u;
    const uint per_sign = nexp << 23;
    uint sign = p / per_sign, rem = p % per_sign;
    bits = (sign << 31) | ((e0 + (rem >> 23)) << 23) | (rem & 0x7fffffu);
    return true;
}

static inline bool pst_controlled(constant PstArgs& a, uint row, uint n, thread uint& bits) {
    if (a.series == 3) return pst_target(a, a.base + row * 3 * a.K + n, bits);
    if (a.series == 2) return n == 0 && pst_target(a, a.base + row, bits);
    return n < a.slots && pst_target(a, a.base + row * a.slots + n, bits);
}

kernel void pst_gen(constant PstArgs& a [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= a.rows * a.K) return;
    uint row = gid / a.K, k = gid % a.K, j = k / 3, part = k % 3;
    uint pb = 0;
    uint bits;
    if (a.series == 3) {
        if (k == 0) pb = (row & 1u) ? 0xbf80u : 0x3f80u;
    } else if (a.series == 2) {
        if (k == 0 && pst_target(a, a.base + row, bits)) pb = bits >> 16;
    } else if (j < a.slots && pst_target(a, a.base + row * a.slots + j, bits)) {
        float s = as_type<float>(bits);
        if (a.series == 1) s *= 0x1p64f;  // exact: the result is normal
        float hi = as_type<float>(as_type<uint>(s) & 0xffff0000u);
        float r1 = s - hi;  // exact
        float mid = as_type<float>(as_type<uint>(r1) & 0xffff0000u);
        float lo = r1 - mid;  // exact, at most 8 significant bits
        float v = part == 0 ? hi : part == 1 ? mid : lo;
        pb = as_type<uint>(v) >> 16;
    }
    a.x[gid] = as_type<bfloat>(ushort(pb));
}

// Reference: the original per-element store of the fp32 cooperative tensor.
template <int ROWS, int NT, int SG, int K>
static inline void pst_ref_project(constant PstArgs& a, int row0, threadgroup bfloat* comb) {
    using namespace mpp::tensor_ops;
    auto A = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        a.x, dextents<int32_t, 2>(K, (int)a.rows));
    auto B = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>(
        a.W, dextents<int32_t, 2>(K, 3 * K));
    constexpr auto desc = matmul2d_descriptor(ROWS, NT,
        static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = A.slice(0, row0);
    uint acc[5] = {};  // exact, positive/negative FTZ, negative zero -> +0, bad
    for (int j = 0; j < 3 * K; j += NT) {
        auto mB = B.slice(0, j);
        auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
        #pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
            if (cT.is_valid_element(i)) cT[i] = 0.0f;
        }
        op.run(mA, mB, cT);
        #pragma unroll
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
            if (!cT.is_valid_element(i)) continue;
            auto idx = cT.get_multidimensional_index(i);
            uint bits;
            if (a.series != 2 && pst_controlled(a, row0 + idx[1], j + idx[0], bits)) {
                // Check the observed FP32 accumulator BEFORE BF16 rounding.
                // The MMA flushes FP32 subnormals and sums signed zeros to +0.
                uint mag = bits & 0x7fffffffu;
                uint expected = mag < 0x00800000u ? 0u : bits;
                uint kind = mag == 0 ? (bits ? 3u : 0u)
                    : mag < 0x00800000u ? ((bits >> 31) ? 2u : 1u) : 0u;
                acc[as_type<uint>(cT[i]) == expected ? kind : 4u]++;
            }
#ifdef PST_MUTATE_REF
            // Mutation hook (run_test_proj_store.sh --mutate): truncating store.
            comb[idx[1] * 3 * K + j + idx[0]] = as_type<bfloat>(ushort(as_type<uint>(cT[i]) >> 16));
#else
            comb[idx[1] * 3 * K + j + idx[0]] = from_float(cT[i]);
#endif
        }
    }
    for (uint k = 0; k < 5; k++)
        if (acc[k]) atomic_fetch_add_explicit(&a.cnt[PST_ACC_EXACT + k], acc[k], memory_order_relaxed);
}

template <int ROWS, int K>
static inline void pst_compare(constant PstArgs& a, int row0, threadgroup const bfloat* prod,
        threadgroup const bfloat* ref, uint tid, uint nthreads) {
    uint mism = 0, nanb = 0, cov = 0, ctl = 0, nz = 0, sub = 0;
    for (uint e = tid; e < (uint)(ROWS * 3 * K); e += nthreads) {
        ushort pb = as_type<ushort>(prod[e]), rb = as_type<ushort>(ref[e]);
        bool pn = (pb & 0x7fffu) > 0x7f80u, rn = (rb & 0x7fffu) > 0x7f80u;
        if (pb != rb) { mism++; if (pn && rn) nanb++; }
        uint row = (uint)row0 + e / (3 * K), n = e % (3 * K), bits;
        if (pst_controlled(a, row, n, bits)) {
            ctl++;
            if (rb == puf_f32_to_bf16_bits(as_type<float>(bits))) cov++;
            if ((rb & 0x7fffu) != 0) nz++;
            if ((rb & 0x7f80u) == 0 && (rb & 0x7fu) != 0) sub++;
        }
    }
    if (mism) atomic_fetch_add_explicit(&a.cnt[PST_MISMATCH], mism, memory_order_relaxed);
    if (nanb) atomic_fetch_add_explicit(&a.cnt[PST_NAN_BITS_DIFF], nanb, memory_order_relaxed);
    if (cov) atomic_fetch_add_explicit(&a.cnt[PST_ROUNDED_TARGET], cov, memory_order_relaxed);
    if (ctl) atomic_fetch_add_explicit(&a.cnt[PST_CONTROLLED], ctl, memory_order_relaxed);
    if (nz) atomic_fetch_add_explicit(&a.cnt[PST_NONZERO], nz, memory_order_relaxed);
    if (sub) atomic_fetch_add_explicit(&a.cnt[PST_SUBNORMAL], sub, memory_order_relaxed);
}

// mgf_project (64 rows, 4 simdgroups) vs the old BF16 (64 x 32) x 3 tiles.
kernel void pst_check64(constant PstArgs& a [[buffer(0)]],
        uint tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup bfloat prod[MGF_ROWS * 3 * MGF_H], ref[MGF_ROWS * 3 * MGF_H];
    mgf_project(a.x, a.W, (int)a.rows, tile, prod);
    pst_ref_project<64, 32, 4, MGF_H>(a, (int)tile * 64, ref);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pst_compare<64, MGF_H>(a, (int)tile * 64, prod, ref, tid, 128);
}

// mgf_project32 (32 rows, 2 simdgroups) vs the old (32 x 32) x 3 tiles.
kernel void pst_check32(constant PstArgs& a [[buffer(0)]],
        uint tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup bfloat prod[32 * 3 * MGF_H], ref[32 * 3 * MGF_H];
    mgf_project32(a.x, a.W, (int)a.rows, tile, prod);
    pst_ref_project<32, 32, 2, MGF_H>(a, (int)tile * 32, ref);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pst_compare<32, MGF_H>(a, (int)tile * 32, prod, ref, tid, 64);
}

// mgf64_project_tg (16 rows, 2 simdgroups) vs the old H64 (16 x MGF64_FWD_N) tiles.
kernel void pst_check_h64(constant PstArgs& a [[buffer(0)]],
        uint tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup bfloat prod[MGF64_CHUNK * 3 * MGF64_H], ref[MGF64_CHUNK * 3 * MGF64_H];
    const int row0 = (int)tile * MGF64_CHUNK;
    mgf64_project_tg(a.x, a.W, (int)a.rows, row0, prod);
    pst_ref_project<MGF64_CHUNK, MGF64_FWD_N, 2, MGF64_H>(a, row0, ref);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pst_compare<MGF64_CHUNK, MGF64_H>(a, row0, prod, ref, tid, 64);
}
#endif

// ---- exhaustive sigmoid check (tests/metal/run_test_sigmoid.sh) --------------
#ifdef PUF_TEST_SIGMOID
PUF_SIG_TEST_KERNELS(mingru, mg_sigmoid)
#endif

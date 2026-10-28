// Metal adapters for the MinGRU train scans and decoder-backward helpers
// (src/algo.cu; MSL in src/metal/shaders/mingru.metal). Included from the end
// of src/metal/kernels.h, so PUF_ADAPTER and puf_mtl::launch are in scope.
//
// The scan adapters take the upstream PrefixScan by a template parameter so
// this header also compiles in tests that include kernels.h without the
// upstream types; any struct with PrefixScan's field names works.
#pragma once

struct MtlMingruScanFwdArgs {
    uint64_t combined, state, input, terminals, scan_h, out, next_state;
    int32_t B, T, H;
};
static_assert(sizeof(MtlMingruScanFwdArgs) == 72, "MingruScanFwdArgs layout");

struct MtlMingruScanBwdArgs {
    uint64_t combined, input, terminals, scan_h, grad_out, grad_next_state,
        grad_combined, grad_state, grad_input;
    int32_t B, T, H;
};
static_assert(sizeof(MtlMingruScanBwdArgs) == 88, "MingruScanBwdArgs layout");

template <typename Scan>
static inline MtlMingruScanFwdArgs puf_mtl_scan_fwd_args(const Scan& s) {
    return MtlMingruScanFwdArgs{puf_mtl::dptr(s.combined_ptr), puf_mtl::dptr(s.state_ptr),
        puf_mtl::dptr(s.input_ptr), puf_mtl::dptr(s.terminals_ptr),
        puf_mtl::dptr(s.scan_h.data), puf_mtl::dptr(s.out.data),
        puf_mtl::dptr(s.next_state.data), (int32_t)s.B, (int32_t)s.T, (int32_t)s.H};
}

template <typename Scan>
static inline MtlMingruScanBwdArgs puf_mtl_scan_bwd_args(const Scan& s,
        const precision_t* grad_out, const precision_t* grad_next_state) {
    return MtlMingruScanBwdArgs{puf_mtl::dptr(s.combined_ptr), puf_mtl::dptr(s.input_ptr),
        puf_mtl::dptr(s.terminals_ptr), puf_mtl::dptr(s.scan_h.data),
        puf_mtl::dptr(grad_out), puf_mtl::dptr(grad_next_state),
        puf_mtl::dptr(s.grad_combined.data), puf_mtl::dptr(s.grad_state.data),
        puf_mtl::dptr(s.grad_input.data), (int32_t)s.B, (int32_t)s.T, (int32_t)s.H};
}

template <typename Scan>
PUF_ADAPTER(mingru_scan_forward, const Scan& scan) {
    puf_mtl::launch(cfg, "mingru_scan_forward", puf_mtl_scan_fwd_args(scan));
}

template <typename Scan>
PUF_ADAPTER(mingru_scan_forward_row, const Scan& scan) {
    puf_mtl::launch(cfg, "mingru_scan_forward_row", puf_mtl_scan_fwd_args(scan));
}

template <typename Scan>
PUF_ADAPTER(mingru_scan_backward, const Scan& scan, const precision_t* grad_out,
        const precision_t* grad_next_state) {
    puf_mtl::launch(cfg, "mingru_scan_backward",
        puf_mtl_scan_bwd_args(scan, grad_out, grad_next_state));
}

template <typename Scan>
PUF_ADAPTER(mingru_scan_backward_row, const Scan& scan, const precision_t* grad_out,
        const precision_t* grad_next_state) {
    puf_mtl::launch(cfg, "mingru_scan_backward_row",
        puf_mtl_scan_bwd_args(scan, grad_out, grad_next_state));
}

struct MtlSumRowsArgs { uint64_t dst, src; int32_t R, C; };
static_assert(sizeof(MtlSumRowsArgs) == 24, "SumRowsArgs layout");
PUF_ADAPTER(sum_rows_to_precision_kernel, precision_t* dst, const float* src, int R, int C) {
    puf_mtl::launch(cfg, "sum_rows_to_precision_kernel",
        MtlSumRowsArgs{puf_mtl::dptr(dst), puf_mtl::dptr(src), R, C});
}

struct MtlAssembleDecoderGradArgs {
    uint64_t dst, grad_logits, grad_value;
    int32_t B_TT, od, od_plus_1;
};
static_assert(sizeof(MtlAssembleDecoderGradArgs) == 40, "AssembleDecoderGradArgs layout");
PUF_ADAPTER(assemble_decoder_grad, precision_t* dst, const float* grad_logits,
        const float* grad_value, int B_TT, int od, int od_plus_1) {
    puf_mtl::launch(cfg, "assemble_decoder_grad", MtlAssembleDecoderGradArgs{
        puf_mtl::dptr(dst), puf_mtl::dptr(grad_logits), puf_mtl::dptr(grad_value),
        B_TT, od, od_plus_1});
}

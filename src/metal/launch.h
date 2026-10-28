// Metal backend launch layer.
//
// Upstream launch sites are written  PUF_LAUNCH(kernel, grid, block, smem, stream)(args...)
// which nvcc expands to  kernel<<<grid, block, smem, stream>>>(args...).
// Here it builds a puf_mtl::Launch<puf_kid_kernel> whose call operator makes an
// unqualified call
//     puf_mtl_kernel((puf_kid_kernel*)nullptr, cfg, args...)
// resolved at the end of the translation unit (ADL on the tag type). Each
// ported kernel provides an adapter overload in src/metal/kernels.h that
// translates pointers to GPU addresses and dispatches the MSL kernel. Kernels
// (or argument-type overloads) without an adapter fall through to the
// variadic catch-all below and abort with "metal: unimplemented kernel <name>".
#pragma once

#include <utility>

#include "compat/cuda_runtime.h"

namespace puf_mtl {

struct LaunchCfg {
    const char* name;
    dim3 grid;
    dim3 block;
    size_t smem;
    cudaStream_t stream;
};

// Converts from any kernel tag pointer; ranks below every exact adapter.
struct Unported {
    template <typename K>
    Unported(K*) {}
};

template <typename K>
struct Launch {
    LaunchCfg cfg;
    template <typename... Args>
    void operator()(Args&&... args) const {
        puf_mtl_kernel((K*)nullptr, cfg, std::forward<Args>(args)...);
    }
};

}  // namespace puf_mtl

// -DPUF_METAL_STRICT_LAUNCH drops the catch-all so a launch without an
// adapter for its exact argument types (e.g. a precision mismatch) is a
// compile error instead of a runtime abort.
#ifndef PUF_METAL_STRICT_LAUNCH
[[noreturn]] void puf_mtl_kernel(puf_mtl::Unported, const puf_mtl::LaunchCfg& cfg, ...);
#endif

#define PUF_LAUNCH(k, g, b, s, st) \
    ::puf_mtl::Launch<puf_kid_##k>{{#k, dim3(g), dim3(b), (size_t)(s), (cudaStream_t)(st)}}

// One tag per upstream __global__ kernel name (overloads share a tag).
struct puf_kid_cast;
struct puf_kid_cast_rew_term;
struct puf_kid_rng_init;
struct puf_kid_sample_logits;
struct puf_kid_snapshot_state;
struct puf_kid_zero_term_state;
struct puf_kid_log_reduce;
struct puf_kid_transpose_102;
struct puf_kid_clamp_precision_kernel;
struct puf_kid_puf_stamp;
struct puf_kid_uniform_scale_kernel;
struct puf_kid_mingru_gate;
struct puf_kid_mingru_scan_forward;
struct puf_kid_mingru_scan_backward;
struct puf_kid_mingru_scan_forward_row;
struct puf_kid_mingru_scan_backward_row;
struct puf_kid_sum_rows_to_precision_kernel;
struct puf_kid_assemble_decoder_grad;
struct puf_kid_add_kernel;
struct puf_kid_muon_sum_sq_partials;
struct puf_kid_muon_sum_sq_reduce;
struct puf_kid_muon_clip_nesterov;
struct puf_kid_muon_l2_normalize;
struct puf_kid_muon_store_update;
struct puf_kid_muon_weight_update;
struct puf_kid_cache_imp_and_v;
struct puf_kid_ppo_loss_compute;
struct puf_kid_ppo_loss_reduce;
struct puf_kid_puff_advantage;

// Shared point-entity encoders (minimal / asteroids).
struct puf_kid_me_materialize_points_kernel;
struct puf_kid_me_linear_max_kernel;
struct puf_kid_me_output_wgrad_kernel;
struct puf_kid_me_grad_entity_kernel;
struct puf_kid_ae_materialize_points_kernel;
struct puf_kid_ae_linear_max_kernel;
struct puf_kid_ae_output_wgrad_kernel;
struct puf_kid_ae_grad_entity_kernel;

// OSRS entity encoder (osrs_colosseum / inferno / pvp / zulrah).
struct puf_kid_osrs_entity_gather_branch;
struct puf_kid_osrs_entity_fused_pool_fwd;
struct puf_kid_osrs_entity_fused_l2_wgrad;
struct puf_kid_osrs_entity_fused_grad_z1;

// Craftax category-embedding encoder.
struct puf_kid_craftax_embed_bag_kernel;
struct puf_kid_craftax_copy_scalars_kernel;
struct puf_kid_craftax_embed_bag_backward_kernel;
struct puf_kid_craftax_fxp_to_precision_kernel;

// NMMO3 conv encoder.
struct puf_kid_n3_c1_im2col_obs;
struct puf_kid_n3_c2_im2col;
struct puf_kid_n3_c2_col2im;
struct puf_kid_n3_rows_to_nchw;
struct puf_kid_n3_nchw_to_rows;
struct puf_kid_n3_embedding_kernel;
struct puf_kid_n3_concat_kernel;
struct puf_kid_n3_relu_kernel;
struct puf_kid_n3_relu_backward_kernel;
struct puf_kid_n3_concat_backward_conv_kernel;
struct puf_kid_n3_embedding_backward_kernel;
struct puf_kid_n3_fxp_to_precision_kernel;

// NetHack encoder and pointer decoder.
struct puf_kid_nh_bias_relu_kernel;
struct puf_kid_nh_bias_kernel;
struct puf_kid_nh_f32_to_precision_kernel;
struct puf_kid_nh_fxp_to_precision_kernel;
struct puf_kid_nh_secband_init_kernel;
struct puf_kid_nh_loc3_gather_kernel;
struct puf_kid_nh_loc3_scatter_kernel;
struct puf_kid_nh_terr_feat_kernel;
struct puf_kid_nh_lab_tok_kernel;
struct puf_kid_nh_lab_dE_scatter_kernel;
struct puf_kid_nh_decode_kernel;
struct puf_kid_nh_crop_kernel;
struct puf_kid_nh_eff_embed_kernel;
struct puf_kid_nh_ekind_grad_kernel;
struct puf_kid_nh_esub_grad_kernel;
struct puf_kid_nh_min_summax_kernel;
struct puf_kid_nh_min_dv_kernel;
struct puf_kid_nh_min_sgate_kernel;
struct puf_kid_nh_min_sgate_bwd_kernel;
struct puf_kid_nh_pass_kernel;
struct puf_kid_nh_passbwd_kernel;
struct puf_kid_nh_blstats_kernel;
struct puf_kid_nh_msg_ids_kernel;
struct puf_kid_nh_msg_pool_kernel;
struct puf_kid_nh_msg_bwd_kernel;
struct puf_kid_nh_spkey_kernel;
struct puf_kid_nh_sp_doorstep_kernel;
struct puf_kid_nh_sp2_dk_kernel;
struct puf_kid_nh_spkey_dE_kernel;
struct puf_kid_nh_concat_kernel;
struct puf_kid_nh_slice_kernel;
struct puf_kid_nh_inv_decode_kernel;
struct puf_kid_nh_inv_table_kernel;
struct puf_kid_nh_inv_sfeat_kernel;
struct puf_kid_nh_inv_gather_kernel;
struct puf_kid_nh_dTinv_scatter_kernel;
struct puf_kid_nh_add_inplace_kernel;
struct puf_kid_nh_relu_bias_bwd_kernel;
struct puf_kid_nh_bias_bwd_kernel;
struct puf_kid_nh_bias_flush_kernel;
struct puf_kid_nh_idemb_kernel;
struct puf_kid_nh_idemb_grad_kernel;
struct puf_kid_nh_dec_assemble_kernel;
struct puf_kid_nh_ptr_rownorm_kernel;
struct puf_kid_nh_ptr3_cos_kernel;
struct puf_kid_nh_dec_dtmp_kernel;
struct puf_kid_nh_ptr3_dq_kernel;
struct puf_kid_nh_spq_bwd_kernel;
struct puf_kid_nh_ptr3_dkmat_kernel;
struct puf_kid_nh_fill_kernel;

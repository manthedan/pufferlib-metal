// NetHack default encoder and pointer decoder adapters.
#pragma once
#ifdef PUFFER_NETHACK
#include "nethack_config.h"
static_assert(NH_MSG_VOCAB == 1024 && NH_MSG_HID == 256 && NH_LABK == 16
    && NH_PASS_DIM == 256 && NH_CONCAT == 1698 && NH_DEC_OD == 742,
    "Metal NetHack shaders require default compile-time policy dimensions");

struct Mtl_nh_bias_relu_kernel {
    uint64_t data;
    uint64_t bias;
    int total;
    int dim;
};
static_assert(sizeof(Mtl_nh_bias_relu_kernel) == 24, "Mtl_nh_bias_relu_kernel layout");
PUF_ENV_ADAPTER(nh_bias_relu_kernel, precision_t* data, const precision_t* bias, int total, int dim) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_bias_relu_kernel", Mtl_nh_bias_relu_kernel{puf_mtl::dptr(data), puf_mtl::dptr(bias), total, dim});
}

struct Mtl_nh_bias_kernel {
    uint64_t data;
    uint64_t bias;
    int total;
    int dim;
};
static_assert(sizeof(Mtl_nh_bias_kernel) == 24, "Mtl_nh_bias_kernel layout");
PUF_ENV_ADAPTER(nh_bias_kernel, precision_t* data, const precision_t* bias, int total, int dim) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_bias_kernel", Mtl_nh_bias_kernel{puf_mtl::dptr(data), puf_mtl::dptr(bias), total, dim});
}

struct Mtl_nh_f32_to_precision_kernel {
    uint64_t dst;
    uint64_t src;
    int n;
};
static_assert(sizeof(Mtl_nh_f32_to_precision_kernel) == 24, "Mtl_nh_f32_to_precision_kernel layout");
PUF_ENV_ADAPTER(nh_f32_to_precision_kernel, precision_t* dst, const float* src, int n) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_f32_to_precision_kernel", Mtl_nh_f32_to_precision_kernel{puf_mtl::dptr(dst), puf_mtl::dptr(src), n});
}

struct Mtl_nh_fxp_to_precision_kernel {
    uint64_t dst;
    uint64_t src;
    int n;
};
static_assert(sizeof(Mtl_nh_fxp_to_precision_kernel) == 24, "Mtl_nh_fxp_to_precision_kernel layout");
PUF_ENV_ADAPTER(nh_fxp_to_precision_kernel, precision_t* dst, const long long* src, int n) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_fxp_to_precision_kernel", Mtl_nh_fxp_to_precision_kernel{puf_mtl::dptr(dst), puf_mtl::dptr(src), n});
}

struct Mtl_nh_secband_init_kernel {
    uint64_t lut;
};
static_assert(sizeof(Mtl_nh_secband_init_kernel) == 8, "Mtl_nh_secband_init_kernel layout");
PUF_ENV_ADAPTER(nh_secband_init_kernel, unsigned char* lut) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_secband_init_kernel", Mtl_nh_secband_init_kernel{puf_mtl::dptr(lut)});
}

struct Mtl_nh_loc3_gather_kernel {
    uint64_t x;
    uint64_t locc_w;
    uint64_t crop_glyph;
    uint64_t lut;
    int B;
};
static_assert(sizeof(Mtl_nh_loc3_gather_kernel) == 40, "Mtl_nh_loc3_gather_kernel layout");
PUF_ENV_ADAPTER(nh_loc3_gather_kernel, precision_t* x, const precision_t* locc_w, const float* crop_glyph, const unsigned char* lut, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_loc3_gather_kernel", Mtl_nh_loc3_gather_kernel{puf_mtl::dptr(x), puf_mtl::dptr(locc_w), puf_mtl::dptr(crop_glyph), puf_mtl::dptr(lut), B});
}

struct Mtl_nh_loc3_scatter_kernel {
    uint64_t acc;
    uint64_t dx;
    uint64_t crop_glyph;
    uint64_t lut;
    int B;
};
static_assert(sizeof(Mtl_nh_loc3_scatter_kernel) == 40, "Mtl_nh_loc3_scatter_kernel layout");
PUF_ENV_ADAPTER(nh_loc3_scatter_kernel, long long* acc, const precision_t* dx, const float* crop_glyph, const unsigned char* lut, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_loc3_scatter_kernel", Mtl_nh_loc3_scatter_kernel{puf_mtl::dptr(acc), puf_mtl::dptr(dx), puf_mtl::dptr(crop_glyph), puf_mtl::dptr(lut), B});
}

struct Mtl_nh_terr_feat_kernel {
    uint64_t tf;
    uint64_t gidx;
    uint64_t obs;
    uint64_t lut;
    uint64_t sblut;
    int B;
};
static_assert(sizeof(Mtl_nh_terr_feat_kernel) == 48, "Mtl_nh_terr_feat_kernel layout");
PUF_ENV_ADAPTER(nh_terr_feat_kernel, precision_t* tf, const float* gidx, const precision_t* obs, const unsigned char* lut, const unsigned char* sblut, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_terr_feat_kernel", Mtl_nh_terr_feat_kernel{puf_mtl::dptr(tf), puf_mtl::dptr(gidx), puf_mtl::dptr(obs), puf_mtl::dptr(lut), puf_mtl::dptr(sblut), B});
}

struct Mtl_nh_lab_tok_kernel {
    uint64_t tok;
    uint64_t gid;
    uint64_t obs;
    uint64_t e_eff;
    uint64_t haz;
    int list_off;
    int is_mon;
    int B;
};
static_assert(sizeof(Mtl_nh_lab_tok_kernel) == 56, "Mtl_nh_lab_tok_kernel layout");
PUF_ENV_ADAPTER(nh_lab_tok_kernel, precision_t* tok, float* gid, const precision_t* obs, const precision_t* e_eff, const unsigned char* haz, int list_off, int is_mon, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_lab_tok_kernel", Mtl_nh_lab_tok_kernel{puf_mtl::dptr(tok), puf_mtl::dptr(gid), puf_mtl::dptr(obs), puf_mtl::dptr(e_eff), puf_mtl::dptr(haz), list_off, is_mon, B});
}

struct Mtl_nh_lab_dE_scatter_kernel {
    uint64_t dE_i;
    uint64_t dtok;
    uint64_t gid;
    int B;
};
static_assert(sizeof(Mtl_nh_lab_dE_scatter_kernel) == 32, "Mtl_nh_lab_dE_scatter_kernel layout");
PUF_ENV_ADAPTER(nh_lab_dE_scatter_kernel, long long* dE_i, const precision_t* dtok, const float* gid, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_lab_dE_scatter_kernel", Mtl_nh_lab_dE_scatter_kernel{puf_mtl::dptr(dE_i), puf_mtl::dptr(dtok), puf_mtl::dptr(gid), B});
}

struct Mtl_nh_decode_kernel {
    uint64_t idx;
    uint64_t obs;
    int B;
};
static_assert(sizeof(Mtl_nh_decode_kernel) == 24, "Mtl_nh_decode_kernel layout");
PUF_ENV_ADAPTER(nh_decode_kernel, float* idx, const precision_t* obs, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_decode_kernel", Mtl_nh_decode_kernel{puf_mtl::dptr(idx), puf_mtl::dptr(obs), B});
}

struct Mtl_nh_crop_kernel {
    uint64_t crop;
    uint64_t idx;
    uint64_t obs;
    int B;
};
static_assert(sizeof(Mtl_nh_crop_kernel) == 32, "Mtl_nh_crop_kernel layout");
PUF_ENV_ADAPTER(nh_crop_kernel, float* crop, const float* idx, const precision_t* obs, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_crop_kernel", Mtl_nh_crop_kernel{puf_mtl::dptr(crop), puf_mtl::dptr(idx), puf_mtl::dptr(obs), B});
}

struct Mtl_nh_eff_embed_kernel {
    uint64_t out;
    uint64_t E;
    uint64_t K;
    uint64_t S;
};
static_assert(sizeof(Mtl_nh_eff_embed_kernel) == 32, "Mtl_nh_eff_embed_kernel layout");
PUF_ENV_ADAPTER(nh_eff_embed_kernel, precision_t* out, const precision_t* E, const precision_t* K, const precision_t* S) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_eff_embed_kernel", Mtl_nh_eff_embed_kernel{puf_mtl::dptr(out), puf_mtl::dptr(E), puf_mtl::dptr(K), puf_mtl::dptr(S)});
}

struct Mtl_nh_ekind_grad_kernel {
    uint64_t dK;
    uint64_t dE;
};
static_assert(sizeof(Mtl_nh_ekind_grad_kernel) == 16, "Mtl_nh_ekind_grad_kernel layout");
PUF_ENV_ADAPTER(nh_ekind_grad_kernel, precision_t* dK, const precision_t* dE) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_ekind_grad_kernel", Mtl_nh_ekind_grad_kernel{puf_mtl::dptr(dK), puf_mtl::dptr(dE)});
}

struct Mtl_nh_esub_grad_kernel {
    uint64_t dS;
    uint64_t dE;
};
static_assert(sizeof(Mtl_nh_esub_grad_kernel) == 16, "Mtl_nh_esub_grad_kernel layout");
PUF_ENV_ADAPTER(nh_esub_grad_kernel, precision_t* dS, const precision_t* dE) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_esub_grad_kernel", Mtl_nh_esub_grad_kernel{puf_mtl::dptr(dS), puf_mtl::dptr(dE)});
}

struct Mtl_nh_min_summax_kernel {
    uint64_t concat;
    uint64_t amax;
    uint64_t v;
    uint64_t vid;
    float eps;
    int n;
    int padv;
    int off;
    int B;
};
static_assert(sizeof(Mtl_nh_min_summax_kernel) == 56, "Mtl_nh_min_summax_kernel layout");
PUF_ENV_ADAPTER(nh_min_summax_kernel, precision_t* concat, float* amax, const precision_t* v, const float* vid, float eps, int n, int padv, int off, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_min_summax_kernel", Mtl_nh_min_summax_kernel{puf_mtl::dptr(concat), puf_mtl::dptr(amax), puf_mtl::dptr(v), puf_mtl::dptr(vid), eps, n, padv, off, B});
}

struct Mtl_nh_min_dv_kernel {
    uint64_t dv;
    uint64_t grad_concat;
    uint64_t amax;
    uint64_t vid;
    float eps;
    int n;
    int padv;
    int off;
    int B;
};
static_assert(sizeof(Mtl_nh_min_dv_kernel) == 56, "Mtl_nh_min_dv_kernel layout");
PUF_ENV_ADAPTER(nh_min_dv_kernel, precision_t* dv, const precision_t* grad_concat, const float* amax, const float* vid, float eps, int n, int padv, int off, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_min_dv_kernel", Mtl_nh_min_dv_kernel{puf_mtl::dptr(dv), puf_mtl::dptr(grad_concat), puf_mtl::dptr(amax), puf_mtl::dptr(vid), eps, n, padv, off, B});
}

struct Mtl_nh_min_sgate_kernel {
    uint64_t concat;
    uint64_t rp;
    uint64_t tok;
    uint64_t gid;
    int underfoot;
    int off;
    int B;
};
static_assert(sizeof(Mtl_nh_min_sgate_kernel) == 48, "Mtl_nh_min_sgate_kernel layout");
PUF_ENV_ADAPTER(nh_min_sgate_kernel, precision_t* concat, const precision_t* rp, const precision_t* tok, const float* gid, int underfoot, int off, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_min_sgate_kernel", Mtl_nh_min_sgate_kernel{puf_mtl::dptr(concat), puf_mtl::dptr(rp), puf_mtl::dptr(tok), puf_mtl::dptr(gid), underfoot, off, B});
}

struct Mtl_nh_min_sgate_bwd_kernel {
    uint64_t drp;
    uint64_t grad_concat;
    uint64_t tok;
    uint64_t gid;
    int underfoot;
    int off;
    int B;
};
static_assert(sizeof(Mtl_nh_min_sgate_bwd_kernel) == 48, "Mtl_nh_min_sgate_bwd_kernel layout");
PUF_ENV_ADAPTER(nh_min_sgate_bwd_kernel, precision_t* drp, const precision_t* grad_concat, const precision_t* tok, const float* gid, int underfoot, int off, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_min_sgate_bwd_kernel", Mtl_nh_min_sgate_bwd_kernel{puf_mtl::dptr(drp), puf_mtl::dptr(grad_concat), puf_mtl::dptr(tok), puf_mtl::dptr(gid), underfoot, off, B});
}

struct Mtl_nh_pass_kernel {
    uint64_t concat;
    uint64_t inv_out;
    uint64_t sfeat;
    uint64_t vid;
    int nopass;
    int passoff;
    int B;
};
static_assert(sizeof(Mtl_nh_pass_kernel) == 48, "Mtl_nh_pass_kernel layout");
PUF_ENV_ADAPTER(nh_pass_kernel, precision_t* concat, const precision_t* inv_out, const precision_t* sfeat, const float* vid, int nopass, int passoff, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_pass_kernel", Mtl_nh_pass_kernel{puf_mtl::dptr(concat), puf_mtl::dptr(inv_out), puf_mtl::dptr(sfeat), puf_mtl::dptr(vid), nopass, passoff, B});
}

struct Mtl_nh_passbwd_kernel {
    uint64_t inv_grad;
    uint64_t grad_concat;
    uint64_t sfeat;
    uint64_t vid;
    int nopass;
    int passoff;
    int B;
};
static_assert(sizeof(Mtl_nh_passbwd_kernel) == 48, "Mtl_nh_passbwd_kernel layout");
PUF_ENV_ADAPTER(nh_passbwd_kernel, precision_t* inv_grad, const precision_t* grad_concat, const precision_t* sfeat, const float* vid, int nopass, int passoff, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_passbwd_kernel", Mtl_nh_passbwd_kernel{puf_mtl::dptr(inv_grad), puf_mtl::dptr(grad_concat), puf_mtl::dptr(sfeat), puf_mtl::dptr(vid), nopass, passoff, B});
}

struct Mtl_nh_blstats_kernel {
    uint64_t out;
    uint64_t obs;
    int B;
};
static_assert(sizeof(Mtl_nh_blstats_kernel) == 24, "Mtl_nh_blstats_kernel layout");
PUF_ENV_ADAPTER(nh_blstats_kernel, precision_t* out, const precision_t* obs, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_blstats_kernel", Mtl_nh_blstats_kernel{puf_mtl::dptr(out), puf_mtl::dptr(obs), B});
}

struct Mtl_nh_msg_ids_kernel {
    uint64_t ids;
    uint64_t obs;
    int B;
};
static_assert(sizeof(Mtl_nh_msg_ids_kernel) == 24, "Mtl_nh_msg_ids_kernel layout");
PUF_ENV_ADAPTER(nh_msg_ids_kernel, float* ids, const precision_t* obs, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_msg_ids_kernel", Mtl_nh_msg_ids_kernel{puf_mtl::dptr(ids), puf_mtl::dptr(obs), B});
}

struct Mtl_nh_msg_pool_kernel {
    uint64_t out;
    uint64_t msg_w;
    uint64_t ids;
    int B;
};
static_assert(sizeof(Mtl_nh_msg_pool_kernel) == 32, "Mtl_nh_msg_pool_kernel layout");
PUF_ENV_ADAPTER(nh_msg_pool_kernel, precision_t* out, const precision_t* msg_w, const float* ids, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_msg_pool_kernel", Mtl_nh_msg_pool_kernel{puf_mtl::dptr(out), puf_mtl::dptr(msg_w), puf_mtl::dptr(ids), B});
}

struct Mtl_nh_msg_bwd_kernel {
    uint64_t dmsg_acc;
    uint64_t grad_concat;
    uint64_t ids;
    int B;
};
static_assert(sizeof(Mtl_nh_msg_bwd_kernel) == 32, "Mtl_nh_msg_bwd_kernel layout");
PUF_ENV_ADAPTER(nh_msg_bwd_kernel, long long* dmsg_acc, const precision_t* grad_concat, const float* ids, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_msg_bwd_kernel", Mtl_nh_msg_bwd_kernel{puf_mtl::dptr(dmsg_acc), puf_mtl::dptr(grad_concat), puf_mtl::dptr(ids), B});
}

struct Mtl_nh_spkey_kernel {
    uint64_t keys;
    uint64_t sp_in;
    uint64_t sp_idx;
    uint64_t spk_w;
    uint64_t e_eff;
    uint64_t obs;
    int B;
};
static_assert(sizeof(Mtl_nh_spkey_kernel) == 56, "Mtl_nh_spkey_kernel layout");
PUF_ENV_ADAPTER(nh_spkey_kernel, precision_t* keys, precision_t* sp_in, float* sp_idx, const precision_t* spk_w, const precision_t* e_eff, const precision_t* obs, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_spkey_kernel", Mtl_nh_spkey_kernel{puf_mtl::dptr(keys), puf_mtl::dptr(sp_in), puf_mtl::dptr(sp_idx), puf_mtl::dptr(spk_w), puf_mtl::dptr(e_eff), puf_mtl::dptr(obs), B});
}

struct Mtl_nh_sp_doorstep_kernel {
    uint64_t concat;
    uint64_t obs;
    int B;
};
static_assert(sizeof(Mtl_nh_sp_doorstep_kernel) == 24, "Mtl_nh_sp_doorstep_kernel layout");
PUF_ENV_ADAPTER(nh_sp_doorstep_kernel, precision_t* concat, const precision_t* obs, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_sp_doorstep_kernel", Mtl_nh_sp_doorstep_kernel{puf_mtl::dptr(concat), puf_mtl::dptr(obs), B});
}

struct Mtl_nh_sp2_dk_kernel {
    uint64_t dkeys;
    uint64_t ptr_dkeys;
    uint64_t keys;
    int B;
};
static_assert(sizeof(Mtl_nh_sp2_dk_kernel) == 32, "Mtl_nh_sp2_dk_kernel layout");
PUF_ENV_ADAPTER(nh_sp2_dk_kernel, precision_t* dkeys, const precision_t* ptr_dkeys, const precision_t* keys, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_sp2_dk_kernel", Mtl_nh_sp2_dk_kernel{puf_mtl::dptr(dkeys), puf_mtl::dptr(ptr_dkeys), puf_mtl::dptr(keys), B});
}

struct Mtl_nh_spkey_dE_kernel {
    uint64_t dE_i;
    uint64_t dkeys;
    uint64_t spk_w;
    uint64_t sp_idx;
    int B;
};
static_assert(sizeof(Mtl_nh_spkey_dE_kernel) == 40, "Mtl_nh_spkey_dE_kernel layout");
PUF_ENV_ADAPTER(nh_spkey_dE_kernel, long long* dE_i, const precision_t* dkeys, const precision_t* spk_w, const float* sp_idx, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_spkey_dE_kernel", Mtl_nh_spkey_dE_kernel{puf_mtl::dptr(dE_i), puf_mtl::dptr(dkeys), puf_mtl::dptr(spk_w), puf_mtl::dptr(sp_idx), B});
}

struct Mtl_nh_concat_kernel {
    uint64_t out;
    uint64_t loc;
    uint64_t glb;
    uint64_t inv;
    uint64_t bl_out;
    uint64_t bl_feats;
    uint64_t msg;
    int B;
};
static_assert(sizeof(Mtl_nh_concat_kernel) == 64, "Mtl_nh_concat_kernel layout");
PUF_ENV_ADAPTER(nh_concat_kernel, precision_t* out, const precision_t* loc, const precision_t* glb, const precision_t* inv, const precision_t* bl_out, const precision_t* bl_feats, const precision_t* msg, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_concat_kernel", Mtl_nh_concat_kernel{puf_mtl::dptr(out), puf_mtl::dptr(loc), puf_mtl::dptr(glb), puf_mtl::dptr(inv), puf_mtl::dptr(bl_out), puf_mtl::dptr(bl_feats), puf_mtl::dptr(msg), B});
}

struct Mtl_nh_slice_kernel {
    uint64_t dst;
    uint64_t src;
    int B;
    int stride;
    int offset;
    int n;
};
static_assert(sizeof(Mtl_nh_slice_kernel) == 32, "Mtl_nh_slice_kernel layout");
PUF_ENV_ADAPTER(nh_slice_kernel, precision_t* dst, const precision_t* src, int B, int stride, int offset, int n) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_slice_kernel", Mtl_nh_slice_kernel{puf_mtl::dptr(dst), puf_mtl::dptr(src), B, stride, offset, n});
}

struct Mtl_nh_inv_decode_kernel {
    uint64_t idx;
    uint64_t obs;
    int B;
    int off;
};
static_assert(sizeof(Mtl_nh_inv_decode_kernel) == 24, "Mtl_nh_inv_decode_kernel layout");
PUF_ENV_ADAPTER(nh_inv_decode_kernel, float* idx, const precision_t* obs, int B, int off) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_inv_decode_kernel", Mtl_nh_inv_decode_kernel{puf_mtl::dptr(idx), puf_mtl::dptr(obs), B, off});
}

struct Mtl_nh_inv_table_kernel {
    uint64_t T;
    uint64_t E;
    uint64_t w1;
};
static_assert(sizeof(Mtl_nh_inv_table_kernel) == 24, "Mtl_nh_inv_table_kernel layout");
PUF_ENV_ADAPTER(nh_inv_table_kernel, precision_t* T, const precision_t* E, const precision_t* w1) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_inv_table_kernel", Mtl_nh_inv_table_kernel{puf_mtl::dptr(T), puf_mtl::dptr(E), puf_mtl::dptr(w1)});
}

struct Mtl_nh_inv_sfeat_kernel {
    uint64_t out;
    uint64_t obs;
    int B;
};
static_assert(sizeof(Mtl_nh_inv_sfeat_kernel) == 24, "Mtl_nh_inv_sfeat_kernel layout");
PUF_ENV_ADAPTER(nh_inv_sfeat_kernel, precision_t* out, const precision_t* obs, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_inv_sfeat_kernel", Mtl_nh_inv_sfeat_kernel{puf_mtl::dptr(out), puf_mtl::dptr(obs), B});
}

struct Mtl_nh_inv_gather_kernel {
    uint64_t out;
    uint64_t T;
    uint64_t b1;
    uint64_t ws;
    uint64_t sfeat;
    uint64_t idx;
    uint64_t Tt;
    uint64_t idxt;
    int B;
};
static_assert(sizeof(Mtl_nh_inv_gather_kernel) == 72, "Mtl_nh_inv_gather_kernel layout");
PUF_ENV_ADAPTER(nh_inv_gather_kernel, precision_t* out, const precision_t* T, const precision_t* b1, const precision_t* ws, const precision_t* sfeat, const float* idx, const precision_t* Tt, const float* idxt, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_inv_gather_kernel", Mtl_nh_inv_gather_kernel{puf_mtl::dptr(out), puf_mtl::dptr(T), puf_mtl::dptr(b1), puf_mtl::dptr(ws), puf_mtl::dptr(sfeat), puf_mtl::dptr(idx), puf_mtl::dptr(Tt), puf_mtl::dptr(idxt), B});
}

struct Mtl_nh_dTinv_scatter_kernel {
    uint64_t dT;
    uint64_t dflat;
    uint64_t idx;
    int64_t n;
    int skip_pad;
};
static_assert(sizeof(Mtl_nh_dTinv_scatter_kernel) == 40, "Mtl_nh_dTinv_scatter_kernel layout");
PUF_ENV_ADAPTER(nh_dTinv_scatter_kernel, long long* dT, const precision_t* dflat, const float* idx, int64_t n, int skip_pad) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_dTinv_scatter_kernel", Mtl_nh_dTinv_scatter_kernel{puf_mtl::dptr(dT), puf_mtl::dptr(dflat), puf_mtl::dptr(idx), n, skip_pad});
}

struct Mtl_nh_add_inplace_kernel {
    uint64_t dst;
    uint64_t src;
    int n;
};
static_assert(sizeof(Mtl_nh_add_inplace_kernel) == 24, "Mtl_nh_add_inplace_kernel layout");
PUF_ENV_ADAPTER(nh_add_inplace_kernel, precision_t* dst, const precision_t* src, int n) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_add_inplace_kernel", Mtl_nh_add_inplace_kernel{puf_mtl::dptr(dst), puf_mtl::dptr(src), n});
}

struct Mtl_nh_relu_bias_bwd_kernel {
    uint64_t grad;
    uint64_t out;
    uint64_t bias_acc;
    int64_t total;
    int dim;
};
static_assert(sizeof(Mtl_nh_relu_bias_bwd_kernel) == 40, "Mtl_nh_relu_bias_bwd_kernel layout");
PUF_ENV_ADAPTER(nh_relu_bias_bwd_kernel, precision_t* grad, const precision_t* out, long long* bias_acc, int64_t total, int dim) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_relu_bias_bwd_kernel", Mtl_nh_relu_bias_bwd_kernel{puf_mtl::dptr(grad), puf_mtl::dptr(out), puf_mtl::dptr(bias_acc), total, dim});
}

struct Mtl_nh_bias_bwd_kernel {
    uint64_t grad;
    uint64_t out;
    uint64_t bias_acc;
    int64_t total;
    int dim;
};
static_assert(sizeof(Mtl_nh_bias_bwd_kernel) == 40, "Mtl_nh_bias_bwd_kernel layout");
PUF_ENV_ADAPTER(nh_bias_bwd_kernel, precision_t* grad, const precision_t* out, long long* bias_acc, int64_t total, int dim) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_bias_bwd_kernel", Mtl_nh_bias_bwd_kernel{puf_mtl::dptr(grad), puf_mtl::dptr(out), puf_mtl::dptr(bias_acc), total, dim});
}

struct Mtl_nh_bias_flush_kernel {
    uint64_t acc;
    uint64_t d0;
    int n0;
    uint64_t d1;
    int n1;
    uint64_t d2;
    int n2;
    uint64_t d3;
    int n3;
    uint64_t d4;
    int n4;
    uint64_t d5;
    int n5;
    uint64_t d6;
    int n6;
};
static_assert(sizeof(Mtl_nh_bias_flush_kernel) == 120, "Mtl_nh_bias_flush_kernel layout");
PUF_ENV_ADAPTER(nh_bias_flush_kernel, const long long* acc, precision_t* d0, int n0, precision_t* d1, int n1, precision_t* d2, int n2, precision_t* d3, int n3, precision_t* d4, int n4, precision_t* d5, int n5, precision_t* d6, int n6) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_bias_flush_kernel", Mtl_nh_bias_flush_kernel{puf_mtl::dptr(acc), puf_mtl::dptr(d0), n0, puf_mtl::dptr(d1), n1, puf_mtl::dptr(d2), n2, puf_mtl::dptr(d3), n3, puf_mtl::dptr(d4), n4, puf_mtl::dptr(d5), n5, puf_mtl::dptr(d6), n6});
}

struct Mtl_nh_idemb_kernel {
    uint64_t concat;
    uint64_t idx_out;
    uint64_t obs;
    uint64_t role_w;
    uint64_t race_w;
    uint64_t gend_w;
    uint64_t algn_w;
    int B;
};
static_assert(sizeof(Mtl_nh_idemb_kernel) == 64, "Mtl_nh_idemb_kernel layout");
PUF_ENV_ADAPTER(nh_idemb_kernel, precision_t* concat, float* idx_out, const precision_t* obs, const precision_t* role_w, const precision_t* race_w, const precision_t* gend_w, const precision_t* algn_w, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_idemb_kernel", Mtl_nh_idemb_kernel{puf_mtl::dptr(concat), puf_mtl::dptr(idx_out), puf_mtl::dptr(obs), puf_mtl::dptr(role_w), puf_mtl::dptr(race_w), puf_mtl::dptr(gend_w), puf_mtl::dptr(algn_w), B});
}

struct Mtl_nh_idemb_grad_kernel {
    uint64_t role_g;
    uint64_t race_g;
    uint64_t gend_g;
    uint64_t algn_g;
    uint64_t grad_concat;
    uint64_t idx;
    int B;
};
static_assert(sizeof(Mtl_nh_idemb_grad_kernel) == 56, "Mtl_nh_idemb_grad_kernel layout");
PUF_ENV_ADAPTER(nh_idemb_grad_kernel, precision_t* role_g, precision_t* race_g, precision_t* gend_g, precision_t* algn_g, const precision_t* grad_concat, const float* idx, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_idemb_grad_kernel", Mtl_nh_idemb_grad_kernel{puf_mtl::dptr(role_g), puf_mtl::dptr(race_g), puf_mtl::dptr(gend_g), puf_mtl::dptr(algn_g), puf_mtl::dptr(grad_concat), puf_mtl::dptr(idx), B});
}

struct Mtl_nh_dec_assemble_kernel {
    uint64_t out;
    uint64_t tmp;
    uint64_t slot_logits;
    uint64_t q;
    uint64_t spkeys;
    int B;
};
static_assert(sizeof(Mtl_nh_dec_assemble_kernel) == 48, "Mtl_nh_dec_assemble_kernel layout");
PUF_ENV_ADAPTER(nh_dec_assemble_kernel, precision_t* out, const precision_t* tmp, const precision_t* slot_logits, const precision_t* q, const precision_t* spkeys, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_dec_assemble_kernel", Mtl_nh_dec_assemble_kernel{puf_mtl::dptr(out), puf_mtl::dptr(tmp), puf_mtl::dptr(slot_logits), puf_mtl::dptr(q), puf_mtl::dptr(spkeys), B});
}

struct Mtl_nh_ptr_rownorm_kernel {
    uint64_t n;
    uint64_t rows;
    int total;
};
static_assert(sizeof(Mtl_nh_ptr_rownorm_kernel) == 24, "Mtl_nh_ptr_rownorm_kernel layout");
PUF_ENV_ADAPTER(nh_ptr_rownorm_kernel, precision_t* n, const precision_t* rows, int total) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_ptr_rownorm_kernel", Mtl_nh_ptr_rownorm_kernel{puf_mtl::dptr(n), puf_mtl::dptr(rows), total});
}

struct Mtl_nh_ptr3_cos_kernel {
    uint64_t slot_logits;
    uint64_t q;
    uint64_t qn;
    uint64_t kmat;
    uint64_t tau;
    int B;
};
static_assert(sizeof(Mtl_nh_ptr3_cos_kernel) == 48, "Mtl_nh_ptr3_cos_kernel layout");
PUF_ENV_ADAPTER(nh_ptr3_cos_kernel, precision_t* slot_logits, const precision_t* q, const precision_t* qn, const precision_t* kmat, const precision_t* tau, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_ptr3_cos_kernel", Mtl_nh_ptr3_cos_kernel{puf_mtl::dptr(slot_logits), puf_mtl::dptr(q), puf_mtl::dptr(qn), puf_mtl::dptr(kmat), puf_mtl::dptr(tau), B});
}

struct Mtl_nh_dec_dtmp_kernel {
    uint64_t dtmp;
    uint64_t g;
    int B;
};
static_assert(sizeof(Mtl_nh_dec_dtmp_kernel) == 24, "Mtl_nh_dec_dtmp_kernel layout");
PUF_ENV_ADAPTER(nh_dec_dtmp_kernel, precision_t* dtmp, const precision_t* g, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_dec_dtmp_kernel", Mtl_nh_dec_dtmp_kernel{puf_mtl::dptr(dtmp), puf_mtl::dptr(g), B});
}

struct Mtl_nh_ptr3_dq_kernel {
    uint64_t dq;
    uint64_t tau_acc;
    uint64_t g;
    uint64_t out;
    uint64_t q;
    uint64_t qn;
    uint64_t kmat;
    uint64_t tau;
    int B;
};
static_assert(sizeof(Mtl_nh_ptr3_dq_kernel) == 72, "Mtl_nh_ptr3_dq_kernel layout");
PUF_ENV_ADAPTER(nh_ptr3_dq_kernel, precision_t* dq, long long* tau_acc, const precision_t* g, const precision_t* out, const precision_t* q, const precision_t* qn, const precision_t* kmat, const precision_t* tau, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_ptr3_dq_kernel", Mtl_nh_ptr3_dq_kernel{puf_mtl::dptr(dq), puf_mtl::dptr(tau_acc), puf_mtl::dptr(g), puf_mtl::dptr(out), puf_mtl::dptr(q), puf_mtl::dptr(qn), puf_mtl::dptr(kmat), puf_mtl::dptr(tau), B});
}

struct Mtl_nh_spq_bwd_kernel {
    uint64_t dq;
    uint64_t spdk;
    uint64_t g;
    uint64_t q;
    uint64_t spkeys;
    int B;
};
static_assert(sizeof(Mtl_nh_spq_bwd_kernel) == 48, "Mtl_nh_spq_bwd_kernel layout");
PUF_ENV_ADAPTER(nh_spq_bwd_kernel, precision_t* dq, precision_t* spdk, const precision_t* g, const precision_t* q, const precision_t* spkeys, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_spq_bwd_kernel", Mtl_nh_spq_bwd_kernel{puf_mtl::dptr(dq), puf_mtl::dptr(spdk), puf_mtl::dptr(g), puf_mtl::dptr(q), puf_mtl::dptr(spkeys), B});
}

struct Mtl_nh_ptr3_dkmat_kernel {
    uint64_t dkmat;
    uint64_t g;
    uint64_t out;
    uint64_t q;
    uint64_t qn;
    uint64_t kmat;
    uint64_t tau;
    int B;
};
static_assert(sizeof(Mtl_nh_ptr3_dkmat_kernel) == 64, "Mtl_nh_ptr3_dkmat_kernel layout");
PUF_ENV_ADAPTER(nh_ptr3_dkmat_kernel, precision_t* dkmat, const precision_t* g, const precision_t* out, const precision_t* q, const precision_t* qn, const precision_t* kmat, const precision_t* tau, int B) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_ptr3_dkmat_kernel", Mtl_nh_ptr3_dkmat_kernel{puf_mtl::dptr(dkmat), puf_mtl::dptr(g), puf_mtl::dptr(out), puf_mtl::dptr(q), puf_mtl::dptr(qn), puf_mtl::dptr(kmat), puf_mtl::dptr(tau), B});
}

struct Mtl_nh_fill_kernel {
    uint64_t p;
    float v;
    int n;
};
static_assert(sizeof(Mtl_nh_fill_kernel) == 16, "Mtl_nh_fill_kernel layout");
PUF_ENV_ADAPTER(nh_fill_kernel, precision_t* p, float v, int n) {
    auto c = cfg; c.smem = 0;
    puf_mtl::launch(c, "nh_fill_kernel", Mtl_nh_fill_kernel{puf_mtl::dptr(p), v, n});
}

struct MtlNhSampleLogitsArgs {
    uint64_t logits, logstd, act_sizes, actions, env_actions, logprobs, value_out,
        rng_states, action_mask;
    int32_t B, fused_cols, num_atns, mask_stride, is_continuous;
    uint64_t hc, veps;
    int32_t hc_stride;
};
static_assert(sizeof(MtlNhSampleLogitsArgs) == 120, "SampleLogitsArgs layout");
PUF_ENV_ADAPTER(sample_logits, Prec dec_out, Prec logstd, int* act_sizes, float* actions,
        float* env_actions, precision_t* logprobs, precision_t* value_out,
        curandStatePhilox4_32_10_t* rng_states, precision_t* action_mask, int mask_stride) {
    bool is_continuous = logstd.data != NULL && numel(logstd.shape) > 0;
    puf_mtl::launch(cfg, "nh_sample_logits", MtlNhSampleLogitsArgs{
        puf_mtl::dptr(dec_out.data), is_continuous ? puf_mtl::dptr(logstd.data) : 0,
        puf_mtl::dptr(act_sizes), puf_mtl::dptr(actions), puf_mtl::dptr(env_actions),
        puf_mtl::dptr(logprobs), puf_mtl::dptr(value_out), puf_mtl::dptr(rng_states),
        puf_mtl::dptr(action_mask), (int32_t)dec_out.shape[0], (int32_t)dec_out.shape[1],
        NUM_ATNS, mask_stride, is_continuous ? 1 : 0, puf_mtl::dptr(d_hc_dev), puf_mtl::dptr(d_veps_dev), d_hc_stride});
}
struct MtlNhCacheImpArgs {
    uint64_t logits, actions, old_logprobs, mask, logstd, act_sizes, imp, values, logps, new_lp;
    int32_t NT, fused_cols, num_atns;
    uint64_t hc, veps;
    int32_t hc_stride;
};
static_assert(sizeof(MtlNhCacheImpArgs) == 120, "CacheImpArgs layout");
PUF_ENV_ADAPTER(cache_imp_and_v, Prec dec_out, const float* actions,
        const precision_t* old_logprobs, const precision_t* action_mask, Prec logstd,
        const int* act_sizes, precision_t* imp_out, precision_t* value_out,
        float* logps, float* new_lp_out) {
    puf_mtl::launch(cfg, "nh_cache_imp_and_v", MtlNhCacheImpArgs{puf_mtl::dptr(dec_out.data),
        puf_mtl::dptr(actions), puf_mtl::dptr(old_logprobs), puf_mtl::dptr(action_mask),
        puf_mtl::dptr(logstd.data), puf_mtl::dptr(act_sizes), puf_mtl::dptr(imp_out),
        puf_mtl::dptr(value_out), puf_mtl::dptr(logps), puf_mtl::dptr(new_lp_out),
        (int32_t)(dec_out.shape[0] * dec_out.shape[1]), (int32_t)dec_out.shape[2], NUM_ATNS,
        puf_mtl::dptr(d_hc_dev), puf_mtl::dptr(d_veps_dev), d_hc_stride});
}

struct MtlNhPPOLossArgs {
    uint64_t partials, grad_logits, grad_logstd, grad_values, logits, logstd,
        values_pred, act_sizes, ent_coef, actions, old_logprobs, advantages, values, returns;
    int32_t num_atns;
    float clip_coef, vf_clip_coef, vf_coef;
    int32_t T_seq, A_total, N, is_continuous;
    uint64_t hc, veps;
    int32_t hc_stride;
    uint64_t mask;
};
static_assert(sizeof(MtlNhPPOLossArgs) == 176, "PPOLossArgs layout");
// Templates keep this header usable by the existing minimal-type Metal tests.
// Production instantiates these with upstream PPOKernelArgs/PPOGraphArgs.
template <typename A, typename G>
PUF_ENV_ADAPTER(ppo_loss_compute, float* partials, A a, G g) {
    puf_mtl::launch(cfg, "nh_ppo_loss_compute", MtlNhPPOLossArgs{puf_mtl::dptr(partials),
        puf_mtl::dptr(a.grad_logits), puf_mtl::dptr(a.grad_logstd),
        puf_mtl::dptr(a.grad_values_pred), puf_mtl::dptr(a.logits), puf_mtl::dptr(a.logstd),
        puf_mtl::dptr(a.values_pred), puf_mtl::dptr(a.act_sizes), puf_mtl::dptr(a.ent_coef),
        puf_mtl::dptr(g.actions), puf_mtl::dptr(g.old_logprobs), puf_mtl::dptr(g.advantages),
        puf_mtl::dptr(g.values), puf_mtl::dptr(g.returns), a.num_atns, a.clip_coef,
        a.vf_clip_coef, a.vf_coef, a.T_seq, a.A_total, a.N, a.is_continuous ? 1 : 0,
        puf_mtl::dptr(d_hc_dev), puf_mtl::dptr(d_veps_dev), d_hc_stride, puf_mtl::dptr(a.action_mask)});
}
#endif  // PUFFER_NETHACK

// Custom ocean encoder adapters. minimal/asteroids: CPU-reference validated.
// Included after upstream types by kernels.h. MSL: shaders/envs_entity.metal.
#pragma once

struct MtlEntityArgs {
    uint64_t dst, argmax, input, weight, grad;
    int32_t B, H, obs_size, self_dim, point_dim, num_points;
};
static_assert(sizeof(MtlEntityArgs) == 64, "EntityArgs layout");

static inline void puf_mtl_entity_launch(const puf_mtl::LaunchCfg& cfg,
        const char* name, MtlEntityArgs args, int n) {
    // Independent output ownership replaces CUDA's shared-memory reductions.
    auto c = cfg;
    c.grid = dim3((n + 255) / 256);
    c.block = dim3(256);
    c.smem = 0;
    puf_mtl::launch(c, name, args);
}

#define PUF_ENTITY_ADAPTERS(prefix, S, F, P) \
static inline void puf_mtl_kernel(puf_kid_##prefix##_materialize_points_kernel*, \
        const puf_mtl::LaunchCfg& c, precision_t* dst, const precision_t* obs, int B, int O) { \
    puf_mtl_entity_launch(c, "entity_materialize", {puf_mtl::dptr(dst), 0, \
        puf_mtl::dptr(obs), 0, 0, B, 0, O, S, F, P}, B * P * (S + F)); \
} \
static inline void puf_mtl_kernel(puf_kid_##prefix##_linear_max_kernel*, \
        const puf_mtl::LaunchCfg& c, precision_t* dst, int* argmax, \
        const precision_t* input, const precision_t* weight, int B, int H) { \
    puf_mtl_entity_launch(c, "entity_linear_max", {puf_mtl::dptr(dst), puf_mtl::dptr(argmax), \
        puf_mtl::dptr(input), puf_mtl::dptr(weight), 0, B, H, 0, S, F, P}, B * H); \
} \
static inline void puf_mtl_kernel(puf_kid_##prefix##_output_wgrad_kernel*, \
        const puf_mtl::LaunchCfg& c, precision_t* dst, const precision_t* grad, \
        const precision_t* input, const int* argmax, int B, int H) { \
    puf_mtl_entity_launch(c, "entity_output_wgrad", {puf_mtl::dptr(dst), puf_mtl::dptr(argmax), \
        puf_mtl::dptr(input), 0, puf_mtl::dptr(grad), B, H, 0, S, F, P}, H * 16); \
} \
static inline void puf_mtl_kernel(puf_kid_##prefix##_grad_entity_kernel*, \
        const puf_mtl::LaunchCfg& c, precision_t* dst, const precision_t* grad, \
        const precision_t* weight, const precision_t* input, const int* argmax, int B, int H) { \
    puf_mtl_entity_launch(c, "entity_grad_input", {puf_mtl::dptr(dst), puf_mtl::dptr(argmax), \
        puf_mtl::dptr(input), puf_mtl::dptr(weight), puf_mtl::dptr(grad), B, H, 0, S, F, P}, B * P * 16); \
}

PUF_ENTITY_ADAPTERS(me, 2, 4, 16)
PUF_ENTITY_ADAPTERS(ae, 4, 5, 26)
#undef PUF_ENTITY_ADAPTERS

#ifdef PUFFER_MINIMAL
static_assert(ME_SELF_DIM == 2 && ME_POINT_DIM == 4 && ME_NUM_POINTS == 16 &&
              ME_ENTITY_HIDDEN == 16, "minimal shader dimensions");
#endif
#ifdef PUFFER_ASTEROIDS
static_assert(AE_SELF_DIM == 4 && AE_POINT_DIM == 5 && AE_NUM_POINTS == 26 &&
              AE_ENTITY_HIDDEN == 16, "asteroids shader dimensions");
#endif

// kernels.h undefines PUF_ADAPTER before including this file.
#define PUF_ENV_ADAPTER(kname, ...) static inline void puf_mtl_kernel( \
    puf_kid_##kname*, const puf_mtl::LaunchCfg& cfg, __VA_ARGS__)

// ---- OSRS entity encoder (shaders/envs_osrs.metal). CUDA-golden validated. ----
#ifdef PUFFER_OSRS_ENTITY_NET
struct MtlOsrsGatherArgs {
    uint64_t flat, obs, table;
    int32_t B, obs_size, obs_start, num_records, obs_features, type_onehot,
        type_code_scale, expansion, table_rows, table_cols;
};
static_assert(sizeof(MtlOsrsGatherArgs) == 64, "OsrsGatherArgs layout");
struct MtlOsrsPoolArgs {
    uint64_t out, argmax, h1, z1, record_flat, l2_w;
    int32_t B, H, num_records, record_features, active_width;
};
struct MtlOsrsBwdArgs {
    uint64_t dst, grad, src, z1, argmax;
    int32_t B, H, num_records;
};
static constexpr int PUF_MTL_OSRS_MAX_RECORDS = 32;  // envs_osrs.metal tiles

// CUDA keeps the item table in __device__ memory; Metal uploads the same host
// table (src/ocean.cu) into one GPU buffer on first use. Rollout buffer threads
// can reach this concurrently, so allocation and upload happen inside a
// thread-safe static initializer: no caller sees the pointer before the copy.
static uint64_t puf_mtl_osrs_item_table() {
    static float* const table = [] {
        float* t = nullptr;
        if (cudaMalloc(&t, sizeof(OSRS_ITEM_OBS_TABLE_DEV)) != cudaSuccess ||
                cudaMemcpy(t, OSRS_ITEM_OBS_TABLE_DEV, sizeof(OSRS_ITEM_OBS_TABLE_DEV),
                    cudaMemcpyHostToDevice) != cudaSuccess) {
            fprintf(stderr, "metal: OSRS item table upload failed\n");
            abort();
        }
        return t;
    }();
    return puf_mtl::dptr(table);
}

PUF_ENV_ADAPTER(osrs_entity_gather_branch, precision_t* flat, const precision_t* obs,
        int B, int obs_size, int obs_start, int num_records, int obs_features,
        int type_onehot, int type_code_scale, OsrsEntityBranchExpansion expansion) {
    MtlOsrsGatherArgs args = {puf_mtl::dptr(flat), puf_mtl::dptr(obs),
        puf_mtl_osrs_item_table(), B, obs_size, obs_start, num_records, obs_features,
        type_onehot, type_code_scale, (int32_t)expansion,
        OSRS_ITEM_OBS_TABLE_ROWS, OSRS_ITEM_OBS_TABLE_COLS};
    static_assert(OSRS_ENTITY_BRANCH_ITEM_TABLE == 1, "shader expansion code");
    puf_mtl::launch(cfg, "osrs_entity_gather_branch", args);
}

PUF_ENV_ADAPTER(osrs_entity_fused_pool_fwd, precision_t* out, int* argmax, precision_t* h1,
        const precision_t* z1, const precision_t* record_flat, const precision_t* l2_w,
        int B, int H, int num_records, int record_features, int active_width) {
    if (num_records > PUF_MTL_OSRS_MAX_RECORDS) {
        fprintf(stderr, "metal: OSRS pool supports <= %d records (got %d)\n",
            PUF_MTL_OSRS_MAX_RECORDS, num_records);
        abort();
    }
    auto c = cfg;
    c.smem = 0;  // static threadgroup tiles
    puf_mtl::launch(c, "osrs_entity_fused_pool_fwd", MtlOsrsPoolArgs{puf_mtl::dptr(out),
        puf_mtl::dptr(argmax), puf_mtl::dptr(h1), puf_mtl::dptr(z1),
        puf_mtl::dptr(record_flat), puf_mtl::dptr(l2_w),
        B, H, num_records, record_features, active_width});
}

PUF_ENV_ADAPTER(osrs_entity_fused_l2_wgrad, precision_t* wgrad, const precision_t* grad,
        const precision_t* h1, const int* argmax, int B, int H, int num_records) {
    auto c = cfg;
    c.block = dim3(256);
    c.smem = 0;
    puf_mtl::launch(c, "osrs_entity_fused_l2_wgrad", MtlOsrsBwdArgs{puf_mtl::dptr(wgrad),
        puf_mtl::dptr(grad), puf_mtl::dptr(h1), 0, puf_mtl::dptr(argmax), B, H, num_records});
}

PUF_ENV_ADAPTER(osrs_entity_fused_grad_z1, precision_t* grad_z1, const precision_t* grad,
        const precision_t* l2_w, const precision_t* z1, const int* argmax,
        int B, int H, int num_records) {
    if (num_records > PUF_MTL_OSRS_MAX_RECORDS) {
        fprintf(stderr, "metal: OSRS grad supports <= %d records (got %d)\n",
            PUF_MTL_OSRS_MAX_RECORDS, num_records);
        abort();
    }
    auto c = cfg;
    c.block = dim3(256);  // shader tiles hidden units 256 at a time
    c.smem = 0;
    puf_mtl::launch(c, "osrs_entity_fused_grad_z1", MtlOsrsBwdArgs{puf_mtl::dptr(grad_z1),
        puf_mtl::dptr(grad), puf_mtl::dptr(l2_w), puf_mtl::dptr(z1), puf_mtl::dptr(argmax),
        B, H, num_records});
}
#endif  // PUFFER_OSRS_ENTITY_NET

// ---- Craftax category-embedding encoder (shaders/envs_craftax.metal). ----
#ifdef PUFFER_CRAFTAX
struct MtlCraftaxArgs { uint64_t dst, src, weight; int32_t B, obs_size, n; };
static_assert(CX_NUM_CELLS == 99 && CX_CHANNELS == 8 && CX_NUM_SCALARS == 51 &&
    CX_EMB_DIM == 16 && CX_VOCAB_TOTAL == 154, "craftax shader dimensions");
static_assert(sizeof(long) == 8, "craftax int64 accumulator");

PUF_ENV_ADAPTER(craftax_embed_bag_kernel, precision_t* concat, const precision_t* obs,
        const precision_t* embed_w, int B, int obs_size) {
    puf_mtl::launch(cfg, "craftax_embed_bag", MtlCraftaxArgs{puf_mtl::dptr(concat),
        puf_mtl::dptr(obs), puf_mtl::dptr(embed_w), B, obs_size, 0});
}
PUF_ENV_ADAPTER(craftax_copy_scalars_kernel, precision_t* concat, const precision_t* obs,
        int B, int obs_size) {
    puf_mtl::launch(cfg, "craftax_copy_scalars", MtlCraftaxArgs{puf_mtl::dptr(concat),
        puf_mtl::dptr(obs), 0, B, obs_size, 0});
}
PUF_ENV_ADAPTER(craftax_embed_bag_backward_kernel, long long* embed_wgrad_i,
        const precision_t* grad_concat, const precision_t* obs, int B, int obs_size) {
    puf_mtl::launch(cfg, "craftax_embed_bag_backward", MtlCraftaxArgs{
        puf_mtl::dptr(embed_wgrad_i), puf_mtl::dptr(obs), puf_mtl::dptr(grad_concat),
        B, obs_size, 0});
}
PUF_ENV_ADAPTER(craftax_fxp_to_precision_kernel, precision_t* dst, const long long* src, int n) {
    puf_mtl::launch(cfg, "craftax_fxp_to_precision", MtlCraftaxArgs{puf_mtl::dptr(dst),
        puf_mtl::dptr(src), 0, 0, 0, n});
}
#endif  // PUFFER_CRAFTAX

// ---- NMMO3 conv encoder (shaders/envs_nmmo3.metal). ----
#ifdef PUFFER_NMMO3
struct MtlN3Args {
    uint64_t dst, src, src2, src3;
    int32_t B, obs_size, OC, spatial, flag, n;
};
static_assert(sizeof(MtlN3Args) == 56, "N3Args layout");
static_assert(N3_MAP_SIZE == 1650 && N3_PLAYER == 47 && N3_EMBED_DIM == 32 &&
    N3_EMBED_VOCAB == 128 && N3_C1_COL_W == 1475 && N3_C2_COL_W == 1152 &&
    N3_CONCAT == 1817, "nmmo3 shader dimensions");
static inline MtlN3Args puf_mtl_n3(const void* dst, const void* src, const void* src2,
        const void* src3, int B, int obs_size = 0, int OC = 0, int spatial = 0,
        int flag = 0, int n = 0) {
    return {puf_mtl::dptr(dst), puf_mtl::dptr(src), puf_mtl::dptr(src2), puf_mtl::dptr(src3),
        B, obs_size, OC, spatial, flag, n};
}
PUF_ENV_ADAPTER(n3_c1_im2col_obs, const precision_t* obs, precision_t* col, int B, int obs_size) {
    puf_mtl::launch(cfg, "n3_c1_im2col_obs", puf_mtl_n3(col, obs, 0, 0, B, obs_size));
}
PUF_ENV_ADAPTER(n3_c2_im2col, const precision_t* input, precision_t* col, int B) {
    puf_mtl::launch(cfg, "n3_c2_im2col", puf_mtl_n3(col, input, 0, 0, B));
}
PUF_ENV_ADAPTER(n3_c2_col2im, const precision_t* col, precision_t* grad_input, int B) {
    puf_mtl::launch(cfg, "n3_c2_col2im", puf_mtl_n3(grad_input, col, 0, 0, B));
}
PUF_ENV_ADAPTER(n3_rows_to_nchw, const precision_t* src, precision_t* dst,
        int B, int OC, int spatial, int relu) {
    puf_mtl::launch(cfg, "n3_rows_to_nchw", puf_mtl_n3(dst, src, 0, 0, B, 0, OC, spatial, relu));
}
PUF_ENV_ADAPTER(n3_nchw_to_rows, const precision_t* src, precision_t* dst,
        int B, int OC, int spatial) {
    puf_mtl::launch(cfg, "n3_nchw_to_rows", puf_mtl_n3(dst, src, 0, 0, B, 0, OC, spatial));
}
PUF_ENV_ADAPTER(n3_embedding_kernel, precision_t* out, const precision_t* obs,
        const precision_t* embed_w, int B, int obs_size) {
    puf_mtl::launch(cfg, "n3_embedding", puf_mtl_n3(out, obs, embed_w, 0, B, obs_size));
}
PUF_ENV_ADAPTER(n3_concat_kernel, precision_t* out, const precision_t* conv_flat,
        const precision_t* embed, const precision_t* obs, int B, int obs_size) {
    puf_mtl::launch(cfg, "n3_concat", puf_mtl_n3(out, conv_flat, embed, obs, B, obs_size));
}
PUF_ENV_ADAPTER(n3_relu_kernel, precision_t* data, int total) {
    puf_mtl::launch(cfg, "n3_relu", puf_mtl_n3(data, 0, 0, 0, 0, 0, 0, 0, 0, total));
}
PUF_ENV_ADAPTER(n3_relu_backward_kernel, precision_t* grad, const precision_t* out, int total) {
    puf_mtl::launch(cfg, "n3_relu_backward", puf_mtl_n3(grad, out, 0, 0, 0, 0, 0, 0, 0, total));
}
PUF_ENV_ADAPTER(n3_concat_backward_conv_kernel, precision_t* conv_grad,
        const precision_t* concat_grad, int B) {
    puf_mtl::launch(cfg, "n3_concat_backward_conv", puf_mtl_n3(conv_grad, concat_grad, 0, 0, B));
}
PUF_ENV_ADAPTER(n3_embedding_backward_kernel, long long* embed_wgrad_i,
        const precision_t* concat_grad, const precision_t* obs, int B, int obs_size) {
    puf_mtl::launch(cfg, "n3_embedding_backward",
        puf_mtl_n3(embed_wgrad_i, concat_grad, obs, 0, B, obs_size));
}
PUF_ENV_ADAPTER(n3_fxp_to_precision_kernel, precision_t* dst, const long long* src, int n) {
    puf_mtl::launch(cfg, "n3_fxp_to_precision", puf_mtl_n3(dst, src, 0, 0, 0, 0, 0, 0, 0, n));
}
#endif  // PUFFER_NMMO3

#include "kernels_nethack.h"

#undef PUF_ENV_ADAPTER

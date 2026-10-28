// Custom ocean encoder goldens (OSRS entity, craftax, nmmo3 conv, NetHack policy).
//
// One host-only harness shared verbatim by both backends:
//   CUDA (CUDA-reference, branch golden-envs): tests/metal_ref/env_golden.h, included
//     at the end of src/pufferl.cu when built with NVCC_EXTRA=-DPUF_ENV_GOLDEN.
//   Metal: tests/metal/env_golden.h, included by src/metal/pufferl_metal.cpp.
//
//   ./puffer generate OUT_ROOT          deterministic inputs -> OUT_ROOT/<case>
//   ./puffer replay GOLDEN_ROOT OUT_ROOT  reuse GOLDEN inputs -> OUT_ROOT/<case>
//
// Each case runs the real Encoder vtable: train/rollout forward and backward.
// Replay restores CUDA's saved forward state before backward, conditioning
// gradients on CUDA's pooling choices. NetHack also exercises its real Decoder
// vtable and conditional sampling/PPO. Actual decoder gradients are checked
// before restoring CUDA gradients at the encoder backward boundary. Tensors
// follow allocator registration order in the tests/metal_ref schema-1 layout.
#pragma once

#ifndef PUF_LAUNCH
#define PUF_LAUNCH(k, g, b, s, st) k<<<g, b, s, st>>>
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

#if defined(PUFFER_OSRS_COLOSSEUM)
#define GOLDEN_OSRS_DESCRIPTOR OSRS_COLOSSEUM_ENTITY_DESCRIPTOR
#elif defined(PUFFER_OSRS_INFERNO)
#define GOLDEN_OSRS_DESCRIPTOR OSRS_INFERNO_ENTITY_DESCRIPTOR
#else
#define GOLDEN_OSRS_DESCRIPTOR OSRS_EQUIPMENT_ENTITY_DESCRIPTOR
#endif

namespace env_golden {

static constexpr const char* prec_dtype = sizeof(precision_t) == 2 ? "bf16" : "f32";
static void upload_precision(precision_t* dst, const std::vector<float>& src) {
    std::vector<precision_t> v(src.size());
    for (size_t i = 0; i < v.size(); i++) v[i] = from_float(src[i]);
    cudaMemcpy(dst, v.data(), v.size() * sizeof(precision_t), cudaMemcpyHostToDevice);
}
// Poison fresh allocations before their first call: a missing write must never
// inherit valid output from allocator reuse or from another test case. The
// 0x7fc1 halfword pattern is NaN as bf16 and as f32 (so it can't hide inside a
// tolerance) and a huge positive int32/int64 (never a valid index or -1).
static void poison_bytes(void* p, size_t bytes) {
    std::vector<uint16_t> v((bytes + 1) / 2, 0x7fc1);
    cudaMemcpy(p, v.data(), bytes, cudaMemcpyHostToDevice);
}
static void poison(Allocator& al) { poison_bytes(al.mem, al.total_bytes); }


static uint32_t hash(uint32_t i, uint32_t seed) {
    uint32_t x = i + 0x9e3779b9u * seed;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
// Uniform in [-scale, scale), 2^-15 resolution (tests/metal_ref pattern()).
static float pattern(uint32_t i, uint32_t seed, float scale) {
    return ((int)(hash(i, seed) & 65535) - 32768) * (scale / 32768.f);
}
static int randint(uint32_t i, uint32_t seed, int n) { return (int)(hash(i, seed) % (uint32_t)n); }

struct Writer {
    std::string dir;
    std::vector<std::string> entries, scalars;
    explicit Writer(const std::string& d) : dir(d) {
        std::string p;
        for (size_t i = 0; i <= d.size(); i++) {
            if (i == d.size() || d[i] == '/') { if (!p.empty()) mkdir(p.c_str(), 0755); }
            if (i < d.size()) p += d[i];
        }
    }
    void scalar(const char* k, double v) {
        char b[128]; snprintf(b, sizeof(b), "\"%s\":%.17g", k, v); scalars.push_back(b);
    }
    void note(const char* k, const std::string& v) {
        scalars.push_back(std::string("\"") + k + "\":\"" + v + "\"");
    }
    // Device or unified pointer; synchronizes first.
    void tensor(const std::string& name, const void* p, std::vector<int64_t> shape,
            const char* dtype, const char* role) {
        cudaDeviceSynchronize();
        int64_t n = 1; for (auto s : shape) n *= s;
        size_t bytes = n * (strcmp(dtype, "u64") == 0 ? 8 : strcmp(dtype, "bf16") == 0 ? 2 : 4);
        std::vector<char> h(bytes);
        cudaMemcpy(h.data(), p, bytes, cudaMemcpyDeviceToHost);
        std::string file = name + ".bin";
        FILE* f = fopen((dir + "/" + file).c_str(), "wb");
        if (!f || fwrite(h.data(), 1, bytes, f) != bytes) { perror(file.c_str()); exit(2); }
        fclose(f);
        std::string e = "\"" + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":[";
        for (size_t i = 0; i < shape.size(); i++) e += (i ? "," : "") + std::to_string(shape[i]);
        entries.push_back(e + "],\"file\":\"" + file + "\",\"role\":\"" + role + "\"}");
    }
    void finish() {
        FILE* f = fopen((dir + "/manifest.json").c_str(), "w");
        fprintf(f, "{\n\"schema_version\":1,\n\"byte_order\":\"little\",\n"
            "\"upstream_commit\":\"ea7f989371270d2948f96728bda8aec5e7634a4c\",\n"
            "\"precision\":\"%s\",\n\"tensors\":{\n",
            sizeof(precision_t) == 2 ? "BF16" : "PRECISION_FLOAT");
        for (size_t i = 0; i < entries.size(); i++) fprintf(f, "%s%s", i ? ",\n" : "", entries[i].c_str());
        fprintf(f, "\n},\n\"scalars\":{\n");
        for (size_t i = 0; i < scalars.size(); i++) fprintf(f, "%s%s", i ? ",\n" : "", scalars[i].c_str());
        fprintf(f, "\n}\n}\n");
        fclose(f);
        printf("wrote %s (%zu tensors)\n", dir.c_str(), entries.size());
    }
};

static void read_file(const std::string& path, void* dst, size_t bytes) {
    FILE* f = fopen(path.c_str(), "rb");
    std::vector<char> h(bytes + 1);
    size_t got = f ? fread(h.data(), 1, bytes + 1, f) : 0;
    if (!f || got != bytes) {
        fprintf(stderr, "env_golden: %s: expected %zu bytes, got %zu\n", path.c_str(), bytes, got);
        exit(2);
    }
    fclose(f);
    cudaMemcpy(dst, h.data(), bytes, cudaMemcpyHostToDevice);
}

static int64_t entry_count(const AllocEntry& e) { return numel(e.shape); }
static std::vector<int64_t> entry_shape(const AllocEntry& e) {
    std::vector<int64_t> s;
    for (int i = 0; i < 8 && e.shape[i]; i++) s.push_back(e.shape[i]);
    return s;
}

// Integer (Int) activation entries; Long entries are recognised by size.
static bool is_int_entry(void* activations, const AllocEntry& e) {
#ifdef PUFFER_OSRS_ENTITY_NET
    OsrsEntityEncoderActivations* a = (OsrsEntityEncoderActivations*)activations;
    if (e.data_ptr == (void**)&a->inv_pool_argmax.data) return true;
    OsrsEntityBranchActivations* br = osrs_entity_branch_activations(a);
    for (int i = 0; i < GOLDEN_OSRS_DESCRIPTOR.num_branches; i++)
        if (e.data_ptr == (void**)&br[i].pool_argmax.data) return true;
#endif
#ifdef PUFFER_MINIMAL
    if (e.data_ptr == (void**)&((MinimalEntityEncoderActivations*)activations)->argmax.data) return true;
#endif
#ifdef PUFFER_ASTEROIDS
    if (e.data_ptr == (void**)&((AsteroidsEntityEncoderActivations*)activations)->argmax.data) return true;
#endif
    (void)activations; (void)e;
    return false;
}
static const char* entry_dtype(void* act, const AllocEntry& e) {
    if (e.elem_size == 8) return "u64";
    return act && is_int_entry(act, e) ? "i32" : e.elem_size == 2 ? "bf16" : "f32";
}

static void poison_forward(void* act) {
#if defined(PUFFER_OSRS_ENTITY_NET)
    auto* a = (OsrsEntityEncoderActivations*)act;
#elif defined(PUFFER_CRAFTAX)
    auto* a = (CraftaxEncoderActivations*)act;
#elif defined(PUFFER_NMMO3)
    auto* a = (NMMO3EncoderActivations*)act;
#elif defined(PUFFER_NETHACK)
    auto* a = (NethackEncoderActivations*)act;
#elif defined(PUFFER_MINIMAL)
    auto* a = (MinimalEntityEncoderActivations*)act;
#elif defined(PUFFER_ASTEROIDS)
    auto* a = (AsteroidsEntityEncoderActivations*)act;
#endif
    poison_bytes(a->out.data, numel(a->out.shape) * sizeof(precision_t));
}

static int reg_index(const Allocator& al, void* data_field) {
    for (int i = 0; i < al.num_regs; i++)
        if (al.regs[i].data_ptr == (void**)data_field) return i;
    return -1;
}

// Pool-argmax provenance for compare_env_golden.py's tie check. Each group is
// "argmax_act:flat_act:h1_act:l2w_param:active_width", groups joined by ';'.
static void note_argmax_groups(Writer& w, void* weights, void* activations,
        const Allocator& params, const Allocator& acts) {
#ifdef PUFFER_OSRS_ENTITY_NET
    OsrsEntityEncoderWeights* ew = (OsrsEntityEncoderWeights*)weights;
    OsrsEntityEncoderActivations* a = (OsrsEntityEncoderActivations*)activations;
    char b[96];
    snprintf(b, sizeof(b), "%d:%d:%d:%d:%d", reg_index(acts, &a->inv_pool_argmax.data),
        reg_index(acts, &a->inv_flat.data), reg_index(acts, &a->inv_h1.data),
        reg_index(params, &ew->inv_l2_w.data), 1);
    std::string groups = b;
    const OsrsEntityEncoderDescriptor* d = &GOLDEN_OSRS_DESCRIPTOR;
    OsrsEntityBranchWeights* bw = osrs_entity_branch_weights(ew);
    OsrsEntityBranchActivations* ba = osrs_entity_branch_activations(a);
    for (int i = 0; i < d->num_branches; i++) {
        int active = d->branches[i].expansion == OSRS_ENTITY_BRANCH_ITEM_TABLE
            ? 1 : d->branches[i].type_onehot;
        snprintf(b, sizeof(b), ";%d:%d:%d:%d:%d", reg_index(acts, &ba[i].pool_argmax.data),
            reg_index(acts, &ba[i].flat.data), reg_index(acts, &ba[i].h1.data),
            reg_index(params, &bw[i].l2_w.data), active);
        groups += b;
    }
    w.note("pool_argmax_groups", groups);
#else
    (void)w; (void)weights; (void)activations; (void)params; (void)acts;
#endif
}

static void dump_alloc(Writer& w, const char* prefix, Allocator& al, void* act, const char* role) {
    for (int i = 0; i < al.num_regs; i++) {
        char name[64]; snprintf(name, sizeof(name), "%s_%02d", prefix, i);
        w.tensor(name, *al.regs[i].data_ptr, entry_shape(al.regs[i]),
            entry_dtype(act, al.regs[i]), role);
    }
}
static void load_alloc(const std::string& dir, const char* prefix, Allocator& al) {
    for (int i = 0; i < al.num_regs; i++) {
        char name[64]; snprintf(name, sizeof(name), "%s_%02d.bin", prefix, i);
        read_file(dir + "/" + name, *al.regs[i].data_ptr,
            entry_count(al.regs[i]) * al.regs[i].elem_size);
    }
}

// ---- Environment-valid observations ------------------------------------------

static void fill_obs(std::vector<float>& obs, int B, int O, uint32_t seed) {
    for (int64_t i = 0; i < (int64_t)B * O; i++) obs[i] = pattern((uint32_t)i, seed, 1.0f);
#ifdef PUFFER_OSRS_ENTITY_NET
    struct Branch { int start, records, features, onehot, scale; bool item; };
    std::vector<Branch> br = {{OSRS_ENTITY_INV_START, OSRS_ENTITY_INV_NUM_RECORDS,
        OSRS_ENTITY_INV_OBS_FEATURES, 0, OSRS_ITEM_OBS_CODE_SCALE, true}};
    const OsrsEntityEncoderDescriptor* d = &GOLDEN_OSRS_DESCRIPTOR;
    for (int i = 0; i < d->num_branches; i++) {
        const OsrsEntityBranchDescriptor& x = d->branches[i];
        br.push_back({x.obs_start, x.num_records, x.obs_features, x.type_onehot,
            x.type_code_scale, x.expansion == OSRS_ENTITY_BRANCH_ITEM_TABLE});
    }
    for (int b = 0; b < B; b++) for (size_t k = 0; k < br.size(); k++) {
        for (int r = 0; r < br[k].records; r++) {
            uint32_t id = (uint32_t)((b * 64 + k) * 64 + r);
            int n = br[k].item ? OSRS_ITEM_OBS_TABLE_ROWS : br[k].onehot + 1;
            int code = randint(id, seed + 1, 10) < 3 ? 0 : 1 + randint(id, seed + 2, n - 1);
            if (b == 1) code = 0;                    // fully inactive row: argmax -1
            if (b == 2) code = 1 + (int)(k % (n - 1));  // identical records: first-wins ties
            obs[(int64_t)b * O + br[k].start + r * br[k].features] =
                (float)code / (float)br[k].scale;
        }
    }
#elif defined(PUFFER_CRAFTAX)
    static const int vocab[CX_CHANNELS] = {64, 8, 2, 16, 16, 16, 16, 16};
    for (int b = 0; b < B; b++) for (int c = 0; c < CX_NUM_CELLS; c++) for (int ch = 0; ch < CX_CHANNELS; ch++) {
        int i = (b * CX_NUM_CELLS + c) * CX_CHANNELS + ch;
        obs[(int64_t)b * O + c * CX_CHANNELS + ch] = (float)randint(i, seed + 1, vocab[ch]);
    }
#elif defined(PUFFER_NETHACK)
    // Byte-encoded engine layout: valid coordinates, signed LE stats, varied
    // glyph kinds, padded inventory/streams, unknown identities and spells.
    std::fill(obs.begin(), obs.end(), 0.0f);
    for (int b = 0; b < B; b++) {
        float* o = obs.data() + (int64_t)b * O;
        auto put = [&](int off, int v, int n) {
            for (int j = 0; j < n; j++) o[off + j] = float((uint32_t(v) >> (8*j)) & 255);
        };
        for (int c = 0; c < NH_MGRID; c++)
            put(2*c, randint(b*NH_MGRID+c, seed, NH_GLYPH_VOCAB), 2);
        for (int j = 0; j < NH_BL_RAW + NH_EX_RAW; j++)
            put(NH_BL_OFF + 4*j, randint(b*123+j, seed+1, 101), 4);
        put(NH_BL_OFF, b % 3 == 0 ? 0 : b % 3 == 1 ? 78 : 39, 4);
        put(NH_BL_OFF+4, b % 3 == 0 ? 0 : b % 3 == 1 ? 20 : 10, 4);
        put(NH_BL_OFF+4*16, -15, 4); // signed AC
        put(NH_BL_OFF+4*26, b%3-1, 4);
        for (int j=0; j<20; j++) put(NH_BL_OFF+4*(NH_BL_RAW+NH_EX_ROLEOH+j),
            j == b%13 || j == 13+b%5 || j == 18+b%2, 4);
        for (int k=0; k<NH_INV; k++) {
            int g = b == 1 || k%7 == 0 ? NH_PAD_GLYPH : NH_OBJ_LO + (b*17+k)%453;
            if (k < 4 && b != 1) g = NH_OBJ_LO + (k == 0 ? 178 : k == 3 ? 207 : 150+k);
            put(NH_INV_OFF+2*k, g, 2);
            put(NH_INVTRUE_OFF+2*k, k%2 ? g : NH_PAD_GLYPH, 2);
            for (int j=0; j<NH_ST_RAW; j++) put(NH_INVST_OFF+k*NH_ST_RAW+j,
                j==1 ? (k%2 ? 128 : 7) : j==5 ? (k<4 ? 1 : 0) | (k==4 ? 2 : 0) | (k==5 ? 8 : 0)
                : randint(b*55+k, seed+j, j==0 ? 4 : j==2 ? 31 : j==6 ? 2 : 4), 1);
        }
        for (int k=0; k<NH_SPELL_SLOTS; k++) {
            int off=NH_BL_OFF+4*(NH_BL_RAW+NH_EXTRA_SHOP+3+4*k);
            bool active = b != 1 && k%3 != 0;
            put(off, active ? 340+k : 0, 4); put(off+4, active ? 1+k%7 : 0, 4);
            put(off+8, active ? (b+k)%101 : 0, 4); put(off+12, active ? 1000+100*k : 0, 4);
        }
        const char* msg = "You hear a distant door open. The goblin hits!";
        for (int j=0; msg[j] && b != 1; j++) o[NH_MSG_OFF+j]=msg[j];
        for (int mon=0; mon<2; mon++) for (int k=0; k<NH_LABK; k++) {
            int off=(mon ? NH_TOKM_OFF : NH_TOKI_OFF)+k*8;
            bool active=b != 1 && k%5 != 0;
            put(off, active ? 1+randint(b*16+k, seed+mon, mon ? 381 : 834) : 0, 2);
            put(off+2, k-8, 1); put(off+3, 8-k, 1);
            put(off+4, k%16, 1); put(off+5, k%4, 1); put(off+6, 12+k, 1);
        }
    }
#elif defined(PUFFER_NMMO3)
    static const int vocab[N3_NFEAT] = {4, 4, 17, 5, 3, 5, 5, 5, 7, 4};
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < N3_MAP_H * N3_MAP_W; c++) for (int f = 0; f < N3_NFEAT; f++) {
            int i = (b * N3_MAP_H * N3_MAP_W + c) * N3_NFEAT + f;
            obs[(int64_t)b * O + c * N3_NFEAT + f] = (float)randint(i, seed + 1, vocab[f]);
        }
        for (int f = 0; f < N3_PLAYER; f++)
            obs[(int64_t)b * O + N3_MAP_SIZE + f] = (float)randint(b * N3_PLAYER + f, seed + 2,
                N3_EMBED_VOCAB);
    }
#endif
}

#ifdef PUFFER_NETHACK
// Conditional heads are tested independently of network rounding. Both epsilon
// endpoints run against identical CUDA logits, masks, teacher actions and RNG.
static void run_nethack_heads(Writer& w, Prec logits, int B,
        const std::string& golden, bool replay) {
    const int A = NH_DEC_OD;
    Allocator al = {};
    auto add = [&](auto& t, int n) { t.shape[0]=n; alloc_register(&al, &t); };
    Float acts={}, env_acts={}, teacher={}, logps={}, new_lp={}, partials={}, losses={}, ec={};
    Prec lp={}, value={}, mask={}, old={}, adv={}, val={}, ret={}, imp={}, live_v={};
    add(acts,B*NUM_ATNS); add(env_acts,B*NUM_ATNS); add(lp,B); add(value,B);
    add(mask,B*A); add(teacher,B*NUM_ATNS); add(old,B); add(adv,B); add(val,B); add(ret,B);
    add(imp,B); add(live_v,B); add(logps,B*A); add(new_lp,B);
    int blocks=(B+PPO_THREADS-1)/PPO_THREADS;
    add(partials,blocks*LOSS_N); add(losses,LOSS_N+1); add(ec,1);
    alloc_create(&al); poison(al);
    int sizes_h[]=ACT_SIZES, *sizes;
    cudaMalloc((void**)&sizes,sizeof(sizes_h));
    cudaMemcpy(sizes,sizes_h,sizeof(sizes_h),cudaMemcpyHostToDevice);
    std::vector<float> m(B*A,1), t(B*NUM_ATNS), v(B);
    for (int b=0; b<B; b++) {
        int off=0;
        for (int h=0; h<NUM_ATNS; h++) {
            // Cover each verb plus masked last actions and all-masked heads.
            int action=h==0 ? b%26 : (b+h)%sizes_h[h];
            t[b*NUM_ATNS+h]=float(action);
            for (int j=0; j<sizes_h[h]; j++)
                m[b*A+off+j] = b%7 == 0 ? 0.0f : j==action || (j+b+h)%3 != 0;
            off += sizes_h[h];
        }
    }
    upload_precision(mask.data, m);
    cudaMemcpy(teacher.data,t.data(),t.size()*4,cudaMemcpyHostToDevice);
    auto fill = [&](Prec& x, int seed, float scale) {
        for (int b=0; b<B; b++) v[b]=pattern(b,seed,scale);
        upload_precision(x.data, v);
    };
    fill(old,24,2); fill(adv,25,1); fill(val,26,1); fill(ret,27,1);
    float ent=0.0065f;
    cudaMemcpy(ec.data,&ent,4,cudaMemcpyHostToDevice);
    curandStatePhilox4_32_10_t* rng;
    cudaMalloc((void**)&rng,(size_t)B*sizeof(*rng));
    if (replay) read_file(golden+"/heads_logits.bin", logits.data, (size_t)B*(A+1)*sizeof(precision_t));
    w.tensor("heads_logits",logits.data,{B,A+1},prec_dtype,"input");
    w.tensor("heads_mask",mask.data,{B,A},prec_dtype,"input");
    w.tensor("heads_teacher",teacher.data,{B,NUM_ATNS},"f32","input");
    for (int e=0; e<2; e++) {
        poison_bytes(acts.data, B*NUM_ATNS*4);
        poison_bytes(env_acts.data, B*NUM_ATNS*4);
        poison_bytes(lp.data, B*sizeof(precision_t));
        poison_bytes(value.data, B*sizeof(precision_t));
        poison_bytes(imp.data, B*sizeof(precision_t));
        poison_bytes(live_v.data, B*sizeof(precision_t));
        poison_bytes(logps.data, B*A*4);
        poison_bytes(new_lp.data, B*4);
        poison_bytes(partials.data, blocks*LOSS_N*4);
        nethack_policy_on_rollout(e ? 100 : 0,100);
        PUF_LAUNCH(rng_init,grid_size(B),BLOCK_SIZE,0,0)(rng,73,B);
        PUF_LAUNCH(sample_logits,grid_size(B),BLOCK_SIZE,0,0)(logits,Prec{},sizes,
            acts.data,env_acts.data,lp.data,value.data,rng,mask.data,A);
        std::string pre=e ? "heads_zero_" : "heads_eps_";
        w.tensor(pre+"actions",acts.data,{B,NUM_ATNS},"f32","output");
        w.tensor(pre+"env_actions",env_acts.data,{B,NUM_ATNS},"f32","output");
        w.tensor(pre+"logprob",lp.data,{B},prec_dtype,"output");
        w.tensor(pre+"value",value.data,{B},prec_dtype,"output");
        w.tensor(pre+"rng",rng,{B,16},"i32","output");
        Prec view={.data=logits.data,.shape={B,1,A+1}};
        PUF_LAUNCH(cache_imp_and_v,grid_size(B),BLOCK_SIZE,0,0)(view,teacher.data,
            old.data,mask.data,Prec{},sizes,imp.data,live_v.data,logps.data,new_lp.data);
        w.tensor(pre+"importance",imp.data,{B},prec_dtype,"output");
        w.tensor(pre+"new_logprob",new_lp.data,{B},"f32","output");
        w.tensor(pre+"logps",logps.data,{B,A},"f32","output");
        // Match the production PPO input boundary (new_lp and logps cached).
        if (replay) {
            read_file(golden+"/"+pre+"logps.bin",logps.data,(size_t)B*A*4);
            read_file(golden+"/"+pre+"new_logprob.bin",new_lp.data,B*4);
        }
        PPOKernelArgs a={.grad_logits=logps.data,.grad_logstd=nullptr,
            .grad_values_pred=new_lp.data,.logits=logits.data,.logstd=nullptr,
            .values_pred=logits.data+A,.act_sizes=sizes,.action_mask=mask.data,
            .num_atns=NUM_ATNS,.clip_coef=0.32f,.vf_clip_coef=1.6f,.vf_coef=0.4f,
            .ent_coef=ec.data,.T_seq=1,.A_total=A,.N=B,.is_continuous=false};
        PPOGraphArgs g={.imp=imp.data,.actions=teacher.data,.old_logprobs=old.data,
            .advantages=adv.data,.values=val.data,.returns=ret.data};
        PUF_LAUNCH(ppo_loss_compute,blocks,PPO_THREADS,0,0)(partials.data,a,g);
        cudaMemset(losses.data,0,(LOSS_N+1)*4);
        PUF_LAUNCH(ppo_loss_reduce,1,LOSS_N,0,0)(losses.data,partials.data,blocks);
        w.tensor(pre+"grad_logits",logps.data,{B,A},"f32","output");
        w.tensor(pre+"grad_value",new_lp.data,{B},"f32","output");
        w.tensor(pre+"losses",losses.data,{LOSS_N+1},"f32","output");
    }
    cudaDeviceSynchronize();
    cudaFree(rng); cudaFree(sizes); cudaFree(al.mem); free(al.regs);
}
#endif

// ---- One case -----------------------------------------------------------------

static void run_case(const std::string& name, int B, int H, const std::string& golden,
        const std::string& out_root) {
    const int O = OBS_SIZE;
    bool replay = !golden.empty();
    std::string gdir = golden + "/" + name;
    Writer w(out_root + "/" + name);
    w.note("operation", "custom_encoder_forward_backward");
    w.note("env", PUFFER_ENV_NAME);
    w.note("mode", replay ? "replay" : "generate");
    w.scalar("B", B); w.scalar("H", H); w.scalar("obs_size", O);

    Encoder enc = {.in_dim = O, .out_dim = H};
    create_custom_encoder(&enc);
    void* weights = enc.create_weights(&enc);
    Allocator params = {}, acts = {}, grads = {}, rollout = {};
    enc.reg_params(weights, &params);
#ifdef PUFFER_NETHACK
    Decoder dec = {.hidden_dim = H, .output_dim = NH_DEC_OD};
    create_custom_decoder(&dec);
    void* dw = dec.create_weights(&dec);
    dec.reg_params(dw, &params);
    void* da = calloc(1, dec.activation_size);
    void* dra = calloc(1, dec.activation_size);
#endif
    alloc_create(&params);
    void* act = calloc(1, enc.activation_size);
    void* ract = calloc(1, enc.activation_size);
    enc.reg_train(weights, act, &acts, &grads, B);
#ifdef PUFFER_NETHACK
    dec.reg_train(dw, da, &acts, &grads, B);
#endif
    enc.reg_rollout(weights, ract, &rollout, B);
#ifdef PUFFER_NETHACK
    dec.reg_rollout(dw, dra, &rollout, B);
#endif
    alloc_create(&acts); alloc_create(&grads); alloc_create(&rollout);
#ifndef PRECISION_FLOAT
    poison(acts); poison(grads); poison(rollout);
#endif

    precision_t *obs_d, *grad_d;
    cudaMalloc((void**)&obs_d, (size_t)B * O * sizeof(precision_t));
    cudaMalloc((void**)&grad_d, (size_t)B * H * sizeof(precision_t));
    if (replay) {
        load_alloc(gdir, "param", params);
        read_file(gdir + "/obs.bin", obs_d, (size_t)B * O * sizeof(precision_t));
    } else {
        for (int i = 0; i < params.num_regs; i++) {
            AllocEntry& e = params.regs[i];
            int64_t n = entry_count(e), cols = e.shape[1] ? e.shape[1] : n;
            std::vector<float> v(n);
            for (int64_t j = 0; j < n; j++)
                v[j] = pattern((uint32_t)j, 100 + i, 1.0f / sqrtf((float)cols));
            upload_precision((precision_t*)*e.data_ptr, v);
        }
        std::vector<float> obs((size_t)B * O);
        fill_obs(obs, B, O, 7);
        upload_precision(obs_d, obs);
    }
    dump_alloc(w, "param", params, nullptr, "param");
    w.tensor("obs", obs_d, {B, O}, prec_dtype, "input");

    Prec input = {.data = obs_d, .shape = {B, O}};
    poison_forward(act);
    Prec out = enc.forward(weights, act, input, 0);
    w.tensor("out", out.data, {B, H}, prec_dtype, "output");
#ifdef PUFFER_NETHACK
    poison_bytes(((NethackDecoderActivations*)da)->out.data, (size_t)B*(NH_DEC_OD+1)*sizeof(precision_t));
    Prec dout = dec.forward(dw, da, out, 0);
    w.tensor("decoder_out", dout.data, {B, NH_DEC_OD+1}, prec_dtype, "output");
#endif
    dump_alloc(w, "acts_fwd", acts, act, "state");
    note_argmax_groups(w, weights, act, params, acts);
    poison_forward(ract);
    Prec rout = enc.forward(weights, ract, input, 0);
    w.tensor("rollout_out", rout.data, {B, H}, prec_dtype, "output");
#ifdef PUFFER_NETHACK
    poison_bytes(((NethackDecoderActivations*)dra)->out.data, (size_t)B*(NH_DEC_OD+1)*sizeof(precision_t));
    Prec drout = dec.forward(dw, dra, rout, 0);
    w.tensor("decoder_rollout_out", drout.data, {B, NH_DEC_OD+1}, prec_dtype, "output");
#endif

    std::vector<float> g((size_t)B * H);
    for (size_t i = 0; i < g.size(); i++) g[i] = pattern((uint32_t)i, 9, 0.5f);
    upload_precision(grad_d, g);
    w.tensor("grad", grad_d, {B, H}, prec_dtype, "input");
    // Backward starts from the golden post-forward state on replay.
    if (replay) load_alloc(gdir, "acts_fwd", acts);
    Prec grad = {.data = grad_d, .shape = {B, H}};
#ifdef PUFFER_NETHACK
    std::vector<float> dg((size_t)B * NH_DEC_OD), dv(B);
    for (size_t i=0; i<dg.size(); i++) dg[i]=pattern(i, 11, 0.5f);
    for (int i=0; i<B; i++) dv[i]=pattern(i, 12, 0.5f);
    float *dg_d, *dv_d;
    cudaMalloc((void**)&dg_d, dg.size()*4); cudaMalloc((void**)&dv_d, B*4);
    cudaMemcpy(dg_d, dg.data(), dg.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dv_d, dv.data(), B*4, cudaMemcpyHostToDevice);
    Float gl = {.data=dg_d, .shape={B, NH_DEC_OD}}, gv = {.data=dv_d, .shape={B}};
    grad = dec.backward(dw, da, gl, {}, gv, 0);
    w.tensor("decoder_grad_input", grad.data, {B,H}, prec_dtype, "output");
    auto* dact = (NethackDecoderActivations*)da;
    w.tensor("decoder_keygrad", dact->keygrad.data, {B,NH_INV_FLAT}, prec_dtype, "output");
    w.tensor("decoder_spdk", dact->spdk.data, {B,NH_SPELL_SLOTS*NH_SPKEY}, prec_dtype, "output");
    // Isolate encoder backward from decoder-rounding while retaining CUDA's
    // actual key/spell gradient consumers and decoder input gradient.
    if (replay) {
        read_file(gdir+"/decoder_grad_input.bin", grad.data, (size_t)B*H*sizeof(precision_t));
        auto* a = (NethackDecoderActivations*)da;
        read_file(gdir+"/decoder_keygrad.bin", a->keygrad.data, (size_t)B*NH_INV_FLAT*sizeof(precision_t));
        read_file(gdir+"/decoder_spdk.bin", a->spdk.data, (size_t)B*NH_SPELL_SLOTS*NH_SPKEY*sizeof(precision_t));
    }
#endif
    enc.backward(weights, act, grad, 0);
    dump_alloc(w, "grads", grads, nullptr, "output");
    dump_alloc(w, "acts_bwd", acts, act, "output");
#ifdef PUFFER_NETHACK
    run_nethack_heads(w, dout, B, gdir, replay);
#endif
    w.finish();

    cudaDeviceSynchronize();
    cudaFree(obs_d); cudaFree(grad_d);
    cudaFree(params.mem); cudaFree(acts.mem); cudaFree(grads.mem); cudaFree(rollout.mem);
    free(params.regs); free(acts.regs); free(grads.regs); free(rollout.regs);
    free(act); free(ract); free(weights);
#ifdef PUFFER_NETHACK
    cudaFree(dg_d); cudaFree(dv_d); free(dw); free(da); free(dra);
    nh_ptr_keygrad = nullptr; nh_ptr_spkeygrad = nullptr;
#endif
}

}  // namespace env_golden

// Default case: each env's configured policy.hidden_size. nmmo3's batch is
// large enough to hit the 256-block cap of its embedding backward; craftax's
// 520 already exceeds its 2048-block cap (grid-stride loop).
#ifndef GOLDEN_HIDDEN
#if defined(PUFFER_OSRS_INFERNO) || defined(PUFFER_CRAFTAX) || defined(PUFFER_NETHACK)
#define GOLDEN_HIDDEN 1024
#elif defined(PUFFER_OSRS_COLOSSEUM) || defined(PUFFER_NMMO3)
#define GOLDEN_HIDDEN 512
#elif defined(PUFFER_ASTEROIDS)
#define GOLDEN_HIDDEN 256
#else
#define GOLDEN_HIDDEN 128
#endif
#endif
#ifndef GOLDEN_BATCH
#if defined(PUFFER_NMMO3)
#define GOLDEN_BATCH 1500
#else
#define GOLDEN_BATCH 520
#endif
#endif

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "";
    std::string golden, out;
    if (mode == "generate" && argc == 3) out = argv[2];
    else if (mode == "replay" && argc == 4) { golden = argv[2]; out = argv[3]; }
    else {
        fprintf(stderr, "usage: %s generate OUT_ROOT | replay GOLDEN_ROOT OUT_ROOT\n", argv[0]);
        return 2;
    }
    std::string env = PUFFER_ENV_NAME;
    cublas_init_handle();
#ifdef PUFFER_NETHACK
    Ini ini = {};
    puf_ini_load_env(&ini, PUFFER_ENV_NAME, 0, nullptr);
    nethack_policy_init(&ini);
    puf_ini_free(&ini);
#endif
    // Partial batch/hidden tiles, then the configured hidden size.
    env_golden::run_case(env + "_small", 13, 40, golden, out);
    env_golden::run_case(env + "_default", GOLDEN_BATCH, GOLDEN_HIDDEN, golden, out);
    return 0;
}

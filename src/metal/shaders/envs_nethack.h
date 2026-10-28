// Default NetHack policy dimensions and engine glyph tables. Mirrors nethack.cu.
#pragma once
#include "common.h"
#include "../nethack_config.h"
#define NH_NUM_OBJECTS 453
#define NH_GLYPH_OBJ_OFF 1906
constant int NH_MAPW = 79, NH_MAPH = 21;
constant int NH_MGRID = NH_MAPW * NH_MAPH;
constant int NH_GLYPH_VOCAB = 5977; // MAX_GLYPH + 1 (NetHack 3.6.6)
constant int NH_PAD_GLYPH = NH_GLYPH_VOCAB - 1; // NO_GLYPH: off-map crop cells
constant int NH_EMBED_DIM = 32;
constant int NH_TERRC_DIM = 8; // v3 crop terrain code width (own tiny table)
constant int NH_RING = 9; // v3 3x3 adjacency ring, full-width monster rows
constant int NH_CROP = 9, NH_CHALF = 4; // NETHACK_CROP, egocentric
constant int NH_CGRID = NH_CROP * NH_CROP;
constant int NH_PW = 5, NH_PH = 5; // patch size (cells)
constant int NH_PX = 16, NH_PY = 5; // patch grid (ceil 79/5, 21/5)
constant int NH_TOK = NH_PX * NH_PY; // 80 global tokens
constant int NH_PCELLS = NH_PW * NH_PH; // cells per patch (off-map -> pad glyph)
// v5 champion-candidate (ENCODER_V5.md): NH_LAB base + five gated changes,
// each independently settable for the per-change screen.
// NH_LOC3: 9-class semantic crop (croplab: danger .998, esc-dir .83;
// classes ~= depth, both help; width flat 128-512 -> 256)
constant int NH_LOCC_DIM = 8;
constant int NH_LOCC_CLASSES = 9; // wall floor door hazard mon item stone other offmap
// v3 local: terrain field (81x8, own table) + adjacency ring (9x32, T_mon)
// + underfoot item (32, T_item). Positions share "what", loc_w keeps "where".
constant int NH_LOC_IN = NH_CGRID * NH_LOCC_DIM;
constant int NH_LOC_HID = 256;
// two-layer local branch (crop -> NH_LOC_H1 -> NH_LOC_HID) so bearing
// at 2-4 tiles has a nonlinearity to be represented in (audit: 0.63 decodable
// from the single linear+relu, 0.39 in the direction logits). NH_LOC3 forces
// the two-layer path at hidden 256.
constant int NH_LOC_H1 = 256; // hidden width; loc_w is (NH_LOC_H1, NH_LOC_IN)
// NH_TERR: landmark table (12x4) + sector radar (8x4x17) -> MLP (terrain lab)
constant int NH_TERRF = 12 * 4 + 8 * 4 * 17; // 592 features
constant int NH_TERR_H1 = 256;
// Global branch: per patch, embed+flatten (25 cells x 32 dims) + normalized
// (dx,dy) patch-center offset from the hero -> 16 -> 128, then elementwise
// MAX over the 80 tokens. The 16-dim bottleneck keeps the fused per-glyph
// gather table (embed+flatten+first layer, NH_TROW cols) L2-resident, and
// fusing the max means the (B, 80, 128) token activations never exist in
// memory. The (dx,dy) slice lives in its own 16x2 weight tensor — same math
// as concatenating onto the flatten.
constant int NH_P1 = 16;
constant int NH_GLB_HID = 128;
constant int NH_TROW = NH_PCELLS * NH_P1; // fused-table row: per-pos 16-dim
constant int NH_PAD_PER_SAMPLE = NH_TOK * NH_PCELLS - NH_MGRID;
constant int NH_HOT_G = 10; // hot-glyph dT smem slots (10x400 int64 = 32KB)
// typed token streams (supervised-lab locked design, 2026-08). Typed K-nearest token
// streams (monsters, items; lean NH_TOK_OBS lists env-side) with deep values
// (2-layer MLP), 8-head attention pools scoring from raw token features, and
// a rank feature; inventory gets the 8-query attention pool (NH_INV_ATTN);
// aux mega-spec heads regress ~26 engine-truth targets off the concat
// (training-time only shaping; targets ride the NH_AUX_OBS extras block and
// are never featurized). All additive beside the champion branches.
// component toggles for leave-one-out ablation
constant int NH_BL_RAW = 27; // NLE_BLSTATS_SIZE
constant int NH_BL_HUNGER = 21, NH_BL_CONDITION = 25;
constant int NH_BL_HP = 10, NH_BL_ENE = 14; // hp/hpmax at 10/11, ene/enemax at 14/15
constant int NH_ACTIONS = 26; // NETHACK_NUM_ACTIONS
constant int NH_OCLASSES = 18; // MAXOCLASSES
constant int NH_EXTRA_SHOP = 2 + NH_OCLASSES; // extra[] index of the shop pair
constant int NH_SPELL_SLOTS = 8; // NETHACK_SPELL_SLOTS
constant int NH_EX_ROLEOH = 2 + NH_OCLASSES + 2 + 1 + 4 * NH_SPELL_SLOTS + 2;
constant int NH_EX_RAW = NH_EX_ROLEOH + 13 + 5 + 2
                               + 1
; // NETHACK_EXTRA_INTS (+role/race/gender one-hots)
// blstats feature map (cumulative offsets; each block documented at its
// kernel branch). hp/ene fracs are the danger ratios the linear bl_w can't
// synthesize from separate cur/max scalars; dnum is one-hot because dungeon
// branch is nominal, not ordinal.
constant int NH_F_HUNGER = 25; // 7-way one-hot
constant int NH_F_COND = NH_F_HUNGER + 7; // 13 condition bits
constant int NH_F_PREV = NH_F_COND + 13; // prev-action one-hot
constant int NH_F_INV = NH_F_PREV + NH_ACTIONS; // inv class counts
constant int NH_F_FRAC = NH_F_INV + NH_OCLASSES; // hp_frac, ene_frac
constant int NH_F_DNUM = NH_F_FRAC + 2; // 8-way one-hot
constant int NH_F_ENGR = NH_F_DNUM + 8; // engraving bits
constant int NH_F_SHOP = NH_F_ENGR + 2; // in-shop, affordability
// spell scalar feature: known count only — per-slot content rides the
// spell-key path (v3 pointer), not the blstats block
constant int NH_F_SPELL = NH_F_SHOP + 2;
// encumbrance ratio (softsign around the wall) + carry capacity /1000
constant int NH_F_WEIGHT = NH_F_SPELL + 1;
constant int NH_F_ROLE = NH_F_WEIGHT + 2; // 13 role + 5 race + 2 gender
constant int NH_F_INTRINS = NH_F_ROLE + 20; // 8 intrinsic bits
// threat block: visible bit, proximity, bearing 8-onehot, adj/near3/vis
// counts, nearest difficulty, difficulty-vs-xplvl, nearest speed
constant int NH_F_THREAT = NH_F_INTRINS + 8;
constant int NH_BL_FEAT = NH_F_THREAT;
constant int NH_EX_INTRINS = NH_EX_ROLEOH + 20;
constant int NH_EX_THREAT = NH_EX_INTRINS + 1;
constant int NH_BL_DNUM = 23;
constant int NH_BL_HID = 64;
// Inventory entity branch: 55 slot glyphs, each embed -> shared 32->32
// linear -> relu. The per-slot vectors are the pointer decoder's keys (slot
// identity lives there); the trunk only gets the pooled summary below. Fused
// per-glyph table T_inv = E @ inv1_w^T (5977xNH_INV_HID) rebuilt per forward.
constant int NH_INV = 55; // NETHACK_INV_SLOTS
// 16-dim per-slot rep: doubles as the pooled-summary bottleneck AND the
// decoder pointer key (unified, patch-encoder style). Halves the pool max
// MACs, the deterministic max-backward atomics, and the max-kernel smem
// footprint (better occupancy) vs the old 32; also shrinks the decoder.
constant int NH_INV_HID = 16;
constant int NH_INV_FLAT = NH_INV * NH_INV_HID;
// The trunk sees a pooled inventory summary, not the 1760-dim flatten: per
// slot 32 -> 128 with the elementwise max folded in (patch-encoder trick, the
// (B,55,128) intermediate never exists). Slot identity for the action heads
// comes from the pointer decoder's keys (inv_out), not the trunk.
constant int NH_INV_POOL = 128;
// Trigram message branch: char-trigram bag over the raw topline. Each trigram
// hashes into NH_MSG_VOCAB buckets, its NH_MSG_HID-dim embed row summed, then
// scaled by 1/sqrt(count+1) (normalized bag / EmbeddingBag sum). The summary
// is concatenated raw (signed, no relu) like the blstats raw features.
constant int NH_MSG_LEN = 128; // raw topline chars in obs tail
constant int NH_MSG_VOCAB = 1024; // champion: ~4.6K distinct trigrams, 1024 buckets tie 4096 (probe + RL n=8)
constant int NH_MSG_LOG2V = 10; // log2(NH_MSG_VOCAB)
constant int NH_MSG_HID = 256; // champion: width ladder 32/128/256 pooled n=13, 256 ahead on mean/median/tail
// NH_INV2 drops the max-pool trunk summary (half-dead in production);
// its slice leaves the concat entirely.
constant int NH_INVP_DIM = 0;
constant int NH_MSG_CONCAT_OFF = NH_LOC_HID + NH_GLB_HID + NH_INVP_DIM + NH_BL_HID + NH_BL_FEAT;
// spell-key path (v3 pointer): per slot, key = spk_w . [e_eff(book glyph) |
// known, lev/7, fail/100, know/20000]; keys feed the CAST pointer head and a
// sum-pooled 16-dim trunk summary. Empty slots are exact zeros end to end.
constant int NH_SPKEY = NH_INV_HID; // 16, shared key width
constant int NH_SPIN = NH_EMBED_DIM + 4; // 36 key inputs/slot
// spell entity block (v6): keys r = relu(spk_w x) -> per-slot MLP 16->64->64
// -> masked sum x0.25 | masked max + 4 exact doorstep scalars [min_fail,
// max_lev, n/8, min_retention]. Same block as inventory/streams; trunk channel
// backprops through the keys (dense gradient; the old spk2 raw-key max-pool
// destroyed identity, AUC .18 vs .97 — this max is over post-depth values).
constant int NH_SPM = 64;
constant int NH_SPELL_SLICE = 2 * NH_SPM + 4;
constant int NH_SPELL_CONCAT_OFF = NH_MSG_CONCAT_OFF + NH_MSG_HID;
// identity embeddings: explicit role/race/gend/align tables in a direct concat channel
constant int NH_IDE_ROLE = 16, NH_IDE_RACE = 8;
constant int NH_IDE_GEND = 8, NH_IDE_ALGN = 8;
constant int NH_IDE_DIM = NH_IDE_ROLE + NH_IDE_RACE + NH_IDE_GEND + NH_IDE_ALGN;
constant int NH_IDE_CONCAT_OFF = NH_SPELL_CONCAT_OFF + NH_SPELL_SLICE;
constant int NH_GMEAN_CONCAT_OFF = NH_IDE_CONCAT_OFF + NH_IDE_DIM;
constant int NH_TERR_VOCAB = 128; // cmap_index+1; 0 unseen; 127 off-map pad
constant int NH_ENT_K = 32;
constant int NH_ENT_F = 6;
constant int NH_ENT_IN = 40;
constant int NH_ENT_HID = 32;
constant int NH_ENT_TAIL = 36;
// v3 typed-level: glyph ranges, tables, two K-nearest token lists
constant int NH_NUMMONS = 381, NH_PET_OFF = 381, NH_DET_OFF = 762;
constant int NH_BODY_OFF = 1144, NH_OBJ_LO = 1906;
constant int NH_MON_ROWS = 384;   // species+1 (0 = pad/empty)
constant int NH_ITEM_ROWS = 840;  // objects 1..453, bodies 454..834, 0 = pad
#ifndef NETHACK_V3_K
#define NETHACK_V3_K 16
#endif
constant int NH_V3K = NETHACK_V3_K;
constant int NH_V3_MONF = 8, NH_V3_ITEMF = 8;
constant int NH_V3_IN = 40, NH_V3_HID = 32, NH_V3_TAIL = 36;
constant int NH_ITBL = NH_GLYPH_VOCAB; // inv/discovery table rows

constant int NH_AP_DIM = NH_P1 + 2;
constant int NH_TK_K = 8;
constant int NH_TK_DIM = NH_P1 + 2; // gated rep + (dx,dy)
constant int NH_ENT_CONCAT_OFF = NH_GMEAN_CONCAT_OFF;
constant int NH_IVA_M = 8; // invattn query heads (invattn8)
constant int NH_IVA_CONCAT_OFF = NH_ENT_CONCAT_OFF;
// lab stream dims: 16 tokens x 48 inputs (emb 32 + dx dy cheb rank + flags +
// diff speed), deep values 48 -> 64 -> 64, 8 heads x 8 dims
constant int NH_LABK = NETHACK_V3_K;
constant int NH_LAB_IN = 48;
constant int NH_LAB_HID = 64;
constant int NH_LAB_HEADS = 8;
constant int NH_AUXH = 32;   // aux head rows (26 used, %8 pad)
constant float NH_AUX_COEF = 0.05f;
constant int NH_LABM_CONCAT_OFF = NH_IVA_CONCAT_OFF + NH_IVA_M * NH_INV_HID;
constant int NH_LABI_CONCAT_OFF = NH_LABM_CONCAT_OFF + NH_LAB_HID;
// NH_INV2 tail: hard wield readout (16, parameterless) + sum channel (64)
constant int NH_WLD_CONCAT_OFF = NH_LABI_CONCAT_OFF + NH_LAB_HID;
constant int NH_ISUM_DIM = 64;
constant int NH_ISUM_CONCAT_OFF = NH_WLD_CONCAT_OFF + NH_INV_HID;
constant int NH_BL_OFF = 2 * NH_MGRID; // blstats offset, obs elements
constant int NH_INV_OFF = NH_BL_OFF + (NH_BL_RAW + NH_EX_RAW) * 4;
// obs v4: per-slot identification-gated state, 8 int8 fields per slot
// [buc, spe(-128=unknown), quan, ero1, ero2, flags, typeknown, rsvd],
// expanded to NH_SFEAT features feeding the slot MLP beside the embed
constant int NH_INVST_OFF = NH_INV_OFF + NH_INV * 2;
constant int NH_ST_RAW = 8; // NLE_INV_STATE_FIELDS
constant int NH_SFEAT = 24; // buc4 + known+spe + quan + ero2 + flags7 + tk + armcat7
// discovered-type glyphs: true otyp glyph once dknown && oc_name_known, else pad
constant int NH_INVTRUE_OFF = NH_INVST_OFF + NH_INV * NH_ST_RAW;
constant int NH_PWORN_DIM = NH_INV_HID; // 16-d worn content mean
// accessory panel: 4 single-occupant slots [amulet | ring A | ring B | eyewear],
// each an exact [r16|sfeat24] copy of the worn slot (zeros when bare); rings in
// inventory order. otyps (onames.h, NetHack 3.6.6): rings 150-177, amulets
// 178-188, lenses/blindfold/towel 207-209. Worn bit = owornmask & W_ACCESSORY.
constant int NH_PACC_DIM = 4 * (NH_INV_HID + NH_SFEAT); // 160
constant int NH_PASS_DIM = 40 + 40 + NH_PWORN_DIM + NH_PACC_DIM;
// V6-min: statistics + lookups everywhere. Inventory: r16 -> MLP 64->64 ->
// sum|max + pass-throughs (wielded/quivered [r|sfeat], worn-mean). Streams:
// tok48 -> rep16 -> MLP 64->64 -> sum|max + token-0 gate (+underfoot for items).
// No encoder attention anywhere; decoder pointers unchanged.
constant int NH_MV = 64;                     // value width (all arms)
constant int NH_MINV_DIM = 2 * NH_MV + 40 + 40 + NH_PWORN_DIM + NH_PACC_DIM;
constant int NH_MINV_OFF = NH_IVA_CONCAT_OFF;                // 690
constant int NH_MLM_DIM = 2 * NH_MV + NH_INV_HID;            // 144
constant int NH_MLM_OFF = NH_MINV_OFF + NH_MINV_DIM;
constant int NH_MLI_DIM = 2 * NH_MV + 2 * NH_INV_HID;        // 160
constant int NH_MLI_OFF = NH_MLM_OFF + NH_MLM_DIM;
constant int NH_CONCAT = NH_MLI_OFF + NH_MLI_DIM; // 1410

constant int NH_MSG_OFF = NH_INVTRUE_OFF + NH_INV * 2; // message block start
constant int NH_OBS_SIZE = NH_MSG_OFF + NH_MSG_LEN

                                 + NH_V3K * (NH_V3_MONF + NH_V3_ITEMF);
constant int NH_TERR_OFF = NH_MSG_OFF + NH_MSG_LEN;
constant int NH_ENTL_OFF = NH_TERR_OFF + NH_MGRID;
constant int NH_OBJM_OFF = NH_TERR_OFF + NH_MGRID;      // v3 (aliases ENTL; exclusive)
constant int NH_VMON_OFF = NH_OBJM_OFF + NH_MGRID * 2;
constant int NH_VITEM_OFF = NH_VMON_OFF + NH_V3K * NH_V3_MONF;
// lean token lists (NH_LAB / env NH_TOK_OBS): appended after every other
// optional block, no dense planes (netlib.h NETHACK_OFF_TOKM mirror)
constant int NH_TOKM_OFF = NH_MSG_OFF + NH_MSG_LEN
;
constant int NH_TOKI_OFF = NH_TOKM_OFF + NH_V3K * NH_V3_MONF;
constant int NH_SORT_BLOCKS = 256; // hist grid (smem histograms)
constant int NH_HOT_T = 16; // hot-glyph smem rows (16x32 int64 = 4KB)


#define NH_GM_QUAL constant
#include "../../../ocean/nethack/glyph_map.h"
#undef NH_GM_QUAL
constant signed char nh_obj_armcat_dev[NH_NUM_OBJECTS] = {
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,2,2,2,2,2,2,2,2,2,
  2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,6,5,5,5,
  5,5,5,5,5,5,5,5,5,1,1,1,1,1,1,1,3,3,3,3,
  4,4,4,4,4,4,4,4,4,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
};
constant int NH_NKIND = NH_GM_NKIND, NH_NSUB = NH_GM_NSUB;
constant float NH_FXP = 16777216.0f;
constant float NH_BL_SCALE[NH_BL_RAW] = {
    1.f/79, 1.f/21, // x, y
    1.f/25, 1.f/125, 1.f/25, 1.f/25, 1.f/25, 1.f/25, 1.f/25,  // str25 str125 dex con int wis cha
    0.1f, // score (log)
    1.f/200, 1.f/200, 1.f/50, // hp, hpmax, depth
    0.1f, // gold (log)
    1.f/100, 1.f/100, 1.f/10, 1.f/10, 1.f/30, // ene, enemax, ac, hd, xp level
    0.1f, 0.1f, // exp points, time (log)
    0.f, // hunger (expanded)
    1.f/4, 0.f, 1.f/50, // cap, dnum (one-hot), dlevel
    0.f, // condition (expanded)
    1.f, // align
};
constant int NH_BL_ISLOG[NH_BL_RAW] = {
    0,0,0,0,0,0,0,0,0, 1, 0,0,0, 1, 0,0,0,0,0, 1,1, 0, 0,0,0, 0, 0,
};constant int NH_DIRS = 8;
constant int NH_DIRHEADS = 6; // move|run|kick|throw|zap|apply
constant int NH_HEADS = 12; // wear|eat|quaff|throw|zap|takeoff|puton|remove|wield|apply|read|drop
constant int NH_SLOT_OD = NH_HEADS * NH_INV; // 660 slot logits
constant int NH_DEC_OD = NH_ACTIONS + NH_SLOT_OD + NH_DIRHEADS * NH_DIRS
                                + NH_SPELL_SLOTS; // logits: verbs|slots|dirs|spell
constant int NH_DEC_LIN = NH_ACTIONS + NH_DIRHEADS * NH_DIRS + 1; // verbs|dirs|value
constant int NH_DEC_PAD = (NH_DEC_LIN + 7) / 8 * 8; // lin rows padded to mult of 8 (cublasLt alignment)
// queries: 12 cosine inv heads + 1 dot-product spell head (row NH_HEADS)
constant int NH_QHEADS = NH_HEADS + 1;
constant int NH_QDIM = NH_QHEADS * NH_INV_HID; // stacked queries
constant int NH_SPELL_BASE = NH_ACTIONS + NH_SLOT_OD + NH_DIRHEADS * NH_DIRS;
// tau is padded to 8 entries (first NH_HEADS live): checkpoints are saved
// compactly and the puffernet loader assumes every tensor is a multiple of
// 8 floats (16-byte bf16 alignment). Pad slots are dead but NOT frozen
// (optimizer weight decay can drift them from init) — never read.
constant int NH_TAU_PAD = 16;

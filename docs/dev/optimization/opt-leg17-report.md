> Historical development log; last recorded at commit `0b9f8592` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

# Optimization relay leg 17 (Opus): learner critical path

Branch `metal-opt17`, worktree `wt-opt17`, baseline **`55f76032`** (merged
leg 16), implementation **`927c6cc6`**. Nothing pushed or merged. **All eight
seed-1 references are exact at the implementation commit; 23/23 gates PASS.** No
accumulation order changed; every kept change is bit-identical and default-on,
each with an env opt-out. Scratch tooling, builds, profiles and mutation logs:
`wt-opt17/build/leg17/` (untracked); final A/B batches in the session
scratchpad `l17logs/fin_*`.

Final validation: every env and precision is faster in the final clean
A/B (all 52 pairs): **Pong +20.0% BF16 / +11.1% FP32** (learner bound),
Cartpole +4.9% / +5.4%, Flappy +7.6% / +1.5%, Breakout +0.9% / +3.0%.
The first final A/B (`91b45015`, overlap always on) found Flappy FP32 -5.1%
and Breakout BF16 -2.8%. A bisect pinned that on the backward dW overlap
contending with the async rollout, and `927c6cc6` now runs the overlap only
while the learner is the critical path.

## Final clean A/B

`build/cleanab.py`, base `build/leg17/base/ENV_PREC` (`55f76032`) vs final
`build/leg17/pol1/ENV_PREC` (`927c6cc6`), absolute paths, alternating order,
`PUF_CLEANAB_COOLDOWN=10 PUF_CLEANAB_IDLE_FIRST=1 PUF_CLEANAB_MAX_FOREIGN=0.01`
(1% foreign-GPU limit), no builds or other work during the batches. Pong uses
precision-separated 2M-step cohorts with n=8 (leg 16's convention). The other
envs use 20M steps with n=6. Only within-batch comparisons; absolute SPS is
never pooled across batches. M SPS, medians:

| env | precision | before | after | change (medians) | paired median | faster pairs |
|---|---|---:|---:|---:|---:|---:|
| Pong | BF16 | 1.4247 | 1.7086 | **+19.93%** | +20.05% | 8/8 |
| Pong | FP32 | 0.9722 | 1.0796 | **+11.05%** | +11.05% | 8/8 |
| Cartpole | BF16 | 12.614 | 13.209 | **+4.71%** | +4.95% | 6/6 |
| Cartpole | FP32 | 11.907 | 12.596 | **+5.79%** | +5.41% | 6/6 |
| Flappy | BF16 | 5.6946 | 6.1323 | **+7.69%** | +7.57% | 6/6 |
| Flappy | FP32 | 4.5821 | 4.6471 | +1.42% | +1.47% | 6/6 |
| Breakout | BF16 | 4.8168 | 4.8606 | +0.91% | +0.91% | 6/6 |
| Breakout | FP32 | 4.3113 | 4.4347 | +2.86% | +2.99% | 6/6 |

Per-epoch medians (run .ini `metal/*`), base -> final: train / rollout /
epoch ms: Pong BF16 45.5/44.3/46.0 -> 37.8/36.9/38.4; Pong FP32
66.8/64.9/67.4 -> 60.0/58.5/60.7; Cartpole BF16 9.9/10.2/10.4 -> 8.5/9.7/9.9;
Flappy BF16 19.0/22.7/23.0 -> 15.9/21.1/21.4; Flappy FP32 27.1/28.3/28.6 ->
23.0/27.9/28.2; Breakout FP32 28.4/30.1/30.4 -> 26.5/29.2/29.6.

**First final A/B (`91b45015`, overlap always on)**, same protocol: Pong
+20.19% / +10.92% (8/8, 8/8); Cartpole +0.20% / +0.05% (3/6, 3/6); Flappy
BF16 +2.84% (6/6) but **FP32 -5.13% (0/6)**; **Breakout BF16 -2.80% (0/6)**,
FP32 -0.34% (1/6). In the regressing runs train time fell but rollout time
rose (Flappy FP32 train 27.2 -> 22.2 ms, rollout 28.3 -> 29.9 ms). Bisect,
n=4 alternating, 20M steps, paired vs base:

| arm | Flappy FP32 | Breakout BF16 |
|---|---:|---:|
| wide tiles, overlap off (`mma3`, `PUF_METAL_BWD_OVERLAP=0`) | -0.57% (1/4) | |
| wide tiles, overlap on (`mma3`) | -6.01% (0/4) | |
| `91b45015`, overlap off | -0.19% (2/4) | +0.93% (3/4) |
| `91b45015`, overlap on | -4.86% (0/4) | -3.47% (0/4) |
| `91b45015`, Muon fragments off | | -3.42% (0/4) |

So the backward overlap's concurrent dW GEMMs shorten the learner but slow
an overlapping async rollout, which is the critical path in Flappy, Breakout
and Cartpole. Commit 7 (below) gates it per epoch.

## Profile before optimizing (baseline `55f76032`)

`PUF_METAL_KPROF=1` (synchronous per-kernel timing; KPROF runs need
`--base.eval-episodes=0`, Pong's eval otherwise hangs), `--base.async=0`,
4M steps. Per learner minibatch (Pong: 16 minibatches/epoch, 3 MinGRU layers
H=64, T=64, 16384 rows):

| Pong BF16 kernel (per minibatch) | us |
|---|---:|
| `mingru_fused_forward_h64` (projection + scan) | 3 x 295 |
| `mingru_fused_backward_h64` (scan + dX) | 3 x 280 |
| MinGRU dW `tensor_ops_gemm_tn_split` (+ reduce 10) | 3 x 158 |
| decoder forward `puf_gemm` (N = 7 + 1 outputs) | 132 |
| encoder forward nt / decoder dX nn | 50 / 46 |
| encoder / decoder dW `sg16` | 47 / 41 |
| Muon (wide_gram 5x8.4, ns5_batch 35, wide_update 5x6.8, gram_buf 5x3.6, misc) | ~160 |
| PPO (loss_compute 13.5, advantage 11, loss_reduce 10, cache 7, assemble 4) | ~46 |

776 train dispatches and 841 barriers per epoch (every dispatch barriered).
FP32 Pong: forward 369, backward 489, dW 336 us; train_gpu 71.3 vs 49.8 ms
per epoch BF16. The three MinGRU kernels are ~80% of learner GPU time and the
H=64 layer kernels move ~12 MB (forward) / ~20 MB (backward) of activations per
layer per minibatch: they run close to the M4's ~120 GB/s plus the projection
GEMM. Cartpole BF16 (H=32, 2 layers): `mingru_fused_backward_dw` 252,
`mingru_fused_forward32` 145, `muon_ns5_batch` **144 us** (one 1024-thread
threadgroup, threadgroup-load bound), decoder GEMM 89. Breakout: backward 721
x 4, forward 399 x 4. The dW GEMMs depend only on the layer's own gradient, yet
ran barriered between the layer chain's dependent kernels.

## Retained changes

1. **`5f4245cd` small-N GEMM for the decoder forward.** `puf_gemm_small_n`
   (FP32/BF16): one thread per output row-chunk for N <= 7, the tiled kernel's
   exact per-element fma chain, alpha/beta finish and rounding. Pong decoder
   forward 132 -> 30 us. `GemmOpts.allow_small_n`; `PUF_METAL_GEMM_SMALL_N=0`
   opts out. Tests: `test_gemm` (host fma oracle, NaN-poisoned outputs, beta 0
   never reads C, padding columns) and `test_bf16`; 10/10 mutations caught.
2. **`968283b2` concurrent dispatch windows + deferred split-K reductions.**
   Runtime: `concurrent_begin/end` on a stream; inside a window only the first
   dispatch keeps its barrier, later ones run concurrently with it and the first
   dispatch after the window is barriered (fatal on nesting / mismatch). GEMM:
   `gemm_rm_deferred` encodes a split-K GEMM's partial pass now and returns a
   `GemmDeferred` whose reduction `gemm_finish` encodes later, with the same
   split count and summation order as `gemm_rm`. `mingru_fused_backward_rm`
   gains `defer_dw_reduce`. Tests: `test_runtime` barrier-count / value window
   test, `test_gemm` deferred paths (7 cases, no-room fallback, pairs in
   windows), `test_bf16`; 8/8 mutations caught.
3. **`34a18391` overlap backward dW with the layer chain** (`algo.cu`
   `BwdOverlap`). While layer i's backward runs, layer i+1's dW GEMM partials
   (or H=32's fused-dW reduction) run in the same concurrent window; reductions
   drain in the following window, so each layer's dW leaves the dependent
   chain. Only when the standard encoder/MinGRU/decoder stack is all-fused.
   `PUF_METAL_BWD_OVERLAP=0` opts out. Barriers/epoch: Pong 841 -> 713,
   Cartpole 263 -> 231; dispatches unchanged. Env-toggle A/B n=4: Pong BF16
   1.472 -> 1.512 M SPS (+2.8%, 4/4 faster), train_gpu 43.84 -> 42.62 ms;
   Cartpole BF16 train_gpu -7.6%, SPS neutral (+0.23% medians, 2/4).
4. **`b522c1d0` wider matmul2d tiles in the fused H=64 layers.** Forward
   projection tile 96 (BF16) / 64 (FP32) columns, backward dX 64, H=32
   projection 96 FP32 / 32 BF16; the per-element K reduction is independent of
   tile width. KPROF Pong forward 295 -> 223 (BF16), 369 -> 286 us (FP32);
   backward 280 -> 263 / 489 -> 457 us; sync train_gpu 49.8 -> 44.1 / 71.3 ->
   63.9 ms per epoch. Breakout FP32 forward 444 -> 406-419 us.
5. **`b05ee65f` keep H=64 backward grad_input on chip.** The chunk's rounded
   grad_input also lives in a 16-row threadgroup tile for the dX epilogue; the
   device copy is optional and `algo.cu` passes nullptr (nothing reads it).
   KPROF Pong backward 273 -> 266 (BF16), 464 -> 437 us (FP32); saves
   2 / 4 MiB of device writes plus reads per layer per minibatch.
6. **`91b45015` simdgroup-fragment GEMM stages in the fused Muon NS steps.**
   For R, C multiples of 8 the three NS GEMM stages of `muon_ns5` /
   `muon_ns5_batch` use 8x8 simdgroup fragments with `muon_wide_mma`'s lane
   mapping (same ascending-K fma chains). `muon_ns5_batch` Cartpole 176 -> 52
   us (train_gpu 13.16 -> 12.05 ms/epoch), Pong 44.6 -> 40.0 us.
   `PUF_METAL_MUON_NS_MMA=0` opts out.
7. **`927c6cc6` overlap only while the learner is critical.** The existing
   per-epoch `metal_dw_policy` already measures how long the learner waited
   for the rollout after training (slack). `g_metal_bwd_overlap = slack <=
   5% of the epoch` (sync training passes slack 0: always on). Pong (~1%
   slack) keeps it; Flappy/Breakout/Cartpole (10-30%) drop it. Either setting
   is bit-identical. `PUF_METAL_BWD_OVERLAP=0 / 1` pins it off / on.

### Deterministic proxies (per epoch, BF16)

From the final A/B run logs (median of each run's per-epoch value).
Learner dispatches per epoch are unchanged in every env. The changes are
kernel shapes, not dispatch counts. Barriers drop only where the overlap
runs:

| env (BF16 / FP32) | train dispatches | barriers base -> final | commits base -> final |
|---|---|---|---|
| Pong | 776 / 760 (same) | 841 -> 708 / 825 -> 691 | 69 -> 64 / 68 -> 61 |
| Cartpole | 200 / 192 (same) | 263 -> 260 / 255 -> 255 | 64 / 64 |
| Flappy | 360 / 352 (same) | 482 -> 474 / 443 -> 474 | 124 -> 116 / 93 -> 124 |
| Breakout | 92 / 92 -> 89 (*) | 129 -> 129 / 125 -> 120 | 38 / 34 -> 32 |

(Flappy/Breakout dispatch, barrier and commit totals include the rollout's
grouped-buffer commits, which vary with timing. Flappy FP32's commit rise is
rollout pairing: learner dispatches are fixed. (*) Breakout FP32's base median
91.88 shows the count already varies run to run: leg 15's timing-driven
decoder-dW side-stream policy moves that GEMM off the train stream.) Bytes: the small-N decoder
GEMM reads the same operands; on-chip grad_input saves 2 MiB (BF16) / 4 MiB
(FP32) of device writes and as much read-back per H=64 layer per minibatch
(Pong: 3 layers x 16 minibatches = 96 / 192 MiB written per epoch less).
The Muon fragment path moves no device bytes differently.

## Rejected / not kept

- **Warp-specialized H=64 forward** (one simdgroup projects chunk n+1 while
  others scan chunk n): 314 vs 292 us, slower (threadgroup memory doubles,
  occupancy falls).
- **Wider BF16 tiles at H=32** (projection 96): slower than 32; kept 32.
- **Backward chunk 8 rows** (~1-2%, inside noise) and **PREFETCH 8**
  (slower): not kept.
- **Grad_input in threadgroup memory with the device write kept**: the gain
  comes from dropping the store; kept only the nullptr form (5).
- **dW folded into the H=64 fused backward**: changes the dW summation order;
  not attempted as default (contract), listed below as an opt-in lever.

## Exactness, gates and mutation tests

Seed-1 (score / checkpoint sha256 prefix), final build `build/leg17/pol1`
(= `927c6cc6` sources; also exact at `ns1` = `91b45015` and after each
earlier commit for the envs it touched): Breakout FP32 844.95672607421875 /
`2ee4da82d356a44c`, BF16 852.43658447265625 / `f5dc197b8d7fde55`; Pong FP32
20.99445915222168 / `5ee574ef2f33c6dd`, BF16 20.996614456176758 /
`b1425713d9e71433`; Flappy FP32 51.408348083496094 / `4eafd59e547e43e1`, BF16
50.560173034667969 / `91886134a50625e1`; Cartpole FP32 200 /
`bf0330663b50b01e`, BF16 199.96110534667969 / `b10ccdb69f45c996`. All exact.
The overlap opt-out (`PUF_METAL_BWD_OVERLAP=0`) is exact too (Pong, Cartpole
and, in the new gate, Cartpole/Breakout both precisions).

**Gates:** `tests/metal/run_all.sh` at `927c6cc6`: **23/23 PASS** (the 22
earlier gates plus the new `overlap_e2e`). New gate, added to
`tests/metal/run_all.sh` as `run overlap_e2e tests/metal/helpers/overlap_e2e.sh`:

    bash tests/metal/helpers/overlap_e2e.sh

It builds Cartpole and Breakout (BF16 and FP32) and checks seed-1 score and
checkpoint sha in three modes: default (auto policy), forced on
(`PUF_METAL_BWD_OVERLAP=1`) and off (`=0`). All 12 pass. The H=64 overlap
path (Pong, Flappy) runs inside the existing `tests/metal/helpers/h64_e2e.sh`
whenever its runs are learner bound. The extended unit tests run inside
existing gates: `mingru`, `mingru_bf16`, `gemm`, `bf16`, `runtime`
(`tests/metal/run_test_{mingru,mingru_bf16,gemm,bf16,runtime}.sh`). The
overlap policy itself cannot be caught by an exactness gate (both settings
are bit-identical); the final A/B barrier counts are its proxy.

regressions (it compiled the changed shaders and syntax-checked host sources
and tests, with no Metal device).

Mutation checks (`build/leg17/mutate.py`: apply one textual mutation, run the
gate, restore): small-N GEMM 10/10 caught; concurrent windows + deferred
GEMMs 8/8; overlap (seed-run gate) 3/3; wide tiles: all caught except one
equivalent mutation and one relaxed-precision compile flag; grad_input on
chip: index, value, skipped device store and host null check caught, three
survivors are equivalent in the tested precision (BF16-only/FP32-only
epilogue lines, `slice(0,0)` with a single 64-column dX tile) or unobservable
(a store through a null device address, which Metal drops); Muon fragments
14/14 caught (both `test_bf16` and `test_gemm`).

## Ranked next levers

1. **H=64 dW folded into the fused backward** (opt-in, order change): the
   backward already holds each 16-row chunk's grad_combined in threadgroup
   memory; per-tile dW partials would drop the 6 MiB grad_combined write and
   the dW GEMM's 8 MiB read (Pong ~3 x 158 us, partly hidden by the overlap).
   Must be an env-gated, default-off variant with its own reference seeds.
2. **Muon wide chain latency** (Pong ~120 us/minibatch, 15 dependent tiny
   dispatches over three 192x64 matrices): fuse gram_buf into update (each
   update tile recomputes its 64x16 gram_buf slice) to drop 5 barriers, or a
   single persistent kernel per matrix if threadgroup memory allows BF16
   staging.
3. **Breakout H=32 `mingru_dw_reduce`** is memory bound (FP32 partials);
   halving partial traffic needs an order-preserving layout change.
4. **Forward scan / projection overlap within a threadgroup** - the
   warp-specialized attempt lost occupancy; a two-threadgroups-per-tile split
   with smaller chunks might not.
5. **Decoder dX small-K kernel** (Pong 46 us, K = 8) analogous to the
   small-N forward.

## Review follow-up

- Runtime window GEMMs now have disjoint outputs, both checked alongside
  producer/window/consumer ordering and exact barrier counts. Removing entry
  ordering fails values in all three repetitions; omitting exit ordering
  fails the barrier assertion. Restoring the runtime passes.
- New `test_bwd_overlap.cpp` runs through the existing GEMM gate in FP32/BF16.
  The real `arch_backward` (four H=64 layers, B=512, T=64) exhausts the real
  4 MiB workspace once per minibatch across three distinct minibatches,
  without clearing it between overlap runs. A test-only forwarding wrapper
  verifies exhaustion, retry at zero and bounds. Gradients are bit-exact
  against overlap disabled, dispatch counts agree, and pending work/offsets
  reset. Deleting `ov_split`'s exhaustion `concurrent_end` fails both
  precisions (`nested concurrent_begin`); restoring it passes both.
- The real `metal_dw_policy` is tested at zero, exactly 5%, adjacent doubles
  below/above 5%, and larger slack at two epoch scales. Flag and subsequent
  schedule assertions cover on->off and off->on transitions: 12 dispatches
  in either mode, 6 barriers with overlap versus 12 disabled. Unconditional
  `true` fails six flag/schedule assertions per precision; restoration passes
  both. This directly distinguishes the otherwise bit-identical schedules.
- Validation: `bash tests/metal/run_all.sh` **23/23 PASS**, including all eight
  H64 seed cases; separate `bash tests/metal/helpers/overlap_e2e.sh` **12/12
  exact** (seed-1 score/checkpoint SHA prefix, default/forced/disabled).
- No production bug was exposed; all production sources, kernel arithmetic
  and defaults are unchanged. Mutation logs and restored runs are under
  `build/leg17/review/` (untracked).

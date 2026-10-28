> Historical development log; last recorded at commit `5629e942` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

# Leg 23 (Opus): small-policy learner backward (H=32 / H=64)

Worktree `wt-opt23`, branch `metal-opt23`, base `b0f7bb84`, final `6cedf3ee`.
MacBook Air 15" M4 (fanless), 24 GB, macOS 26.4, AC power for every
measurement below. Nothing pushed or merged; the classic_snake snapshot in the
worktree stays untracked.

## 1. Summary

One idea shipped, default on and bit-exact: **an exact fma-corrected sigmoid**
(`puf_sigmoid_exact`, `src/metal/shaders/common.h`). The correctly rounded
IEEE divisions `1/(1+z)` and `z/(1+z)` become one hardware reciprocal
(`fast::divide(1, d)`) plus a Markstein fma correction:

```
z = exp(-|x|); d = 1 + z; n = x >= 0 ? 1 : z;
y = fast::divide(1, d); q = n * y;
return fma(fma(-q, d, n), y, q);
```

Over **all 2^32 fp32 inputs** (NaN included) it returns the bits of the old
formula in every translation unit that defines a sigmoid, so every seed
reference, golden and gate is unchanged. Every learner-bound env gets faster
except Pong FP32, which is neutral (+0.4%).

- `7cd17d5f` metal: exact fma-corrected sigmoid for the MinGRU learner kernels
  (`mg_sigmoid` in mingru.metal: H32/H64 fused forward and backward, also used
  by the generic paths), with the test and gate.
- `6cedf3ee` metal: exact fma-corrected sigmoid in the rollout and wide kernels
  (`puf_sigmoid` rollout.metal, `rollout_sigmoid` rollout_forward.h,
  `mg_sigmoid` wide.metal), with the test extended to those translation units.

`-DPUF_SIG_IEEE_DIV` compiles the old division back in (the test uses it as a
reference). Nothing order- or precision-changing was added, so no opt-in
switches or new references.

## 2. Profile (KPROF=2, async off, eval off; one run per arm, AC, clean GPU)

Top learner kernels, base `b0f7bb84` and final, avg us per call (calls per
run identical in both arms):

| Kernel | Breakout BF16 | Breakout FP32 | Pong BF16 | Pong FP32 |
|---|---:|---:|---:|---:|
| fused backward (H32 `_dw` / `_h64`) | 673 -> 561 (-17%) | 876 -> 809 (-8%) | 263 -> 241 (-8%) | 454 -> 442 (-3%) |
| fused forward (H32 / `_h64`) | 296 -> 211 (-29%) | 348 -> 282 (-19%) | 223 -> 192 (-14%) | 290 -> 282 (-3%) |
| rollout step (packed) | 200 -> 161 (-20%) | 172 -> 161 (-6%) | 82 -> 73 (-11%) | 70 -> 66 (-6%) |
| encoder / decoder dW (`puf_gemm_tn_sg16`) | 383 -> 375 | 501 -> 512 | 47 -> 46 | 53 -> 52 |
| `mingru_dw_reduce` | 186 -> 186 | 183 -> 187 | - | - |
| H64 dW (`tensor_ops_gemm_tn_split`) | - | - | 159 -> 159 | 310 -> 309 |
| **total GPU ms per run** | 745 -> 632 (-15%) | 851 -> 799 (-6%) | 665 -> 618 (-7%) | 953 -> 931 (-2%) |

Raw: `wt-opt23/build/leg23/profile/{base,final}_{breakout,pong}_{bf16,fp32}.out`
(Breakout 4M steps, Pong 1M).

After the change, Breakout BF16 per minibatch is: fused backward ~2.2 ms
(4 calls), forward ~0.84, encoder dW 0.75 (2 calls), dW reduce 0.74. Ablation
on the H32 backward (in-process microbench, B=2048, T=32, under a foreign GPU
load, so only the proportions matter): BF16 scan ~28%, dW partials ~15%,
dX ~27%; FP32 scan ~40%, dW ~21%, dX ~22%. The H64 FP32 kernels move about
20 MB per call and barely react to ALU savings: they are close to bandwidth
bound.

## 3. Microbench (in-process alternation, B=2048, T=32)

Measured while a browser kept ~60-69% of the GPU busy (see section 7), so
absolute numbers are about 2x leg 21's; the variants alternate every round
inside one process and the output hashes are identical:

| Kernel | base us | sigmoid us | change |
|---|---:|---:|---:|
| BF16 H32 fused forward | ~412 | ~337 | -18% |
| BF16 H32 fused backward + dW partials | ~800 | ~728 | -9% |
| FP32 H32 fused forward | ~486 | ~377 | -22% |
| FP32 H32 fused backward + dW partials | ~1105 | ~990 | -10% |

## 4. End-to-end A/B (cleanab, alternating, cooldown 10 s, base vs final)

| Cohort | n | Base median SPS | Final median SPS | Median | Paired median | Faster |
|---|---:|---:|---:|---:|---:|---:|
| Breakout BF16 | 6 | 6.2615M | 6.4683M | +3.30% | +3.97% | 5/6 |
| Breakout FP32 | 6 | 4.8730M | 5.2474M | +7.68% | +7.39% | 6/6 |
| Cartpole BF16 | 6 | 15.3602M | 16.2989M | +6.11% | +5.31% | 6/6 |
| Cartpole FP32 | 6 | 12.7242M | 13.9475M | +9.61% | +11.44% | 6/6 |
| Flappy BF16 | 8 | 7.0345M | 7.4990M | +6.60% | +6.59% | 8/8 |
| Flappy FP32 | 8 | 4.8707M | 5.0855M | +4.41% | +1.66% | 6/8 |
| Pong BF16 (2M) | 8 | 1.6661M | 1.7672M | +6.07% | +5.63% | 8/8 |
| Pong FP32 (2M) | 8 | 1.0556M | 1.0599M | +0.41% | +0.50% | 5/8 |
| Freeway BF16 (8.4M), under load | 6 | 0.8880M | 0.9265M | +4.34% | +4.95% | 6/6 |
| Freeway FP32 (8.4M), under load | 6 | 0.4282M | 0.4036M | -5.75% | +2.60% | 4/6 |
| snake BF16 (2.1M), under load | 6 | 0.2381M | 0.2497M | +4.85% | +2.65% | 4/6 |
| snake FP32 (2.1M), under load | 6 | 0.1790M | 0.1803M | +0.75% | +0.51% | 4/6 |

The Freeway and classic_snake controls (`ab/wide`) ran while the browser was
back at 57-83% of GPU time. They used relaxed guards (foreign 95%, bandwidth
floor 40/40), and absolute SPS is about 60% of leg 22's. The per-run SPS is
bimodal (load phases), so the Freeway FP32 median of -5.75% contradicts its
+2.60% paired median. Read these four rows as "no regression detected" only,
with low confidence. A first attempt with a 3% foreign limit stalled and was
aborted.

The arms are base `b0f7bb84` builds and the final tree's builds (the same
source as the A/B variant `c2`). Breakout, and Cartpole BF16, come from a clean rerun
(`ab/sig2`, foreign limit 3%, idle-first off, default bandwidth floors).
The other cohorts come from the first batch (`ab/sig`), which ran with relaxed
guards (foreign 95%, bandwidth floor 40 GB/s). These cohorts ran after the browser load ended (19:17): Cartpole FP32,
Pong, then Flappy. One minute sample during Cartpole FP32 still showed 7.5%
browser GPU time.
The first batch's Breakout and Cartpole BF16 cohorts ran fully under the browser load
(Breakout BF16 -3.7% median / 2 of 6 pairs faster, FP32 +12.6%, Cartpole BF16
-8.8%) and were discarded as noise and rerun. No absolute SPS is
pooled across batches. Raw: `wt-opt23/build/leg23/ab/{sig,sig2,wide}/`.

Deterministic proxies: dispatches, barriers and device bytes are unchanged
(same kernels and grids; KPROF call counts identical per run). The only change
is per-element ALU: an IEEE fp32 division sequence is replaced by one
reciprocal estimate, one multiply and two fmas.

## 5. Exactness and tests

- Seed 1, final tree: all eight references EXACT (Breakout FP32
  844.95672607421875/2ee4da82d356a44c, BF16 852.43658447265625/f5dc197b8d7fde55;
  Pong 20.99445915222168/5ee574ef2f33c6dd, 20.996614456176758/b1425713d9e71433;
  Flappy 51.408348083496094/4eafd59e547e43e1, 50.560173034667969/91886134a50625e1;
  Cartpole 200/bf0330663b50b01e, 199.96110534667969/b10ccdb69f45c996). The same
  held for the intermediate learner-only build.
- **Gates: 40/40 PASS** at `6cedf3ee` (`tests/metal/run_all.sh`: the 39 base gates
  including CUDA golden replays, wide_e2e, h64/overlap e2e, large_e2e, plus the
  new `sigmoid` gate). Log: `wt-opt23/build/leg23/gates_final.log`.
- New test `tests/metal/test_sigmoid.mm` + `tests/metal/run_test_sigmoid.sh`.
  Each translation unit compiles `sig_check_*` / `sig_dump_*` kernels (only
  under `-DPUF_TEST_SIGMOID`), so the shipped function is compared in its own
  compilation context: `test.metallib:mingru`, `:rollout`, `:rollout_fwd`,
  `test.rollout.metallib:rollout`, `test.wide.metallib:wide`.
  1. Exhaustive: all 2^32 fp32 bit patterns, shipped sigmoid vs the
     IEEE-division formula compiled alongside: 0 mismatches in all five units,
     NaNs included.
  2. Independent oracle: a CPU double-precision `1/(1+e)` / `e/(1+e)` (with the
     GPU's own `exp(-|x|)` as input, so only the division is judged) on
     329,814,117 strided samples, 164,907,059 of them negative: 0 differences.
  3. Mutations: `--mutate` (residual from a rounded product instead of an fma)
     gives 23,685,559 exhaustive mismatches plus 1,826,275 oracle differences;
     `--mutate2` (no correction step) gives 78,448,994 plus 6,046,681. Both FAIL;
     restored code passes. A full run takes about 22 s; `--quick` checks 5 of 16
     batches.
- A Newton refinement of the reciprocal before the correction was tried first
  and is not needed: the exhaustive check passes without it.

New gate commands:

```sh
bash tests/metal/run_test_sigmoid.sh            # gate `sigmoid` (exhaustive, ~22 s)
bash tests/metal/run_test_sigmoid.sh --mutate   # must FAIL
bash tests/metal/run_test_sigmoid.sh --mutate2  # must FAIL
```

## 6. Rejected (all exact unless noted; diffs in `wt-opt23/build/leg23/rejected/`)

| Idea | Result |
|---|---|
| Threadgroup-staged dW reduce (`dw_reduce_tg.diff`: 96 TGs x 256 threads, 64-tile chunks, same order) | Exact but slower (~360 vs ~240-274 us under load). The reduce streams 12 MB of fp32 partials and is bandwidth bound, as leg 21 found. |
| H32 grad_input staged in grad_x (`gin_stage.diff`, `PUF_METAL_GIN_STAGE`) | BF16 unchanged, FP32 -2% (noise). Reverted. |
| FP32 backward phase split (`bwd_split_and_ablation.diff`): all 128 threads precompute [h_tilde, z, s] plus a `simd_ballot` sign mask before the serial scan | ~1027 vs ~1015 us. No gain. |
| BF16 vectorized dX epilogue (`bwd_vec_epilogue.diff`): bf16(G@W) staged in `comb`, uint4 add with grad_input | 676 vs 679 us. No gain. |
| Encoder dW (`enc_dw_sg16_variants.diff`): sg16 TN with register-prefetch pipelining; K-staged cooperative loads (KS 32/64/128) with float4 or scalar TG stores; BF16 with `red` aliased onto As/Bs (4 KiB TG) | Clean GPU, M=32 N=118 K=65536: FP32 base 905 us, KS128 831 (-8%), KS64 855, scalar-store variants the same; BF16 base 675, KS128 1127, KS64 869, aliased 677. All hash-identical. Only FP32 improves, worth about 0.15 ms per minibatch (~1% of Breakout FP32 GPU time), which doesn't pay for a new shared-GEMM kernel with tests and Freeway/snake A/Bs. Load-only and MMA-only ablations (365 / 360 of 1090 us under load) show poor overlap, which points at occupancy. |

## 7. Measurement caveats

- For most of the session a browser helper kept 60-69% of GPU time
  (WindowServer ~15%), and `l3_bw` read 62-68 GB/s instead of the usual >80.
  To get any cohorts at all, the local untracked `build/cleanab.py` gained
  `PUF_CLEANAB_MIN_BW` (pre-run floor, default 80) and `PUF_CLEANAB_MIN_BW_POST`
  (default 60). Defaults are unchanged; the first batch used 40/40. The load
  ended around 19:17, and the cohorts reported in section 4 were rerun or run
  after that. A background `gpuwatch.sh` logged the top GPU clients every
  minute (`build/leg23/gpuwatch.log`).
- The microbench in section 3 and the rejected-idea timings (except encoder dW)
  were measured under that load. They are good for direction within one
  process, not for absolute values.
- Exactness depends on the reciprocal estimate's behavior on this
  GPU/compiler (M4, macOS 26.4 Metal toolchain). The Markstein step makes
  the result correctly rounded whenever the estimate is within an ulp, which
  is what the exhaustive check confirms here. On other Apple GPUs or a future
  compiler, the `sigmoid` gate will catch any change. If it fails,
  `-DPUF_SIG_IEEE_DIV` restores the old code.

## 8. Ranked next levers

1. **H64 FP32 backward and forward** (Pong FP32: 442 + 282 us per call, 56%
   of GPU time; ALU savings barely moved it). Cut bytes. For example, recompute
   `scan_h` from the minibatch's immutable `train_state` plus `combined`
   instead of re-reading it, or stash fewer FP32 intermediates between forward
   and backward. Must stay bit-identical (same fp ops, only where values come
   from).
2. **H32 FP32 backward scan** (~40% of an 809 us backward). Same idea: the
   per-step reads dominate. The 24 KiB FP32 TG tile limits occupancy to about
   2 TGs per core, and half the threads idle during the serial scan.
3. **Encoder dW occupancy** (375 / 512 us x2 per minibatch on Breakout). The
   ablation shows load and MMA phases that don't overlap. Splitting into more
   TGs with the same 32-way K partition (same partial boundaries) is the
   remaining exact option; FP32 KS128 staging is a ready -8% if someone
   bundles it with other GEMM work (`enc_dw_sg16_variants.diff`).
4. **dW reduce** (186 us x4 per minibatch on Breakout): bandwidth bound on
   12 MB of fp32 partials. Only an order-changing (opt-in) fused reduction or
   fewer partials would help.
5. **Rollout step** (161 us per step on Breakout): now the second-largest item
   on Breakout BF16 after the backward.

## Portability

The bit-exact claims above apply only to exhaustive validation on Apple M4
with the macOS 26.4 and 27 compilers. MSL's reciprocal accuracy bounds do
**not** guarantee correct rounding after the fma correction, even when the
estimate is within one ULP; the contrary statement in section 7 is incorrect.
For example, at `x = 2^-23`, `z = 0x3f7ffffe` and `d = 0x3fffffff`, the
conforming estimate `y = 0.5` produces corrected `0x3f000000`, whereas IEEE
division produces `0x3f000001`. Metal `exp` also depends on the device and
compiler, so cross-device bit identity was never guaranteed.

The fma-corrected sigmoid remains the default. The sigmoid gate reports bit
mismatch counts and maximum ULP error over all 2^32 inputs, requiring at most
one ULP against that device's IEEE-division sigmoid. `PUF_SIG_REQUIRE_EXACT=1`
requires bit equality instead and is set by `tests/metal/run_all.sh` for the
validated M4. NaN outputs must remain NaNs; only exact mode requires matching
NaN payloads and zero signs. `-DPUF_SIG_IEEE_DIV` restores IEEE division.

The sampled CPU oracle uses the GPU's own `z` and rounded denominator,
independently validates the IEEE reference, and accepts RN/RZ and permitted
subnormal flushing (including either zero sign), as specified in sections
8.1, 8.2 and 8.5 of Apple's [Metal Shading Language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).
`--mutate3` injects exactly two ULP at signed zero and must fail both gate
modes; `--mutate` and `--mutate2` retain the original mutation checks.

Review validation: both full gate modes pass all five shader targets with
zero bit mismatches and max ULP 0. Quick mutation checks reject both original
mutations in exact mode and the two-ULP mutation in both modes. The
rounded-product mutation stays within one ULP
in that sample and therefore passes portable mode, while removing the
correction fails its sampled oracle. Run the original mutation commands
above with `PUF_SIG_REQUIRE_EXACT=1` to require rejection. Breakout seed 1,
checked via the gate runner's `tests/metal/helpers/seedrun.sh` helper, still matches
both BF16 and FP32 score/checkpoint references exactly. No timing
measurements were taken for this review.

> Historical development log; last recorded at commit `42f830a9` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

# Leg 22: exact H512/H1024 learner paths

Machine: Mac mini M4, 10 GPU cores, 16 GiB unified memory, macOS 27.0.1,
Xcode 27.1. Branch `metal-opt22`, base `d606b131`. All figures below are
from this machine; no laptop SPS figures are used. Builds export
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.

## Outcome

Final production revision: `ca945bd2`. Six-pair clean A/B cohorts show median
SPS gains of **130.43% / 254.31%** for Tetris BF16 / FP32 and
**79.21% / 130.73%** for WEF BF16 / FP32, with all six pairs faster in every
target cohort. The twelve control cohorts range from -0.58% to +1.71%.
All **35 final gates pass**, including the required small seed references,
wide_e2e, the new H512/H1024 native checks, and 51 caught mutations.
The earlier `3377ab68` candidate was rejected for a WEF BF16 regression; the
revision and its measured cause are documented below.

The kept implementation has three independent default-on switches. Setting
any to `0` restores that component's previous implementation:

- `PUF_METAL_GEMM_LARGE_TN`: recurrent H512/H1024 TN tiles and ragged encoder
  TN partials. The original dispatcher chooses split count, K chunk, workspace
  layout and reduction first; only the spatial computation within each partial
  changes. Unsplit products retain the full ascending-K accumulator.
- `PUF_METAL_MUON_LARGE`: admit H1024 recurrent matrices to the per-matrix
  Muon batch plan. Unaligned large gradients are copied into aligned scratch
  for normalization and all five NS steps, then copied back bit-for-bit.
  Already-aligned matrices incur neither extra allocation nor copies.
  Gram-buffer/update use full-K spatial tensor products with the original
  `alpha*acc`, then `fma(beta, old, ...)`, then storage rounding. The Gram
  kernel and product kernels preserve one-output-per-thread scalar fallbacks
  for smaller matrices sharing a batch. Norm trees remain per matrix.
- `PUF_METAL_GEMM_LARGE_LINEAR`: recurrent NT and residual dX, at M>=4096,
  use 64x64 output tiles with 128 threads. BF16 dX still computes
  `bf16(bf16(A@B)+D)`. Smaller shapes retain their previous dispatches.

Primary and rollout metallibs are unchanged byte-for-byte. New entries live
in the auxiliary wide library. No precision/order-changing opt-in is added.

## Baseline and workload choice

Ran the supplied gate suite before profiling or editing production code.
The supplied runner initially had a stale 31st `proj_store` entry pointing to
an absent test, and `env_golden` required missing CMake. Removed the stale
local runner entry, installed CMake 4.4.4, built the ignored fast-nle dependency,
and reran env_golden successfully: all 30 intended baseline gates passed.
The other worktree's suite reached `GATES DONE` before the first profile or
benchmark. No process was killed or reniced. Baseline evidence is under
`build/leg22/{gates_base.out,base_env_golden_rerun.log,verify_base.out}`.

Chosen environments use default geometry and need no external data:

| Environment | H / layers | Agents / horizon | Minibatch / updates per epoch | Observation |
|---|---:|---:|---:|---:|
| Tetris | 512 / 2 | 4096 / 128 | 65536 / 21 | 234 |
| WEF | 1024 / 4 | 512 / 32 | 4096 / 8 | 110 |

Tetris's configured layer count 2.20258 is cast to 2. WEF is a bundled
continuous-action environment with cheap CPU stepping relative to its learner.
Both build in FP32 and BF16 and fit comfortably in 16 GiB. Boxoban needs
external levels absent here. g2048 builds, but its default five-layer,
65536-minibatch H1024 FP32 learner exceeds the measured Metal working-set
budget. Its nine recurrent activation tensors per layer alone require
11.25 GiB (`5*9*65536*1024*4`). Adding only the global recurrent input-gradient
buffer, encoder output, recurrent weights, recurrent gradients and momentum
already reaches **11.9258 GiB**, versus this device's **11.8400 GiB** reported
budget, before rollout, other parameters, optimizer scratch and replay data.
It is not a viable default FP32 benchmark under this stock budget. This is
an allocation lower bound; no OOM was induced. Evidence:
`build/leg22/metal_memory_budget.txt`, `encoder_reg_train`, `mingru_reg_train`
and `muon_init` in `src/algo.cu`.

## Profile before changes

`PUF_METAL_KPROF=2`, `--base.async=0`, evaluation disabled, default geometry;
Tetris 1,048,576 steps and WEF 131,072 steps. Profile instrumentation serializes
GPU timing, so these are attribution measurements, not accepted SPS comparisons.
Raw logs: `build/leg22/profile/{tetris,wef}_{bf16,f32}.out`.

| Workload | CPU env ms/epoch | Learner ms/epoch | Peak footprint, bytes |
|---|---:|---:|---:|
| Tetris BF16 | 9.362 | 10,670 | 2,193,606,768 |
| Tetris FP32 | 9.747 | 26,640 | 4,202,843,640 |
| WEF BF16 | 7.742 | 3,019 | 739,968,176 |
| WEF FP32 | 11.86 | 6,751 | 1,116,145,104 |

Both are learner-bound. Recurrent kernel cost per layer per minibatch (ms):

| Workload | dW TN | Forward NT | dX NN+add | Scan forward | Scan backward |
|---|---:|---:|---:|---:|---:|
| Tetris BF16 | 143.475 | 27.516 | 27.236 | 4.442 | 7.408 |
| Tetris FP32 | 386.613 | 40.855 | 40.593 | 8.687 | 14.567 |
| WEF BF16 | 9.537 | 7.280 | 7.245 | 0.531 | 0.896 |
| WEF FP32 | 47.317 | 9.649 | 23.795 | 1.063 | 1.796 |

Tetris encoder dW costs 15.314/15.928 ms (BF16/FP32); encoder NT costs
4.801/6.853 ms. Its existing three-matrix Muon batch costs 1.175/2.315 ms
for Gram, 1.882/2.107 ms for Gram-buffer, and 5.565/6.601 ms for update,
per NS stage. WEF FP32 generic recurrent Muon Gram/Gram-buffer/update cost
15.380/2.009/6.345 ms each. WEF BF16 update costs 5.879 ms; aligned Gram
3.521 ms. Its generic scalar GEMM aggregate mixes unaligned Gram and
Gram-buffer, averaging 3.453 ms; those cannot be separated by name alone.
The BF16 WEF flat gradient has an eight-byte decoder-logstd offset, so even
NS steps are unaligned. FP32's corresponding offset is sixteen bytes.

## Tuning, rejected work, and the Muon revision

Only spatial tiles and thread counts were searched. K traversal, split
partitions, accumulation dtype and storage boundaries were fixed. Every
candidate had to pass independent numerical checks before timing.

| Search | Valid / tried | Time | Selection |
|---|---:|---:|---|
| Initial TN | 18/18 | 277.17 s | H512 FP32 64x128/64; H512 BF16 64x64/64; H1024 64x32/64 |
| Extended TN | 7/8 | 117.14 s | No improvement; 64x256/64 was numerically wrong |
| NT/dX | 18/18 | 192.24 s | 64x64/128, both widths and precisions |
| Initial mixed-batch Gram | 16/16 | 65.74 s | Superseded after native profiling |
| Production-shaped Gram | 24/24 | 167.59 s | FP32 64x64/64; BF16 64x32/64 |
| Extended Gram | 8/8 | 53.66 s | No improvement |
| Production-shaped products | 24/24 | 152.62 s | BF16 64x32/64 |
| Extended products | 8/8 | 50.75 s | FP32 128x64/128 |
| Larger product boundary | 10/12 | 75.42 s | No improvement; 512x32/128 numerically wrong in both precisions |

Notation is output rows x columns / threads. Ragged encoder TN uses
64x64/64. Tiny per-shape wins were not used to add more dispatch branches.
All new learner kernels fit the M4 limits; no new scan threadgroup allocation
is shipped. The incorrect large tiles were excluded, not treated as harmless
floating-point variation. Their compiler/backend failure cause was not resolved.

Reproduction commands (raw JSON and logs stay in the named output directories):

```sh
python3 tools/metal/tune_large.py --output build/leg22/tune_tn
python3 tools/metal/tune_large.py --extended --output build/leg22/tune_tn_extended
python3 tools/metal/tune_large.py --linear --output build/leg22/tune_linear
python3 tools/metal/tune_large_stages.py muon --output build/leg22/tune_muon_production
python3 tools/metal/tune_large_stages.py muon --extended --output build/leg22/tune_muon_extended
python3 tools/metal/tune_large_stages.py products --output build/leg22/tune_muon_products
python3 tools/metal/tune_large_stages.py products --extended --output build/leg22/tune_products_extended
python3 tools/metal/tune_large_stages.py products --boundary --output build/leg22/tune_products_boundary
```

The final Muon tuner measures wrapper-only production-shaped batches:
Tetris's encoder plus two 1536x512 matrices; WEF's encoder, decoder and four
3072x1024 matrices, including the eight-byte BF16 logstd offset. It first runs
the broader independent mixed/ragged reference tests. Reference dispatches
are excluded from timing. The initial mixed benchmark had only one large tall
H1024 matrix and did not represent WEF's four-matrix batch or the real encoder.
It selected a mathematically exact but slow implementation.

The first combined candidate (`3377ab68`) therefore **was not retained as the
final result**. Six clean pairs per row, saved baseline versus that candidate:

| Workload | Baseline median SPS | Candidate median SPS | Median change | Paired median change | Faster pairs |
|---|---:|---:|---:|---:|---:|
| Tetris BF16 | 50,351.5 | 107,488.5 | +113.48% | +113.47% | 6/6 |
| Tetris FP32 | 20,660.5 | 69,220.5 | +235.04% | +235.35% | 6/6 |
| WEF BF16 | 5,498.5 | 2,744.5 | -50.09% | -50.09% | 0/6 |
| WEF FP32 | 2,392.5 | 4,070.5 | +70.14% | +70.12% | 6/6 |

These are separate intermediate cohorts in `build/leg22/ab/`, not pooled
with final measurements. Tetris used 1,048,576 steps; WEF used 131,072.
Scores/checkpoints were exact even in the rejected case. The native Muon-off
comparison identified the component; the subsequent synchronous profile
identified the stage:

| Intermediate profile | Gram ms/stage | Gram-buffer ms/stage | Update ms/stage | Learner ms/epoch |
|---|---:|---:|---:|---:|
| WEF BF16, initial candidate | 87.582 | 5.317 | 20.032 | 5,572 |
| WEF BF16, aligned staging | 11.304 | 5.135 | 18.785 | 2,178 |
| WEF FP32, initial candidate | 18.182 | 11.975 | 37.246 | 3,868 |

Aligned staging fixed the unaligned Gram cost without changing any bits.
It passed independent stage/wrapper references and native score/full-SHA
checks before the next revision. Then the product sweep addressed the
remaining expensive Gram-buffer/update stages. Finally, one-output-per-thread
fallbacks removed the smaller encoder's loss of concurrency: the original
64x64/64-thread Gram fallback serialized up to 64 outputs per thread.
Evidence: `profile_final/` (the intermediate candidate), `profile_staged/`,
`staged_e2e_wef_*.out`, and `muon_before_tiles/`, all under `build/leg22/`.
Final combined attribution is in the next section.

NT/dX isolated screening timings, original -> selected (ms):

| Shape / precision | NT | dX |
|---|---:|---:|
| M65536 H512 FP32 | 39.515 -> 30.231 | 38.814 -> 29.295 |
| M65536 H512 BF16 | 27.492 -> 27.101 | 27.496 -> 26.817 |
| M4096 H1024 FP32 | 9.411 -> 7.518 | 16.040 -> 7.394 |
| M4096 H1024 BF16 | 6.833 -> 6.798 | 6.922 -> 6.790 |

These screening samples are not cleanab SPS claims.

A sliced MinGRU forward prototype split H across threadgroups with at most
32 KiB scratch, full-K projection and exact per-unit scan/carry. All 26
candidates passed combined/scan/output/state comparisons, including terminal
resets and carried horizons; six store/carry mutations failed and restored
versions passed. Backward stayed generic because dX depends on all hidden
slices. The prototype was never connected to the policy and is not shipped.

Against the retiled ordinary projection, slice128/chunk16/128 threads took
43.657 vs 38.904 ms at H512 FP32 and 18.776 vs 8.570 ms at H1024 FP32
(fused vs projection+scan). BF16 H1024 took 7.953 vs 7.315 ms; its earlier
best chunk32 variant was only 1.4% ahead of the old projection+scan. H512
BF16 alone saved about 2 ms (29.179 vs 31.233), a small fraction of learner
time. This did not justify adding a precision-specific scan path. Archive:
`build/leg22/rejected_mingru/`, sweep `tune_mingru/`, mutation log
`mg_mutations.log`. This rejection has isolated evidence, not a native
cleanab speedup claim.

## Final profile and remaining cost

Same synchronous KPROF setup and step counts as the baseline. These are
attribution measurements, not the cleanab SPS results below. Raw logs:
`build/leg22/profile_release/{tetris,wef}_{bf16,f32}.out`.

| Workload | CPU env ms/epoch | Learner ms/epoch | Peak footprint, bytes |
|---|---:|---:|---:|
| Tetris BF16 | 9.373 | 4,774 | 2,204,878,984 |
| Tetris FP32 | 9.713 | 7,276 | 4,241,411,600 |
| WEF BF16 | 7.457 | 1,641 | 818,038,008 |
| WEF FP32 | 8.442 | 2,433 | 1,246,561,768 |

Final recurrent kernel cost per layer/minibatch (ms):

| Workload | dW TN | Forward NT | dX NN+add | Scan forward | Scan backward |
|---|---:|---:|---:|---:|---:|
| Tetris BF16 | 31.075 | 27.967 | 27.680 | 4.234 | 7.093 |
| Tetris FP32 | 68.398 | 32.205 | 31.370 | 8.199 | 13.723 |
| WEF BF16 | 7.272 | 7.249 | 7.236 | 0.515 | 0.860 |
| WEF FP32 | 10.479 | 8.254 | 13.854 | 1.001 | 1.686 |

The unchanged scan kernels are shown for context; their timing variation is
not attributed to a scan optimization. Tetris encoder dW falls to
4.952/9.379 ms (BF16/FP32), with original split partitions and reduction.

Final Muon times are **per whole batch per NS stage** (Tetris three matrices,
WEF six), unlike the baseline WEF per-matrix generic costs:

| Workload | Gram ms | Gram-buffer ms | Update ms |
|---|---:|---:|---:|
| Tetris BF16 | 0.710 | 0.248 | 0.653 |
| Tetris FP32 | 0.788 | 0.301 | 0.814 |
| WEF BF16 | 10.824 | 2.502 | 7.422 |
| WEF FP32 | 16.689 | 3.387 | 10.697 |

The isolated production-shaped sweeps and native profiles are kept separate.
Real learner buffers and surrounding kernels change cache/working-set behavior;
primitive timing is not substituted for end-to-end measurement.

## Controlled end-to-end measurements

Final target and control cohorts are in `build/leg22/ab_final/`. Each row is
its own six-pair cohort; no absolute SPS is pooled across rows or batches.

Target workloads:

| Workload | Baseline median SPS | Release median SPS | Median change | Paired median change | Faster pairs |
|---|---:|---:|---:|---:|---:|
| tetris BF16 | 50,369.5 | 116,068.0 | +130.43% | +130.31% | 6/6 |
| tetris FP32 | 20,699.5 | 73,341.0 | +254.31% | +255.00% | 6/6 |
| wef BF16 | 5,545.5 | 9,938.0 | +79.21% | +79.05% | 6/6 |
| wef FP32 | 2,414.5 | 5,571.0 | +130.73% | +130.64% | 6/6 |

Shared-code controls:

| Workload | Baseline median SPS | Release median SPS | Median change | Paired median change | Faster pairs |
|---|---:|---:|---:|---:|---:|
| breakout BF16 | 5,611,146.5 | 5,707,112.5 | +1.71% | +1.91% | 5/6 |
| breakout FP32 | 4,936,596.5 | 4,907,829.0 | -0.58% | -0.45% | 1/6 |
| cartpole BF16 | 14,521,073.5 | 14,489,469.0 | -0.22% | -0.06% | 3/6 |
| cartpole FP32 | 13,976,204.5 | 13,962,737.5 | -0.10% | -0.15% | 3/6 |
| pong BF16 | 1,811,587.0 | 1,812,393.0 | +0.04% | +0.03% | 4/6 |
| pong FP32 | 1,102,274.0 | 1,101,707.0 | -0.05% | -0.08% | 2/6 |
| flappy BF16 | 7,566,535.5 | 7,603,639.5 | +0.49% | +0.41% | 6/6 |
| flappy FP32 | 5,266,029.5 | 5,265,245.0 | -0.01% | -0.14% | 2/6 |
| freeway BF16 | 1,532,521.0 | 1,533,423.0 | +0.06% | +0.06% | 4/6 |
| freeway FP32 | 1,110,799.0 | 1,112,775.5 | +0.18% | +0.18% | 5/6 |
| snake BF16 | 573,710.0 | 573,864.0 | +0.03% | -0.04% | 3/6 |
| snake FP32 | 405,608.5 | 405,669.0 | +0.01% | +0.12% | 4/6 |

No substantial control regression was observed. The worst median loss was
Breakout FP32 at -0.58% (-0.45% paired, 1/6 pairs faster); this small measured
loss is retained rather than rounded to zero. Other negative medians are
between -0.22% and -0.01%. These cohorts do not establish zero performance
change. All accepted runs used a foreign-GPU limit of 1%; the largest accepted
fraction was 0.91%. Raw summaries are tabulated in
`build/leg22/ab_final_summary.json`. Each of the sixteen cohorts contains six
accepted pairs, for 192 accepted runs total.

`build/cleanab.py` and `build/l3_bw` copied from tools-local. The foreign-GPU
guard was checked against real per-process ioreg nanosecond counters: a rollout
test recorded about 678 ms GPU work and WindowServer 2.66 ms in a one-second
sample (`build/leg22/guard_check.txt`). Accepted cohorts use six alternating
pairs, ten-second cooldown before each arm, GPU idle wait and bandwidth guard:

```sh
export PUF_CLEANAB_IDLE_FIRST=1 PUF_CLEANAB_MAX_FOREIGN=0.01
export PUF_CLEANAB_COOLDOWN=10 PUF_CLEANAB_ALTERNATE_ORDER=1
unset PUF_CLEANAB_RELAX_AFTER
python3 build/cleanab.py ABS_OUTDIR 6 STEPS ABS_BASE/puffer ABS_FINAL/puffer
```

The local helper copy received one reporting-only fix: print `n/a` rather
than divide by zero when a baseline timing field is zero. The initial-candidate Tetris
BF16 cohort saved all twelve accepted samples before hitting that formatting
error; those samples were retained, and remaining cohorts resumed separately.
Sampling, guard thresholds and acceptance logic were not changed. The guard
rejected attempts exceeding 1% foreign GPU time, including two attempts for
the final Tetris FP32 release sample in pair six (1.4% and 1.1%). The third
attempt passed at 0.5%. Rejected raw logs are retained.

Final target timing uses Tetris 1,048,576 steps and WEF 65,536 steps,
with unchanged default geometry. The six control environments use 20 million
steps for Breakout/Cartpole/Flappy, 2 million for Pong, 8,388,608 for Freeway,
and 2,097,152 for classic_snake, in both precisions.

Evaluation is disabled for timing; the runtime excludes epoch zero from its
summary. Tetris therefore reports one measured epoch after warmup and WEF
reports three. Every cohort compares saved `d606b131`
binaries with the combined candidate, using default model and minibatch
geometry. Median SPS and median within-pair ratios are reported separately;
absolute SPS is never pooled across batches. No builds, profiles, tests or
other benchmarks run concurrently with accepted measurements.

## Exactness and gates

The final suite passed **35/35 gates** at production revision `ca945bd2`: all
30 original gates and five additions. All eight required small seed references
match, as does `wide_e2e`. The new native gate passed twenty runs (five modes
for each of four configurations); the separate release comparison added twelve
runs against generic/default/saved baseline. All scores and full checkpoint
SHA256 values match. All 27 new mutations were caught and restored versions
passed; all 24 existing wide mutations were caught.

Evidence: `build/leg22/gates_final.out`, `build/leg22/verify_release.out`,
`build/{h64_e2e,overlap_e2e,wide_e2e,large_e2e,wide_mutations,large_mutations}.log`.
The final audit also verifies all 192 accepted A/B runs, original primary and
rollout metallib hashes, unchanged snake snapshot hashes and clean production
sources after mutations. Only the two report files change after this tested
production revision.

Independent references cover all new entry points:

- TN: original dispatcher with the new option disabled, every split partial,
  ascending-FMA CPU samples, output poisoning and allocation canaries.
  H512/H1024 recurrent shapes, K64/65/2064/4096/65536, ragged widths
  109/110/111 and 233/234/235, deferred workspace offsets and opt-out.
- Muon: original normalization and GEMM/copy/store pipeline, every Gram,
  Gram-buffer and update in all five NS stages, aligned and unaligned
  matrices, both orientations, mixed old/new/ragged sizes, production wrapper
  selection counts and opt-out. Original mixed batches include 1536x512,
  3072x1024, 1024x3072, 1024x1024, 513x512 and 129x128. Additional cases
  528x512 and 1024x1040 exercise fast tiles with ragged output edges;
  512x234 and 1024x110 cover the actual encoders. Short scratch must reject
  without dispatching work. A queued-work test verifies twelve GPU staging
  copies, avoiding the runtime's idle-stream CPU memcpy shortcut.
- Linear: original projection and NN+add pipelines plus independent CPU
  ascending-FMA samples, BF16 double rounding, poisoning and guards,
  H512/H1024, M64/4096/4160 and generic M4097 fallback, opt-out.
- Mutation suite: eight TN, fourteen Muon (six shader variants, two stage
  selectors, six staging copy/selection mutations), five linear (including
  BF16 rounding): **27 new mutations**. Each mutant must fail for the intended
  reason, then the original must pass. Existing 24 wide mutations remain.
- Native short seed-1 runs: Tetris 1,048,576 steps, WEF 65,536, 128 evaluation
  episodes/agents. Compare generic all-off, default, each component off and
  the saved baseline. Exact full checkpoint SHA256 plus score is required.

These full reference values were established with saved `d606b131` binaries.
All 24 initial-candidate native runs matched them. All twelve final release
runs (generic, default and saved baseline for four configurations) also
matched exactly. The final suite also passed every individual component opt-out.

| Environment / precision | Score | Full checkpoint SHA256 |
|---|---:|---|
| tetris_bf16 | 27.340425491333008 | `6ceed0713c2937f49f62aa05d29bba07fdf7bd1499b992b72a900f9870b8be74` |
| tetris_f32 | 27.677595138549805 | `c2ba1c5f371cf70d0e280ed131edde06395448248c7130f60e77e0be4f7b9e04` |
| wef_bf16 | -0.3426969051361084 | `6708c9fca4b82ef9c5ac5a0bb72a09eac6e231ddfea987948ad5c4db5d30008c` |
| wef_f32 | 1.0028723478317261 | `e08030a0e8ae1877799391c188b262a460d0fbdfc1cf73815cdf2f6dd98de267` |

Initial logs: `build/leg22/final_e2e_{tetris,wef}_{bf16,f32}.out`; release logs:
`build/leg22/release_e2e_{tetris,wef}_{bf16,f32}.out`. Underlying native
logs/checkpoints are in the printed `build/metal/large_e2e.*` directories.

The original small seed-1 references checked by `h64_e2e` and `overlap_e2e`:

| Environment | FP32 score / SHA prefix | BF16 score / SHA prefix |
|---|---|---|
| Breakout | 844.95672607421875 / `2ee4da82d356a44c` | 852.43658447265625 / `f5dc197b8d7fde55` |
| Pong | 20.99445915222168 / `5ee574ef2f33c6dd` | 20.996614456176758 / `b1425713d9e71433` |
| Flappy | 51.408348083496094 / `4eafd59e547e43e1` | 50.560173034667969 / `91886134a50625e1` |
| Cartpole | 200 / `bf0330663b50b01e` | 199.96110534667969 / `b10ccdb69f45c996` |

Five added gates are registered in the supplied local `tests/metal/run_all.sh`;
the tracked aggregate driver is `tests/metal/run_leg22_gates.sh`:

```sh
bash tests/metal/run_test_large_gemm.sh
bash tests/metal/run_test_large_muon.sh
bash tests/metal/run_test_large_linear.sh
python3 tests/metal/mutate_large.py
bash tests/metal/run_test_large_e2e.sh
```

## Deterministic proxies and memory

TN and linear retain dispatch/barrier counts, unique
operand bytes and split workspace/reduction passes. Spatial tile counts change;
no savings in unique bytes are claimed from cache reuse. Muon retains all
normalization and intermediate storage rounding. WEF's four H1024 recurrent
matrices now join the existing encoder/decoder batch. FP32 removes 928 learner
dispatches and runtime barriers per epoch (116 per minibatch); BF16 removes
864 (108 per minibatch) after accounting for aligned staging. There are eight
minibatches per epoch. Measured dispatch counts are 1385 -> 457 for WEF FP32
and 1393 -> 529 for BF16. Tetris remains at 954 FP32 / 975 BF16 dispatches.

The selected recurrent dW tile launch counts per layer/minibatch are:

| H / storage | Original groups × threads | Selected groups × threads |
|---|---:|---:|
| 512 FP32 | 384 × 128 | 96 × 64 |
| 512 BF16 | 384 × 128 | 192 × 64 |
| 1024 either | 1536 × 128 | 1536 × 64 |

NT/dX change 64x32/128-thread output tiles to 64x64/128-thread tiles,
halving threadgroup counts at the selected shapes. Ragged encoder partials
keep the same 64x64 spatial tile and original K partitions, but launch 64
rather than 256 threads per tile. These are launch proxies; instruction
count, register spilling and cache traffic are not inferred from them.

WEF removes five Gram-buffer and five update destination-initialization copies per recurrent matrix:
`5 * 4 layers * 8 minibatches * (1024^2 + 3072*1024) * 2 * sizeof(precision)`
= 5 GiB FP32 / 2.5 GiB BF16 logical read+write traffic per epoch before
staging. BF16 staging adds 768 MiB read+write traffic, leaving **1.75 GiB net
saved per epoch**; aligned FP32 retains the full **5 GiB** saving. These are
logical traffic proxies, not measured DRAM bandwidth. Concurrent per-matrix
Muon scratch adds approximately 80 MiB FP32; BF16 adds 40 MiB for batching
plus 24 MiB for aligned staging, or **64 MiB**, plus 4 KiB bookkeeping.
Already-batched Tetris adds neither staging nor batch scratch. No new scan
threadgroup allocation is shipped.

## Ranked next levers and limits

1. H1024 Muon Gram remains costly in native batches: 10.824 ms BF16 and
   16.689 ms FP32 per NS stage. Investigate cache reuse and matrix scheduling
   while retaining each full-K accumulation and per-matrix norm tree. Larger
   spatial tiles in this sweep stopped paying off; two extreme tiles failed
   exactness and need compiler-level investigation before reconsideration.
2. Tetris recurrent dW remains the largest individual learner kernel at
   31.075/68.398 ms per layer/minibatch (BF16/FP32). New split partitions
   would violate this default exactness contract. Further work should target
   operand reuse or launch scheduling with the existing partitions.
3. H1024 FP32 dX remains 13.854 ms per layer/minibatch in the native profile,
   despite faster isolated timing. Investigate native memory/cache behavior
   before adding another tile branch.
4. Wide recurrent fusion needs a different memory ownership plan, especially
   backward dX across hidden slices. The tested forward-only path saved too
   little in BF16 H512 and lost elsewhere; it is lower priority than the
   measured GEMM/Muon costs.

These are next hypotheses, not demonstrated gains. The bounded, exact spatial
sweeps and native revision cycle above are the stopping point for this leg.

Full default-length training has not been claimed. Tetris defaults to about
1.924 billion steps; WEF to 75 million. Short native runs exercise exact
checkpoint/score agreement, while primitive tests compare every relevant
intermediate. Coverage of other H512/H1024 environment-specific encoder and
decoder shapes is not implied by the two target workloads.

## Commits and reproduction

Production commits, in order:

- `14b129e9`: exact H512/H1024 TN tiles and ragged split partials.
- `a1e56026`: initial large Muon batch eligibility and Gram fragments.
- `3377ab68`: exact recurrent projection and residual dX tiles.
- `d0ae1cd6`: aligned scratch staging for unaligned large Muon inputs.
- `ca945bd2`: tiled large Muon products and scalar fallback concurrency.

The final behavior includes the last two revisions; the intermediate Muon
candidate is not the recommended endpoint. All commits are on `metal-opt22`;
no push or merge was performed. The classic_snake snapshot remains untracked.

Build a target with `./build.sh ENV DIR/puffer --metal [--float]`, give `DIR`
config/resources symlinks, then run the independent native reference check:

```sh
python3 tests/metal/large_e2e.py DIR/puffer ENV --reference ABS_BASE/puffer
bash tests/metal/run_leg22_gates.sh
bash tests/metal/run_all.sh
```

`ENV` is `tetris` or `wef`. Omitting `--quick` exercises each component opt-out
as well as all-off and default. The tracked gate script builds both precisions
and both targets. The complete supplied suite is local/ignored; the five new
gate commands and all their test sources are tracked.

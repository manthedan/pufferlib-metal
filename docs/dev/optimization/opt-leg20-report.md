> Historical development log; last recorded at commit `a2e93a5a` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

# Optimization relay leg 20 (Astra): H=128/256 MinGRU tier

Worktree `wt-opt20`, branch `metal-opt20`, baseline `3ef39b79`. No push,
merge, branch switch, or writes to another worktree. The supplied
`config/classic_snake.ini` and `ocean/classic_snake/` remain untracked.
Machine: MacBook Air M4, macOS 26.4.1 (25E253), Xcode Metal 4 / SDK 26.5.
Every build exports `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.

Untracked snapshot SHA256 (recorded only; the files are never staged):

- `config/classic_snake.ini`: `99ac4be2cc34a7983168897a472b7535859bae00067200aac5da129d0e0ffcfa`
- `ocean/classic_snake/classic_snake.h`: `cd794a5b2a1c3a48dde3ca9342383d7667300acbb2c066d57ef6473eca6af606`

## Implementation commits

- `38a39417`: exact wide TN tiles (unsplit and fixed split-K), ragged N,
  lazy wide library, independent GEMM/partial references.
- `9822bef2`: templated H128/H256 MinGRU, checked-in tile table, offline
  tuner and exact recurrent tests.
- `c1c841dc`: wide Muon fragments, intermediate reference tests, mutation
  suite and end-to-end wide seed gates.

## Profile first

Both supplied snake builds succeeded. Snake retains H256/L2, obs1008,
4096 agents, two buffers, T64, minibatch8192. Freeway was selected for
H128: its default 16384 agents, H128/L4, T64, minibatch16384 give ample
learner work; its CPU environment is cheap. Enduro's 256-agent default
would instead emphasize actor turnaround. No policy or training geometry
was changed for the benchmark.

Baseline KPROF=2, async=0, evaluation disabled, 2,097,152 steps. Times in
microseconds per minibatch; recurrent rows are **per layer**. KPROF
serializes dispatches and is attribution, not throughput evidence.
Raw profiles: `build/leg20/profile/{classic_snake,freeway}_{bf16,f32}.out`.

| learner item | Snake BF16 | Snake FP32 | Freeway BF16 | Freeway FP32 |
|---|---:|---:|---:|---:|
| recurrent projection NT | 1,020 | 1,401 | 562 | 755 |
| forward scan | 264 | 504 | 263 | 509 |
| backward scan | 438 | 848 | 430 | 828 |
| dX + highway add | 1,000 | 1,360 | 526 | 717 |
| recurrent dW TN / split TN | 1,591 | 5,501 | 577 | 2,361 |
| encoder forward | 1,337 | 1,934 | 126 | 117 |
| encoder dW | 4,736 | 4,910 | 192 | 259 |
| Muon Gram (all wide matrices, each of 5 stages) | 118 | 151 | 81 | 85 |
| Muon Gram-buffer (all wide matrices, each stage) | 375 | 340 | 72 | 66 |
| Muon update (all wide matrices, each stage) | 1,248 | 1,178 | 199 | 185 |

Snake has 32 minibatches/epoch, Freeway 64. Snake's encoder dW is exactly
the predicted scalar unsplit fallback, (M,N,K)=(256,1008,8192).
Freeway's scalar CPU env costs ~33–35 ms per synchronous epoch versus
~799/1,521 ms learner time (BF16/FP32). Snake costs ~15 ms env versus
~789/1,156 ms learner. Both are suitable learner targets.

Async traces used 8,388,608 steps and default geometry. Snake's baseline
learner/epoch is ~904/906 ms BF16 and ~1339/1343 ms FP32; Freeway BF16
~970/976 ms. These diagnostic runs are not pooled with clean benchmarks.
The first Freeway FP32 async trace overlapped an exploratory microbenchmark;
it is excluded from timing claims. Trace CSVs and `trace_timeline.py`
summaries are beside the profiles. No rollout kernel was changed.

## Kept implementation

1. **Wide TN tensor-ops GEMM.** The final scalar fallback
   for dense TN, M128/256, N>=32 and N%32!=0, K>0 divisible by16,
   alpha1/beta0, aligned pointers changes. Earlier SG/split decisions are
   untouched for ragged shapes. Masked edge stores and bounded operand tensors
   retain the full ascending-K chain. `PUF_METAL_GEMM_RAGGED=0` restores it.
   The same spatial tiles also replace ordinary aligned TN at M=3N,
   N128/256, and the H128 split-K partial kernel. Split counts, K chunk
   boundaries, workspace offsets, deferred reductions and alpha/beta
   finishing are unchanged. `PUF_METAL_GEMM_WIDE_TN=0` restores aligned
   TN; `GemmOpts.allow_wide_tn=false` provides an independent test oracle.
2. **Templated H128/H256 learner forward/backward.** One threadgroup owns
   64 whole-sequence rows. Chunked projections feed the serial per-unit
   scan from threadgroup memory. Combined is still saved for backward;
   projection recomputation is avoided. The backward keeps grad_combined
   and the highway gradient on chip for dX, but writes grad_combined for
   the original dW GEMM. Per-step BF16 state rounding and the BF16 dX
   double-rounding boundary are unchanged. In-place grad_out/dX is safe
   because each group reads all its own chunk before writing it.
   `PUF_METAL_FUSED_MINGRU_WIDE=0` restores the generic layer; the existing
   global MinGRU opt-out still applies. Whole-stack fusion also permits
   the existing backward overlap policy to run for these widths.
3. **Wide Muon Gram-buffer/update fragments.** The existing ascending-K
   8x8 fragment method now covers min(R,C)=128/256 when both dimensions
   are multiples of16. Batched odd matrices retain the scalar fallback;
   min64 keeps the original arithmetic. All intermediate rounding and
   `ns_finish` are identical. `PUF_METAL_MUON_WIDE_MMA=0` restores the old
   stages. Neither stage count nor scratch size changes.
4. **Separate lazy wide library.** The new kernels live in
   `puffer.wide.metallib`, embedded as `__PUF_METAL,__wide`; standalone
   tests can use a sidecar or `PUF_METAL_WIDE_LIB`. It has an independent
   precision marker. Primary and grouped-rollout metallibs are
   byte-identical to baseline in both precisions. No wide entry point is
   loaded for small policies. `PUF_METAL_PIPELINE_INFO=1` reports actual
   pipeline thread and scratch limits.

## Offline tuning and rejected variants

`python3 tools/metal/tune_wide.py` builds candidate auxiliary libraries and
matching host dispatchers without editing defaults. Every candidate must
pass independent bit-exact references before its benchmark runs. The
checked-in table is `src/metal/shaders/wide_config.h`; shape class is
64-row tiles of whole sequences, T dividing64 and M divisible by64.

The initial sweep tried chunks4/8/16, 64/128 threads, output widths32/64/128,
skipping combinations above 32 KiB. Four-row chunks fail the installed
Metal primitive's static assertion (M must be a multiple of8 or16).
All 42 compiling candidates passed exactness. A second sweep added 256
threads and H256 output width256: 10 more exact candidates. Search time
was 140.15 + 30.56 seconds, including compilation, reference tests and
microbenchmarks. These diagnostic timings select candidates; clean
end-to-end cohorts decide retention.

| H | precision | benchmark (M,T) | chunk | threads | projection/dX tile N | forward scratch | backward scratch |
|---|---|---|---:|---:|---:|---:|---:|
|128|FP32|(16384,64)|8|128|128|12 KiB|16 KiB|
|128|BF16|(16384,64)|16|128|128|12 KiB|16 KiB|
|256|FP32|(8192,64)|8|256|256|24 KiB|32 KiB|
|256|BF16|(8192,64)|16|256|256|24 KiB|32 KiB|

The device reports 32 KiB threadgroup memory, SIMD width 32 and these
pipelines' maximum threads 1024. Register counts are not exposed by this
runtime query. I also serialized M4 pipeline binary archives and inspected
their build tables. The installed metal-objdump cannot initialize an
applegpu_g16g disassembler and exposes no decoded register counts there;
no measured register-count claim is made. Tile timings guided the occupancy tradeoff; hardware occupancy counters
were not measured, and fitting scratch alone was not treated as evidence
of good occupancy.

The initial H256 chunk8/128-thread/64-column prototype was slower than
generic (FP32 forward 1.96 vs 1.79ms, backward 2.89 vs 2.13ms). Selected wide
columns/256 threads recover it (forward 1.58 vs 1.79ms, backward 1.73 vs 2.13ms).
H128 FP32 selected forward/backward ~0.80/0.97ms versus 1.35/1.58ms generic;
BF16 ~0.65/0.72ms versus 1.00/0.97ms. H256 BF16 ~1.14/1.24ms versus
1.49/1.46ms in its selection run. Do not compare absolute times across
these separate diagnostic runs.

Ragged TN tuning tried output M16/32/64, N16/32/64, threads64/128: 18
variants in 53.84 seconds, all exact in both storage types. Winners for
snake: FP32 64x64/64threads (~1.88ms versus 5.01ms scalar), BF16
32x64/64threads (~1.41ms versus 4.72ms). The initial 64x32/128threads FP32
candidate regressed (~7.40 vs 4.91 ms) and was rejected. Raw tuning output:
`build/leg20/tune{,2,_ragged}.out` and candidate directories.

A follow-up exact test on snake's aligned recurrent TN (768,256,8192)
found FP32 5.84 -> 1.41ms, BF16 1.80 -> 1.05ms with the selected tiles.
This justified extending the same implementation beyond ragged encoder dW.
Tests compare every split FP32 partial as well as the final alpha/beta
result, including a nonzero deferred workspace offset and a short last
K chunk. Split-K partitioning itself is unchanged.

No split-K partition, accumulation order, precision, model, CPU env,
rollout ownership or default training configuration was changed. No
split-fp16 emulation or opt-in numerical variant was retained.

## Exactness and gates

Short seed 1 gates use default geometry, 2,097,152 steps for snake and
4,194,304 for Freeway, 128 evaluation episodes/agents. Default, all new
paths disabled, each path disabled individually, and the saved baseline
binary agree in score and full checkpoint SHA256.

| env | precision | score | SHA256 prefix |
|---|---|---:|---|
|snake|BF16|0.3309859037399292|3755cdf9ef43ac2b|
|snake|FP32|0.36666667461395264|51a58ac22caf7ee5|
|Freeway|BF16|0.0703125|415d06d1cdbf5895|
|Freeway|FP32|0.0859375|91459c402a7a088e|

These short scores are exactness references, not convergence claims.
Evaluation runs only after training. The native evaluator rewrites the
saved INI's agent count to 128; training still uses 4096 Snake /16384
Freeway agents, verified by the training output and profiling shapes.
The 128-agent evaluation setting is not a training-geometry override.
The e2e gate compares trained bytes and verifies fewer learner dispatches,
so an eligibility mutation that silently disables fusion cannot pass it.
On a clean checkout without the untracked snake snapshot, the permanent
H256 gate uses Breakout's standard dense policy at H256/L2/T64/minibatch8192.

New gate commands (also added to `tests/metal/run_all.sh`):

```
tests/metal/run_test_wide_gemm.sh
tests/metal/run_test_wide.sh
tests/metal/run_test_wide_muon.sh
python3 tests/metal/mutate_wide.py
tests/metal/run_test_wide_e2e.sh
```

The GEMM test checks H128/H256, N1007/1008/1009, actual snake K8192,
NaN destinations, guards, the original scalar kernel and sampled CPU
ascending-fma references. MinGRU tests cover every supported T, terminals,
multiple tiles, carried horizons, optional grad_input, in-place dX,
rejected shapes/alignments/aliasing and index overflow. Muon tests check
every intermediate over five stages with tall/wide/square/mixed/odd
matrices, both ping-pong directions, fresh NaNs and guards. Independent
wrong-precision and missing-library opt-out tests cover lazy loading.

All **30 gates passed at implementation `c1c841dc`**, after correcting
execute permissions on the two supplied untracked h64/overlap gate scripts
and rerunning them. The first runner log records those two launch failures
(126), not numerical failures. Logs: `build/leg20/gates.out`,
`h64_e2e_rerun.log`, `overlap_e2e_rerun.log`; per-gate logs are in `build/`.
Final documentation-head verification uses `build/leg20/gates_final.sh`.
It requires 30 distinct PASS results, no failures, an unchanged HEAD and
no tracked source modifications. The log and exact commit are recorded
in `build/leg20/gates_final.out` and `build/leg20/gates_final_head.txt`.

All **20/20 mutations were caught**, without modifying production sources:
GEMM and split stores, both forward/backward H128/H256 entries, both Muon
stages, and forward/backward cross-chunk carries, across both precisions.

All eight required seed 1 references remain exact (default and the existing
H64/overlap opt-out/forced modes):

| env | FP32 score / SHA prefix | BF16 score / SHA prefix |
|---|---|---|
|Breakout|844.95672607421875 / 2ee4da82d356a44c|852.43658447265625 / f5dc197b8d7fde55|
|Pong|20.99445915222168 / 5ee574ef2f33c6dd|20.996614456176758 / b1425713d9e71433|
|Flappy|51.408348083496094 / 4eafd59e547e43e1|50.560173034667969 / 91886134a50625e1|
|Cartpole|200 / bf0330663b50b01e|199.96110534667969 / b10ccdb69f45c996|

## Final isolated profile

Same KPROF procedure; per layer-minibatch, microseconds. Final attribution
runs are separate from baseline timing batches and from clean A/B.

| item | Snake BF16 | Snake FP32 | Freeway BF16 | Freeway FP32 |
|---|---:|---:|---:|---:|
| fused forward |1177|1756|763|1022|
| fused backward + dX |1269|1971|824|1180|
| recurrent TN / split TN |1098|1555|597|786|
| encoder dW |1467|2074|207|331|
| Muon Gram-buffer, each stage |68|82|20|24|
| Muon update, each stage |232|272|52|63|

Freeway encoder dW retains the original SG split path (not the new
ragged path); its measured time is included above. Separate serialized
profile batches can move even unchanged kernels, so these values are
attribution, not controlled speedup measurements.

The new wide pipelines report SIMD32, maximum 1024 threads, and the scratch
sizes in the tile table. GEMM and Muon variants use zero static threadgroup
memory. The residual H256 forward/backward remains compute-intensive;
there is no H32-style projection recomputation or fused dW order change.

## Measurement results

All clean cohorts use
`build/cleanab.py`, n>=6, alternating order, 10-second cooldown, pre/post
bandwidth guards and sampled foreign-GPU accounting. Following leg19's
desktop-load workaround: `PUF_CLEANAB_IDLE_FIRST=0`,
`PUF_CLEANAB_MAX_FOREIGN=0.03`. The bandwidth quiet-wait remains enabled;
strict zero-utilization idle-first is not used. No builds, tests or other
GPU work overlap accepted benchmark trials. Absolute SPS is never pooled
across batches. The final cohorts contain 180 accepted trials; sampled
foreign-GPU fractions span 0.0000–0.0124. No trial required a retry.

Final implementation versus saved `3ef39b79` binaries, n=6 per row.
Binaries are `build/leg20/{base,final}/ENV_PREC/puffer`; baseline builds
come from a git archive inside this worktree. Exact cohort commands and
component-arm specs are in `build/leg20/measure_final.sh`.
Snake uses 2,097,152 steps; Freeway uses 8,388,608. The supplied helper
keeps default base seed 73 (selfplay is disabled). Exactness and full-budget
validation explicitly use seed 1. Values below are SPS medians;
paired change is the median of the six within-cycle ratios.

| workload | baseline → final SPS | median change | paired change | faster pairs |
|---|---:|---:|---:|---:|
| Snake BF16 |284,070.5 → 489,933|+72.47%|+72.09%|6/6|
| Snake FP32 |198,907.5 → 360,707|+81.34%|+81.40%|6/6|
| Freeway BF16 |1,098,143.5 → 1,366,237|+24.41%|+24.22%|6/6|
| Freeway FP32 |571,611 → 859,407.5|+50.35%|+50.35%|6/6|

Component arms share those same alternating cohorts. Everything else is
final; each comparison disables just the named component.

| component enabled | workload | median change | paired change | faster pairs |
|---|---|---:|---:|---:|
| wide MinGRU |Snake BF16|+5.44%|+4.47%|5/6|
| wide MinGRU |Snake FP32|+2.26%|+2.26%|6/6|
| wide MinGRU |Freeway BF16|+12.13%|+12.80%|6/6|
| wide MinGRU |Freeway FP32|+12.33%|+12.64%|6/6|
| recurrent TN tile |Freeway BF16|+6.18%|+6.24%|6/6|
| recurrent TN tile |Freeway FP32|+32.79%|+34.05%|6/6|

Freeway FP32 shows a throughput decline consistent with thermal drift:
final SPS runs from 995,282 to 826,770 across the six cycles (baseline 581,140 to 560,833). All accepted
trials remain included. No pooled absolute SPS or inference of sustained
speedup is made from this short cohort. Full-budget validation is recorded
separately below. The serialized BF16 split-kernel profile looked slightly
slower, but its repeated end-to-end opt-out comparison supports retention.

Raw accepted summaries, all stdout/stderr and guard values:
`build/leg20/final_ab_{ENV}_{bf16,f32}/`. An earlier candidate, before the
aligned TN extension, has separate `build/leg20/ab_*` cohorts: paired
Snake gains +59.42/+22.75%, Freeway +16.28/+13.80% (BF16/FP32), 6/6 each.
Those results are not pooled with the final cohorts.

Small-policy controls use the same n=6 alternating protocol, 20M steps
except Pong at 2M. Every paired median is between -0.37% and +0.51%.
These small signed differences are reported as observed, not claimed
as improvements or proof of mathematically zero performance change.

| workload | baseline → final SPS | median change | paired change | faster pairs |
|---|---:|---:|---:|---:|
|Breakout BF16|5,423,795.0 → 5,462,293.5|+0.71%|+0.51%|5/6|
|Breakout FP32|4,816,180.0 → 4,810,703.5|-0.11%|-0.16%|2/6|
|Pong BF16|1,721,288.5 → 1,720,623.0|-0.04%|+0.00%|3/6|
|Pong FP32|1,093,017.5 → 1,089,243.5|-0.35%|-0.32%|2/6|
|Flappy BF16|7,213,545.0 → 7,210,596.5|-0.04%|+0.03%|4/6|
|Flappy FP32|5,167,796.5 → 5,184,226.0|+0.32%|+0.22%|5/6|
|Cartpole BF16|14,464,992.5 → 14,424,205.5|-0.28%|-0.37%|2/6|
|Cartpole FP32|13,868,784.5 → 13,867,086.0|-0.01%|-0.13%|3/6|

Small-policy learner/total dispatch and command-buffer medians are
unchanged apart from tiny existing adaptive-policy differences on
Breakout. Flappy FP32 and Cartpole show small barrier-count variation
from the existing adaptive policy; no new wide pipeline is selected.
Primary and grouped-rollout library bytes match baseline in both
precisions. The largest negative paired shift is Cartpole BF16's 0.37%.

Full-budget sustained validation used one on/off pair per workload,
with the final binary in both arms. Generic means all four new opt-outs
set to 0. Configured budgets are 200M Snake /1.2B Freeway; native training
rounds down to whole epochs. Seed 1, default training geometry, 128 final
evaluation episodes/agents, checkpoint interval 100000. `--sweep.downsample=64` increases saved curve resolution.
These single full-length pairs validate exactness and sustained behavior;
they are not an additional n=6 speedup cohort and are not pooled into the
short-run estimates.

All four pairs match final score and **complete checkpoint SHA256**
(prefix shown below). All saved numeric metrics are finite.

| workload | actual training steps | final score, both arms | SHA256 prefix, both arms |
|---|---:|---:|---|
|Snake BF16|199,753,728|68.10526275634766|`d59be507d2abcca7`|
|Snake FP32|199,753,728|62.40625|`720325ccc1a98712`|
|Freeway BF16|1,199,570,944|21.3359375|`ca82f0fbeb6be7a9`|
|Freeway FP32|1,199,570,944|21.34375|`b4bb35cb064b9109`|

Thermal diagnostic: median SPS from the first/last 100 training dashboard
samples after epoch 2, excluding evaluation and duplicate epochs. Values
below are kSPS and inherit the dashboard's display rounding. Every
optimized late rate remains above its generic counterpart.

| workload | generic early → late kSPS | optimized early → late kSPS |
|---|---:|---:|
|Snake BF16|233.2 → 206.3|469.7 → 362.3|
|Snake FP32|152.2 → 143.7|264.4 → 234.0|
|Freeway BF16|825.3 → 730.8|1000.0 → 947.1|
|Freeway FP32|455.3 → 445.1|661.9 → 616.6|

**Logging limitation found during validation:** native saved INI metrics
are based on episode-completion history. `log_history_bin_mean` in
`src/pufferl.cu` fills trailing empty bins with its last sample. Freeway
FP32's optimized file repeats epoch 1115 and 2129.4 ms across its last 16
bins despite training reaching epoch 1144 with dashboard SPS near 620K.
Those repeated bins are not a valid late-time window. The table above
therefore uses training dashboard samples for all eight runs. Original
INI-derived fields are retained in the JSON artifacts for audit; they are
not used for sustained claims. This existing logger behavior is unchanged.

Full-run guard settings: 30-second cooldown, bandwidth quiet-wait
(pre>=80GB/s, post>=60GB/s), idle-first off, sampled foreign-GPU limit 3%.
All eight runs passed their guards on the first attempt; observed foreign
fractions were 0.0020–0.0040 and bandwidth checks
91.4–93.2GB/s. No other GPU test or build overlapped them.
The full validation driver is `build/leg20/full_validate.py`; complete
commands, stdout, metrics, checkpoints and per-arm/per-pair JSON results
remain under `build/leg20/full/`, with progress in
`build/leg20/full_validate.out`.

## Deterministic proxies

The ragged and Muon changes preserve dispatch count, command buffers,
barriers and allocated intermediates. Their unique operand/output byte
sets are unchanged; tile reuse changes device load multiplicity and cache
traffic, which is not claimed as a measured DRAM byte reduction.

MinGRU removes two dispatches per layer-minibatch (128 per snake epoch,
512 per Freeway epoch). It removes 3MH projection reads, 3MH gradient reads,
and 2MH highway-gradient writes/reads, at the storage element size, per
layer-minibatch. That is 32 MiB BF16 /64 MiB FP32 for both benchmark
minibatches, or 2/4 GiB per snake epoch and 8/16 GiB per Freeway epoch of
logical device traffic. Buffer capacity is unchanged. Additional barrier
reductions come from the already-existing overlap policy now becoming
eligible; no synchronization rule was weakened.

Observed counters, baseline → final, per reported epoch in the short
cohorts (the last empty rollout makes some totals fractional):

| workload | learner dispatches | total dispatches | barriers | command buffers |
|---|---:|---:|---:|---:|
| Snake BF16 |1449 → 1321|2877 → 2749|2877 → 2621|115.7 → 115.7|
| Snake FP32 |1417 → 1289|2735 → 2607|2735 → 2479|115.7 → 115.7|
| Freeway BF16 |3849 → 3337|8022 → 7510|8022 → 6999|235.4 → 233.4|
| Freeway FP32 |3785 → 3273|7738 → 7226|7738 → 6714|234.4 → 232.4|

The Freeway command-buffer reduction is from the existing adaptive
backward policy becoming eligible with fusion. Aligned and split TN
changes preserve the original split workspace and all reduction passes,
so their dispatch/barrier/command-buffer/unique-byte deltas are zero,
just as for the ragged tile and Muon stages.

## Ranked next levers

1. **Recurrent projection/backward tiling at H128/H256.** The selected
   fused forward plus backward still costs about 3.73 ms per Snake layer
   and 2.20 ms per Freeway layer in the FP32 isolated profile. Investigate
   weight-fragment reuse or a different sequence-row schedule while
   preserving each reduction and BF16 store boundary. H256 backward is
   already at 32 KiB; increasing the chunk is not a viable FP32 knob.
2. **Snake encoder NT and residual wide TN.** Encoder forward and dW each
   remain roughly 2 ms per minibatch in FP32. Ragged dW now reaches tensor
   ops, but this leg did not tune encoder NT or change split-K partitions.
   A larger shape-specific spatial search is preferable to changing
   numerical order.
3. **H512/H1024 need a different memory plan.** Even the smallest tested
   legal FP32 chunk at H512 would exceed the current backward scratch
   allowance. Do not extend the width whitelist without a new staging
   strategy and independent exactness tests.
4. **Rollout and Muon follow measured demand.** Wide rollout is unchanged;
   learner time still dominates these two default workloads. The original
   Muon Gram stage remains a candidate, but the costly Gram-buffer/update
   chains have already been shortened here. Re-profile a different default
   environment before choosing either as the next relay target.

The CUDA reference was useful for the ideas of weight reuse and layer
folding, but its policy math was not copied. Split-fp16 FP32 emulation,
new split-K partitions, and layer folding were not implemented. The kept
changes all retain the original mathematical and rounding contracts.

## Review follow-up

Addressed both Sol P2 test findings without changing production code,
kernel arithmetic, defaults or seed references; no production bug was exposed.

- Added mixed batches containing `(129,128)` and `(256,257)`, which match
  the wide Muon width whitelist but require the divisibility fallback.
  Every intermediate remains bit-exact against the independent pre-leg20
  stages over five iterations and both ping-pong directions. Byte-exact
  canaries surround every logical allocation and are checked before poison
  is refreshed, including input buffers. Removing both `%16` guards corrupts
  the `(129,128)` trailing canary and leaves 256 incorrect `(256,257)` values
  on the first update, in both FP32 and BF16.
- Added production `muon_ns_wide_rm` tests for H128/H256 mixed batches in
  fresh flag-enabled/disabled processes. Existing KPROF kernel counts assert
  five dispatches of each wide stage when enabled and only original stages
  when disabled, accounting separately for the independent reference.
  Final results and guards also match. Permanently disabling `use_wide_mma`
  preserves numerical results but now fails the kernel-selection assertion.

`mutate_wide.py` now catches **24/24 mutations**, including both reviewer
mutations in FP32/BF16. Mutants use generated shader/host copies; each new
case fails for the intended reason and passes again with the original
library/binary. `bash tests/metal/run_all.sh`: **30/30 PASS**. All eight small-env
seed references and all four wide-policy scores/checkpoints remain exact;
wide full SHA256 values also match the saved pre-follow-up logs.

Evidence: `build/leg20/review_new_mutations.log`,
`build/leg20/review_gates.log`, `build/leg20/review_refs_check.log`, and
`build/metal/wide_mutations/` (individual failure/restoration logs).
The classic_snake snapshot remains unchanged and untracked.

> Historical development log; last recorded at commit `4df07db6` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

# Optimization relay leg 18 (Sol): learner critical path II

Branch `metal-opt18`, worktree `wt-opt18`, baseline `17db51e3` (leg 17
implementation `927c6cc6`, merged at `e1ac8645`). No push, merge, branch
switch, new worktree, or git identity override. Scratch builds, rejected
patches, profiles and raw trials are under `build/leg18/` (untracked).

## Outcome

**No production optimization is retained.** Two serious bit-preserving
attempts regress their targets: Muon Gram-buffer/update fusion loses on
Pong BF16; the blocked dW partial layout loses on Breakout BF16 and FP32.
This meets the relay's stopping criterion: neither lever delivers ~2%.
Production sources, defaults, shader payloads and host layout are restored
to `17db51e3`. The only committed changes are this report and the relay
entry. **23/23 gates PASS; all eight seed-1 references remain exact.**

The completed clean cohorts are reported separately below. Muon FP32's
incomplete wall-clock cohort is explicitly excluded; its profile also
shows higher kernel cost. There is no claim of a final throughput gain.

## Profile before optimizing

Saved all eight baseline binaries with `./build.sh ENV DIR/puffer --metal
[--float]`, alongside config/resources symlinks. Baseline source was clean.
KPROF attribution used `PUF_METAL_KPROF=2`, `--base.async=0`, 4M steps,
evaluation disabled (required to avoid the known KPROF/evaluation hang).
Raw files: `build/leg18/profile/ENV_PREC.out`. KPROF serializes dispatches;
its summed times are attribution, not an async speed claim. Shapes and
stream IDs are included. All times below are microseconds per minibatch
unless a multiplier is shown.

| learner item | Pong BF16 | Pong FP32 | Cartpole BF16 | Cartpole FP32 | Breakout BF16 | Breakout FP32 |
|---|---:|---:|---:|---:|---:|---:|
| recurrent forward | 3 × 225.32 | 3 × 277.91 | 2 × 172.42 | 2 × 159.65 | 4 × 412.26 | 4 × 419.66 |
| recurrent backward, including dX (and H32 dW partials) | 3 × 249.28 | 3 × 427.21 | 2 × 301.20 | 2 × 326.91 | 4 × 743.33 | 4 × 880.94 |
| H64 recurrent dW split GEMM | 3 × 151.82 | 3 × 308.22 | — | — | — | — |
| H64 recurrent dW reduction | 3 × 15.71 | 3 × 15.91 | — | — | — | — |
| H32 dW partial reduction | — | — | 2 × 32.73 | 2 × 38.53 | 4 × 186.04 | 4 × 183.90 |
| decoder forward, small-N | 30.52 | 61.43 | 14.29 | 19.52 | 51.52 | 114.19 |
| decoder dX | 46.35 | 31.05 | 40.49 | 22.19 | 95.78 | 72.76 |
| encoder forward | 51.43 | 31.76 | 41.47 | 23.66 | 226.51 | 410.62 |
| encoder dW | 48.87 | 54.35 | 47.88 | 62.10 | 616.33 | 817.35 |
| decoder dW | 41.94 | 44.81 | 43.39 | 41.65 | 77.87 | 123.17 |
| PPO loss + reduction + cache + advantage + decoder-gradient assembly | 46.02 | 44.92 | 58.13 | 56.09 | 110.67 | 129.64 |
| small-matrix Muon batch | 33.14 | 32.11 | 52.33 | 51.14 | 132.81 | 132.60 |
| wide-matrix Muon chain | 106.70 | 99.26 | — | — | — | — |

Pong wide Muon breaks down as five batched Gram / Gram-buffer / update
stages at **8.54 / 3.62 / 6.88 us BF16**, **7.07 / 3.72 / 6.78 us FP32**,
plus sum-square / normalize / store at 3.94 / 4.81 / 2.75 us BF16 and
3.86 / 4.77 / 2.78 us FP32. Those stages cover all three 192×64 recurrent
weights in one grid. The fifteen chain stages are already only about
95 / 88 us; eliminating five stages has limited headroom.

Pong learner dispatches are 776 / 760 per epoch, with 16 minibatches.
Cartpole has 200 / 192 and eight minibatches. Breakout has three
minibatches; its train-stream count depends on the existing decoder-dW
side policy. The H32 partial reduction moves 12 MiB per Breakout layer,
reading 1,024 FP32 partial tiles for each of 3,072 weights.

### Async command-buffer timeline

`PUF_METAL_TRACE=FILE`, 4M steps, evaluation disabled, default async and
policy choices; analyzed with `python3 tools/metal/trace_timeline.py FILE`.
The middle half of the epochs is reported. These are separate diagnostic
runs; do not compare their absolute SPS to a benchmark cohort.

| env / precision | epoch ms | learner CB spans ms | union of all CB spans ms | learner CBs/epoch |
|---|---:|---:|---:|---:|
| Pong BF16 | 41.63 | 38.82 | 39.38 | 6.9 |
| Pong FP32 | 62.11 | 60.63 | 60.81 | 3.0 |
| Cartpole BF16 | 10.25 | 8.76 | 9.23 | 1.0 |
| Cartpole FP32 | 10.50 | 8.91 | 9.31 | 1.0 |
| Breakout BF16 | 27.14 | 22.12 | 24.14 | 4.0 |
| Breakout FP32 | 32.28 | 25.95 | 27.82 | 4.0 |

Pong remains learner dominated. Its BF16 trace switches between epochs
with 256-dispatch CB chunks and epochs with per-minibatch side-stream
handoffs when rollout slack changes the existing policies. FP32 is three
large learner CBs, with approximately 67 us median inter-CB gaps. BF16's
median is about 133 us. Breakout's learner CB gaps are approximately
290 / 361 us, including its side-stream handoffs. Cartpole has one
learner CB per epoch, so submission gaps cannot explain most learner time.
Raw timelines and analyzer output remain in `build/leg18/profile/`.

## Attempt 1: fuse the Muon Gram-buffer/update chain

**Design.** Each 16×16 update tile recomputes its rounded 64×16
Gram-buffer slice in threadgroup memory, then consumes it for the update.
The ascending-K simdgroup fragment chains, coefficient finish, BF16
rounding, source ping-pong and scale store are unchanged. Eligibility is a
homogeneous batch with min(R,C)=64 and both dimensions divisible by 16;
other shapes retain the old chain. The prototype uses
`PUF_METAL_MUON_CHAIN=0/1` in one binary, avoiding host/library-layout
confounding in the A/B. No new entry point is introduced.

**Correctness.** The existing independent unfused Muon-tail comparisons
were extended with homogeneous tall, wide and misaligned M64 cases,
separately in FP32 and BF16. Both pass. The mixed-shape fallback still
passes. Trial patch: `build/leg18/muon_chain_trial.diff`.

**Profile.** The fused update costs **19.84 us BF16 / 18.04 us FP32**,
versus the original Gram-buffer + update at **10.50 / 10.50 us**. Repeated
slice computation outweighs the removed launch and barrier. It eliminates
five dispatches per minibatch, or **80 dispatches and barriers per Pong
epoch**, but increases work.

**Clean A/B.** Completed BF16 cohort: n=6, precision separated, 2M steps.
The same binary runs with the flag off and on, alternating order, 10 s
cooldown, per-trial idle, pre/post bandwidth checks, and strict <1% foreign
GPU guard; no builds overlap accepted trials. No accepted runs are removed.
All six pairs are slower.

| candidate / target | off M SPS | on M SPS | medians change | paired median | faster pairs |
|---|---:|---:|---:|---:|---:|
| Muon chain / Pong BF16 | 1.705852 | 1.670652 | -2.06% | -2.08% | 0/6 |

Measured learner GPU span rises 37.705→38.495 ms/epoch. Median learner
dispatches fall 776→696; total barriers fall 710.8→630.8. Commits are
66.76→65.76 (timing-dependent rollout/side-stream totals). The intended
dispatch reduction is real, and wall clock rejects it.
One accepted thermal excursion gives a -14.30% pair; the paired median
still loses 2.08%. Absolute SPS is never pooled with another cohort.

The FP32 cohort was stopped while our own cleanab process waited for GPU
idle after sustained foreign activity. Its three complete pairs and one
unmatched candidate, including rejected guard attempts, are preserved in
`ab_chain_fp32/`; they are diagnostic only. No FP32 wall-clock speed claim
uses this incomplete cohort. No other process was killed or reprioritized.

## Attempt 2: blocked H32 dW partial layout

**Design.** Change `(tile, weight)` storage to
`(weight / 32, tile, weight % 32)`. Each reducer warp reads its next 32
weights contiguously over consecutive partial tiles rather than striding
12 KiB between 256-byte slices. The backward computes identical 64-row
FP32 partials; every output still adds tiles from zero in ascending order.
The same existing backward and reducer entry points handle both layouts,
selected in one binary with `PUF_METAL_DW_LAYOUT=0/1`. No partials are
compressed and no accumulation order changes.

**Independent reference and mutation quality.**
`build/leg18/layout_test.sh` builds both precisions. It exercises T=1,2,8,
16,32,64 and 3,17,1,024 tiles, fresh poisoned scratch and output guards,
and compares each FP32 partial after host reindexing as well as final dW,
dX, highway and state gradients. An independent CPU loop sums the
original partials in ascending tile order and applies the final storage
rounding. Both precisions pass; the existing MinGRU suites pass too.

An initial reverse-order mutation survived the two-layout comparison,
because both modes shared the modified reducer. Adding the independent
CPU oracle fixes this false-pass. Final mutations: skipped partial store
and reversed eight-partial addition order, each in FP32 and BF16:
**4/4 caught**, followed by both restored references passing. Logs:
`mutation_layout_final.out`, `mutation_layout_*_PREC.out`,
`layout_reference_final.out`.

**Profile screen.** Same prototype binary, layout off/on, KPROF, 1M steps:
reduction **185.04→175.34 us BF16**, **183.56→178.08 us FP32**. Backward
partial production is 821.32→792.29 / 948.55→945.54 us in these short
profiles. Foreign GPU activity was present during this screen; these are
attribution only and the small deltas are not speed claims.

**Deterministic proxies.** With either policy pinned, dispatch, barrier
and command-buffer topology are unchanged. Partial bytes remain **12 MiB
written + 12 MiB read per layer**, or 144 MiB each direction per Breakout
epoch (four layers × three minibatches). A layout change alone does not
halve this traffic. Default policy choices remain automatic; their
per-epoch decoder-dW placement can change total stream counters with
timing, especially in FP32.

**Clean A/B**, same binary, layout off/on, precision-separated n=6,
20M steps, alternating order, 10 s cooldown, idle-first and strict <1%
foreign-GPU guard, bandwidth checks unchanged. No compilation or other
GPU tests overlap accepted trials. The initial foreign-busy attempts and
idle waits are retained. Our waiting benchmark was paused for the
23-gate suite; no layout trial had been accepted before that pause.
Accepted trials begin only after the suite ended and a quiet interval
became available. Absolute SPS is never pooled with another batch.

| target | off M SPS | on M SPS | medians change | paired median | faster pairs |
|---|---:|---:|---:|---:|---:|
| Breakout BF16 | 4.858283 | 4.583836 | -5.65% | -5.43% | 0/6 |
| Breakout FP32 | 4.416201 | 4.259937 | -3.54% | -3.53% | 0/6 |

BF16 learner GPU span is 22.100→22.745 ms/epoch; rollout GPU span is
17.44→19.06 ms (+9.3%). Dispatches remain 92 on the train stream,
barriers 130.8 and commits 39.78 per epoch in both arms. The isolated
reducer's small profile gain does not survive async execution. More
scattered producer stores and changed cache behavior are plausible
causes, but this was not isolated experimentally. The wall-clock result
is the reason for rejection, not that explanation.

FP32 learner GPU span rises 25.680→26.720 ms/epoch; rollout wall
29.34→30.42 ms. Its automatic side placement moves train dispatches
90.13→90.63, total barriers 124.35→122.90 and commits 38.65→38.15;
these are run-median epoch averages, not a new dispatch optimization.
The independent seed gate checks both overlap regimes. FP32 loses in
all six pairs too.

Raw cohorts: `ab_layout_bf16/`, `ab_layout_fp32/`; runner
`bash build/leg18/ab_layout.sh`; summary
`python3 build/leg18/table.py build/leg18/ab_layout_{bf16,fp32}`.
Rejected source: `build/leg18/dw_layout_trial.diff`. Counter tables use
medians of each run's `metal_summary` epoch averages. Because these
cohorts used a relative output directory, training `.ini` files live
under the binary directory's matching `build/leg18/ab_*/logs/` subtree;
raw stdout/stderr and accepted summaries live in the cohort directories
above. The older `build/leg17/proxies.py` expects absolute log paths and
was not used for the reported counter values.

### Measurement commands and final retained proxies

All completed cohorts use the same guard settings:

```bash
export PUF_CLEANAB_COOLDOWN=10 PUF_CLEANAB_IDLE_FIRST=1
export PUF_CLEANAB_MAX_FOREIGN=0.01 PUF_CLEANAB_ALTERNATE_ORDER=1
python3 build/cleanab.py build/leg18/ab_chain_bf16 6 2000000 \
  "$PWD/build/leg18/chain/pong_bf16/puffer@PUF_METAL_MUON_CHAIN=0" \
  "$PWD/build/leg18/chain/pong_bf16/puffer@PUF_METAL_MUON_CHAIN=1"
bash build/leg18/ab_layout.sh
```

Saved scripts/raw attempts remain intact, including every rejected
foreign-GPU attempt. No guard threshold was relaxed. There are **no kept
runtime changes**: final dispatches, barriers, command-buffer schedules
and bytes per epoch are the baseline's. The final binary-section audit
below also rules out an unused shader payload or host-layout change.

## Final validation

The restored production source tree equals `17db51e3` byte for byte
(`git diff --exit-code 17db51e3 -- src tests tools build.sh`).
**23/23 gates PASS** with `bash tests/metal/run_all.sh` on this final source
tree (`build/leg18/gates_final.out`). Documentation changes do not alter
that tested tree. All eight seed-1 references are exact; H64 also passes
the generic opt-outs, and H32 passes auto / forced / disabled overlap.

| env | BF16 score / SHA256 prefix | FP32 score / SHA256 prefix |
|---|---|---|
| Breakout | 852.43658447265625 / `f5dc197b8d7fde55` | 844.95672607421875 / `2ee4da82d356a44c` |
| Pong | 20.996614456176758 / `b1425713d9e71433` | 20.99445915222168 / `5ee574ef2f33c6dd` |
| Flappy | 50.560173034667969 / `91886134a50625e1` | 51.408348083496094 / `4eafd59e547e43e1` |
| Cartpole | 199.96110534667969 / `b10ccdb69f45c996` | 200 / `bf0330663b50b01e` |

`python3 build/leg18/audit.py` compares all eight saved baseline native
binaries with the native binaries rebuilt by the final end-to-end gates.
**8/8 have byte-identical host sections** (`__TEXT`, `__DATA`,
`__DATA_CONST`), primary shader payload and auxiliary rollout payload.
This checks Breakout and Flappy for code/layout regressions as well as
Pong and Cartpole. No final wall-clock performance delta is claimed for
identical retained code and payloads. Native UUID/load-command metadata
is outside this section audit.

Primary SHA256, the same for every env in its precision:

- BF16: `e7e037fef40552b420ab5c586bda7a22c45f877202c6c860197e1e4cdf7d8b90`
- FP32: `9e88a30c6aaf174ab50141953a77ae70594011b25d5df8b1a71069d7934a55f6`

The layout prototype separately passed all 12 Cartpole/Breakout
seed cases in `build/leg18/layout_overlap_e2e.out` (default, forced overlap,
overlap disabled, both precisions).

No new production kernel or entry point is retained, so **no new final
gate is added** to `tests/metal/run_all.sh`. The independent prototype gate
and mutation commands, after applying the saved rejected layout patch,
are:

```bash
git apply build/leg18/dw_layout_trial.diff
bash build/leg18/layout_test.sh
python3 build/leg18/mutate_layout.py
git apply -R build/leg18/dw_layout_trial.diff
```

These scratch tests are not part of the final default gate suite because
they require the rejected layout. They preserve the independent CPU
oracle and four caught mutations for reproduction.

## Ranked next levers

1. **Move the next pass to rollout ownership/completion transport** (leg
   16 lever 1): retain four original env workers while coalescing ready GPU
   requests without a second host handoff. This is a distinct hypothesis
   from another learner tile tweak; preserve per-buffer time, state, RNG
   and the actual buffer count.
2. **If returning to the Breakout learner, profile the ragged encoder dW
   separately.** It remains 616 / 817 us per minibatch BF16 / FP32. An
   order-preserving tensor/simdgroup path with the identical split count
   and per-split ascending-K chain has more theoretical headroom than the
   small dW-reducer layout change. Require independent partial-level
   comparisons before trusting an end-to-end match.
3. **Small-K decoder dX is a narrow opportunity.** Pong is 46 / 31 us per
   minibatch. Even removing it completely saves only 0.74 / 0.50 ms per
   epoch, about 2% / <1% before overhead; a real replacement saves less.
   Group this with a larger structural hypothesis if revisited.
4. **Forward projection/scan overlap needs an occupancy argument before
   another prototype.** Leg 17's warp specialization lost occupancy;
   whole-sequence row tiles must reduce latency while retaining enough
   independent threadgroups and the exact projection/scan order.
5. **H64 dW folding remains opt-in and lowest priority.** It changes the
   dW summation order. Any future version must be default-off, have an
   explicit flag, independent references/mutations and its own learning
   validation. No such variant was implemented in this leg.

**Recommendation: STOP this learner bottleneck pass.** The two measured
levers both regress, so stop under the relay's ~2% rule. Shift the next
leg to rollout ownership/completion transport. Reopen learner work only
with a distinct structural hypothesis and a fresh clean A/B; do not keep
these prototypes or infer a gain from their isolated reducer/dispatch
proxies.

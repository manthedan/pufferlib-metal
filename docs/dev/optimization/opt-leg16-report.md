> Historical development log; last recorded at commit `5b63c30d` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

# Optimization relay leg 16 (Sol): small-env rollout latency II

Branch `metal-opt16`, worktree `wt-opt16`, baseline **`51e0a5fb`**,
implementation **`e05a2301`**. Subsequent report edits do not change code.
Nothing pushed. **All eight required seed-1 references are exact; 22/22
gates PASS.** No arithmetic, buffer count or RNG assignment changed.
Evidence and rejected prototypes: `wt-opt16/build/leg16/`.

Final validation complete. Flappy and Cartpole improve; Breakout and Pong
show no measurable regression in either precision. Pong is neutral.

## Final clean A/B

`build/cleanab.py`, n=6 per binary/precision (Pong n=8), alternating order,
10 s cooldown, 20M steps per run. The final Pong check separates precisions
and uses 2M steps to reduce GPU duty cycle; the mixed 8M cohort is also
reported below. Every final trial waits for 0% GPU
utilization, uses the bandwidth/cooldown checks and a **1% foreign-GPU
limit**, stricter than the original 3% guard. Rejected attempts are saved,
never relaxed. No builds overlap these measurements. Absolute binary paths
under `build/leg16/{base,final2}/ENV_PREC`; raw results `ab_final2_ENV/`.
Only within-batch comparisons are used. M SPS, medians:

| env | precision | before | after | change | faster pairs |
|---|---|---:|---:|---:|---:|
| Flappy | BF16 | 5.367197 | 5.696133 | +6.13% | 6/6 |
| Flappy | FP32 | 4.179458 | 4.587650 | +9.77% | 6/6 |
| Cartpole | BF16 | 11.444216 | 12.594511 | +10.05% | 6/6 |
| Cartpole | FP32 | 10.503849 | 11.994515 | +14.19% | 6/6 |
| Breakout | BF16 | 4.874249 | 4.907227 | +0.68% | 6/6 |
| Breakout | FP32 | 4.365699 | 4.377209 | +0.26% | 4/6 |
| Pong | BF16 | 1.423906 | 1.424760 | +0.06% (flat) | 3/8 |
| Pong | FP32 | 0.975469 | 0.975421 | 0.00% (flat) | 5/8 |

Flappy paired medians: +6.07%/+9.89%; Cartpole +10.10%/+14.12%; Breakout
+0.68%/+0.24%. Pong paired medians are -0.049%/+0.049%: neutral.
`final2_table.py` verifies six accepted summaries per binary and guard limits;
Pong's eight accepted pairs are in `ab_pong_low_{bf16,fp32}`. Absolute SPS
is never pooled across cohorts.

## Retained changes

- **`dc7c1a12` — safe lazy progress words.** Unused multi-buffer async
  rollout streams opt into `stream_host_wait_only`. Shared-event signals,
  feedback checks and dependency verification remain; only the final
  progress-word barrier/blit is omitted. Any later cross-stream consumer
  performs a verified host wait before enqueueing work, including checking
  transitive waits that Metal may have silently released. No GPU consumer
  trusts a missing progress witness. The role is fixed before first enqueue.
  `PUF_METAL_LAZY_PROGRESS=0` restores progress writes.
- **`702b5b85`, `725fe79e` — paired rollout dispatches.** Grid Y selects
  separate `RolloutStepArgs` for the original buffers. Kernel arithmetic,
  state strides, output addresses, rounding and Philox arrays stay unchanged.
  A leader runs the pair's contiguous env range after combined GPU completion;
  its OpenMP team keeps the original total thread budget. Unused workers
  sleep. The actual buffer count stays four. Default group2 for eligible
  async rollouts; sync default stays group1. `PUF_METAL_ROLLOUT_GROUP=1`
  opts out; `=2/4` forces an eligible size. Single-buffer paths remain group1.
- **`e05a2301` — auxiliary shader library.** Grouped entry points load
  lazily from `__PUF_METAL,__rollgroup`; the primary `__metallib` is exactly
  baseline in both precisions. Native binaries still ship as one file.
  Standalone harnesses use `puffer.rollout.metallib` beside the executable;
  `PUF_METAL_ROLLOUT_LIB` overrides only this auxiliary for tests. Each
  library has its own precision check. This avoids disturbing existing
  pipelines when adding unused grouped kernels (see crossover below).
- **Test commits `9cb85446`, `499ad505`, `5aa889e3`.** Force held stores
  through the GPU, require dispatch reduction as well as matching model
  bytes, and mutation-test every new entry point separately.

`primary_audit.out` checks the embedded sections of all eight final native
binaries. Their primary SHA256 hashes equal baseline:

- BF16: `c42b1ae5591061f02a22c73db0171f365da389ba2794f898108445e8c3aae8bd`
- FP32: `599d2c8dea019c8da053ff5364a8ae41053566fef81df4ea47400b658e5b34c9`

## Timeline and critical path

Before editing, reproduced leg 15 with `PUF_METAL_TRACE`, 20M steps, BF16,
`tools/metal/trace_timeline.py`, middle 77 epochs. Initial idle check: 0%.

| baseline env | epoch ms | union of CB spans ms | learner CB spans ms | queue / GPU / wake p50 us |
|---|---:|---:|---:|---:|
| Flappy | 24.51 | 21.44 | 18.00 | 139.7 / 57.6 / 32.0 |
| Cartpole | 11.54 | 10.29 | 9.36 | 166.0 / 50.6 / 37.0 |

Flappy kernels are 87.2 us contended versus 49.7 uncontended; deferred
elapsed reads remain 0.1 us. Submission/wake latency and learner contention
remain the problem. Flappy issues two learner CBs (256+104 dispatches),
Cartpole one (200).

Pairing halves rollout commits: **256 -> 128 per Flappy epoch, 128 -> 64
Cartpole**, retaining all four original buffers. The pre-isolation paired
trace gives Flappy queue/GPU/wake 98.8/93.8/66.6 us and Cartpole
102.3/64.2/63.2. Each dispatch now processes two buffers, so spans are not
per-buffer timings. Four-buffer traces increase wake/env overhead further.

Final `e05a2301` traces were captured after the gate suite in a hotter
regime: Flappy 27.47 ms/epoch, union 25.64, learner 24.19, queue/GPU/wake
108.3/117.0/70.2 us; Cartpole 11.02, union 10.59, learner 10.50,
108.2/67.6/65.8 us. These are attribution, **not a cross-regime speed
comparison**; SPS claims use the clean cohorts above. Flappy retains
rollout slack beyond training; Cartpole rollout finish coincides with train
sync, making it learner bound. Traces: `traces/base_*`, `final_*`, `final2_*`.

## Experiments in evaluation order

Exploratory cohorts also use cleanab n=6, alternating, 10 s cooldown and
the 3% guard without relaxation. Final cohorts use the stricter protocol
above. No other process was killed or reprioritized; idle waits sometimes
took minutes. Values from different batches are never pooled as absolute SPS.

### 1. Safe lazy progress words — kept

20M steps, `ab_lazy_{flappy,cartpole}`:

| env | BF16 M SPS before -> after | FP32 M SPS before -> after |
|---|---:|---:|
| Flappy | 5.357277 -> 5.423244 (+1.23%, 6/6 faster) | 4.155775 -> 4.166215 (+0.25%, 3/6; flat) |
| Cartpole | 11.098750 -> 11.466215 (+3.31%, 6/6) | 10.251021 -> 10.590529 (+3.31%, 6/6) |

### 2. Same-bits multi-buffer fusion — group2 kept

10M steps, same-binary group1/2/4, `ab_group_{flappy,cartpole}`; lazy words
already enabled. Original buffer/RNG assignment stays fixed.

| env/precision | group1 M SPS | group2 | group4 | group2 gain |
|---|---:|---:|---:|---:|
| Flappy BF16 | 5.391496 | 5.704872 | 5.387360 | +5.81%, 6/6 |
| Flappy FP32 | 4.122162 | 4.533998 | 4.509541 | +9.99%, 6/6 |
| Cartpole BF16 | 11.582948 | 12.521639 | 11.784979 | +8.10%, 6/6 |
| Cartpole FP32 | 10.754747 | 12.003266 | 11.836987 | +11.61%, 6/6 |

Group4 BF16 is flat: Flappy -0.08% (3/6); Cartpole +1.74% median but
-0.06% median paired change (3/6). FP32 gains less than group2. Reject four
as the default. Grouping increases learner overlap: Flappy learner
18.235 -> 19.140 ms BF16, 25.950 -> 27.350 FP32, while epoch latency falls.

Smaller OpenMP teams, BF16, `ab_group_threads_*`: Flappy pair/4 threads
5.796458M -> pair/2 5.788597M (-0.14%, 1/6); Cartpole pair/8 12.518913M
-> pair/4 12.549826M (+0.25%, 4/6). Both flat, **rejected**. Smaller-team
group4 loses 3.42%/4.37% versus standard group2 (0/6 in both envs).

### 3. Predictive wake — timer-only prototype rejected

`ab_latency_flappy`, BF16, 10M steps. Per-stream timer estimate sleeps with
`mach_wait_until`, then retains the real event wait. No spinning or early
env access; held-output checks pass. This estimator learns host-observed
delay including its own sleep and is unsuitable at this scale.

| variant | M SPS | change | learner ms |
|---|---:|---:|---:|
| control | 5.719181 | — | 18.92 |
| 20 us lead | 2.654977 | -53.58%, 0/6 faster | 24.14 |
| 40 us lead | 2.653285 | -53.61%, 0/6 | 24.60 |

Ideal-timer trace replay suggested only 1.1/1.5 us saved per original
Flappy/Cartpole step and 4.6 us for paired Flappy; measured behavior is much
worse. No predictive code is retained. Leg 15's raw-wait spinning was not repeated.

### 4. Learner command-buffer chunking — rejected

Same cohort/control, learner-only cap 128/64 versus 256 dispatches:
5.679773M / 5.686633M (-0.69%/-0.57%, 3/6 faster each), learner
19.07/19.12 ms (+0.8%/+1.1%). No throughput win and the learner slows.
The installed SDK's MTL4CommandQueueDescriptor exposes label and
feedbackQueue, no public priority setting. Shorter CBs do not shorten the
individual learner kernels. Prototypes remain only under `experiments/`.

## Layout and library controls

The pre-isolation full A/B (`ab_final_*`) gains Flappy/Cartpole but gives
small mixed Breakout shifts (-0.23% BF16, +0.25% FP32). Checked layout before
blaming kernels: shader growth is isolated, data-segment base unchanged,
but new host statics move global totals by 64 bytes. The 56-byte counter
object crosses a 128-byte cache line in that build. A five-way BF16 n=6
control pins decoder dW side placement for all variants (`ab_layout_breakout`):

| host / library / counter layout | M SPS | vs baseline |
|---|---:|---:|
| baseline / baseline | 4.810498 | — |
| current / enlarged | 4.835298 | +0.52%, 3/6 faster |
| current / baseline | 4.831903 | +0.44%, 4/6 |
| baseline / enlarged | 4.802743 | -0.16%, 3/6 |
| current / enlarged / aligned counters | 4.811684 | +0.02%, 3/6 |

Current vs baseline median paired change is -0.04%: neutral. Shader
crossover changes little. Aligning counters loses 0.49% versus current,
so **rejected**, never committed. No counter-layout fix is justified.

Pong's first n=6 cohort crossed a broad regime change despite the 3% guard:
BF16 1.401418 -> 1.387138M (-1.02%; paired -0.263%); FP32
0.956055 -> 0.949299M (-0.71%; paired -0.395%). A stricter 1% n=6
crossover with per-trial idle (`ab_pong_quiet`) gives:

| precision | baseline host/library | current/enlarged library | current/baseline library |
|---|---:|---:|---:|
| BF16 | 1.408740 | 1.411419 (+0.19%) | 1.407995 (-0.05%) |
| FP32 | 0.970008 | 0.966256 (-0.39%) | 0.970188 (+0.02%) |

The baseline library beats the enlarged library in 5/6 FP32 pairs on the
same host (median +0.41%). Some enlarged-library runs are intermittently
low. Rather than accept that unrelated context change, `e05a2301` restores
the byte-identical primary and loads new kernels from the auxiliary only
when grouped execution is requested. No arithmetic changes. All final
performance cohorts are repeated after this fix; earlier evidence is retained.

The final isolated-library mixed 8M Pong cohort (`ab_final2_pong`) still
crosses a nonstationary GPU regime with negligible foreign work: BF16
1.366379 -> 1.355730M (-0.78%, paired -0.213%); FP32
0.948318 -> 0.963256M (+1.58%, paired -0.310%). Individual paired changes
include -9.27%/+4.88% BF16 and +30.05% FP32. These raw median shifts are
not a credible gain/regression attribution.

The final check therefore keeps each precision in its own lower-duty batch:
n=8, 2M steps, same binaries, same 10 s cooldown, per-trial idle and 1%
guard (`ab_pong_low_*`). BF16 1.423906 -> 1.424760M (+0.06%); FP32
0.975469 -> 0.975421M (-0.005%). Paired medians -0.049%/+0.049%, wins
3/8 and 5/8. Median learner GPU times are also flat (BF16 45.31/45.30 ms,
FP32 66.43/66.42). Treat Pong as neutral; no Pong speedup is claimed.
Both long mixed and lower-duty cohorts remain available, rather than
selecting only favorable runs. No runtime code or binary changed between them.

## Exactness, gates and mutation tests

At `e05a2301`, `./build.sh ENV DIR/puffer --metal [--float]`, then
`tests/metal/helpers/seedrun.sh DIR ENV 1`, fresh `final2` run directories.
All eight required score/checkpoint prefixes match (`seeds_final2.out`).
No changed numerical reference, gate expectation or five-seed cohort needed.

| env | FP32 score / SHA256 prefix | BF16 score / SHA256 prefix |
|---|---|---|
| Breakout | 844.95672607421875 / `2ee4da82d356a44c` | 852.43658447265625 / `f5dc197b8d7fde55` |
| Pong | 20.99445915222168 / `5ee574ef2f33c6dd` | 20.996614456176758 / `b1425713d9e71433` |
| Flappy | 51.408348083496094 / `4eafd59e547e43e1` | 50.560173034667969 / `91886134a50625e1` |
| Cartpole | 200 / `bf0330663b50b01e` | 199.96110534667969 / `b10ccdb69f45c996` |

**22/22 gates PASS** via this worktree's `tests/metal/run_all.sh`
(`gates_final2.out`), including H64 default/opt-out seeds, both replay/golden
precisions, wait detection, shared IO and held pack/group ordering. Group
unit checks run inside both rollout gates; grouped training inside pack_order.
Legacy embedding and the shader-payload layout check still pass.

- Group unit checks: 20 cases per precision, H32 splits 1/2 and H64 1/2/4,
  2/4 buffers, B17/B64, 17 recurrent steps. Packed weights and ordinary
  outputs get fresh NaNs (0x7fc1 halfwords); state/RNG are independent
  carried inputs. Compare every output/state/RNG byte-for-byte to independent
  original calls; check finite active outputs and red zones.
- Integration: group1/2/4 model bytes match in async, sync, copied IO, reset
  and time-major observation cases. Held 3 ms GPU steps also match for
  shared/copied IO. Dispatch assertions prevent silent fallback. Group1
  points the auxiliary at a nonexistent file, proving it is not loaded.
- **20/20 final kernel mutations caught:** wrong grid-Y arguments and
  dropping stores for Y>0, each of five entry points separately in both
  precisions (`aux_mutations_all.out`). The same audit passed before isolation.
- **3/3 synchronization mutations caught:** missing lazy fallback wait
  exposes NaNs to explicit/legacy consumers; raw_wait instead of verified
  fallback fails transitive timeout detection; missing grouped step wait
  changes the checkpoint under held steps.
- **2/2 loader mutations caught:** eager auxiliary load fails the primary-only
  case; removing the auxiliary precision check fails wrong-precision rejection.
  Both rollout gates check entry-point isolation and reject wrong auxiliary
  precision; primary-only lookup works with a deliberately missing auxiliary.
- **False-pass test corrected:** an initial missing-wait mutation survived
  because cudaMemsetAsync took its idle CPU shortcut, bypassing the hold.
  Replaced it with a direct GPU fill. Fresh source/destination NaNs and three
  changing answers now catch the missing wait; the unmodified checks pass.

Mutant binaries are isolated and correct source is restored before measuring.
No unsafe progress-word skip, raw-wait spin, or changed buffer count is used
for any retained change or speed claim. Rejected sources/logs remain in build/leg16.

## Ranked next levers

1. Keep four env workers while coalescing pairs of ready GPU requests,
   preserving each buffer's time/RNG order. First isolate why wake latency
   grows with pairing: fixed worker ownership versus completion transport.
   Avoid a second host handoff on the critical path.
2. Shorten individual learner dispatches along whole-sequence row tiles,
   preserving accumulation order and occupancy. CB-only chunking failed;
   Cartpole is now learner bound and Flappy still suffers contention.
3. Stage small encoder inputs and decoder sampling logits in threadgroup
   memory, with explicit matching BF16 rounding, to avoid device-memory
   round trips/barriers. Retain NaN poisoning, independent references and
   separate store/wait mutations for every entry point.

The final2 mixed Pong cohort completed but is still nonstationary: BF16
1.366379 -> 1.355730M (-0.78%, paired -0.213%); FP32
0.948318 -> 0.963256M (+1.58%, paired -0.310%). Per-pair changes include
-9.27%/+4.88% BF16 and +30.05% FP32 despite negligible foreign GPU work.
These raw medians are not a credible code-speed claim. An additional
Pong check separates precisions and lowers duty cycle: n=8, 2M steps, same
10-second cooldown, per-trial idle and 1% guard. The mixed cohort remains
reported; no binary or runtime code changes for this check.

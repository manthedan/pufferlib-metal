> Historical development log; last recorded at commit `62a27f1b` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

# Optimization relay leg 19 (Opus): rollout step chain

Branch `metal-opt19`, baseline `d54ff927` (merged `metal` after leg 18).
Target: the fixed host<->GPU latency in each async rollout step. On
Metal 4, a step costs about 100 us from commit to GPU start and 40-90 us
from GPU end to the blocked host waking up. That is over half of a
Cartpole or Flappy step. Nothing was pushed or merged.

## Outcome

**Kept (default on, bit-exact):**

1. **Host-gated, pre-committed rollout steps.** While step t runs, the
   worker encodes step t+1 and commits it behind a queue wait on a gate
   event. It opens the gate right after the env step. Submission latency
   then overlaps the GPU step, the wake and the env step.
2. **Four-buffer gated groups when the learner waits.** With gating, one
   rollout queue of four-buffer steps beats two queues of pairs, but only
   where the rollout is clearly the critical path. A learner-slack policy
   decides once per run (Flappy: groups of four; Cartpole: pairs).

**Opt-in only (measured, below the ~2% bar or too costly):** a bounded
completion poll before blocking waits (`PUF_METAL_ROLLOUT_SPIN_US`), and an
env-thread override (`PUF_ENV_THREADS`).

**Rejected:** clamping the env thread count by default, and a GPU-side
epoch hand-off (cross-queue wait instead of the host sync in
`rollout_start`).

Final clean A/B, baseline `d54ff927` vs final, M SPS medians.
Breakout/Pong final = `c5` (identical rollout/learner code paths to the
final commit for single-buffer envs; host-layout check below):

| env | prec | baseline | final | change | paired median | faster pairs |
|---|---|---:|---:|---:|---:|---:|
| Flappy | BF16 | 6.1317 | 7.2232 | **+17.80%** | +17.81% | 6/6 |
| Flappy | FP32 | 4.6545 | 5.1768 | **+11.22%** | +11.22% | 6/6 |
| Cartpole | BF16 | 13.2465 | 14.5224 | **+9.63%** | +9.70% | 6/6 |
| Cartpole | FP32 | 12.5657 | 13.8320 | **+10.08%** | +10.19% | 6/6 |
| Breakout | BF16 | 4.8618 | 5.5348 | **+13.84%** | +13.91% | 6/6 |
| Breakout | FP32 | 4.4347 | 4.8098 | **+8.46%** | +8.49% | 6/6 |
| Pong (2M) | BF16 | 1.7141 | 1.7218 | +0.45% | +0.56% | 7/8 |
| Pong (2M) | FP32 | 1.0805 | 1.0921 | +1.08% | +1.12% | 7/8 |
| Breakout full 55M | BF16 | 4.8218 | 5.5253 | **+14.59%** | +14.70% | 3/3 |

Exactness: all eight seed-1 references are exact at the final commit,
and also with the opt-in knobs (spin, thread override, pinned groups,
gate off). Gates: 25/25 PASS (23 old plus 2 new); see below.

## Measurement protocol and deviations

- `build/cleanab.py`, alternating order, 10 s cooldown, bandwidth guard.
  Flappy, Cartpole and Breakout: n=6, 20M steps per precision. Pong is a
  precision-separated n=8, 2M-step cohort (leg 16 style). Breakout BF16
  also has a full-length (55M, default) n=3 interleaved cohort. Flappy's
  default length is 20M, so its cohorts are already full length. Runner:
  `build/leg19/abn.sh TAG "envs" ARM...`. Paired analysis:
  `build/leg19/pairs.py`.
- **Deviation:** the final cohorts use `PUF_CLEANAB_IDLE_FIRST=0` and a
  **3% foreign-GPU limit**, not 1%. After about 20:30 the desktop session
  (WindowServer and AccessibilityUIServer, the user's remote screen)
  produced a continuous 2-5% foreign GPU load. Under the 1% limit, or
  while waiting for 0% utilization, cleanab made no progress: 10 attempts
  in 20 minutes all failed. Each arm's foreign GPU fraction is in
  `build/leg19/ab/*/*/log.txt`, and rejected attempts are kept. Gains of
  this size are far above that noise, and all comparisons stay within a
  batch.
- **Contamination re-measured:** a foreign `csnake`/`csnake_f32`
  training session (another session's port) ran from about 19:57 to
  20:29 PDT. The first lever-1 cohort (`ab/lever1`, Flappy BF16 only,
  5 pairs, +10.5%) ran from 19:28 to about 19:58 and is treated as
  preliminary. It was **discarded and every number above was re-measured
  after 20:30**: `ab/final` (lever 1 + spin), `ab/final5` (lever 1 +
  groups, Pong), `ab/final6` (final Flappy/Cartpole), `ab/layout6`,
  `ab/full6`, `ab/policy`, `ab/handoff`, `ab/group4*` and `ab/explore2`.
  The partial `ab/explore` started under csnake and was discarded too.
- Background jobs ran at nice 0. zsh's `BG_NICE` had put an early `&`
  launch at nice 5; that batch was killed before producing any result.

## Profile before optimizing (trace, BF16, baseline `d54ff927`)

Per rollout step, `PUF_METAL_TRACE` + `tools/metal/trace_timeline.py`,
middle epochs. Times are in us, p50 (mean):

| env | commit->GPU start | GPU | GPU end->wake | env step | step total | epoch | learner GPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cartpole | 109 (113) | 64 (93) | 63 (57) | 32 (34) | 279 (303) | 9.90 ms | 8.45 ms |
| Flappy | 95 (113) | 89 (125) | 64 (65) | 21 (24) | 274 (333) | 21.54 ms | 15.83 ms |
| Breakout | 97 (124) | 476 (519) | 67 (67) | 169 (182) | 879 (917) | 29.62 ms | 22.56 ms |
| Pong | 113 (139) | 292 (283) | 86 (86) | 75 (84) | 604 (614) | 41.70 ms | 39.81 ms |

Pong is learner-bound (the learner GPU is busy 95% of the epoch). The
other three are bound by the rollout chain. In Cartpole and Flappy,
commit->start plus wake (~170 us) is about 60% of a step.

## 1. Host-gated pre-committed rollout steps (kept)

### Runtime API (`src/metal/runtime.{h,mm}`)

The API is `gate_arm`, `gate_commit`, `gate_open(run)` and `gate_ran`:

- **Arming.** `gate_arm(s)` commits the open work. The next command buffer
  must hold exactly one dispatch; it is committed behind
  `[queue waitForEvent:gate value:++target]`.
- **Safety against MTL4's silent ~4 s queue-wait release.** The gated
  dispatch is indirect. Its grid is copied, by a blit at the start of the
  command buffer (so behind the wait), from stage words that the host
  fills only in `gate_open`. A wait released early therefore runs an
  empty grid. `gate_ran(v)` compares the live grid with the real one, and
  the worker re-encodes the step. `gate_open(run=false)` cancels a step.
- **Feedback hazard.** MTL4 holds commit i's feedback back while the next
  commit on the same queue is blocked on a queue wait. The probe measured
  this, and `test_gate strict` reports one commit held. So strict joins
  (`join_all_feedback`, reached from main-thread `cudaMemcpy`/`cudaMemset`)
  skip the commit just before a pending gated commit. A gated stream also
  settles lagged feedback one observation later.
- **Restrictions.** Gating needs a host-wait-only stream (nothing on the
  GPU may wait on a gated commit). It is disabled with `PUF_METAL_KPROF`
  and with strict feedback. Reusing a slot whose gated commit is still
  pending is fatal.
- **Review fix (Codex).** If the 4 s release overlaps `gate_open`, the
  GPU could copy the stage before the signal. `gate_open` now fences the
  caller's input writes before it publishes the grid, not only before
  the signal (commit `9615e9af`).

### Worker (`src/pufferl.cu`, `metal_gated_horizon`)

- Step t+1 (single packed or grouped dispatch, same args) is committed
  gated before the worker waits for step t's `COPY_END`. It is released
  right after the env step.
- If `gate_ran` is false, the worker logs "gated step released early",
  cancels t+1, and redoes t ungated.
- No gated commit is ever pending at the end of a horizon, so none can be
  pending at checkpoint, eval or shutdown.
- Eligibility: async, shared env IO, packed steps on every buffer, CPU
  envs. A single buffer's stream is now host-wait-only when gating is on.
- `PUF_METAL_ROLLOUT_GATE=0` restores the old loop.
  `PUF_METAL_LAZY_PROGRESS=0` also disables gating.

### Effect (trace, BF16, p50 us; gated queue time counted from gate_open)

| env | release->start | wake | env step | step total | epoch ms |
|---|---:|---:|---:|---:|---:|
| Breakout | 97 → **26** | 67 → 47 | 169 → 186 | 879 → 797 | 29.62 → 26.02 |
| Pong | 113 → **25** | 86 → 41 | 75 → 49 | 604 → 273 | 41.70 → 40.08 |
| Flappy (pairs) | 95 → 98 | 64 → 36 | 21 → 17 | 274 → 268 | 21.54 → 20.56 |
| Cartpole (pairs) | 109 → 77 | 63 → 34 | 32 → 28 | 279 → 264 | 9.90 → 9.44 |

- **Single queue.** With one rollout queue, a released step starts in
  ~25 us.
- **Two rollout queues.** With two queues of pairs (Flappy, Cartpole), a
  released step still starts 70-100 us after `gate_open`, even with the
  learner idle (uncontended p50 74 / 67 us). The step still gains,
  because its wake and env step shrink: the worker stays hot encoding the
  next step.
- **Two-queue anomaly.** This MTL4 behavior is not understood. It led to
  §2.

**Proxies (per epoch, BF16):**

- **Commits and waits.** Commits and CPU waits are unchanged; Breakout
  drops 39.8 → 33.8 commits because the dW side-stream policy flipped
  (below).
- **Gated commits.** 30.8 (Breakout), 60.8 (Pong), 61.6 (Cartpole pairs),
  125.2 (Flappy pairs).
- **Barriers.** One blit→dispatch barrier more per gated step.
- **GPU time.** Rollout GPU time rises 0.5-0.6 ms/epoch on Flappy and
  Cartpole (indirect dispatch + blit). Breakout's falls 17.1 → 16.1 ms
  (less learner contention per step).

**Gating only, clean n=6** (`ab/final5`, arm `c3`):

| env | change vs baseline | paired | pairs |
|---|---:|---:|---:|
| Flappy BF16 / FP32 | +10.41% / +8.09% | +10.50% / +7.24% | 6/6, 5/5 |
| Cartpole BF16 / FP32 | +9.91% / +10.70% | +9.90% / +10.81% | 6/6, 6/6 |
| Breakout BF16 / FP32 | +13.90% / +8.47% | +13.87% / +8.57% | 6/6, 6/6 |

**Learner policies.** Breakout's learner is now the critical path (train
23.1 ms vs epoch 23.7 ms). The slack policy now picks no dW side stream
and backward overlap on. Pinning the old choices loses
(`ab/policy`, n=6, vs adaptive default):

| pinned | Breakout BF16 | Breakout FP32 |
|---|---:|---:|
| `DW_SIDE=1` | -1.80% (0/6) | -2.11% (0/6) |
| `BWD_OVERLAP=0` | -2.61% (0/6) | -1.90% (0/6) |
| both (old policy) | -4.16% (0/6) | -3.64% (0/6) |

So the adaptive policies do the right thing.

**Tests:**

- `tests/metal/run_test_gate.sh`, unit cases:
  - `pipeline`: 2 threads × 400 pipelined steps, each reading the input
    written right before its gate opened;
  - `cancel`;
  - `strict`: a strict join with a closed gate returns at once;
  - `spin`;
  - `early`: a 4.6 s closed gate makes Metal release the wait, the
    dispatch runs empty, `gate_ran` is false and the redo works;
  - `misuse`: fatal paths.

  Mutations: 6/6 caught (`build/leg19/mutate_gate.py`): no strict-join
  cap, grid not released, no queue wait, `gate_ran` always true, dispatch
  reading the stage directly, no blit→dispatch barrier.
- `tests/metal/run_test_rollout_gate.sh`, end to end on Flappy BF16:
  - final checkpoint bytes, gate off vs on, for each config: default
    grouped (policy), pairs, groups of four, ungrouped, single buffer;
  - a 4.5 s env-step hold (`PUF_METAL_TEST_GATE_HOLD`) forces the
    early-release recovery and still matches;
  - held GPU steps (`PUF_METAL_TEST_STEP_HOLD`) match.

  Mutations: 4/4 caught (`build/leg19/mutate_gate_e2e.py`).

## 2. Four-buffer gated groups when the learner waits (kept)

Leg 16 found four-buffer groups worse than pairs (more wake/env overhead
per worker). With gating, one queue of four-buffer steps avoids the
two-queue start latency above.

Groups of four vs pairs, both gated (`ab/final5`, n=6):

| env | BF16 | FP32 |
|---|---:|---:|
| Flappy | **+6.86%** (6/6) | **+3.98%** (5/5) |
| Cartpole | +1.29% (5/6) | **-1.28%** (0/6) |

Cartpole's learner becomes critical (its train time rises with the
bigger rollout steps). That makes the choice a policy decision, not a
default:

- **When it decides.** In `metal_dw_policy`, once per run: the median
  learner slack of async epochs 3-8, measured under pairs, must be
  > 5% (the same threshold as the dW policies) to switch to four-buffer
  groups.
- **Observed slack.** Flappy BF16 7-14% and FP32 8-9% (groups of four).
  Cartpole BF16 0.4-3% and FP32 0.8-2% (pairs kept).
- **Logging.** The decision is printed to stderr ("learner slack ... ->
  rollout groups of N").
- **OpenMP team cap.** The merged team of a group is capped at all cores
  but two (8 on the M4 Air) unless `PUF_ENV_THREADS` is set. Cartpole in
  groups of four would otherwise run a 16-thread team: -2.4% at 16,
  +0.5% at 8 or 4 vs pairs (`ab/group4t`, n=4). Only grouped teams are
  capped; Pong's single-buffer team is unchanged.
- **Override.** `PUF_METAL_ROLLOUT_GROUP=2/4` pins the group size.

Final vs gating-only (`ab/final6` vs `ab/final5`): Flappy gains the
group-of-four margin, and Cartpole keeps pairs (same as gating only).

Proxies with groups of four (Flappy BF16, per epoch):

- commits 130.1 → 69.5 and CPU waits 129.1 → 68.5;
- gated commits 65.5;
- CPU cores 0.70 → 0.59 (one worker instead of two).

## 3. Bounded spin-then-block wait (opt-in, not default)

`stream_wait_spin(s, us)` polls the stream's completion (`isb` pause) for
up to `us` before the blocking `waitUntilSignaledValue`. The blocking loop
re-checks completion, so the poll can only shorten a wait. Verification,
feedback and the stall diagnostics are unchanged.

- **Knob and metrics.** `PUF_METAL_ROLLOUT_SPIN_US=N` turns it on for
  rollout streams. `metal/spin_hits` / `spin_misses` and a new
  `metal/cpu_cores` (getrusage CPU seconds / wall seconds) price it.
- **Tests.** `test_gate spin` checks short waits, a 30 ms hold past a
  2 ms budget, and gated steps on a polling stream. Mutations: 4/4 caught
  (`build/leg19/mutate_spin.py`).

Clean n=6 on top of gating (pairs), 1000 us budget (`ab/final`):

| env | SPS vs gated | paired | pairs | CPU cores (busy) |
|---|---:|---:|---:|---|
| Flappy BF16 | +2.63% | +2.47% | 6/6 | 0.70 → **2.36** |
| Flappy FP32 | +1.14% | +0.98% | 5/6 | 0.59 → 2.32 |
| Cartpole BF16 | +0.96% | +0.46% | 6/6 | 0.81 → 2.41 |
| Cartpole FP32 | +1.14% | +1.21% | 6/6 | 0.82 → 2.39 |
| Breakout BF16 | +0.61% | +0.55% | 4/6 | 0.78 → 1.53 |
| Breakout FP32 | +0.01% | +0.14% | 4/6 | 0.67 → 1.41 |

- **Wake latency.** Gating already halved the wake (63-67 → 34-47 us),
  and Breakout is now learner-bound.
- **Verdict.** The poll buys 0-2.6% for about **3x the process CPU time**
  (+0.7-1.7 busy cores) on a fanless laptop. Below the ~2% bar except
  Flappy BF16, so it is not a default.
- **Thermal check not needed.** Flappy's cohorts are full length (20M
  default). No full-length thermal A/B was needed because nothing kept
  spins.

## 4. Env OpenMP thread clamp (opt-in override only)

`PUF_ENV_THREADS=N` overrides `vec.num_threads`; threads do not change
results (per-env RNG). Exploratory n=3 (`ab/explore2`, gated `c2`, 3%
foreign limit):

| env | threads | SPS vs config | env ms/epoch |
|---|---|---:|---|
| Pong (16) | 10 / 8 / 4 | +0.0% / -0.9% / +0.8% | 3.3 → 2.5 / 2.6 / 1.9 |
| Cartpole (16, pairs) | 8 / 4 | +0.5% / +0.1% | 0.85 → 0.62 / 0.55 |
| Flappy (8) | 4 | +0.3% | 1.35 → 1.17 |
| Breakout (8) | 4 | +0.2% | 3.9 → 4.1 |

- **Effect.** Env time drops, but the envs are no longer on the critical
  path, so SPS is flat (<1%).
- **Default kept.** The config's count stays the default. Grouped
  four-buffer teams are capped (§2).
- **Leg 16 agreement.** This matches leg 16's finding that smaller
  per-pair teams are flat.

## 5. GPU-side epoch hand-off (rejected)

`rollout_start` copies the actor snapshot and packs rollout weights on
the default stream, then host-syncs. The learner is critical again on
Breakout, and Cartpole's learner idles ~0.8 ms/epoch.

**Attempt.** Replace the sync with an event that the rollout streams and
the train stream wait on, on the GPU. The `muon->lr` write also moves
into learner stream order. It stayed bit-exact.

**Result.** The cross-queue wait made the learner's first command buffer
start *later*: +0.46 ms after `train_impl` vs +0.25 ms (trace). It also
added one wait-only submit per rollout stream per epoch. Cartpole BF16,
n=6 (`ab/handoff`): **-1.10%**, paired -1.85%, 0/6. Reverted; the patch is
kept at `build/leg19/handoff_gpu_wait.patch`.

**What would work instead.** Do the copy+pack at the tail of the
previous train command buffer, so that no cross-queue wait is needed;
see the next levers.

## Busy loops

The busy loops (`rollout_finish`, worker BUF_RUNNING/BUF_WAITING) are
unchanged.

- **CPU cost is small.** `metal/cpu_cores` puts the whole process at
  0.5-0.9 busy cores in async training (gated, default threads).
- **Why.** The loops cover only the learner/rollout slack, which gating
  shrank (Cartpole's main thread waits 0.2-0.6 ms per 9 ms epoch).
- **Pause instructions.** An `isb`/`yield` pause would not lower that.
- **Parking.** Parking would add a 30-60 us wake on the epoch critical
  path.

## Exactness, gates, new gate commands

- **Seeds.** Seed 1, all eight references exact at the final commit
  (`build/leg19/seeds.sh c7`):
  - Breakout: BF16 852.43658447265625/f5dc197b8d7fde55, FP32
    844.95672607421875/2ee4da82d356a44c;
  - Pong: BF16 20.996614456176758/b1425713d9e71433, FP32
    20.99445915222168/5ee574ef2f33c6dd;
  - Flappy: BF16 50.560173034667969/91886134a50625e1, FP32
    51.408348083496094/4eafd59e547e43e1;
  - Cartpole: BF16 199.96110534667969/b10ccdb69f45c996, FP32
    200/bf0330663b50b01e.
- **Seeds with knobs.** Also exact: Flappy BF16 with
  `PUF_METAL_ROLLOUT_SPIN_US=1000 PUF_ENV_THREADS=4`; Cartpole FP32 with
  `PUF_METAL_ROLLOUT_SPIN_US=300 PUF_METAL_ROLLOUT_GROUP=4`; Breakout BF16
  with `PUF_METAL_ROLLOUT_GATE=0`.
- **Gates.** `bash tests/metal/run_all.sh`: 25/25 PASS at the final commit.
  These are the 23 existing gates plus two new ones, added to
  `tests/metal/run_all.sh`:
  - `tests/metal/run_test_gate.sh` (runtime unit test; cases `pipeline`,
    `cancel`, `strict`, `spin`, `early`, `misuse`);
  - `tests/metal/run_test_rollout_gate.sh [TRAINER]` (end-to-end
    byte-identity, Flappy BF16 by default, builds `./puffer`).
- **Mutation scripts** (scratch, untracked): `build/leg19/mutate_gate.py`
  (6/6), `build/leg19/mutate_gate_e2e.py` (4/4), and
  `build/leg19/mutate_spin.py` (4/4).
- **Host layout.** Breakout BF16 `c5` vs `c6` (group policy added):
  +0.04%, paired +0.12% (`ab/layout6`). No layout effect.
- **Reviews.** Two Codex (Sol) reviews; all findings were fixed:
  - the gate-open fence (P1);
  - the test counter delta (P3);
  - the non-Metal build of `metal_env_team` (P1);
  - the epoch-zero `cpu_cores` NaN (P2).

## Commits

- `507dad6f` metal: host-gated commits (gate_arm/gate_commit/gate_open/gate_ran)
- `996ab979` metal: pipeline rollout steps behind host gates
- `9615e9af` metal: order gate inputs before publishing the gated grid
- `8a4935e2` metal: bounded completion poll before blocking host waits
- `6f8f3587` metal: opt-in rollout wait poll, env thread override, CPU-cores metric
- `ecde04f0` metal: four-buffer gated rollout groups when the learner waits
- `5feb680e` metal: keep the shared env loop building without Metal; guard cpu_cores
- docs commit (this report)

## Ranked next levers

1. **Learner critical path again.** After this leg, Breakout (train
   23.1 of 23.7 ms), Cartpole (8.3-8.8 of 9.0-9.5 ms) and Pong are
   learner-bound, and Flappy is close (16.8 of 18.1 ms). The next leg
   should go back to the learner. Breakout's ragged encoder dW (leg 18)
   is the most promising.
2. **Tail-of-train snapshot.** Encode the actor copy and weight pack at
   the end of the previous train command buffer, behind the learner's
   last kernels. The host sync and copy then leave `rollout_start`
   without a cross-queue wait (the form that lost in §5). Upper bound
   about 0.2-0.3 ms/epoch, Cartpole ~2-3%.
3. **Two-queue gate start latency.** Gated steps on one of two rollout
   queues start 70-100 us after release, vs 25 us with a single queue.
   Understanding this (MTL4 firmware queue-wait scheduling?) could
   recover ~50 us/step for Cartpole's pairs, where groups of four lose to
   learner contention. A cheap experiment: both pairs gated on one shared
   gate event, or both pairs on one queue.
4. **Spin.** Revisit only with a CPU/power budget, e.g. a predictive poll
   that sleeps until about 20 us before the expected completion.

**Recommendation: STOP the rollout-chain bottleneck for now.** Gating
removed most of the submission latency, and every env except Flappy is
now learner-bound. The remaining rollout-side lever (spin) costs ~3x CPU
for ≤2.6%. Move the next leg to the learner, and to lever 2 (tail-of-train
snapshot) if it is cheap. If lever 3 is understood, a short follow-up on
Cartpole's pairs is worth it.

## Review follow-up

Closed all three P2 test gaps with test-only changes. No production bug
was exposed; production code, kernel arithmetic and defaults are unchanged.

- `test_gate spin` now holds work for 20 ms inside a 200 ms spin budget
  and requires `spin_hit` to advance, independently of the short-work waits.
- `test_bwd_overlap.cpp` calls the real `metal_dw_policy` with controlled
  slack: below, at and just above 5%, an unsorted median case, excluded
  warm-up epochs, exactly six samples from epochs 3-8, and an immutable
  decision/sample window under later opposing slack. Runs in FP32/BF16 via
  `bash tests/metal/run_test_bwd_overlap.sh`, already in the GEMM gate.
- `run_test_rollout_gate.sh` now expires a gate at hold 150 with forced
  four-buffer groups and requires recovery plus byte-identical checkpoints.

Review mutations (`python3 build/leg19/mutate_review.py`): **3/3 caught**:
microseconds treated as nanoseconds, unconditional pairs, and recovery
re-encoding only two buffers. Each failed its intended assertion; each
restored test passed. Original mutation scripts also caught **14/14**:
`mutate_gate.py` 6/6, `mutate_gate_e2e.py` 4/4, `mutate_spin.py` 4/4.

`bash tests/metal/run_all.sh`: **25/25 PASS**, including all eight seed-1
reference scores/checkpoint hash prefixes. No new gate was needed.

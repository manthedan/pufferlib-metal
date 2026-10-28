> Historical development log; last recorded at commit `5b63c30d` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

## Leg 16 (Sol): small-env rollout latency II

Branch `metal-opt16`, baseline `51e0a5fb`, implementation **`e05a2301`**
(final documentation commit changes no code). Nothing pushed.
Full report: [opt-leg16-report.md](opt-leg16-report.md).

**Kept, all bit-exact:**

- `dc7c1a12`: omit progress-word barrier/blits on unused multi-buffer async
  rollout streams. Any later GPU consumer instead performs a verified host
  wait, including transitive timeout/feedback checks. Immutable stream role;
  `PUF_METAL_LAZY_PROGRESS=0` restores writes.
- `702b5b85`, `725fe79e`: two original buffers per packed dispatch, independent
  argument/state/Philox/output addresses, unchanged arithmetic. A leader
  steps the pair's envs after GPU completion; unused workers sleep. The
  actual buffer count stays four. `PUF_METAL_ROLLOUT_GROUP=1` opts out.
- `e05a2301`: lazy auxiliary library for grouped entry points; primary
  metallib byte-identical to baseline in both precisions, verified in all
  eight binaries. Keeps unused group kernels from disturbing single-buffer
  pipelines. Both libraries embed in __PUF_METAL; standalone tests use a
  sidecar. Independent precision guards and lazy-load tests.

**Profile first:** baseline Flappy BF16 24.51 ms/epoch, learner 18.00,
queue/GPU/wake p50 140/58/32 us; contended/uncontended kernels 87/50 us.
Cartpole 11.54 ms, learner 9.36, queue/GPU/wake 166/51/37. Pairing halves
rollout commits (256->128 Flappy; 128->64 Cartpole), though dispatch/wake
spans grow. Cartpole is now learner bound. Final hot post-gate traces and
all earlier traces are in `wt-opt16/build/leg16/traces`; only clean within-
batch A/Bs support speed claims.

**Final clean A/B**, alternating, 10 s cooldown, per-trial GPU idle and a
stricter 1% foreign-GPU guard. M SPS, n=6 each (Pong n=8, separate 2M-step
lower-duty precision batches; other envs 20M):

| env | BF16 before -> after | FP32 before -> after |
|---|---:|---:|
| Flappy | 5.3672 -> **5.6961 (+6.13%)** | 4.1795 -> **4.5877 (+9.77%)** |
| Cartpole | 11.4442 -> **12.5945 (+10.05%)** | 10.5038 -> **11.9945 (+14.19%)** |
| Breakout | 4.8742 -> 4.9072 (+0.68%) | 4.3657 -> 4.3772 (+0.26%) |
| Pong | 1.4239 -> 1.4248 (flat) | 0.97547 -> 0.97542 (flat) |

Flappy/Cartpole: 6/6 faster pairs in both precisions. Breakout 6/6 and 4/6.
Pong paired medians -0.049%/+0.049%, wins 3/8 and 5/8. Long mixed 8M
Pong cohorts have large regime swings even with low foreign GPU work; all
are reported, no absolute SPS pooled and no Pong speedup claimed.

**Exactness:** all eight required seed-1 score/checkpoint prefixes exact at
`e05a2301`, no numerical cohort or gate expectation change. Breakout FP32
844.95672607421875 / `2ee4da82d356a44c`; BF16 852.43658447265625 /
`f5dc197b8d7fde55`. Full eight-value table in the report.

**Tests:** **22/22 gates PASS** at final implementation. Fresh NaNs, guards,
independent per-buffer state/RNG, finite-output checks; held GPU steps plus
model bytes and dispatch-count assertions. **20/20** final auxiliary-kernel
store/selector mutation runs caught (all five entry points, both precisions),
**3/3** wait/verification mutations caught and **2/2** loader mutations caught.
An initial false-pass test used cudaMemsetAsync's idle CPU shortcut; direct
GPU fill corrected it, and the missing wait then failed.

**Rejected:** four-buffer default; smaller OpenMP teams (flat); timer-only
predictive wake (-53.6%, real event wait retained, no spinning); learner CB
caps 128/64 (-0.7%/-0.6%, learner slower); aligned global counters (-0.49%
vs current in a five-way Breakout host/library control). The enlarged primary
library's quiet Pong FP32 cohort lost 0.39%; old-library crossover recovered
it, motivating the retained auxiliary split. No unsafe progress skip or
raw-wait spin was repeated. All prototypes/logs under `wt-opt16/build/leg16`.

**Next levers, ranked:** coalesce ready GPU requests while preserving four
env workers and each buffer's ordering; shorten individual learner dispatches
on whole-sequence tiles without reducing occupancy (CB-only caps failed);
stage small encoder inputs / sampling logits in threadgroup memory with
matching BF16 rounding to remove device-memory round trips/barriers.

> Historical development log; last recorded at commit `3ef39b79` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

## Leg 19 (Opus): rollout step chain

Branch `metal-opt19`, baseline `d54ff927`. Full report:
[opt-leg19-report.md](opt-leg19-report.md). Nothing pushed or merged.

Profile first (BF16 trace, p50 per rollout step): commit→GPU start
95-113 us and GPU end→host wake 63-86 us on every env. Together they
were ~60% of a Cartpole/Flappy step (274-279 us). Pong was already
learner-bound (learner GPU 95% of the epoch).

**Kept (default on, bit-exact):**

1. **Host-gated pre-committed rollout steps.** The worker commits step
   t+1 behind an MTL4 queue wait on a gate event, then opens the gate
   after the env step. New runtime API: `gate_arm`/`gate_commit`/
   `gate_open`/`gate_ran`. The gated dispatch is indirect, and its grid
   is published only at open, so Metal's silent ~4 s queue-wait release
   runs an empty grid. The worker detects that and redoes the step.
   Strict joins skip the feedback that MTL4 holds behind a blocked commit.
   With one rollout queue, release→start is ~25 us (was ~100). With two
   queues it stays 70-100 us, and why is not understood.
2. **Four-buffer gated groups when the learner waits.** A slack policy
   picks groups of four if the median learner slack over async epochs 3-8
   is >5%. Otherwise it keeps pairs: groups of four lose 1.3% on Cartpole
   FP32. Grouped OpenMP teams are capped at ncpu-2.

**Opt-in:**

- `PUF_METAL_ROLLOUT_SPIN_US`, a bounded completion poll: +0-2.6% for
  ~3x process CPU (0.7 → 2.4 busy cores).
- `PUF_ENV_THREADS` thread override: flat (<1%).
- New metrics: `metal/cpu_cores`, `metal/spin_hits`, `metal/spin_misses`.

**Rejected:** GPU-side epoch hand-off through a cross-queue wait
(Cartpole BF16 -1.10%, 0/6; the learner starts later), and clamping env
threads by default.

Clean cohorts: n=6, 20M steps; Pong n=8, 2M steps; alternating order,
10 s cooldown. **Deviation:** idle-first was off and the foreign-GPU
limit was 3%, because the desktop session's continuous 2-5% GPU load
blocked the 1% guard. A foreign csnake run (19:57-20:29 PDT)
contaminated the first lever-1 batch. It was discarded, and every number
below was re-measured afterwards.

| env / prec | M SPS base → final | change | paired | faster pairs |
|---|---:|---:|---:|---:|
| Flappy BF16 | 6.1317 → 7.2232 | **+17.80%** | +17.81% | 6/6 |
| Flappy FP32 | 4.6545 → 5.1768 | **+11.22%** | +11.22% | 6/6 |
| Cartpole BF16 | 13.2465 → 14.5224 | **+9.63%** | +9.70% | 6/6 |
| Cartpole FP32 | 12.5657 → 13.8320 | **+10.08%** | +10.19% | 6/6 |
| Breakout BF16 | 4.8618 → 5.5348 | **+13.84%** | +13.91% | 6/6 |
| Breakout FP32 | 4.4347 → 4.8098 | **+8.46%** | +8.49% | 6/6 |
| Breakout BF16 55M (n=3) | 4.8218 → 5.5253 | **+14.59%** | +14.70% | 3/3 |
| Pong BF16 2M (n=8) | 1.7141 → 1.7218 | +0.45% | +0.56% | 7/8 |
| Pong FP32 2M (n=8) | 1.0805 → 1.0921 | +1.08% | +1.12% | 7/8 |

Gating alone gives Flappy +10.4/+8.1%, Cartpole +9.9/+10.7% and Breakout
+13.9/+8.5% (BF16/FP32). Groups of four add Flappy's +6.9/+4.0%. On
Breakout, gating makes the learner critical, which flips the adaptive dW
policies (no side stream, overlap on). Pinning the old policies loses
3.6-4.2%.

**Validation:**

- **Gates.** 25/25 PASS. That is the 23 existing gates plus two new ones:
  `tests/metal/run_test_gate.sh` (runtime unit cases `pipeline`,
  `cancel`, `strict`, `spin`, `early`, `misuse`) and
  `tests/metal/run_test_rollout_gate.sh`. The second checks end-to-end
  checkpoint byte identity with the gate off vs on across five rollout
  layouts, a forced early release and held GPU steps.
- **Seeds.** All eight seed-1 references are exact, including with the
  opt-in knobs.
- **Mutations.** 14/14 caught.
- **Reviews.** Two Codex (Sol) reviews; all findings fixed (gate-open
  fence P1, non-Metal build P1, two minor).

**Recommendation: STOP the rollout-chain bottleneck.** Every env except
Flappy is now learner-bound: Breakout trains 23.1 of 23.7 ms, Cartpole
8.3-8.8 of 9.0-9.5 ms. Move the next leg to the learner (Breakout's
ragged encoder dW first). Two cheap side levers:

- do the snapshot copy+pack at the tail of the previous train command
  buffer (~0.2-0.3 ms/epoch);
- investigate the two-queue gated start latency (~50 us/step on
  Cartpole's pairs).

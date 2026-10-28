> Historical development log; last recorded at commit `1af50fc9` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

## Leg 17 (Opus): learner critical path

Branch `metal-opt17` (worktree `wt-opt17`), base `55f76032`, implementation
`927c6cc6`. Full report: `docs/dev/optimization/opt-leg17-report.md`. Nothing pushed or
merged. **All eight seed-1 references exact; 23/23 gates PASS.** No accumulation
order changed. Every change is bit-identical, default-on, with an env
opt-out.

**Profile (base, KPROF, Pong BF16 per minibatch):** fused H=64 forward 3 x
295 us, backward 3 x 280, MinGRU dW split-K GEMM 3 x 158, decoder forward
GEMM 132, Muon ~160, PPO ~46; 776 learner dispatches and 841 barriers per
epoch. Cartpole: `muon_ns5_batch` 144 us per minibatch.

**Kept:**
1. `5f4245cd`: small-N GEMM (N <= 7) for the decoder forward: 132 -> 30 us
   (`PUF_METAL_GEMM_SMALL_N=0`).
2. `968283b2`: runtime concurrent dispatch windows and deferred split-K
   reductions (`gemm_rm_deferred` / `gemm_finish`).
3. `34a18391`: backward dW GEMMs overlap the layer chain in concurrent
   windows; Pong barriers 841 -> 708 per epoch (`PUF_METAL_BWD_OVERLAP`).
4. `b522c1d0`: wider matmul2d tiles in the fused MinGRU layers: Pong
   forward 295 -> 223 / 369 -> 286 us, backward 280 -> 263 / 489 -> 457
   (BF16 / FP32).
5. `b05ee65f`: H=64 backward grad_input stays in threadgroup memory; device
   copy optional (2/4 MiB less traffic per layer per minibatch).
6. `91b45015`: simdgroup-fragment GEMM stages in the fused Muon NS steps:
   Cartpole `muon_ns5_batch` 176 -> 52 us (`PUF_METAL_MUON_NS_MMA=0`).
7. `927c6cc6`: overlap gated per epoch on learner slack <= 5% (same signal
   as `metal_dw_policy`). The first final A/B with it always on regressed
   rollout-bound Flappy FP32 (-5.1%) and Breakout BF16 (-2.8%); a bisect
   pinned that on the overlap contending with the async rollout.

**Final clean A/B** (base vs `927c6cc6`; cleanab alternating, cooldown 10 s,
idle-first, 1% foreign limit; Pong 2M n=8 precision-separated, others 20M
n=6). M SPS medians, paired medians, faster pairs:

| env | BF16 | FP32 |
|---|---:|---:|
| Pong | 1.4247 -> **1.7086 (+19.93%)**, +20.05%, 8/8 | 0.9722 -> **1.0796 (+11.05%)**, +11.05%, 8/8 |
| Cartpole | 12.614 -> **13.209 (+4.71%)**, +4.95%, 6/6 | 11.907 -> **12.596 (+5.79%)**, +5.41%, 6/6 |
| Flappy | 5.6946 -> **6.1323 (+7.69%)**, +7.57%, 6/6 | 4.5821 -> 4.6471 (+1.42%), +1.47%, 6/6 |
| Breakout | 4.8168 -> 4.8606 (+0.91%), +0.91%, 6/6 | 4.3113 -> 4.4347 (+2.86%), +2.99%, 6/6 |

**Tests:** new reference tests for every entry point (small-N GEMM host
oracle, concurrent-window barrier counts, deferred GEMMs, null grad_input,
Muon fragment shapes). Mutations: 10/10 small-N, 8/8 windows/deferred, 3/3
overlap seeds, 14/14 Muon fragments, grad_input all non-equivalent caught.
New gate `tests/metal/helpers/overlap_e2e.sh` (Cartpole + Breakout, BF16 +
FP32, overlap auto / forced on / off, each seed-1 exact), run from
`tests/metal/run_all.sh` as `overlap_e2e`.

**Rejected:** warp-specialized H=64 forward (314 vs 292 us), wide BF16 H=32
projection tiles, backward chunk 8 / prefetch 8, the always-on overlap.

**Next levers, ranked:** opt-in H=64 dW folded into the fused backward
(order change; drops 14 MiB of grad_combined traffic per layer); Muon wide
chain latency (15 tiny dependent dispatches, ~120 us per Pong minibatch);
Breakout `mingru_dw_reduce` partial traffic; forward scan/projection overlap
without occupancy loss; small-K decoder dX kernel (46 us).

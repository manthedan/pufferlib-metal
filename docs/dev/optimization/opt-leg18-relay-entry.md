> Historical development log; last recorded at commit `4df07db6` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

## Leg 18 (Sol): learner critical path II

Branch `metal-opt18`, baseline `17db51e3`. **Documentation-only outcome;
no production change retained.** Full report:
[opt-leg18-report.md](opt-leg18-report.md). Nothing pushed or merged.

Profile first: Pong BF16/FP32 recurrent forward 225/278 us, backward
249/427 us, dW split GEMM 152/308 us per layer-minibatch; wide Muon
107/99 us per minibatch. Cartpole backward+dW 301/327 us per layer,
Muon batch 52/51 us. Breakout dW partial reduction 186/184 us per layer;
encoder dW 616/817 us. Async trace and command-buffer gap breakdowns are
in the report.

Two serious bit-preserving attempts, **both rejected**. Completed clean
cohorts use n=6, alternating order, 10 s cooldown, idle-first, strict <1%
foreign GPU guard and unchanged bandwidth guards; no builds/tests overlap
accepted trials. Pong is a separate BF16 2M-step cohort; Breakout uses
precision-separated 20M-step cohorts. Never pool absolute SPS across them.

| attempt / target | M SPS off → on | medians change | paired median | faster pairs |
|---|---:|---:|---:|---:|
| Muon fusion / Pong BF16 | 1.705852 → 1.670652 | -2.06% | -2.08% | 0/6 |
| dW layout / Breakout BF16 | 4.858283 → 4.583836 | -5.65% | -5.43% | 0/6 |
| dW layout / Breakout FP32 | 4.416201 → 4.259937 | -3.54% | -3.53% | 0/6 |

Muon fusion drops 80 learner dispatches and barriers per epoch but
recomputes Gram-buffer slices; fused update 19.84/18.04 us versus two
old stages totaling 10.50/10.50 us. Its incomplete FP32 wall cohort is
excluded after foreign GPU activity prevented completion.

The blocked partial layout preserves every FP32 partial and ascending
sum order. The reducer's isolated improvement is small; async learner
and rollout times rise. Traffic stays 12 MiB written plus 12 MiB read per
layer. Dispatch topology is unchanged at fixed policies. Exact references
include both overlap regimes and all 12 H32 end-to-end mode cases.

**Validation:** 23/23 gates PASS on the final restored source tree; all
eight seed-1 score/checkpoint references exact. All eight rebuilt native
binaries have byte-identical host sections, primary shaders and auxiliary
rollout shaders versus baseline, including Breakout and Flappy. No final
throughput gain is claimed. Four layout store/order mutations are caught
in FP32/BF16; an initial shared-reducer false-pass was fixed with a CPU
ascending-sum oracle. No new production kernel, flag or gate remains.
Scratch reference/mutation commands and saved rejected patches are in
the report; the final runner remains `bash tests/metal/run_all.sh`.

**Recommendation: STOP this learner bottleneck pass under the relay's
~2% rule.** Move the next leg to rollout ownership/completion transport
(leg 16 lever 1). If learner work is reopened with a fresh hypothesis,
Breakout's ragged encoder dW has more headroom than another partial-layout
tweak. Small-K decoder dX has <2% practical headroom; H64 dW folding stays
default-off, opt-in and lowest priority because it changes summation order.

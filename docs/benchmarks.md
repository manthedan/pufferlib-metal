# Recorded Metal benchmarks

Record dated **2026-10-26** (the source did not retain separate per-cohort
timestamps). See the [benchmark methodology](dev/optimization-history.md#methodology)
for the protocol and its load/guard exceptions. Hardware: fanless 15-inch
MacBook Air M4, 24 GB; macOS 26.4; **AC power**. Comparison: before versus
after the FMA-corrected sigmoid change. These are historical observations.

| Configuration | Runs | Base median SPS | Final median SPS |
|---|---:|---:|---:|
| Breakout BF16 | 6 | 6.2615M | 6.4683M |
| Breakout FP32 | 6 | 4.8730M | 5.2474M |
| Cartpole BF16 | 6 | 15.3602M | 16.2989M |
| Cartpole FP32 | 6 | 12.7242M | 13.9475M |
| Flappy BF16 | 8 | 7.0345M | 7.4990M |
| Flappy FP32 | 8 | 4.8707M | 5.0855M |
| Pong BF16 (2M steps) | 8 | 1.6661M | 1.7672M |
| Pong FP32 (2M steps) | 8 | 1.0556M | 1.0599M |
| Freeway BF16 (8.4M steps, concurrent load) | 6 | 0.8880M | 0.9265M |
| Freeway FP32 (8.4M steps, concurrent load) | 6 | 0.4282M | 0.4036M |

Method: alternating before/after runs, 10-second cooldown, median SPS. Breakout and Cartpole BF16 were rerun with a 3% foreign-GPU limit and default
bandwidth floors. Other small-policy cohorts ran after the main browser load
ended but retained relaxed guards; one Cartpole FP32 sample still saw 7.5%
browser GPU time. Freeway ran with browser load at 57–83%, a 95% foreign-GPU
limit and 40/40 GB/s bandwidth floors. Its FP32 median change was -5.75% while
the paired median was +2.60%, so the wide control has low confidence. Full thermal
telemetry and a complete command/config manifest were not retained in the report.
The historical untracked snake experiment is not a reproducible public benchmark
and is recorded only in the optimization history. No speed or learning-parity guarantee follows
from these measurements. Future benchmark updates should record full commands,
config hashes, toolchain version, thermal state and concurrent load.

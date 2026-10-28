# Recorded Metal benchmarks

Record dated **2026-10-26**, the date of measurement commit `6cedf3ee` (the
original log does not record separate per-cohort timestamps). Source:
[leg 23 development report, section 4](dev/optimization/opt-leg23-report.md#4-end-to-end-ab-cleanab-alternating-cooldown-10-s-base-vs-final).
Machine: fanless 15-inch MacBook Air M4, 24 GB; macOS 26.4; **AC power**.
Comparison: base `b0f7bb84` versus `6cedf3ee`. These are historical observations,
not new measurements on the cleanup host.

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

Method: alternating before/after runs, 10-second cooldown, median SPS. Small-policy
cohorts were run after a competing browser GPU load ended; Freeway was measured
with that load back at 57–83% GPU time and relaxed bandwidth guards. Full thermal
telemetry and a complete command/config manifest were not retained in the report.
The historical untracked snake experiment is not a reproducible public benchmark
and is kept only in the archive. No speed or learning-parity guarantee follows
from these measurements. Future benchmark updates should record full commands,
config hashes, toolchain version, thermal state and concurrent load.

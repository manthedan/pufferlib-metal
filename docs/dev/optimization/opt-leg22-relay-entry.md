> Historical development log; last recorded at commit `42f830a9` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

## Leg 22 (Astra): H=512/1024 tier

Mac mini M4, 16 GiB; base `d606b131`, branch `metal-opt22`, final production
`ca945bd2`. Default-size Tetris (H512/L2) and WEF (H1024/L4) fit and are
learner-bound in both precisions. g2048 FP32's default allocation lower bound
is 11.9258 GiB versus the measured 11.8400 GiB Metal budget, before all buffers;
it was not used as a viable stock-budget benchmark, and no OOM was induced.

Kept exact recurrent/ragged TN spatial tiles, large Muon batching with aligned
scratch staging and tiled products, and recurrent NT/dX tiles. Switches
`PUF_METAL_GEMM_LARGE_TN=0`, `PUF_METAL_MUON_LARGE=0`, and
`PUF_METAL_GEMM_LARGE_LINEAR=0` restore the old components independently.
Split partitions, reduction order, normalization and every BF16 rounding
boundary are preserved. Primary/rollout libraries retain their baseline hashes.
Sliced MinGRU forward was tested and rejected: losses at three of four target
shapes and a small isolated BF16 H512 gain did not justify shipping it.

Six alternating pairs per cohort, ten-second cooldown, idle/bandwidth wait,
foreign-GPU threshold 1%, no relaxation. Saved baseline versus final release:

| Workload | Baseline median SPS | Release median SPS | Median change | Paired median change | Faster pairs |
|---|---:|---:|---:|---:|---:|
| tetris BF16 | 50,369.5 | 116,068.0 | +130.43% | +130.31% | 6/6 |
| tetris FP32 | 20,699.5 | 73,341.0 | +254.31% | +255.00% | 6/6 |
| wef BF16 | 5,545.5 | 9,938.0 | +79.21% | +79.05% | 6/6 |
| wef FP32 | 2,414.5 | 5,571.0 | +130.73% | +130.64% | 6/6 |

The twelve control cohorts (four small envs, Freeway and snake, both precisions)
range from -0.58% to +1.71% median SPS, with no substantial regression observed.
The worst loss is Breakout FP32 (-0.45% paired, 1/6 pairs faster); retained in
the report. No absolute SPS is pooled across batches or compared with laptops.

The first combined candidate regressed WEF BF16 by 50.09% despite exact outputs.
Native profiling exposed an unaligned Gram bottleneck missed by the first
mixed-shape tuner. Staging the whole NS chain into aligned scratch, measuring
production-shaped batches, tiling products and restoring scalar fallback
concurrency produced the final gains above. Incorrect extreme TN/product tiles
were rejected. WEF removes 928 FP32 / 864 BF16 learner dispatches and barriers
per epoch, and 5 GiB / 1.75 GiB logical copy traffic respectively; extra scratch
is about 80 MiB / 64 MiB. Tetris dispatch counts are unchanged.

**35/35 gates pass**, including all eight required small seed references and
wide_e2e. The final large native gate passes generic/default/each-component-off
for Tetris and WEF in both precisions (20 runs), and twelve separate release
runs match the saved baseline full SHA and score. All 27 new mutations were
caught with restored passes; all 24 existing wide mutations were caught.
The final audit verifies 192 accepted A/B runs and intact library/snapshot
hashes. No order- or precision-changing option was introduced.

New gates, all registered in the supplied local `tests/metal/run_all.sh`:

```sh
bash tests/metal/run_test_large_gemm.sh
bash tests/metal/run_test_large_muon.sh
bash tests/metal/run_test_large_linear.sh
python3 tests/metal/mutate_large.py
bash tests/metal/run_test_large_e2e.sh
```

Tracked aggregate: `bash tests/metal/run_leg22_gates.sh`. Production commits:
`14b129e9`, `a1e56026`, `3377ab68`, `d0ae1cd6`, `ca945bd2`.
No push or merge. The snake snapshot remains untracked.

Next levers: native H1024 Muon Gram/cache scheduling, Tetris TN operand reuse
with unchanged partitions, and native FP32 H1024 dX behavior. The bounded
spatial sweeps stopped paying off; wide recurrent backward needs a different
memory ownership plan. Full default-length Tetris/WEF training was not run;
short native full-SHA checks and every-stage primitive references are covered.
Report: [opt-leg22-report.md](opt-leg22-report.md).

> Historical development log; last recorded at commit `144ac8d5` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

## Leg 20 (Astra): H=128/256 MinGRU tier

Branch `metal-opt20`, baseline `3ef39b79`. Full report:
[opt-leg20-report.md](opt-leg20-report.md). Nothing pushed or merged;
the supplied classic_snake snapshot remains untracked.

**Profile first:** Snake H256/L2 (obs1008, 4096 agents, two buffers,
minibatch8192) and Freeway H128/L4 (16384 agents, four buffers,
minibatch16384), both default geometry and both precisions. Freeway's
cheap CPU env takes only ~33–35 ms against ~799/1521 ms of synchronous
learner work (BF16/FP32). Snake's encoder dW (256,1008,8192) falls to the
scalar unsplit path and costs ~4.7–4.9 ms per minibatch. Wide recurrent
dW and Muon Gram-buffer/update are also substantial.

**Kept, default on and bit-exact:**

1. `38a39417`: masked ragged-N TN tensor ops, plus spatial retile of
   H128/H256 recurrent TN. The H128 split path retains every original K
   chunk, partial and reduction order. Opt-outs:
   `PUF_METAL_GEMM_RAGGED=0`, `PUF_METAL_GEMM_WIDE_TN=0`.
2. `9822bef2`: templated wide MinGRU projection/scan and backward/dX.
   Forward saves combined; backward keeps the highway gradient on chip.
   Original BF16 state and dX rounding remain intact. Opt-out:
   `PUF_METAL_FUSED_MINGRU_WIDE=0`.
3. `c1c841dc`: extend the existing exact 8x8 Muon fragments to minimum
   dimensions 128/256; mixed odd matrices keep the scalar fallback.
   Opt-out: `PUF_METAL_MUON_WIDE_MMA=0`.

New kernels are in a lazy `puffer.wide.metallib`. Primary and grouped
rollout metallibs remain byte-identical to baseline. No rollout kernel,
training configuration, split-K partition or numerical variant changed.

The offline tuner tries only order-preserving spatial/chunk parameters.
52 compiling MinGRU candidates passed exactness in 170.71 seconds total;
24 four-row candidates failed the installed primitive's static assertion.
18 TN variants took 53.84 seconds. Selected MinGRU chunks are 8/16
(FP32/BF16), threads and column widths equal H; scratch is 12/16 KiB at
H128 and 24/32 KiB at H256. Larger FP32 chunks do not fit. Register counts
could not be decoded by the installed M4 archive/disassembly tools.

Clean cohorts: n=6, alternating order, 10-second cooldown, bandwidth
quiet waits and sampled foreign-GPU accounting. Same desktop workaround
as leg19: `PUF_CLEANAB_IDLE_FIRST=0`, `PUF_CLEANAB_MAX_FOREIGN=0.03`.
Snake uses 2,097,152 steps, Freeway 8,388,608, with the default base
seed 73. Exactness/full-budget runs use seed 1. No accepted trial overlaps
another test, build or GPU workload; no absolute SPS is pooled across
batches.

| workload | SPS baseline → final | median change | paired change | faster pairs |
|---|---:|---:|---:|---:|
| Snake BF16 |284,070.5 → 489,933|+72.47%|+72.09%|6/6|
| Snake FP32 |198,907.5 → 360,707|+81.34%|+81.40%|6/6|
| Freeway BF16 |1,098,143.5 → 1,366,237|+24.41%|+24.22%|6/6|
| Freeway FP32 |571,611 → 859,407.5|+50.35%|+50.35%|6/6|

Within those cohorts, MinGRU fusion contributes paired +4.47/+2.26%
for Snake and +12.80/+12.64% for Freeway (BF16/FP32). The Freeway TN tile
contributes +6.24/+34.05%. Eight small-policy control cohorts span
-0.37% to +0.51% paired, with no material regression observed.

Full-budget validation also passes: one generic/optimized pair at 200M
Snake and 1.2B Freeway in each precision, with identical final score and
full checkpoint SHA256 in all four pairs. Late optimized dashboard SPS
remains higher in every workload. These are single-pair thermal/exactness
checks, not additional n=6 performance claims. The report gives scores,
hashes, guards and early/late rates, and documents an existing sparse
history issue that makes trailing Freeway INI timing bins misleading.

**Deterministic proxies:** fusion removes 128 learner dispatches/epoch on
Snake, 512 on Freeway; logical device tensor-pass traffic falls 2/4 GiB
and 8/16 GiB per epoch respectively (BF16/FP32). The existing overlap
policy becomes eligible, further reducing barriers and two Freeway
command buffers. TN and Muon preserve dispatch counts, intermediate
capacity and unique operand byte sets; cache/load multiplicity changes.

**Validation:** 30 gates, including five new wide gates; 20/20 new
mutations caught. All eight established seed 1 references match. Wide
short-run scores and full checkpoint SHA256 match baseline, all paths
off, and each individual opt-out in both precisions. New commands are
listed in the full report and added to `tests/metal/run_all.sh`.

**Next:** recurrent projection/backward tiling and weight reuse, then
Snake encoder NT/residual TN. H512 needs a different scratch strategy
before extending eligibility. Wide rollout remains secondary on these
default workloads. The initial H256 narrow-column fusion and initial
FP32 ragged tile regressed and were rejected; no split-fp16 or other
order/precision-changing path was retained.

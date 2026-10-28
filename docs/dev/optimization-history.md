# Metal optimization history

These records summarize optimization rounds 12, 16–20, 22 and 23. Missing
round numbers have no report in this collection. Gate counts below describe
the tested revision; the [suite index](../../tests/metal/README.md) describes
current coverage. Historical Snake measurements used an untracked
`classic_snake` snapshot and are not reproducible public benchmarks.

## Methodology

Use [tests/metal/bench.sh](../../tests/metal/bench.sh) as the shared benchmark
yardstick. Its default is three FP32 Breakout runs of 20,000,000 steps;
`--env`, `-n`, `--steps`, `--bin` and `--no-build` select other cases.
`--ab A B` interleaves binaries A then B on each repetition; `--full` retains
the configured budget and evaluation. Short timing runs disable evaluation
and set checkpoint interval 100000. The reported `metal_summary` excludes
epoch zero. SPS is steps divided by wall time over the remaining epochs;
report per-binary medians and per-run spread. GPU spans from overlapping
queues must not be added to infer wall time.

Historical controlled A/B runs used an untracked `cleanab.py` helper on top
of this measurement contract. They alternated arm order, cooled down for
10 seconds before each arm, and normally used six pairs, a GPU idle wait,
pre/post bandwidth floors of 80/60 GB/s and a 1% foreign-GPU limit. These
guards and alternating order are **not implemented by bench.sh itself**.
Keep rejected guard attempts, compare only within a cohort, and report both
the change of medians and median of paired ratios. Do not pool absolute SPS
across batches, machines or thermal regimes. No builds, profiles or other
tests overlapped accepted controlled trials. KPROF serializes dispatches:
its timings attribute cost but cannot establish asynchronous throughput gains.
Use `--base.eval-episodes=0` with KPROF to avoid the observed evaluation hang.

Protocol exceptions matter:

| Rounds | Geometry, repetitions and departures |
|---|---|
| 16–18 | Small policies normally 20M steps, n=6; Pong precision-separated 2M, n=8 in 16/17 and n=6 for the rejected round-18 Muon trial. Round-16 exploratory cohorts used a 3% foreign limit; final cohorts used 1%. Long mixed-precision Pong cohorts were nonstationary and do not support speed claims. |
| 19 | Small policies as above; Breakout BF16 also 55M, n=3. Idle-first disabled and foreign limit 3% because continuous desktop GPU work blocked the 1% guard. An initial overlapping training cohort was discarded and remeasured. |
| 20 | n=6, Snake 2,097,152 steps, Freeway 8,388,608, default benchmark seed 73. Idle-first disabled, foreign limit 3%, bandwidth guard retained; accepted foreign fractions 0.0000–0.0124. Separate seed-1 full-budget on/off pairs used 30-second cooldown, 80/60 GB/s floors and 3% foreign limit. |
| 22 | n=6, strict 1% foreign limit, idle/bandwidth waits, no relaxation. Tetris 1,048,576 steps (one measured epoch), WEF 65,536 (three measured epochs); default geometry. Largest accepted foreign fraction 0.91%. 192 accepted target/control runs. |
| 23 | Fanless 15-inch M4 laptop, 24 GB, macOS 26.4, AC power. Breakout and Cartpole BF16 were rerun with 3% foreign limit, idle-first off and default bandwidth floors. Cartpole FP32, Pong and Flappy retained 95% foreign / 40 GB/s floor settings from the first batch but ran after the main browser load ended; a Cartpole FP32 sample still saw 7.5% browser GPU time. Wide controls ran under 57–83% browser GPU load with 95% foreign and 40/40 GB/s floors. These controls have low confidence. |

Round 23 used n=6 for Breakout/Cartpole, n=8 for Flappy/Pong, and 2M Pong
steps. Wide controls used n=6, Freeway 8.4M and Snake 2.1M as reported.
The discarded loaded small-policy cohorts measured Breakout BF16 -3.7%
(2/6 faster), FP32 +12.6%, and Cartpole BF16 -8.8%. The retained Freeway
FP32 control has -5.75% change of medians but +2.60% paired median; bimodal
load prevents a strong regression conclusion. The date recorded for this
measurement revision is 2026-10-26; separate per-cohort timestamps, full
thermal telemetry and a complete command/config manifest were not retained.
Future measurements should retain those records and config/toolchain hashes.

Correctness gates precede timing: independent CPU/FP64 or original scalar
references, CUDA captures, poisoned outputs and allocation guards, per-stage
comparisons, selection/dispatch assertions, mutations, then end-to-end
score and checkpoint comparisons. Exact changes preserve accumulation order,
split-K partitions, BF16 store boundaries and per-buffer RNG ownership.
Counters and logical tensor traffic explain a result; neither proves a
speedup or measures DRAM traffic. Changes to numerical order need separate
validation and must not inherit an exact checkpoint claim.

## Chronological changes

| Round | Retained change and configurations | Measured result | Correctness evidence |
|---|---|---|---|
| 12 | BF16 custom encoders for Minimal, Asteroids, four OSRS environments, Craftax, NMMO3 and NetHack; remove FP32 fallback. FP32 accumulators and fixed-point sums remain; OSRS forward GELU tile stays unrounded. | No performance claim. All 27 bounded BF16/FP32/CUDA training runs completed. | 18 CUDA cases repeat exactly; 1196 tensor comparisons + 290 independent exact/FP64 checks per run, 64 poisoned primitive checks; 1196 FP32 tensors unchanged; 21/21 gates. |
| 16 | Safe lazy progress words on host-wait-only streams, paired rollout dispatches retaining four original buffers, lazy auxiliary library isolating new kernels. | Flappy +6.13%/+9.77%, Cartpole +10.05%/+14.19% (BF16/FP32); Pong neutral. | All eight small seed references exact; 22/22 gates; 20/20 store/selector, 3/3 synchronization and 2/2 loader mutations caught. |
| 17 | Small-N decoder GEMM; concurrent backward dW windows and deferred reductions; wider H64 tiles; on-chip grad_input with optional device store; small Muon fragments; overlap only at learner slack <=5%. | Pong +19.93%/+11.05%; Cartpole +4.71%/+5.79%; Flappy +7.69%/+1.42%; Breakout +0.91%/+2.86%. | Eight seed references exact; 23/23 gates; 10/10 small-N, 8/8 window/deferred, 3/3 overlap, 14/14 Muon mutations caught; workspace exhaustion and policy-boundary controls. |
| 18 | No retained production change after Muon fusion and dW layout trials. | Both attempts regressed; see rejected table. | Restored binaries' host/primary/rollout sections identical in all eight builds; 23/23 gates; seed references exact. Independent CPU sum oracle catches 4/4 layout mutations. |
| 19 | Precommitted rollout behind host gates; indirect grid publication detects silent queue-wait release and redoes empty work. Four-buffer groups selected when median learner slack in epochs 3–8 exceeds 5%; grouped teams capped at ncpu−2. | Flappy +17.80%/+11.22%; Cartpole +9.63%/+10.08%; Breakout +13.84%/+8.46%; Pong +0.45%/+1.08%. | Eight seed references exact, including opt-ins; 25/25 gates; 14/14 mutations plus 3/3 follow-up mutations caught; held GPU/early-release recovery. |
| 20 | Ragged and recurrent H128/H256 TN spatial tiles, fused wide projection/scan and backward/dX, wide Muon fragments; lazy wide library. | Snake +72.47%/+81.34%; Freeway +24.41%/+50.35%. | 30/30 gates, 24/24 mutations after follow-up; stage references and full checkpoint/score equality across opt-outs. Full-budget pairs exact at 200M Snake / 1.2B Freeway configured steps. |
| 22 | Exact H512/H1024 recurrent/ragged TN and NT/dX tiles; large Muon batching, aligned scratch staging and tiled products with scalar fallback concurrency. | Tetris +130.43%/+254.31%; WEF +79.21%/+130.73%. | 35/35 gates, 27 new + 24 existing wide mutations; 20 native mode runs plus 12 release comparisons exact in score/full SHA. |
| 23 | FMA-corrected sigmoid in learner, rollout and wide shaders; IEEE-division compile-time fallback. | Breakout +3.30%/+7.68%, Cartpole +6.11%/+9.61%, Flappy +6.60%/+4.41%, Pong +6.07%/+0.41%. | 40/40 historical gates; all 2^32 FP32 patterns, NaNs included, exact in five translation units on validated M4 compilers; eight seed references unchanged. |

### Recorded throughput cohorts

Tables preserve the original reported precision. Percentages above are changes
of medians, BF16 then FP32. Hardware differed: rounds 16–20 used an M4 laptop;
round 22 used an M4 with 10 GPU cores and 16 GiB, macOS 27.0.1 / Xcode 27.1.
The following tables are separate cohorts, not cumulative speedup estimates.

#### Round 16

M SPS.

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

#### Round 17

M SPS.

| env | precision | before | after | change (medians) | paired median | faster pairs |
|---|---|---:|---:|---:|---:|---:|
| Pong | BF16 | 1.4247 | 1.7086 | **+19.93%** | +20.05% | 8/8 |
| Pong | FP32 | 0.9722 | 1.0796 | **+11.05%** | +11.05% | 8/8 |
| Cartpole | BF16 | 12.614 | 13.209 | **+4.71%** | +4.95% | 6/6 |
| Cartpole | FP32 | 11.907 | 12.596 | **+5.79%** | +5.41% | 6/6 |
| Flappy | BF16 | 5.6946 | 6.1323 | **+7.69%** | +7.57% | 6/6 |
| Flappy | FP32 | 4.5821 | 4.6471 | +1.42% | +1.47% | 6/6 |
| Breakout | BF16 | 4.8168 | 4.8606 | +0.91% | +0.91% | 6/6 |
| Breakout | FP32 | 4.3113 | 4.4347 | +2.86% | +2.99% | 6/6 |

#### Round 19

M SPS.

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

#### Round 20

SPS (M denotes millions).

| workload | baseline → final SPS | median change | paired change | faster pairs |
|---|---:|---:|---:|---:|
| Snake BF16 |284,070.5 → 489,933|+72.47%|+72.09%|6/6|
| Snake FP32 |198,907.5 → 360,707|+81.34%|+81.40%|6/6|
| Freeway BF16 |1,098,143.5 → 1,366,237|+24.41%|+24.22%|6/6|
| Freeway FP32 |571,611 → 859,407.5|+50.35%|+50.35%|6/6|

#### Round 22

SPS (M denotes millions).

| Workload | Baseline median SPS | Release median SPS | Median change | Paired median change | Faster pairs |
|---|---:|---:|---:|---:|---:|
| tetris BF16 | 50,369.5 | 116,068.0 | +130.43% | +130.31% | 6/6 |
| tetris FP32 | 20,699.5 | 73,341.0 | +254.31% | +255.00% | 6/6 |
| wef BF16 | 5,545.5 | 9,938.0 | +79.21% | +79.05% | 6/6 |
| wef FP32 | 2,414.5 | 5,571.0 | +130.73% | +130.64% | 6/6 |

#### Round 23

SPS (M denotes millions).

| Cohort | n | Base median SPS | Final median SPS | Median | Paired median | Faster |
|---|---:|---:|---:|---:|---:|---:|
| Breakout BF16 | 6 | 6.2615M | 6.4683M | +3.30% | +3.97% | 5/6 |
| Breakout FP32 | 6 | 4.8730M | 5.2474M | +7.68% | +7.39% | 6/6 |
| Cartpole BF16 | 6 | 15.3602M | 16.2989M | +6.11% | +5.31% | 6/6 |
| Cartpole FP32 | 6 | 12.7242M | 13.9475M | +9.61% | +11.44% | 6/6 |
| Flappy BF16 | 8 | 7.0345M | 7.4990M | +6.60% | +6.59% | 8/8 |
| Flappy FP32 | 8 | 4.8707M | 5.0855M | +4.41% | +1.66% | 6/8 |
| Pong BF16 (2M) | 8 | 1.6661M | 1.7672M | +6.07% | +5.63% | 8/8 |
| Pong FP32 (2M) | 8 | 1.0556M | 1.0599M | +0.41% | +0.50% | 5/8 |
| Freeway BF16 (8.4M), under load | 6 | 0.8880M | 0.9265M | +4.34% | +4.95% | 6/6 |
| Freeway FP32 (8.4M), under load | 6 | 0.4282M | 0.4036M | -5.75% | +2.60% | 4/6 |
| snake BF16 (2.1M), under load | 6 | 0.2381M | 0.2497M | +4.85% | +2.65% | 4/6 |
| snake FP32 (2.1M), under load | 6 | 0.1790M | 0.1803M | +0.75% | +0.51% | 4/6 |

Round 16 Pong's low-duty paired medians were -0.049%/+0.049% (3/8, 5/8
wins); its FP32 median change was also reported as -0.005%. No Pong gain
was claimed. Round 20's eight small-policy control paired medians span
-0.37% to +0.51%. Round 22's twelve control median changes span -0.58% to
+1.71%; the worst is Breakout FP32, -0.45% paired with 1/6 faster pairs.
These observations do not establish mathematically zero performance change.

Round 20 full-budget actual steps were 199,753,728 for Snake and
1,199,570,944 for Freeway. Generic/optimized scores were exactly
68.10526275634766 / 62.40625 for Snake BF16/FP32 and
21.3359375 / 21.34375 for Freeway; full checkpoint hashes matched.
Those single pairs are exactness/sustained-behavior checks, not n=6 estimates.
Freeway's stale trailing INI bins were excluded from sustained-rate claims;
see the [logging correction](parity-investigations.md#training-log-history).
Round 22 did not run full default-length Tetris or WEF training.

## Tried and rejected

“Not measured” means the source supplied no numerical effect. Rows marked
“not attempted” or “opt-in retained” distinguish exclusions from reverted
implementations. Diagnostic timings are not controlled end-to-end gains.

| Round / idea | Where measured | Measured effect | Why rejected or restricted |
|---|---|---|---|
| 12: round OSRS forward GELU before pooling | BF16 encoder mutation | Incorrectly rounded tile fails the gate; no throughput measurement | Only saved backward h1 is rounded in CUDA; forward must keep FP32 tile. |
| 16: four-buffer default | Flappy / Cartpole, 10M, n=6 | Group1/group2/group4 M SPS: Flappy BF16 5.391496/5.704872/5.387360, FP32 4.122162/4.533998/4.509541; Cartpole BF16 11.582948/12.521639/11.784979, FP32 10.754747/12.003266/11.836987. Group4 BF16 -0.08% Flappy; Cartpole +1.74% median but -0.06% paired (both 3/6). | Pairs win; bigger wake/env overhead. Later reconsidered only with gating and slack policy in round 19. |
| 16: smaller OpenMP teams | Paired BF16 Flappy / Cartpole | 5.796458→5.788597M (-0.14%, 1/6); 12.518913→12.549826M (+0.25%, 4/6). Smaller-team group4 -3.42%/-4.37% versus standard group2 (0/6 each). | Flat or slower. |
| 16: timer-only predictive wake | Flappy BF16, 10M, n=6 | Control 5.719181M; 20 us lead 2.654977M (-53.58%, 0/6), 40 us lead 2.653285M (-53.61%, 0/6); learner 18.92→24.14/24.60 ms. | Estimator learns its own sleep delay; actual event wait retained. |
| 16: learner command-buffer caps 128/64 instead of 256 | Same Flappy cohort | 5.679773M/5.686633M (-0.69%/-0.57%, 3/6 each), learner 19.07/19.12 ms (+0.8%/+1.1%). | Shorter buffers do not shorten individual kernels; no public queue-priority setting in installed SDK. |
| 16: aligned global counters | Breakout BF16 five-way host/library control | Current 4.835298M; aligned 4.811684M, -0.49% vs current. | No counter-layout benefit. |
| 16: grouped kernels in primary library | Quiet Pong FP32, n=6 | Baseline 0.970008M, enlarged 0.966256M (-0.39%), current host/original library 0.970188M (+0.02%); original library +0.41% paired, 5/6. | Unused shaders disturbed existing pipelines; retained separate lazy library. |
| 16: unsafe progress skip / raw-wait spin | Referenced earlier work, not repeated | No new measurement | Preserve verified transitive waits; no unsafe skip or spin adopted. |
| 17: warp-specialized H64 forward | Projection/scan overlap microprofile | 314 vs 292 us | Doubled scratch lowers occupancy. |
| 17: H32 BF16 projection width 96 | Fused forward | Slower than width 32; no magnitude reported | Kept width 32. |
| 17: backward chunk 8 / PREFETCH 8 | Fused backward | Chunk gain ~1–2%, inside noise; prefetch slower, no magnitude reported | No reliable gain. |
| 17: on-chip grad_input while keeping device write | H64 backward | No separate number reported | Benefit required dropping the unused store; retained nullptr form only. |
| 17: unconditional backward overlap | First final A/B | Flappy FP32 -5.13% (0/6), Breakout BF16 -2.80% (0/6), FP32 -0.34% (1/6). Bisect overlap on/off Flappy -4.86%/-0.19%, Breakout -3.47%/+0.93%. | Faster learner contends with critical rollout; restrict to slack <=5%. |
| 17/18: fold H64 dW into backward | Not attempted as default | No measurement; potential 6 MiB write + 8 MiB read saved per layer | Changes summation order; would need explicit numerical variant and learning evidence. |
| 18: fuse Muon Gram-buffer/update | Pong BF16 2M, n=6 | 1.705852→1.670652M (-2.06%, paired -2.08%, 0/6); fused BF16/FP32 19.84/18.04 us vs old two-stage 10.50/10.50 us. | Recomputed slices outweigh 80 fewer dispatches/barriers per epoch; incomplete FP32 cohort excluded. |
| 18: blocked H32 dW partial layout | Breakout 20M, n=6 each precision | BF16 4.858283→4.583836M (-5.65%, paired -5.43%, 0/6); FP32 4.416201→4.259937M (-3.54%, paired -3.53%, 0/6). | Isolated reducer 185.04→175.34 / 183.56→178.08 us did not survive async execution; still 12 MiB written + 12 MiB read per layer. Cache explanation remains unproven. |
| 19: pin old learner policies after gating | Breakout n=6, BF16/FP32 | DW_SIDE=1 -1.80%/-2.11%; BWD_OVERLAP=0 -2.61%/-1.90%; both -4.16%/-3.64%, all 0/6. | Learner becomes critical; adaptive policy works better. |
| 19: unconditional gated group4 | Cartpole | BF16 +1.29% (5/6), FP32 -1.28% (0/6); 16-thread group -2.4%, 8 or 4 threads +0.5% vs pairs (n=4). | Learner contention; keep pairs at low slack and cap grouped teams. |
| 19: default completion spin (opt-in retained) | Gated pairs, 1000 us, n=6 | Flappy BF16/FP32 +2.63%/+1.14%, Cartpole +0.96%/+1.14%, Breakout +0.61%/+0.01%. Busy cores respectively 0.70→2.36 / 0.59→2.32, 0.81→2.41 / 0.82→2.39, 0.78→1.53 / 0.67→1.41. | About 3x process CPU for limited benefit; not default. |
| 19: default env-thread clamp (override retained) | Gated n=3 | Pong 10/8/4 threads +0.0%/-0.9%/+0.8%; Cartpole 8/4 +0.5%/+0.1%; Flappy 4 +0.3%; Breakout 4 +0.2%. | Less env time does not improve critical path; keep configured counts. |
| 19: GPU-side epoch handoff | Cartpole BF16 n=6 | -1.10%, paired -1.85%, 0/6; first learner start +0.46 vs +0.25 ms | Cross-queue wait starts learner later and adds wait-only submits. |
| 19: park or pause busy host loops | Not implemented | Process already 0.5–0.9 busy cores; estimated parking wake 30–60 us | Loops cover short slack; no demonstrated benefit. |
| 20: four-row MinGRU chunks | Offline tuner | 24 candidates fail primitive static assertion | Installed primitive requires M multiple of 8 or 16. |
| 20: H256 chunk8/128-thread/64-column fusion | FP32 isolated kernels | Forward 1.96 vs 1.79 ms; backward 2.89 vs 2.13 ms | Slower than generic; full-width columns/256 threads retained. |
| 20: initial ragged TN 64x32/128 threads | Snake FP32 | ~7.40 vs 4.91 ms | Replaced by selected spatial tile. |
| 20: larger FP32 chunks / extend same scratch plan to H512 | Scratch feasibility | H256 backward already 32 KiB; smallest legal H512 FP32 chunk exceeds allowance | Requires a different staging plan, not a wider whitelist. |
| 20: split-fp16 emulation, changed split-K, layer folding | Not implemented | No measurement | Would change precision or reduction order; excluded from exact path. |
| 22: initial mixed-batch large Muon candidate | Native WEF BF16 n=6 | 5,498.5→2,744.5 SPS (-50.09%, paired -50.09%, 0/6); Gram 87.582 ms/stage, aligned staging 11.304 ms/stage. | Mixed tuner missed real unaligned four-matrix batch; superseded by aligned staging and production-shaped tuning. |
| 22: tiled scalar fallback with serialized small outputs | Mixed Muon batch | Original fallback serialized up to 64 outputs/thread; no separate SPS delta | Restore one-output-per-thread fallback concurrency. |
| 22: extended TN / Gram / product sweeps | Spatial tuning | Extended TN 7/8 valid, Gram 8/8, product boundary 10/12; no further gain. TN 64x256/64 wrong; product 512x32/128 wrong in both precisions. | Incorrect extremes excluded; compiler cause unresolved. Stop bounded search instead of adding tiny shape branches. |
| 22: sliced MinGRU forward | Isolated, 26 exact candidates | H512 FP32 43.657 vs 38.904 ms; H1024 FP32 18.776 vs 8.570; BF16 H1024 7.953 vs 7.315 (earlier chunk32 only 1.4% ahead of old projection); H512 BF16 29.179 vs 31.233. | Loses three shapes; small H512 BF16 benefit does not justify new path; backward needs all hidden slices. Never connected to policy. |
| 23: Newton reciprocal refinement before FMA correction | Exhaustive sigmoid check | All 2^32 inputs pass without it on validated M4; no timing delta | Unnecessary extra arithmetic. |
| 23: threadgroup-staged dW reduction | Loaded microbench, 96 groups × 256 threads, 64-tile chunks | ~360 vs ~240–274 us | Slower; streams 12 MB FP32 partials, bandwidth bound. |
| 23: H32 grad_input staged in grad_x | Loaded microbench | BF16 unchanged, FP32 -2% (noise) | Reverted; no gain. |
| 23: FP32 backward phase split | Loaded microbench, 128 threads precompute h_tilde/z/s plus sign ballot | ~1027 vs ~1015 us | No gain. |
| 23: BF16 vectorized dX epilogue | Loaded microbench, staged product + uint4 add | 676 vs 679 us | No gain. |
| 23: encoder dW prefetch / cooperative K staging / aliased scratch | Clean GPU, M32 N118 K65536 | FP32 base 905 us, KS128 831 (-8%), KS64 855; scalar stores same. BF16 base 675, KS128 1127, KS64 869, aliased 677. Loaded load-only/MMA-only ablations 365/360 of 1090 us. | Only FP32 improves, ~0.15 ms/minibatch (~1% of Breakout FP32 GPU time); not enough for another shared kernel. All hashes identical. |

## Remaining headroom and limits

- H64 FP32 forward/backward remains near bandwidth limits: round-23 Pong
  282 + 442 us/call, 56% of GPU time. Recompute identical intermediates or
  retain fewer device copies while preserving operation order.
- H32 FP32 backward scan was ~40% of 809 us; 24 KiB scratch limits occupancy
  and half the threads idle in serial scan. Encoder dW load/MMA overlap and
  operand reuse deserve a structural hypothesis, not another reducer layout.
- Large native Muon Gram remains 10.824/16.689 ms per NS stage (BF16/FP32);
  Tetris recurrent TN 31.075/68.398 ms per layer/minibatch; H1024 FP32 dX
  13.854 ms. Preserve norm trees and split partitions when testing cache reuse.
- Wide recurrent fusion needs new memory ownership, particularly backward
  dX across hidden slices. H256 backward already uses 32 KiB scratch.
- Tail-of-train snapshot copy/pack could remove ~0.2–0.3 ms/epoch. Two gated
  queues started 70–100 us after release versus ~25 us for one queue; the
  cause remains unexplained. Default spin is not justified by current CPU cost.
- The sigmoid's bit identity is measured on M4 with macOS 26.4 and 27
  compilers, not guaranteed by MSL reciprocal bounds. Even a one-ULP estimate
  need not correct to IEEE rounding: x=2^-23, z=0x3f7ffffe, d=0x3fffffff,
  y=0.5 yields 0x3f000000 rather than 0x3f000001. Portable testing allows
  <=1 ULP against device IEEE division and validates a sampled CPU oracle
  with RN/RZ/subnormal-flush rules; exact mode also checks NaN payloads and
  zero signs. The default suite sets `PUF_SIG_REQUIRE_EXACT=1` on validated
  M4; `-DPUF_SIG_IEEE_DIV` restores division. A two-ULP mutation fails both
  modes; the rounded-product mutation may pass portable mode.

Longer multi-seed custom-environment learning, full default-length large-policy
training and other device/compiler combinations remain separate validation
work. The historical g2048 FP32 allocation lower bound on the 16 GiB M4 was
11.9258 GiB versus an 11.8400 GiB budget; it was excluded without inducing OOM.

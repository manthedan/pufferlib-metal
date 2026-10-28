> Historical development log; last recorded at commit `e4a6802c` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

# H1024 Metal/CUDA discrepancy diagnosis

2026-10-26. Worktree: `<repo>/wt-goldwide`, branch `metal-goldwide`, starting commit `562c331b` (Metal base `ef62360b`). Golden CUDA math revision: `ea7f989371270d2948f96728bda8aec5e7634a4c`. All builds use `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`; Python numerical work uses the shared `.venv`. No tracked production source, golden data, other worktree, or existing learner configuration was changed. The requested untracked classic_snake files were copied in for the gate suite. No push or merge was performed.

## Finding

The first large backward difference is **one hidden-component element of the backward scan's `grad_combined`**, caused by a forward projection rounding across MinGRU's derivative discontinuity at zero. The TN dW and NN dX GEMMs are correct on their respective inputs. CUDA follows the negative branch of the independent float64 reference; Metal follows the positive branch of a valid FP32 dot product. This is an ill-conditioned cross-implementation golden comparison, rather than a defective Metal backward kernel or defective CUDA golden capture.

The upstream hidden activation is `g(a) = a + 0.5` for `a >= 0`, otherwise `sigmoid(a)`. Its value is continuous at zero, but its one-sided derivatives are **1** and **0.25**. Thus an arbitrarily small projection perturbation across zero can produce a fourfold change in the local hidden derivative while forward values agree to FP32 precision.

Only `(batch=1, time=48, hidden=817)` crosses branches in the L1/B4/T64/I36/H1024 fixture (zero-based indices):

| Quantity | Value |
|---|---:|
| Float64 projection, entirely from golden inputs | `-2.384857582249378e-9` |
| Float64 dot product using Metal's rounded encoder output | `-3.251539544557591e-9` |
| Metal tensor_ops NT projection | `+1.147246408983449e-9` |
| Independent CPU ascending FP32 `fmaf` loop, same operands | `+1.147246408983449e-9` (bit-identical to Metal) |
| Independent CPU reverse FP32 `fmaf` loop | `+1.220488599074088e-9` |
| Sum of absolute products for that dot | `0.2044148581981533` |
| Standard `gamma_1024 * sum_abs_products` FP32 bound | `1.247725436111538e-5` |
| Float64 canonical `grad_combined[1,48,817]` | `-3.052075868061584e-6` |
| Metal `grad_combined[1,48,817]` | `-1.220830290549202e-5` |
| Independently predicted branch-gradient delta | `-9.156227604184751e-6` |

The CPU FMA reproduction is decisive: the disputed projection is an ordinary FP32 accumulation result. No approximation to imitate CUDA was added to a production kernel.

## Stage isolation and causality

Metal intermediates were first dumped using a build-directory-only copy of the original reader, before changing the harness: encoder output, combined projection, scan cache, decoder input gradient, combined gradient, scan direct-input gradient, state gradient, projection dW, encoder gradient and encoder dW. The captured CUDA **policy** manifest contains final gradients, layer outputs/state and original inputs/parameters, but does **not** contain the policy's combined projection or scan-gradient intermediates. Those CUDA intermediates are therefore not claimed as direct observations. Standalone CUDA scan cases do contain their own `grad_combined`, and all H1024 plain/row scan cases pass. The original wide CPU validator checks policy forward, not these full-policy backward tensors; this diagnosis independently reconstructs the missing backward intermediates.

An independent NumPy float64 forward/backward reference was computed from the golden inputs, and a separate Accelerate `cblas_dgemm` float64 reference is now used by the native harness. Error normalization below is `max_abs / max_abs(reference)`, matching the original test, not an L2 norm.

| Comparison | Maximum absolute error | Normalized error |
|---|---:|---:|
| Metal scan combined gradient vs full float64, before branch accounting | `9.1562270e-6` | `6.4169e-2` |
| Same scan comparison, excluding the single crossing element | `2.2359e-11` | — |
| Metal dW vs float64 GEMM on the **same Metal operands** | `4.1599e-11` | `5.9333e-7` |
| Metal dX + highway vs float64 on the **same Metal operands** | `9.8561e-11` | `3.9725e-7` |
| CUDA dW vs canonical float64 policy backward | `4.1679e-11` | `5.9447e-7` |
| CUDA encoder gradient vs canonical float64 | `4.5205e-11` | `1.8220e-7` |
| CUDA encoder dW vs canonical float64 | `6.9739e-10` | `3.7558e-7` |

The derivative change affects neither the recurrent-state derivative formula nor the direct highway derivative. That explains why `layer_0_grad_state` and forward/state outputs remain close while the three gradients downstream of `grad_combined` differ.

The complete discrepancy is predicted by propagating one independently calculated scalar delta:

- `delta_dW[817, :] = delta_dc * encoder_output[1,48,:]`;
- `delta_dX[1,48,:] = delta_dc * projection_weight[817,:]`;
- `delta_encoder_dW = delta_dX.T @ obs`.

After this analytic correction, the native harness's remaining Metal/CUDA normalized errors are **`2.08e-7` for projection dW, `4.55e-7` for encoder gradient, and `7.52e-7` for encoder dW**. The original unexplained errors were `1.205e-2`, `5.762e-4` and `7.553e-5`. There are no sign crossings in the H256/H512 one-layer control cases; their smallest float64 hidden magnitudes are approximately `2.00e-7` and `1.72e-7`.

## Kernel selection and bounds

With `PUF_METAL_GEMM_LOG=1`, all three relevant dense GEMMs select ordinary `GEMM_TENSOR_OPS` (path 1):

| Stage | Row-major operation | M | N | K |
|---|---|---:|---:|---:|
| Forward MinGRU projection | NT | 256 | 3072 | 1024 |
| Projection dW | TN | 3072 | 1024 | 256 |
| Projection contribution to dX | NN | 256 | 1024 | 3072 |

The leg 20 wide TN selection requires N=128 or N=256, so it cannot select this H1024 dW. Its K=256 also excludes every split-K route, which requires K>=2048. H1024 fused MinGRU is ineligible. T=64 uses the plain scan: 16 threadgroups of 256 threads for B*H=4096, one thread per `(b,h)`, with no H-sized threadgroup-memory allocation. The maximum combined index is 786431; the combined buffer is 3 MiB and dW is 12 MiB. There is no H1024 threadgroup-size, shared-buffer-size or integer-overflow defect in this execution.

## Harness changes only

Changes remain on `metal-goldwide`; no `metal-h1024fix` worktree was needed.

1. Add a minimal one-lane regression showing identical FP32 forward values and fourfold derivative changes across tiny positive/negative hidden values.
2. For L1/H1024 FP32 only, independently reconstruct the complete policy in float64. A branch exception requires a sign crossing within the ordinary FP32 dot-product error bound, including propagated encoder error. Validate encoder/projection/cache/output, decoder dX, branch-aware scan gradients and state/direct-input gradients, and each backward GEMM on its actual operands. Independently require CUDA's final gradients to match the canonical float64 backward. These checks use `1e-5`, stricter than the existing policy bound.
3. Apply only the analytic derivative-branch delta to CUDA expectations for the three affected gradients; preserve the original `1e-4` policy comparison bound and print the raw error as well. No blanket tolerance widening or skipped tensor is used.
4. Add optional `PUF_POLICY_DUMP_DIR` dumps and `tests/metal/diagnose_policy_backward.py` for independent NumPy reproduction.
5. Handle the separately discovered BF16 cancellation case with per-element float64 validation, as described below.

Negative controls inject wrong scan gradients, projection dW, dX, an out-of-bound projection sign change, a corrupted in-memory CUDA expected gradient, and a wrong BF16 cancellation result. All are rejected. The golden files themselves are never modified.

An original-reader/revised-reader comparison of **30 dumped tensors across H256/H512/H1024 is bitwise identical**. Production results are unchanged, including the originally disputed H1024 outputs.

## BF16

The original `run_test_goldwide.sh` uses `set -e` and stopped at the first FP32 `test_mingru` failure. Although it built BF16 binaries, it never executed the BF16 numerical fixtures.

I ran both BF16 runs with fused policy selection disabled and enabled. The original reader finds exactly one additional failing check in each run: `gate_h1024/state_out[3915]` (b=3, h=843). Other BF16 wide operation/policy checks pass their pre-existing bounds, including H1024 policy backward.

| Quantity | Value |
|---|---:|
| Hidden / gate / initial state | `0.48828125 / -0.54296875 / -0.57421875` |
| Float64 exact gate state | `-4.3031267678e-6` |
| CUDA BF16 state | `-4.321336746216e-6` |
| Metal BF16 state | `-4.261732101440e-6` |
| Metal/CUDA absolute difference | `5.960464477539e-8` (2 BF16 output ULPs) |
| CUDA / Metal error against float64 | `1.8210e-8 / 4.1395e-8` |

This is cancellation in a weighted sum of values near one, after FP32 sigmoid/lerp evaluation. The correctly rounded float64 BF16 value is **`0xb690 = -4.291534423828125e-6`**. CUDA (`0xb691`) and Metal (`0xb68f`) are each exactly **one BF16 ULP from that reference**, even though they are two ULPs apart from one another.

The harness retains direct one-ULP comparison everywhere else. For an element outside it, a cancellation exception requires `abs(exact) < 2^-8 * condition`, where `condition = abs((1-z)*state) + abs(z*h_tilde)`, and **both** CUDA and Metal must independently be within the original one-ULP limit of the correctly rounded float64 result. That reference replaces the pairwise expectation only for the validated element. No absolute-error floor or larger ULP limit is introduced. A deliberately wrong value of about `-1e-5` is rejected.

## Validation and reproducibility

The complete revised `tests/metal/run_test_goldwide.sh ../golden` finished with **exit 0**. It covers both FP32 captures, both BF16 captures, generic and eligible fused policy paths, and Freeway/Tetris train replay in both precisions with fused selection on/off and both noise probes. The CPU audit verifies all 276 cases, 3,312 tensor files and 3,645 transferred hashes. All eight train comparisons pass, including exact RNG/integer comparisons. No golden file or hash was updated.

The copied full gate runner completed. **All 31 applicable gates pass**, including runtime/rollout gates, both original BF16 golden runs, original train replays, all environment gates, wide GEMM/MinGRU/Muon, all 24 built-in mutation checks, four wide end-to-end environment/precision pairs and exhaustive projection-store checking. Wide end-to-end checkpoints match byte-for-byte across all six optimization settings in each pair.

The raw runner exits **1**, not 0: I edited the MinGRU reader build script before its first invocation finished, causing a shell-script read/write race (`-o: command not found`, exit 127). The clean targeted `run_test_mingru.sh ../golden/run_a` rerun passes. Its remaining two failures are **unavailable tests**, `tests/metal/run_test_protein.sh` and `tests/metal/run_test_sweep_cli.sh`, absent from this branch. This branch still includes `src/metal/protein_stub.h`; those tests and the CPU Protein implementation belong to `wt-protein` and were not imported as an unrelated feature change. Thus the raw summary is 30 passes plus three command failures; after the successful MinGRU rerun, the effective result is 31 passes and two unavailable checks. No numerical gate failed.

Additional validation: the minimal regression passes; all **seven** diagnosis negative controls are rejected; shell syntax, Python compilation and `git diff --check` pass. `build/h1024-diagnosis/gate_summary.json` records raw and effective outcomes.

All eight seed-1 reference score/hash pairs match in all **20** checked modes (H64 default/opt-out; H32 default/forced overlap/opt-out):

| Case | Seed-1 score | Checkpoint SHA-256 prefix |
|---|---:|---|
| breakout_bf16 | `852.43658447265625` | `f5dc197b8d7fde55` |
| breakout_f32 | `844.95672607421875` | `2ee4da82d356a44c` |
| cartpole_bf16 | `199.96110534667969` | `b10ccdb69f45c996` |
| cartpole_f32 | `200` | `bf0330663b50b01e` |
| flappy_bf16 | `50.560173034667969` | `91886134a50625e1` |
| flappy_f32 | `51.408348083496094` | `4eafd59e547e43e1` |
| pong_bf16 | `20.996614456176758` | `b1425713d9e71433` |
| pong_f32 | `20.99445915222168` | `5ee574ef2f33c6dd` |

Main artifacts, relative to `wt-goldwide`:

- `build/h1024-diagnosis/goldwide-fixed.log`: complete passing goldwide run;
- `build/h1024-diagnosis/dump.log`, `original-recheck.log`: original failing reader;
- `build/h1024-diagnosis/comparison.json`, `numpy-final.log`, `policy_layer_*_reference.npz`: NumPy float64 analysis;
- `build/h1024-diagnosis/policy_layer_*.bin`: Metal intermediates;
- `build/h1024-diagnosis/corrected-fp32.log`, `corrected-bf16.log`: focused native reference checks;
- `build/h1024-diagnosis/fma_probe.cpp`, `fma_probe.log`: independent FP32 accumulation reproduction;
- `build/h1024-diagnosis/unchanged_tensors.json`: original/revised tensor SHA-256 equality;
- `build/h1024-diagnosis/mutation_results.json`, `mutation_*.log`: negative controls;
- `build/h1024-diagnosis/gates-run.log`, `mingru-rerun.log`, `gate_summary.json`, `seed_reference_results.json`, and `build/*.log`: broader gate evidence.

Reproduce the optional independent NumPy comparison after building the gates:

```bash
cd <repo>/wt-goldwide
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export PATH=<repo>/.venv/bin:$PATH
PUF_POLICY_DUMP_DIR=build/h1024-diagnosis \
  build/metal/goldwide/fp32/test_mingru ../golden/wide_a
python3 tests/metal/diagnose_policy_backward.py ../golden/wide_a build/h1024-diagnosis
```

## Review follow-up

Addressed both P2 review findings on `metal-goldwide`. The BF16 cancellation reference now rounds directly from binary64 to BF16, with round-to-nearest-even and no intermediate FP32 conversion. A committed regression checks all 32,639 adjacent positive finite BF16 pairs, their exact midpoints and neighboring binary64 values, in both signs, including subnormals, plus overflow, signed zero, infinities and NaN. The review example `(hidden, gate, state) = (-3.65625, 1.875, -0.1640625)` rounds to `0x378b`; conversion through FP32 incorrectly produces `0x378a`.

`run_test_goldwide.sh` now builds and runs committed host controls against the actual FP32 policy exception validator and BF16 gate-state checker before reading the goldens. Valid branch and cancellation cases must pass. Six negative controls must exit **1** with the expected numerical diagnostic: an FP32 sign crossing beyond its dot-product error bound, incorrect scan dC, projection dW, and dX, and either Metal or CUDA BF16 results two ULPs from the independently rounded reference. Setup errors and crashes do not count as rejections. The sign-crossing control remains within the general projection comparison tolerance, so it specifically tests the separate roundoff bound.

Reproduce with `bash tests/metal/check_goldwide_controls.sh` after a normal or `--build-only` goldwide build. `python3 tests/metal/mutate_goldwide.py` compiles isolated reader copies under `build/metal`; tracked sources and golden files are unchanged. Each mutant accepts a known bad fixture (exit 0), then fails the same control gate called by the full goldwide runner (exit 1):

| Reader mutation | Bad fixture exit | Control gate exit | Result |
|---|---:|---:|---|
| Remove FP32 sign-crossing bound | 0 | 1 | Caught |
| Bypass float64 validation | 0 | 1 | Caught |
| Reintroduce double -> float -> BF16 | 0 | 1 | Caught |
| Restored readers | — | 0 | Pass |

The full `tests/metal/run_test_goldwide.sh ../golden` rerun exits **0**, covering both captures in FP32 and BF16, generic/fused selection, all eight Freeway/Tetris train comparisons, and both noise probes per comparison. The CPU audit again verifies 276 cases, 3,312 tensor files and 3,645 hashes. All six committed negative controls are rejected, and the BF16 midpoint/double-rounding regression passes. Production kernels and golden files are unchanged.

The full applicable gate-suite rerun exits **0: all 31 gates pass**. The two Protein gates absent from this branch were excluded before execution. This includes both original BF16 captures, all environment gates, 20 reference score/checkpoint checks across the H64 and overlap modes, runtime/rollout gating and recovery, wide GEMM/MinGRU/Muon, all 24 existing wide mutation checks, all four wide end-to-end environment/precision pairs (byte-identical checkpoints across six settings per pair), and exhaustive BF16 projection-store checking. There were no command failures or targeted reruns. macOS Bash 3.2 syntax checks, Python compilation and `git diff --check` pass; all commands use `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` and the project venv supplies NumPy.

New evidence, relative to `wt-goldwide`:

- `build/h1024-diagnosis/review-goldwide.log`: full passing FP32/BF16 wide run and eight train comparisons;
- `build/metal/goldwide/reference_controls/*.log`: six required numerical rejections;
- `build/h1024-diagnosis/review-mutations.log` and `build/metal/goldwide/mutations/{results.json,*.log}`: all three weakened validators caught, bad-fixture exits 0, mutated gate exits 1, restored gate exits 0;
- `build/h1024-diagnosis/review-gates-run.log`, `review-gates/*.log`, and `review-summary.json`: all 31 applicable gates and the combined results.

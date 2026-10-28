# Robot-arm CUDA trajectory fixtures

Captured 2026-10-26 on a Linux x86-64 host with an RTX 3090 (sm_86), CUDA 12.8.93, `nvcc -O2
-arch=sm_86 -std=c++17` (default FMA; no fast-math). Reference checkout:
`920a88e92f333b00e6b201659129c041aaa6c10f` (`golden-envs`). Its robot-arm
sources are byte-identical to this port's base `b0f7bb84`:

| Upstream file | SHA256 |
|---|---|
| robot_arm.h | f955119339cd227eea5374b333b65c7ab3b94820e1e29347aad51bfe1c00bdbf |
| robot_arm_cuda.cuh | 9d0ecb84429182e34bb988627fb15ab952f1888debc5a092929598a64ebb84fc |
| robot_arm.cu | 0177d30a97ed554fb0f674dd6649504c98441f8cd64182c72066d622864c2c98 |

Scratch directory: `<cuda-reference>/robot-arm-oojWWcdf`. Sources were
exported with `git archive`, without modifying the reference checkout.
Builds and GPU runs used `nice -n 19`; `nvidia-smi` was checked immediately
before GPU runs. The GPU became occupied once; captures/training waited for
the other user's process to finish.

`tests/metal/robot_arm_trajectory.cpp` calls the actual CUDA kernels, with
seeds **1, 2, 43, 999**, in pick, stack, and basketball modes. Env 0 receives
zero actions; others receive deterministic integer-generated triangle waves.
Capture includes initialization and 640 steps. The committed ordinary files
retain initialization plus the first **four** steps. The reset fixture starts
two steps before each mode's timeout, and includes terminal/reset at step 2.
It is an independent capture, not synthesized from CPU output.

Each file has eight little-endian uint32 header fields:
`[0x52415452, 1, modes=3, envs=4, frames=5, width=140, obs_bytes, 0]`.
Payload is float32, shape `[mode, frame, env, 140]`, with:

- 69 observations converted to float32 for storage;
- reward and terminal;
- 69 unquantized observations computed by `ra_observe` on the GPU.

The three fixtures total **100,896 bytes**. BF16 and FP32 builds have the
same physics. Upstream **also rounds FP32 observations through BF16**;
the port preserves full FP32 observations. The comparison rounds CPU FP32
observations only for the stored-observation check, and separately compares
the unquantized observations to detect physics errors.
It also requires CPU FP32 storage to equal the raw observations exactly,
and CPU BF16 storage to equal their round-to-nearest-even conversion.

Gate: absolute raw-observation limits at steps 0–4 are
`[5e-7, 1e-6, 3e-6, 1.2e-5, 4e-5]`; stored observations `4e-5`, rewards
`1e-7`, terminals exact. No relative tolerance. Longer trajectories diverge
rapidly: even a one-ULP initial joint perturbation amplifies substantially.
Do not extend the tight gate window without studying that sensitivity.
Reset fixtures also pin the reference's different host/device basketball RNG
argument evaluation orders. The gravity-term mutation must fail this gate.

To regenerate in an isolated export on CUDA-reference, copy the capture source into
that export, configure CUDA/NCCL on PATH, and compile each precision:

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
# Set define=-DPRECISION_FLOAT for FP32, empty for BF16.
nice -n 19 nvcc -O2 -arch=sm_86 -std=c++17 -DRA_TRAJECTORY_CUDA $define \
  -Xcompiler=-Wno-narrowing --diag-suppress=2361 \
  -I. -Isrc -Ivendor -Iraylib-5.5_linux_amd64/include \
  -x cu -c tests/metal/robot_arm_trajectory.cpp -o capture.o
nice -n 19 nvcc capture.o raylib-5.5_linux_amd64/lib/libraylib.a \
  -lGL -lm -lpthread -o capture
nvidia-smi
# Run only when no other compute process is using the GPU.
nice -n 19 ./capture cuda.bin 640
nice -n 19 ./capture cuda_reset.bin 4 reset  # BF16 only
```

Trim each ordinary capture with
`python tests/metal/compare_robot_arm.py cuda.bin --trim fixture.bin --steps 4`.
See `docs/dev/investigations/robot-arm-port-report.md` and `docs/dev/investigations/robot-arm-results.json` for
the long-trajectory errors, performance and training comparison.

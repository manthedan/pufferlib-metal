# CUDA BF16 custom encoder references

The reference checkout on CUDA-reference is `<cuda-reference>/pufferlib-envs-bf16`,
branch `golden-envs-bf16`. It combines `golden-envs` (`920a88e9`) with
`golden-bf16` (`13497e22`, `da2516e4`), and carries a verbatim copy of
`tests/metal/env_golden.h` as `tests/metal_ref/env_golden.h`.

This is historical capture provenance: `env_golden.sh` was a reference-branch
helper and is not shipped in this fork. Inside that reference checkout, after
configuring CUDA/NCCL and installing parser tools:

```bash
export M4="$(command -v m4)"
bash tests/metal_ref/env_golden.sh "$PUFFER_GOLDEN_DIR/envs_bf16"
```

The generator builds default BF16 for OSRS Zulrah/PVP/Colosseum/Inferno,
Craftax, NMMO3, NetHack, Minimal and Asteroids; it writes both `run_a` and
`run_b`, `determinism.json`, and SHA256 inventory. The M4 override is needed
by the relocated bison used to build fast-nle on CUDA-reference. Native training on
that older checkout also needs `NVCC_EXTRA="-include tests/metal_ref/cuda_launch.h"`
for NetHack's launch wrappers (the current Metal source defines them itself).

Mac gates (numpy required, put the shared venv's bin first on PATH):

```bash
tests/metal/run_test_env_bf16.sh
tests/metal/run_test_env_golden.sh --bf16 ../golden/envs_bf16
PUF_ENV_GOLDEN_RUN=run_b tests/metal/run_test_env_golden.sh --bf16 ../golden/envs_bf16
# Unchanged FP32 reference gate:
tests/metal/run_test_env_golden.sh ../golden/envs
```

The primitive gate poisons destinations and changes inputs across two trials.
It checks exact transforms/fixed-point scatters and FP64 arithmetic. The
network gate uses leg 9's policy budget (4 BF16 ulp or 2^-5 tensor max), with
additional exact and FP64 checks on replay intermediates. Metal's direct
arithmetic bound remains 1 BF16 ulp or 2^-12 tensor max; NetHack GEMM+bias
accounts for both upstream stores. CUDA's own FP64 error is measured and
reported separately: cuBLAS can exceed that stricter bound, as in leg 9.
FP32 outputs from conditional sampling/PPO keep the existing 1e-5 gate;
actions, BF16 conditional sampling/importance outputs, inputs, parameters
and backward integer state are exact. Pooling
choices are independently checked against each backend's unrounded GELU
scores, including inactive rows and strict first-winner identical records.

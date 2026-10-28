> Historical development log; last recorded at commit `8c866d2f` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

## Leg 12 (Sol): BF16 env encoders

Branch `metal-opt12`, worktree `wt-opt12`, from `7538b089`; nothing pushed.
Commits `d7a1eb36` (all custom env shaders/adapters, remove FP32 fallback,
default-BF16 sweep), `fbc44a63` (BF16 replay and poisoned FP64 gates),
`e537d03c` (complete-case and exact sampling checks), plus the report/source
commit carrying Asteroids' configured H=256 golden case. Full report:
`docs/dev/optimization/opt-leg12-report.md`.

**All nine envs ported:** minimal, asteroids, craftax, nmmo3, nethack,
osrs_colosseum/inferno/zulrah/pvp. Explicit FP32 arithmetic with BF16 RNE at
CUDA stores; FP32 index/PPO scratch and int64 fixed-point sums stay intact.
OSRS keeps its unrounded FP32 GELU tile while rounding saved h1. `--float`
continues to work; `env_sweep.py` now builds default precision.

**CUDA:** CUDA-reference `<cuda-reference>/pufferlib-envs-bf16`, branch
`golden-envs-bf16`, commits `177d287c`, `3f389444`, `bce6d7ca`, `7f027ee0`.
Goldens in `<cuda-reference>/golden_envs_bf16`, Mac `golden/envs_bf16`.
Two runs, **18 cases, zero differing tensors**. Metal passes both runs:
1196 tensor comparisons + 290 independent exact/FP64 checks per run.
Full-network tolerance is leg 9's 4 BF16 ulp or 2^-5 norm; direct Metal
arithmetic passes 1 ulp or 2^-12 norm, with explicit prior-GEMM rounding
propagation for bias. Copies/scatters/index state and BF16 conditional
sampling outputs are exact. CUDA's own direct FP64 error exceeds that
stricter bound in 23 measurements (max norm 0.007497); Metal still passes
its stricter bounds. Worst Metal/CUDA network norm is 0.009346 (NetHack).

**Poisoning:** 64 primitive checks, two changed-input trials, pure outputs
poisoned every call; full BF16 allocations poisoned before first use;
requested replay cases required. Missing-write mutations for entity,
Craftax and NMMO3, and an incorrectly rounded OSRS tile, all fail.

**Training:** all nine envs, seed 1, 64 epochs with sweep geometry on Metal
BF16/FP32 and CUDA BF16; finite losses and checkpoints in all 27 runs.
Final Metal BF16 / FP32 / CUDA BF16 scores: minimal -381.43/-382.13/-383.76;
asteroids 1.80/1.84/1.33; Zulrah .159/.167/.184; PVP .00433/.00223/.00279;
Colosseum .00117/.00130/.00112; Inferno .0270/.0290/.0316;
Craftax 5.00/4.87/4.88; NMMO3 1/1/1; NetHack 2.83/3.65/2.41.
Four smaller envs also pass two-epoch actual-default-geometry FP32/BF16
smokes. These are smoke-level comparisons, not multi-seed convergence
claims; NetHack has wide final-window differences and NMMO3's metric is flat.

**Preservation:** FP32 env goldens pass; 1196 additional FP32 tensor outputs
and inputs are byte-identical with baseline `7538b089` shaders. Breakout
seed 1 is exact: FP32 **844.95672607421875 / 2ee4da82d356a44c**; BF16
**852.43658447265625 / f5dc197b8d7fde55**.

**Final verification:** **21/21 gates PASS**, supplied runner copied to
`build/gates_leg12/run.sh` with this worktree's cd and BF16 env/primitive
and seed checks. Default BF16 `tools/metal/env_sweep.sh` (no --out):
**81/85 OK**, expected drive/matsci/robot_arm/osrs only. Final commit is
recorded in `build/gates_leg12/commit.txt`; complete logs, training evidence
and mutation checks are under `build/gates_leg12/` and `build/leg12/`.

No custom encoder needs a fallback. Remaining work: longer multi-seed env
learning and BF16 encoder tuning; optional NMMO3 attention remains outside
the default convolution port. No other branches/worktrees or other users'
processes were changed.

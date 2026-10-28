> Historical development log; last recorded at commit `49e63975` (tree archived from `e68239d9`). Statements and gate counts describe that revision, not current support.

# Training log history fix

2026-10-26, Mac mini M4, macOS 27, branch `metal-logfix`, base `b0f7bb84`.
Work is confined to `<repo>/wt-logfix`. Existing untracked snake files
are excluded. Builds use Xcode's `DEVELOPER_DIR` and the mini's Python venv.
Implementation and regression tests: commit `8d3335b8`.

## Causes and upstream comparison

The pinned upstream reference is `tests/metal/fixtures/log_history/upstream_log_history_bin_mean.cu.inc` (extracted in cleanup leg 2); the CLI parity audit
in `docs/dev/investigations/protein-port-report.md` establishes that the native trainer supplies
both saved INI arrays and Protein's downsampled observations.

1. **Incorrect bin assignment is upstream-shared.** Both implementations
   added the current sample before checking a boundary, closed at most one
   bin per sample, then incremented the boundary once. A sample jumping many
   boundaries was assigned to an earlier bin, and subsequent samples kept
   filling displaced bins. Remaining bins repeated the last mean. The exact
   last array element was overwritten with the last history value, but that
   did not repair preceding bins or stale history values.
2. **History admission depends on the objective; stale trainer snapshots are
   a port-side defect.** Both trainers throttle ordinary logs to 0.6 seconds;
   both explicitly bypass that throttle for the final epoch. `vec_log` emits
   `env/n=0` but omits other episode metrics when no episode has completed.
   Upstream replaces `last_log` wholesale and appends only if the objective
   exists. Thus upstream can also omit the final epoch from the training
   curve when the objective is an episode metric. Its final trainer fields
   remain current, but the objective disappears (result score falls back to
   zero, and dense-key artifact binning can fail on a missing episode key).
   The port instead kept the entire previous episode-bearing snapshot,
   updated only `agent_steps` and `uptime`, and required both completed
   episodes and a newly present objective before appending. Consequently
   `epoch`, performance timings, utilization, losses, and `env/n` could all
   stay stale. The final artifact append copied that stale snapshot. This
   explains why forcing a final append did not fix leg 20's epoch 1115
   versus actual epoch 1144. A non-episode sweep objective was unnecessarily
   episode-gated in the port as well.

## Resulting behavior

For `P > 1`, let `F` be the final history step and `w = F / (P - 1)`.
The first `P-1` outputs are arithmetic sample means in the half-open intervals
`[j*w, (j+1)*w)`. A sample exactly on a boundary belongs to the following bin.
The last output is the exact final snapshot, not a mean. Before assigning a
sample, the implementation closes **all** earlier bins. Empty bins carry the
last non-empty bin's mean; leading empty bins use the key's first observed
value. Absent keys contribute no samples. This also supports episode keys
that first appear after a non-episode objective has already started history;
the old dense-key reader would fail on those missing keys.
`P == 1` still returns only the last value. Single-sample and zero-step
histories remain defined. These rules intentionally correct upstream's
boundary behavior even for dense histories.

Each actual logging event now refreshes every field present in `new_log`.
Only absent fields, normally episode statistics, carry forward. In particular,
`env/n` becomes zero when no new episodes completed, while the last observed
episode score remains available. Once the objective has appeared, every
logging event enters history, including the unthrottled final epoch. There
is no additional rollout, evaluation, synchronization, or change to training
kernels, RNG, optimizer updates, logging cadence, or checkpoint writes.

Protein uses the same binning and complete final training snapshot as the
INI. Carried episode scores are observations of the last known statistic,
not newly completed episodes; intermediate means and costs can change from
upstream and from the old port, so subsequent sweep suggestions can change.
Negative objective values remain valid. If no objective ever appeared, the
existing zero-score fallback is returned as **one** point at actual final
cost and steps. Previously `result.points` advertised the requested count
while only slot zero was populated, exposing spurious zero-cost/zero-step
observations (also upstream-shared). Existing final evaluation/self-play
score overlays are retained. Wall-clock metrics naturally vary by run.

## Regression tests

`tests/metal/run_test_log_history.sh` is registered as `log_history` in the
tracked `tests/metal/run_all.sh`.

* `test_log_history.py` extracts the actual host-only history functions from
  the single-file trainer and compiles `test_log_history.c` with Clang C11,
  AddressSanitizer, and UndefinedBehaviorSanitizer. Nine hand-calculated
  fixtures cover dense data, exact boundaries, sparse data, skipped bins,
  carry-forward of a mean, leading gaps, one sample, one point, zero steps,
  a duplicate final evaluation overlay, and late/absent keys. A separate
  snapshot assertion checks fresh epoch/timing fields and retained negative
  episode score when `env/n=0`. The numeric comparator explicitly rejects
  NaN; an all-NaN binning mutant exits 1 instead of passing through `fabs`.
* The mutation substitutes the exact old binning function from the pinned
  upstream file in an isolated generated header. The same test exits 1,
  including sparse-bin failures. Production files are never mutated.
* The gate builds FP32 Freeway; `test_log_history_train.py` runs it in fresh
  artifact directories. The main case uses 512 agents, horizon 64, H128/L4, seed 1,
  60 epochs (1,966,080 steps), 32 output points, and no final evaluation.
  The episode case enables the test-only `PUF_TEST_NO_INTERMEDIATE_LOG`
  schedule: it logs every epoch until the objective first appears, then
  suppresses ordinary logs until the forced final epoch. The resulting curve
  has exactly two distinct epochs (31 and 60), so the final `env/n=0`, fresh
  epoch, advancing cost, and paired step/epoch checks do not depend on the
  0.6-second wall-clock throttle.
* A two-epoch run covers no completed episodes (final epoch 2, 65,536 steps,
  one explicit Protein fallback). Another 60-epoch run uses the trainer
  objective `metal/epoch_ms`, independent of episode completion, and checks
  episode keys that first appear later. The test reads the real `TrainResult`
  result-fd payload and compares its score, cost, and step arrays with the
  INI, allowing only FP32 serialization rounding.

The final logging gate passes. Its host evidence is
`build/metal/log_history/host-zw8b5yg2/`; real training evidence is
`build/metal/log_history/train-tkb7y1zt/`. The scheduled episode run ended
with epochs `[31, 31, 31, 60]` and final `env/n=0`. A separately compiled
stale-snapshot mutant (old snapshot plus only fresh `agent_steps`/`uptime`)
failed the same training regression with all 32 epoch outputs stuck at 31;
the production source was restored before the final runners.

## Full gates

Command (Bash 3.2; no external `timeout`):

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export PATH="$HOME/metal_puffer/.venv/bin:$PATH"
export PUFFER_GOLDEN_DIR="$HOME/metal_puffer/golden"
bash tests/metal/run_all.sh > build/metal/log_history/gates-summary.log 2>&1
```

**Final run: 40/40 gates PASS, runner exit status 0, ending in `GATES DONE`.**
This includes the updated `log_history` gate, Protein/sweep CLI, FP32/BF16
replay and environment goldens, all wide/large mutation and end-to-end gates,
and `goldwide`. With the copied external fixtures, `goldwide` verified 276
cases and 3,645 SHA256 files (1,941,212,704 raw tensor bytes); its six negative
controls were rejected. Individual gate logs are `build/<gate>.log`.

All 20 seeded score/checkpoint comparisons in `h64_e2e` and `overlap_e2e`
passed: default/opt-out for Pong and Flappy, and default/forced/opt-out for
Cartpole and Breakout, each in both precisions. Every reference score and
checkpoint SHA256 prefix is unchanged:

| Environment | Precision | Score | SHA256 prefix |
|---|---|---:|---|
| Pong | BF16 | 20.996614456176758 | b1425713d9e71433 |
| Pong | FP32 | 20.99445915222168 | 5ee574ef2f33c6dd |
| Flappy | BF16 | 50.560173034667969 | 91886134a50625e1 |
| Flappy | FP32 | 51.408348083496094 | 4eafd59e547e43e1 |
| Cartpole | BF16 | 199.96110534667969 | b10ccdb69f45c996 |
| Cartpole | FP32 | 200 | bf0330663b50b01e |
| Breakout | BF16 | 852.43658447265625 | f5dc197b8d7fde55 |
| Breakout | FP32 | 844.95672607421875 | 2ee4da82d356a44c |

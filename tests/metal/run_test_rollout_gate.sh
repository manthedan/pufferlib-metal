#!/bin/bash
# Pipelined gated rollout steps (pufferl.cu metal_gated_horizon) must be
# bit-exact: final checkpoints with PUF_METAL_ROLLOUT_GATE=0 and default
# (gated) must match byte for byte, for grouped (default: four-buffer
# groups once the learner-slack policy picks them, pairs before),
# pairs (PUF_METAL_ROLLOUT_GROUP=2), four-buffer groups throughout (=4),
# ungrouped (=1) and single-buffer async rollouts. Also:
#   - the gated runs really gate (metal_summary gated_commits > 0) and never
#     hit the early-release path;
#   - PUF_METAL_TEST_GATE_HOLD=N stalls the env step of the N-th gated step
#     4.5 s, past Metal's ~4 s queue-wait release: the recovery path must run
#     (stderr "released early") and the weights must still match, including
#     forced four-buffer groups so recovery must re-encode every buffer;
#   - PUF_METAL_TEST_STEP_HOLD=1 holds each step's GPU work ~3 ms: the env
#     step must wait for its own step, not the gated next one.
# Deterministic slack-policy cases call metal_dw_policy in test_bwd_overlap.cpp
# (run_test_bwd_overlap.sh, also included by the existing GEMM gate).
# Run from the repo root:
#   tests/metal/run_test_rollout_gate.sh [TRAINER_BINARY]   (default: Flappy BF16)
set -e
export DEVELOPER_DIR=${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}
mkdir -p build/metal
if [ $# -eq 0 ]; then
    TRAIN_OUT=$(mktemp -d "${PUFFER_GATE_RUN_DIR:-$PWD/build/metal}/rollout_gate.XXXXXX")
    trap 'rm -f "$TRAIN_OUT/puffer"' EXIT
    ./build.sh flappy "$TRAIN_OUT/puffer" --metal
    set -- "$TRAIN_OUT/puffer"
fi
python3 - "$1" <<'PY'
import hashlib, os, re, subprocess, sys, tempfile
from pathlib import Path

binary = Path(sys.argv[1]).resolve()
out = Path(tempfile.mkdtemp(prefix="test_rollout_gate.", dir="build/metal")).resolve()
fails = 0

def run(tag, args, env):
    root = out / tag
    e = dict(os.environ)
    for k in ("PUF_METAL_ROLLOUT_GATE", "PUF_METAL_ROLLOUT_GROUP", "PUF_METAL_TEST_GATE_HOLD",
              "PUF_METAL_TEST_STEP_HOLD", "PUF_METAL_LAZY_PROGRESS", "PUF_METAL_KPROF",
              "PUF_METAL_STRICT_FEEDBACK"):
        e.pop(k, None)
    e.update(env)
    cmd = [str(binary), "train", "--base.seed=1", "--train.total-timesteps=2000000",
           "--base.eval-episodes=0", "--base.checkpoint-interval=100000",
           "--base.log-dir=" + str(root / "logs"),
           "--base.checkpoint-dir=" + str(root / "checkpoints")] + args
    r = subprocess.run(cmd, env=e, capture_output=True, text=True)
    (out / (tag + ".log")).write_text(r.stdout + r.stderr)
    if r.returncode:
        raise SystemExit(tag + ": trainer failed, log " + str(out / (tag + ".log")))
    paths = list((root / "checkpoints").rglob("*.bin"))
    assert len(paths) == 1, paths
    m = re.search(r"gated_commits=([0-9.e+]+)", r.stdout)
    gated = float(m.group(1)) if m else 0.0
    early = r.stderr.count("released early")
    run.groups = re.findall(r"rollout groups of (\d)", r.stderr)
    return paths[0].read_bytes(), gated, early

def check(name, ok):
    global fails
    print(("  ok    " if ok else "  FAIL  ") + name, flush=True)
    if not ok: fails += 1

configs = [
    ("grouped", [], {}),
    ("pairs", [], {"PUF_METAL_ROLLOUT_GROUP": "2"}),
    ("group4", [], {"PUF_METAL_ROLLOUT_GROUP": "4"}),
    ("group1", [], {"PUF_METAL_ROLLOUT_GROUP": "1"}),
    ("single", ["--vec.num-buffers=1"], {}),
]
for name, args, env in configs:
    print(name, flush=True)
    ref, g0, _ = run(name + "_off", args, dict(env, PUF_METAL_ROLLOUT_GATE="0"))
    check("gate off: no gated commits", g0 == 0)
    on, g1, e1 = run(name + "_on", args, env)
    check("gated: %.0f gated commits/epoch" % g1, g1 > 0)
    check("gated: no early release", e1 == 0)
    check("gated: final model bytes match gate off", on == ref)
    if name == "grouped":
        # Timing-dependent (learner slack), so reported, not required: on an
        # idle M4 Flappy switches from pairs to four-buffer groups mid-run.
        print("  info  group policy decision: %s" % (run.groups or "none"), flush=True)
    if name not in ("group1", "pairs"):
        held, g2, e2 = run(name + "_hold", args, dict(env, PUF_METAL_TEST_GATE_HOLD="150"))
        check("held env step 4.5 s: early-release recovery ran (%d)" % e2, e2 >= 1)
        check("held env step 4.5 s: final model bytes match", held == ref)
    if name == "grouped":
        sh, g3, e3 = run(name + "_stephold", args, dict(env, PUF_METAL_TEST_STEP_HOLD="1"))
        check("held GPU steps: gated and no early release", g3 > 0 and e3 == 0)
        check("held GPU steps: final model bytes match", sh == ref)
    print("  " + name + " model " + hashlib.sha256(ref).hexdigest()[:16], flush=True)
print(("ALL PASS" if not fails else "%d FAILED" % fails) + " (logs: " + str(out) + ")")
sys.exit(1 if fails else 0)
PY

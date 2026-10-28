#!/usr/bin/env python3
"""End-to-end train warm-start regression and isolated skipped-load mutant."""

from pathlib import Path
import hashlib
import os
import subprocess
import sys
import tempfile


def run(cmd, log, env=None, ok=True):
    merged = dict(os.environ)
    if env:
        merged.update(env)
    with log.open("w") as fp:
        result = subprocess.run(cmd, env=merged, stdout=fp, stderr=subprocess.STDOUT)
    if ok and result.returncode != 0:
        raise AssertionError(f"command failed ({result.returncode}); see {log}")
    if not ok and result.returncode == 0:
        raise AssertionError(f"command unexpectedly passed; see {log}")
    return result


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: test_warm_start.py PRODUCTION MUTANT OUT")
    production = Path(sys.argv[1]).resolve()
    mutant = Path(sys.argv[2]).resolve()
    out = Path(sys.argv[3]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run-", dir=out)).resolve()

    common = [
        "--base.seed=11",
        "--base.eval-episodes=0",
        "--base.checkpoint-interval=100000",
        "--vec.total-agents=128",
        "--vec.num-buffers=2",
        "--vec.num-threads=2",
        "--policy.hidden-size=64",
        "--policy.num-layers=1",
        "--train.horizon=16",
        "--train.minibatch-size=2048",
        "--train.replay-ratio=1",
        "--train.total-timesteps=2048",
    ]

    def train(binary, name, extra=(), env=None, ok=True):
        dest = root / name
        cmd = [str(binary), "train", *common,
               f"--base.run-id={name}",
               f"--base.log-dir={dest / 'logs'}",
               f"--base.checkpoint-dir={dest / 'checkpoints'}", *extra]
        result = run(cmd, root / f"{name}.log", env=env, ok=ok)
        checkpoints = list((dest / "checkpoints").rglob("*.bin"))
        return result, checkpoints

    # Checkpoint creation and the unchanged default/explicit-None path.
    _, source_files = train(production, "source")
    assert len(source_files) == 1, source_files
    source = source_files[0].resolve()
    _, none_files = train(production, "explicit_none",
                          ["--base.load-model-path=None"])
    assert len(none_files) == 1, none_files
    assert source.read_bytes() == none_files[0].read_bytes(), \
        "explicit default load_model_path changed seeded checkpoint bytes"

    # The loaded checkpoint's deterministic first rollout is the reference
    # policy output. Warm-start train must produce the identical first rollout.
    eval_output = root / "eval-output.bin"
    eval_cmd = [str(production), "eval", str(source), "--headless",
                "--base.seed=11", "--base.eval-episodes=1", "--base.eval-agents=128",
                "--vec.num-buffers=2", "--vec.num-threads=2",
                "--policy.hidden-size=64", "--policy.num-layers=1",
                "--train.horizon=16", "--train.minibatch-size=2048"]
    run(eval_cmd, root / "eval.log",
        env={"PUF_WARM_START_TEST_OUTPUT": str(eval_output)})

    warm_output = root / "warm-output.bin"
    loaded_weights = root / "loaded-weights.bin"
    momentum = root / "warm-momentum.bin"
    _, warm_files = train(
        production, "warm", [f"--base.load-model-path={source}"],
        env={
            "PUF_WARM_START_TEST_OUTPUT": str(warm_output),
            "PUF_WARM_START_TEST_WEIGHTS": str(loaded_weights),
            "PUF_WARM_START_TEST_MOMENTUM": str(momentum),
        })
    assert len(warm_files) == 1, warm_files
    assert loaded_weights.read_bytes() == source.read_bytes(), \
        "warm-start master weights do not equal checkpoint"
    assert not any(momentum.read_bytes()), "warm-start restored optimizer momentum"
    assert warm_output.read_bytes() == eval_output.read_bytes(), \
        "first warm-start rollout differs from loaded policy output"
    assert "optimizer, step, replay, and RNG state are fresh" in \
        (root / "warm.log").read_text()

    # A half-sized (BF16-like) file and a different policy shape both fail with
    # the explicit format/shape diagnostic before the first rollout.
    bad_precision = root / "bad-precision.bin"
    bad_precision.write_bytes(source.read_bytes()[:source.stat().st_size // 2])
    train(production, "bad_precision",
          [f"--base.load-model-path={bad_precision}"], ok=False)
    bad_text = (root / "bad_precision.log").read_text()
    assert "checkpoint shape/precision mismatch" in bad_text
    assert "serialized FP32 master weights" in bad_text

    train(production, "bad_shape",
          [f"--base.load-model-path={source}", "--policy.hidden-size=32"], ok=False)
    assert "checkpoint shape/precision mismatch" in \
        (root / "bad_shape.log").read_text()

    mutant_output = root / "mutant-output.bin"
    train(mutant, "mutant", [f"--base.load-model-path={source}"],
          env={"PUF_WARM_START_TEST_OUTPUT": str(mutant_output)})
    assert mutant_output.read_bytes() != eval_output.read_bytes(), \
        "skipped-load mutant escaped output comparison"

    restored_output = root / "restored-output.bin"
    train(production, "restored", [f"--base.load-model-path={source}"],
          env={"PUF_WARM_START_TEST_OUTPUT": str(restored_output)})
    assert restored_output.read_bytes() == eval_output.read_bytes()

    print("warm_start: checkpoint -> first outputs exact; fresh optimizer PASS")
    print("warm_start: shape/precision errors PASS; skipped-load mutant CAUGHT; restored PASS")
    print("warm_start: default checkpoint sha256=" +
          hashlib.sha256(source.read_bytes()).hexdigest())
    print("warm_start: logs=" + str(root))


if __name__ == "__main__":
    main()

"""Serial Metal build/train smoke sweep; standard library only.

Usage: tools/metal/env_sweep.sh [ENV ...] [--resume] [--out build/metal/sweep]
No arguments enumerates every ocean/ directory, including support directories.
JSON contains exact commands and exit status; the Markdown table is generated
from those observations. A passing run must complete two learner updates.
"""
import argparse
import configparser
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time


def run(cmd, log, timeout):
    start = time.monotonic()
    with log.open("w") as f:
        f.write("$ " + shlex.join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            f.write(f"\nSWEEP_TIMEOUT after {timeout}s\n")
            code = 124
    return code, round(time.monotonic() - start, 3)


def first_error(text):
    for line in text.splitlines():
        if re.search(r"error:|Error:|Assertion|assertion|failed|Abort|SWEEP_TIMEOUT|not found|No such file", line, re.I):
            return line.strip()[:450]
    return next((s.strip()[:450] for s in reversed(text.splitlines()) if s.strip()), "no diagnostic")


def smoke_args(env, out):
    ini = configparser.ConfigParser(interpolation=None, strict=False, inline_comment_prefixes=("#", ";"))
    ini.read(["config/default.ini", f"config/{env}.ini"])
    per_env = max(1, int(float(ini.get("env", "num_agents", fallback="1").strip("[]").split(",")[0])))
    if env == "moba":
        per_env = 5 if int(ini.get("env", "script_opponents", fallback="1")) else 10
    # Respect single-buffer env constraints; keep whole multi-agent envs.
    buffers = min(2, int(ini.get("vec", "num_buffers")))
    unit = math.lcm(buffers, buffers * per_env)
    agents = ((256 + unit - 1) // unit) * unit
    values = {
        "train.total_timesteps": agents * 16 * 2,
        "vec.total_agents": agents, "vec.num_buffers": buffers, "vec.num_threads": 2,
        "vec.num_policies": 1, "vec.hist_policy_percent": 0,
        "selfplay.enabled": 0, "selfplay.eval_bot_games": 0, "selfplay.eval_games": 0,
        "train.horizon": 16, "train.minibatch_size": agents * 16,
        "policy.hidden_size": 64, "policy.num_layers": 2,
        "base.eval_episodes": 0, "base.checkpoint_dir": str(out / "checkpoints"),
        "base.log_dir": str(out / "logs"), "base.run_id": "smoke",
    }
    # train_epoch_gpu floors replay_ratio * rollout_batch / minibatch_size.
    # With a full-batch minibatch, ratios below one silently skip the learner.
    if float(ini.get("train", "replay_ratio")) < 1:
        values["train.replay_ratio"] = 1
    # Metal is single-device; multi-GPU (NCCL) defaults such as craftax's run one rank.
    if int(float(ini.get("train", "gpus", fallback="1"))) != 1:
        values["train.gpus"] = 1
    return [f"--{k}={v}" for k, v in values.items()]


def save(results, out):
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    rows = ["| Environment | Class | Observation |", "|---|---|---|"]
    for env, r in sorted(results.items()):
        detail = r["detail"].replace("|", "\\|").replace("\n", " ")
        rows.append(f"| {env} | {r['class']} | {detail} |")
    (out / "table.md").write_text("\n".join(rows) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("envs", nargs="*")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("build/metal/sweep"))
    parser.add_argument("--build-timeout", type=int, default=300)
    parser.add_argument("--train-timeout", type=int, default=45)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    out = args.out.resolve()
    if not out.is_relative_to(root):
        parser.error("--out must stay inside this worktree")
    out.mkdir(parents=True, exist_ok=True)
    results_file = out / "results.json"
    results = json.loads(results_file.read_text()) if results_file.exists() else {}
    envs = args.envs or sorted(p.name for p in Path("ocean").iterdir() if p.is_dir())
    for env in envs:
        if not re.fullmatch(r"[a-z0-9_]+", env) or not Path("ocean", env).is_dir():
            parser.error(f"unknown environment: {env}")
        if args.resume and env in results:
            continue
        binary = f"build/metal/bin_{env}"
        build = ["/bin/bash", "./build.sh", env, binary, "--metal"]
        bc, bt = run(build, out / f"{env}.build.log", args.build_timeout)
        r = {"build_command": build, "build_exit": bc, "build_seconds": bt}
        if bc:
            diagnostic = (out / f"{env}.build.log").read_text(errors="replace")
            r.update({"class": "BUILD_FAIL", "detail": first_error(diagnostic)})
            if "does not support CUDA envs" in diagnostic:
                r.update({"class": "CUDA_ONLY", "detail": "This build selected a CUDA simulator; Metal requires a CPU implementation. robot_arm --metal uses robot_arm_cpu.h."})
            elif env == "osrs" and not Path("ocean/osrs/osrs.h").exists():
                r.update({"class": "SKIPPED", "detail": "Shared OSRS support directory, no standalone environment/header/config; build attempted."})
        else:
            train = [binary, "train"] + smoke_args(env, out)
            tc, tt = run(train, out / f"{env}.train.log", args.train_timeout)
            r.update({"train_command": train, "train_exit": tc, "train_seconds": tt})
            diagnostic = (out / f"{env}.train.log").read_text(errors="replace")
            kernel = re.search(r"metal: unimplemented kernel (\w+)", diagnostic)
            nonfinite = re.search(r"(?<![\w/])(?:nan|[-+]?inf)(?!\w)", diagnostic, re.I)
            if kernel:
                r.update({"class": "UNPORTED_KERNEL", "detail": kernel.group(1)})
            elif tc or nonfinite:
                r.update({"class": "RUNTIME_FAIL", "detail": "Nonfinite train output" if nonfinite else first_error(diagnostic)})
            elif not re.search(r"Epoch\s+2\b", diagnostic):
                r.update({"class": "RUNTIME_FAIL", "detail": "Exited without evidence of two learner epochs"})
            else:
                r.update({"class": "OK", "detail": "Two learner updates completed; finite losses (small policy/batch; evaluation disabled)."})
        results[env] = r
        save(results, out)
        print(f"{env}: {r['class']} — {r['detail']}", flush=True)


if __name__ == "__main__":
    main()

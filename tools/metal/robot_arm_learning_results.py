#!/usr/bin/env python3
"""Summarize completed, full-budget robot_arm parity runs (no training work)."""
import argparse
import configparser
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


METRICS = ("success_rate", "episode_return", "grasp_rate", "lift_rate")
STEPS = (1_000_000_000 // (4096 * 128)) * (4096 * 128)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(folder, backend):
    import numpy as np

    assert (folder / "exit_code.txt").read_text().strip() == "0", folder
    paths = list((folder / "logs").rglob("*.ini"))
    assert len(paths) == 1, paths
    ini = configparser.ConfigParser(interpolation=None, strict=False)
    ini.read(paths[0])
    for section, key, expected in (
        ("train", "total_timesteps", 1_000_000_000),
        ("train", "horizon", 128), ("train", "minibatch_size", 32768),
        ("policy", "hidden_size", 256), ("policy", "num_layers", 2),
        ("base", "eval_episodes", 1024), ("base", "eval_agents", 1024),
    ):
        assert int(ini[section][key]) == expected, (folder, section, key)
    metrics = {k: [float(x) for x in v.split(",")]
               for k, v in ini["metrics"].items()}
    assert all(len(v) == 5 and all(map(math.isfinite, v))
               for v in metrics.values()), folder
    assert metrics["agent_steps"][-1] == STEPS, folder
    assert metrics["epoch"][-1] == 1907, folder
    assert metrics["env/n"][-1] >= 1024, folder
    assert metrics["env/score"][-1] == metrics["env/success_rate"][-1], folder
    checkpoints = list((folder / "checkpoints").rglob(f"{STEPS:016d}.bin"))
    assert len(checkpoints) == 1, checkpoints
    checkpoint = checkpoints[0]
    weights = np.fromfile(checkpoint, dtype="<f4")
    assert weights.size > 400000 and np.isfinite(weights).all(), checkpoint
    seed = int(ini["base"]["seed"])
    assert folder.name == f"seed{seed}"
    result = {
        "backend": backend, "seed": seed,
        "steps": STEPS, "epochs": 1907,
        "training_seconds": metrics["uptime"][-1],
        "mean_sps": STEPS / metrics["uptime"][-1],
        "eval_episodes_observed": int(metrics["env/n"][-1]),
        "eval": {k: metrics[f"env/{k}"][-1] for k in METRICS},
        "curve": {
            "mean_steps": metrics["agent_steps"][:-1],
            **{k: metrics[f"env/{k}"][:-1] for k in METRICS},
            "mean_sps": metrics["sps"][:-1],
        },
        "checkpoint_sha256": digest(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "artifact": str(paths[0]),
        "artifact_sha256": digest(paths[0]),
        "started_utc": (folder / "start.txt").read_text().strip(),
        "runner_observed_end_utc": (folder / "end.txt").read_text().strip(),
    }
    telemetry = (folder / "telemetry.txt").read_text()
    if backend == "metal":
        summary = re.search(r"^metal_summary (.*)$",
                            (folder / "train.log").read_text(), re.M)
        assert summary, folder
        result["metal_summary"] = {
            k: float(v) for k, v in re.findall(r"(\S+)=(\S+)", summary[1])
        }
        result["thermal"] = {
            "samples": telemetry.count("No thermal warning level has been recorded"),
            "performance_warning_free_samples": telemetry.count(
                "No performance warning level has been recorded"),
            "power_watts": None,
        }
    else:
        rows = []
        for line in telemetry.splitlines():
            if "," in line:
                try:
                    rows.append([float(re.match(r"\s*([\d.]+)", x)[1])
                                 for x in line.split(",")])
                except (TypeError, ValueError):
                    continue
        if rows:
            rows = [row for row in rows if row[2] > 0] or rows
            result["telemetry"] = {
                name: {"min": min(col), "mean": statistics.mean(col), "max": max(col)}
                for name, col in zip(
                    ("temperature_c", "power_w", "gpu_percent", "memory_mib", "sm_mhz"),
                    zip(*rows))
            }
    return result


def summarize(runs):
    return {
        key: {"mean": statistics.mean(values),
              "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
              "min": min(values), "max": max(values)}
        for key in METRICS
        for values in [[r["eval"][key] for r in runs]]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("build/ra"))
    parser.add_argument("--output", type=Path,
                        default=Path("build/ra/results.json"))
    parser.add_argument("--allow-pending-cuda", action="store_true",
                        help="Report completed Metal seeds with CUDA explicitly pending")
    args = parser.parse_args()
    runs = [read_run(folder, backend)
            for backend in ("metal", "cuda")
            for folder in sorted((args.root / backend).glob("seed*"))
            if (folder / "exit_code.txt").exists()]
    finished = {(r["backend"], r["seed"]) for r in runs}
    required = ("metal",) if args.allow_pending_cuda else ("metal", "cuda")
    assert finished >= {
        (backend, seed) for backend in required for seed in (42, 43, 44)
    }, "The three planned seeds must finish on each required backend"
    pending = [{"backend": backend, "seed": seed}
               for backend in ("metal", "cuda") for seed in (42, 43, 44)
               if (backend, seed) not in finished]
    result = {
        "schema_version": 1,
        "requested_steps": 1_000_000_000,
        "actual_steps": STEPS,
        "curve_definition": "Four equal-width half-open step bins, arithmetic means of logged snapshots; final eval is separate. Missing episode metrics carry forward as in logfix-report.md.",
        "runs": runs,
        "pending": pending,
        "primary_seeds_summary": {
            backend: summarize([r for r in runs if r["backend"] == backend and r["seed"] <= 44])
            for backend in ("metal", "cuda")
            if any(r["backend"] == backend and r["seed"] <= 44 for r in runs)
        },
        "all_seeds_summary": {
            backend: summarize([r for r in runs if r["backend"] == backend])
            for backend in ("metal", "cuda")
            if any(r["backend"] == backend for r in runs)
        },
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Backend | Seed | Success | Return | Grasp | Lift | SPS")
    for r in runs:
        e = r["eval"]
        print(f'{r["backend"]} | {r["seed"]} | {e["success_rate"]:.6f} | '
              f'{e["episode_return"]:.5f} | {e["grasp_rate"]:.6f} | '
              f'{e["lift_rate"]:.6f} | {r["mean_sps"]:.0f}')


if __name__ == "__main__":
    main()

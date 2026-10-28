#!/usr/bin/env python3
"""Compare captures; float32 payload has [obs69, reward, terminal, raw_obs69]."""
import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    header = np.fromfile(path, dtype="<u4", count=8)
    assert len(header) == 8 and tuple(header[:2]) == (0x52415452, 1), path
    assert tuple(header[[2, 3, 5]]) == (3, 4, 140), header
    data = np.fromfile(path, dtype="<f4", offset=32).reshape(
        int(header[2]), int(header[4]), int(header[3]), int(header[5]))
    assert np.isfinite(data).all(), f"nonfinite trajectory: {path}"
    return header, data


def bf16_round(values):
    bits = values.copy().view(np.uint32)
    bits += 0x7FFF + ((bits >> 16) & 1)
    bits &= 0xFFFF0000
    return bits.view(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path, nargs="?")
    ap.add_argument("--trim", type=Path)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    rh, ref = load(args.reference)
    if args.trim:
        assert args.steps < ref.shape[1]
        rh[4] = args.steps + 1
        with args.trim.open("wb") as out:
            out.write(rh.tobytes())
            out.write(ref[:, :args.steps + 1].tobytes())
        return
    ch, cpu = load(args.candidate)
    assert rh[6] == ch[6], "precision mismatch"
    ticks = min(ref.shape[1], cpu.shape[1])
    ref, cpu = ref[:, :ticks], cpu[:, :ticks]
    diff = np.abs(ref.astype(np.float64) - cpu.astype(np.float64))
    report = {"precision_bytes": int(rh[6]), "steps": ticks - 1, "modes": {}}
    for m, name in enumerate(("pick", "stack", "basketball")):
        rows = {}
        for t in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 600, ticks - 1):
            if t >= ticks:
                continue
            rows[str(t)] = {
                label: {"max": float(diff[m, t, :, sl].max()),
                        "mean": float(diff[m, t, :, sl].mean())}
                for label, sl in (("obs", slice(0, 69)), ("reward", slice(69, 70)),
                                  ("terminal", slice(70, 71)), ("raw_obs", slice(71, 140)))
            }
        raw_max = diff[m, :, :, 71:].max(axis=(1, 2))
        first = {}
        for limit in (1e-5, 1e-4, 1e-3, 1e-2, 0.1):
            indices = np.flatnonzero(raw_max > limit)
            first[str(limit)] = int(indices[0]) if indices.size else None
        term = np.flatnonzero(diff[m, :, :, 70].max(axis=1) != 0)
        report["modes"][name] = {"at_step": rows, "first_raw_obs_error_gt": first,
            "first_terminal_mismatch": int(term[0]) if term.size else None,
            "raw_obs_all_max": float(raw_max.max()),
            "raw_obs_all_mean": float(diff[m, :, :, 71:].mean())}
    if args.gate:
        assert args.steps == 4, "gate tolerances cover exactly the first four steps"
        assert ticks >= args.steps + 1, "capture shorter than gate window"
        # Tolerances are calibrated against independently captured CUDA data.
        # FP32 upstream stores BF16-rounded observations too: compare that path
        # after applying the same rounding, and check unquantized physics below.
        n = args.steps + 1
        obs = cpu[:, :n, :, :69]
        raw = cpu[:, :n, :, 71:]
        expected_storage = bf16_round(raw) if rh[6] == 2 else raw
        assert np.array_equal(obs, expected_storage), "CPU observation precision gate"
        if rh[6] == 4:
            obs = bf16_round(obs)
        assert np.allclose(obs, ref[:, :n, :, :69], atol=4e-5, rtol=0), "observation gate"
        for t, tol in enumerate((5e-7, 1e-6, 3e-6, 1.2e-5, 4e-5)):
            error = diff[:, t, :, 71:].max()
            assert error <= tol, f"physics gate: step {t}, error {error:g} > {tol:g}"
        assert np.allclose(cpu[:, :n, :, 69], ref[:, :n, :, 69], atol=1e-7, rtol=0), "reward gate"
        assert np.array_equal(cpu[:, :n, :, 70], ref[:, :n, :, 70]), "terminal gate"
        print(f"robot_arm trajectory PASS: {n} frames, 3 modes, 4 seeds, obs_bytes={rh[6]}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    elif not args.gate:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

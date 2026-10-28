"""Compare optional PUF_POLICY_DUMP_DIR dumps with CUDA and NumPy float64.

Usage: diagnose_policy_backward.py GOLDEN_RUN METAL_DUMP_DIR
The CUDA policy manifest has final gradients, not scan intermediates; this
reference derives the latter independently from the recorded policy inputs.
"""

import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("golden_run", type=Path)
parser.add_argument("dump_dir", type=Path)
args = parser.parse_args()
base = args.golden_run
dump = args.dump_dir
results = {}


def metric(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    err = np.abs(a - b)
    return dict(
        max_abs=float(err.max()),
        norm=float(err.max() / max(np.abs(b).max(), 1e-300)),
        rms=float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)),
        argmax=list(np.unravel_index(err.argmax(), err.shape)),
    )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


for H in [256, 512, 1024]:
    name = f"policy_layer_h{H}"
    folder = base / name
    m = json.loads((folder / "manifest.json").read_text())
    s = m["scalars"]
    B, T = s["B"], s["T"]
    d = {
        k: np.fromfile(folder / v["file"], dtype=np.float32)
        .reshape(v["shape"])
        .astype(np.float64)
        for k, v in m["tensors"].items()
        if v["dtype"] == "f32"
    }

    def metal(k, shape):
        return (
            np.fromfile(dump / f"{name}_{k}.bin", dtype=np.float32)
            .reshape(shape)
            .astype(np.float64)
        )

    x = d["obs"] @ d["param_0"].T
    c = x @ d["param_2"].T
    h = d["initial_state"][0, s["agent_off"] : s["agent_off"] + B].copy()
    hp = np.zeros_like(x)
    y = np.zeros_like(x)
    for t in range(T):
        h = np.where(d["terminals"][:, t, None] != 0, 0, h)
        hp[:, t] = h
        hid, z, p = np.split(c[:, t], 3, axis=-1)
        z, p = sigmoid(z), sigmoid(p)
        ht = np.where(hid >= 0, hid + 0.5, sigmoid(hid))
        h = h + (ht - h) * z
        y[:, t] = p * h + (1 - p) * x[:, t]
    g = np.concatenate((d["dlogits"], d["dvalue"]), axis=-1) @ d["param_1"]
    dh = d["grad_next_state"][:, 0].copy()
    gc = np.zeros_like(c)
    gi = np.zeros_like(x)
    for t in reversed(range(T)):
        hid, z, p = np.split(c[:, t], 3, axis=-1)
        z, p = sigmoid(z), sigmoid(p)
        ht = np.where(hid >= 0, hid + 0.5, sigmoid(hid))
        h = hp[:, t] + (ht - hp[:, t]) * z
        total = dh + g[:, t] * p
        gc[:, t] = np.concatenate(
            (
                total * z * np.where(hid >= 0, 1, ht * (1 - ht)),
                total * (ht - hp[:, t]) * z * (1 - z),
                g[:, t] * (h - x[:, t]) * p * (1 - p),
            ),
            axis=-1,
        )
        gi[:, t] = g[:, t] * (1 - p)
        dh = np.where(d["terminals"][:, t, None] != 0, 0, total * (1 - z))
    dw = gc.reshape(-1, 3 * H).T @ x.reshape(-1, H)
    dx = gc @ d["param_2"] + gi
    ew = dx.reshape(-1, H).T @ d["obs"].reshape(-1, s["input_dim"])
    refs = dict(
        enc_out=x,
        combined=c,
        scan_h=hp,
        dec_grad_in=g,
        grad_combined=gc,
        grad_input=gi,
        grad_state=dh,
        dw=dw,
        encoder_grad=dx,
        enc_dw=ew,
    )
    out = {}
    for key, ref in refs.items():
        out[key + "/metal_vs_f64"] = metric(metal(key, ref.shape), ref)
    for key, ref in [
        ("param_2_grad", dw),
        ("encoder_grad", dx.reshape(-1, H)),
        ("param_0_grad", ew),
        ("layer_0_out", y),
        ("layer_0_grad_state", dh[:, None]),
    ]:
        out[key + "/cuda_vs_f64"] = metric(d[key], ref)
    mc = metal("combined", c.shape)
    idx = np.argwhere((mc[..., :H] >= 0) != (c[..., :H] >= 0))
    out["branch_mismatches"] = len(idx)
    out["smallest_hidden"] = float(np.min(np.abs(c[..., :H])))
    for i in idx[:30]:
        i = tuple(i)
        print("SIGN FLIP", H, i, c[i], mc[i])
    # isolate Metal GEMMs from all upstream differences
    mgc = metal("grad_combined", gc.shape)
    mx = metal("enc_out", x.shape)
    mgi = metal("grad_input", gi.shape)
    out["dw/metal_vs_f64_same_operands"] = metric(
        metal("dw", dw.shape), mgc.reshape(-1, 3 * H).T @ mx.reshape(-1, H)
    )
    out["dx/metal_vs_f64_same_operands"] = metric(
        metal("encoder_grad", dx.shape), mgc @ d["param_2"] + mgi
    )
    results[name] = out
    # Independently predict downstream error from derivative branches only.
    delta = np.zeros_like(gc)
    for b, t, h in idx:

        def slope(v):
            z = sigmoid(v)
            return 1.0 if v >= 0 else z * (1 - z)

        delta[b, t, h] = gc[b, t, h] * (slope(mc[b, t, h]) / slope(c[b, t, h]) - 1)
    ddw = delta.reshape(-1, 3 * H).T @ x.reshape(-1, H)
    ddx = delta @ d["param_2"]
    dew = ddx.reshape(-1, H).T @ d["obs"].reshape(-1, s["input_dim"])
    out["dw/branch_accounted_vs_cuda"] = metric(
        metal("dw", dw.shape), d["param_2_grad"] + ddw
    )
    out["dx/branch_accounted_vs_cuda"] = metric(
        metal("encoder_grad", dx.shape), d["encoder_grad"].reshape(dx.shape) + ddx
    )
    out["enc_dw/branch_accounted_vs_cuda"] = metric(
        metal("enc_dw", ew.shape), d["param_0_grad"] + dew
    )
    np.savez(dump / f"{name}_reference.npz", **refs)
print(json.dumps(results, indent=2, default=lambda x: int(x)))
(dump / "comparison.json").write_text(
    json.dumps(results, indent=2, default=lambda x: int(x)) + "\n"
)

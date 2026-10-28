"""Independent BF16 arithmetic checks on the actual replay intermediates.

One BF16 ulp (or 2^-12 of tensor max for cancellation) is the leg-9 direct
GEMM budget. Copies, categorical transforms and fixed-point scatters are exact.
Each backend is checked using its own intermediate inputs; backward uses the
CUDA saved forward state, as the replay does. No cross-backend network error
can excuse an incorrect primitive here.
"""

import numpy as np


def rne(x):
    u = np.asarray(x, dtype=np.float32).copy().view(np.uint32)
    u += np.uint32(0x7FFF) + ((u >> 16) & 1)
    return (u & np.uint32(0xFFFF0000)).view(np.float32)


def spacing(x):
    x = np.abs(np.asarray(x, dtype=np.float64))
    return np.exp2(np.maximum(-126, np.floor(np.log2(np.maximum(x, 2.0**-126)))) - 7)


def validate(data, golden, env, groups=None, reference=False):
    rows = {}

    def exact(name, got, want):
        rows[name] = {"pass": bool(np.array_equal(got, want)), "exact": True}

    def rounded(name, got, want, extra=0):
        want = np.asarray(want, dtype=np.float64).reshape(got.shape)
        err = np.abs(got.astype(np.float64) - want)
        scale = float(np.max(np.abs(want)))
        bound = np.maximum(spacing(want) + extra, 2.0**-12 * scale)
        ok = bool(np.all(np.isfinite(got)) and np.all(err <= bound))
        rows[name] = {
            "max_abs_vs_fp64": float(err.max()),
            "max_ref": scale,
            "norm_vs_fp64": float(err.max()) / scale if scale else float(err.max()),
            "budget": "1 bf16 ulp or 2^-12 tensor max, plus prior GEMM rounding for bias",
        }
        if reference:
            # cuBLAS can exceed one ulp (leg 9); measure its error, rather
            # than weakening the Metal-vs-fp64 gate to match cuBLAS.
            rows[name].update(reference_measurement=True, within_budget=ok)
        else:
            rows[name]["pass"] = ok

    def f(i):
        return data[f"acts_fwd_{i:02d}"]

    def b(i):
        return data[f"acts_bwd_{i:02d}"]

    def gf(i):
        return golden[f"acts_fwd_{i:02d}"]

    def p(i):
        return data[f"param_{i:02d}"].astype(np.float64)

    def grad(i):
        return data[f"grads_{i:02d}"]

    obs, g = data["obs"], data["grad"].astype(np.float64)
    B, H = g.shape
    if env.startswith("osrs_"):
        rounded("global wgrad GEMM", grad(0), g.T @ obs.astype(np.float64))
        exact("saved obs", f(1), obs)
        for am_name, (flat_name, h1_name, l2_name, active_w) in (groups or {}).items():
            ai = int(am_name.rsplit("_", 1)[1])
            fi = int(flat_name.rsplit("_", 1)[1])
            hi = int(h1_name.rsplit("_", 1)[1])
            wi = int(l2_name.rsplit("_", 1)[1])
            R = f(fi).shape[0] // B
            exact(f"gather {fi}", f(fi), gf(fi))
            rounded(f"L1 GEMM {fi}", f(fi + 1), f(fi).astype(np.float64) @ p(wi - 1).T)
            z = f(fi + 1).astype(np.float64)
            hp = 0.5 * z * (1 + np.tanh(0.7978845608028654 * (z + 0.044715 * z**3)))
            rounded(f"saved GELU {hi}", f(hi), hp)
            am = gf(ai)
            hp = gf(hi).astype(np.float64).reshape(B, R, 16)
            wg = np.zeros_like(p(wi))
            dz = np.zeros((B, R, 16))
            for h in range(H):
                active = am[:, h] >= 0
                ids = np.maximum(0, am[:, h])
                wg[h] = (g[:, h, None] * hp[np.arange(B), ids] * active[:, None]).sum(
                    axis=0
                )
                dz[np.arange(B), ids] += g[:, h, None] * p(wi)[h] * active[:, None]
            z = gf(fi + 1).astype(np.float64).reshape(B, R, 16)
            t = np.tanh(0.7978845608028654 * (z + 0.044715 * z**3))
            dz *= 0.5 * (1 + t) + 0.5 * z * (1 - t * t) * 0.7978845608028654 * (
                1 + 3 * 0.044715 * z * z
            )
            rounded(f"L2 wgrad {wi}", grad(wi), wg)
            rounded(f"GELU grad {fi}", b(hi + 1), dz)
            rounded(
                f"L1 wgrad GEMM {wi - 1}",
                grad(wi - 1),
                b(hi + 1).astype(np.float64).T @ gf(fi).astype(np.float64),
            )
    elif env == "nethack":
        # Registration names provide schema indices only. Arithmetic below is
        # independent fp64, with the two explicit upstream BF16 stores.
        import re
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2] / "ocean/nethack/nethack.cu"
        ).read_text()

        def fields(fn, allocator):
            body = source.split("static void " + fn + "(", 1)[1].split("\n}", 1)[0]
            return re.findall(
                r"alloc_register\(" + allocator + r",\s*&\w+->(\w+)\)", body
            )

        act_names = fields("nethack_encoder_reg_train", "acts") + [
            "dec_" + n for n in fields("nethack_decoder_reg_train", "acts")
        ]
        param_names = fields("nethack_encoder_reg_params", "alloc") + [
            "dec_" + n for n in fields("nethack_decoder_reg_params", "alloc")
        ]
        af = {n: f(i) for i, n in enumerate(act_names)}
        pw = {n: p(i) for i, n in enumerate(param_names)}
        blocks = [
            ("x_local", "loc_w", "loc_b", "loc_h1"),
            ("loc_h1", "loc2_w", "loc2_b", "loc_out"),
            ("terr_tf", "terr1_w", "terr1_b", "terr_h"),
            ("terr_h", "terr2_w", "terr2_b", "glb_out"),
            ("bl_feats", "bl_w", "bl_b", "bl_out"),
            ("inv_out", "mv1_w", "mv1_b", "mh1"),
            ("mh1", "mv2_w", "mv2_b", "mvv"),
            ("lm_tok", "mr_w", "mr_b", "mrp"),
            ("mrp", "mm1_w", "mm1_b", "mh1m"),
            ("mh1m", "mm2_w", "mm2_b", "mvm"),
            ("li_tok", "ir_w", "ir_b", "irp"),
            ("irp", "im1_w", "im1_b", "mh1i"),
            ("mh1i", "im2_w", "im2_b", "mvi"),
            ("spk_keys", "spm1_w", "spm1_b", "sph1"),
            ("sph1", "spm2_w", "spm2_b", "spv"),
            ("concat", "proj_w", "proj_b", "out"),
        ]
        for inp, weight, bias, out in blocks:
            w = pw[weight]
            mm = af[inp].reshape(-1, w.shape[1]).astype(np.float64) @ w.T
            want = np.maximum(0, rne(mm).astype(np.float64) + pw[bias])
            rounded(
                "GEMM+bias/relu " + out, af[out].reshape(want.shape), want, spacing(mm)
            )
        for inp, weight, out in [
            ("out", "dec_lin_w", "dec_tmp"),
            ("out", "dec_q_w", "dec_q"),
            ("inv_out", "dec_k_w", "dec_kmat"),
        ]:
            w = pw[weight]
            want = af[inp].reshape(-1, w.shape[1]).astype(np.float64) @ w.T
            rounded("decoder GEMM " + out, af[out].reshape(want.shape), want)
        for name in [
            "glyph_idx",
            "crop_glyph",
            "inv_idx",
            "invt_idx",
            "spell_idx",
            "ide_idx",
            "msg_ids",
            "lm_gid",
            "li_gid",
        ]:
            exact("indices " + name, af[name], gf(act_names.index(name)))
    elif env in ("minimal", "asteroids"):
        S, F, R = (2, 4, 16) if env == "minimal" else (4, 5, 26)
        points = np.concatenate(
            [np.broadcast_to(obs[:, None, :S], (B, R, S)), obs[:, S:].reshape(B, R, F)],
            axis=2,
        )
        exact("materialize", f(0), points)
        rounded(
            "entity GEMM", f(1), points.reshape(-1, S + F).astype(np.float64) @ p(0).T
        )
        x = np.maximum(0, f(1).astype(np.float64))
        scores = x @ p(1).T
        exact("pool argmax", f(4), scores.argmax(axis=1))
        rounded("pool", data["out"], scores.max(axis=1))
        xg = np.maximum(0, gf(1).astype(np.float64))
        am = gf(4)
        wg = np.zeros_like(p(1))
        dx = np.zeros_like(xg)
        for h in range(H):
            wg[h] = (g[:, h, None] * xg[np.arange(B), am[:, h]]).sum(axis=0)
            dx[np.arange(B), am[:, h]] += g[:, h, None] * p(1)[h]
        dx *= gf(1) > 0
        rounded("output wgrad", grad(1), wg)
        rounded("entity grad", b(3), dx)
        rounded(
            "input wgrad GEMM",
            grad(0),
            b(3).reshape(-1, 16).astype(np.float64).T
            @ gf(0).reshape(-1, S + F).astype(np.float64),
        )
    elif env == "craftax":
        offsets = np.array([0, 64, 72, 74, 90, 106, 122, 138])
        ids = obs[:, :792].astype(int).reshape(B, 99, 8) + offsets
        concat = np.concatenate(
            [rne(p(0)[ids].sum(axis=2)).reshape(B, -1), obs[:, 792:]], axis=1
        )
        exact("embedding bag and scalars", f(0), concat)
        exact("saved obs", f(2), obs)
        rounded("projection GEMM", data["out"], concat.astype(np.float64) @ p(1).T)
        rounded("projection wgrad GEMM", grad(1), g.T @ gf(0).astype(np.float64))
        rounded("concat grad GEMM", b(0), g @ p(1))
        sums = np.zeros((154, 16), dtype=np.int64)
        q = np.rint(
            b(0)[:, :1584].astype(np.float64).reshape(B, 99, 16) * 2**24
        ).astype(np.int64)
        for ch in range(8):
            np.add.at(sums, ids[:, :, ch].ravel(), q.reshape(-1, 16))
        exact("fixed-point scatter", b(3), sums)
        exact("fixed-point flush", grad(0), rne(sums.astype(np.float64) / 2**24))
    elif env == "nmmo3":
        exact("saved obs", f(11), obs)
        rounded("conv1 GEMM", f(3), f(2).astype(np.float64) @ p(0).T)
        exact(
            "conv1 transpose/relu",
            f(0).reshape(B, 128, 12),
            np.maximum(0, f(3).reshape(B, 12, 128).transpose(0, 2, 1)),
        )
        rounded("conv2 GEMM", f(7), f(6).astype(np.float64) @ p(1).T)
        exact(
            "conv2 transpose",
            f(4).reshape(B, 128, 2),
            f(7).reshape(B, 2, 128).transpose(0, 2, 1),
        )
        ids = obs[:, 1650:1697].astype(int)
        exact("embedding copy", f(8).reshape(B, 47, 32), p(2)[ids])
        concat = np.concatenate(
            [
                f(4).reshape(B, 256),
                f(8).reshape(B, 1504),
                obs[:, 1650:1697],
                obs[:, -10:],
            ],
            axis=1,
        )
        exact("concat", f(9), concat)
        rounded(
            "projection GEMM/relu",
            data["out"],
            np.maximum(0, concat.astype(np.float64) @ p(3).T),
        )
        g *= gf(10) > 0
        rounded("projection wgrad GEMM", grad(3), g.T @ gf(9).astype(np.float64))
        rounded("concat grad GEMM", b(9), g @ p(3))
        exact("conv2 grad slice", b(5).reshape(B, 256), b(9)[:, :256])
        exact(
            "conv2 grad transpose",
            b(7),
            b(5).reshape(B, 128, 2).transpose(0, 2, 1).reshape(-1, 128),
        )
        rounded(
            "conv2 wgrad GEMM",
            grad(1),
            b(7).astype(np.float64).T @ gf(6).astype(np.float64),
        )
        rounded("conv2 col grad GEMM", b(6), b(7).astype(np.float64) @ p(1))
        exact(
            "conv1 grad transpose",
            b(3),
            b(1).reshape(B, 128, 12).transpose(0, 2, 1).reshape(-1, 128),
        )
        rounded(
            "conv1 wgrad GEMM",
            grad(0),
            b(3).astype(np.float64).T @ gf(2).astype(np.float64),
        )
        sums = np.zeros((128, 32), dtype=np.int64)
        q = np.rint(
            b(9)[:, 256:1760].astype(np.float64).reshape(B * 47, 32) * 2**24
        ).astype(np.int64)
        np.add.at(sums, ids.ravel(), q)
        exact("fixed-point scatter", b(12), sums)
        exact("fixed-point flush", grad(2), rne(sums.astype(np.float64) / 2**24))
    return rows

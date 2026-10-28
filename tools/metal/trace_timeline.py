#!/usr/bin/env python3
"""Summarize a PUF_METAL_TRACE timeline of an async CPU-env training run.

usage: PUF_METAL_TRACE=/tmp/t.csv ./puffer train ...
       tools/metal/trace_timeline.py /tmp/t.csv [first_epoch last_epoch]

The CSV (src/metal/runtime.mm trace_dump) holds every command-buffer commit
(kind 0: host commit window t0..t1, GPU span g0..g1 from commit feedback, n =
dispatches), every blocking host wait (kind 1) and trace_mark() records
(kind 2) from pufferl.cu: tags 1-6 per rollout worker step (sid = buffer,
n = t: 1 start, 2 encoded, 3 committed, 4 wait returned, 5 timing events
read, 6 env stepped), 11/12/13 learner (train_impl start, encoded, synced),
14 rollout_finish; kind 3 records gate_open of a gated commit (commit
tag 2), whose queue time then counts from the gate's release. Times are mach-absolute seconds, the clock of the
commit feedback GPU timestamps. Reports, over the middle half of the
epochs by default: the learner's command buffers, rollout commit queueing,
GPU spans and wake-up latency (split by overlap with learner GPU spans),
per-step host phases and the GPU-busy union.
"""
import csv, sys, statistics as st
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    for k in ("kind","sid","tag","n","v"): r[k] = int(r[k])
    for k in ("t0","t1","g0","g1"): r[k] = float(r[k])
commits = [r for r in rows if r["kind"] == 0]
waits = [r for r in rows if r["kind"] == 1]
marks = [r for r in rows if r["kind"] == 2]
# kind 3: gate_open of gated commit (sid, v) (tag 1 released, 0 cancelled)
gate_open = {(r["sid"], r["v"]): r["t0"] for r in rows if r["kind"] == 3}
# stream roles
disp = defaultdict(int); ncom = defaultdict(int)
for c in commits: disp[c["sid"]] += c["n"]; ncom[c["sid"]] += 1
print("streams (sid: commits, dispatches):", {k: (ncom[k], disp[k]) for k in sorted(disp)})
# epoch windows from marks 11 (train begin), 14 (rollout finish)
ep = {}
last = None
for m in marks:
    if m["tag"] == 11: last = m["n"]
    if m["tag"] == 12 and last is not None: ep.setdefault(last, {})[12] = m["t0"]
    elif m["tag"] in (11, 13, 14): ep.setdefault(m["n"], {})[m["tag"]] = m["t0"]
eps = sorted(e for e in ep if 11 in ep[e] and 14 in ep[e])
a = int(sys.argv[2]) if len(sys.argv) > 2 else eps[len(eps)//4]
b = int(sys.argv[3]) if len(sys.argv) > 3 else eps[3*len(eps)//4]
sel = [e for e in eps if a <= e <= b]
T0 = ep[sel[0]][11]; T1 = ep[sel[-1]][14]
print(f"epochs {sel[0]}..{sel[-1]} ({len(sel)}), wall per epoch {(T1-T0)/len(sel)*1e3:.2f} ms")
# train stream = the one with most dispatches per commit
train_sid = max(disp, key=lambda k: disp[k] / ncom[k])
inwin = lambda t: T0 <= t <= T1
tc = [c for c in commits if c["sid"] == train_sid and inwin(c["t0"])]
print(f"train sid {train_sid}: {len(tc)/len(sel):.1f} commits/epoch, GPU busy {sum(c['g1']-c['g0'] for c in tc)/len(sel)*1e3:.2f} ms/epoch")
for e in sel[:3]:
    w = ep[e]
    print(f"  epoch {e}: train_impl {(w[12]-w[11])*1e3:.2f} ms encode, train sync at +{(w[13]-w[11])*1e3:.2f}, rollout_finish at +{(w[14]-w[11])*1e3:.2f}")
    for c in tc:
        if w[11] <= c["t0"] <= w[14]:
            print(f"     train cb n={c['n']:4d} commit +{(c['t0']-w[11])*1e3:7.3f} gpu [{(c['g0']-w[11])*1e3:7.3f}, {(c['g1']-w[11])*1e3:7.3f}] dur {(c['g1']-c['g0'])*1e3:.3f}")
train_iv = [(c["g0"], c["g1"]) for c in tc]
def overlaps(g0, g1):
    return any(x0 < g1 and g0 < x1 for x0, x1 in train_iv)
# per rollout step
mk = {}
for m in marks:
    if m["tag"] <= 6: mk[(m["sid"], m["n"], m["tag"], )] = mk.get((m["sid"], m["n"], m["tag"]), []) + [m["t0"]]
# Rollout streams: one per worker buffer, the most-committing non-train ones
# (excludes the default and copy streams).
nbuf = len({m["sid"] for m in marks if m["tag"] <= 6})
roll_sids = sorted((k for k in ncom if k != train_sid), key=lambda k: -ncom[k])[:nbuf]
rc = defaultdict(list)
for c in commits:
    if c["sid"] in roll_sids and inwin(c["t0"]): rc[c["sid"]].append(c)
steps = defaultdict(list)
for sid, cs in rc.items():
    for c in cs:
        if c["n"] == 0: continue
        # Gated commits (tag 2) wait for the host's gate_open: queue from then.
        ready = max(c["t1"], gate_open.get((c["sid"], c["v"]), c["t1"])) if c["tag"] == 2 else c["t1"]
        steps["gated"].append(c["tag"] == 2)
        if c["tag"] == 2: steps["precommit_us"].append((ready - c["t1"]) * 1e6)
        steps["queue_us"].append((c["g0"] - ready) * 1e6)
        steps["gpu_us"].append((c["g1"] - c["g0"]) * 1e6)
        steps["contended"].append(overlaps(c["g0"], c["g1"]))
        # wake: first wait on this sid with v>=c.v ending after g1
        steps["commit_call_us"].append((c["t1"] - c["t0"]) * 1e6)
wk = defaultdict(list)
for w in waits:
    if w["sid"] in roll_sids and inwin(w["t0"]): wk[w["sid"]].append(w)
cmap = {(c["sid"], c["v"]): c for cs in rc.values() for c in cs}
wake = []
for sid, ws in wk.items():
    for w in ws:
        c = cmap.get((sid, w["v"]))
        if c and c["g1"] > 0: wake.append((w["t1"] - c["g1"]) * 1e6)
def summ(name, xs):
    xs = sorted(xs)
    if not xs: return
    print(f"  {name:<22} n={len(xs):6d} mean {st.mean(xs):8.1f}  p50 {xs[len(xs)//2]:8.1f}  p10 {xs[len(xs)//10]:8.1f}  p90 {xs[9*len(xs)//10]:8.1f}")
print("rollout commits (us):" + (f" {sum(steps['gated'])} of {len(steps['gated'])} gated: queue from gate_open" if any(steps["gated"]) else ""))
summ("commit->gpu start", steps["queue_us"])
summ("gated commit->open", steps["precommit_us"])
summ("gpu duration", steps["gpu_us"])
summ("commit call", steps["commit_call_us"])
summ("gpu end->host wake", wake)
cont = steps["contended"]
print(f"  steps overlapping learner GPU: {sum(cont)}/{len(cont)}")
qc = [q for q, c in zip(steps["queue_us"], cont) if c]; qn = [q for q, c in zip(steps["queue_us"], cont) if not c]
summ("queue (contended)", qc); summ("queue (uncontended)", qn)
gc = [q for q, c in zip(steps["gpu_us"], cont) if c]; gn = [q for q, c in zip(steps["gpu_us"], cont) if not c]
summ("gpu (contended)", gc); summ("gpu (uncontended)", gn)
# host phases per step from marks
ph = defaultdict(list)
keys = sorted({(k[0], k[1]) for k in mk})
for (buf, t) in keys:
    for i, ts in enumerate(mk.get((buf, t, 1), [])):
        try:
            v = [mk[(buf, t, g)][i] for g in range(1, 7)]
        except (KeyError, IndexError):
            continue
        if not inwin(v[0]): continue
        ph["encode(1-2)"].append((v[1]-v[0])*1e6)
        ph["record/commit(2-3)"].append((v[2]-v[1])*1e6)
        ph["sync wait(3-4)"].append((v[3]-v[2])*1e6)
        ph["elapsed(4-5)"].append((v[4]-v[3])*1e6)
        ph["env(5-6)"].append((v[5]-v[4])*1e6)
        ph["total(1-6)"].append((v[5]-v[0])*1e6)
print("host phases per step (us):")
for k in ph: summ(k, ph[k])
# loop gap 6 -> next 1
gap = []
for (buf, t) in keys:
    if t == 0: continue
    a6 = mk.get((buf, t-1, 6), []); b1 = mk.get((buf, t, 1), [])
    for x, y in zip(a6, b1):
        if inwin(x): gap.append((y-x)*1e6)
summ("loop gap(6-next 1)", gap)
# GPU busy: union of all CB intervals within the window
iv = sorted((c["g0"], c["g1"]) for c in commits if c["g0"] > 0 and T0 <= c["g0"] <= T1)
busy = 0; cur0 = cur1 = None
for x0, x1 in iv:
    if cur1 is None or x0 > cur1:
        if cur1 is not None: busy += cur1 - cur0
        cur0, cur1 = x0, x1
    else: cur1 = max(cur1, x1)
if cur1 is not None: busy += cur1 - cur0
print(f"GPU busy (union of CB spans): {busy/len(sel)*1e3:.2f} ms/epoch of {(T1-T0)/len(sel)*1e3:.2f}")
riv = sorted((c["g0"], c["g1"]) for cs in rc.values() for c in cs if c["g0"] > 0)
rb = 0; cur0 = cur1 = None
for x0, x1 in riv:
    if cur1 is None or x0 > cur1:
        if cur1 is not None: rb += cur1 - cur0
        cur0, cur1 = x0, x1
    else: cur1 = max(cur1, x1)
if cur1 is not None: rb += cur1 - cur0
print(f"rollout CB union: {rb/len(sel)*1e3:.2f} ms/epoch")

import sys, numpy as np
import production.live_replay as lr
from production.arm_prior import _globals
rec = sys.argv[1]; simulate = "--sim" in sys.argv
log = []
orig = lr.postprocess_output
def wrapped(output, calibrator, lag, labels, *a, **k):
    m = calibrator.measured_frame(lag)
    res = orig(output, calibrator, lag, labels, *a, **k)
    log.append((None if m is None else np.array(m), res[2]))
    return res
lr.postprocess_output = wrapped
r = lr.replay_live(rec, simulate=simulate)
print(r.summary(), len(log))
L = r.labels; D = np.array([0, -1.0, 0])
def elev(v): return np.degrees(np.arcsin(np.clip(v[1] / np.linalg.norm(v), -1, 1)))
def ang(a, b): return np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
# which half of log corresponds to the main pass? compare final shank to measured
for name, sub in (("first", log[0::2]), ("second", log[1::2]), ("head", log[:len(r.times)])):
    err = []
    for i in range(0, len(r.times), 50):
        m = sub[i][0]
        if m is None: continue
        g = _globals(r.final_local[i].astype(np.float64))
        err.append(ang(g[4] @ D, m[L.index("ls")] @ D))
    print(name, np.round(np.median(err), 2))
X = np.array([1.0, 0, 0]); Y = np.array([0, 1.0, 0])
rows = []
for i, t in enumerate(r.times):
    m = log[2 * i][0]
    gr = _globals(r.raw_local[i].astype(np.float64)); gf = _globals(r.final_local[i].astype(np.float64))
    row = [t, elev(gr[1] @ D), ang(gr[1] @ D, gr[4] @ D), elev(gr[2] @ D), ang(gr[2] @ D, gr[5] @ D),
           elev(gf[1] @ D), elev(gf[4] @ D), ang(gf[1] @ D, gf[4] @ D), elev(gf[2] @ D), elev(gf[5] @ D), ang(gf[2] @ D, gf[5] @ D),
           elev(gf[0] @ Y), elev(gf[3] @ Y)]
    if m is not None:
        row += [elev(m[L.index("lw")] @ X), elev(m[L.index("rw")] @ -X)]
    else:
        row += [np.nan, np.nan]
    rows.append(row)
rows = np.array(rows)
np.save("analysis/scratch_birddog/rows2_%s.npy" % ("sim" if simulate else "live"), rows)
step = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 0.5
print("   t | RAW thL kneeL thR kneeR | FIN thL shL kneeL thR shR kneeR | pelY sp1Y | lwEl rwEl")
for s in np.arange(0, rows[-1, 0] + step, step):
    mk = (rows[:, 0] >= s) & (rows[:, 0] < s + step)
    if mk.any():
        v = np.nanmean(rows[mk], 0)
        print(f"{s:5.1f} | " + " ".join(f"{x:4.0f}" for x in v[1:5]) + " | " + " ".join(f"{x:4.0f}" for x in v[5:11]) + " | " + " ".join(f"{x:4.0f}" for x in v[11:13]) + " | " + " ".join(f"{x:4.0f}" for x in v[13:15]))
print("weights: t | wL wR")
W = np.array([[ (log[2*i][1].get("birddog") or {}).get("weights", {}).get(k, np.nan) for k in ("left","right")] for i in range(len(r.times))], float)
np.save("analysis/scratch_birddog/w_live.npy", W)
for s in np.arange(0, rows[-1, 0] + step, step):
    mk = (rows[:, 0] >= s) & (rows[:, 0] < s + step)
    if mk.any(): print(f"W {s:5.1f} | " + " ".join(f"{x:4.2f}" for x in np.nanmean(W[mk], 0)))

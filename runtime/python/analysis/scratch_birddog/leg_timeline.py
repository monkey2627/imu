"""Bird-dog leg timeline: live final output vs measured shank, per second."""
import sys, numpy as np
import production.live_replay as lr
from production.arm_prior import _globals

rec = sys.argv[1] if len(sys.argv) > 1 else "recordings/birddog_20261123_005257.jsonl"
simulate = "--sim" in sys.argv
measured_log = []
orig = lr.postprocess_output
def wrapped(output, calibrator, lag, labels, *a, **k):
    m = calibrator.measured_frame(lag)
    measured_log.append(None if m is None else np.array(m))
    return orig(output, calibrator, lag, labels, *a, **k)
lr.postprocess_output = wrapped
r = lr.replay_live(rec, simulate=simulate)
print(r.summary())
labels = r.labels
D = np.array([0, -1.0, 0])
def elev(v): return np.degrees(np.arcsin(np.clip(v[1] / np.linalg.norm(v), -1, 1)))
def ang(a, b): return np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
rows = []
# measured_log has one per postprocess call; replay may also call for rules check -> take last len(times)
ml = measured_log[-len(r.times):] if len(measured_log) >= len(r.times) else [None] * len(r.times)
for i, t in enumerate(r.times):
    out = []
    for loc in (r.raw_local[i], r.final_local[i]):
        g = _globals(loc.astype(np.float64))
        th_l, th_r = g[1] @ D, g[2] @ D
        sh_l, sh_r = g[4] @ D, g[5] @ D
        pel_down = g[0] @ D
        out.append((elev(th_l), elev(sh_l), ang(th_l, sh_l), elev(th_r), elev(sh_r), ang(th_r, sh_r), elev(g[0] @ np.array([0, 1.0, 0]))))
    m = ml[i]
    if m is not None:
        msl, msr = m[labels.index("ls")] @ D, m[labels.index("rs")] @ D
        meas = (elev(msl), elev(msr))
    else:
        meas = (np.nan, np.nan)
    rows.append((t, *out[0], *out[1], *meas))
rows = np.array(rows)
print(len(measured_log), len(r.times))
print(" t   | RAW thL shL kneeL  thR shR kneeR pelUp | FINAL thL shL kneeL thR shR kneeR pelUp | meas shL shR")
for s in np.arange(0, rows[-1, 0] + 1, 1.0):
    m = (rows[:, 0] >= s) & (rows[:, 0] < s + 1)
    if m.any():
        v = rows[m].mean(0)
        print(f"{s:4.0f} | " + " ".join(f"{x:5.0f}" for x in v[1:8]) + " | " + " ".join(f"{x:5.0f}" for x in v[8:15]) + " | " + " ".join(f"{x:5.0f}" for x in v[15:17]))
np.save("analysis/scratch_birddog/rows_%s.npy" % ("sim" if simulate else "live"), rows)

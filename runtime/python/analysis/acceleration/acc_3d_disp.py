"""3-axis displacement per movement (ZUPT both ends, edge-bias removed): if the length is right
(~0.5 m for a forearm raise/lower) but the direction is wrong, the axes are misassigned.
Also: how often the acceleration value actually changes (update rate) during movement.
python -m analysis.acceleration.acc_3d_disp <recording> <label> <a> <b> ...
"""
import json, sys
import numpy as np
from production.input_recording import resolve_recording_path

rows = [json.loads(l) for l in resolve_recording_path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if l.strip()]
ticks = [r for r in rows if r.get("type") == "tick"]
t = np.array([r["t"] for r in ticks]); t -= t[0]
args = sys.argv[2:]
for k in range(0, len(args), 3):
    lab, a, b = args[k], float(args[k + 1]), float(args[k + 2])
    acc = np.array([r["samples"][lab]["reference_adjusted_accel_xyz"] for r in ticks], float)
    pre = (t >= a - 0.3) & (t < a); post = (t > b) & (t <= b + 0.3); m = (t >= a) & (t <= b)
    tt = t[m]
    x = acc[m] - (acc[pre].mean(0) + np.outer((tt - a) / (b - a), acc[post].mean(0) - acc[pre].mean(0)))
    v = np.vstack([np.zeros(3), np.cumsum((x[1:] + x[:-1]) / 2 * np.diff(tt)[:, None], 0)])
    v -= np.outer((tt - tt[0]) / (tt[-1] - tt[0]), v[-1])
    d = np.sum((v[1:] + v[:-1]) / 2 * np.diff(tt)[:, None], 0)
    changes = np.mean(np.any(np.diff(acc[m], axis=0) != 0, axis=1))
    print(f"{lab} {a:5.1f}-{b:5.1f}s  disp {np.round(d, 2)}  |d| {np.linalg.norm(d):.2f} m   peak|a| {np.linalg.norm(acc[m], axis=1).max():.1f}"
          f"  value changes in {changes*100:.0f}% of rows ({len(tt)/(b-a):.0f} rows/s)")

"""Orientation-dependent leakage: a body-frame accelerometer bias b shows up in the world vertical as
(Q(t) b)_y.  Fit b (and a constant c) from static windows at different orientations, then integrate the
corrected vertical acceleration over each movement (ZUPT at both ends).
python -m analysis.acceleration.acc_bias_fit
"""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from production.input_recording import resolve_recording_path

REC = "unity_20261103_002526"
STATIC = [(0.0, 0.5), (1.5, 2.5), (4.25, 5.0), (7.5, 8.5), (10.0, 10.75), (12.5, 13.5), (15.0, 15.75),
          (17.25, 17.75), (19.25, 20.0), (23.25, 24.0)]
MOVES = [(2.6, 4.1, +1), (5.2, 7.1, -1), (8.6, 10.0, +1), (11.1, 12.3, -1), (13.6, 14.8, +1), (15.9, 17.2, -1),
         (17.9, 19.1, +1), (20.3, 22.0, -1), (22.1, 23.1, +1), (24.3, 25.4, -1)]

rows = [json.loads(l) for l in resolve_recording_path(REC).read_text(encoding="utf-8").splitlines() if l.strip()]
ticks = [r for r in rows if r.get("type") == "tick"]
t = np.array([r["t"] for r in ticks]); t -= t[0]


def integrate(ay, a, b):
    pre = (t >= a - 0.3) & (t < a); post = (t > b) & (t <= b + 0.3); m = (t >= a) & (t <= b)
    tt = t[m]
    acc = ay[m] - np.interp(tt, [a, b], [ay[pre].mean(), ay[post].mean()])
    v = np.r_[0, np.cumsum((acc[1:] + acc[:-1]) / 2 * np.diff(tt))]
    v -= np.interp(tt, [tt[0], tt[-1]], [0, v[-1]])
    return np.sum((v[1:] + v[:-1]) / 2 * np.diff(tt))


for lab in ("lw", "rw"):
    s = [r["samples"][lab] for r in ticks]
    ay = np.array([x["reference_adjusted_accel_xyz"] for x in s], float)[:, 1]
    for qname in ("raw_quat_xyzw", "quat_xyzw"):
        Q = Rotation.from_quat([x[qname] for x in s]).as_matrix()
        X, y = [], []
        for a, b in STATIC:
            m = (t >= a) & (t < b)
            X.append(np.r_[Q[m, 1, :].mean(0), 1.0]); y.append(ay[m].mean())
        X, y = np.array(X), np.array(y)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        corr = ay - (Q[:, 1, :] @ coef[:3] + coef[3])
        before = [integrate(ay, a, b) for a, b, _ in MOVES]
        after = [integrate(corr, a, b) for a, b, _ in MOVES]
        print(f"{lab} [{qname}] static vertical acc {np.round(y, 2)} -> fit residual rms {np.sqrt((resid**2).mean()):.2f}, "
              f"|b| {np.linalg.norm(coef[:3]):.2f}")
        print("   expected  " + " ".join(f"{'+' if s_ > 0 else '-'}0.4" for *_, s_ in MOVES))
        print("   before    " + " ".join(f"{v:+.2f}" for v in before))
        print("   corrected " + " ".join(f"{v:+.2f}" for v in after))

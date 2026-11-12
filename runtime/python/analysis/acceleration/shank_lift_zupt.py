"""Shank height change per movement from the gravity-free acceleration, done carefully:
real row timestamps, still detection, per-movement bias from the preceding still window,
zero-velocity at both ends (linear velocity drift removed), vertical displacement per movement.
python -m analysis.acceleration.shank_lift_zupt <recording> [labels]
"""
import json, sys
import numpy as np
from scipy.spatial.transform import Rotation

RATE_STILL = 20.0     # deg/s
ACC_STILL = 0.6       # m/s^2
MIN_STILL = 0.25      # s
MIN_MOVE = 0.3        # s


def analyse(rows, label):
    ticks = [r for r in rows if r.get("type") == "tick"]
    t = np.array([r["t"] for r in ticks])
    acc = np.array([r["samples"][label]["reference_adjusted_accel_xyz"] for r in ticks], float)
    rot = Rotation.from_quat([r["samples"][label]["quat_xyzw"] for r in ticks])
    rate = np.r_[0, np.degrees((rot[:-1].inv() * rot[1:]).magnitude()) / np.maximum(np.diff(t), 1e-3)]
    raw_still = (rate < RATE_STILL) & (np.linalg.norm(acc, axis=1) < ACC_STILL)
    # still = raw_still sustained for MIN_STILL
    still = np.zeros(len(t), bool)
    start = None
    for i, s in enumerate(np.r_[raw_still, False]):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if t[i - 1] - t[start] >= MIN_STILL:
                still[start:i] = True
            start = None
    segs, i = [], 0
    while i < len(t):
        if still[i]:
            i += 1; continue
        j = i
        while j < len(t) and not still[j]:
            j += 1
        if i > 0 and j < len(t) and t[j - 1] - t[i] >= MIN_MOVE:
            pre = slice(max(0, i - 15), i); post = slice(j, min(len(t), j + 15))
            b0, b1 = acc[pre, 1].mean(), acc[post, 1].mean()
            seg_t = t[i - 1:j + 1]
            a = acc[i - 1:j + 1, 1] - np.interp(seg_t, [seg_t[0], seg_t[-1]], [b0, b1])
            v = np.r_[0, np.cumsum((a[1:] + a[:-1]) / 2 * np.diff(seg_t))]
            v -= np.interp(seg_t, [seg_t[0], seg_t[-1]], [0, v[-1]])          # zero velocity at both ends
            h = np.r_[0, np.cumsum((v[1:] + v[:-1]) / 2 * np.diff(seg_t))]
            tilt = Rotation.from_quat([ticks[k]["samples"][label]["quat_xyzw"] for k in (i - 1, j)])
            segs.append((t[i] - t[0], t[j - 1] - t[i], h[-1], np.abs(v).max(),
                         np.degrees((tilt[0].inv() * tilt[1]).magnitude())))
        i = j
    return t[0], segs


def main():
    path = sys.argv[1]
    labels = sys.argv[2].split(",") if len(sys.argv) > 2 else ["rs", "ls"]
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    t0 = next(r["t"] for r in rows if r.get("type") == "tick")
    print("events:", [(round(r["t"] - t0, 1), r["name"]) for r in rows if r.get("type") == "event"
                      and "Calibrate" in r["name"]])
    for label in labels:
        _, segs = analyse(rows, label)
        print(f"{label}: movements (start s, duration s, height change m, peak |v| m/s, rotation deg)")
        for s in segs:
            flag = "  <==" if abs(s[2]) > 0.15 else ""
            print(f"   {s[0]:6.1f} {s[1]:5.2f}  dh {s[2]:+.2f}  v {s[3]:.2f}  rot {s[4]:4.0f}{flag}")


if __name__ == "__main__":
    main()

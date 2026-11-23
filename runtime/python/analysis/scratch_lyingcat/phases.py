"""Cat/cow phase windows from the waist's VERTICAL TRANSLATION (accel ZUPT), independent of its pitch.

Motion segments = windows where the smoothed |linear accel| or the waist angular speed is above
the recording's still level; each segment's ZUPT vertical displacement says up (cat) or down (cow).
The still gap after a segment is labelled with that direction.  Diagnostic only (not production)."""
import json, sys, numpy as np
from scipy.spatial.transform import Rotation as R
from production.accel_frame import AccelerationFrameFix


def load(stem):
    fix = AccelerationFrameFix(); T, A, Q = [], [], []
    for line in open(f"recordings/{stem}.jsonl", encoding="utf-8"):
        d = json.loads(line)
        if d.get("type") != "tick": continue
        s = d["samples"]["waist"]; T.append(d["t"]); v = fix.correct_sample(s)
        A.append(v if v is not None else [np.nan] * 3); Q.append(s["quat_xyzw"])
    return np.array(T), np.nan_to_num(np.array(A, float)), np.array(Q)


def phases(stem, t0=9.0, t1=None, min_dy=0.04):
    T, a, Q = load(stem)
    if t1 is None:      # stop before getting up: waist pitch > 40 deg
        y = np.degrees(np.arcsin(np.clip(R.from_quat(Q).as_matrix()[:, 1, 1], -1, 1)))
        t1 = T[(T > t0) & (np.abs(R.from_quat(Q).as_matrix()[:, 1, :]).max(1) > 2)].min() if False else None
    dt = np.median(np.diff(T))
    rot = R.from_quat(Q)
    w = np.r_[0, np.degrees((rot[1:] * rot[:-1].inv()).magnitude()) / dt]
    k = 20
    a_s = np.convolve(np.linalg.norm(a, axis=1), np.ones(k) / k, "same")
    w_s = np.convolve(w, np.ones(k) / k, "same")
    base_a = np.percentile(a_s[T > t0], 20); base_w = np.percentile(w_s[T > t0], 20)
    moving = (a_s > 1.8 * base_a) | (w_s > max(3.0, 2.0 * base_w))
    segs, i, n = [], 0, len(T)
    while i < n:
        if not moving[i] or T[i] < t0: i += 1; continue
        j = i
        while j < n and moving[j]: j += 1
        if j - i >= 15:
            seg = a[i:j]; tt = np.arange(j - i) * dt
            v = np.cumsum(seg, 0) * dt; v -= np.outer(tt / tt[-1], v[-1]); p = np.cumsum(v, 0) * dt
            segs.append((T[i], T[j - 1], p[-1, 1]))
        i = j
    segs = [s for s in segs if abs(s[2]) >= min_dy]
    out = []
    for (s0, e0, dy), nxt in zip(segs, segs[1:] + [None]):
        end = nxt[0] if nxt else e0 + 3.0
        if end - e0 > 0.6:
            out.append(("cat" if dy > 0 else "cow", e0 + 0.2, end - 0.2, dy))
    return segs, out


if __name__ == "__main__":
    for stem in sys.argv[1:]:
        segs, ph = phases(stem)
        print(stem)
        for s in segs: print(f"   seg {s[0]:6.2f}-{s[1]:6.2f} dy {100*s[2]:+6.1f} cm")
        for p in ph: print(f"   {p[0]} {p[1]:6.2f}-{p[2]:6.2f}")

"""Shank vertical displacement per motion segment (ZUPT between still periods)."""
import sys, json, numpy as np
from production.accel_frame import AccelerationFrameFix
rec = sys.argv[1]
fix = AccelerationFrameFix()
T, A = [], {"ls": [], "rs": []}
for line in open(rec, encoding="utf-8"):
    d = json.loads(line)
    if d.get("type") != "tick": continue
    T.append(d["t"])
    for lab in A:
        s = d["samples"][lab]
        v = fix.correct_sample(s)
        A[lab].append(v if v is not None else [np.nan] * 3)
T = np.array(T)
dt = np.median(np.diff(T))
print("dt", dt, "n", len(T))
for lab in A:
    a = np.nan_to_num(np.array(A[lab], float))
    mag = np.linalg.norm(a, axis=1)
    k = 12
    sm = np.convolve(mag, np.ones(k) / k, "same")
    still = sm < 0.35
    # segments of motion between still periods
    i = 0; n = len(T); out = []
    while i < n:
        if still[i]: i += 1; continue
        j = i
        while j < n and not still[j]: j += 1
        if j - i >= 15:
            seg = a[i:j]; tt = np.arange(j - i) * dt
            v = np.cumsum(seg, 0) * dt
            v -= np.outer(tt / tt[-1], v[-1])          # ZUPT: velocity zero at both ends
            p = np.cumsum(v, 0) * dt
            out.append((T[i], T[j - 1], p[-1, 1], p[:, 1].max(), p[:, 1].min()))
        i = j
    print(f"--- {lab}")
    for s, e, dy, mx, mn in out:
        if abs(dy) > 0.03 or mx > 0.05:
            print(f"  {s:6.2f}-{e:6.2f}  dy {dy*100:6.1f} cm  (max {mx*100:5.1f}, min {mn*100:6.1f})")

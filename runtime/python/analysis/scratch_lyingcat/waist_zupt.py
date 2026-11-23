"""Diagnostic: waist vertical displacement per motion segment (ZUPT) vs waist pitch change."""
import sys, json, numpy as np
from production.accel_frame import AccelerationFrameFix
from analysis.scratch_lyingcat.signal import elev
for stem in sys.argv[1:]:
    fix = AccelerationFrameFix(); T, A = [], []
    for line in open(f"recordings/{stem}.jsonl", encoding="utf-8"):
        d = json.loads(line)
        if d.get("type") != "tick": continue
        T.append(d["t"]); v = fix.correct_sample(d["samples"]["waist"]); A.append(v if v is not None else [np.nan]*3)
    T = np.array(T); a = np.nan_to_num(np.array(A, float)); dt = np.median(np.diff(T))
    cap = np.load(f"analysis/scratch_lyingcat/cap_{stem}_prod.npz"); pitch = elev(cap["uncal"][:, 4, :, 1])
    mag = np.linalg.norm(a, axis=1); sm = np.convolve(mag, np.ones(15)/15, "same")
    still = sm < float(sys.argv[0:1] and 0.08)
    print(f"== {stem}  |a| median {np.median(mag):.3f} p90 {np.percentile(mag,90):.3f}")
    i, n = 0, len(T)
    while i < n:
        if still[i] or T[i] < 9 or T[i] > 52: i += 1; continue
        j = i
        while j < n and not still[j]: j += 1
        if j - i >= 20:
            seg = a[i:j]; tt = np.arange(j - i) * dt
            v = np.cumsum(seg, 0) * dt; v -= np.outer(tt / tt[-1], v[-1]); p = np.cumsum(v, 0) * dt
            print(f"  {T[i]:6.2f}-{T[j-1]:6.2f}  dy {p[-1,1]*100:6.1f} cm  pitch {pitch[i]:6.1f} -> {pitch[j-1]:6.1f} (d {pitch[j-1]-pitch[i]:+6.1f})")
        i = j

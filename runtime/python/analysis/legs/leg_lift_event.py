"""Does the user's shank show a lift movement when the coach raises that leg?
Coach phase from the recorded coach pose (thigh above -45 deg elevation = leg raised).
For every coach raise: the user's shank angular path (deg) and peak rate in [start-1.5 s, start+3 s],
for the leg the coach raises vs. the other leg.
python -m analysis.legs.leg_lift_event <recording> ...
"""
import json, sys
import numpy as np
from scipy.spatial.transform import Rotation
from production.input_recording import resolve_recording_path
from production.pose_calibration import unity_quat_to_smpl

for name in sys.argv[1:]:
    rows = [json.loads(l) for l in resolve_recording_path(name).read_text(encoding="utf-8").splitlines() if l.strip()]
    ticks = [r for r in rows if r.get("type") == "tick"]
    coach = [r for r in rows if r.get("type") == "coach"]
    if len(coach) < 100:
        continue
    t = np.array([r["t"] for r in ticks])
    path = {}
    for lab in ("ls", "rs"):
        rot = Rotation.from_quat([r["samples"][lab]["quat_xyzw"] for r in ticks])
        step = np.r_[0, np.degrees((rot[:-1].inv() * rot[1:]).magnitude())]
        path[lab] = np.cumsum(step)
    ct = np.array([r["t"] for r in coach])
    el = {}
    for lab, joint in (("ls", 1), ("rs", 2)):           # coach thigh (SMPL hip joint global), bone axis -y
        d = np.array([unity_quat_to_smpl(r["q"][joint]) @ np.array([0.0, -1.0, 0.0]) for r in coach])
        el[lab] = np.degrees(np.arcsin(np.clip(d[:, 1], -1, 1)))
    print(f"== {name}  ({(t[-1]-t[0]):.0f}s)  coach thigh elevation range L {el['ls'].min():.0f}..{el['ls'].max():.0f}"
          f" R {el['rs'].min():.0f}..{el['rs'].max():.0f}")
    seg = lambda lab, a, b: path[lab][np.searchsorted(t, b) - 1] - path[lab][np.searchsorted(t, a)]
    for lab in ("ls", "rs"):
        other = "rs" if lab == "ls" else "ls"
        up = el[lab] > -45
        starts = ct[1:][up[1:] & ~up[:-1]]
        for s in starts:
            raised, still = seg(lab, s - 1.5, s + 3.0), seg(other, s - 1.5, s + 3.0)
            print(f"   coach raises {lab} at {s - t[0]:6.1f}s: user {lab} path {raised:5.0f} deg | other leg {other} {still:5.0f} deg")
    # quiet reference: windows where the coach keeps both thighs down for >= 4.5 s
    down = (el["ls"] < -60) & (el["rs"] < -60)
    quiet = []
    i = 0
    while i < len(ct):
        if down[i]:
            j = i
            while j < len(ct) and down[j]:
                j += 1
            if ct[j - 1] - ct[i] >= 4.5:
                quiet += [seg(l, ct[i], ct[i] + 4.5) for l in ("ls", "rs")]
            i = j
        else:
            i += 1
    if quiet:
        print(f"   coach both legs down (4.5 s windows): user shank path median {np.median(quiet):.0f}, max {np.max(quiet):.0f} deg (n={len(quiet)})")

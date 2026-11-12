"""Margin check on the coach captures: between 'knee extended' and 'knee bent' frames of an action,
how far apart are the thigh and shank pointing directions (pelvis-yaw frame)?
python -m analysis.legs.leg_state_margin <humanpose_ours dir>
"""
import sys
from pathlib import Path
import numpy as np
from analysis.legs.thigh_vs_shank_knn import load, knee_angle, I

for path in sorted(Path(sys.argv[1]).glob("*.humanpose_joints.json"), key=lambda p: int(p.name.split(".")[0])):
    name, rot, pos = load(path)
    out = []
    for side in "lr":
        k = knee_angle(pos, side)
        ext, bent = k < 30, k > 70
        if ext.sum() < 30 or bent.sum() < 30:
            continue
        def dirs(a, b):
            d = pos[:, I[b + "_" + side]] - pos[:, I[a + "_" + side]]
            return d / np.linalg.norm(d, axis=-1, keepdims=True)
        res = []
        for a, b in (("femur", "tibia"), ("tibia", "talus")):
            d = dirs(a, b)
            me, mb = d[ext].mean(0), d[bent].mean(0)
            gap = np.degrees(np.arccos(np.clip(me @ mb / np.linalg.norm(me) / np.linalg.norm(mb), -1, 1)))
            spread = np.degrees(np.arccos(np.clip(d[ext] @ (me / np.linalg.norm(me)), -1, 1))).mean()
            el = lambda v: np.degrees(np.arcsin(v[1] / np.linalg.norm(v)))
            res.append(f"{'thigh' if a == 'femur' else 'shank'} gap {gap:4.0f}° (el ext {el(me):+4.0f}/bent {el(mb):+4.0f})")
        out.append(f"{side}: n_ext {ext.sum():4d} n_bent {bent.sum():4d} | " + " | ".join(res))
    if out:
        print(name); [print("   " + o) for o in out]

"""Leg-lift amplitude and timing: predicted thigh vs. measured shank (live recording, read-only).

For each leg-lift episode (shank sensor pitch rises > 12 deg above its kneeling baseline):
peak shank pitch (sensor), peak predicted thigh pitch, implied knee flexion at the peak
(angle between predicted thigh axis and measured shank axis; ~0 = straight leg), and the
delay between the shank and the predicted thigh reaching half of their own rise.
Pitch = angle from straight down (0 vertical, 90 horizontal).

    python -m analysis.leg_lift_response BirdDog_RingSkinC_Recorded_20261112_212104
"""
import sys

import numpy as np

from analysis.imudino_jitter_stages import LABELS, fk, load, quat_xyzw_to_mat, resolve_recording_path, unity_to_smpl
from production.calibration import bone_orientation

DOWN = np.array([0, -1.0, 0])


def pitch_of(rot, axis=(0, -1, 0)):
    v = rot @ np.asarray(axis, float)
    return np.degrees(np.arccos(np.clip(v @ DOWN, -1, 1))), v


for name in sys.argv[1:]:
    header, _, preds, quats, acc, pred_tick, _, raw_q, fin_q = load(resolve_recording_path(name))
    g = fk(unity_to_smpl(quat_xyzw_to_mat(raw_q)))
    ori = bone_orientation(quats, LABELS).numpy().astype(float)
    t = pred_tick - 1                       # frame each prediction represents (lag 1)
    print(f"== {name[-15:]}")
    for side, thigh_j, s in (("L", 1, 2), ("R", 2, 3)):
        sh_p, sh_v = pitch_of(ori[t, s])
        th_p, th_v = pitch_of(g[:, thigh_j])
        knee = np.degrees(np.arccos(np.clip((sh_v * th_v).sum(-1), -1, 1)))
        base = np.percentile(sh_p, 20)
        up = sh_p > base + 12
        idx = np.flatnonzero(up)
        if not len(idx):
            print(f"  {side}: no lift"); continue
        for e in np.split(idx, np.flatnonzero(np.diff(idx) > 15) + 1):
            if len(e) < 15:
                continue
            lo, hi = max(e[0] - 30, 0), min(e[-1] + 30, len(t) - 1)
            k = e[np.argmax(sh_p[e])]
            seg = slice(lo, hi + 1)
            def half_time(x):
                b, top = np.median(x[lo:e[0]]) if e[0] > lo else x[lo], x[seg].max()
                above = np.flatnonzero(x[seg] >= b + 0.5 * (top - b))
                return (t[lo + above[0]]) if len(above) else np.nan
            d = (half_time(th_p) - half_time(sh_p)) * 1000 / 60
            print(f"  {side}-lift t={preds[e[0]]['t']:.1f}-{preds[e[-1]]['t']:.1f}s  shank {base:.0f}->{sh_p[e].max():.0f}  "
                  f"thigh {np.median(th_p[lo:e[0]]) if e[0] > lo else th_p[lo]:.0f}->{th_p[e].max():.0f}  "
                  f"knee flex at shank peak {knee[k]:.0f}  thigh half-rise delay {d:+.0f} ms")

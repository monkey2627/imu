"""Fit one fixed sensor-side rotation C for the acceleration (in the tracker's RAW frame, raw_quat_xyzw),
on half the arm movements; test on the other half: does integrating the corrected acceleration reproduce
the kinematic displacement (shoulder-pivot model)?  Also compares C across trackers.
python -m analysis.acceleration.acc_frame_fix
"""
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot
from production.calibration import bone_orientation
from production.input_recording import resolve_recording_path
from production.pose_calibration import BONE_AXES

REC, L = "unity_20261103_002526", 0.40
MOVES = [(2.6, 4.1), (5.2, 7.1), (8.6, 10.0), (11.1, 12.3), (13.6, 14.8), (15.9, 17.2), (22.1, 23.1), (24.3, 25.4)]
S2SMPL = np.diag([-1.0, 1.0, -1.0])
labels = ["lw", "rw", "ls", "rs", "waist"]
rows = [json.loads(l) for l in resolve_recording_path(REC).read_text(encoding="utf-8").splitlines() if l.strip()]
ticks = [r for r in rows if r.get("type") == "tick"]
t = np.array([r["t"] for r in ticks]); t -= t[0]
g = np.arange(0, t[-1], 0.01)
interp = lambda x: np.stack([np.interp(g, t, x[:, k]) for k in range(x.shape[1])], 1)


def integrate(acc, a, b):
    """ZUPT double integration over [a, b] (edge bias removed): 3-D displacement."""
    pre = (g >= a - 0.3) & (g < a); post = (g > b) & (g <= b + 0.3); m = (g >= a) & (g <= b)
    tt = g[m]
    x = acc[m] - (acc[pre].mean(0) + np.outer((tt - a) / (b - a), acc[post].mean(0) - acc[pre].mean(0)))
    v = np.vstack([np.zeros(3), np.cumsum((x[1:] + x[:-1]) / 2 * 0.01, 0)])
    v -= np.outer((tt - tt[0]) / (tt[-1] - tt[0]), v[-1])
    return np.sum((v[1:] + v[:-1]) / 2 * 0.01, 0)


for lab in ("lw", "rw"):
    i = labels.index(lab)
    ori = np.stack([bone_orientation(np.array([r["samples"][l]["quat_xyzw"] for l in labels]), labels).numpy()[i]
                    for r in ticks]).astype(float)
    pos = gaussian_filter1d(interp(ori @ (L * BONE_AXES[lab])), 4, axis=0)            # SMPL world
    exp = np.gradient(np.gradient(pos, 0.01, axis=0), 0.01, axis=0) @ S2SMPL.T        # -> server world
    q = np.array([r["samples"][lab]["raw_quat_xyzw"] for r in ticks], float)
    q[np.sum(q * np.roll(q, 1, 0), 1) < 0] *= -1
    Rraw = Rot.from_quat(interp(q) / np.linalg.norm(interp(q), axis=1, keepdims=True)).as_matrix()
    for field in ("accel_xyz", "reference_adjusted_accel_xyz"):
        mea = gaussian_filter1d(interp(np.array([r["samples"][lab][field] for r in ticks], float)), 4, axis=0)
        loc_m = np.einsum("tji,tj->ti", Rraw, mea)                                      # into the raw sensor frame
        loc_e = np.einsum("tji,tj->ti", Rraw, exp)
        fit_idx, test_idx = MOVES[0::2], MOVES[1::2]
        sel = lambda moves: np.any([(g >= a) & (g <= b) for a, b in moves], axis=0)
        C, _ = Rot.align_vectors(loc_e[sel(fit_idx)], loc_m[sel(fit_idx)])
        corr = lambda X, Y: np.sum(X * Y) / np.sqrt(np.sum(X * X) * np.sum(Y * Y))
        fixed = np.einsum("tij,tj->ti", Rraw, C.apply(loc_m))                           # corrected, server world
        print(f"{lab} [{field}] C = {np.round(C.as_rotvec() / np.linalg.norm(C.as_rotvec()), 2)} x "
              f"{np.degrees(C.magnitude()):.0f} deg;  held-out corr before {corr(loc_m[sel(test_idx)], loc_e[sel(test_idx)]):+.2f}"
              f" after {corr(C.apply(loc_m[sel(test_idx)]), loc_e[sel(test_idx)]):+.2f}")
        if field == "accel_xyz":
            for a, b in MOVES:
                d_true = (pos[np.searchsorted(g, b)] - pos[np.searchsorted(g, a)]) @ S2SMPL.T
                d_raw, d_fix = integrate(mea, a, b), integrate(fixed, a, b)
                tag = "fit " if (a, b) in fit_idx else "test"
                print(f"   {tag} {a:5.1f}-{b:5.1f}s  true {np.round(d_true, 2)} |{np.linalg.norm(d_true):.2f}|"
                      f"   raw {np.round(d_raw, 2)}   fixed {np.round(d_fix, 2)} |{np.linalg.norm(d_fix):.2f}|"
                      f"  err {np.linalg.norm(d_fix - d_true):.2f} m")

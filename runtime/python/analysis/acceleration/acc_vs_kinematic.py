"""Measured SlimeVR linear acceleration vs. the acceleration implied by the orientation stream.
A straight arm swinging about a fixed shoulder: sensor position p(t) = R_forearm(t) @ (L * bone_axis);
expected world acceleration = d2p/dt2.  Compared per movement: gain, correlation, best time shift,
and the residual (what the integration sees as error).
python -m analysis.acceleration.acc_vs_kinematic
"""
import json
import numpy as np
from production.calibration import bone_orientation
from production.input_recording import resolve_recording_path
from production.pose_calibration import BONE_AXES

REC, L = "unity_20261103_002526", 0.40
MOVES = [(2.6, 4.1), (5.2, 7.1), (8.6, 10.0), (11.1, 12.3), (13.6, 14.8), (15.9, 17.2), (22.1, 23.1), (24.3, 25.4)]
S2SMPL = np.diag([-1.0, 1.0, -1.0])
rows = [json.loads(l) for l in resolve_recording_path(REC).read_text(encoding="utf-8").splitlines() if l.strip()]
ticks = [r for r in rows if r.get("type") == "tick"]
t = np.array([r["t"] for r in ticks]); t -= t[0]
labels = ["lw", "rw", "ls", "rs", "waist"]
# resample everything on a uniform 100 Hz grid
g = np.arange(0, t[-1], 0.01)
for lab in ("lw", "rw"):
    i = labels.index(lab)
    ori = np.stack([bone_orientation(np.array([r["samples"][l]["quat_xyzw"] for l in labels]), labels).numpy()[i]
                    for r in ticks]).astype(float)
    acc = np.array([r["samples"][lab]["reference_adjusted_accel_xyz"] for r in ticks], float) @ S2SMPL.T
    pos = ori @ (L * BONE_AXES[lab])
    pos_g = np.stack([np.interp(g, t, pos[:, k]) for k in range(3)], 1)
    from scipy.ndimage import gaussian_filter1d
    pos_g = gaussian_filter1d(pos_g, 4, axis=0)                      # ~40 ms smoothing before differentiating
    exp = np.gradient(np.gradient(pos_g, 0.01, axis=0), 0.01, axis=0)
    mea = np.stack([np.interp(g, t, acc[:, k]) for k in range(3)], 1)
    mea = gaussian_filter1d(mea, 4, axis=0)
    print(f"== {lab}   per movement: |expected| rms, |measured| rms, best gain, corr at best shift, shift (ms; + = measured lags), residual rms")
    for a, b in MOVES:
        m = (g >= a) & (g <= b)
        best = None
        for sh in range(-30, 31):                                    # -300..+300 ms
            e = exp[np.roll(m, sh)] if True else None
            x = mea[m]; y = exp[np.roll(m, -sh)]
            c = np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y))
            if best is None or c > best[0]:
                best = (c, sh)
        c, sh = best
        y = exp[np.roll(m, -sh)]; x = mea[m]
        gain = np.sum(x * y) / np.sum(y * y)
        e0 = exp[m]
        print(f"   {a:5.1f}-{b:5.1f}s  exp {np.sqrt((e0**2).sum(1).mean()):.2f}  meas {np.sqrt((x**2).sum(1).mean()):.2f}"
              f"  gain {gain:+.2f}  corr {c:+.2f} @ {sh*10:+4d} ms   corr@0 {np.sum(x*e0)/np.sqrt(np.sum(x*x)*np.sum(e0*e0)):+.2f}"
              f"  resid {np.sqrt(((x - e0)**2).sum(1).mean()):.2f}")


def kabsch_check():
    """Is there ONE fixed rotation (world-side or sensor-side) that aligns measured with expected?"""
    from scipy.spatial.transform import Rotation as Rot
    from scipy.ndimage import gaussian_filter1d
    for lab in ("lw", "rw"):
        i = labels.index(lab)
        ori = np.stack([bone_orientation(np.array([r["samples"][l]["quat_xyzw"] for l in labels]), labels).numpy()[i]
                        for r in ticks]).astype(float)
        acc = np.array([r["samples"][lab]["reference_adjusted_accel_xyz"] for r in ticks], float) @ S2SMPL.T
        pos = ori @ (L * BONE_AXES[lab])
        pos_g = gaussian_filter1d(np.stack([np.interp(g, t, pos[:, k]) for k in range(3)], 1), 4, axis=0)
        exp = np.gradient(np.gradient(pos_g, 0.01, axis=0), 0.01, axis=0)
        mea = gaussian_filter1d(np.stack([np.interp(g, t, acc[:, k]) for k in range(3)], 1), 4, axis=0)
        R_g = Rot.from_matrix(np.stack([Rot.from_matrix(ori).as_quat()[:, k] for k in range(4)], 1) * 0 + 0) if False else None
        quat = Rot.from_matrix(ori).as_quat()
        quat_g = np.stack([np.interp(g, t, quat[:, k]) for k in range(4)], 1)
        Rg = Rot.from_quat(quat_g / np.linalg.norm(quat_g, axis=1, keepdims=True)).as_matrix()
        m = np.zeros(len(g), bool)
        for a, b in MOVES:
            m |= (g >= a) & (g <= b)
        for name, X, Y in (("world-side", mea[m], exp[m]),
                           ("sensor-side", np.einsum("tji,tj->ti", Rg[m], mea[m]), np.einsum("tji,tj->ti", Rg[m], exp[m]))):
            r, rssd = Rot.align_vectors(Y, X)
            Xa = r.apply(X)
            c = np.sum(Xa * Y) / np.sqrt(np.sum(Xa * Xa) * np.sum(Y * Y))
            print(f"{lab} best fixed {name} rotation: {np.degrees(r.magnitude()):5.1f} deg -> corr {c:+.2f}  "
                  f"resid rms {np.sqrt(((Xa - Y)**2).sum(1).mean()):.2f}  (expected rms {np.sqrt((Y**2).sum(1).mean()):.2f})")


kabsch_check()

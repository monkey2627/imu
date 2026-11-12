"""Motion features of a shank over given windows: rotation path (deg), raw linear-acceleration RMS and
peak (m/s^2, all axes and vertical), and the ratio acc-RMS / rotation-rate.  A real lift moves the
shank ~0.4 m with little rotation; wiggles/kicks rotate it without carrying it.
python -m analysis.legs.leg_motion_features <recording> <label> <t0> <t1> [<label> <t0> <t1> ...]
"""
import json, sys
import numpy as np
from scipy.spatial.transform import Rotation
from production.input_recording import resolve_recording_path

rows = [json.loads(l) for l in resolve_recording_path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if l.strip()]
ticks = [r for r in rows if r.get("type") == "tick"]
t = np.array([r["t"] for r in ticks]); t -= t[0]
args = sys.argv[2:]
for k in range(0, len(args), 3):
    lab, a, b = args[k], float(args[k + 1]), float(args[k + 2])
    m = (t >= a) & (t < b)
    acc = np.array([r["samples"][lab]["reference_adjusted_accel_xyz"] for r in ticks], float)[m]
    rot = Rotation.from_quat([r["samples"][lab]["quat_xyzw"] for r in ticks])
    step = np.r_[0, np.degrees((rot[:-1].inv() * rot[1:]).magnitude())][m]
    dur = b - a
    rms = np.sqrt((acc ** 2).sum(1).mean()); vrms = np.sqrt((acc[:, 1] ** 2).mean())
    print(f"{lab} {a:5.1f}-{b:5.1f}s: rot path {step.sum():5.0f} deg ({step.sum()/dur:4.0f} deg/s)  acc rms {rms:4.2f} "
          f"(vert {vrms:4.2f}) peak {np.linalg.norm(acc, axis=1).max():4.2f}  acc/rot {rms / max(step.sum()/dur, 1) * 100:5.2f}")

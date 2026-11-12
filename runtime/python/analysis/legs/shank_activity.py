"""Per-0.5 s activity of both shanks (no coach needed): angular path in the window, and the
pointing (azimuth vs. waist forward, elevation) plus angle to the first-seconds kneeling pose.
python -m analysis.legs.shank_activity <recording> [base_from_s base_to_s]
"""
import json, sys
import numpy as np
from scipy.spatial.transform import Rotation
from production.calibration import bone_orientation
from production.input_recording import load_input_recording, resolve_recording_path
from production.offline_compare import _expand
from production.pose_calibration import BONE_AXES, _best_yaw

path = resolve_recording_path(sys.argv[1])
rec = load_input_recording(path)
rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
quats, _, _ = _expand(rec, rows)
ori = np.stack([bone_orientation(q, rec.labels).numpy() for q in quats]).astype(np.float64)
wi = rec.labels.index("waist")
b0, b1 = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (None, None)
print(f"{path.name}: {len(ori)/60:.1f}s")
step = {}
base = {}
for lab in ("ls", "rs"):
    r = Rotation.from_matrix(ori[:, rec.labels.index(lab)])
    step[lab] = np.r_[0, np.degrees((r[:-1].inv() * r[1:]).magnitude())]
    if b0 is not None:
        base[lab] = Rotation.from_matrix(ori[int(b0*60):int(b1*60), rec.labels.index(lab)]).mean()
print("t     | spine pitch | ls: path/0.5s az el toBase | rs: path/0.5s az el toBase")
for t in range(0, len(ori), 30):
    yaw = _best_yaw([ori[t, wi]])
    sp = ori[t, wi] @ np.array([0.0, 1.0, 0.0])
    line = f"{t/60:5.1f} | {np.degrees(np.arctan2(np.hypot(sp[0], sp[2]), sp[1])):4.0f}"
    for lab in ("ls", "rs"):
        d = ori[t, rec.labels.index(lab)] @ BONE_AXES[lab]
        az = np.degrees((np.arctan2(d[0], d[2]) - yaw + np.pi) % (2*np.pi) - np.pi)
        tb = np.degrees((base[lab].inv() * Rotation.from_matrix(ori[t, rec.labels.index(lab)])).magnitude()) if base else np.nan
        line += f" | {lab} {step[lab][t:t+30].sum():5.0f} {az:+5.0f} {np.degrees(np.arcsin(np.clip(d[1],-1,1))):+4.0f} {tb:4.0f}"
    print(line)

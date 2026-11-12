"""Per-wear templates: angle of each shank (and the waist) to its kneeling-base hold and to its
raised-leg hold over time.  Is the 'leg raised' state separable from orientation alone?
python -m analysis.legs.leg_template_signal <recording>
"""
import json, sys
import numpy as np
from scipy.spatial.transform import Rotation
from production.calibration import bone_orientation
from production.input_recording import load_input_recording, resolve_recording_path
from production.offline_compare import _expand
from production.pose_calibration import BONE_AXES

path = resolve_recording_path(sys.argv[1])
rec = load_input_recording(path)
rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
quats, _, events = _expand(rec, rows)
ev = {e["name"]: f for f, e in events}
ori = np.stack([bone_orientation(q, rec.labels).numpy() for q in quats]).astype(np.float64)
B, L, R = ev["birdDogBaseCalibrate"], ev["birdDogLeftReachCalibrate"], ev["birdDogRightReachCalibrate"]
mean = lambda i, f: Rotation.from_matrix(ori[f - 60:f, i]).mean()
idx = {l: rec.labels.index(l) for l in rec.labels}
temps = {"rs": (mean(idx["rs"], B), mean(idx["rs"], L)), "ls": (mean(idx["ls"], B), mean(idx["ls"], R)),
         "waist(L)": (mean(idx["waist"], B), mean(idx["waist"], L)), "waist(R)": (mean(idx["waist"], B), mean(idx["waist"], R))}
for name, (b, r) in temps.items():
    rel = (b.inv() * r).as_rotvec(); i = idx[name[:5].strip("(") if name.startswith("waist") else name]
    axis_local = rel / np.linalg.norm(rel)
    print(f"{name}: base->raised {np.degrees(np.linalg.norm(rel)):.0f} deg, local axis {axis_local.round(2)}"
          + (f" (bone axis {BONE_AXES[name]})" if name in BONE_AXES else ""))
print("\nt  | rs: to base / to raised | ls: to base / to raised | waist to base")
for t in range(0, len(ori), 30):
    line = f"{t/60:5.1f}"
    for lab in ("rs", "ls"):
        r = Rotation.from_matrix(ori[t, idx[lab]])
        b, s = temps[lab]
        line += f" | {lab} {np.degrees((b.inv()*r).magnitude()):4.0f} {np.degrees((s.inv()*r).magnitude()):4.0f}"
    w = Rotation.from_matrix(ori[t, idx["waist"]])
    line += f" | w {np.degrees((temps['waist(L)'][0].inv()*w).magnitude()):4.0f}"
    print(line)

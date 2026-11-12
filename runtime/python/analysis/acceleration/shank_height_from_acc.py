"""Can the raised leg be told from kneeling by integrating the shank's vertical acceleration?
Zero-velocity resets while the shank is still; prints the integrated height per 0.5 s.
python -m analysis.acceleration.shank_height_from_acc <recording>
"""
import json, sys
import numpy as np
from scipy.spatial.transform import Rotation
from production.calibration import bone_orientation
from production.input_recording import load_input_recording, resolve_recording_path
from production.offline_compare import _expand

path = resolve_recording_path(sys.argv[1])
rec = load_input_recording(path)
rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
quats, accels, events = _expand(rec, rows)
ev = {e["name"]: f for f, e in events}
# rows repeat when ticks > 1: use unique samples only, with their true timing
print("events (s):", {k: round(v / 60, 1) for k, v in ev.items()})
for lab in ("rs", "ls"):
    i = rec.labels.index(lab)
    ori = np.stack([bone_orientation(q, rec.labels).numpy()[i] for q in quats]).astype(np.float64)
    acc_y = accels[:, i, 1].astype(np.float64)         # server and SMPL share +Y up
    rate = np.r_[0, np.degrees(Rotation.from_matrix(ori[1:]).inv().__mul__(Rotation.from_matrix(ori[:-1])).magnitude())] * 60
    still = np.convolve(rate < 10, np.ones(15) / 15, "same") > 0.99
    v = h = 0.0; heights = []
    for t in range(len(acc_y)):
        if still[t]:
            v = 0.0
        else:
            v += acc_y[t] / 60
        h += v / 60
        heights.append(h)
    heights = np.asarray(heights)
    print(f"{lab}: height change (m) every 1 s:")
    print("  " + " ".join(f"{t}s:{heights[t*60]:+.2f}" for t in range(0, len(heights) // 60)))

"""Model upper arm during the bird-dog reach holds, raw input vs. the recording's own final
calibration applied throughout (in-sample for the forearm heading).
python -m analysis.legs.bird_dog_reach_eval <recording> [models]
"""
import json, sys
import numpy as np
import torch
from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from production.calibration import bone_orientation
from production.input_recording import load_input_recording, resolve_recording_path
from production.offline_compare import _expand, run_model, joint_positions, calibrated_orientations
from production.pose_calibration import _best_yaw, _yaw

torch.set_grad_enabled(False)
path = resolve_recording_path(sys.argv[1])
models = sys.argv[2].split(",") if len(sys.argv) > 2 else ["waist_mobileposer_amass", "waist_streamposer_ours_offsetfree"]
rec = load_input_recording(path)
rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
quats, accels, events = _expand(rec, rows)
raw = np.stack([bone_orientation(q, rec.labels).numpy() for q in quats]).astype(np.float64)
_, cal, _ = calibrated_orientations(rec, quats, events)
inputs = {"raw": raw.astype(np.float32),
          "calibrated": (cal.headings[None] @ raw @ cal.corrections[None]).astype(np.float32)}
ev = {e["name"]: f for f, e in events}
holds = {"L reach (lw)": (ev["birdDogLeftReachCalibrate"] - 300, ev["birdDogLeftReachCalibrate"], 16, 18, 20, 1),
         "R reach (rw)": (ev["birdDogRightReachCalibrate"] - 300, ev["birdDogRightReachCalibrate"], 17, 19, 21, -1)}
body = ParametricModel(paths.smpl_file)
wi = rec.labels.index("waist")
print(f"{path.name}: upper arm azimuth vs trunk forward (+ = outward) / elevation / elbow flex, mean over the last 5 s of each reach step")
for cond, ori in inputs.items():
    for m in models:
        j = joint_positions(body, run_model(m, rec, quats, accels, ori), 1)
        line = f"  {cond:10s} {m:34s}"
        for name, (a, b, sh, el, wr, side) in holds.items():
            vals = []
            for t in range(a, b, 5):
                facing = _yaw(_best_yaw([ori[t, wi].astype(np.float64)]))
                p = j[t] @ facing
                up = p[el] - p[sh]; fo = p[wr] - p[el]
                up /= np.linalg.norm(up); fo /= np.linalg.norm(fo)
                vals.append((np.degrees(np.arctan2(side * up[0], up[2])), np.degrees(np.arcsin(up[1])),
                             np.degrees(np.arccos(np.clip(up @ fo, -1, 1)))))
            v = np.mean(vals, 0)
            line += f" | {name} {v[0]:+5.0f} {v[1]:+4.0f} e{v[2]:3.0f}"
        print(line, flush=True)

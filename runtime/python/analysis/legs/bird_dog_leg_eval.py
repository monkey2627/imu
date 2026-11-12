"""Bird-dog legs: is the raised leg's shank orientation distinguishable from the kneeling base,
and what do the models predict for its thigh / knee?  Also forearm/upper-arm after calibration.
python -m analysis.legs.bird_dog_leg_eval <calibration recording> [models]
"""
import json, sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from production.calibration import bone_orientation
from production.input_recording import load_input_recording, resolve_recording_path
from production.offline_compare import _expand, run_model, joint_positions, calibrated_orientations
from production.pose_calibration import BONE_AXES, _best_yaw, _yaw

torch.set_grad_enabled(False)
path = resolve_recording_path(sys.argv[1])
models = sys.argv[2].split(",") if len(sys.argv) > 2 else [
    "waist_mobileposer_amass", "waist_streamposer_amass", "waist_line14_amass",
    "waist_mobileposer_ours_offsetfree", "waist_streamposer_ours_offsetfree", "waist_line14_ours_offsetfree"]
rec = load_input_recording(path)
rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
quats, accels, events = _expand(rec, rows)
raw = np.stack([bone_orientation(q, rec.labels).numpy() for q in quats]).astype(np.float64)
_, cal, msgs = calibrated_orientations(rec, quats, events)
print(msgs[-1][1][:120])
ori = (cal.headings[None] @ raw @ cal.corrections[None]).astype(np.float32)
ev = {e["name"]: f for f, e in events}
B, L, R = ev["birdDogBaseCalibrate"], ev["birdDogLeftReachCalibrate"], ev["birdDogRightReachCalibrate"]
wi = rec.labels.index("waist")
win = lambda f: range(f - 180, f, 5)          # last 3 s of the step
mean = lambda lab, f: Rotation.from_matrix(ori[f - 60:f, rec.labels.index(lab)]).mean().as_matrix()

def rel(vec, t):
    yaw = _best_yaw([ori[t, wi].astype(np.float64)])
    az = np.degrees((np.arctan2(vec[0], vec[2]) - yaw + np.pi) % (2 * np.pi) - np.pi)
    return az, np.degrees(np.arcsin(np.clip(vec[1] / np.linalg.norm(vec), -1, 1)))

print("\nInput shank (calibrated): angle between the raised-leg hold and the kneeling base (full rotation)")
for lab, f in (("rs", L), ("ls", R)):
    a = mean(lab, B); b = mean(lab, f)
    print(f"  {lab}: base az/el {rel(a @ BONE_AXES[lab], B-1)}  raised az/el {rel(b @ BONE_AXES[lab], f-1)}"
          f"  -> rotation between them {np.degrees(Rotation.from_matrix(a.T @ b).magnitude()):.0f} deg")
print("\nInput forearm (calibrated) during its reach: az (+=toward body-left) / el")
for lab, f in (("lw", L), ("rw", R)):
    v = [rel(ori[t, rec.labels.index(lab)] @ BONE_AXES[lab], t) for t in win(f)]
    print(f"  {lab}: {np.mean(v, 0).round(0)}")

body = ParametricModel(paths.smpl_file)
print("\nModel, mean over the last 3 s of each step. legs: thigh el (0=horizontal,-90=down) / knee flex;"
      " arm: upper-arm az out / el / elbow flex")
for m in models:
    j = joint_positions(body, run_model(m, rec, quats, accels, ori), 1)
    def seg(a, b, c, t):
        u = j[t, b] - j[t, a]; w = j[t, c] - j[t, b]
        return u, np.degrees(np.arccos(np.clip(u @ w / np.linalg.norm(u) / np.linalg.norm(w), -1, 1)))
    out = []
    for name, f, (hip, knee, ankle) in (("base R-leg", B, (2, 5, 8)), ("L-reach R-leg", L, (2, 5, 8)),
                                        ("R-reach L-leg", R, (1, 4, 7))):
        v = []
        for t in win(f):
            u, flex = seg(hip, knee, ankle, t)
            v.append((rel(u, t)[1], flex))
        v = np.mean(v, 0); out.append(f"{name} th{v[0]:+4.0f} k{v[1]:3.0f}")
    for name, f, (sh, el, wr, side) in (("L arm", L, (16, 18, 20, 1)), ("R arm", R, (17, 19, 21, -1))):
        v = []
        for t in win(f):
            u, flex = seg(sh, el, wr, t)
            az, e = rel(u, t)
            v.append((side * az, e, flex))
        v = np.mean(v, 0); out.append(f"{name} {v[0]:+4.0f} {v[1]:+4.0f} e{v[2]:3.0f}")
    print(f"  {m:34s} " + " | ".join(out), flush=True)

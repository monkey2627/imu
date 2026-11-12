"""Effect of the straight-arm prior (production/arm_prior.py) on real recordings.

1. unity_20261103_002526 (standing; heading-only arms-forward calibration from hold F1, as in
   arms_calib_ablation): upper arm azimuth (0 = forward) / elbow flex per hold, before -> after.
2. calibration_20261103_215533 (bird dog; its own calibration): upper arm vs trunk forward in the reaches.
Also reports the prior's weight per hold, so holds it must not touch (hanging, sideways) are visible.

python -m analysis.arms.arm_prior_eval   (from realUse/)
"""
import json
import numpy as np
import torch

from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from production.arm_prior import straighten_reaching_arms
from production.calibration import bone_orientation
from production.input_recording import load_input_recording, resolve_recording_path
from production.offline_compare import _expand, run_model, calibrated_orientations
from production.pose_calibration import _best_yaw, _yaw
from analysis.arms.arms_calib_ablation import calibrate

MODELS = ["waist_mobileposer_amass", "waist_streamposer_ours_offsetfree", "waist_line14_amass"]
STAND_HOLDS = {"F1 fwd palm-down": (4.25, 5.0), "F2 fwd palm-down": (15.0, 15.75), "O1 fwd palm-in": (10.0, 10.75),
               "O2 fwd palm-in": (19.25, 20.0), "S sideways": (23.25, 24.0), "H hanging": (12.6, 13.4)}


def arm_stats(body, poses, ori, wi, frames, rel_axis="facing"):
    """Mean (upper-arm azimuth out, elevation, elbow flex) per arm over `frames`, and mean prior weight."""
    res, weights = [], []
    for t in frames:
        p_local = torch.from_numpy(poses[t]).float()
        _, j = body.forward_kinematics(p_local[None])[:2]
        j = j[0].numpy()
        facing = _yaw(_best_yaw([ori[t, wi].astype(np.float64)]))
        p = j @ facing
        row = []
        for sh, el, wr, side in ((16, 18, 20, 1), (17, 19, 21, -1)):
            up = p[el] - p[sh]; fo = p[wr] - p[el]
            up /= np.linalg.norm(up); fo /= np.linalg.norm(fo)
            row.append((np.degrees(np.arctan2(side * up[0], up[2])), np.degrees(np.arcsin(up[1])),
                        np.degrees(np.arccos(np.clip(up @ fo, -1, 1)))))
        res.append(row)
    return np.mean(res, 0)


def main():
    torch.set_grad_enabled(False)
    body = ParametricModel(paths.smpl_file)
    # 1. standing recording
    path = resolve_recording_path("unity_20261103_002526")
    rec = load_input_recording(path)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    quats, accels, _ = _expand(rec, rows)
    raw = np.stack([bone_orientation(q, rec.labels).numpy() for q in quats])
    ori, _ = calibrate(rec.labels, raw, two_pose=False)
    wi = rec.labels.index("waist")
    print("== standing, heading-only calibration.  per hold: L az/el/elbow -> after | R ... | prior weight L,R")
    for m in MODELS:
        poses = run_model(m, rec, quats, accels, ori)
        fore = lambda t: (ori[t, rec.labels.index("lw")], ori[t, rec.labels.index("rw")])
        after = np.stack([straighten_reaching_arms(p, fore(t)) for t, p in enumerate(poses)]).astype(np.float32)
        weights = np.array([straighten_reaching_arms(poses[t], fore(t), return_weights=True)[1] for t in range(0, len(poses), 5)])
        print(f"  {m}")
        for h, (a, b) in STAND_HOLDS.items():
            fr = range(int(a * 60), int(b * 60), 5)
            s0, s1 = arm_stats(body, poses, ori, wi, fr), arm_stats(body, after, ori, wi, fr)
            w = weights[int(a * 12):int(b * 12)].mean(0)
            print(f"    {h:18s} L {s0[0,0]:+4.0f}/{s0[0,1]:+4.0f}/{s0[0,2]:3.0f} -> {s1[0,0]:+4.0f}/{s1[0,1]:+4.0f}/{s1[0,2]:3.0f}"
                  f" | R {s0[1,0]:+4.0f}/{s0[1,1]:+4.0f}/{s0[1,2]:3.0f} -> {s1[1,0]:+4.0f}/{s1[1,1]:+4.0f}/{s1[1,2]:3.0f}"
                  f" | w {w[0]:.2f},{w[1]:.2f}")
    # 2. bird dog calibration recording
    path = resolve_recording_path("calibration_20261103_215533")
    rec = load_input_recording(path)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    quats, accels, events = _expand(rec, rows)
    raw = np.stack([bone_orientation(q, rec.labels).numpy() for q in quats]).astype(np.float64)
    _, cal, _ = calibrated_orientations(rec, quats, events)
    ori = (cal.headings[None] @ raw @ cal.corrections[None]).astype(np.float32)
    ev = {e["name"]: f for f, e in events}
    print("\n== bird dog reaches (last 3 s of each reach step).  reaching arm: az/el/elbow -> after | weight")
    for m in MODELS:
        poses = run_model(m, rec, quats, accels, ori)
        fore = lambda t: (ori[t, rec.labels.index("lw")], ori[t, rec.labels.index("rw")])
        after = np.stack([straighten_reaching_arms(p, fore(t)) for t, p in enumerate(poses)]).astype(np.float32)
        line = f"  {m:34s}"
        for name, f, k in (("L reach", ev["birdDogLeftReachCalibrate"], 0), ("R reach", ev["birdDogRightReachCalibrate"], 1)):
            fr = range(f - 180, f, 5)
            s0, s1 = arm_stats(body, poses, ori, wi, fr), arm_stats(body, after, ori, wi, fr)
            w = np.mean([straighten_reaching_arms(poses[t], fore(t), return_weights=True)[1][k] for t in fr])
            line += (f" | {name} {s0[k,0]:+4.0f}/{s0[k,1]:+4.0f}/{s0[k,2]:3.0f} -> "
                     f"{s1[k,0]:+4.0f}/{s1[k,1]:+4.0f}/{s1[k,2]:3.0f} w {w:.2f}")
        print(line, flush=True)


if __name__ == "__main__":
    main()

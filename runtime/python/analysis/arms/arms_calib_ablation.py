"""Same recording, three inputs: raw (no Unity calibration), heading-only arms-forward
calibration, and stand + arms-forward two-pose calibration (the old Unity flow).  Calibrated
on hold F1 + the arms-hanging window, evaluated on all holds (F2/O1/O2/S are held out).

python -m analysis.arms.arms_calib_ablation unity_20261103_002526   (from realUse/)
"""
import json
import sys
import numpy as np
import torch

from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from production.calibration import bone_orientation
from production.input_recording import load_input_recording, resolve_recording_path
from production.offline_compare import _expand, run_model, joint_positions
from production.pose_calibration import PoseCalibrator, _best_yaw, _yaw
from analysis.arms.arms_input_timeline import arm_state

STAND_AT, FORWARD_AT = 2.4, 5.0            # end of the 0.5 s calibration windows (s)
HOLDS = {"F1 fwd palm-down (calib)": (4.25, 5.0), "F2 fwd palm-down": (15.0, 15.75),
         "O1 fwd palm-in": (10.0, 10.75), "O2 fwd palm-in": (19.25, 20.0), "S sideways": (23.25, 24.0)}
MODELS = ["waist_mobileposer_amass", "waist_streamposer_amass", "waist_line14_amass",
          "waist_mobileposer_ours_offsetfree", "waist_streamposer_ours_offsetfree", "waist_line14_ours_offsetfree"]


def calibrate(labels, ori, two_pose):
    cal = PoseCalibrator(labels)
    msgs = []
    for t, o in enumerate(ori):
        cal.correct(torch.from_numpy(o))
        if two_pose and t == int(STAND_AT * 60):
            cal.request_stand()
        if t == int(FORWARD_AT * 60):
            cal.request_arms_forward()
        m = cal.advance(1)
        if m:
            msgs.append(m)
    out = cal.headings[None] @ ori.astype(np.float64) @ cal.corrections[None]   # constant correction everywhere
    return out.astype(np.float32), msgs


def main():
    torch.set_grad_enabled(False)
    path = resolve_recording_path(sys.argv[1])
    rec = load_input_recording(path)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    quats, accels, _ = _expand(rec, rows)
    raw = np.stack([bone_orientation(q, rec.labels).numpy() for q in quats])
    wi = rec.labels.index("waist")
    body = ParametricModel(paths.smpl_file)
    inputs = {"raw": raw}
    for name, two in (("heading-only", False), ("two-pose", True)):
        inputs[name], msgs = calibrate(rec.labels, raw, two)
        print(f"{name}: {msgs}")
    for cond, ori in inputs.items():
        print(f"\n=== input: {cond}")
        rows_out = {h: [] for h in HOLDS}
        for h, (a, b) in HOLDS.items():
            vals = []
            for t in range(int(a * 60), int(b * 60), 5):
                facing = _yaw(_best_yaw([ori[t, wi].astype(np.float64)]))
                vals.append([arm_state(facing.T @ ori[t, rec.labels.index(l)], l)[:3] for l in ("lw", "rw")])
            v = np.nanmean(np.asarray(vals, dtype=float), 0)
            print(f"  input {h:26s} L az {v[0,0]:+4.0f} el {v[0,1]:+4.0f} tw {v[0,2]:+4.0f} | "
                  f"R az {v[1,0]:+4.0f} el {v[1,1]:+4.0f} tw {v[1,2]:+4.0f}")
        for m in MODELS:
            j = joint_positions(body, run_model(m, rec, quats, accels, ori), 1)
            line = f"  {m:34s}"
            for h, (a, b) in HOLDS.items():
                res = []
                for t in range(int(a * 60), int(b * 60), 5):
                    facing = _yaw(_best_yaw([ori[t, wi].astype(np.float64)]))
                    p = j[t] @ facing
                    for sh, el, wr, side in ((16, 18, 20, 1), (17, 19, 21, -1)):
                        up = p[el] - p[sh]; fo = p[wr] - p[el]
                        up /= np.linalg.norm(up); fo /= np.linalg.norm(fo)
                        res.append((np.degrees(np.arctan2(side * up[0], up[2])),
                                    np.degrees(np.arccos(np.clip(up @ fo, -1, 1)))))
                r = np.asarray(res).reshape(-1, 2, 2).mean(0)
                line += f" | {h[:2]} {r[0,0]:+4.0f}/{r[0,1]:2.0f} {r[1,0]:+4.0f}/{r[1,1]:2.0f}"
            print(line, flush=True)


if __name__ == "__main__":
    main()

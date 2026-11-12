"""Timeline of the forearm input in the waist-facing frame, plus the live model's upper arm.

For each arm: bone axis azimuth (0 = forward, 90 = straight out to that side), elevation,
twist relative to the SMPL ideal of 'level forward, palms down' (R_y(-+90)), and the
total angle to that ideal.  python -m analysis.arms.arms_input_timeline <recording> [model]
"""
import json
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from production.calibration import bone_orientation
from production.input_recording import load_input_recording, resolve_recording_path
from production.offline_compare import _expand, run_model, joint_positions
from production.pose_calibration import ARMS_FORWARD, BONE_AXES, _best_yaw, _yaw


def arm_state(local, lab):
    """local: forearm rotation in the facing frame -> (azimuth out to side, elevation, palm-down twist, total)."""
    d = local @ BONE_AXES[lab]
    side = 1.0 if lab == "lw" else -1.0                      # +x is the body's left
    az = np.degrees(np.arctan2(side * d[0], d[2]))
    el = np.degrees(np.arcsin(np.clip(d[1], -1, 1)))
    # palm normal of the SMPL forearm frame (T-pose palms down -> -y)
    palm = local @ np.array([0.0, -1.0, 0.0])
    # twist about the bone axis: angle of palm vs. world-down projected perpendicular to the bone
    down = np.array([0.0, -1.0, 0.0]) - d * (-d[1])
    down_n = np.linalg.norm(down)
    if down_n < 0.3:
        tw = np.nan
    else:
        down /= down_n
        tw = np.degrees(np.arctan2(np.dot(np.cross(down, palm), d), np.dot(down, palm)))
    total = np.degrees(Rotation.from_matrix(ARMS_FORWARD[lab].T @ local).magnitude())
    return az, el, tw, total


def main():
    torch.set_grad_enabled(False)
    path = resolve_recording_path(sys.argv[1])
    models = sys.argv[2].split(",") if len(sys.argv) > 2 else ["waist_mobileposer_amass"]
    rec = load_input_recording(path)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    quats, accels, events = _expand(rec, rows)
    ori = np.stack([bone_orientation(q, rec.labels).numpy() for q in quats])
    wi = rec.labels.index("waist")
    body = ParametricModel(paths.smpl_file)
    preds = {}
    for m in models:
        poses = run_model(m, rec, quats, accels, ori)
        preds[m] = joint_positions(body, poses, 1)
    step = 15
    print(f"{path.name}: {len(quats)/60:.1f}s.  per arm: az(0=fwd,90=out) / elev / palm twist(0=palm down,"
          f" +90=palm in for... sign see code) / total-vs-fwd-palmdown;  model: upper-arm az / elev / elbow flex")
    for t in range(0, len(quats), step):
        facing = _yaw(_best_yaw([ori[t, wi].astype(np.float64)]))
        line = f"{t/60:6.2f}s"
        for lab in ("lw", "rw"):
            i = rec.labels.index(lab)
            az, el, tw, tot = arm_state(facing.T @ ori[t, i], lab)
            line += f" | {lab} {az:+4.0f} {el:+4.0f} {tw:+5.0f} {tot:4.0f}"
        for m, j in preds.items():
            p = j[t] @ facing
            for sh, elb, wr, side in ((16, 18, 20, 1), (17, 19, 21, -1)):
                up = p[elb] - p[sh]; fo = p[wr] - p[elb]
                up /= np.linalg.norm(up); fo /= np.linalg.norm(fo)
                uaz = np.degrees(np.arctan2(side * up[0], up[2])); uel = np.degrees(np.arcsin(up[1]))
                flex = np.degrees(np.arccos(np.clip(up @ fo, -1, 1)))
                line += f" | U {uaz:+4.0f} {uel:+4.0f} e{flex:3.0f}"
        print(line)


if __name__ == "__main__":
    main()

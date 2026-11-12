"""Real recordings: at every armsForwardCalibrate hold, is the calibrated forearm input really
'level, forward, palms down', and what upper arm does each model predict?

python -m analysis.arms.arms_forward_real <rec1> <rec2> ...   (from realUse/)
"""
import json
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from production.input_recording import load_input_recording, resolve_recording_path
from production.offline_compare import _expand, run_model, joint_positions, calibrated_orientations
from production.pose_calibration import ARMS_FORWARD, BONE_AXES, _best_yaw, _yaw
from production.calibration import bone_orientation

MODELS = ["waist_mobileposer_amass", "waist_streamposer_amass", "waist_line14_amass",
          "waist_mobileposer_ours_offsetfree", "waist_streamposer_ours_offsetfree", "waist_line14_ours_offsetfree"]
WIN = 30


def deg(m):
    return float(np.degrees(Rotation.from_matrix(m).magnitude()))


def main():
    torch.set_grad_enabled(False)
    body = ParametricModel(paths.smpl_file)
    for name in sys.argv[1:]:
        path = resolve_recording_path(name)
        rec = load_input_recording(path)
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        quats, accels, events = _expand(rec, rows)
        arms = [f for f, e in events if e["name"] == "armsForwardCalibrate"]
        if not arms:
            continue
        f = arms[-1]
        end = min(len(quats), f + 90)
        ori, cal, msgs = calibrated_orientations(rec, quats[:end], [(a, e) for a, e in events if a < end])
        # state right after the arms solve = final state (no later arm events inside [:end])
        raw = np.stack([bone_orientation(q, rec.labels).numpy() for q in quats[f - WIN + 1:f + 1]]).astype(np.float64)
        win = cal.headings[None] @ raw @ cal.corrections[None]
        wi = rec.labels.index("waist")
        facing = _yaw(_best_yaw([win[:, wi].mean(0) if False else Rotation.from_matrix(win[:, wi]).mean().as_matrix()]))
        print(f"\n== {path.name}  (arms hold t={f/60:.1f}s)  {[m for _, m in msgs if 'arm' in m][-1:]}")
        for lab in ("lw", "rw"):
            i = rec.labels.index(lab)
            m = Rotation.from_matrix(win[:, i]).mean().as_matrix()
            local = facing.T @ m                     # in the body-facing frame
            d = local @ BONE_AXES[lab]
            az = np.degrees(np.arctan2(d[0], d[2])); el = np.degrees(np.arcsin(d[1]))
            err = deg(local.T @ ARMS_FORWARD[lab])
            # twist about the bone axis between measurement and target
            tw = Rotation.from_matrix(ARMS_FORWARD[lab].T @ local).as_rotvec()
            twist = np.degrees(np.dot(tw, BONE_AXES[lab]))
            print(f"  input {lab}: bone az {az:+5.0f} (0=fwd, + = toward left)  elev {el:+4.0f}  "
                  f"total err vs target {err:4.0f}  twist {twist:+4.0f}")
        for model_id in MODELS:
            poses = run_model(model_id, rec, quats[:end], accels[:end], ori[:end])
            j = joint_positions(body, poses[f - WIN + 1:f + 1], 1) @ facing     # rows: facing^T p
            out = []
            for sh, el, wr in ((16, 18, 20), (17, 19, 21)):
                up = j[:, el] - j[:, sh]; fo = j[:, wr] - j[:, el]
                up /= np.linalg.norm(up, axis=-1, keepdims=True); fo /= np.linalg.norm(fo, axis=-1, keepdims=True)
                fwd = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1))).mean()
                elbow = np.degrees(np.arccos(np.clip((up * fo).sum(-1), -1, 1))).mean()
                fo_fwd = np.degrees(np.arccos(np.clip(fo[:, 2], -1, 1))).mean()
                out.append(f"upper {fwd:3.0f}/elbow {elbow:3.0f}/forearm {fo_fwd:3.0f}")
            print(f"  {model_id:36s} L {out[0]} | R {out[1]}", flush=True)


if __name__ == "__main__":
    main()

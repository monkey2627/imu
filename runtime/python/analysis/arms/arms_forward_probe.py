"""Probe: can the models tell 'arms straight forward' from 'upper arm sideways + elbow 90 deg'?

Both end poses give the forearm sensor exactly the same global orientation (SMPL R_y(-+90)),
so only the motion (acceleration / orientation trajectory) can separate them.
Synthetic SMPL-frame input: orientation = global rotation of the sensed bone, acceleration =
2nd derivative of a point 70% down the forearm (gravity-free, m/s^2), waist/shanks static.

python -m analysis.arms.arms_forward_probe   (from realUse/)
"""
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from production.model_backends import (
    WAIST_MODEL_FILES, apply_input_offsets, make_waist_session, resolve_waist_model)
from production.calibration import identity_heading
from production.layout import ResolvedLayout

FPS = 60
LABELS = ["lw", "rw", "ls", "rs", "waist"]
KEYS = {"lw": "10:0", "rw": "2:0", "ls": "6:0", "rs": "4:0", "waist": "8:0"}
R = lambda ax, deg: Rotation.from_euler(ax, deg, degrees=True).as_matrix()
LEFT = dict(down=R("z", -90), side=np.eye(3), fwd=R("y", -90), lat="x")
RIGHT = dict(down=R("z", 90), side=np.eye(3), fwd=R("y", 90))


def arm_globals(kind, s):
    """(upper, forearm) global rotation of the LEFT arm at progress s in [0,1]."""
    if kind == "A":   # straight arm raised in the sagittal plane, pronating to palms down
        raise_ = R("x", -90 * s) @ LEFT["down"]
        twist_end = (R("x", -90) @ LEFT["down"]).T @ LEFT["fwd"]          # about the bone axis
        tw = Rotation.from_matrix(twist_end).as_rotvec() * s
        g = raise_ @ Rotation.from_rotvec(tw).as_matrix()
        return g, g
    # B: abduct to T-pose (0..0.5), then flex the elbow horizontally (0.5..1)
    if s < 0.5:
        g = R("z", -90 * (1 - 2 * s))
        return g, g
    return np.eye(3), R("y", -90 * (2 * s - 1))


def mirror(m):
    """Left-arm rotation -> right arm (reflect x)."""
    f = np.diag([-1.0, 1.0, 1.0])
    return f @ m @ f


def sequence(kind, hang=1.0, move=1.2, hold=4.0):
    n_hang, n_move, n_hold = int(hang * FPS), int(move * FPS), int(hold * FPS)
    s = np.r_[np.zeros(n_hang), 0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, n_move)), np.ones(n_hold)]
    T = len(s)
    local = np.tile(np.eye(3), (T, 24, 1, 1))
    for t, si in enumerate(s):
        up, fo = arm_globals(kind, si)
        local[t, 16], local[t, 18] = up, up.T @ fo
        upr, for_ = mirror(up), mirror(fo)
        local[t, 17], local[t, 19] = upr, upr.T @ for_
    return local.astype(np.float32), n_hang + n_move


def synth_input(body, local):
    pose = torch.from_numpy(local)
    glob, joints = body.forward_kinematics(pose)[:2]
    glob, joints = glob.numpy(), joints.numpy()
    ori = np.stack([glob[:, 18], glob[:, 19], glob[:, 4], glob[:, 5], glob[:, 0]], 1)
    pos = np.stack([joints[:, 18] + 0.7 * (joints[:, 20] - joints[:, 18]),
                    joints[:, 19] + 0.7 * (joints[:, 21] - joints[:, 19]),
                    joints[:, 4], joints[:, 5], joints[:, 0]], 1)
    acc = np.zeros_like(pos)
    acc[1:-1] = (pos[2:] - 2 * pos[1:-1] + pos[:-2]) * FPS ** 2
    return ori.astype(np.float32), acc.astype(np.float32)


def run(model_id, ori, acc, lag):
    resolved = ResolvedLayout("wrists_shanks_waist", LABELS, KEYS)
    session = make_waist_session(model_id, resolved, identity_heading(LABELS), loaded=resolve_waist_model(model_id))
    ori_in = np.stack([apply_input_offsets(model_id, LABELS, torch.from_numpy(o)).numpy() for o in ori])
    out = np.tile(np.eye(3, dtype=np.float32), (len(ori), 24, 1, 1))
    last = out[0]
    for t in range(len(ori)):
        feature = np.concatenate((acc[t].reshape(-1) / 30.0, ori_in[t].reshape(-1))).astype(np.float32)
        if hasattr(session, "push_observation"):
            session.push_observation(feature, torch.from_numpy(ori_in[t]), torch.from_numpy(acc[t]))
        else:
            session.push_feature(feature)
        if t == 0:
            for _ in range(130):    # prime with the first (arms hanging) frame
                session.push_observation(feature, torch.from_numpy(ori_in[t]), torch.from_numpy(acc[t])) \
                    if hasattr(session, "push_observation") else session.push_feature(feature)
        if t % 2 == 1:
            last = session.infer().detach().numpy()
        if t - lag >= 0:
            out[t - lag] = last
    return out


def arm_metrics(body, local):
    """Per frame: left/right upper-arm angle from forward (deg, 0=forward, 90=sideways), elbow flex (deg)."""
    _, j = body.forward_kinematics(torch.from_numpy(local))[:2]
    j = j.numpy()
    res = []
    for sh, el, wr, side in ((16, 18, 20, 1), (17, 19, 21, -1)):
        up = j[:, el] - j[:, sh]; fo = j[:, wr] - j[:, el]
        up /= np.linalg.norm(up, axis=-1, keepdims=True); fo /= np.linalg.norm(fo, axis=-1, keepdims=True)
        fwd = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))
        elbow = np.degrees(np.arccos(np.clip((up * fo).sum(-1), -1, 1)))
        elev = np.degrees(np.arcsin(np.clip(up[:, 1], -1, 1)))
        res.append((fwd, elbow, elev))
    return res


def main():
    torch.set_grad_enabled(False)
    body = ParametricModel(paths.smpl_file)
    models = [m for m in WAIST_MODEL_FILES if "deadbug" not in m] if len(sys.argv) < 2 else sys.argv[1].split(",")
    seqs = {k: sequence(k) for k in ("A", "B")}
    inputs = {k: synth_input(body, v[0]) for k, v in seqs.items()}
    oa, ob = inputs["A"][0][-1], inputs["B"][0][-1]
    print("final forearm input A vs B (deg):",
          [round(float(np.degrees(Rotation.from_matrix(oa[i].T @ ob[i]).magnitude())), 3) for i in range(2)])
    for k in ("A", "B"):
        gt = arm_metrics(body, seqs[k][0])
        print(f"GT {k} end: L fwd {gt[0][0][-1]:.0f} elbow {gt[0][1][-1]:.0f} | R fwd {gt[1][0][-1]:.0f} elbow {gt[1][1][-1]:.0f}")
    print("\nupper-arm angle from forward (0=forward, 90=sideways) / elbow flex, L|R;"
          " at motion end+0.3s and after 4 s hold")
    for model_id in models:
        lag = 5 if "streamposer" in model_id else 9 if "line14" in model_id else 10
        line = f"{model_id:36s}"
        for k in ("A", "B"):
            ori, acc = inputs[k]
            pred = run(model_id, ori, acc, lag)
            m = arm_metrics(body, pred)
            end = seqs[k][1]
            for name, t in (("mv", end + 18), ("hold", len(ori) - lag - 2)):
                line += (f" | {k}-{name} L{m[0][0][t]:3.0f}/{m[0][1][t]:3.0f}"
                         f" R{m[1][0][t]:3.0f}/{m[1][1][t]:3.0f}")
        print(line, flush=True)


if __name__ == "__main__":
    main()

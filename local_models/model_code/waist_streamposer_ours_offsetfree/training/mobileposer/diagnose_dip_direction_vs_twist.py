"""Is the large upper-arm error on DIP real, or mostly axial twist (unobservable / annotation dependent)?
Bone DIRECTION error (angle between predicted and reference parent->child joint vectors, deg) versus the
full orientation error, for v2_base on the DIP-IMU test split (real IMU as given).
Also: direction error of the arm as seen in the TORSO frame (removes the contribution of torso/pelvis error).

Run from SparseIMU/:  python -m mobileposer.diagnose_dip_direction_vs_twist
"""
import sys
from pathlib import Path

import numpy as np
import torch

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent))
sys.path.insert(0, str(_DIR.parent.parent))

from mobileposer.config import paths
from mobileposer.articulate.model import ParametricModel
from mobileposer.evaluate_v2_models import ModulePredictor
from mobileposer.evaluate_dip_real import load_dip
from drift_eval_common import angle_between_rotmats

BONES = {'upper arm (16->18)': (16, 18, [16, 17]), 'forearm (18->20)': (18, 20, [18, 19]),
         'thigh (1->4)': (1, 4, [1, 2]), 'shank (4->7)': (4, 7, [4, 5])}
SIDE = {16: 17, 18: 19, 1: 2, 4: 5, 20: 21, 7: 8}


def vec_angle(a, b):
    a, b = a / a.norm(dim=-1, keepdim=True), b / b.norm(dim=-1, keepdim=True)
    return torch.rad2deg(torch.acos((a * b).sum(-1).clamp(-1, 1)))


def main():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    body = ParametricModel(paths.smpl_file)
    model = ModulePredictor('checkpoints/v2_base_poser', 'checkpoints/v2_base_joints', False, dev)
    acc = {k: [] for k in BONES}
    n = 0
    for s in load_dip(body):
        T = len(s['pose'])
        pred = model(s['acc'], s['ori'])[:T]
        G, J = body.forward_kinematics(s['pose'], torch.ones(10), tran=None, calc_mesh=False)
        Gp, Jp = body.forward_kinematics(pred, torch.ones(10), tran=None, calc_mesh=False)
        G, Gp, J, Jp = G.reshape(T, 24, 3, 3), Gp.reshape(T, 24, 3, 3), J.reshape(T, 24, 3), Jp.reshape(T, 24, 3)
        for name, (p, c, js) in BONES.items():
            full = angle_between_rotmats(Gp[:, js], G[:, js]).mean(-1)
            dr, dr_t = [], []
            for a in js:
                ch = c if a == p else SIDE[c]
                v, vp = J[:, ch] - J[:, a], Jp[:, ch] - Jp[:, a]
                dr.append(vec_angle(v, vp))
                # in the torso (spine3, joint 9) frame of each skeleton
                vt = (G[:, 9].transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)
                vpt = (Gp[:, 9].transpose(-1, -2) @ vp.unsqueeze(-1)).squeeze(-1)
                dr_t.append(vec_angle(vt, vpt))
            acc[name].append((T, full.mean().item(), torch.stack(dr).mean().item(), torch.stack(dr_t).mean().item()))
        n += T
    print(f'DIP test, {n} frames, v2_base, real IMU (deg)')
    print('bone'.ljust(22) + 'full orientation   direction only   direction in torso frame')
    for name, rows in acc.items():
        w = np.array([r[0] for r in rows], float)
        v = np.average([r[1:] for r in rows], axis=0, weights=w)
        print(f'{name:22s}{v[0]:12.1f}{v[1]:16.1f}{v[2]:22.1f}')


if __name__ == '__main__':
    main()

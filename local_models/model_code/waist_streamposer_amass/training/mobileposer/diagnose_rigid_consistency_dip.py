"""Can rigid-skeleton consistency between the predicted JOINT POSITIONS (Joints module) and the predicted ROTATIONS
(Poser -> forward kinematics) improve the unsensed joints?  DIP-IMU test split, real IMU, v2_base.

For the unsensed limb joints (knees, elbows, hips, shoulders) report root-relative position error (cm):
  fk      : forward kinematics of the Poser rotations (rigid by construction)
  joints  : raw Joints-module positions (NOT rigid: bone lengths vary)
  proj    : Joints positions projected onto the rigid skeleton (each child = parent + bone_length * unit(pred child - pred parent))
  avg     : mean of fk and proj positions (a first, untuned fusion)
and the bone-length violation of the raw Joints output.  If 'joints' is not better than 'fk', or 'avg' does not
beat 'fk', consistency between these two predictions alone cannot add information (both come from the same IMU input).

Run from SparseIMU/:  python -m mobileposer.diagnose_rigid_consistency_dip
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
from mobileposer.evaluate_canon_models import best_ckpt
from mobileposer.evaluate_drift_anchorpose import prepare_imu

NAMES = {'knees': [4, 5], 'elbows': [18, 19], 'hips': [1, 2], 'shoulders': [16, 17], 'wrists': [20, 21], 'ankles': [7, 8]}


@torch.no_grad()
def joints_positions(model, acc, ori):
    imu = prepare_imu(acc, ori).to(model.dev).unsqueeze(0)
    return model.joints(imu, [imu.shape[1]]).cpu().reshape(-1, 24, 3)


def project_rigid(pj, parent, length):
    out = torch.zeros_like(pj)
    out[:, 0] = pj[:, 0]
    for i in range(1, 24):
        v = pj[:, i] - pj[:, parent[i]]
        out[:, i] = out[:, parent[i]] + length[i] * v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return out


def main():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    body = ParametricModel(paths.smpl_file)
    model = ModulePredictor('checkpoints/v2_base_poser', 'checkpoints/v2_base_joints', False, dev)
    parent = body.parent
    acc = {k: {n: [] for n in NAMES} for k in ('fk', 'joints', 'proj', 'avg')}
    viol, wts = [], []
    for s in load_dip(body):
        T = len(s['pose'])
        _, J = body.forward_kinematics(s['pose'], torch.ones(10), tran=None, calc_mesh=False)
        J = J.reshape(T, 24, 3)
        length = torch.zeros(24)
        for i in range(1, 24):
            length[i] = (J[:, i] - J[:, parent[i]]).norm(dim=-1).mean()
        _, Jfk = body.forward_kinematics(model(s['acc'], s['ori'])[:T], torch.ones(10), tran=None, calc_mesh=False)
        Jfk = Jfk.reshape(T, 24, 3)
        pj = joints_positions(model, s['acc'], s['ori'])[:T]
        pr = project_rigid(pj, parent, length)
        cand = {'fk': Jfk - Jfk[:, :1], 'joints': pj - pj[:, :1], 'proj': pr - pr[:, :1]}
        cand['avg'] = 0.5 * (cand['fk'] + cand['proj'])
        ref = J - J[:, :1]
        for k, P in cand.items():
            for n, js in NAMES.items():
                acc[k][n].append((P[:, js] - ref[:, js]).norm(dim=-1).mean().item() * 100)
        bl = torch.stack([(pj[:, i] - pj[:, parent[i]]).norm(dim=-1) for i in range(1, 24)], 1)
        viol.append(((bl - length[1:]).abs() / length[1:]).mean().item() * 100)
        wts.append(T)
    print('root-relative position error (cm), DIP test, v2_base')
    print('method'.ljust(8) + ''.join(n.rjust(11) for n in NAMES))
    for k in acc:
        print(k.ljust(8) + ''.join(f'{np.average(acc[k][n], weights=wts):11.2f}' for n in NAMES))
    print(f'\nbone-length violation of the raw Joints output: mean {np.average(viol, weights=wts):.1f}% of the bone length')


if __name__ == '__main__':
    main()

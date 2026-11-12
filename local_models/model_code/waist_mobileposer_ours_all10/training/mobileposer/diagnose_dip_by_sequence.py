"""Where is the headroom on the DIP-IMU test split?  Per-sequence and per-joint error of v2_base (real IMU,
as-given inputs; DIP is already calibrated).  Unaligned global orientation error, deg, every 5th frame.

Run from SparseIMU/:  python -m mobileposer.diagnose_dip_by_sequence
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
from mobileposer.evaluate_canon_models import GROUPS
from mobileposer.evaluate_dip_real import load_dip
from drift_eval_common import angle_between_rotmats

NAMES = ['pelvis', 'L hip', 'R hip', 'spine1', 'L knee', 'R knee', 'spine2', 'L ankle', 'R ankle', 'spine3', 'L foot', 'R foot',
         'neck', 'L collar', 'R collar', 'head', 'L shoulder', 'R shoulder', 'L elbow', 'R elbow', 'L wrist', 'R wrist', 'L hand', 'R hand']


def main():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    body = ParametricModel(paths.smpl_file)
    model = ModulePredictor('checkpoints/v2_base_poser', 'checkpoints/v2_base_joints', False, dev)
    seqs = load_dip(body)
    rows, per_joint, wts = [], [], []
    for s in seqs:
        T = len(s['pose'])
        G, _ = body.forward_kinematics(s['pose'], torch.ones(10), tran=None, calc_mesh=False)
        G = G.reshape(T, 24, 3, 3)
        Gp, _ = body.forward_kinematics(model(s['acc'], s['ori'])[:T], torch.ones(10), tran=None, calc_mesh=False)
        e = angle_between_rotmats(Gp.reshape(T, 24, 3, 3), G).numpy()           # [T,24]
        per_joint.append(e.mean(0) * T)
        wts.append(T)
        # posture descriptors from the reference: mean knee flexion proxy = angle between thigh and shank, mean torso tilt
        thigh_shank = angle_between_rotmats(G[:, 1], G[:, 4]).mean().item()
        torso = torch.rad2deg(torch.acos(G[:, 3, 1, 1].clamp(-1, 1))).mean().item()        # tilt of spine1 'up' axis from world up
        rows.append((s['name'], T, [float(e[:, js].mean()) for js in GROUPS.values()], thigh_shank, torso))
    print('per joint mean error (deg), frame-weighted:')
    pj = np.sum(per_joint, axis=0) / sum(wts)
    for i in np.argsort(-pj):
        print(f'  {i:2d} {NAMES[i]:11s} {pj[i]:5.1f}')
    print('\nper sequence (sorted by thigh error): name, frames, groups', list(GROUPS), ', mean thigh-shank angle, mean torso tilt')
    for name, T, g, ts, to in sorted(rows, key=lambda r: -r[2][2]):
        print(f'  {name:14s} {T:5d}  ' + ' '.join(f'{x:5.1f}' for x in g) + f'   thigh-shank {ts:5.1f}  torso-tilt {to:5.1f}')


if __name__ == '__main__':
    main()

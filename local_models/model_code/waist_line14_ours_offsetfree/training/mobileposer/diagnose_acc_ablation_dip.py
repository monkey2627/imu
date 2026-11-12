"""How much does the trained network actually USE the accelerations on real DIP-IMU data?
v2_base on the DIP test split with (a) real acc, (b) acc set to zero, (c) acc time-shuffled across the sequence
(keeps the statistics, destroys the alignment).  Unaligned global orientation error per group (deg) and the
bone DIRECTION error of upper arm / thigh.  If the error barely changes, the network ignores the acceleration
and the physics information in it is unused.

Run from SparseIMU/:  python -m mobileposer.diagnose_acc_ablation_dip
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
from mobileposer.diagnose_dip_direction_vs_twist import vec_angle
from drift_eval_common import angle_between_rotmats


def main():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    body = ParametricModel(paths.smpl_file)
    model = ModulePredictor('checkpoints/v2_base_poser', 'checkpoints/v2_base_joints', False, dev)
    g = torch.Generator().manual_seed(0)
    conds = {'real acc': lambda a: a, 'acc = 0': lambda a: torch.zeros_like(a),
             'acc shuffled in time': lambda a: a[torch.randperm(len(a), generator=g)]}
    res = {c: [] for c in conds}
    wts = []
    for s in load_dip(body):
        T = len(s['pose'])
        G, J = body.forward_kinematics(s['pose'], torch.ones(10), tran=None, calc_mesh=False)
        G, J = G.reshape(T, 24, 3, 3), J.reshape(T, 24, 3)
        wts.append(T)
        for c, f in conds.items():
            pred = model(f(s['acc']), s['ori'])[:T]
            Gp, Jp = body.forward_kinematics(pred, torch.ones(10), tran=None, calc_mesh=False)
            Gp, Jp = Gp.reshape(T, 24, 3, 3), Jp.reshape(T, 24, 3)
            e = angle_between_rotmats(Gp, G).mean(0).numpy()
            d_arm = (vec_angle(J[:, 18] - J[:, 16], Jp[:, 18] - Jp[:, 16]).mean() + vec_angle(J[:, 19] - J[:, 17], Jp[:, 19] - Jp[:, 17]).mean()) / 2
            d_thigh = (vec_angle(J[:, 4] - J[:, 1], Jp[:, 4] - Jp[:, 1]).mean() + vec_angle(J[:, 5] - J[:, 2], Jp[:, 5] - Jp[:, 2]).mean()) / 2
            res[c].append([float(e[js].mean()) for js in GROUPS.values()] + [d_arm.item(), d_thigh.item()])
    print('DIP test, v2_base, real IMU as given (deg, frame-weighted)')
    print('condition'.ljust(24) + '  '.join(n[:15].rjust(15) for n in GROUPS) + '  upperarm dir  thigh dir')
    for c in conds:
        print(c.ljust(24) + '  '.join(f'{x:15.1f}' for x in np.average(res[c], axis=0, weights=wts)))


if __name__ == '__main__':
    main()

"""Sim-to-real acceleration bandwidth mismatch on DIP-IMU.
The training synthesis is a smoothed second difference (spacing 4 frames @30 Hz = triangular kernel of half-width
8 samples @60 Hz on the acceleration); real IMU acceleration is full-band (impacts, shank swing).  Low-pass the REAL
acceleration with a triangular kernel of half-width n (60 Hz samples) BEFORE feeding the model and measure
(a) agreement with the synthesised acceleration of the reference pose (relative acc R2, shanks / forearms),
(b) pose error on the DIP test split (v2_base and v3_base; unaligned global orientation per group + bone directions).
n = 0 is the as-given input; n = 8 matches the synthesis bandwidth.

Run from SparseIMU/:  python -m mobileposer.diagnose_acc_filter_dip
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
from mobileposer.no_head_layouts import synthesis_metadata, _synthesize
from mobileposer.surface_imu import DEFAULT_CONFIG
from mobileposer.diagnose_dip_direction_vs_twist import vec_angle
from drift_eval_common import angle_between_rotmats

WIDTHS = [0, 4, 8, 12]


def main():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    body = ParametricModel(paths.smpl_file)
    syn = synthesis_metadata('surface', DEFAULT_CONFIG)
    models = {'v2_base': ModulePredictor('checkpoints/v2_base_poser', 'checkpoints/v2_base_joints', False, dev),
              'v3_base': ModulePredictor('checkpoints/v3_base_poser', 'checkpoints/v3_base_joints', False, dev)}
    print('n   relative-acc R2: Lforearm Rforearm Lshank Rshank | model    ' + '  '.join(n[:12].rjust(12) for n in GROUPS) + '  upperarm-dir thigh-dir')
    for n in WIDTHS:
        seqs = load_dip(body, acc_filter=n)
        m_rel, s_rel = [], []
        res = {k: [] for k in models}
        wts = []
        for s in seqs:
            T = len(s['pose'])
            _, _, acc_s = _synthesize(body, s['pose'], torch.ones(10), torch.zeros(T, 3), 'wrists_shanks_waist', syn, 128)
            sl = slice(10, T - 10)
            m_rel.append((s['acc'][sl][:, :4] - s['acc'][sl][:, 4:5]).numpy()); s_rel.append((acc_s[sl][:, :4] - acc_s[sl][:, 4:5]).numpy())
            G, J = body.forward_kinematics(s['pose'], torch.ones(10), tran=None, calc_mesh=False)
            G, J = G.reshape(T, 24, 3, 3), J.reshape(T, 24, 3)
            wts.append(T)
            for k, fn in models.items():
                Gp, Jp = body.forward_kinematics(fn(s['acc'], s['ori'])[:T], torch.ones(10), tran=None, calc_mesh=False)
                Gp, Jp = Gp.reshape(T, 24, 3, 3), Jp.reshape(T, 24, 3)
                e = angle_between_rotmats(Gp, G).mean(0).numpy()
                d_arm = (vec_angle(J[:, 18] - J[:, 16], Jp[:, 18] - Jp[:, 16]).mean() + vec_angle(J[:, 19] - J[:, 17], Jp[:, 19] - Jp[:, 17]).mean()) / 2
                d_th = (vec_angle(J[:, 4] - J[:, 1], Jp[:, 4] - Jp[:, 1]).mean() + vec_angle(J[:, 5] - J[:, 2], Jp[:, 5] - Jp[:, 2]).mean()) / 2
                res[k].append([float(e[js].mean()) for js in GROUPS.values()] + [d_arm.item(), d_th.item()])
        m, sy = np.concatenate(m_rel), np.concatenate(s_rel)
        r2 = [1 - ((m[:, k] - sy[:, k]) ** 2).sum() / (m[:, k] ** 2).sum() for k in range(4)]
        for i, (k, v) in enumerate(res.items()):
            head = f'{n:<3d} {"".join(f"{x:9.2f}" for x in r2) if i == 0 else " " * 36}   | {k:8s}'
            print(head + '  '.join(f'{x:12.1f}' for x in np.average(v, axis=0, weights=wts)), flush=True)


if __name__ == '__main__':
    main()

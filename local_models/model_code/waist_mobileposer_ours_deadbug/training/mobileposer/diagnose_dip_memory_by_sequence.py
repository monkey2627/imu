"""Does long causal memory help on REAL IMU sequences that contain a posture change?  DIP-IMU test split, per sequence.
Deployed baseline (46-frame online window, bidirectional v3_base / v3_canon) vs streamed causal memory (causal125) vs
offline bidirectional.  Thigh [1,2] unaligned global orientation error and thigh bone-direction error (deg), sequences
sorted by how much the reference knee flexion varies (std of the thigh-shank angle) - sequences that enter/leave
a bent-knee posture are the ones where history matters.

Run from SparseIMU/:  python -m mobileposer.diagnose_dip_memory_by_sequence
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
from mobileposer.evaluate_causal_holds import load
from mobileposer.diagnose_realtime_gap import window_predict
from mobileposer.diagnose_dip_direction_vs_twist import vec_angle
from mobileposer.evaluate_dip_real import load_dip
from mobileposer.evaluate_v2_models import ModulePredictor
from drift_eval_common import angle_between_rotmats


def main():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    body = ParametricModel(paths.smpl_file)
    base = ModulePredictor('checkpoints/v3_base_poser', 'checkpoints/v3_base_joints', False, dev)
    canon = load('checkpoints/v3_canon_poser', 'checkpoints/v3_canon_joints', False, dev)
    causal = load('checkpoints/v3_causal125_poser', 'checkpoints/v3_causal125_joints', True, dev)
    rows = []
    for s in load_dip(body):
        T = len(s['pose'])
        frames = list(range(0, T, 3))
        G, J = body.forward_kinematics(s['pose'], torch.ones(10), tran=None, calc_mesh=False)
        G, J = G.reshape(T, 24, 3, 3), J.reshape(T, 24, 3)
        flex = angle_between_rotmats(G[:, 1], G[:, 4])
        outs = {'base 46-win': window_predict(base, s['acc'], s['ori'], frames), 'base offline': base(s['acc'], s['ori'])[:T][frames],
                'canon 46-win': window_predict(canon, s['acc'], s['ori'], frames), 'causal stream': causal(s['acc'], s['ori'])[:T][frames]}
        r = {'name': s['name'], 'T': T, 'flex_std': flex.std().item(), 'flex_max': flex.max().item()}
        for k, pred in outs.items():
            Gp, Jp = body.forward_kinematics(pred, torch.ones(10), tran=None, calc_mesh=False)
            Gp, Jp = Gp.reshape(-1, 24, 3, 3), Jp.reshape(-1, 24, 3)
            e = angle_between_rotmats(Gp[:, [1, 2]], G[frames][:, [1, 2]]).mean().item()
            d = (vec_angle(J[frames][:, 4] - J[frames][:, 1], Jp[:, 4] - Jp[:, 1]).mean() + vec_angle(J[frames][:, 5] - J[frames][:, 2], Jp[:, 5] - Jp[:, 2]).mean()).item() / 2
            r[k] = (e, d)
        rows.append(r)
        print('  done', s['name'], flush=True)
    keys = ['base 46-win', 'base offline', 'canon 46-win', 'causal stream']
    print('\nthigh orientation error / thigh direction error (deg) per sequence; sorted by knee-flexion std')
    print('sequence'.ljust(16) + 'frames  flex_std flex_max  ' + '   '.join(k.rjust(15) for k in keys))
    for r in sorted(rows, key=lambda r: -r['flex_std']):
        print(f'{r["name"]:16s}{r["T"]:6d}  {r["flex_std"]:7.1f} {r["flex_max"]:8.1f}  ' + '   '.join(f'{r[k][0]:7.1f}/{r[k][1]:6.1f}' for k in keys))
    w = np.array([r['T'] for r in rows], float)
    top = [r for r in rows if r['flex_std'] > 12]
    for name, sel in (('ALL', rows), ('flex_std > 12 deg (posture changes)', top)):
        ww = np.array([r['T'] for r in sel], float)
        print(f'{name} ({len(sel)} seq): ' + '   '.join(f'{k}: {np.average([r[k][0] for r in sel], weights=ww):.1f}/{np.average([r[k][1] for r in sel], weights=ww):.1f}' for k in keys))


if __name__ == '__main__':
    main()

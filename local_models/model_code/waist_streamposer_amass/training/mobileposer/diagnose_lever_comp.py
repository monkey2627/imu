"""Does lever-arm compensation recover the acceleration of the UNSENSED joint (elbow / knee) relative to the waist?
  a_J,rel = a_sensor - (alpha x r + w x (w x r)) - a_waist      (from sensed orientations + the template lever arm r)
compared with the true relative joint acceleration (second difference, spacing 4 like the training synthesis, of
J_bone - p_waist from the reference pose).  R2 against the raw relative acceleration (a_sensor - a_waist) shows the gain.
(a) synthetic IMU from DIP reference poses (formula / sign / finite-difference noise check)
(b) REAL DIP-IMU measurements (acceleration low-passed with the training-synthesis kernel).
Lever arms are those of the neutral template; sensors: lw rw ls rs (forearms: elbow, shanks: knee).

Run from SparseIMU/:  MOBILEPOSER_CTX=1 python -m mobileposer.diagnose_lever_comp
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
from mobileposer.body_ctx import body_context
from mobileposer.evaluate_dip_real import load_dip, triangle_smooth
from mobileposer.models.relacc import lever_features
from mobileposer.no_head_layouts import synthesis_metadata, _synthesize, LAYOUTS
from mobileposer.surface_imu import DEFAULT_CONFIG, surface_kinematics

BONE_J = [18, 19, 4, 5]
NAMES = ['L forearm/elbow', 'R forearm/elbow', 'L shank/knee', 'R shank/knee']


def second_diff4(x, fps=30.0):
    n = 4
    return (x[2 * n:] + x[:-2 * n] - 2 * x[n:-n]) * fps * fps / (n * n)


def r2(true, est):
    return 1 - ((true - est) ** 2).sum() / (true ** 2).sum()


def analyse(label, body, seqs, ctx, source):
    raw_t, comp_t, tru = [], [], []
    for s in seqs:
        T = len(s['pose'])
        acc, ori = source(s, T)                                              # world acc [T,5,3] (m/s^2), ori [T,5,3,3]
        imu = torch.cat([(acc / 30.0).reshape(T, 15), ori.reshape(T, 45), ctx.expand(T, -1)], dim=1).unsqueeze(0)
        f = lever_features(imu)[0]                                           # [T,36]; f[:, :12] = (a_J - a_waist) in the waist frame, *3/30
        Rw = ori[:, 4]
        comp_world = (Rw.unsqueeze(1) @ (f[:, :12].reshape(T, 4, 3) / 0.1).unsqueeze(-1)).squeeze(-1)
        raw_world = acc[:, :4] - acc[:, 4:5]
        _, J = body.forward_kinematics(s['pose'], torch.ones(10), tran=None, calc_mesh=False)
        J = J.reshape(T, 24, 3)
        _, _, pts = surface_kinematics(body, s['pose'], torch.ones(10), torch.zeros(T, 3), LAYOUTS['wrists_shanks_waist']['labels'], synthesis_metadata('surface', DEFAULT_CONFIG)['attachments'], 128)
        rel_pos = J[:, BONE_J] - pts[:, 4:5]                                 # joint relative to the waist sensor point
        true = second_diff4(rel_pos)                                          # [T-8,4,3]
        sl = slice(4, T - 4)
        raw_t.append(raw_world[sl].numpy()); comp_t.append(comp_world[sl].numpy()); tru.append(true.numpy())
    raw_t, comp_t, tru = np.concatenate(raw_t), np.concatenate(comp_t), np.concatenate(tru)
    print(f'\n{label}: R2 of estimating the relative acceleration of the unsensed joint')
    print('                    raw sensor-waist   lever-compensated   |true| rms (m/s^2)')
    for k in range(4):
        print(f'  {NAMES[k]:16s} {r2(tru[:, k], raw_t[:, k]):12.2f} {r2(tru[:, k], comp_t[:, k]):18.2f} {np.sqrt((tru[:, k] ** 2).sum(-1).mean()):14.2f}')


def main():
    body = ParametricModel(paths.smpl_file)
    ctx = body_context(torch.zeros(10)).unsqueeze(0)
    seqs = load_dip(body)
    syn = synthesis_metadata('surface', DEFAULT_CONFIG)

    def synthetic(s, T):
        ori, _, acc = _synthesize(body, s['pose'], torch.ones(10), torch.zeros(T, 3), 'wrists_shanks_waist', syn, 128)
        return acc[:, :5], ori[:, :5]

    def real(s, T):
        a = triangle_smooth(s['acc'], 4)                                      # training-synthesis bandwidth (half-width 4 @30 Hz)
        return a, s['ori']

    analyse('(a) synthetic IMU from the reference pose', body, seqs[:8], ctx, synthetic)
    analyse('(b) REAL DIP-IMU (low-passed)', body, seqs, ctx, real)


if __name__ == '__main__':
    main()

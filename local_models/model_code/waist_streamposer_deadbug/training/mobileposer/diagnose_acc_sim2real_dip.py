"""Is the synthetic acceleration (what we train on) the same quantity as the REAL DIP-IMU acceleration?
Per limb sensor: synthesise the acceleration from the REFERENCE pose with our surface-attachment IMU model
(exactly the training synthesis) and compare with the measured DIP-IMU acceleration (same sensor, world frame,
30 Hz).  Reported on the RELATIVE acceleration (sensor - pelvis sensor; gravity cancels, so a gravity convention
mismatch in the DIP files does not matter) and on the raw acceleration:
  R2   = 1 - |a_meas - a_syn|^2 / |a_meas|^2   (1 = perfect, <=0 = no better than zero)
  corr = Pearson correlation of the stacked xyz components
  scale= least-squares gain a_meas ~ scale * a_syn
If R2 is low even with the reference pose, the acceleration physics (and any model trained on synthetic
acceleration) is limited by the sim-to-real gap, not by pose errors.

Run from SparseIMU/:  python -m mobileposer.diagnose_acc_sim2real_dip
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
from mobileposer.no_head_layouts import LAYOUTS, synthesis_metadata, _synthesize
from mobileposer.surface_imu import DEFAULT_CONFIG
from mobileposer.evaluate_dip_real import load_dip

NAMES = ['L forearm', 'R forearm', 'L shank', 'R shank', 'pelvis/waist']


def stats(meas, syn):
    r2 = 1 - ((meas - syn) ** 2).sum() / (meas ** 2).sum()
    corr = np.corrcoef(meas.reshape(-1), syn.reshape(-1))[0, 1]
    scale = (meas * syn).sum() / (syn * syn).sum()
    return float(r2), float(corr), float(scale)


def main():
    body = ParametricModel(paths.smpl_file)
    syn = synthesis_metadata('surface', DEFAULT_CONFIG)
    meas_rel, syn_rel, meas_raw, syn_raw = [[] for _ in range(4)]
    for s in load_dip(body):
        T = len(s['pose'])
        ori_s, _, acc_s = _synthesize(body, s['pose'], torch.ones(10), torch.zeros(T, 3), 'wrists_shanks_waist', syn, 128)
        sl = slice(10, T - 10)                                  # the synthesis zero-pads / smooths the ends
        a_m, a_s = s['acc'][sl].numpy(), acc_s[sl].numpy()
        meas_raw.append(a_m); syn_raw.append(a_s)
        meas_rel.append(a_m[:, :4] - a_m[:, 4:5]); syn_rel.append(a_s[:, :4] - a_s[:, 4:5])
    meas_raw, syn_raw = np.concatenate(meas_raw), np.concatenate(syn_raw)
    meas_rel, syn_rel = np.concatenate(meas_rel), np.concatenate(syn_rel)
    print(f'{len(meas_raw)} frames; mean |a_meas| per sensor (m/s^2):', np.round(np.linalg.norm(meas_raw, axis=-1).mean(0), 2),
          ' mean |a_syn|:', np.round(np.linalg.norm(syn_raw, axis=-1).mean(0), 2))
    print('\nRELATIVE acceleration (sensor - pelvis):  sensor        R2    corr   scale')
    for k in range(4):
        r2, c, sc = stats(meas_rel[:, k], syn_rel[:, k])
        print(f'                                           {NAMES[k]:12s}{r2:6.2f}{c:7.2f}{sc:7.2f}')
    print('\nRAW acceleration (as given by the file vs synthesis; gravity convention matters here):')
    for k in range(5):
        r2, c, sc = stats(meas_raw[:, k], syn_raw[:, k])
        print(f'                                           {NAMES[k]:12s}{r2:6.2f}{c:7.2f}{sc:7.2f}')
    mean_meas = meas_raw.mean((0)); print('\nmean measured raw acc per sensor (xyz), gravity left in the data would show as ~9.8 on one axis:\n', np.round(mean_meas, 2))


if __name__ == '__main__':
    main()

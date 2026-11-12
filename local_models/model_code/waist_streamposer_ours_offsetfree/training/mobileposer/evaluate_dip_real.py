"""Standard held-out TEST SET with REAL IMUs: DIP-IMU test split (subjects s_09, s_10), never used in training.

Our layout wrists_shanks_waist maps to the DIP-IMU sensors [7, 8, 11, 12, 2] = left/right lower arm,
left/right lower leg, pelvis (the 6-IMU set of TransPose/PIP/PNP minus the head).  MobilePoser's own DIP
preprocessing uses [7, 8, 9, 10, 0, 2] (upper legs, head) and cannot be reused.
Caveat: the DIP pelvis sensor is rigid with SMPL joint 0 while our synthetic 'waist' sits on spine1 (joint 3).

Calibration: DIP orientations carry per-sensor constant bone offsets; as in evaluate_causal_captures.py an
ORACLE per-sequence constant (fit against the reference global rotations) is removed ('oracle-calib'),
the as-given inputs are also reported.  The fit residual is printed so that a global-frame mismatch is visible.

UNALIGNED global orientation error per joint group (deg), frame-weighted over the test split, every 5th frame.

Run from SparseIMU/:  python -m mobileposer.evaluate_dip_real
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent))
sys.path.insert(0, str(_DIR.parent.parent))

from mobileposer.config import paths
from mobileposer.articulate import math
from mobileposer.articulate.model import ParametricModel
from mobileposer.evaluate_canon_models import GROUPS, Predictors
from mobileposer.evaluate_v2_models import ModulePredictor
from mobileposer.evaluate_causal_holds import load
from mobileposer.diagnose_realtime_gap import window_predict, group_err_frames
from drift_eval_common import angle_between_rotmats

IMU_MASK = [7, 8, 11, 12, 0, 2] if os.environ.get('MOBILEPOSER_NSENS') == '6' else [7, 8, 11, 12, 2]   # lw, rw, ls, rs, (head,) pelvis
BONE = [18, 19, 4, 5, 3]                # SMPL joints the sensors are calibrated to (waist -> spine1)
SUBJECTS = ['s_09', 's_10']


def triangle_smooth(acc, n):
    """Match the bandwidth of the training synthesis (second difference with spacing n_syn frames @30 Hz == a
    triangular kernel of half-width n_syn on the acceleration).  acc [T,S,3] at the ORIGINAL rate; n in samples."""
    if n <= 0:
        return acc
    k = torch.tensor([(n - abs(i)) / n ** 2 for i in range(-n, n + 1)], dtype=acc.dtype)
    x = acc.permute(1, 2, 0).reshape(-1, 1, acc.shape[0])
    x = torch.nn.functional.pad(x, (n, n), mode='replicate')
    return torch.nn.functional.conv1d(x, k.view(1, 1, -1)).reshape(acc.shape[1], acc.shape[2], -1).permute(2, 0, 1).contiguous()


def load_dip(body, acc_filter=0, subjects=None):
    """acc_filter: half-width (in 60 Hz samples) of a triangular low-pass on the real acceleration (0 = as given)."""
    seqs = []
    for sub in (subjects or SUBJECTS):
        for name in sorted(os.listdir(paths.raw_dip / sub)):
            d = pickle.load(open(paths.raw_dip / sub / name, 'rb'), encoding='latin1')
            acc = torch.from_numpy(d['imu_acc'][:, IMU_MASK]).float()
            ori = torch.from_numpy(d['imu_ori'][:, IMU_MASK]).float()
            pose = torch.from_numpy(d['gt']).float()
            for _ in range(4):                      # fill nan with nearest neighbours (as process_dipimu)
                for x in (acc, ori):
                    x[1:].masked_scatter_(torch.isnan(x[1:]), x[:-1][torch.isnan(x[1:])])
                    x[:-1].masked_scatter_(torch.isnan(x[:-1]), x[1:][torch.isnan(x[:-1])])
            acc = triangle_smooth(acc, acc_filter)
            st = 60 // int(os.environ.get('MOBILEPOSER_FPS', 30))
            acc, ori, pose = acc[6:-6:st].contiguous(), ori[6:-6:st].contiguous(), pose[6:-6:st].contiguous()
            if torch.isnan(acc).any() or torch.isnan(ori).any() or torch.isnan(pose).any() or len(pose) < 60:
                continue
            seqs.append({'name': f'{sub}/{name}', 'acc': acc, 'ori': ori,
                         'pose': math.axis_angle_to_rotation_matrix(pose).reshape(-1, 24, 3, 3)})
    return seqs


def calibrate(ori, G):
    out, resid = ori.clone(), []
    for k, j in enumerate(BONE):
        M = (ori[:, k].transpose(-1, -2) @ G[:, j]).mean(0)
        U, S, Vt = torch.linalg.svd(M)
        if torch.det(U @ Vt) < 0:
            U[:, -1] *= -1
        out[:, k] = ori[:, k] @ (U @ Vt)
        resid.append(angle_between_rotmats(out[:, k], G[:, j]).mean().item())
    return out, resid


def main():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    body = ParametricModel(paths.smpl_file)
    P0 = Predictors(dev)
    m3 = load('checkpoints/v3_canon_poser', 'checkpoints/v3_canon_joints', False, dev)
    models = {'v1_base (orig, 32h)': P0.baseline, 'v1_canon': P0.canon,
              'v2_base (48h)': ModulePredictor('checkpoints/v2_base_poser', 'checkpoints/v2_base_joints', False, dev),
              'v2_canon (48h)': ModulePredictor('checkpoints/v2_canon_poser', 'checkpoints/v2_canon_joints', True, dev),
              'v3_canon offline': m3,
              'v3_canon 46-window': lambda a, o: None,
              'causal125 streamed': load('checkpoints/v3_causal125_poser', 'checkpoints/v3_causal125_joints', True, dev)}
    seqs = load_dip(body)
    print(f'{len(seqs)} DIP test sequences, {sum(len(s["pose"]) for s in seqs)} frames @30 Hz', flush=True)
    res = {(m, c): [] for m in models for c in ('oracle-calib', 'as-given')}
    weights, resids = [], []
    for s in seqs:
        T = len(s['pose'])
        G, _ = body.forward_kinematics(s['pose'], torch.ones(10), tran=None, calc_mesh=False)
        G = G.reshape(T, 24, 3, 3)
        ori_cal, resid = calibrate(s['ori'], G)
        resids.append(resid)
        frames = list(range(0, T, 5))
        weights.append(len(frames))
        for cal, ori in (('oracle-calib', ori_cal), ('as-given', s['ori'])):
            for m, fn in models.items():
                pred = window_predict(m3, s['acc'], ori, frames) if m == 'v3_canon 46-window' else fn(s['acc'], ori)[:T][frames]
                res[(m, cal)].append(group_err_frames(body, pred, s['pose'][frames]))
        print('  done', s['name'], 'calibration residual (deg) lw rw ls rs pelvis:', np.round(resid, 1), flush=True)
    w = np.array(weights, float)
    print('\nmean calibration residual per sensor (lw, rw, ls, rs, pelvis->spine1):', np.round(np.average(resids, axis=0, weights=w), 1))
    hdr = '  '.join(n[:17].rjust(17) for n in GROUPS)
    for cal in ('oracle-calib', 'as-given'):
        print(f'\n=== DIP-IMU test split, real IMU, {cal} ===')
        print('model'.ljust(24) + hdr)
        for m in models:
            print(m.ljust(24) + '  '.join(f'{x:17.1f}' for x in np.average(res[(m, cal)], axis=0, weights=w)))


if __name__ == '__main__':
    main()

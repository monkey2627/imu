"""Why is REAL-TIME use much worse than our offline numbers?  Two differences were never measured:
  (1) calibration: our reported real-IMU numbers use ORACLE per-sensor constant calibration (computed
      from the reference); real use cannot do that.  Here: oracle vs as-given (inputs/ours1.pt, which
      only has the sensor-only constant-yaw fit).
  (2) online inference: real-time runs a short sliding window (MobilePoser forward_online: 40 past + 5
      future frames, bidirectional LSTM over the 46-frame window, output of frame t read at index 40),
      not the whole sequence with full context.
Evaluate on the 10 real captures (UNALIGNED global orientation error per group, deg), on every 5th frame
(same frames for all conditions), for v2_base and v2_canon.

Run from SparseIMU/:  python -m mobileposer.diagnose_realtime_gap
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
from mobileposer.evaluate_canon_models import GROUPS
from mobileposer.evaluate_v2_models import ModulePredictor
from drift_eval_common import angle_between_rotmats

PAST, FUT = 40, 5


def window_predict(fn, acc, ori, frames):
    """pose at each requested frame from a 46-frame window (past 40, future 5), like forward_online"""
    T = len(acc)
    out = []
    for t in frames:
        a, b = max(0, t - PAST), min(T, t + FUT + 1)
        pose = fn(acc[a:b], ori[a:b])
        out.append(pose[t - a])
    return torch.stack(out)


def group_err_frames(body, pred, gt_pose):
    """pred, gt_pose: [N,24,3,3] local rotations of the selected frames (independent frames)"""
    gp, _ = body.forward_kinematics(pred, torch.ones(10), tran=None, calc_mesh=False)
    gq, _ = body.forward_kinematics(gt_pose, torch.ones(10), tran=None, calc_mesh=False)
    e = angle_between_rotmats(gp.reshape(-1, 24, 3, 3), gq.reshape(-1, 24, 3, 3)).mean(0).numpy()
    return [float(e[js].mean()) for js in GROUPS.values()]


def main():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    body = ParametricModel(paths.smpl_file)
    models = {'v2_base': ModulePredictor('checkpoints/v2_base_poser', 'checkpoints/v2_base_joints', False, dev),
              'v2_canon': ModulePredictor('checkpoints/v2_canon_poser', 'checkpoints/v2_canon_joints', True, dev)}
    conds = ['offline+oracle-calib', 'offline+as-given', 'online window+oracle-calib', 'online window+as-given']
    res = {(m, c): [] for m in models for c in conds}
    for cap in sorted(Path('results/ours_multi_finetune/captures').iterdir(), key=lambda p: int(p.name.split('_')[0])):
        t = torch.load(cap / 'targets.pt', map_location='cpu', weights_only=False)
        gt = t['pose'].reshape(-1, 24, 3, 3)
        d = torch.load(cap / 'inputs/wrists_shanks_waist/ours1.pt', map_location='cpu', weights_only=False)
        acc = d['acc'][0]
        ori_raw = d['ori'][0].clone()
        ori_cal = ori_raw.clone()
        Tc = min(len(ori_raw), len(t['global_pose']))
        for k, j in enumerate(d['layout_joints']):
            M = (ori_raw[:Tc, k].transpose(-1, -2) @ t['global_pose'][:Tc, j]).mean(0)
            U, S, Vt = torch.linalg.svd(M)
            if torch.det(U @ Vt) < 0:
                U[:, -1] *= -1
            ori_cal[:, k] = ori_raw[:, k] @ (U @ Vt)
        T = min(len(acc), len(gt))
        frames = list(range(0, T, 5))
        gsel = gt[frames]
        for m, fn in models.items():
            for cal_name, ori in (('oracle-calib', ori_cal), ('as-given', ori_raw)):
                off = fn(acc[:T], ori[:T])[:T][frames]
                res[(m, f'offline+{cal_name}')].append(group_err_frames(body, off, gsel))
                on = window_predict(fn, acc[:T], ori[:T], frames)
                res[(m, f'online window+{cal_name}')].append(group_err_frames(body, on, gsel))
        print('  done', cap.name.split('__')[0], flush=True)
    hdr = '  '.join(n[:17].rjust(17) for n in GROUPS)
    print('\ncondition / model'.ljust(46) + hdr)
    for m in models:
        for c in conds:
            print(f'{c} / {m}'.ljust(46) + '  '.join(f'{x:17.1f}' for x in np.mean(res[(m, c)], axis=0)))


if __name__ == '__main__':
    main()

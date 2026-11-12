"""Real-time (past-only) long memory vs the current real-time window on the 10 REAL captures.

Deployed baseline  : v3_canon (bidirectional), 46-frame window (40 past + 5 future), as forward_online.
Causal long memory : v3_causal125 / v3_causal1000, streamed from the capture start, past frames only.
Also the offline bidirectional upper reference.  Real IMU with oracle per-sensor constant calibration and
as-given inputs; UNALIGNED global orientation error per joint group (deg), every 5th frame.

Run from SparseIMU/:  python -m mobileposer.evaluate_causal_captures
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
from mobileposer.evaluate_causal_holds import load
from mobileposer.diagnose_realtime_gap import window_predict, group_err_frames


def main():
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    body = ParametricModel(paths.smpl_file)
    bidir = load('checkpoints/v3_canon_poser', 'checkpoints/v3_canon_joints', False, dev)
    c125 = load('checkpoints/v3_causal125_poser', 'checkpoints/v3_causal125_joints', True, dev)
    c1000 = load('checkpoints/v3_causal1000_poser', 'checkpoints/v3_causal1000_joints', True, dev)
    conds = ['bidir 46-window (deployed)', 'bidir offline (non-causal)', 'causal125 streamed', 'causal1000 streamed']
    res = {(c, cal): [] for c in conds for cal in ('oracle-calib', 'as-given')}
    thigh = {}
    for cap in sorted(Path('results/ours_multi_finetune/captures').iterdir(), key=lambda p: int(p.name.split('_')[0])):
        t = torch.load(cap / 'targets.pt', map_location='cpu', weights_only=False)
        gt = t['pose'].reshape(-1, 24, 3, 3)
        d = torch.load(cap / 'inputs/wrists_shanks_waist/ours1.pt', map_location='cpu', weights_only=False)
        acc, ori_raw = d['acc'][0], d['ori'][0].clone()
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
        ex = cap.name.split('__')[0]
        for cal, ori in (('oracle-calib', ori_cal), ('as-given', ori_raw)):
            outs = {conds[0]: window_predict(bidir, acc[:T], ori[:T], frames),
                    conds[1]: bidir(acc[:T], ori[:T])[:T][frames],
                    conds[2]: c125(acc[:T], ori[:T])[:T][frames],
                    conds[3]: c1000(acc[:T], ori[:T])[:T][frames]}
            for c, pred in outs.items():
                e = group_err_frames(body, pred, gsel)
                res[(c, cal)].append(e)
                thigh[(ex, c, cal)] = e[2]
        print('  done', ex, flush=True)
    hdr = '  '.join(n[:17].rjust(17) for n in GROUPS)
    for cal in ('oracle-calib', 'as-given'):
        print(f'\n=== real IMU, {cal}: mean over 10 captures ===')
        print('model'.ljust(30) + hdr)
        for c in conds:
            print(c.ljust(30) + '  '.join(f'{x:17.1f}' for x in np.mean(res[(c, cal)], axis=0)))
    print('\nper exercise thigh[1,2] error (oracle-calib)')
    print('  exercise'.ljust(20) + '  '.join(c[:14].rjust(14) for c in conds))
    for ex in sorted({k[0] for k in thigh}, key=lambda s: int(s.split('_')[0])):
        print(f'  {ex:16s}' + '  '.join(f'{thigh[(ex, c, "oracle-calib")]:14.1f}' for c in conds))


if __name__ == '__main__':
    main()

"""Compare pose models trained on the 32 h AMASS set (v1) and on the 48 h set (v2: +HDM05,
HUMAN4D, HumanEva, KIT, MoSh, PosePrior, SFU, SSM, TCDHands), each with world-frame inputs
('base') or waist-frame canonical inputs ('canon').  wrists_shanks_waist, Poser+Joints only.

  v1_base  : the original full checkpoint (checkpoints/no_head_5imu_surface/.../base_model.pth)
  v1_canon : checkpoints/canon_poser|canon_joints
  v2_base  : checkpoints/v2_base_poser|v2_base_joints
  v2_canon : checkpoints/v2_canon_poser|v2_canon_joints

Tests (UNALIGNED global orientation error per joint group, deg): rotation test on 12 upright walking
sequences (seen by all models) and the 10 real captures (never seen; IMU synthesized from the reference,
and real IMU with ORACLE per-sensor constant calibration).

Run from SparseIMU/:  python -m mobileposer.evaluate_v2_models
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent))
sys.path.insert(0, str(_DIR.parent.parent))

from mobileposer.config import paths
from mobileposer.articulate.model import ParametricModel
from mobileposer.models.poser import Poser
from mobileposer.models.joints import Joints
from mobileposer.evaluate_drift_anchorpose import prepare_imu, LAYOUT_NAME
from mobileposer.evaluate_canon_models import Predictors, best_ckpt, rot, group_err, GROUPS
from mobileposer.no_head_layouts import _synthesize, synthesis_metadata
from mobileposer.surface_imu import DEFAULT_CONFIG
from drift_eval_common import load_long_sequences


class ModulePredictor:
    def __init__(self, poser_root, joints_root, canon, device):
        self.dev, self.canon = device, canon
        self.poser = Poser.load_from_checkpoint(best_ckpt(poser_root), map_location=device).to(device).eval()
        self.joints = Joints.load_from_checkpoint(best_ckpt(joints_root), map_location=device).to(device).eval()

    @torch.no_grad()
    def __call__(self, acc, ori):
        if self.canon:
            Rw = ori[:, 4]
            RwT = Rw.transpose(-1, -2)
            acc = (RwT.unsqueeze(1) @ acc[:, :5].unsqueeze(-1)).squeeze(-1)
            ori = RwT.unsqueeze(1) @ ori[:, :5]
        imu = prepare_imu(acc, ori).to(self.dev).unsqueeze(0)
        T = imu.shape[1]
        pj = self.joints(imu, [T])
        pose = self.poser._reduced_global_to_full(self.poser(torch.cat((pj, imu), -1), [T])).cpu().reshape(-1, 24, 3, 3)
        if self.canon:
            pose[:, 0] = Rw[:len(pose)] @ pose[:, 0]
        return pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    dev = torch.device(args.device)
    body = ParametricModel(paths.smpl_file)
    syn = synthesis_metadata('surface', DEFAULT_CONFIG)
    P0 = Predictors(dev)
    models = {'v1_base': P0.baseline, 'v1_canon': P0.canon,
              'v2_base': ModulePredictor('checkpoints/v2_base_poser', 'checkpoints/v2_base_joints', False, dev),
              'v2_canon': ModulePredictor('checkpoints/v2_canon_poser', 'checkpoints/v2_canon_joints', True, dev)}
    names = list(GROUPS)
    hdr = '  '.join(n[:17].rjust(17) for n in names)

    print('=== A. rotation test (12 upright walking seqs, rotated about a horizontal axis; seen by all models) ===')
    print('condition / model'.ljust(28) + hdr)
    seqs = load_long_sequences(min_frames=1800, max_seqs=12)
    for cname, R in (('0 (as is)', None), ('pitch(x) 90', rot('x', 90))):
        acc_e = {m: [] for m in models}
        for s in seqs:
            p = s['pose'].clone()[:900].reshape(-1, 24, 3, 3); tr = s['tran'].clone()[:900]
            if R is not None:
                p[:, 0] = R @ p[:, 0]; tr = (R @ tr.T).T
            ori, _, acc = _synthesize(body, p, torch.ones(10), tr, LAYOUT_NAME, syn, 128)
            for m, fn in models.items():
                acc_e[m].append(group_err(body, fn(acc, ori), p))
        for m in models:
            print(f'{cname} / {m}'.ljust(28) + '  '.join(f'{x:17.1f}' for x in np.mean(acc_e[m], axis=0)))

    print('\n=== B. 10 real captures (unseen), UNALIGNED global error deg, mean over captures ===')
    root = Path('results/ours_multi_finetune/captures')
    rows = {(src, m): [] for src in ('synthetic-from-reference IMU', 'real IMU, oracle-calibrated') for m in models}
    per_ex = {}
    for cap in sorted(root.iterdir(), key=lambda p: int(p.name.split('_')[0])):
        t = torch.load(cap / 'targets.pt', map_location='cpu', weights_only=False)
        gt = t['pose'].reshape(-1, 24, 3, 3); Tg = len(gt)
        d = torch.load(cap / 'inputs/wrists_shanks_waist/ours1.pt', map_location='cpu', weights_only=False)
        ori_s, _, acc_s = _synthesize(body, gt, torch.ones(10), torch.zeros(Tg, 3), LAYOUT_NAME, syn, 128)
        ori_r = d['ori'][0].clone(); acc_r = d['acc'][0]
        Tc = min(len(ori_r), len(t['global_pose']))
        for k, j in enumerate(d['layout_joints']):
            M = (ori_r[:Tc, k].transpose(-1, -2) @ t['global_pose'][:Tc, j]).mean(0)
            U, S, Vt = torch.linalg.svd(M)
            if torch.det(U @ Vt) < 0:
                U[:, -1] *= -1
            ori_r[:, k] = ori_r[:, k] @ (U @ Vt)
        ex = cap.name.split('__')[0]
        for src, (acc, ori) in (('synthetic-from-reference IMU', (acc_s, ori_s)), ('real IMU, oracle-calibrated', (acc_r, ori_r))):
            for m, fn in models.items():
                e = group_err(body, fn(acc, ori), gt)
                rows[(src, m)].append(e)
                per_ex[(ex, src, m)] = e
    print('source / model'.ljust(46) + hdr)
    for (src, m), v in rows.items():
        print(f'{src} / {m}'.ljust(46) + '  '.join(f'{x:17.1f}' for x in np.mean(v, axis=0)))
    print('\nper exercise (real IMU, oracle-calibrated): thigh[1,2] error per model')
    print('  exercise'.ljust(20) + '  '.join(m.rjust(9) for m in models))
    for ex in sorted({k[0] for k in per_ex}, key=lambda s: int(s.split('_')[0])):
        print(f'  {ex:16s}' + '  '.join(f'{per_ex[(ex, "real IMU, oracle-calibrated", m)][2]:9.1f}' for m in models))
    print('\nper exercise (real IMU, oracle-calibrated): pelvis[0] error per model')
    for ex in sorted({k[0] for k in per_ex}, key=lambda s: int(s.split('_')[0])):
        print(f'  {ex:16s}' + '  '.join(f'{per_ex[(ex, "real IMU, oracle-calibrated", m)][0]:9.1f}' for m in models))


if __name__ == '__main__':
    main()

"""Baseline (world-frame inputs) vs WAIST-FRAME-canonicalized Poser+Joints (wrists_shanks_waist).

Canonical model: all sensors' acc/ori rotated by R_waist^T, outputs in that frame; the
root's world orientation is recovered as R_waist @ predicted canonical root. Only the pose
modules are retrained (Poser, Joints); translation/contact are not evaluated here.

Tests (UNALIGNED global orientation error per joint group, deg):
  A. rotation test: held-out upright walking sequences, whole body rotated about a horizontal axis
  B. 10 real captures (evaluation only): IMU synthesized from the reference pose; and real IMU with
     ORACLE per-sensor constant calibration (diagnostic upper bound, uses the reference)

Run from SparseIMU/:  python -m mobileposer.evaluate_canon_models
"""
import argparse
import glob
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR.parent))
sys.path.insert(0, str(_DIR.parent.parent))

from mobileposer.config import paths
from mobileposer.utils.model_utils import load_model
from mobileposer.articulate.model import ParametricModel
from mobileposer.models.poser import Poser
from mobileposer.models.joints import Joints
from mobileposer.evaluate_drift_anchorpose import prepare_imu, LAYOUT_NAME
from mobileposer.no_head_layouts import _synthesize, synthesis_metadata
from mobileposer.surface_imu import DEFAULT_CONFIG
from drift_eval_common import load_long_sequences, angle_between_rotmats

GROUPS = {'pelvis[0]': [0], 'SENSED[4,5,18,19]': [4, 5, 18, 19], 'thigh[1,2]': [1, 2], 'spine2/3[6,9]': [6, 9],
          'neck/head[12-15]': [12, 13, 14, 15], 'upperarm[16,17]': [16, 17]}


def best_ckpt(root):
    files = glob.glob(f'{root}/*/*/epoch=*.ckpt') + glob.glob(f'{root}/*/epoch=*.ckpt')
    def loss(f):
        m = re.search(r'validation_step_loss=([0-9.]+)', f)
        return float(m.group(1).rstrip('.')) if m else 1e9
    if all(loss(f) >= 1e9 for f in files):                    # validation loss was NaN (padded rows in the transformer eval fast path): take the last epoch
        return max(files, key=lambda f: int(re.search(r"epoch=([0-9]+)", f).group(1)))
    return min(files, key=loss)


def rot(axis, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    if axis == 'x': return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)
    if axis == 'y': return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)


class Predictors:
    def __init__(self, device):
        self.dev = device
        self.base = load_model(str(paths.root_dir / 'checkpoints/no_head_5imu_surface/wrists_shanks_waist/1/base_model.pth')).to(device).eval()
        self.poser = Poser.load_from_checkpoint(best_ckpt('checkpoints/canon_poser'), map_location=device).to(device).eval()
        self.joints = Joints.load_from_checkpoint(best_ckpt('checkpoints/canon_joints'), map_location=device).to(device).eval()

    @torch.no_grad()
    def baseline(self, acc, ori):
        imu = prepare_imu(acc, ori).to(self.dev)
        self.base.reset()
        pose, _, _, _ = self.base.forward_offline(imu.unsqueeze(0), [imu.shape[0]])
        return (pose.squeeze(0) if pose.dim() == 4 else pose).cpu().reshape(-1, 24, 3, 3)

    @torch.no_grad()
    def canon(self, acc, ori):
        Rw = ori[:, 4]
        RwT = Rw.transpose(-1, -2)
        acc_c = (RwT.unsqueeze(1) @ acc[:, :5].unsqueeze(-1)).squeeze(-1)
        ori_c = RwT.unsqueeze(1) @ ori[:, :5]
        imu = prepare_imu(acc_c, ori_c).to(self.dev).unsqueeze(0)
        T = imu.shape[1]
        pj = self.joints(imu, [T])
        pose = self.poser._reduced_global_to_full(self.poser(torch.cat((pj, imu), -1), [T])).cpu().reshape(-1, 24, 3, 3)
        pose[:, 0] = Rw[:len(pose)] @ pose[:, 0]
        return pose


def group_err(body, pred, gt):
    T = min(len(pred), len(gt))
    gp, _ = body.forward_kinematics(pred[:T], torch.ones(10), tran=None, calc_mesh=False)
    gq, _ = body.forward_kinematics(gt[:T], torch.ones(10), tran=None, calc_mesh=False)
    e = angle_between_rotmats(gp.reshape(T, 24, 3, 3), gq.reshape(T, 24, 3, 3)).mean(0).numpy()
    return [float(e[js].mean()) for js in GROUPS.values()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    dev = torch.device(args.device)
    body = ParametricModel(paths.smpl_file)
    syn = synthesis_metadata('surface', DEFAULT_CONFIG)
    P = Predictors(dev)
    names = list(GROUPS)
    hdr = '  '.join(n[:17].rjust(17) for n in names)

    # ---- A. rotation test
    seqs = load_long_sequences(min_frames=1800, max_seqs=12)
    conds = [('0 (as is)', None), ('yaw(y) 90', rot('y', 90)), ('pitch(x) 45', rot('x', 45)), ('pitch(x) 90', rot('x', 90)),
             ('roll(z) 90', rot('z', 90)), ('pitch(x) 180', rot('x', 180))]
    print('=== A. rotation test (upright walking, whole body rotated), UNALIGNED global error deg ===')
    print('condition / model'.ljust(30) + hdr)
    for cname, R in conds:
        acc_e = {'baseline': [], 'canon': []}
        for s in seqs:
            p = s['pose'].clone()[:900].reshape(-1, 24, 3, 3); tr = s['tran'].clone()[:900]
            if R is not None:
                p[:, 0] = R @ p[:, 0]; tr = (R @ tr.T).T
            ori, _, acc = _synthesize(body, p, torch.ones(10), tr, LAYOUT_NAME, syn, 128)
            acc_e['baseline'].append(group_err(body, P.baseline(acc, ori), p))
            acc_e['canon'].append(group_err(body, P.canon(acc, ori), p))
        for m in ('baseline', 'canon'):
            print(f'{cname} / {m}'.ljust(30) + '  '.join(f'{x:17.1f}' for x in np.mean(acc_e[m], axis=0)))

    # ---- B. real captures
    print('\n=== B. 10 real captures (reference = HumanPose pseudo-labels), UNALIGNED global error deg; mean over captures ===')
    root = Path('results/ours_multi_finetune/captures')
    rows = {(src, m): [] for src in ('synthetic-from-reference IMU', 'real IMU, oracle-calibrated') for m in ('baseline', 'canon')}
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
            for m, fn in (('baseline', P.baseline), ('canon', P.canon)):
                e = group_err(body, fn(acc, ori), gt)
                rows[(src, m)].append(e)
                per_ex[(ex, src, m)] = e
    print('source / model'.ljust(44) + hdr)
    for (src, m), v in rows.items():
        print(f'{src} / {m}'.ljust(44) + '  '.join(f'{x:17.1f}' for x in np.mean(v, axis=0)))
    print('\nper exercise (synthetic-from-reference IMU): thigh[1,2] / pelvis[0]   baseline -> canon')
    for ex in sorted({k[0] for k in per_ex}, key=lambda s: int(s.split('_')[0])):
        b = per_ex[(ex, 'synthetic-from-reference IMU', 'baseline')]
        c = per_ex[(ex, 'synthetic-from-reference IMU', 'canon')]
        print(f'  {ex:14s} thigh {b[2]:5.1f} -> {c[2]:5.1f}   pelvis {b[0]:5.1f} -> {c[0]:5.1f}   upperarm {b[5]:5.1f} -> {c[5]:5.1f}')


if __name__ == '__main__':
    main()

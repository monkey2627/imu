"""Evaluation-only check of the BASE (not finetuned) layout models on the 10 real
self-collected captures (data/raw/ours/<n>_<exercise>/1), using the already
extracted inputs and references in results/ours_multi_finetune/captures/<id>/:
  inputs/<layout>/ours1.pt : physical IMU only (acc, ori), reference not used as input
  targets.pt               : HumanPose-retargeted SMPL pseudo-labels (SlimeVR avatar
                             solved from ALL trackers -> NOT mocap ground truth)
Error = ROOT-ALIGNED GLOBAL joint-angle error. Joints split per layout into SENSED
(bone carrying an IMU: layout_joints + root) and UNSENSED (the rest of the reduced set).
These captures are used for evaluation ONLY (never for training / method design).

Run from SparseIMU/:  python -m mobileposer.evaluate_ours_captures
"""
import argparse
import json
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
from mobileposer.evaluate_drift_anchorpose import prepare_imu
from mobileposer.no_head_layouts import _synthesize, synthesis_metadata
from mobileposer.surface_imu import DEFAULT_CONFIG
from drift_eval_common import angle_between_rotmats

REDUCED = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
GROUPS = {'thigh/hip[1,2]': [1, 2], 'knee[4,5]': [4, 5], 'spine[3,6,9]': [3, 6, 9], 'neck/head/clav[12-15]': [12, 13, 14, 15],
          'upperarm/shoulder[16,17]': [16, 17], 'forearm/elbow[18,19]': [18, 19]}
NAME = {'1_卧位猫': 'cat-cow(quadruped)', '2_坐位': 'sitting', '3_站位': 'standing', '4_臀桥': 'glute bridge(supine)',
        '5_侧桥': 'side bridge', '6_卷腹': 'crunch(supine)', '7_死虫': 'dead bug(supine)', '8_鸟狗': 'bird-dog(quadruped)',
        '9_屈膝': 'knee bend', '10_站立位伸展': 'standing stretch'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captures', default='results/ours_multi_finetune/captures')
    ap.add_argument('--ckpt-root', default='checkpoints/no_head_5imu_surface')
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--oracle-calibration', action='store_true',
                    help='remove ONE constant sensor-to-bone offset per sensor per capture, estimated from the reference '
                         '(uses the reference => diagnostic upper bound, not a deployable calibration)')
    ap.add_argument('--synthetic-from-reference', action='store_true',
                    help='replace the real IMU by IMU SYNTHESIZED from the reference pose (zero translation): separates real-IMU problems from posture generalization')
    args = ap.parse_args()
    device = torch.device(args.device)
    body = ParametricModel(paths.smpl_file)
    models = {}

    res = {}   # (exercise, layout) -> dict
    for cap in sorted(Path(args.captures).iterdir()):
        if not (cap / 'targets.pt').exists():
            continue
        ex = cap.name.split('__')[0]
        tgt = torch.load(cap / 'targets.pt', map_location='cpu', weights_only=False)
        gt = tgt['pose'].reshape(-1, 24, 3, 3)
        for lay_dir in sorted((cap / 'inputs').iterdir()):
            layout = lay_dir.name
            d = torch.load(lay_dir / 'ours1.pt', map_location='cpu', weights_only=False)
            if layout not in models:
                models[layout] = load_model(f'{args.ckpt_root}/{layout}/1/base_model.pth').to(device).eval()
            model = models[layout]
            ori_in = d['ori'][0].clone()
            if args.oracle_calibration:
                Tg = min(len(ori_in), len(tgt['global_pose']))
                for k, j in enumerate(d['layout_joints']):
                    R_in, R_ref = ori_in[:Tg, k], tgt['global_pose'][:Tg, j]
                    M = (R_in.transpose(-1, -2) @ R_ref).mean(0)
                    U, S, Vt = torch.linalg.svd(M)
                    if torch.det(U @ Vt) < 0:
                        U[:, -1] *= -1
                    ori_in[:, k] = ori_in[:, k] @ (U @ Vt)
            acc_in = d['acc'][0]
            if args.synthetic_from_reference:
                syn = synthesis_metadata('surface', DEFAULT_CONFIG)
                Tg = len(gt)
                with torch.no_grad():
                    ori_in, _, acc_in = _synthesize(body, gt, torch.ones(10), torch.zeros(Tg, 3), layout, syn, 128)
            imu = prepare_imu(acc_in, ori_in).to(device)
            with torch.no_grad():
                model.reset()
                pred, _, _, _ = model.forward_offline(imu.unsqueeze(0), [imu.shape[0]])
                pred = (pred.squeeze(0) if pred.dim() == 4 else pred).cpu().reshape(-1, 24, 3, 3)
                T = min(len(pred), len(gt))
                pp, gg = pred[:T].clone(), gt[:T].clone()
                pp[:, 0] = torch.eye(3)
                gg[:, 0] = torch.eye(3)
                gp, _ = body.forward_kinematics(pp, torch.ones(10), tran=None, calc_mesh=False)
                gq, _ = body.forward_kinematics(gg, torch.ones(10), tran=None, calc_mesh=False)
                e = angle_between_rotmats(gp.reshape(T, 24, 3, 3), gq.reshape(T, 24, 3, 3)).mean(0).numpy()   # [24]
            sensed = set(d['layout_joints'])
            uns = [j for j in REDUCED if j not in sensed]
            sen = [j for j in REDUCED if j in sensed]
            res[(ex, layout)] = {'sensed': float(e[sen].mean()), 'unsensed': float(e[uns].mean()),
                                 'per_joint': e, 'frames': T, 'sensed_joints': sorted(sensed)}
            print(f'  {ex} {layout}: sensed={res[(ex, layout)]["sensed"]:.1f} unsensed={res[(ex, layout)]["unsensed"]:.1f}', flush=True)

    layouts = sorted({k[1] for k in res})
    exercises = sorted({k[0] for k in res}, key=lambda s: int(s.split('_')[0]))
    print('\n=== wrists_shanks_waist: error (deg, root-aligned global) per exercise vs pseudo-label reference ===')
    print('exercise'.ljust(24) + 'frames  sensed  UNSENSED  ' + '  '.join(g[:14].rjust(14) for g in GROUPS))
    for ex in exercises:
        r = res.get((ex, 'wrists_shanks_waist'))
        if r is None:
            continue
        print((ex + ' ' + NAME.get(ex, ''))[:23].ljust(24) + f'{r["frames"]:6d}  {r["sensed"]:6.1f}  {r["unsensed"]:8.1f}  '
              + '  '.join(f'{r["per_joint"][js].mean():14.1f}' for js in GROUPS.values()))
    print('\n=== mean over exercises, per layout: sensed vs UNSENSED error ===')
    for lay in layouts:
        rr = [res[(ex, lay)] for ex in exercises if (ex, lay) in res]
        print(f'{lay:26s} n={len(rr):2d}  sensed={np.mean([r["sensed"] for r in rr]):5.1f}  UNSENSED={np.mean([r["unsensed"] for r in rr]):5.1f}')
    print('\nNOTE: reference = HumanPose-retargeted pseudo-labels, not mocap; 1 capture per exercise; evaluation only.')


if __name__ == '__main__':
    main()

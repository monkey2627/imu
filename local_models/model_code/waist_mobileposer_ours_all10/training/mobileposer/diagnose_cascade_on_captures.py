"""Why does the base model fail on lying / quadruped / side-bridge postures even for
the bones that carry an IMU?  (evaluate_ours_captures: sensed-bone error 27-50 deg,
identical with IMU synthesized from the reference pose itself.)

Hypothesis tested here: the Joints->Poser CASCADE. MobilePoser's Poser is fed the
Joints module's predicted joint positions; Poser was TRAINED with ground-truth positions
(+0.04 m Gaussian noise). In unusual postures the Joints module is wrong in a
non-Gaussian way and misleads Poser. We compare, on the same inputs (IMU synthesized
from the reference pose, so no real-IMU effects):
  (a) normal:   Poser input = Joints-module prediction
  (b) oracle:   Poser input = ground-truth joint positions (+ the training noise level 0.04)
  and report Joints-module position error. Error = root-aligned global, deg.

Run from SparseIMU/:  python -m mobileposer.diagnose_cascade_on_captures
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
from mobileposer.utils.model_utils import load_model
from mobileposer.articulate.model import ParametricModel
from mobileposer.evaluate_drift_anchorpose import prepare_imu
from mobileposer.no_head_layouts import _synthesize, synthesis_metadata
from mobileposer.surface_imu import DEFAULT_CONFIG
from drift_eval_common import angle_between_rotmats

REDUCED = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
SENSED = {4, 5, 18, 19, 3}   # wrists_shanks_waist


def root_aligned_err(body, pred, gt):
    T = min(len(pred), len(gt))
    pp, gg = pred[:T].clone(), gt[:T].clone()
    pp[:, 0] = torch.eye(3)
    gg[:, 0] = torch.eye(3)
    gp, _ = body.forward_kinematics(pp, torch.ones(10), tran=None, calc_mesh=False)
    gq, _ = body.forward_kinematics(gg, torch.ones(10), tran=None, calc_mesh=False)
    return angle_between_rotmats(gp.reshape(T, 24, 3, 3), gq.reshape(T, 24, 3, 3)).mean(0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captures', default='results/ours_multi_finetune/captures')
    ap.add_argument('--layout', default='wrists_shanks_waist')
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    device = torch.device(args.device)
    body = ParametricModel(paths.smpl_file)
    model = load_model(f'checkpoints/no_head_5imu_surface/{args.layout}/1/base_model.pth').to(device).eval()
    syn = synthesis_metadata('surface', DEFAULT_CONFIG)
    sen = [j for j in REDUCED if j in SENSED]
    uns = [j for j in REDUCED if j not in SENSED]

    print('exercise'.ljust(26) + 'normal(sensed/UNS)   oracle-joints->Poser(sensed/UNS)   Joints-module pos err (cm)')
    rows = []
    for cap in sorted(Path(args.captures).iterdir(), key=lambda p: int(p.name.split('_')[0])):
        if not (cap / 'targets.pt').exists():
            continue
        tgt = torch.load(cap / 'targets.pt', map_location='cpu', weights_only=False)
        gt = tgt['pose'].reshape(-1, 24, 3, 3)
        T = len(gt)
        with torch.no_grad():
            ori, _, acc = _synthesize(body, gt, torch.ones(10), torch.zeros(T, 3), args.layout, syn, 128)
            imu = prepare_imu(acc, ori).to(device).unsqueeze(0)
            _, gt_joint = body.forward_kinematics(gt, torch.ones(10), tran=None, calc_mesh=False)
            gt_joint = gt_joint.reshape(T, 24, 3)
            model.reset()
            pred_joints = model.joints(imu, [T])                             # [1,T,72]
            pose_a = model._reduced_global_to_full(model.pose(torch.cat((pred_joints, imu), -1), [T]))
            noisy = gt_joint.reshape(1, T, -1).to(device) + 0.04 * torch.randn(1, T, 72, device=device)
            pose_b = model._reduced_global_to_full(model.pose(torch.cat((noisy, imu), -1), [T]))
            ea = root_aligned_err(body, pose_a.cpu().reshape(-1, 24, 3, 3), gt)
            eb = root_aligned_err(body, pose_b.cpu().reshape(-1, 24, 3, 3), gt)
            jerr = (pred_joints.cpu().reshape(T, 24, 3) - gt_joint).norm(dim=-1).mean().item() * 100
        rows.append((ea[sen].mean(), ea[uns].mean(), eb[sen].mean(), eb[uns].mean(), jerr))
        print(cap.name.split('__')[0].ljust(26) + f'{ea[sen].mean():6.1f} / {ea[uns].mean():6.1f}          '
              f'{eb[sen].mean():6.1f} / {eb[uns].mean():6.1f}                   {jerr:6.1f}', flush=True)
    r = np.array(rows)
    print('mean'.ljust(26) + f'{r[:,0].mean():6.1f} / {r[:,1].mean():6.1f}          {r[:,2].mean():6.1f} / {r[:,3].mean():6.1f}                   {r[:,4].mean():6.1f}')


if __name__ == '__main__':
    main()

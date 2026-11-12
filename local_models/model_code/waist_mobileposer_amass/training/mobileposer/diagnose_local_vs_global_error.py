"""No-training diagnostic: is the hip/knee LOCAL rotation error just the
composition of independent parent/child GLOBAL errors (Poser predicts global
rotations of 16 reduced joints, local ones come from inverse_kinematics_R)?

For each joint j in (L_hip, R_hip, L_knee, R_knee) with parent p:
  local_err  : angle(R_local_pred, R_local_gt)            (what we measure so far)
  child_glob : angle(R_glob_pred[j], R_glob_gt[j])         (SIP-style global metric)
  par_glob   : angle(R_glob_pred[p], R_glob_gt[p])
  indep      : sqrt(child_glob^2 + par_glob^2) if errors were independent
  corr       : mean cosine between the child/parent global error rotvecs
If local_err ~ indep and corr ~ 0, the flat global-rotation head is wasting
accuracy -> hierarchical/local-loss head (option A) has real room. If
local_err << indep (strong positive corr, errors cancel in the difference),
the head already exploits the structure -> option A is pointless.

Run from SparseIMU/:
    python -m mobileposer.diagnose_local_vs_global_error --min_frames 1800 --max_seqs 30
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_DIR = Path(__file__).resolve().parent
_BASE = _DIR.parent
_CODE = _BASE.parent
sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_CODE))

from mobileposer.config import paths
from mobileposer.utils.model_utils import load_model
from mobileposer.articulate.model import ParametricModel
from mobileposer.evaluate_drift_anchorpose import to_training_basis, prepare_imu, LAYOUT_NAME
from mobileposer.no_head_layouts import _synthesize, synthesis_metadata
from mobileposer.surface_imu import DEFAULT_CONFIG
from mobileposer.orientation_fusion import _so3_log
from mobileposer.joints_swing_fusion import JOINT_PARENT, HIP_KNEE_JOINTS
from drift_eval_common import load_long_sequences, angle_between_rotmats

NAMES = {1: 'L_hip', 2: 'R_hip', 4: 'L_knee', 5: 'R_knee'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--min_frames', type=int, default=1800)
    ap.add_argument('--max_seqs', type=int, default=30)
    ap.add_argument('--checkpoint', default=str(paths.root_dir / 'checkpoints/no_head_5imu_surface/wrists_shanks_waist/1/base_model.pth'))
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint).to(device)
    model.eval()
    bodymodel = ParametricModel(paths.smpl_file)
    synthesis = synthesis_metadata('surface', DEFAULT_CONFIG)
    sequences = load_long_sequences(min_frames=args.min_frames, max_seqs=args.max_seqs)
    print(f'Loaded {len(sequences)} long sequences', flush=True)

    acc = {j: {'local': [], 'child': [], 'par': [], 'cos': []} for j in HIP_KNEE_JOINTS}
    for seq in sequences:
        gt_pose, gt_tran = to_training_basis(seq['pose'], seq['tran'])
        shape = torch.ones(10)
        with torch.no_grad():
            ori, joint, a = _synthesize(bodymodel, gt_pose, shape, gt_tran, LAYOUT_NAME, synthesis, 128)
            imu = prepare_imu(a, ori)
            model.reset()
            pose_pred, _, _, _ = model.forward_offline(imu.to(device).unsqueeze(0), [imu.shape[0]])
            pose_pred = pose_pred.squeeze(0).cpu() if pose_pred.dim() == 4 else pose_pred.cpu()
            T = min(len(gt_pose), len(pose_pred))
            pose_pred, gt = pose_pred[:T].reshape(T, 24, 3, 3), gt_pose[:T].reshape(T, 24, 3, 3)
            g_pred, _ = bodymodel.forward_kinematics(pose_pred, shape, tran=None, calc_mesh=False)
            g_gt, _ = bodymodel.forward_kinematics(gt, shape, tran=None, calc_mesh=False)
            for j in HIP_KNEE_JOINTS:
                p = JOINT_PARENT[j]
                acc[j]['local'].append(angle_between_rotmats(pose_pred[:, j], gt[:, j]).numpy())
                acc[j]['child'].append(angle_between_rotmats(g_pred[:, j], g_gt[:, j]).numpy())
                acc[j]['par'].append(angle_between_rotmats(g_pred[:, p], g_gt[:, p]).numpy())
                ec = _so3_log(g_pred[:, j] @ g_gt[:, j].transpose(-1, -2))   # global-frame error rotvec
                ep = _so3_log(g_pred[:, p] @ g_gt[:, p].transpose(-1, -2))
                cos = torch.nn.functional.cosine_similarity(ec, ep, dim=-1)
                acc[j]['cos'].append(cos.numpy())
        print('  seq done', flush=True)

    print('\njoint    local   child_glob  par_glob  sqrt(c^2+p^2)  corr(cos)')
    tot = {k: [] for k in ('local', 'child', 'par')}
    for j in HIP_KNEE_JOINTS:
        l, c, p, cs = (np.concatenate(acc[j][k]) for k in ('local', 'child', 'par', 'cos'))
        indep = np.sqrt(c ** 2 + p ** 2).mean()
        print(f'{NAMES[j]:7s} {l.mean():6.3f}  {c.mean():9.3f}  {p.mean():8.3f}  {indep:12.3f}  {cs.mean():9.3f}')
        tot['local'].append(l); tot['child'].append(c); tot['par'].append(p)
    print(f'pooled  {np.concatenate(tot["local"]).mean():6.3f}  {np.concatenate(tot["child"]).mean():9.3f}  {np.concatenate(tot["par"]).mean():8.3f}')


if __name__ == '__main__':
    main()

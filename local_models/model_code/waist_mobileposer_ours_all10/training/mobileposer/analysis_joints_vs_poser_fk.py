"""Cheap diagnostic (no training) for a different camera-free Creation Point 2
candidate, after `leg_consistency_fusion` was ruled out (see
project_2026_novelty_search_history item 22: a residual CORRELATING with
error does not imply it carries usable CORRECTION direction).

This candidate is structurally different: `net.py::forward` computes TWO
independent estimates of joint position from the SAME frame of IMU input --
`Joints` module's direct 72D regression (`pred_joints`), and the full pose
`Poser` module's output, which (via forward kinematics) ALSO implies a full
set of joint positions. These are never reconciled at inference (`pred_joints`
is only used as an *input feature* to Poser/FootContact/Velocity, never
cross-checked against Poser's own FK-implied positions). This mirrors the
camera_fusion.py recipe exactly (two decent, independently-produced estimates
of the same quantity -> learn a per-frame trust weight -> convex combine) --
if the pattern holds, it could be ported directly instead of inventing a new
residual-correction mechanism.

The necessary condition for any such fusion to help (standard decision-theory
fact, and the reason CameraFusionGRU works): neither estimate can dominate
the other on (nearly) all frames -- there must be real complementary
information, i.e. each source wins a non-trivial fraction of the time.

Run from SparseIMU/ (mobileposer conda env):
    python -m mobileposer.analysis_joints_vs_poser_fk --min_frames 1800 --max_seqs 30
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

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

from drift_eval_common import load_long_sequences, SEGMENTS

HIP_KNEE_JOINTS = SEGMENTS['Hip'] + SEGMENTS['Knee']   # [1, 2, 4, 5]
ALL_JOINTS = list(range(24))


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
    print(f'Loaded {len(sequences)} long sequences')

    all_err_joints, all_err_poserfk = [], []       # per-joint, HIP_KNEE only, meters
    all_disagree = []                                # ||pred_joints - poser_fk_joints||, HIP_KNEE only

    for seq in sequences:
        gt_pose, gt_tran = to_training_basis(seq['pose'], seq['tran'])
        shape = torch.ones(10)

        with torch.no_grad():
            # ground-truth root-relative joint positions (tran=None -> root-relative FK)
            _, gt_joint = bodymodel.forward_kinematics(gt_pose, shape, tran=None, calc_mesh=False)

            ori, joint, acc = _synthesize(bodymodel, gt_pose, shape, gt_tran, LAYOUT_NAME, synthesis, 128)
            imu = prepare_imu(acc, ori).to(device)
            model.reset()
            pose_pred, pred_joints, _, _ = model.forward_offline(imu.unsqueeze(0), [imu.shape[0]])
            pose_pred = pose_pred.squeeze(0).cpu() if pose_pred.dim() == 4 else pose_pred.cpu()
            pred_joints = pred_joints.squeeze(0).cpu().view(-1, 24, 3) if pred_joints.dim() == 3 else pred_joints.cpu().view(-1, 24, 3)

            # Poser's own FK-implied root-relative joint positions
            _, poser_fk_joint = bodymodel.forward_kinematics(pose_pred, shape, tran=None, calc_mesh=False)

            T = min(len(gt_joint), len(pred_joints), len(poser_fk_joint))
            gt_j = gt_joint[:T, HIP_KNEE_JOINTS]
            pj = pred_joints[:T, HIP_KNEE_JOINTS]
            pf = poser_fk_joint[:T, HIP_KNEE_JOINTS]

            err_joints = (pj - gt_j).norm(dim=-1)        # [T,4]
            err_poserfk = (pf - gt_j).norm(dim=-1)        # [T,4]
            disagree = (pj - pf).norm(dim=-1)              # [T,4]

        all_err_joints.append(err_joints.numpy())
        all_err_poserfk.append(err_poserfk.numpy())
        all_disagree.append(disagree.numpy())
        print(f'  {seq["source"]}: mean_err_joints={err_joints.mean():.4f}m  '
              f'mean_err_poserfk={err_poserfk.mean():.4f}m  mean_disagree={disagree.mean():.4f}m', flush=True)

    ej = np.concatenate(all_err_joints, axis=0)       # [N,4]
    ep = np.concatenate(all_err_poserfk, axis=0)      # [N,4]
    dd = np.concatenate(all_disagree, axis=0)         # [N,4]

    print(f'\n=== Pooled over {ej.shape[0]} frames x 4 hip/knee joints ===')
    print(f'Mean position error -- Joints-direct: {ej.mean()*100:.2f} cm, Poser-FK: {ep.mean()*100:.2f} cm')
    print(f'Mean disagreement (Joints vs Poser-FK): {dd.mean()*100:.2f} cm')

    joints_wins = (ej < ep)
    poserfk_wins = (ep < ej)
    print(f'Joints-direct closer to GT: {100*joints_wins.mean():.1f}% of (frame,joint) pairs')
    print(f'Poser-FK closer to GT:      {100*poserfk_wins.mean():.1f}% of (frame,joint) pairs')

    # Is disagreement itself informative about which source to trust, i.e. is there
    # a usable signal (analogous to checking a blend is learnable, not just "always pick A")?
    which_better = np.where(ej < ep, -1, 1).astype(float)  # -1 = Joints better, +1 = Poser-FK better
    rho, p = spearmanr(dd.flatten(), np.abs(ej - ep).flatten())
    print(f'\nSpearman(disagreement magnitude, |err_joints - err_poserfk|) = {rho:.4f}, p = {p:.3e}  '
          f'(expect positive: bigger disagreement -> bigger stakes in picking correctly)')

    # Oracle potential: if we could always pick the better one, what would hip/knee error be?
    oracle = np.minimum(ej, ep)
    print(f'\nOracle (always pick the better source) mean error: {oracle.mean()*100:.2f} cm '
          f'vs Joints-direct-only {ej.mean()*100:.2f} cm vs Poser-FK-only {ep.mean()*100:.2f} cm')
    print(f'Oracle improvement over the better single source: '
          f'{100*(1 - oracle.mean()/min(ej.mean(), ep.mean())):.1f}%')


if __name__ == '__main__':
    main()

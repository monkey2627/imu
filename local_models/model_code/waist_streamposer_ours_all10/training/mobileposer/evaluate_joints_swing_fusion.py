"""Evaluate the trained SwingFusionGRU on the SAME long-sequence AMASS
benchmark used to validate the motivating oracle test
(analysis_joints_swing_correction_oracle.py), not just the standard
short-window training-data holdout used by train_joints_swing_fusion.py.

Run from SparseIMU/ (mobileposer conda env):
    python -m mobileposer.evaluate_joints_swing_fusion --min_frames 1800 --max_seqs 30
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

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
from mobileposer.joints_swing_fusion import (
    SwingFusionGRU, HIP_KNEE_JOINTS, build_rest_dir, compute_swing_twist_components,
)

from drift_eval_common import load_long_sequences, angle_between_rotmats

NAMES = {1: 'L_hip', 2: 'R_hip', 4: 'L_knee', 5: 'R_knee'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--min_frames', type=int, default=1800)
    ap.add_argument('--max_seqs', type=int, default=30)
    ap.add_argument('--checkpoint', default=str(paths.root_dir / 'checkpoints/no_head_5imu_surface/wrists_shanks_waist/1/base_model.pth'))
    ap.add_argument('--fusion-ckpt', default='runtime_swing_fusion_v1/last.pt')
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint).to(device)
    model.eval()

    fusion = SwingFusionGRU().to(device)
    blob = torch.load(args.fusion_ckpt, map_location=device, weights_only=False)
    fusion.load_state_dict(blob['state_dict'])
    fusion.eval()
    h = blob['manifest']['history'][-1]
    print(f'Loaded fusion from {args.fusion_ckpt}: short-window holdout baseline={h["holdout_baseline_deg"]:.2f} deg, '
          f'corrected={h["holdout_corrected_deg"]:.2f} deg')

    bodymodel = ParametricModel(paths.smpl_file)
    synthesis = synthesis_metadata('surface', DEFAULT_CONFIG)
    rest_dir = build_rest_dir(bodymodel)

    sequences = load_long_sequences(min_frames=args.min_frames, max_seqs=args.max_seqs)
    print(f'Loaded {len(sequences)} long sequences')

    per_joint_base = {j: [] for j in HIP_KNEE_JOINTS}
    per_joint_corr = {j: [] for j in HIP_KNEE_JOINTS}
    per_seq_base, per_seq_corr = [], []

    for seq in sequences:
        gt_pose, gt_tran = to_training_basis(seq['pose'], seq['tran'])
        shape = torch.ones(10)

        with torch.no_grad():
            ori, joint, acc = _synthesize(bodymodel, gt_pose, shape, gt_tran, LAYOUT_NAME, synthesis, 128)
            imu = prepare_imu(acc, ori).to(device)
            model.reset()
            pose_pred, pred_joints, _, _ = model.forward_offline(imu.unsqueeze(0), [imu.shape[0]])
            pose_pred = pose_pred.squeeze(0).cpu() if pose_pred.dim() == 4 else pose_pred.cpu()
            pred_joints = pred_joints.squeeze(0).cpu().view(-1, 24, 3) if pred_joints.dim() == 3 else pred_joints.cpu().view(-1, 24, 3)

            T = min(len(gt_pose), len(pose_pred), len(pred_joints))
            pose_pred, pred_joints, gt_pose_t = pose_pred[:T], pred_joints[:T], gt_pose[:T]
            R_sp, R_st, R_tp = compute_swing_twist_components(bodymodel, pose_pred, pred_joints, rest_dir)

            sp = R_sp.unsqueeze(0).to(device)
            st = R_st.unsqueeze(0).to(device)
            tp = R_tp.unsqueeze(0).to(device)
            corrected, weight, _ = fusion(sp, st, tp)
            corrected = corrected.squeeze(0).cpu()

            poser_hipknee = pose_pred[:, HIP_KNEE_JOINTS]
            gt_hipknee = gt_pose_t[:, HIP_KNEE_JOINTS]
            base_err = angle_between_rotmats(poser_hipknee, gt_hipknee)   # [T,4]
            corr_err = angle_between_rotmats(corrected, gt_hipknee)      # [T,4]

        for i, j in enumerate(HIP_KNEE_JOINTS):
            per_joint_base[j].append(base_err[:, i].numpy())
            per_joint_corr[j].append(corr_err[:, i].numpy())
        per_seq_base.append(base_err.mean().item())
        per_seq_corr.append(corr_err.mean().item())
        print(f'  {seq["source"]}: baseline={base_err.mean():.2f} deg, corrected={corr_err.mean():.2f} deg, '
              f'mean_weight={weight.mean().item():.3f}', flush=True)

    print(f'\n=== Per-joint result over {len(sequences)} sequences ===')
    pooled_base, pooled_corr = [], []
    for j in HIP_KNEE_JOINTS:
        b = np.concatenate(per_joint_base[j])
        c = np.concatenate(per_joint_corr[j])
        frac_improved = (c < b).mean()
        print(f'{NAMES[j]:8s}: Poser_only={b.mean():.2f} deg  corrected={c.mean():.2f} deg  '
              f'({100*(1 - c.mean()/b.mean()):+.1f}%)  frac_frames_improved={100*frac_improved:.1f}%')
        pooled_base.append(b)
        pooled_corr.append(c)

    pb = np.concatenate(pooled_base)
    pc = np.concatenate(pooled_corr)
    print(f'\nPooled (4 joints, {len(pb)} frame-joint pairs): Poser_only={pb.mean():.2f} deg, '
          f'corrected={pc.mean():.2f} deg ({100*(1 - pc.mean()/pb.mean()):+.1f}%)')

    per_seq_base = np.array(per_seq_base)
    per_seq_corr = np.array(per_seq_corr)
    n_better = (per_seq_corr < per_seq_base).sum()
    w_stat, p_w = wilcoxon(per_seq_base, per_seq_corr, alternative='greater')
    print(f'Per-sequence paired Wilcoxon (baseline > corrected): p = {p_w:.3e}; '
          f'{n_better}/{len(sequences)} sequences improved')


if __name__ == '__main__':
    main()

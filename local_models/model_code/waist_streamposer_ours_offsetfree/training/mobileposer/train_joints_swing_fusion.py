"""Train SwingFusionGRU on top of a frozen, already-trained MobilePoserNet.

Pipeline per sequence (all deterministic given the frozen backbone, so caching
only needs to happen once, same as train_leg_consistency_fusion.py):
  1. Run the frozen base_model once to get the full predicted pose and the
     Joints module's independently-regressed joint positions.
  2. Swing-twist decompose each hip/knee joint's rotation relative to its
     rest-pose bone direction, producing two independent SWING estimates
     (Poser's own, and the one implied by Joints' child-joint position) plus
     Poser's TWIST (kept unchanged -- not observable from either position
     source). See joints_swing_fusion.py / analysis_joints_swing_correction_oracle.py.
  3. Train SwingFusionGRU to learn a per-frame, per-joint SLERP blend between
     the two swing estimates, supervised directly against ground-truth
     hip/knee rotation. The base network's four submodules are never updated.

Dataset-level (not subject-level) train/holdout split, same salt as
camera_fusion/orientation_fusion/leg_consistency_fusion (`split_subsets`)
so this module is evaluated on the exact same held-out AMASS subsets.
"""
import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn

from mobileposer.config import model_config, paths
from mobileposer.utils.model_utils import load_model
from mobileposer.articulate.model import ParametricModel
from mobileposer.evaluate_no_head_5imu import load_sequences, prepare_imu
from mobileposer.train_camera_fusion import split_subsets
from mobileposer.joints_swing_fusion import (
    SwingFusionGRU, HIP_KNEE_JOINTS, build_rest_dir, compute_swing_twist_components,
)


def geodesic_loss(R_pred, R_gt):
    diff = R_pred.transpose(-1, -2) @ R_gt
    trace = diff[..., 0, 0] + diff[..., 1, 1] + diff[..., 2, 2]
    theta = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    return theta.mean()


def angle_err_deg(R1, R2):
    diff = R1.transpose(-1, -2) @ R2
    trace = diff[..., 0, 0] + diff[..., 1, 1] + diff[..., 2, 2]
    theta = torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0))
    return torch.rad2deg(theta)


@torch.no_grad()
def cache_one(base_model, bodymodel, rest_dir, seq):
    imu = prepare_imu(seq['acc'], seq['ori'])
    base_model.reset()
    pose_pred, pred_joints, _, _ = base_model.forward_offline(imu.unsqueeze(0).to(next(base_model.parameters()).device),
                                                               [imu.shape[0]])
    pose_pred = pose_pred.squeeze(0).cpu() if pose_pred.dim() == 4 else pose_pred.cpu()
    pred_joints = pred_joints.squeeze(0).cpu().view(-1, 24, 3) if pred_joints.dim() == 3 else pred_joints.cpu().view(-1, 24, 3)

    gt_pose = seq['pose'].float()
    T = min(len(gt_pose), len(pose_pred), len(pred_joints))
    pose_pred, pred_joints = pose_pred[:T], pred_joints[:T]
    R_sp, R_st, R_tp = compute_swing_twist_components(bodymodel, pose_pred, pred_joints, rest_dir)
    gt_hipknee = gt_pose[:T][:, HIP_KNEE_JOINTS]
    poser_hipknee = pose_pred[:, HIP_KNEE_JOINTS]
    return R_sp, R_st, R_tp, gt_hipknee, poser_hipknee


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/no_head_5imu_surface/wrists_shanks_waist/1/base_model.pth')
    parser.add_argument('--data-root', default='data/no_head_5imu_surface_processed/wrists_shanks_waist')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-seq-per-subset', type=int, default=None, help='For smoke tests only')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache-file', type=Path, default=None)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_config.device = device

    data_root = Path(args.data_root)

    if args.cache_file is not None and args.cache_file.exists():
        print(f'Loading cached features from {args.cache_file}')
        blob = torch.load(args.cache_file, map_location='cpu', weights_only=False)
        cached, holdout_subsets = blob['cached'], blob['holdout_subsets']
    else:
        base_model = load_model(args.checkpoint).to(device).eval()
        for p in base_model.parameters():
            p.requires_grad_(False)
        bodymodel = ParametricModel(paths.smpl_file)
        rest_dir = build_rest_dir(bodymodel)

        split, holdout_subsets = split_subsets(data_root)
        print(json.dumps({'subsets': split, 'holdout_subsets': sorted(holdout_subsets)}, indent=2))

        cached = {'train': [], 'holdout': []}
        all_seqs = load_sequences(data_root, max_seq=None)
        for seq in all_seqs:
            subset = seq['source'].split('[')[0]
            fold = split.get(subset)
            if fold is None:
                continue
            if args.max_seq_per_subset is not None:
                count_so_far = sum(1 for s in cached[fold] if s[0] == subset)
                if count_so_far >= args.max_seq_per_subset:
                    continue
            R_sp, R_st, R_tp, gt_hipknee, poser_hipknee = cache_one(base_model, bodymodel, rest_dir, seq)
            cached[fold].append((subset, seq['source'], R_sp, R_st, R_tp, gt_hipknee, poser_hipknee))
        if args.cache_file is not None:
            torch.save({'cached': cached, 'holdout_subsets': holdout_subsets}, args.cache_file)
            print(f'Saved cache to {args.cache_file}')
    print(json.dumps({'train_sequences': len(cached['train']), 'holdout_sequences': len(cached['holdout'])}))

    fusion = SwingFusionGRU(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(fusion.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in fusion.parameters())
    print(json.dumps({'fusion_param_count': n_params}))

    def run_holdout():
        fusion.eval()
        baseline_errs, corrected_errs, per_seq = [], [], []
        with torch.no_grad():
            for subset, source, R_sp, R_st, R_tp, gt_hipknee, poser_hipknee in cached['holdout']:
                sp = R_sp.unsqueeze(0).to(device)
                st = R_st.unsqueeze(0).to(device)
                tp = R_tp.unsqueeze(0).to(device)
                gh = gt_hipknee.unsqueeze(0).to(device)
                corrected, weight, _ = fusion(sp, st, tp)
                b_err = angle_err_deg(poser_hipknee.to(device).unsqueeze(0), gh).mean().item()
                c_err = angle_err_deg(corrected, gh).mean().item()
                baseline_errs.append(b_err)
                corrected_errs.append(c_err)
                per_seq.append({'source': source, 'baseline_deg': b_err, 'corrected_deg': c_err,
                                 'mean_weight': weight.mean().item()})
        fusion.train()
        return baseline_errs, corrected_errs, per_seq

    history = []
    for epoch in range(args.epochs):
        fusion.train()
        train_losses = []
        order = list(range(len(cached['train'])))
        random.Random(args.seed * 1000 + epoch).shuffle(order)
        for idx in order:
            subset, source, R_sp, R_st, R_tp, gt_hipknee, poser_hipknee = cached['train'][idx]
            sp = R_sp.unsqueeze(0).to(device)
            st = R_st.unsqueeze(0).to(device)
            tp = R_tp.unsqueeze(0).to(device)
            gh = gt_hipknee.unsqueeze(0).to(device)
            corrected, _, _ = fusion(sp, st, tp)
            loss = geodesic_loss(corrected, gh)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())

        baseline_errs, corrected_errs, _ = run_holdout()
        record = {
            'epoch': epoch + 1,
            'train_loss_rad': sum(train_losses) / max(len(train_losses), 1),
            'holdout_baseline_deg': sum(baseline_errs) / max(len(baseline_errs), 1),
            'holdout_corrected_deg': sum(corrected_errs) / max(len(corrected_errs), 1),
        }
        history.append(record)
        print(json.dumps(record), flush=True)

    manifest = {
        'status': 'joints_swing_fusion_hipknee_correction',
        'checkpoint': args.checkpoint,
        'data_root': str(data_root),
        'holdout_subsets': sorted(holdout_subsets),
        'fusion_param_count': n_params,
        'epochs': args.epochs,
        'hidden': args.hidden,
        'seed': args.seed,
        'history': history,
        'limitations': [
            'dataset-level (AMASS subset) holdout split, not subject-level',
            'twist component always kept from Poser (not observable from joint positions); only the swing component (2 of 3 DOF) can ever be corrected',
            'validated oracle headroom (analysis_joints_swing_correction_oracle.py) is +8.4% pooled on long sequences, consistently positive across all 4 joints but with asymmetric magnitude (L side ~+4-9% naive, R side strongly negative naive) -- a real learned improvement must beat the naive always-apply baseline, not just reach the oracle',
        ],
    }
    (args.output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    torch.save({'state_dict': fusion.state_dict(), 'manifest': manifest}, args.output / 'last.pt')

    _, _, per_seq_final = run_holdout()
    (args.output / 'holdout_detail.json').write_text(json.dumps(per_seq_final, indent=2, ensure_ascii=False))
    print(f"Saved: {args.output / 'holdout_detail.json'}")


if __name__ == '__main__':
    main()

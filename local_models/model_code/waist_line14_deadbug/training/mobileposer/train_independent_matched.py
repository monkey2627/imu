"""Fair-comparison control: retrain the two INDEPENDENT networks
(CameraFusionGRU, OrientationFusionGRU) using the SAME correlated-failure
noise generator (synthetic_slam.synthesize_slam_joint) that JointAnchorFusion
trains on -- each independent network only sees its own axis's noisy signal
(ignores the cross-axis correlation), but now experiences the identical
marginal noise distribution, so any remaining gap vs JointAnchorFusion is
attributable to the shared-hidden-state architecture, not a train/test
distribution mismatch.

Run from SparseIMU/:
    python -m mobileposer.train_independent_matched \
        --tran-cache /tmp/tran_cache.pt --ori-cache /tmp/ori_cache.pt \
        --output runtime_independent_matched
"""
import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn

from mobileposer.synthetic_slam import synthesize_slam_joint, stable_seed, sample_time_constants_loguniform
from mobileposer.camera_fusion import CameraFusionGRU
from mobileposer.orientation_fusion import OrientationFusionGRU
from mobileposer.train_joint_anchor_fusion import merge_caches, angle_between_rotmats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tran-cache', type=Path, required=True)
    ap.add_argument('--ori-cache', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--slam-rms-m', type=float, default=0.66)
    ap.add_argument('--slam-rms-deg', type=float, default=3.0)
    ap.add_argument('--time-constant-s', type=float, default=None,
                     help='Override both pos/rot OU time constants with a single value '
                          '(for the time-constant sweep). Default: use synthesize_slam_joint '
                          'defaults (0.3/0.4, real-calibrated).')
    ap.add_argument('--time-constant-range', type=float, nargs=2, default=None, metavar=('LOW', 'HIGH'),
                     help='Domain-randomize: sample pos/rot OU time constants INDEPENDENTLY '
                          'log-uniform from [LOW, HIGH] seconds, resampled every training step '
                          '(for AdaptiveAnchorFusion stage 1: a robust-across-tc independent '
                          'checkpoint). Mutually exclusive with --time-constant-s.')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.time_constant_s is not None and args.time_constant_range is not None:
        raise ValueError('--time-constant-s and --time-constant-range are mutually exclusive')
    tc_kwargs = {} if args.time_constant_s is None else \
        {'pos_time_constant_s': args.time_constant_s, 'rot_time_constant_s': args.time_constant_s}
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tran_blob = torch.load(args.tran_cache, map_location='cpu', weights_only=False)
    ori_blob = torch.load(args.ori_cache, map_location='cpu', weights_only=False)
    merged = {'train': merge_caches(tran_blob['cached'], ori_blob['cached'], 'train'),
              'holdout': merge_caches(tran_blob['cached'], ori_blob['cached'], 'holdout')}
    print(json.dumps({'train_sequences': len(merged['train']), 'holdout_sequences': len(merged['holdout'])}))

    cam = CameraFusionGRU(hidden=args.hidden).to(device)
    ori = OrientationFusionGRU(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(list(cam.parameters()) + list(ori.parameters()), lr=args.lr)
    print(json.dumps({'cam_params': sum(p.numel() for p in cam.parameters()),
                       'ori_params': sum(p.numel() for p in ori.parameters())}))

    def run_holdout(seed_offset=0):
        cam.eval(); ori.eval()
        rows = []
        with torch.no_grad():
            for subset, source, tran_pred, tran_gt, ori_pred, ori_gt in merged['holdout']:
                seed = stable_seed(source, seed_offset)
                if args.time_constant_range is not None:
                    pos_tc, rot_tc = sample_time_constants_loguniform(
                        *args.time_constant_range, rng=random.Random(seed))
                    holdout_tc_kwargs = {'pos_time_constant_s': pos_tc, 'rot_time_constant_s': rot_tc}
                else:
                    holdout_tc_kwargs = tc_kwargs
                slam_pos, slam_ori, _ = synthesize_slam_joint(
                    tran_gt, ori_gt, fps=30.0, target_rms_m=args.slam_rms_m,
                    target_rms_deg=args.slam_rms_deg, seed=seed, **holdout_tc_kwargs)
                imu_pos_b, slam_pos_b = tran_pred.unsqueeze(0).to(device), slam_pos.unsqueeze(0).to(device)
                imu_ori_b, slam_ori_b = ori_pred.unsqueeze(0).to(device), slam_ori.unsqueeze(0).to(device)
                cam_fused, _, _ = cam(imu_pos_b, slam_pos_b)
                ori_fused, _, _ = ori(imu_ori_b, slam_ori_b)
                pos_err = (cam_fused.squeeze(0).cpu() - tran_gt).norm(dim=-1).mean().item()
                ori_err = angle_between_rotmats(ori_fused.squeeze(0).cpu(), ori_gt).mean().item()
                rows.append({'source': source, 'indep_matched_pos_m': pos_err, 'indep_matched_ori_deg': ori_err,
                             'pure_imu_pos_m': (tran_pred - tran_gt).norm(dim=-1).mean().item(),
                             'pure_imu_ori_deg': angle_between_rotmats(ori_pred, ori_gt).mean().item()})
        cam.train(); ori.train()
        return rows

    history = []
    for epoch in range(args.epochs):
        cam.train(); ori.train()
        losses = []
        order = list(range(len(merged['train'])))
        random.Random(args.seed * 1000 + epoch).shuffle(order)
        for idx in order:
            subset, source, tran_pred, tran_gt, ori_pred, ori_gt = merged['train'][idx]
            if args.time_constant_range is not None:
                pos_tc, rot_tc = sample_time_constants_loguniform(*args.time_constant_range)
                step_tc_kwargs = {'pos_time_constant_s': pos_tc, 'rot_time_constant_s': rot_tc}
            else:
                step_tc_kwargs = tc_kwargs
            slam_pos, slam_ori, _ = synthesize_slam_joint(
                tran_gt, ori_gt, fps=30.0, target_rms_m=args.slam_rms_m,
                target_rms_deg=args.slam_rms_deg, seed=None, **step_tc_kwargs)
            imu_pos_b, slam_pos_b = tran_pred.unsqueeze(0).to(device), slam_pos.unsqueeze(0).to(device)
            imu_ori_b, slam_ori_b = ori_pred.unsqueeze(0).to(device), slam_ori.unsqueeze(0).to(device)
            pos_target, ori_target = tran_gt.unsqueeze(0).to(device), ori_gt.unsqueeze(0).to(device)

            cam_fused, _, _ = cam(imu_pos_b, slam_pos_b)
            ori_fused, _, _ = ori(imu_ori_b, slam_ori_b)
            loss = (nn.functional.mse_loss(cam_fused, pos_target)
                    + 20.0 * nn.functional.mse_loss(ori_fused, ori_target))
            if not torch.isfinite(loss):
                # Rare extreme OU noise realization (more likely at large time
                # constants, where excursions can persist/compound) can blow up
                # the loss on a single sequence; skip this step rather than
                # corrupting the weights with a NaN/Inf gradient.
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(cam.parameters()) + list(ori.parameters()), 1.0)
            opt.step()
            losses.append(loss.item())

        rows = run_holdout()
        record = {'epoch': epoch + 1, 'train_loss': sum(losses) / max(len(losses), 1),
                  'pure_imu_pos_m': sum(r['pure_imu_pos_m'] for r in rows) / len(rows),
                  'pure_imu_ori_deg': sum(r['pure_imu_ori_deg'] for r in rows) / len(rows),
                  'indep_matched_pos_m': sum(r['indep_matched_pos_m'] for r in rows) / len(rows),
                  'indep_matched_ori_deg': sum(r['indep_matched_ori_deg'] for r in rows) / len(rows)}
        history.append(record)
        print(json.dumps(record), flush=True)

    torch.save({'cam_state_dict': cam.state_dict(), 'ori_state_dict': ori.state_dict(), 'history': history},
               args.output / 'last.pt')
    final_rows = run_holdout()
    (args.output / 'holdout_detail.json').write_text(json.dumps(final_rows, indent=2, ensure_ascii=False))
    print(f"\nSaved: {args.output / 'holdout_detail.json'}")


if __name__ == '__main__':
    main()

"""Joint ablation of the unified AnchorPose pipeline (anchorpose_pipeline.py) on
the 30 held-out long AMASS sequences: baseline / +cam / +ori / +swing / all.

Metrics (per sequence, then averaged; paired Wilcoxon all-vs-baseline):
  tran_m        mean root-translation error (m)           <- camera fusion
  root_ori_deg  mean root global-orientation error (deg)  <- orientation fusion
  hipknee_deg   mean L/R hip+knee LOCAL error (deg)       <- swing fusion
  glob_ori_deg  mean GLOBAL (not root-aligned) orientation error over the 15
                reduced joints                            <- root ori + local joints
  mpjpe_cm      world-frame joint-position error, FK(pose, tran) vs GT (cm)
                                                          <- all three together
SLAM signals are synthetic (calibrated OU noise, see synthetic_slam.py).

Run from SparseIMU/:
    python -m mobileposer.evaluate_anchorpose_pipeline --max_seqs 30
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

from mobileposer.anchorpose_pipeline import AnchorPosePipeline
from mobileposer.evaluate_drift_anchorpose import prepare_imu, LAYOUT_NAME
from mobileposer.no_head_layouts import _synthesize, synthesis_metadata
from mobileposer.surface_imu import DEFAULT_CONFIG
from mobileposer.synthetic_slam import synthesize_slam_position, synthesize_slam_orientation, stable_seed
from mobileposer.joints_swing_fusion import HIP_KNEE_JOINTS
from drift_eval_common import load_long_sequences, angle_between_rotmats, FPS

REDUCED15 = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
CONFIGS = {
    'baseline': dict(use_cam=False, use_ori=False, use_swing=False),
    '+cam(CP1 trans)': dict(use_cam=True, use_ori=False, use_swing=False),
    '+ori(CP1 rot)': dict(use_cam=False, use_ori=True, use_swing=False),
    '+ori(propagate)': dict(use_cam=False, use_ori=True, use_swing=False, ori_mode='propagate'),
    '+swing(CP2)': dict(use_cam=False, use_ori=False, use_swing=True),
    'all': dict(use_cam=True, use_ori=True, use_swing=True),
}
METRICS = ['tran_m', 'root_ori_deg', 'hipknee_deg', 'glob_ori_deg', 'mpjpe_cm']


def metrics(pipe, pose, tran, gt_pose, gt_tran_rel, gt_joint_world, shape):
    T = min(len(pose), len(gt_pose))
    pose, tran, gt_pose, gt_tran_rel, gt_joint_world = pose[:T], tran[:T], gt_pose[:T], gt_tran_rel[:T], gt_joint_world[:T]
    g_pred, j_pred = pipe.bodymodel.forward_kinematics(pose, shape, tran=tran, calc_mesh=False)
    g_gt, _ = pipe.bodymodel.forward_kinematics(gt_pose, shape, tran=None, calc_mesh=False)
    return {
        'tran_m': (tran - gt_tran_rel).norm(dim=-1).mean().item(),
        'root_ori_deg': angle_between_rotmats(pose[:, 0], gt_pose[:, 0]).mean().item(),
        'hipknee_deg': angle_between_rotmats(pose[:, HIP_KNEE_JOINTS], gt_pose[:, HIP_KNEE_JOINTS]).mean().item(),
        'glob_ori_deg': angle_between_rotmats(g_pred.reshape(T, 24, 3, 3)[:, REDUCED15], g_gt.reshape(T, 24, 3, 3)[:, REDUCED15]).mean().item(),
        'mpjpe_cm': 100.0 * (j_pred.reshape(T, 24, 3) - gt_joint_world).norm(dim=-1).mean().item(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--min_frames', type=int, default=1800)
    ap.add_argument('--max_seqs', type=int, default=30)
    ap.add_argument('--slam-time-constant-s', type=float, default=0.3)
    ap.add_argument('--slam-rms-m', type=float, default=0.66)
    ap.add_argument('--slam-rms-deg', type=float, default=3.0)
    ap.add_argument('--device', default=None)
    ap.add_argument('--dip-file', default=None, help='DIP-GT layout data (no translation GT): evaluates ori+swing only')
    ap.add_argument('--min_dip_frames', type=int, default=300)
    ap.add_argument('--cam-ckpt', default=None)
    ap.add_argument('--ori-ckpt', default=None)
    ap.add_argument('--swing-ckpt', default=None)
    ap.add_argument('--noise-offset', type=int, default=0, help='shifts the SLAM noise seeds (use 0/10/20 for 3 noise realisations)')
    args = ap.parse_args()

    kw = {k: v for k, v in (('cam_ckpt', args.cam_ckpt), ('ori_ckpt', args.ori_ckpt), ('swing_ckpt', args.swing_ckpt)) if v}
    pipe = AnchorPosePipeline(device=args.device, **kw)
    synthesis = synthesis_metadata('surface', DEFAULT_CONFIG)
    dip = args.dip_file is not None
    if dip:
        data = torch.load(args.dip_file, map_location='cpu')
        sequences = [{'acc': a, 'ori': o, 'pose': p.reshape(-1, 24, 3, 3), 'tran': torch.zeros(len(p), 3), 'source': f'dip[{i}]'}
                     for i, (a, o, p) in enumerate(zip(data['acc'], data['ori'], data['pose'])) if len(p) >= args.min_dip_frames]
        configs = {k: v for k, v in CONFIGS.items() if not v['use_cam']}
        metric_names = ['root_ori_deg', 'hipknee_deg', 'glob_ori_deg']
    else:
        sequences = load_long_sequences(min_frames=args.min_frames, max_seqs=args.max_seqs)
        configs, metric_names = CONFIGS, METRICS
    print(f'Loaded {len(sequences)} sequences (dip={dip})', flush=True)

    res = {c: {m: [] for m in METRICS} for c in configs}
    cp3 = []
    for seq in sequences:
        gt_pose, gt_tran = seq['pose'].clone(), seq['tran'].clone()
        shape = torch.ones(10)
        if dip:
            acc, ori = seq['acc'], seq['ori']
        else:
            ori, _, acc = _synthesize(pipe.bodymodel, gt_pose, shape, gt_tran, LAYOUT_NAME, synthesis, 128)
        imu = prepare_imu(acc, ori)
        gt_tran_rel = gt_tran - gt_tran[:1]
        slam_pos = synthesize_slam_position(gt_tran, fps=FPS, target_rms_m=args.slam_rms_m,
                                            time_constant_s=args.slam_time_constant_s, seed=stable_seed(seq['source'], offset=args.noise_offset)) - gt_tran[:1]
        slam_ori = synthesize_slam_orientation(gt_pose[:, 0], fps=FPS, target_rms_deg=args.slam_rms_deg,
                                               seed=stable_seed(seq['source'], offset=args.noise_offset + 1))
        _, gt_joint_world = pipe.bodymodel.forward_kinematics(gt_pose, shape, tran=gt_tran_rel, calc_mesh=False)
        gt_joint_world = gt_joint_world.reshape(-1, 24, 3)
        for name, flags in configs.items():
            out = pipe(imu, slam_pos, slam_ori, **flags)
            m = metrics(pipe, out['pose'], out['tran'], gt_pose, gt_tran_rel, gt_joint_world, shape)
            for k, v in m.items():
                res[name][k].append(v)
            if name == '+cam(CP1 trans)':   # Creation Point 3: trust weight (1=trust IMU) vs per-frame IMU error
                T_ = len(out['weights']['cam'])
                cp3.append((out['weights']['cam'].mean(dim=-1).numpy(),
                            (out['base_tran'][:T_] - gt_tran_rel[:T_]).norm(dim=-1).numpy()))
        print(f'  {seq["source"]}: done', flush=True)

    if cp3:
        from scipy.stats import spearmanr
        w_all, e_all = np.concatenate([w for w, _ in cp3]), np.concatenate([e for _, e in cp3])
        rho_pool = spearmanr(w_all, e_all).correlation
        rhos = np.array([spearmanr(w, e).correlation for w, e in cp3])
        print(f'\n[CP3 trust weight] pooled Spearman(weight, IMU per-frame err)={rho_pool:.3f}  '
              f'per-seq mean={np.nanmean(rhos):.3f}  negative in {int((rhos < 0).sum())}/{len(rhos)} seqs  n_frames={len(w_all)}')

    base = {m: np.array(res['baseline'][m]) for m in metric_names}
    print('\n' + 'config'.ljust(18) + ''.join(m.rjust(15) for m in metric_names))
    for name in configs:
        print(name.ljust(18) + ''.join(f'{np.mean(res[name][m]):15.3f}' for m in metric_names))
    print('\nchange vs baseline (+ = better), improved-seqs, paired Wilcoxon p:')
    for name in list(configs)[1:]:
        cells = []
        for m in metric_names:
            v, b = np.array(res[name][m]), base[m]
            try:
                p = wilcoxon(b, v).pvalue
            except ValueError:
                p = float('nan')
            cells.append(f'{100 * (1 - v.mean() / b.mean()):+6.1f}% {int((v < b).sum()):2d}/{len(v)} p={p:.0e}')
        print(name.ljust(18) + ' | '.join(cells))
    print('columns:', metric_names)


if __name__ == '__main__':
    main()

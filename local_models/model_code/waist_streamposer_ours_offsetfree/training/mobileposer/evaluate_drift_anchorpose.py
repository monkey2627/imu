"""Add AnchorPose (wrists_shanks_waist layout, IMU-only vs camera-fused) to the
existing multi-method drift-comparison benchmark (see code/drift_comparison_report.md,
which already covers MobilePoser/IMUCoCo/PNP/SlimeVR on the same AMASS long-
sequence protocol via code/drift_eval_common.py).

Reuses, does not re-derive:
  - drift_eval_common.load_long_sequences: the SAME long AMASS sequences as the
    other four methods (guarantees a fair, identical-data comparison).
  - no_head_layouts._synthesize: the SAME surface-IMU synthesis this project
    already uses for its own training/eval data, applied here to the drift
    benchmark's ground-truth pose/tran instead of the original AMASS pipeline
    output -- the shared loader's raw acc/ori are for the STANDARD 6-slot
    layout (L/R wrist, L/R hip, head, pelvis), which has no shank sensor and
    is not compatible with our wrists_shanks_waist layout.
  - synthetic_slam.synthesize_slam_position + camera_fusion.CameraFusionGRU:
    the same translation-fusion pipeline used throughout this project
    (domain-randomized-over-time-constant independent checkpoint).
  - The exact tran_err formula from mobileposer/evaluate_drift.py, copied
    verbatim: tran_err = (tran_pred - (gt_tran - gt_tran[:1])).norm(dim=-1) --
    so the resulting Tran(m) numbers are directly comparable to the existing
    table's PNP/MobilePoser/IMUCoCo rows, not a differently-defined metric.

Run from SparseIMU/ (mobileposer conda env):
    python -m mobileposer.evaluate_drift_anchorpose --min_frames 900 --max_seqs 3   # smoke test
    python -m mobileposer.evaluate_drift_anchorpose --min_frames 1800 --max_seqs 30  # full benchmark
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_DIR = Path(__file__).resolve().parent            # mobileposer/
_BASE = _DIR.parent                                # base_mobileposer/
_CODE = _BASE.parent                                # code/
sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_CODE))

from mobileposer.config import paths
from mobileposer.utils.model_utils import load_model
from mobileposer.articulate.model import ParametricModel
from mobileposer.no_head_layouts import _synthesize, synthesis_metadata
from mobileposer.surface_imu import DEFAULT_CONFIG
from mobileposer.camera_fusion import CameraFusionGRU

# CORRECTED 2026-10-01 (was a bug for over a week): this used to re-apply
# amass_rot to data from drift_eval_common.load_long_sequences, on the false
# premise that its source (data/processed_datasets/*.pt) was raw/unrotated
# AMASS. It is NOT -- process.py::process_amass() already applies this exact
# amass_rot once when building those .pt files (see process.py's "align AMASS
# global frame with DIP" comment). no_head_layouts.py::_load_amass_dataset
# (the actual wrists_shanks_waist training-data pipeline) separately reads
# raw *_poses.npz and applies amass_rot once too -- so both pipelines apply
# the rotation EXACTLY ONCE relative to raw AMASS, and data already loaded
# via load_long_sequences must NOT be rotated again here.
#
# The double-rotation silently misaligned gravity for IMU synthesis, which
# inflated hip/knee angular error ~5-10x (e.g. 31-46 deg vs a corrected
# 4-9 deg, matching the training pipeline's own independently-measured
# 5.8-6.6 deg baseline on the same AMASS subset) and inflated translation
# drift error ~10-20x (e.g. 2-5m vs a corrected 0.07-0.5m). The original
# "implausibly small" numbers this function was written to fix were, as far
# as can now be determined, the correct ones -- this project's
# wrists_shanks_waist layout (shank-mounted IMUs) plausibly has a large,
# genuine translation-tracking advantage via its ZUPT-style foot-contact
# velocity fallback (net.py's contact_vel) over layouts without a shank/foot
# sensor, which is a real structural reason, not a sign of a bug, for its
# pure-IMU baseline to beat PNP/MobilePoser/IMUCoCo by a wide margin even
# before any camera fusion. See project_anchorpose_current_state_20261001.md
# for the discovery writeup and which downstream numbers need re-deriving.


def to_training_basis(pose, tran):
    """No-op (identity passthrough), kept only so existing call sites across
    this project's analysis/evaluate/diagnose scripts do not need editing.
    Data from drift_eval_common.load_long_sequences is already in the
    correct training basis -- see the correction note above."""
    return pose.clone(), tran.clone()
from mobileposer.synthetic_slam import synthesize_slam_position, stable_seed

from drift_eval_common import load_long_sequences, LUMBAR_JOINTS, angle_between_rotmats, FPS

LAYOUT_NAME = 'wrists_shanks_waist'


def prepare_imu(acc, ori):
    """Matches evaluate_no_head_5imu.py::prepare_imu exactly (acc_scale=30)."""
    ns = int(__import__("os").environ.get("MOBILEPOSER_NSENS", 5))
    acc = (acc[:, :ns] / 30.0).float()
    ori = ori[:, :ns].float()
    return torch.cat([acc.flatten(1), ori.flatten(1)], dim=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--min_frames', type=int, default=1800)
    ap.add_argument('--max_seqs', type=int, default=30)
    ap.add_argument('--checkpoint', default=str(paths.root_dir / 'checkpoints/no_head_5imu_surface/wrists_shanks_waist/1/base_model.pth'))
    ap.add_argument('--camera-fusion-ckpt', default='runtime_indep_tcrand/last.pt')
    ap.add_argument('--slam-time-constant-s', type=float, default=0.3,
                     help='Real-calibrated deployment value (see AnchorPose 4.0 section), not the training-time domain-randomized range.')
    ap.add_argument('--slam-rms-m', type=float, default=0.66)
    ap.add_argument('--checkpoint-s', type=int, default=120)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='drift_results_anchorpose.npz')
    args = ap.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint).to(device)
    model.eval()

    cam = CameraFusionGRU().to(device)
    cam_blob = torch.load(args.camera_fusion_ckpt, map_location=device, weights_only=False)
    cam.load_state_dict(cam_blob['cam_state_dict'] if 'cam_state_dict' in cam_blob else cam_blob['state_dict'])
    cam.eval()

    bodymodel = ParametricModel(paths.smpl_file)
    synthesis = synthesis_metadata('surface', DEFAULT_CONFIG)

    sequences = load_long_sequences(min_frames=args.min_frames, max_seqs=args.max_seqs)
    print(f'Loaded {len(sequences)} long sequences')

    max_frames = args.checkpoint_s * FPS + 1
    lumbar_sum = np.zeros(max_frames)
    tran_sum_imu = np.zeros(max_frames)
    tran_sum_fused = np.zeros(max_frames)
    count = np.zeros(max_frames)

    for seq in sequences:
        gt_pose, gt_tran = to_training_basis(seq['pose'], seq['tran'])  # match backbone's training basis
        T = gt_pose.shape[0]
        shape = torch.ones(10)

        with torch.no_grad():
            ori, joint, acc = _synthesize(bodymodel, gt_pose, shape, gt_tran, LAYOUT_NAME, synthesis, 128)
            imu = prepare_imu(acc, ori).to(device)

            model.reset()
            pose_pred, _, tran_pred, _ = model.forward_offline(imu.unsqueeze(0), [imu.shape[0]])
            pose_pred = pose_pred.squeeze(0).cpu() if pose_pred.dim() == 4 else pose_pred.cpu()
            tran_pred = tran_pred.squeeze(0).cpu() if tran_pred.dim() == 3 else tran_pred.cpu()

            gt_tran_rel = gt_tran - gt_tran[:1]

            slam_pos = synthesize_slam_position(
                gt_tran, fps=FPS, target_rms_m=args.slam_rms_m,
                time_constant_s=args.slam_time_constant_s, seed=stable_seed(seq['source']))
            slam_pos_rel = slam_pos - gt_tran[:1]  # same reference frame as tran_pred

            tran_pred_b = tran_pred.unsqueeze(0).to(device)
            slam_pos_b = slam_pos_rel.unsqueeze(0).to(device)
            tran_fused, _, _ = cam(tran_pred_b, slam_pos_b)
            tran_fused = tran_fused.squeeze(0).cpu()

            lumbar_err = angle_between_rotmats(pose_pred[:, LUMBAR_JOINTS], gt_pose[:, LUMBAR_JOINTS]).mean(dim=-1).numpy()
            tran_err_imu = (tran_pred - gt_tran_rel).norm(dim=-1).numpy()
            tran_err_fused = (tran_fused - gt_tran_rel).norm(dim=-1).numpy()

        n = min(T, max_frames)
        lumbar_sum[:n] += lumbar_err[:n]
        tran_sum_imu[:n] += tran_err_imu[:n]
        tran_sum_fused[:n] += tran_err_fused[:n]
        count[:n] += 1
        print(f'  {seq["source"]}: T={T}', flush=True)

    valid = count > 0
    lumbar_avg = np.where(valid, lumbar_sum / np.maximum(count, 1), 0.0)
    tran_avg_imu = np.where(valid, tran_sum_imu / np.maximum(count, 1), 0.0)
    tran_avg_fused = np.where(valid, tran_sum_fused / np.maximum(count, 1), 0.0)

    checkpoint_frame = min(args.checkpoint_s * FPS, max_frames - 1)
    print(f'\n@ {args.checkpoint_s}s (frame {checkpoint_frame}, n_valid_seqs={int(count[checkpoint_frame])}):')
    print(f'  Lumbar5 (IMU-only; unaffected by camera fusion by construction): {lumbar_avg[checkpoint_frame]:.2f} deg')
    print(f'  Tran(m) IMU-only:      {tran_avg_imu[checkpoint_frame]:.3f} m')
    print(f'  Tran(m) +camera fused: {tran_avg_fused[checkpoint_frame]:.3f} m')

    np.savez(args.out, lumbar=lumbar_avg, tran_imu=tran_avg_imu, tran_fused=tran_avg_fused, count=count)
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()

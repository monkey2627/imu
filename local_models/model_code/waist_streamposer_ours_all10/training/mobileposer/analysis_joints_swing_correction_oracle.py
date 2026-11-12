"""Oracle test (no training): does the position-level complementarity between
Joints-direct regression and Poser-FK (see analysis_joints_vs_poser_fk.py,
9.0% position-error oracle headroom, 30/30 sequences consistent) actually
translate into a ROTATION-level improvement for hip/knee -- the thing SIP/
Angular Error actually measures? Position-level promise does not guarantee
rotation-level promise (see leg_consistency_fusion's failure: a real
correlation did not carry usable directional information) -- this must be
checked before investing in training a real fusion module.

Method: swing-twist decomposition relative to each hip/knee joint's rest-pose
bone direction (hip's child=knee, knee's child=ankle). Poser's own local
rotation already implies a "swing" (bending direction, 2 observable DOF) and
a "twist" (rotation about the bone axis, NOT observable from joint positions
alone, so always kept from Poser). The candidate correction: replace Poser's
implied swing with the swing implied by Joints-direct's child-joint position,
keep Poser's twist unchanged. Compare corrected-rotation angular error vs
Poser's original, per frame/joint, plus an oracle (always pick whichever is
closer to GT) to get the achievable upper bound before any learned blending.

Run from SparseIMU/ (mobileposer conda env):
    python -m mobileposer.analysis_joints_swing_correction_oracle --min_frames 1800 --max_seqs 30
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

from drift_eval_common import load_long_sequences, angle_between_rotmats


def axis_angle_to_rotmat(v: torch.Tensor) -> torch.Tensor:
    """[...,3] axis-angle -> [...,3,3] rotation matrix, batched Rodrigues formula."""
    theta = v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    k = v / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zeros = torch.zeros_like(kx)
    K = torch.stack([
        torch.stack([zeros, -kz, ky], dim=-1),
        torch.stack([kz, zeros, -kx], dim=-1),
        torch.stack([-ky, kx, zeros], dim=-1),
    ], dim=-2)
    I = torch.eye(3, device=v.device, dtype=v.dtype).expand(*v.shape[:-1], 3, 3)
    sin_t = torch.sin(theta).unsqueeze(-1)
    cos_t = torch.cos(theta).unsqueeze(-1)
    return I + sin_t * K + (1.0 - cos_t) * (K @ K)


# joint -> child joint whose position defines the observable "swing" direction
JOINT_CHILD = {1: 4, 2: 5, 4: 7, 5: 8}   # L_hip->L_knee, R_hip->R_knee, L_knee->L_ankle, R_knee->R_ankle
JOINT_PARENT = {1: 0, 2: 0, 4: 1, 5: 2}  # whose GLOBAL rotation frames the observed direction
HIP_KNEE_JOINTS = list(JOINT_CHILD.keys())


def shortest_arc_rotation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[...,3] unit vectors -> [...,3,3] minimal rotation taking a to b."""
    v = torch.cross(a, b, dim=-1)
    c = (a * b).sum(-1, keepdim=True)
    s = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = v / s
    theta = torch.atan2(s, c)  # [...,1], robust, in [0, pi]
    return axis_angle_to_rotmat(axis * theta)


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
    shape = torch.ones(10)
    zero_joint, _ = bodymodel.get_zero_pose_joint_and_vertex(shape)
    zero_joint = zero_joint[0]  # [24,3]
    rest_dir = {j: torch.nn.functional.normalize(zero_joint[c] - zero_joint[j], dim=-1) for j, c in JOINT_CHILD.items()}

    sequences = load_long_sequences(min_frames=args.min_frames, max_seqs=args.max_seqs)
    print(f'Loaded {len(sequences)} long sequences')

    all_base_err, all_corr_err = {j: [] for j in HIP_KNEE_JOINTS}, {j: [] for j in HIP_KNEE_JOINTS}

    for seq in sequences:
        gt_pose, gt_tran = to_training_basis(seq['pose'], seq['tran'])

        with torch.no_grad():
            ori, joint, acc = _synthesize(bodymodel, gt_pose, shape, gt_tran, LAYOUT_NAME, synthesis, 128)
            imu = prepare_imu(acc, ori).to(device)
            model.reset()
            pose_pred, pred_joints, _, _ = model.forward_offline(imu.unsqueeze(0), [imu.shape[0]])
            pose_pred = pose_pred.squeeze(0).cpu() if pose_pred.dim() == 4 else pose_pred.cpu()
            pred_joints = pred_joints.squeeze(0).cpu().view(-1, 24, 3) if pred_joints.dim() == 3 else pred_joints.cpu().view(-1, 24, 3)

            pose_global_pred, poser_fk_joint = bodymodel.forward_kinematics(pose_pred, shape, tran=None, calc_mesh=False)
            T = min(len(gt_pose), len(pose_pred), len(pred_joints), len(poser_fk_joint))
            pose_pred, pred_joints = pose_pred[:T], pred_joints[:T]
            pose_global_pred, poser_fk_joint = pose_global_pred[:T], poser_fk_joint[:T]
            gt_pose_t = gt_pose[:T]

            for j, c in JOINT_CHILD.items():
                R_parent_global = pose_global_pred[:, JOINT_PARENT[j]]                       # [T,3,3]
                obs_dir_joints = torch.nn.functional.normalize(pred_joints[:, c] - pred_joints[:, j], dim=-1)
                dir_poser = torch.nn.functional.normalize(poser_fk_joint[:, c] - poser_fk_joint[:, j], dim=-1)
                obs_local_joints = (R_parent_global.transpose(-1, -2) @ obs_dir_joints.unsqueeze(-1)).squeeze(-1)
                dir_local_poser = (R_parent_global.transpose(-1, -2) @ dir_poser.unsqueeze(-1)).squeeze(-1)

                rd = rest_dir[j].expand_as(obs_local_joints)
                R_swing_poser = shortest_arc_rotation(rd, dir_local_poser)
                R_swing_target = shortest_arc_rotation(rd, obs_local_joints)

                R_poser_local = pose_pred[:, j]
                R_twist_poser = R_swing_poser.transpose(-1, -2) @ R_poser_local
                R_corrected = R_swing_target @ R_twist_poser

                base_err = angle_between_rotmats(R_poser_local, gt_pose_t[:, j])
                corr_err = angle_between_rotmats(R_corrected, gt_pose_t[:, j])
                all_base_err[j].append(base_err.numpy())
                all_corr_err[j].append(corr_err.numpy())

        print(f'  {seq["source"]}: done', flush=True)

    print(f'\n=== Per-joint swing-correction oracle test, {len(sequences)} sequences ===')
    names = {1: 'L_hip', 2: 'R_hip', 4: 'L_knee', 5: 'R_knee'}
    pooled_base, pooled_corr = [], []
    for j in HIP_KNEE_JOINTS:
        b = np.concatenate(all_base_err[j])
        c = np.concatenate(all_corr_err[j])
        oracle = np.minimum(b, c)
        frac_improved = (c < b).mean()
        print(f'{names[j]:8s}: n={len(b)}  Poser_only={b.mean():.2f} deg  swing_corrected={c.mean():.2f} deg  '
              f'({100*(1 - c.mean()/b.mean()):+.1f}%)  oracle={oracle.mean():.2f} deg  '
              f'({100*(1 - oracle.mean()/b.mean()):+.1f}%)  frac_frames_improved={100*frac_improved:.1f}%')
        pooled_base.append(b)
        pooled_corr.append(c)

    pb = np.concatenate(pooled_base)
    pc = np.concatenate(pooled_corr)
    oracle = np.minimum(pb, pc)
    print(f'\nPooled (4 joints): Poser_only={pb.mean():.2f} deg, swing_corrected={pc.mean():.2f} deg '
          f'({100*(1 - pc.mean()/pb.mean()):+.1f}%), oracle={oracle.mean():.2f} deg '
          f'({100*(1 - oracle.mean()/pb.mean()):+.1f}%)')


if __name__ == '__main__':
    main()

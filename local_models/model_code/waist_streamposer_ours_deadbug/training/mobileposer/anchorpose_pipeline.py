"""AnchorPose unified inference pipeline: ONE entry point that chains the
frozen MobilePoser backbone with the three external "reconcile two estimates
of the same quantity" modules.

    IMU --> backbone (frozen) --> pose (local rotmats), tran, pred joints
              |
              +-- OrientationFusionGRU : root orientation  (IMU  vs waist-camera SLAM heading)
              +-- CameraFusionGRU      : root translation   (IMU  vs waist-camera SLAM position)
              +-- SwingFusionGRU       : hip/knee swing     (Joints-module vs Poser-FK, no camera)

The three stages touch disjoint outputs (root rotation, root translation,
hip/knee local rotation), so they commute and can be toggled independently
for ablation. Creation Point 3 (trust weights correlate with real IMU error)
is read from the `weights` entries returned here.
"""
from pathlib import Path

import torch

from mobileposer.config import paths
from mobileposer.utils.model_utils import load_model
from mobileposer.articulate.model import ParametricModel
from mobileposer.camera_fusion import CameraFusionGRU
from mobileposer.orientation_fusion import OrientationFusionGRU
from mobileposer.joints_swing_fusion import (
    SwingFusionGRU, HIP_KNEE_JOINTS, build_rest_dir, compute_swing_twist_components,
)

DEFAULT_BASE = str(paths.root_dir / 'checkpoints/no_head_5imu_surface/wrists_shanks_waist/1/base_model.pth')


def _hidden(blob, default=32):
    return int(blob.get('manifest', {}).get('hidden', default))


class AnchorPosePipeline:
    def __init__(self, base_ckpt=DEFAULT_BASE,
                 cam_ckpt='runtime_camera_fusion_recal/last.pt',
                 ori_ckpt='runtime_orientation_fusion_dr/last.pt',
                 swing_ckpt='runtime_swing_fusion_v1/last.pt',
                 device=None):
        self.device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.model = load_model(base_ckpt).to(self.device).eval()

        blob = torch.load(cam_ckpt, map_location=self.device, weights_only=False)
        self.cam = CameraFusionGRU(hidden=_hidden(blob)).to(self.device)
        self.cam.load_state_dict(blob['cam_state_dict'] if 'cam_state_dict' in blob else blob['state_dict'])
        self.cam.eval()

        blob = torch.load(ori_ckpt, map_location=self.device, weights_only=False)
        self.ori = OrientationFusionGRU(hidden=_hidden(blob)).to(self.device)
        self.ori.load_state_dict(blob['state_dict'])
        self.ori.eval()

        blob = torch.load(swing_ckpt, map_location=self.device, weights_only=False)
        self.swing = SwingFusionGRU(hidden=_hidden(blob)).to(self.device)
        self.swing.load_state_dict(blob['state_dict'])
        self.swing.eval()

        self.bodymodel = ParametricModel(paths.smpl_file)
        self.rest_dir = build_rest_dir(self.bodymodel)

    @torch.no_grad()
    def __call__(self, imu, slam_pos=None, slam_ori=None, use_cam=True, use_ori=True, use_swing=True,
                 ori_mode='hold_global'):
        """imu: [T,60] (prepare_imu format). slam_pos: [T,3] in the same
        start-relative frame as the backbone's tran. slam_ori: [T,3,3] global
        root rotation. Stages whose SLAM input is missing are skipped.

        Returns dict(pose[T,24,3,3] local, tran[T,3], weights{cam,ori,swing},
        base_pose, base_tran) on CPU. base_* are the untouched backbone outputs."""
        dev = self.device
        self.model.reset()
        pose, pred_joints, tran, _ = self.model.forward_offline(imu.to(dev).unsqueeze(0), [imu.shape[0]])
        pose = pose.squeeze(0).cpu() if pose.dim() == 4 else pose.cpu()
        pose = pose.reshape(-1, 24, 3, 3)
        tran = tran.squeeze(0).cpu() if tran.dim() == 3 else tran.cpu()
        pred_joints = (pred_joints.squeeze(0) if pred_joints.dim() == 3 else pred_joints).cpu().view(-1, 24, 3)
        T = min(len(pose), len(tran), len(pred_joints))
        pose, tran, pred_joints = pose[:T], tran[:T], pred_joints[:T]
        out = {'base_pose': pose.clone(), 'base_tran': tran.clone(), 'weights': {}}
        pose = pose.clone()

        if use_swing:   # hip/knee local rotations; reads the UNFUSED backbone outputs
            R_sp, R_st, R_tp = compute_swing_twist_components(self.bodymodel, pose, pred_joints, self.rest_dir)
            corrected, w, _ = self.swing(R_sp.unsqueeze(0).to(dev), R_st.unsqueeze(0).to(dev), R_tp.unsqueeze(0).to(dev))
            pose[:, HIP_KNEE_JOINTS] = corrected.squeeze(0).cpu()
            out['weights']['swing'] = w.squeeze(0).cpu()

        if use_ori and slam_ori is not None:   # root global orientation
            fused, w, _ = self.ori(pose[:, 0].unsqueeze(0).to(dev), slam_ori[:T].unsqueeze(0).to(dev))
            fused = fused.squeeze(0).cpu()
            if ori_mode == 'hold_global':
                # The backbone predicts every limb's GLOBAL orientation from its own sensors; the root error
                # is not common-mode to them. Keep the children's global orientation fixed and re-express
                # their local rotation relative to the corrected root (SMPL root children: 1, 2, 3).
                delta = fused.transpose(-1, -2) @ pose[:, 0]          # R_new^T R_old
                for c in (1, 2, 3):
                    pose[:, c] = delta @ pose[:, c]
            elif ori_mode != 'propagate':
                raise ValueError(ori_mode)
            pose[:, 0] = fused
            out['weights']['ori'] = w.squeeze(0).cpu()

        if use_cam and slam_pos is not None:   # root translation
            fused, w, _ = self.cam(tran.unsqueeze(0).to(dev), slam_pos[:T].unsqueeze(0).to(dev))
            tran = fused.squeeze(0).cpu()
            out['weights']['cam'] = w.squeeze(0).cpu()

        out['pose'], out['tran'] = pose, tran
        return out

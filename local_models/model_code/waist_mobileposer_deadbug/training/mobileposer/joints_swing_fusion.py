"""Learned SLERP-style fusion of two independent SWING-rotation estimates for
each hip/knee joint, replacing the bilateral-leg residual-corrector approach
that was ruled out (see project_2026_novelty_search_history item 22: a signal
correlating with error does not imply it carries usable correction
direction).

This candidate is structurally different -- and directly mirrors
orientation_fusion.py's already-successful recipe (SO(3) geodesic blend
between two independently-produced rotation estimates of the SAME quantity),
just applied to a different pair of signals:
  - Poser's own implied SWING for the joint (the component of its predicted
    local rotation observable from joint position alone -- see
    analysis_joints_swing_correction_oracle.py for the swing-twist
    decomposition).
  - The SWING implied by the Joints module's independently-regressed child
    joint position (same frame, same backbone, different sub-network).
Twist (rotation about the bone's own axis) is NOT observable from either
position source, so it is always kept from Poser's own prediction unchanged.

Validated before training (analysis_joints_vs_poser_fk.py +
analysis_joints_swing_correction_oracle.py, 30 held-out AMASS long
sequences): naive "always apply Joints' implied swing" is a net loss
(-4.8% pooled, asymmetric across joints), but the ORACLE (always pick
whichever swing is closer to ground truth) is consistently positive across
all 4 hip/knee joints (+2.2% to +13.8%, pooled +8.4%) -- the same
"sometimes-A-sometimes-B, but learnable" pattern that made CameraFusionGRU
and OrientationFusionGRU work, not the all-or-nothing pattern that doomed
the bilateral-leg corrector.
"""
import torch
from torch import nn

from mobileposer.orientation_fusion import _so3_log, _so3_exp

JOINT_CHILD = {1: 4, 2: 5, 4: 7, 5: 8}    # L_hip->L_knee, R_hip->R_knee, L_knee->L_ankle, R_knee->R_ankle
JOINT_PARENT = {1: 0, 2: 0, 4: 1, 5: 2}
HIP_KNEE_JOINTS = list(JOINT_CHILD.keys())
N_JOINTS = len(HIP_KNEE_JOINTS)


def shortest_arc_rotation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[...,3] unit vectors -> [...,3,3] minimal rotation taking a to b."""
    v = torch.cross(a, b, dim=-1)
    c = (a * b).sum(-1, keepdim=True)
    s = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = v / s
    theta = torch.atan2(s, c)
    return _so3_exp(axis * theta)


def compute_swing_twist_components(bodymodel, pose_pred, pred_joints, rest_dir):
    """pose_pred: [T,24,3,3] local rotmats (frozen backbone's full predicted
    pose). pred_joints: [T,24,3] root-relative Joints-module positions.
    rest_dir: dict{joint: [3]} unit rest-pose bone directions (see
    analysis_joints_swing_correction_oracle.py for how to build this from
    bodymodel.get_zero_pose_joint_and_vertex).

    Returns (R_swing_poser, R_swing_target, R_twist_poser), each
    [T,len(HIP_KNEE_JOINTS),3,3], in HIP_KNEE_JOINTS order.
    """
    shape = torch.ones(10)
    pose_global_pred, poser_fk_joint = bodymodel.forward_kinematics(pose_pred, shape, tran=None, calc_mesh=False)
    swing_poser, swing_target, twist_poser = [], [], []
    for j in HIP_KNEE_JOINTS:
        c = JOINT_CHILD[j]
        R_parent_global = pose_global_pred[:, JOINT_PARENT[j]]
        obs_dir_joints = torch.nn.functional.normalize(pred_joints[:, c] - pred_joints[:, j], dim=-1)
        dir_poser = torch.nn.functional.normalize(poser_fk_joint[:, c] - poser_fk_joint[:, j], dim=-1)
        obs_local_joints = (R_parent_global.transpose(-1, -2) @ obs_dir_joints.unsqueeze(-1)).squeeze(-1)
        dir_local_poser = (R_parent_global.transpose(-1, -2) @ dir_poser.unsqueeze(-1)).squeeze(-1)

        rd = rest_dir[j].expand_as(obs_local_joints)
        R_sp = shortest_arc_rotation(rd, dir_local_poser)
        R_st = shortest_arc_rotation(rd, obs_local_joints)
        R_tp = R_sp.transpose(-1, -2) @ pose_pred[:, j]

        swing_poser.append(R_sp)
        swing_target.append(R_st)
        twist_poser.append(R_tp)
    return torch.stack(swing_poser, dim=1), torch.stack(swing_target, dim=1), torch.stack(twist_poser, dim=1)


def build_rest_dir(bodymodel):
    shape = torch.ones(10)
    zero_joint, _ = bodymodel.get_zero_pose_joint_and_vertex(shape)
    zero_joint = zero_joint[0]
    return {j: torch.nn.functional.normalize(zero_joint[c] - zero_joint[j], dim=-1) for j, c in JOINT_CHILD.items()}


class SwingFusionGRU(nn.Module):
    """R_swing_poser, R_swing_target: [B,T,4,3,3] -- the two independent swing
    estimates for (L_hip, R_hip, L_knee, R_knee), in that fixed order.
    R_twist_poser: [B,T,4,3,3] -- Poser's own twist component, always kept.
    Returns (corrected_hipknee[B,T,4,3,3], weight[B,T,4], state)."""

    def __init__(self, hidden=32):
        super().__init__()
        self.gru = nn.GRU(9 * N_JOINTS, hidden, batch_first=True)
        self.head = nn.Linear(hidden, N_JOINTS)
        nn.init.zeros_(self.head.bias)  # start near sigmoid(0)=0.5, no prior toward either source

    @staticmethod
    def _features(R_swing_poser, R_swing_target):
        poser_rel = torch.zeros_like(R_swing_poser[..., 0])
        poser_rel[:, 1:] = _so3_log(R_swing_poser[:, :-1].transpose(-1, -2) @ R_swing_poser[:, 1:])
        target_rel = torch.zeros_like(poser_rel)
        target_rel[:, 1:] = _so3_log(R_swing_target[:, :-1].transpose(-1, -2) @ R_swing_target[:, 1:])
        discrepancy = _so3_log(R_swing_poser.transpose(-1, -2) @ R_swing_target)
        feat = torch.cat([poser_rel, target_rel, discrepancy], dim=-1)   # [B,T,4,9]
        return feat.flatten(-2)                                          # [B,T,36]

    def forward(self, R_swing_poser, R_swing_target, R_twist_poser, state=None):
        if R_swing_poser.shape != R_swing_target.shape or R_swing_poser.shape[-3] != N_JOINTS:
            raise ValueError(f'Expected [B,T,{N_JOINTS},3,3] tensors')
        feat = self._features(R_swing_poser, R_swing_target)
        h, state = self.gru(feat, state)
        weight = torch.sigmoid(self.head(h))                 # [B,T,4], 1 = trust Poser's own swing
        blend = (1.0 - weight).unsqueeze(-1)                  # [B,T,4,1]
        rel_rotvec = _so3_log(R_swing_poser.transpose(-1, -2) @ R_swing_target)   # [B,T,4,3]
        step = _so3_exp(rel_rotvec * blend)                   # [B,T,4,3,3]
        fused_swing = R_swing_poser @ step
        corrected = fused_swing @ R_twist_poser
        return corrected, weight, state

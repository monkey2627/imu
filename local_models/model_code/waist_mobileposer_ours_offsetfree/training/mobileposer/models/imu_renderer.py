"""Differentiable IMU rendering from predicted global SMPL pose.

The benchmark stores world linear acceleration divided by ``amass.acc_scale``
and bone-calibrated sensor orientations.  This module deliberately keeps the
same conventions so it can be used as a consistency loss without changing the
existing model interface.
"""

import torch

from mobileposer.articulate import math as art_math


def _synth_acc(points, fps, smooth_n=4):
    """Differentiable equivalent of the project's AMASS IMU synthesis filter."""
    out = torch.zeros_like(points)
    T = points.shape[1]
    if T >= 3:
        out[:, 1:-1] = (points[:, 2:] - 2.0 * points[:, 1:-1] + points[:, :-2]) * float(fps) ** 2
    if T > smooth_n * 2:
        i = torch.arange(T - smooth_n * 2, device=points.device)
        smooth = (points[:, i + smooth_n * 2] + points[:, i] - 2.0 * points[:, i + smooth_n])
        smooth = smooth * (float(fps) ** 2 / smooth_n ** 2)
        # Match no_head_layouts._syn_acc: the stride-smooth samples are assigned
        # into the [smooth_n:-smooth_n] slice (length T-2*smooth_n).
        out = torch.cat((out[:, :smooth_n], smooth, out[:, T - smooth_n:]), dim=1)
    return out


def _so3_log(R):
    """Stable rotation-matrix logarithm, returning an axis-angle vector."""
    # The inputs are adjacent frames, so angles are small in normal operation.
    vee = torch.stack((R[..., 2, 1] - R[..., 1, 2],
                       R[..., 0, 2] - R[..., 2, 0],
                       R[..., 1, 0] - R[..., 0, 1]), dim=-1)
    half_sin = vee.norm(dim=-1) * 0.5
    half_cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.atan2(half_sin, half_cos)
    scale = torch.where(half_sin > 1e-5, theta / (2.0 * half_sin), torch.ones_like(theta) * 0.5)
    return vee * scale[..., None]


def render_imu(bodymodel, pose_local, labels, attachment_config, fps, shape=None):
    """Render ``(acc, ori)`` for a batch of local SMPL poses.

    Args:
        bodymodel: ``ParametricModel`` instance.
        pose_local: [B,T,24,3,3] local rotations.
        labels: sensor attachment names in benchmark order.
        attachment_config: loaded surface attachment config.
        fps: sequence rate.

    Returns:
        ``acc`` [B,T,N,3] in m/s^2 and ``ori`` [B,T,N,3,3].
    """
    if pose_local.ndim != 5 or pose_local.shape[-2:] != (3, 3):
        raise ValueError("pose_local must have shape [B,T,24,3,3]")
    B, T = pose_local.shape[:2]
    if shape is None:
        shape = pose_local.new_zeros(B, 10)
    if shape.ndim == 1:
        shape = shape.unsqueeze(0).expand(B, -1)
    flat_shape = shape[:, None, :].expand(B, T, 10).reshape(B * T, 10)
    flat_pose = pose_local.reshape(B * T, 24, 3, 3)
    zero_j, zero_v = bodymodel.get_zero_pose_joint_and_vertex(flat_shape)
    local_t = art_math.transformation_matrix(flat_pose, bodymodel.joint_position_to_bone_vector(zero_j))
    global_t = bodymodel.forward_kinematics_T(local_t)
    global_t = global_t.clone()
    global_t[..., -1:] -= torch.matmul(global_t, art_math.append_zero(zero_j, dim=-1).unsqueeze(-1))
    glob = global_t[..., :3, :3].reshape(B, T, 24, 3, 3)
    bone_ids = [attachment_config['attachments'][x]['joint'] for x in labels]
    mounts = pose_local.new_tensor([attachment_config['attachments'][x]['mount_rotation'] for x in labels])
    ori = glob[:, :, bone_ids] @ mounts
    # The stored benchmark orientation is bone-calibrated: mounting rotation is
    # removed after constructing the device frame.
    ori = ori @ mounts.transpose(-1, -2)
    ids = torch.tensor([attachment_config['attachments'][x]['vertices'] for x in labels], device=pose_local.device)
    bary = pose_local.new_tensor([attachment_config['attachments'][x]['weights'] for x in labels])
    skin = bodymodel._skinning_weights[ids.reshape(-1)]
    vertex_t = torch.einsum('bjxy,vj->bvxy', global_t, skin)
    vertex_ids = ids.reshape(-1)
    v_h = art_math.append_one(zero_v[:, vertex_ids], dim=-1)
    selected = torch.matmul(vertex_t, v_h.unsqueeze(-1)).squeeze(-1)[..., :3]
    selected = selected.reshape(B * T, len(labels), 3, 3)
    points = (selected * bary[None, :, :, None]).sum(2).reshape(B, T, len(labels), 3)
    acc = _synth_acc(points, fps)
    return acc, ori


def orientation_residual(pred_ori, obs_ori):
    """Geodesic residual in radians for valid observed rotations."""
    R = pred_ori.transpose(-1, -2) @ obs_ori
    vee = torch.stack((R[..., 2, 1] - R[..., 1, 2],
                       R[..., 0, 2] - R[..., 2, 0],
                       R[..., 1, 0] - R[..., 0, 1]), dim=-1)
    half_sin = vee.norm(dim=-1) * 0.5
    half_cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.atan2(half_sin, half_cos) * 2.0


def angular_velocity_residual(pred_ori, obs_ori, fps):
    """Compare finite-difference angular velocities in the world frame."""
    def velocity(R):
        rel = R[:, 1:].transpose(-1, -2) @ R[:, :-1]
        w = _so3_log(rel) * float(fps)
        return torch.cat((w[:, :1], w), dim=1)
    return velocity(pred_ori) - velocity(obs_ori)

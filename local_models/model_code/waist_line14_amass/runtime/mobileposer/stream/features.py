"""Pelvis-relative IMU features, root-relative targets, and the inverse map back to SMPL local rotations.

Sensor slots (pnp5 layout): 0 lw, 1 rw, 2 ls, 3 rs, 4 pelvis.  acc: world free acceleration (m/s^2), ori: bone-calibrated
global orientation (3x3).  All functions work on [..., T, 5, ...] tensors on any device.
"""
import torch
import torch.nn.functional as F

import mobileposer.articulate as art

ROOT = 4
ACC_SCALE = 30.0
GYRO_SCALE = 4.0
FPS = 60
REDUCED = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]      # predicted joints (PNP's reduced set, root excluded)
IGNORED = [0, 7, 8, 10, 11, 20, 21, 22, 23]
PARENT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
N_JOINT_OUT = 23 * 3
N_ROT_OUT = len(REDUCED) * 6


FRAMES = ('rel', 'world', 'both')       # pelvis-relative (default), world (as the windowed MobilePoser baseline), both


def n_features(gyro=True, frame='rel', local=False):
    rel = 36 + 3 + 3 + 12 + (15 if gyro else 0)
    world = 15 + 45 + (15 if gyro else 0)
    return {'rel': rel, 'world': world, 'both': rel + world}[frame] + (12 if local else 0)


def local_angular_velocity(ori):
    """angular velocity of each limb sensor in its OWN frame [B,T,12]: a hinge (elbow/knee) is a fixed direction here,
    so the per-wearing hinge geometry is directly visible to the network"""
    w = angular_velocity(ori)[:, :, :ROOT]                                            # world frame
    return (ori[:, :, :ROOT].transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1).flatten(2) / GYRO_SCALE


def angular_velocity(ori):
    """World-frame angular velocity from consecutive orientations, backward difference (causal).  ori [B,T,S,3,3] -> [B,T,S,3]"""
    dR = ori[:, 1:] @ ori[:, :-1].transpose(-1, -2)
    # log map of dR directly (vee of the skew part times theta/sin(theta)); far faster than the quaternion route
    v = torch.stack([dR[..., 2, 1] - dR[..., 1, 2], dR[..., 0, 2] - dR[..., 2, 0], dR[..., 1, 0] - dR[..., 0, 1]], -1) / 2
    cos = ((dR.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos)
    w = v * (theta / torch.sin(theta).clamp(min=1e-6)).unsqueeze(-1) * FPS
    return torch.cat([torch.zeros_like(w[:, :1]), w], 1)


def imu_features(acc, ori, gyro=True, frame='rel', local=False):
    """acc [B,T,5,3], ori [B,T,5,3,3] -> [B,T,F]: pelvis-relative and/or world-frame features (see FRAMES),
    plus (local=True) the limb sensors' angular velocity in their own frames, appended last."""
    if local:
        return torch.cat([imu_features(acc, ori, gyro, frame), local_angular_velocity(ori)], -1)
    B, T = acc.shape[:2]
    if frame != 'rel':
        world = [acc.reshape(B, T, 15) / ACC_SCALE, ori.reshape(B, T, 45)]
        if gyro:
            world.append(angular_velocity(ori).reshape(B, T, 15) / GYRO_SCALE)
        world = torch.cat(world, -1)
        if frame == 'world':
            return world
        return torch.cat([imu_features(acc, ori, gyro, 'rel'), world], -1)
    RrT = ori[:, :, ROOT].transpose(-1, -2)                                         # [B,T,3,3]
    ori_rel = RrT.unsqueeze(2) @ ori[:, :, :ROOT]                                     # [B,T,4,3,3]
    up = RrT[..., :, 1]                                                               # world +y in the root frame
    acc_root = (RrT @ acc[:, :, ROOT].unsqueeze(-1)).squeeze(-1) / ACC_SCALE
    acc_rel = (RrT.unsqueeze(2) @ (acc[:, :, :ROOT] - acc[:, :, ROOT:]).unsqueeze(-1)).squeeze(-1) / ACC_SCALE
    feats = [ori_rel.reshape(B, T, 36), up, acc_root, acc_rel.reshape(B, T, 12)]
    if gyro:
        w = angular_velocity(ori)
        feats.append((RrT.unsqueeze(2) @ w.unsqueeze(-1)).squeeze(-1).reshape(B, T, 15) / GYRO_SCALE)
    return torch.cat(feats, -1)


def targets(body, pose_local):
    """pose_local [N,24,3,3] -> root-relative joint positions [N,69] (root frame) and reduced root-relative global 6D [N,90]."""
    G, J = body.forward_kinematics(pose_local, calc_mesh=False)[:2]
    G, J = G.view(-1, 24, 3, 3), J.view(-1, 24, 3)
    R0T = G[:, 0].transpose(-1, -2)
    jrel = (R0T.unsqueeze(1) @ (J[:, 1:] - J[:, :1]).unsqueeze(-1)).squeeze(-1)
    grel = R0T.unsqueeze(1) @ G[:, REDUCED]
    return jrel.reshape(-1, N_JOINT_OUT), art.math.rotation_matrix_to_r6d(grel.reshape(-1, 3, 3)).view(-1, N_ROT_OUT)


def to_local_pose(body, r6d, root_ori):
    """r6d [T,90] root-relative reduced globals + measured pelvis orientation [T,3,3] -> SMPL local rotations [T,24,3,3]."""
    T = r6d.shape[0]
    grel = art.math.r6d_to_rotation_matrix(r6d.reshape(-1, 6)).view(T, len(REDUCED), 3, 3)
    G = torch.empty(T, 24, 3, 3, device=r6d.device)
    G[:, 0] = root_ori
    red = {j: k for k, j in enumerate(REDUCED)}
    for j in range(1, 24):
        G[:, j] = root_ori @ grel[:, red[j]] if j in red else G[:, PARENT[j]]
    return body.inverse_kinematics_R(G).view(T, 24, 3, 3)


# Anchored outputs: the parent of a sensed segment is predicted RELATIVE to that segment's measured orientation,
# G_parent = G_sensor (root-relative, measured) @ R_rel.  The calibration (RSB) aligns every dataset's sensors to its own GT
# bone frames, so the measured forearm/shank orientation carries the dataset's twist convention to the unsensed parent.
ANCHORS = {'none': {}, 'arms': {16: 0, 17: 1}, 'limbs': {16: 0, 17: 1, 1: 2, 2: 3}, 'armtw': {}}     # joint -> sensor slot
# 'armtw': the upper arm keeps only the DIRECTION of the network output; its twist is solved in closed form so that the
# elbow twist (twist of G_upper^T G_forearm about the upper-arm rest axis) equals an AMASS prior tau + a bounded residual.
TWIST_ARMS = ((16, 0), (17, 1))                                   # (joint, forearm sensor slot)
N_EXTRA = {'armtw': 2}


def min_rotation(a, d):
    """minimal rotation taking unit a [3] to unit d [...,3]"""
    a = a.expand_as(d)
    v = torch.cross(a, d, dim=-1)
    c = (a * d).sum(-1, keepdim=True)
    K = torch.zeros(*d.shape[:-1], 3, 3, device=d.device, dtype=d.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    K = K - K.transpose(-1, -2)
    return torch.eye(3, device=d.device) + K + K @ K / (1 + c).clamp(min=1e-4).unsqueeze(-1)


def twist_angle(R, a):
    """signed twist (rad) of R [...,3,3] about unit a: 2 atan2(vee(R - R^T).a, 1 + tr R)"""
    vee = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    return 2 * torch.atan2((vee * a).sum(-1), 1 + R.diagonal(dim1=-2, dim2=-1).sum(-1))


def rot_about(a, ang):
    """rotation by ang [...] about the fixed unit axis a [3] (Rodrigues)"""
    K = torch.zeros(3, 3, device=a.device)
    K[0, 1], K[0, 2], K[1, 2] = -a[2], a[1], -a[0]
    K = K - K.t()
    s, c = torch.sin(ang)[..., None, None], torch.cos(ang)[..., None, None]
    return torch.eye(3, device=a.device) + s * K + (1 - c) * (K @ K)


def elbow_twist(G_upper, G_forearm, a):
    return twist_angle(G_upper.transpose(-1, -2) @ G_forearm, a)


def root_relative_ori(ori):
    """ori [B,T,5,3,3] -> limb sensor orientations relative to the pelvis sensor [B,T,4,3,3]"""
    return ori[:, :, ROOT].transpose(-1, -2).unsqueeze(2) @ ori[:, :, :ROOT]


def compose(out, ori_rel, anchor, axes=None, tau=None, bound=0.0):
    """network output [B,T,90(+extra)] + measured ori_rel [B,T,4,3,3] -> root-relative global 6D [B,T,90].
    axes [2,3] upper-arm rest axes, tau [2] elbow-twist priors (rad), bound: residual range (rad) for 'armtw'."""
    r6d = out[..., :N_ROT_OUT]
    if anchor == 'none':
        return r6d
    B, T = r6d.shape[:2]
    R = art.math.r6d_to_rotation_matrix(r6d.reshape(-1, 6)).view(B, T, len(REDUCED), 3, 3)
    parts = list(R.unbind(2))
    for j, slot in ANCHORS[anchor].items():
        k = REDUCED.index(j)
        parts[k] = ori_rel[:, :, slot] @ parts[k]
    if anchor == 'armtw':
        extra = out[..., N_ROT_OUT:]
        for i, (j, slot) in enumerate(TWIST_ARMS):
            k, a = REDUCED.index(j), axes[i]
            S = min_rotation(a, parts[k] @ a)                                           # direction from the network
            target = tau[i] + bound * torch.tanh(extra[..., i])
            psi = twist_angle(S.transpose(-1, -2) @ ori_rel[:, :, slot], a) - target   # left twist about a adds exactly
            parts[k] = S @ rot_about(a, psi)
    R = torch.stack(parts, 2)
    return art.math.rotation_matrix_to_r6d(R.reshape(-1, 3, 3)).view(B, T, N_ROT_OUT)


def random_mount(n, sigma_deg, device):
    """[n,5,3,3] random constant sensor-to-bone offsets (axis uniform, angle ~ |N(0, sigma)|)."""
    axis = F.normalize(torch.randn(n, 5, 3, device=device), dim=-1)
    ang = torch.randn(n, 5, 1, device=device).abs() * sigma_deg * torch.pi / 180
    return art.math.axis_angle_to_rotation_matrix((axis * ang).view(-1, 3)).view(n, 5, 3, 3)

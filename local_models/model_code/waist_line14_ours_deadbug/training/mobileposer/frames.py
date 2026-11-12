"""Input frames for the IMU network.
world : absolute orientations / accelerations in the world frame (MobilePoser default)
pelvis: everything expressed in the PELVIS sensor's frame (the encoding of TransPose / PIP / PNP: heading AND tilt removed)
yaw   : everything expressed in the gravity-aligned HEADING frame of the pelvis: only the yaw (rotation about the vertical
        axis) is removed, the tilt - hence the direction of gravity, which separates standing / sitting / lying - stays in the data.
Targets (global rotations, joint positions) are expressed in the same frame; the predicted root is rotated back by Rf."""
import os

import torch


def frame_kind():
    k = os.environ.get('MOBILEPOSER_FRAME')
    if not k and os.environ.get('MOBILEPOSER_CANON_WAIST') == '1':
        k = 'pelvis'
    return k or 'world'


def frame_slot():
    return int(os.environ.get('MOBILEPOSER_FRAME_SLOT', 4 if int(os.environ.get('MOBILEPOSER_NSENS', 5)) == 5 else 5))


def frame_rot(R, kind):
    """R [T,3,3] pelvis-sensor orientation -> Rf [T,3,3] (columns = the frame axes in world coordinates)"""
    if kind == 'pelvis':
        return R
    if kind == 'yaw':
        f, r = R[:, :, 2], R[:, :, 0]                                   # pelvis forward (+z) and right (+x) axes in the world frame
        use_f = (f[:, [0, 2]].norm(dim=-1) > 0.3)
        yaw_f = torch.atan2(f[:, 0], f[:, 2])
        yaw_r = torch.atan2(r[:, 0], r[:, 2]) - torch.pi / 2             # lying on the back/front: the forward axis is vertical
        yaw = torch.where(use_f, yaw_f, yaw_r)
        c, s = torch.cos(yaw), torch.sin(yaw)
        Rf = torch.zeros_like(R)
        Rf[:, 0, 0], Rf[:, 0, 2], Rf[:, 1, 1], Rf[:, 2, 0], Rf[:, 2, 2] = c, s, 1.0, -s, c
        return Rf
    return torch.eye(3, dtype=R.dtype, device=R.device).expand_as(R)

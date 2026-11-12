"""Realistic (randomized) calibration sequences for line 9.

Unlike the original fixed P0-P6 protocol (identical trajectory, perfect nominal pose, fixed world heading, which lets an
encoder learn the constant ``R_bone`` and read ``R_off = R_bone^T R_meas`` from one frame), every sample here has
  * constant + slowly varying joint-pose deviations from the nominal protocol (``dev_deg`` RMS per joint),
  * per-phase motion amplitude scaling (0.75-1.25) and a random monotone time warp,
  * a random global heading (yaw) applied to the root.
Heading is removed from the encoder input by ``heading_normalize`` (pelvis-sensor yaw at frame 0).  The yaw offset of the
pelvis sensor itself is unobservable against heading, so it is defined away (``identity_offsets(..., gauge=True)``).
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

from mobileposer.articulate import math as AM
from mobileposer.line9.calib import identity_offsets, synth_sequence
from mobileposer.line9.fixed_protocol import make_protocol_pose

STRIDE = 4


def _gen(key: str, tag: str, seed: int = 20261013) -> torch.Generator:
    d = hashlib.sha256(f"{key}|{tag}|{seed}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(d[:8], "little") % (2**63))


def ry(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta), torch.sin(theta)
    z, o = torch.zeros_like(theta), torch.ones_like(theta)
    return torch.stack((torch.stack((c, z, s), -1), torch.stack((z, o, z), -1), torch.stack((-s, z, c), -1)), -2)


def randomized_pose(key: str, dev_deg: float, heading: float | None = None):
    """Return (pose matrices [T,24,3,3], phase_id [T], heading psi)."""
    proto = make_protocol_pose()
    pose, phase = proto.pose_aa.clone(), proto.phase_id.clone()
    T = len(phase)
    g = _gen(key, "proto")
    base = pose[0].clone()
    amp = 0.75 + 0.5 * torch.rand(7, generator=g)
    pose = base[None] + amp[phase][:, None, None] * (pose - base[None])
    inc = torch.exp(0.35 * torch.randn(12, generator=g))
    cum = torch.cat((torch.zeros(1), torch.cumsum(inc, 0)))
    cum = cum / cum[-1]
    s = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, 13), cum.numpy()) * (T - 1)
    i0 = np.floor(s).astype(int).clip(0, T - 2)
    w = torch.tensor(s - i0, dtype=torch.float32)
    pose = pose[i0] * (1 - w)[:, None, None] + pose[i0 + 1] * w[:, None, None]
    phase = phase[torch.tensor(np.rint(s).astype(int).clip(0, T - 1))]
    # dev_deg is the target RMS deviation of the *global sensor-bone orientation* at the stance; the chain pelvis->wrist
    # accumulates ~8 joint deviations, so the per-joint RMS is dev_deg / 3.
    sigma = math.radians(dev_deg) / 3.0 / math.sqrt(3.0)
    const = torch.randn(24, 3, generator=g) * sigma
    slow = torch.randn(1, 72, 8, generator=g) * sigma * 0.5
    slow = torch.nn.functional.interpolate(slow, size=T, mode="linear", align_corners=True)[0].T.reshape(T, 24, 3)
    dev = const[None] + slow
    dev[:, 22:] = 0.0
    R = AM.axis_angle_to_rotation_matrix(pose.reshape(-1, 3)).reshape(T, 24, 3, 3)
    D = AM.axis_angle_to_rotation_matrix(dev.reshape(-1, 3)).reshape(T, 24, 3, 3)
    R = R @ D
    psi = float((torch.rand(1, generator=g) * 2 - 1) * math.pi) if heading is None else float(heading)
    R[:, 0] = ry(torch.tensor(psi)) @ R[:, 0]
    return R, phase, psi


@torch.no_grad()
def synth_random_calibration(model, config, key, beta, offset_key, rot_deg, lever_cm, dev_deg, heading=None):
    """One randomized calibration sample -> dict with stride-4 raw imu [T/4,60], phase_id, R_off, psi."""
    R, phase, psi = randomized_pose(key, dev_deg, heading)
    R_off, lever, _ = identity_offsets(offset_key, rot_deg, lever_cm, gauge=True)
    beta = torch.as_tensor(beta, dtype=torch.float32).reshape(10)
    acc, ori = synth_sequence(model, config, R, beta, torch.zeros(len(R), 3), R_off, lever)
    imu = torch.cat((acc.flatten(1) / 30.0, ori.flatten(1)), 1)[::STRIDE].contiguous()
    return {"imu": imu, "phase_id": phase[::STRIDE].clone(), "rotation": R_off, "psi": psi}


def heading_normalize(imu: torch.Tensor) -> torch.Tensor:
    """Rotate a calibration sequence [...,T,60] about the vertical so the pelvis sensor frame-0 yaw is zero."""
    lead = imu.shape[:-1]
    acc = imu[..., :15].reshape(*lead, 5, 3)
    ori = imu[..., 15:].reshape(*lead, 5, 3, 3)
    R0 = ori[..., 0, 4, :, :]
    psi = torch.atan2(R0[..., 0, 2], R0[..., 2, 2])
    M = ry(-psi)[..., None, None, :, :]
    acc2 = (M @ acc[..., None]).squeeze(-1)
    ori2 = M @ ori
    return torch.cat((acc2.reshape(*lead, 15), ori2.reshape(*lead, 45)), -1)

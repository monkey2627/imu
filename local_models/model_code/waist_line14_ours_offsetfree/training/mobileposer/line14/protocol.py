"""Dynamic (fast) calibration protocol for line 14 and a protocol-parametrised randomised synthesis.

The line-9 P0-P6 protocol is slow (phases of 3-6 s, ~2 cycles).  The velocity/acceleration of the limbs (and hence how well the
rotation centres, i.e. the sensor->joint vectors, are constrained by a = alpha x r + omega x (omega x r)) scales with frequency, so
here every excitation phase is a fast swing of 1.2-2.5 Hz that lasts ~2 s:

  D0 stand (1 s) | D1 arm flex/ext swing | D2 arm abduction swing | D3 fast elbow flexion (T pose)
  | D4 fast squats | D5 fast leg swing/kick | D6 fast march          (13 s in total, same 7 phase ids as P0-P6)
"""
from __future__ import annotations

import math

import numpy as np
import torch

from mobileposer.articulate import math as AM
from mobileposer.line9.calib import identity_offsets, synth_sequence
from mobileposer.line9.fixed_protocol import JOINTS, FPS, Protocol
from mobileposer.line9.robust_calib import _gen, ry

DYN_DURATIONS = (1, 2, 2, 2, 2, 2, 2)


def _env(n):
    u = torch.linspace(0.0, 1.0, n)
    return torch.sin(math.pi * u) ** 2           # smooth start/stop: returns exactly to the neutral pose


def _arms_down(n):
    out = torch.zeros(n, 24, 3)
    out[:, JOINTS["lshoulder"], 2] = -1.15
    out[:, JOINTS["rshoulder"], 2] = 1.15
    return out


def _seg_d0(n):
    return _arms_down(n)


def _seg_d1(n, f=1.5, amp=0.9):                  # shoulder flexion/extension swing (x axis)
    out = _arms_down(n); u = torch.linspace(0.0, 1.0, n); T = n / FPS
    w = amp * _env(n) * torch.sin(2 * math.pi * f * T * u)
    out[:, JOINTS["lshoulder"], 0] = w; out[:, JOINTS["rshoulder"], 0] = w
    return out


def _seg_d2(n, f=1.5, amp=0.55):                 # shoulder abduction swing around the arms-down baseline (z axis)
    out = _arms_down(n); u = torch.linspace(0.0, 1.0, n); T = n / FPS
    w = amp * _env(n) * torch.sin(2 * math.pi * f * T * u)
    out[:, JOINTS["lshoulder"], 2] += w; out[:, JOINTS["rshoulder"], 2] -= w
    return out


def _seg_d3(n, f=1.5, amp=0.95):                 # arms to T pose, fast elbow flexion
    out = torch.zeros(n, 24, 3); u = torch.linspace(0.0, 1.0, n); T = n / FPS
    edge = 0.15
    sh = torch.where(u < edge, 0.5 - 0.5 * torch.cos(math.pi * u / edge),
                     torch.where(u > 1 - edge, 0.5 - 0.5 * torch.cos(math.pi * (1 - u) / edge), torch.ones_like(u)))
    out[:, JOINTS["lshoulder"], 2] = -1.15 * (1 - sh); out[:, JOINTS["rshoulder"], 2] = 1.15 * (1 - sh)
    central = ((u - edge) / (1 - 2 * edge)).clamp(0, 1)
    bend = amp * (0.5 - 0.5 * torch.cos(2 * math.pi * f * T * (1 - 2 * edge) * central)) * sh
    out[:, JOINTS["lelbow"], 2] = bend; out[:, JOINTS["relbow"], 2] = -bend
    return out


def _seg_d4(n, f=1.2):                           # fast squats
    out = _arms_down(n); u = torch.linspace(0.0, 1.0, n); T = n / FPS
    s = (0.5 - 0.5 * torch.cos(2 * math.pi * f * T * u)) * _env(n).sqrt()
    out[:, JOINTS["lhip"], 0] = -0.62 * s; out[:, JOINTS["rhip"], 0] = -0.62 * s
    out[:, JOINTS["lknee"], 0] = 1.15 * s; out[:, JOINTS["rknee"], 0] = 1.15 * s
    return out


def _seg_d5(n, f=1.5):                           # fast alternating leg swing/kick
    out = _arms_down(n); u = torch.linspace(0.0, 1.0, n); T = n / FPS
    c = 2 * math.pi * f * T * u
    l = 0.85 * torch.relu(torch.sin(c)) * _env(n); r = 0.85 * torch.relu(torch.sin(c + math.pi)) * _env(n)
    out[:, JOINTS["lhip"], 0] = -0.45 * l; out[:, JOINTS["rhip"], 0] = -0.45 * r
    out[:, JOINTS["lknee"], 0] = 1.25 * l; out[:, JOINTS["rknee"], 0] = 1.25 * r
    return out


def _seg_d6(n, f=2.5):                           # fast march
    out = _arms_down(n); u = torch.linspace(0.0, 1.0, n); T = n / FPS
    c = 2 * math.pi * f * T * u
    l = 0.45 * torch.relu(torch.sin(c)) * _env(n); r = 0.45 * torch.relu(torch.sin(c + math.pi)) * _env(n)
    out[:, JOINTS["lhip"], 0] = -l; out[:, JOINTS["rhip"], 0] = -r
    out[:, JOINTS["lknee"], 0] = 0.9 * l; out[:, JOINTS["rknee"], 0] = 0.9 * r
    return out


def make_dynamic_protocol(durations=DYN_DURATIONS) -> Protocol:
    makers = (_seg_d0, _seg_d1, _seg_d2, _seg_d3, _seg_d4, _seg_d5, _seg_d6)
    pieces, phases = [], []
    for i, (m, sec) in enumerate(zip(makers, durations)):
        n = int(sec) * FPS
        pieces.append(m(n)); phases.append(torch.full((n,), i, dtype=torch.long))
    return Protocol(torch.cat(pieces).float(), torch.cat(phases), FPS)


def randomized_pose_p(proto: Protocol, key: str, dev_deg: float, heading: float | None = None, amp_scale: float = 1.0):
    """Same randomisation as mobileposer.line9.robust_calib.randomized_pose but for an arbitrary 7-phase protocol."""
    pose, phase = proto.pose_aa.clone(), proto.phase_id.clone()
    T = len(phase)
    g = _gen(key, "proto")
    base = pose[0].clone()
    amp = (0.75 + 0.5 * torch.rand(7, generator=g)) * amp_scale
    pose = base[None] + amp[phase][:, None, None] * (pose - base[None])
    inc = torch.exp(0.35 * torch.randn(12, generator=g))
    cum = torch.cat((torch.zeros(1), torch.cumsum(inc, 0)))
    cum = cum / cum[-1]
    s = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, 13), cum.numpy()) * (T - 1)
    i0 = np.floor(s).astype(int).clip(0, T - 2)
    w = torch.tensor(s - i0, dtype=torch.float32)
    pose = pose[i0] * (1 - w)[:, None, None] + pose[i0 + 1] * w[:, None, None]
    phase = phase[torch.tensor(np.rint(s).astype(int).clip(0, T - 1))]
    sigma = math.radians(dev_deg) / 3.0 / math.sqrt(3.0)
    const = torch.randn(24, 3, generator=g) * sigma
    slow = torch.randn(1, 72, 8, generator=g) * sigma * 0.5
    slow = torch.nn.functional.interpolate(slow, size=T, mode="linear", align_corners=True)[0].T.reshape(T, 24, 3)
    dev = const[None] + slow
    dev[:, 22:] = 0.0
    R = AM.axis_angle_to_rotation_matrix(pose.reshape(-1, 3) + 1e-6).reshape(T, 24, 3, 3)   # +eps: exact-zero axis-angle gives NaN
    D = AM.axis_angle_to_rotation_matrix(dev.reshape(-1, 3)).reshape(T, 24, 3, 3)
    R = R @ D
    psi = float((torch.rand(1, generator=g) * 2 - 1) * math.pi) if heading is None else float(heading)
    R[:, 0] = ry(torch.tensor(psi)) @ R[:, 0]
    return R, phase, psi


@torch.no_grad()
def synth_random_calibration_p(model, config, proto, key, beta, offset_key, rot_deg, lever_cm, dev_deg, stride=4, heading=None, amp_scale=1.0):
    R, phase, psi = randomized_pose_p(proto, key, dev_deg, heading, amp_scale)
    R_off, lever, _ = identity_offsets(offset_key, rot_deg, lever_cm, gauge=True)
    beta = torch.as_tensor(beta, dtype=torch.float32).reshape(10)
    acc, ori = synth_sequence(model, config, R, beta, torch.zeros(len(R), 3), R_off, lever)
    imu = torch.cat((acc.flatten(1) / 30.0, ori.flatten(1)), 1)[::stride].contiguous()
    return {"imu": imu, "phase_id": phase[::stride].clone(), "rotation": R_off, "psi": psi}

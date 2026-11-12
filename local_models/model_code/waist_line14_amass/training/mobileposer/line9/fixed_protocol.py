"""Programmatic P0--P6 calibration protocol for line 9.

The protocol is deliberately generated in SMPL *local* axis-angle space.  It
does not use AMASS motion labels, and every identity receives the same motion
timing while its beta and fixed sensor offsets are kept separate.  The output
of :func:`synthesize_protocol` is the same 60-channel representation used by
``line9.data``: five 3-D accelerations divided by 30, followed by five 3x3
bone-calibrated orientation matrices.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch

from mobileposer.articulate import math
from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from mobileposer.line9.calib import LABELS, CONFIG, identity_offsets, synth_sequence
from mobileposer.surface_imu import load_attachments


FPS = 60
DEFAULT_DURATIONS = (3, 6, 6, 6, 6, 5, 5)  # P0 ... P6, 37 s at 60 fps
PHASE_NAMES = ("P0_stand", "P1_arms", "P2_elbows", "P3_shoulders", "P4_squat", "P5_knee", "P6_march")
JOINTS = {"pelvis": 0, "lhip": 1, "rhip": 2, "lknee": 4, "rknee": 5,
          "lshoulder": 16, "rshoulder": 17, "lelbow": 18, "relbow": 19}


@dataclass(frozen=True)
class Protocol:
    pose_aa: torch.Tensor       # [T, 24, 3], local SMPL axis-angle, radians
    phase_id: torch.Tensor      # [T], values 0..6
    fps: int = FPS

    @property
    def frames(self) -> int:
        return int(self.pose_aa.shape[0])

    @property
    def seconds(self) -> float:
        return self.frames / float(self.fps)


def _smooth(n: int, lo: float, hi: float, cycles: float = 1.0) -> torch.Tensor:
    """Cosine ramp with zero velocity at both ends."""
    if n < 1:
        return torch.empty(0)
    if n == 1:
        return torch.tensor([hi], dtype=torch.float32)
    u = torch.linspace(0.0, 1.0, n)
    wave = 0.5 - 0.5 * torch.cos(2.0 * np.pi * cycles * u)
    if cycles == 1.0:
        wave = 0.5 - 0.5 * torch.cos(np.pi * u)
    return lo + (hi - lo) * wave


def _hold(n: int) -> torch.Tensor:
    return torch.zeros(n, dtype=torch.float32)


def _put(seg: torch.Tensor, joint: int, axis: int, values: torch.Tensor) -> None:
    seg[:, joint, axis] = values


def _segment_p0(n: int) -> torch.Tensor:
    out = torch.zeros(n, 24, 3)
    _put(out, JOINTS["lshoulder"], 2, torch.full((n,), -1.15))
    _put(out, JOINTS["rshoulder"], 2, torch.full((n,), 1.15))
    return out


def _segment_p1(n: int) -> torch.Tensor:
    # SMPL rest shoulders are close to a T-pose.  Opposite z signs account for
    # the mirrored local frames of the left/right shoulder joints.
    out = torch.zeros(n, 24, 3)
    down_l, down_r = -1.15, 1.15
    u = torch.linspace(0.0, 1.0, n)
    lift = 0.5 - 0.5 * torch.cos(4.0 * np.pi * u)
    vals = down_l * (1.0 - lift)
    _put(out, JOINTS["lshoulder"], 2, vals)
    vals = down_r * (1.0 - lift)
    _put(out, JOINTS["rshoulder"], 2, vals)
    return out


def _segment_p2(n: int) -> torch.Tensor:
    out = torch.zeros(n, 24, 3)
    # Enter/leave T-pose smoothly; flex elbows in the central interval.
    u = torch.linspace(0.0, 1.0, n)
    edge = 0.15
    shoulder = torch.where(u < edge, 0.5 - 0.5 * torch.cos(np.pi * u / edge),
                           torch.where(u > 1.0 - edge,
                                       0.5 - 0.5 * torch.cos(np.pi * (1.0 - u) / edge),
                                       torch.ones_like(u)))
    _put(out, JOINTS["lshoulder"], 2, -1.15 * (1.0 - shoulder))
    _put(out, JOINTS["rshoulder"], 2, 1.15 * (1.0 - shoulder))
    central = ((u - edge) / (1.0 - 2.0 * edge)).clamp(0.0, 1.0)
    bend = 0.95 * (0.5 - 0.5 * torch.cos(2.0 * np.pi * 3.0 * central)) * shoulder
    _put(out, JOINTS["lelbow"], 2, bend)
    _put(out, JOINTS["relbow"], 2, -bend)
    return out


def _segment_p3(n: int) -> torch.Tensor:
    out = torch.zeros(n, 24, 3)
    u = torch.linspace(0.0, 1.0, n)
    # Windowed circles excite two axes and return exactly to arms-down.
    envelope = torch.sin(np.pi * u) ** 2
    z = 0.55 * envelope * torch.sin(2.0 * np.pi * 2.0 * u)
    x = 0.34 * envelope * torch.cos(2.0 * np.pi * 2.0 * u)
    _put(out, JOINTS["lshoulder"], 2, -1.15 + z)
    _put(out, JOINTS["rshoulder"], 2, 1.15 - z)
    _put(out, JOINTS["lshoulder"], 0, x); _put(out, JOINTS["rshoulder"], 0, x)
    return out


def _segment_p4(n: int) -> torch.Tensor:
    out = _segment_p0(n)
    u = torch.linspace(0.0, 1.0, n)
    squat = 0.62 * (0.5 - 0.5 * torch.cos(2.0 * np.pi * 2.0 * u))
    knee = 1.15 * (0.5 - 0.5 * torch.cos(2.0 * np.pi * 2.0 * u))
    _put(out, JOINTS["lhip"], 0, -squat); _put(out, JOINTS["rhip"], 0, -squat)
    _put(out, JOINTS["lknee"], 0, knee); _put(out, JOINTS["rknee"], 0, knee)
    return out


def _segment_p5(n: int) -> torch.Tensor:
    out = _segment_p0(n)
    u = torch.linspace(0.0, 1.0, n)
    cycles = 2.0 * np.pi * 2.0 * u
    left = 0.85 * torch.relu(torch.sin(cycles))
    right = 0.85 * torch.relu(torch.sin(cycles + np.pi))
    _put(out, JOINTS["lhip"], 0, -0.45 * left); _put(out, JOINTS["rhip"], 0, -0.45 * right)
    _put(out, JOINTS["lknee"], 0, 1.25 * left); _put(out, JOINTS["rknee"], 0, 1.25 * right)
    return out


def _segment_p6(n: int) -> torch.Tensor:
    out = _segment_p0(n)
    u = torch.linspace(0.0, 1.0, n)
    cycles = 2.0 * np.pi * 2.5 * u
    left = 0.45 * torch.relu(torch.sin(cycles))
    right = 0.45 * torch.relu(torch.sin(cycles + np.pi))
    _put(out, JOINTS["lhip"], 0, -left); _put(out, JOINTS["rhip"], 0, -right)
    _put(out, JOINTS["lknee"], 0, 0.9 * left); _put(out, JOINTS["rknee"], 0, 0.9 * right)
    return out


def make_protocol_pose(fps: int = FPS, durations=DEFAULT_DURATIONS) -> Protocol:
    """Return the fixed P0--P6 local-axis-angle protocol and integer phase IDs."""
    if int(fps) != FPS:
        raise ValueError("the line-9 fixed protocol is defined at 60 Hz")
    if len(durations) != 7 or any(int(x) < 1 for x in durations):
        raise ValueError("durations must contain seven positive phase durations")
    makers = (_segment_p0, _segment_p1, _segment_p2, _segment_p3, _segment_p4, _segment_p5, _segment_p6)
    pieces, phases = [], []
    for i, (maker, seconds) in enumerate(zip(makers, durations)):
        frames = int(seconds) * FPS
        pieces.append(maker(frames)); phases.append(torch.full((frames,), i, dtype=torch.long))
    pose = torch.cat(pieces, dim=0).float()
    phase = torch.cat(phases, dim=0)
    return Protocol(pose, phase, FPS)


def validate_protocol(protocol: Protocol, model: Optional[ParametricModel] = None,
                      beta: Optional[torch.Tensor] = None) -> dict:
    """Validate finite local rotations, phase boundaries, and beta-dependent lengths."""
    if protocol.pose_aa.ndim != 3 or tuple(protocol.pose_aa.shape[1:]) != (24, 3):
        raise ValueError("pose_aa must have shape [T,24,3]")
    if protocol.phase_id.shape != (protocol.frames,) or set(protocol.phase_id.tolist()) != set(range(7)):
        raise ValueError("phase_id must cover P0..P6 exactly")
    if not torch.isfinite(protocol.pose_aa).all():
        raise ValueError("non-finite protocol pose")
    # Every phase must be contiguous and non-empty.
    transitions = torch.where(protocol.phase_id[1:] != protocol.phase_id[:-1])[0].tolist()
    if len(transitions) != 6 or protocol.phase_id[0] != 0 or protocol.phase_id[-1] != 6:
        raise ValueError("phase IDs are not contiguous P0..P6")
    boundary_jump = max(float((protocol.pose_aa[i + 1] - protocol.pose_aa[i]).norm(dim=-1).max()) for i in transitions)
    if boundary_jump > 1e-4:
        raise ValueError(f"discontinuous phase boundary: {boundary_jump:.6g} rad")
    out = {"frames": protocol.frames, "seconds": protocol.seconds, "phase_boundary_jump_max_rad": boundary_jump, "phase_counts": {
        PHASE_NAMES[i]: int((protocol.phase_id == i).sum()) for i in range(7)}}
    if model is not None:
        shape = torch.zeros(10) if beta is None else torch.as_tensor(beta, dtype=torch.float32).reshape(10)
        j, _ = model.get_zero_pose_joint_and_vertex(shape)
        lengths = []
        for child, parent in enumerate(model.parent):
            if parent is not None:
                lengths.append(float((j[0, child] - j[0, parent]).norm()))
        out["bone_length_mean_m"] = float(np.mean(lengths))
        out["bone_length_min_m"] = float(np.min(lengths))
        if not (out["bone_length_min_m"] > 0):
            raise ValueError("invalid beta-dependent bone length")
    return out


def session_offsets(identity: str, session_id, rot_deg: float, lever_cm: float, data_seed: int = 20261013):
    """Deterministic installation parameters, independent across sessions of an identity."""
    key = f"{identity}::session::{session_id}"
    return identity_offsets(key, rot_deg, lever_cm, data_seed)


@torch.no_grad()
def synthesize_protocol(identity: str, beta, *, session_id: int | str = 0, rot_deg: float = 5.0, lever_cm: float = 3.0,
                        model: Optional[ParametricModel] = None, config=None,
                        protocol: Optional[Protocol] = None) -> dict:
    """Synthesize one identity's fixed calibration IMU and phase IDs."""
    model = model or ParametricModel(paths.smpl_file)
    config = config or load_attachments(CONFIG, model)
    protocol = protocol or make_protocol_pose()
    beta = torch.as_tensor(beta, dtype=torch.float32).reshape(10)
    validate_protocol(protocol, model, beta)
    pose = math.axis_angle_to_rotation_matrix(protocol.pose_aa).reshape(-1, 24, 3, 3)
    tran = torch.zeros(protocol.frames, 3)
    R_off, lever, rotvec = session_offsets(identity, session_id, rot_deg, lever_cm)
    acc, ori = synth_sequence(model, config, pose, beta, tran, R_off, lever)
    imu = torch.cat((acc.flatten(1) / 30.0, ori.flatten(1)), dim=1).float()
    digest = hashlib.sha256(protocol.pose_aa.numpy().tobytes() + protocol.phase_id.numpy().tobytes()).hexdigest()
    acc_norm = acc.norm(dim=-1).flatten()
    return {"imu": imu, "phase_id": protocol.phase_id.clone(), "pose_aa": protocol.pose_aa.clone(),
            "beta": beta.clone(), "identity": identity, "session_id": session_id, "fps": FPS,
            "rot_off": R_off, "rotvec_off": rotvec, "lever": lever,
            "protocol": {"phase_names": PHASE_NAMES, "durations_s": list(DEFAULT_DURATIONS),
                         "digest_sha256": digest, "acc_norm_p99_m_s2": float(torch.quantile(acc_norm, 0.99)),
                         "rot_deg_rms": rot_deg, "lever_cm": lever_cm}}


@torch.no_grad()
def validate_zero_offset(identity: str, beta, *, protocol: Optional[Protocol] = None) -> dict:
    """Check that the zero-offset path is finite and preserves protocol shape/lengths."""
    model = ParametricModel(paths.smpl_file)
    config = load_attachments(CONFIG, model)
    protocol = protocol or make_protocol_pose()
    beta = torch.as_tensor(beta, dtype=torch.float32).reshape(10)
    pose = math.axis_angle_to_rotation_matrix(protocol.pose_aa).reshape(-1, 24, 3, 3)
    zero_R = torch.eye(3).repeat(5, 1, 1)
    zero_l = torch.zeros(5, 3)
    acc, ori = synth_sequence(model, config, pose, beta, torch.zeros(protocol.frames, 3), zero_R, zero_l)
    finite = bool(torch.isfinite(acc).all() and torch.isfinite(ori).all())
    return {"finite": finite, "frames": int(len(acc)), "imu_channels": 60,
            "orientation_orthogonality_max": float((ori @ ori.transpose(-1, -2) - torch.eye(3)).abs().max())}


def save_protocol(output: str | Path, identity: str, beta, **kwargs) -> Path:
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(synthesize_protocol(identity, beta, **kwargs), output)
    return output


__all__ = ["Protocol", "make_protocol_pose", "validate_protocol", "synthesize_protocol",
           "validate_zero_offset", "session_offsets", "save_protocol", "PHASE_NAMES", "DEFAULT_DURATIONS"]

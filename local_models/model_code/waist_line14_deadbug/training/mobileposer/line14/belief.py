"""Causal joint belief-state model for calibration-free PNP5 pose recovery.

The model keeps sensor-frame relationship as a slowly varying latent belief,
not as a deterministic pre-processing correction.  It never consumes a
future frame and exposes uncertainty for the latent sensor state.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from mobileposer.articulate import math as M


class CausalBeliefFilter(nn.Module):
    """Recurrent state-space approximation with pose and IMU observation heads."""

    def __init__(self, hidden: int = 256, sensor_latent: int = 15, pose_dim: int = 96):
        super().__init__()
        if min(hidden, sensor_latent, pose_dim) <= 0:
            raise ValueError("dimensions must be positive")
        self.hidden = int(hidden)
        self.sensor_latent = int(sensor_latent)
        self.pose_dim = int(pose_dim)
        self.encoder = nn.Sequential(nn.Linear(120, hidden), nn.LayerNorm(hidden), nn.Tanh())
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.pose_head = nn.Linear(hidden, pose_dim)
        self.velocity_head = nn.Linear(hidden, pose_dim)
        self.sensor_mean = nn.Linear(hidden, sensor_latent)
        self.sensor_logvar = nn.Linear(hidden, 5)
        self.acceleration_head = nn.Linear(hidden, 15)

    def forward(self, imu: Tensor, state=None):
        if imu.ndim != 3 or imu.shape[-1] != 60:
            raise ValueError("imu must have shape [B,T,60]")
        previous = None
        gru_state = state
        if isinstance(state, tuple):
            gru_state, previous = state
        delta = torch.zeros_like(imu)
        if previous is not None:
            delta[:, :1] = imu[:, :1] - previous[:, None]
        if imu.shape[1] > 1:
            delta[:, 1:] = imu[:, 1:] - imu[:, :-1]
        z, next_gru_state = self.rnn(self.encoder(torch.cat((imu, delta), -1)), gru_state)
        pose = self.pose_head(z)
        velocity = self.velocity_head(z)
        sensor_mean = 0.65 * torch.tanh(self.sensor_mean(z)).reshape(*imu.shape[:2], 5, 3)
        sensor_logvar = self.sensor_logvar(z).clamp(-8.0, 4.0)
        bone_rot = M.r6d_to_rotation_matrix(pose.reshape(-1, 6)).reshape(*imu.shape[:2], 16, 3, 3)
        # PNP5 sensors are attached to SMPL joints [18, 19, 4, 5, 0].
        # Their slots in joint_set.reduced are [14, 15, 4, 5, 0].
        sensor_bone = bone_rot[:, :, (14, 15, 4, 5, 0)]
        mount = self._so3_exp(sensor_mean.reshape(-1, 3)).reshape(*imu.shape[:2], 5, 3, 3)
        orientation_prediction = sensor_bone @ mount
        acceleration_prediction = self.acceleration_head(z)
        return {"pose": pose, "velocity": velocity, "sensor_mean": sensor_mean,
                "sensor_logvar": sensor_logvar,
                "orientation_prediction": orientation_prediction,
                "acceleration_prediction": acceleration_prediction,
                "state": (next_gru_state, imu[:, -1].detach())}

    @staticmethod
    def _so3_exp(vector: Tensor) -> Tensor:
        v = vector.reshape(-1, 3)
        theta2 = v.square().sum(-1, keepdim=True).clamp_min(1e-8)
        theta = theta2.sqrt()
        a = torch.sin(theta) / theta
        b = (1.0 - torch.cos(theta)) / theta2
        x, y, z = v.unbind(-1)
        zero = torch.zeros_like(x)
        k = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
        eye = torch.eye(3, device=v.device, dtype=v.dtype).expand(len(v), 3, 3)
        return eye + a[..., None] * k + b[..., None] * (k @ k)

    def step(self, imu: Tensor, state: Optional[Tensor] = None):
        if imu.ndim != 2 or imu.shape[-1] != 60:
            raise ValueError("imu must have shape [B,60]")
        out = self.forward(imu[:, None], state)
        return {k: (v[:, 0] if k != "state" else v) for k, v in out.items()}


def belief_loss(output: dict[str, Tensor], imu: Tensor, target_pose: Tensor,
                pose_weight: float = 1.0, obs_weight: float = 0.1,
                smooth_weight: float = 1e-3, uncertainty_weight: float = 1e-3,
                mask: Optional[Tensor] = None) -> Tensor:
    """Supervised pose plus self-consistent observation and slow-state losses."""
    pose = output["pose"]
    if pose.shape != target_pose.shape:
        raise ValueError("target_pose shape must match output pose")
    if mask is None:
        mask = torch.ones(imu.shape[:2], dtype=torch.bool, device=imu.device)
    weights = mask.to(pose.dtype).unsqueeze(-1)
    denom = weights.sum().clamp_min(1.0)
    loss = pose_weight * ((pose - target_pose).square() * weights).sum() / (denom * pose.shape[-1])
    # Orientation observations are generated from predicted bone orientation
    # and the latent sensor-to-bone transform, not by pre-correcting the IMU.
    measured = imu[..., 15:].reshape(*imu.shape[:2], 5, 3, 3)
    pred = output["orientation_prediction"]
    chordal_error = (pred - measured).square().sum((-1, -2))
    logvar = output["sensor_logvar"]
    nll = 0.5 * (chordal_error * torch.exp(-logvar) + logvar)
    loss = loss + obs_weight * ((nll * mask.unsqueeze(-1)).sum() / (denom * 5.0))
    acc_loss = (output["acceleration_prediction"] - imu[..., :15]).square().mean(-1)
    loss = loss + 0.05 * ((acc_loss * mask).sum() / denom)
    if imu.shape[1] > 1:
        pair = mask[:, 1:] & mask[:, :-1]
        if pair.any():
            diff = (output["sensor_mean"][:, 1:] - output["sensor_mean"][:, :-1]).square().mean((-1, -2))
            loss = loss + smooth_weight * diff[pair].mean()
    # Penalize unjustified uncertainty only weakly; the model can still widen
    # its posterior when the observation is ambiguous.
    if imu.shape[1] > 1:
        pair = mask[:, 1:] & mask[:, :-1]
        if pair.any():
            target_velocity = target_pose[:, 1:] - target_pose[:, :-1]
            vel_error = (output["velocity"][:, 1:] - target_velocity).square().mean(-1)
            loss = loss + 0.1 * vel_error[pair].mean()
    loss = loss + uncertainty_weight * (output["sensor_logvar"].exp() * mask.unsqueeze(-1)).sum() / (denom * 5.0)
    return loss


def belief_smoke(device: str = "cpu") -> dict[str, float]:
    torch.manual_seed(14027)
    dev = torch.device(device)
    model = CausalBeliefFilter(hidden=32).to(dev).eval()
    x = torch.randn(2, 17, 60, device=dev)
    with torch.no_grad():
        ref = model(x)
        changed = x.clone(); changed[:, 9:] = torch.randn_like(changed[:, 9:]) * 100
        prefix = model(x[:, :9])["pose"]
        changed_prefix = model(changed[:, :9])["pose"]
        future_error = float((prefix - changed_prefix).abs().max())
        state = None; streamed = []
        for t in range(x.shape[1]):
            out = model.step(x[:, t], state); state = out["state"]; streamed.append(out["pose"])
        stream_error = float((torch.stack(streamed, 1) - ref["pose"]).abs().max())
        finite = all(bool(torch.isfinite(v).all()) for k, v in ref.items() if k != "state")
    if future_error != 0.0 or stream_error > 1e-5 or not finite:
        raise AssertionError((future_error, stream_error, finite))
    return {"future_prefix_max_abs_error": future_error, "streaming_max_abs_error": stream_error, "finite": float(finite)}


if __name__ == "__main__":
    print(belief_smoke())

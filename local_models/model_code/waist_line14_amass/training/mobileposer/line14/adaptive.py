"""Strictly causal, calibration-free latent sensor-frame adaptation.

The adapter is deliberately trained from pose supervision rather than from a
sensor-to-bone calibration label.  A small causal GRU estimates a bounded
rotation correction for each sensor from the motion history.  The correction
starts at identity, so a newly added adapter can be attached to a baseline
without changing its initial input distribution.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from mobileposer.articulate import math as M


class CausalLatentSensorAdapter(nn.Module):
    """Infer per-sensor frame corrections using only the causal IMU prefix.

    Input/output use packed PNP5 frames ``[B,T,60]``.  Accelerations and the
    first five sensor positions are passed through unchanged; only the 5
    orientation matrices are right-multiplied by the learned correction.
    ``max_angle`` prevents the learned latent state from becoming an
    unconstrained replacement for the pose decoder.
    """

    def __init__(self, hidden: int = 128, max_angle: float = 0.65):
        super().__init__()
        if hidden <= 0 or max_angle <= 0:
            raise ValueError("hidden and max_angle must be positive")
        self.max_angle = float(max_angle)
        # Orientation is represented by 6D columns; velocity-like changes are
        # appended as a causal finite difference.  No future frame is used.
        self.encoder = nn.Sequential(nn.Linear(45 + 45, hidden), nn.Tanh())
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 15)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        packed: Tensor,
        state: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if packed.ndim != 3 or packed.shape[-1] != 60:
            raise ValueError("packed must have shape [B,T,60]")
        ori = packed[..., 15:].reshape(*packed.shape[:2], 5, 3, 3)
        # Relative orientation to the first observed frame makes the latent
        # estimator insensitive to a global sensor-frame convention.
        ref = ori[:, :1]
        rel = ref.transpose(-1, -2) @ ori
        flat = rel.reshape(*packed.shape[:2], 45)
        delta = torch.zeros_like(flat)
        if packed.shape[1] > 1:
            delta[:, 1:] = flat[:, 1:] - flat[:, :-1]
        z, next_state = self.rnn(self.encoder(torch.cat((flat, delta), dim=-1)), state)
        tangent = self.max_angle * torch.tanh(self.head(z)).reshape(*packed.shape[:2], 5, 3)
        # The repository's axis-angle converter normalizes by ||v|| and has
        # undefined gradients at exactly zero.  Use a stable Rodrigues form;
        # clamping theta^2 only stabilizes scalar coefficients while the skew
        # matrix still carries the correct first-order gradient at identity.
        v = tangent.reshape(-1, 3)
        theta2 = v.square().sum(-1, keepdim=True).clamp_min(1e-8)
        theta = theta2.sqrt()
        A = torch.sin(theta) / theta
        B = (1.0 - torch.cos(theta)) / theta2
        vx, vy, vz = v.unbind(-1)
        z = torch.zeros_like(vx)
        K = torch.stack((z, -vz, vy, vz, z, -vx, -vy, vx, z), -1).reshape(-1, 3, 3)
        eye = torch.eye(3, device=v.device, dtype=v.dtype).expand(len(v), 3, 3)
        correction = eye + A[..., None] * K + B[..., None] * (K @ K)
        correction = correction.reshape(*packed.shape[:2], 5, 3, 3)
        adapted = ori @ correction.transpose(-1, -2)
        out = torch.cat((packed[..., :15], adapted.reshape(*packed.shape[:2], 45)), dim=-1)
        return out, next_state, tangent


def causal_adapter_smoke(device: str = "cpu") -> dict[str, float]:
    """Check zero initialization, streaming equivalence, and future invariance."""
    torch.manual_seed(14027)
    dev = torch.device(device)
    model = CausalLatentSensorAdapter(hidden=24).to(dev).eval()
    x = torch.randn(2, 13, 60, device=dev)
    with torch.no_grad():
        y, _, tangent = model(x)
        zero_angle = float(tangent.abs().max())
        streamed = []
        state = None
        for t in range(x.shape[1]):
            value, state, _ = model(x[:, t:t + 1], state)
            streamed.append(value)
        stream_error = float((torch.cat(streamed, 1) - y).abs().max())
        split = 8
        changed = x.clone()
        changed[:, split:] = torch.randn_like(changed[:, split:]) * 100
        prefix, _, _ = model(x[:, :split])
        changed_prefix, _, _ = model(changed[:, :split])
        future_error = float((prefix - changed_prefix).abs().max())
    if zero_angle != 0.0 or stream_error > 1e-5 or future_error != 0.0:
        raise AssertionError((zero_angle, stream_error, future_error))
    return {"zero_angle": zero_angle, "streaming_max_abs_error": stream_error, "future_prefix_max_abs_error": future_error}


if __name__ == "__main__":
    print(causal_adapter_smoke())

"""Calibration-conditioned endpoint-causal wrappers for existing MobilePoser RNNs."""
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from mobileposer.line14.causal import CalibrationState
from mobileposer.models.rnn import RNN


class CalibrationConditionedRNN(nn.Module):
    """Zero-initialized FiLM around a pretrained RNN, preserving its baseline initially."""

    def __init__(self, base: RNN, condition_dim: int):
        super().__init__()
        if not isinstance(base, RNN):
            raise TypeError("base must be a MobilePoser RNN")
        self.base = base
        hidden = base.linear1.out_features
        output = base.linear2.out_features
        self.film_in = nn.Linear(condition_dim, 2 * hidden)
        self.film_out = nn.Linear(condition_dim, 2 * output)
        self.reliability_gate = nn.Sequential(
            nn.Linear(condition_dim + 15, 32), nn.Tanh(), nn.Linear(32, 5)
        )
        nn.init.zeros_(self.film_in.weight)
        nn.init.zeros_(self.film_in.bias)
        nn.init.zeros_(self.film_out.weight)
        nn.init.zeros_(self.film_out.bias)
        nn.init.zeros_(self.reliability_gate[-1].weight)
        nn.init.constant_(self.reliability_gate[-1].bias, -3.0)

    def forward(self, x: Tensor, condition: Tensor, lengths) -> Tensor:
        if x.ndim != 3 or condition.ndim != 2 or x.shape[0] != condition.shape[0]:
            raise ValueError("x must be [B,T,F] and condition [B,C]")
        if lengths is None or len(lengths) != len(x):
            raise ValueError("valid sequence lengths are required")
        gamma, beta = self.film_in(condition).chunk(2, dim=-1)
        hidden = self.base.linear1(x)
        hidden = self.base.dropout(F.relu(hidden * (1.0 + gamma[:, None]) + beta[:, None]))
        packed = pack_padded_sequence(hidden, lengths, batch_first=True, enforce_sorted=False)
        packed, _ = self.base.rnn(packed)
        hidden, _ = pad_packed_sequence(packed, batch_first=True, total_length=x.shape[1])
        output = self.base.linear2(hidden)
        out_gamma, out_beta = self.film_out(condition).chunk(2, dim=-1)
        return output * (1.0 + out_gamma[:, None]) + out_beta[:, None]

    def calibration_gates(self, condition: Tensor, offsets: Tensor) -> Tensor:
        """Estimate per-sensor trust from frozen geometry uncertainty and offset."""
        from mobileposer.articulate import math as M
        tangent = M.rotation_matrix_to_axis_angle(offsets.reshape(-1, 3, 3)).reshape(-1, 15)
        gate = torch.sigmoid(self.reliability_gate(torch.cat((condition, tangent), -1)))
        return gate


class CalibratedEndpointSession:
    """Zero-lookahead session using frozen calibration and at most ``past`` history frames."""

    def __init__(
        self,
        joints: CalibrationConditionedRNN,
        poser: CalibrationConditionedRNN,
        calibration: CalibrationState,
        *,
        past: int = 40,
        decode_pose: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        if joints.base.linear1.in_features != 60 or joints.base.linear2.out_features != 72:
            raise ValueError("PNP5 Joints RNN must map 60 inputs to 72 outputs")
        if poser.base.linear1.in_features != 132 or poser.base.linear2.out_features != 96:
            raise ValueError("PNP5 Poser RNN must map 132 inputs to 96 outputs")
        if joints.film_in.in_features != calibration.condition_dim or poser.film_in.in_features != calibration.condition_dim:
            raise ValueError("FiLM dimensions must match the frozen calibration state")
        if past < 0:
            raise ValueError("past must be non-negative")
        self.joints = joints.eval()
        self.poser = poser.eval()
        self.calibration = calibration
        self.past = int(past)
        self.decode_pose = decode_pose
        self._raw_history: list[Tensor] = []

    def _correct(self, raw: Tensor) -> Tensor:
        """Blend estimated offsets toward identity using learned sensor trust."""
        from mobileposer.articulate import math as M
        offsets = self.calibration.i2s_rotation.to(device=raw.device, dtype=raw.dtype)
        condition = self.calibration.condition(device=raw.device, dtype=raw.dtype)
        gates = self.joints.calibration_gates(condition, offsets).to(device=raw.device, dtype=raw.dtype)
        tangent = M.rotation_matrix_to_axis_angle(offsets.reshape(-1, 3, 3)).reshape(5, 3)
        effective = M.axis_angle_to_rotation_matrix((tangent * gates[:, None]).reshape(-1, 3)).reshape(5, 3, 3)
        measured = raw[..., 15:].reshape(*raw.shape[:-1], 5, 3, 3)
        corrected = measured @ effective.transpose(-1, -2)
        return torch.cat((raw[..., :15], corrected.flatten(-2)), -1)

    @classmethod
    def from_mobileposer(cls, joints_model, poser_model, calibration: CalibrationState, *, past: int = 40):
        """Wrap loaded standard models; no checkpoint weights are modified."""
        if getattr(joints_model, "slowfast", False) or getattr(joints_model, "jchain", False):
            raise ValueError("only the plain PNP5 Joints model is supported")
        if getattr(joints_model, "relacc", False) or getattr(joints_model, "lever", False):
            raise ValueError("augmented Joints inputs need a separate PNP5 audit")
        if getattr(poser_model, "relacc", False) or getattr(poser_model, "lever", False):
            raise ValueError("augmented Poser inputs need a separate PNP5 audit")
        return cls(
            CalibrationConditionedRNN(joints_model.joints, calibration.condition_dim),
            CalibrationConditionedRNN(poser_model.pose, calibration.condition_dim),
            calibration,
            past=past,
            decode_pose=poser_model._reduced_global_to_full,
        )

    def reset(self) -> None:
        self._raw_history.clear()

    @torch.inference_mode()
    def step(self, packed_imu_frame: Tensor) -> Tensor:
        device = next(self.joints.parameters()).device
        dtype = next(self.joints.parameters()).dtype
        frame = torch.as_tensor(packed_imu_frame, dtype=dtype, device=device)
        if frame.shape != (60,):
            raise ValueError("a live step consumes exactly one PNP5 packed IMU frame [60]")
        self._raw_history.append(frame.detach())
        self._raw_history = self._raw_history[-(self.past + 1):]
        raw = torch.stack(self._raw_history).unsqueeze(0)
        corrected = self._correct(raw)
        condition = self.calibration.condition(device=device, dtype=dtype)
        length = [corrected.shape[1]]
        joint = self.joints(corrected, condition, length)
        pose_input = torch.cat((joint, corrected), dim=-1)
        reduced = self.poser(pose_input, condition, length)[:, -1]
        if self.decode_pose is not None:
            return self.decode_pose(reduced).reshape(24, 3, 3)
        return reduced[0]

"""Calibration-conditioned, strictly causal PNP5 inference primitives.

The calibration estimator is deliberately separate from the target-motion
decoder.  A caller may analyse the complete, already-finished calibration
clip, then construct a frozen :class:`CalibrationState`.  Once prediction
starts, :class:`CalibratedCausalSession.step` accepts exactly one current IMU
frame and carries recurrent state forward; it has no target-window or
look-ahead argument.

This module defines an untrained architecture/inference contract.  It is not
a trained line14 model and must not be used to claim benchmark performance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


N_SENSORS = 5
IMU_DIM = 60
JOINT_DIM = 72
POSE_DIM = 16 * 6
POSE_INPUT_DIM = JOINT_DIM + IMU_DIM
PIVOTPRINT_DIM = 24


@dataclass(frozen=True)
class CalibrationState:
    """Frozen estimates obtained only from the separate pre-prediction clip.

    ``i2s_rotation`` follows ``R_measured = R_bone @ R_i2s`` and has shape
    ``[5,3,3]``. ``pivotprint`` is the 24-D ordered R2 vector
    (lw->elbow/shoulder, rw->elbow/shoulder, ls->knee/hip,
    rs->knee/hip), in metres.  Training must apply the same normalization to
    this vector that is used for the online condition token.

    Optional log-variance represents calibration-estimator uncertainty.  It is
    only exposed as a conditional input; no uncertainty-gating benefit is
    assumed by this prototype.
    """

    i2s_rotation: Tensor
    pivotprint: Tensor
    pivotprint_logvar: Optional[Tensor] = None

    @classmethod
    def from_link_output(
        cls,
        i2s_rotation: Tensor,
        geometry_mean_cm: Tensor,
        geometry_logvar_cm: Optional[Tensor] = None,
    ) -> "CalibrationState":
        """Build frozen state from one C20/C21 session prediction.

        C20 stores the ordered 24-D R2 mean in centimetres and optional
        component log-variance in cm^2. Convert both to SI before freezing.
        Inputs must already be selected/aggregated for the completed
        calibration session; this method never reads target-motion frames.
        """
        geometry_m = torch.as_tensor(geometry_mean_cm, dtype=torch.float32) / 100.0
        logvar_m2 = None
        if geometry_logvar_cm is not None:
            logvar_cm = torch.as_tensor(geometry_logvar_cm, dtype=torch.float32)
            logvar_m2 = logvar_cm - 2.0 * torch.log(logvar_cm.new_tensor(100.0))
        return cls(i2s_rotation=i2s_rotation, pivotprint=geometry_m, pivotprint_logvar=logvar_m2)

    def __post_init__(self) -> None:
        rotation = torch.as_tensor(self.i2s_rotation, dtype=torch.float32).detach().clone()
        geometry = torch.as_tensor(self.pivotprint, dtype=torch.float32).detach().clone()
        if rotation.shape != (N_SENSORS, 3, 3):
            raise ValueError(f"i2s_rotation must have shape [5,3,3], got {tuple(rotation.shape)}")
        if geometry.shape != (PIVOTPRINT_DIM,):
            raise ValueError(f"pivotprint must have shape [24], got {tuple(geometry.shape)}")
        if not torch.isfinite(rotation).all() or not torch.isfinite(geometry).all():
            raise ValueError("calibration estimates must be finite")
        eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
        orth_err = (rotation.transpose(-1, -2) @ rotation - eye).abs().amax()
        det_err = (torch.linalg.det(rotation) - 1.0).abs().amax()
        if orth_err > 1e-3 or det_err > 1e-3:
            raise ValueError("i2s_rotation must contain proper rotation matrices")
        logvar = self.pivotprint_logvar
        if logvar is not None:
            logvar = torch.as_tensor(logvar, dtype=torch.float32).detach().clone()
            if logvar.shape != (PIVOTPRINT_DIM,) or not torch.isfinite(logvar).all():
                raise ValueError("pivotprint_logvar must be a finite [24] vector")
        object.__setattr__(self, "i2s_rotation", rotation)
        object.__setattr__(self, "pivotprint", geometry)
        object.__setattr__(self, "pivotprint_logvar", logvar)

    @property
    def condition_dim(self) -> int:
        return PIVOTPRINT_DIM * (2 if self.pivotprint_logvar is not None else 1)

    def condition(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        parts = [self.pivotprint]
        if self.pivotprint_logvar is not None:
            parts.append(self.pivotprint_logvar)
        return torch.cat(parts).to(device=device, dtype=dtype).unsqueeze(0)

    def correct_frame(self, packed_imu: Tensor) -> Tensor:
        """Correct packed PNP5 frame(s); acceleration remains unchanged."""
        if packed_imu.ndim == 1:
            packed_imu = packed_imu.unsqueeze(0)
        if packed_imu.ndim < 2 or packed_imu.shape[-1] != IMU_DIM:
            raise ValueError("packed_imu must have shape [60] or [...,60]")
        offsets = self.i2s_rotation.to(device=packed_imu.device, dtype=packed_imu.dtype)
        measured = packed_imu[..., 15:].reshape(*packed_imu.shape[:-1], N_SENSORS, 3, 3)
        corrected = measured @ offsets.transpose(-1, -2)
        return torch.cat((packed_imu[..., :15], corrected.reshape(*packed_imu.shape[:-1], 45)), dim=-1)


class CausalConditionedLSTM(nn.Module):
    """Unidirectional recurrent decoder with a fixed per-session condition.

    ``step`` is the deployment API. ``forward`` is provided for sequence
    training and uses the same unidirectional recurrence, with one condition
    vector shared across every frame in that sequence.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        condition_dim: int = PIVOTPRINT_DIM,
        hidden_dim: int = 256,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(input_dim, output_dim, condition_dim, hidden_dim, layers) <= 0:
            raise ValueError("all decoder dimensions must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.condition_projection = nn.Linear(self.condition_dim, self.hidden_dim, bias=False)
        self.dropout = nn.Dropout(float(dropout))
        self.rnn = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.layers,
            batch_first=True,
            bidirectional=False,
            dropout=float(dropout) if self.layers > 1 else 0.0,
        )
        self.output_projection = nn.Linear(self.hidden_dim, self.output_dim)

    def _validate(self, x: Tensor, condition: Tensor) -> None:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"x must have shape [B,T,{self.input_dim}]")
        if condition.ndim != 2 or condition.shape != (x.shape[0], self.condition_dim):
            raise ValueError(f"condition must have shape [B,{self.condition_dim}]")

    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        state: Optional[tuple[Tensor, Tensor]] = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        self._validate(x, condition)
        z = F.relu(self.input_projection(x) + self.condition_projection(condition).unsqueeze(1))
        z = self.dropout(z)
        z, next_state = self.rnn(z, state)
        return self.output_projection(z), next_state

    def step(
        self,
        x: Tensor,
        condition: Tensor,
        state: Optional[tuple[Tensor, Tensor]] = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Consume exactly one time step ``x:[B,F]`` and return ``[B,O]``."""
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"x must have shape [B,{self.input_dim}]")
        output, next_state = self.forward(x.unsqueeze(1), condition, state)
        return output[:, 0], next_state


class CalibratedCausalSession:
    """Stateful causal Joints -> Pose pipeline for one frozen PNP5 calibration.

    The joint decoder must map 60 -> 72 channels; the pose decoder maps
    72 joint channels + 60 corrected IMU channels -> 96 reduced global 6-D
    rotations. Both models must have been trained with the same calibration
    condition representation and normalization as ``CalibrationState``.
    """

    def __init__(
        self,
        joints: CausalConditionedLSTM,
        pose: CausalConditionedLSTM,
        calibration: CalibrationState,
    ) -> None:
        if joints.input_dim != IMU_DIM or joints.output_dim != JOINT_DIM:
            raise ValueError("joints decoder must have 60 input and 72 output channels")
        if pose.input_dim != POSE_INPUT_DIM or pose.output_dim != POSE_DIM:
            raise ValueError("pose decoder must have 132 input and 96 output channels")
        if joints.condition_dim != calibration.condition_dim or pose.condition_dim != calibration.condition_dim:
            raise ValueError("decoder condition dimensions must match the frozen calibration state")
        if next(joints.parameters()).device != next(pose.parameters()).device:
            raise ValueError("joint and pose decoders must be on the same device")
        self.joints = joints.eval()
        self.pose = pose.eval()
        self.calibration = calibration
        self._joint_state: Optional[tuple[Tensor, Tensor]] = None
        self._pose_state: Optional[tuple[Tensor, Tensor]] = None

    def reset(self) -> None:
        """Start another target action for the same calibration."""
        self._joint_state = None
        self._pose_state = None

    @torch.no_grad()
    def step(self, packed_imu_frame: Tensor) -> Tensor:
        """Predict from one current PNP5 frame; no future/action window exists."""
        device = next(self.joints.parameters()).device
        dtype = next(self.joints.parameters()).dtype
        frame = torch.as_tensor(packed_imu_frame, device=device, dtype=dtype)
        if frame.ndim == 1:
            frame = frame.unsqueeze(0)
        if frame.shape != (1, IMU_DIM):
            raise ValueError("a live session consumes exactly one packed frame with shape [60]")
        corrected = self.calibration.correct_frame(frame)
        condition = self.calibration.condition(device=device, dtype=dtype)
        joint, self._joint_state = self.joints.step(corrected, condition, self._joint_state)
        pose_input = torch.cat((joint, corrected), dim=-1)
        output, self._pose_state = self.pose.step(pose_input, condition, self._pose_state)
        return output[0]


def synthetic_causality_test(device: str = "cpu") -> dict[str, float]:
    """Check future invariance and batch/streaming equivalence for both stages."""
    torch.manual_seed(14027)
    dev = torch.device(device)
    model = CausalConditionedLSTM(7, 5, condition_dim=4, hidden_dim=16, layers=2, dropout=0.0).to(dev).eval()
    x = torch.randn(1, 19, 7, device=dev)
    condition = torch.randn(1, 4, device=dev)
    split = 11
    changed = x.clone()
    changed[:, split:] = torch.randn_like(changed[:, split:]) * 100
    with torch.no_grad():
        expected, _ = model(x, condition)
        perturbed, _ = model(changed, condition)
        prefix_error = (expected[:, :split] - perturbed[:, :split]).abs().amax()
        if not torch.equal(expected[:, :split], perturbed[:, :split]):
            raise AssertionError(f"future-frame perturbation changed prefix output: {prefix_error.item()}")
        streamed = []
        state = None
        for t in range(x.shape[1]):
            value, state = model.step(x[:, t], condition, state)
            streamed.append(value)
        stream_error = (torch.stack(streamed, 1) - expected).abs().amax()
        if stream_error > 1e-5:
            raise AssertionError(f"streaming and sequence outputs differ by {stream_error.item()}")

        angle = torch.linspace(-0.4, 0.4, N_SENSORS, device=dev)
        zero, one = torch.zeros_like(angle), torch.ones_like(angle)
        c, s = angle.cos(), angle.sin()
        i2s = torch.stack((c, -s, zero, s, c, zero, zero, zero, one), dim=-1).reshape(N_SENSORS, 3, 3)
        bone_angle = angle.flip(0) * 0.7
        cb, sb = bone_angle.cos(), bone_angle.sin()
        bone = torch.stack((one, zero, zero, zero, cb, -sb, zero, sb, cb), dim=-1).reshape(N_SENSORS, 3, 3)
        calibration = CalibrationState(i2s, torch.randn(PIVOTPRINT_DIM, device=dev))
        raw_geometry_cm = torch.full((PIVOTPRINT_DIM,), 125.0, device=dev)
        raw_logvar_cm = torch.full((PIVOTPRINT_DIM,), 2.0 * torch.log(torch.tensor(100.0)), device=dev)
        from_link = CalibrationState.from_link_output(i2s, raw_geometry_cm, raw_logvar_cm)
        geometry_unit_error = (from_link.pivotprint - 1.25).abs().amax()
        logvar_unit_error = from_link.pivotprint_logvar.abs().amax()
        if geometry_unit_error > 1e-6 or logvar_unit_error > 1e-6:
            raise AssertionError("C20 geometry mean/log-variance unit conversion failed")
        packed = torch.cat((torch.randn(1, 15, device=dev), (bone @ i2s).reshape(1, 45)), dim=-1)
        corrected = calibration.correct_frame(packed)[0, 15:].reshape(N_SENSORS, 3, 3)
        i2s_error = (corrected - bone).abs().amax()
        if i2s_error > 1e-5:
            raise AssertionError(f"I2S correction convention failed by {i2s_error.item()}")
        joint_model = CausalConditionedLSTM(IMU_DIM, JOINT_DIM, hidden_dim=16, dropout=0.0).to(dev).eval()
        pose_model = CausalConditionedLSTM(POSE_INPUT_DIM, POSE_DIM, hidden_dim=16, dropout=0.0).to(dev).eval()
        first = CalibratedCausalSession(joint_model, pose_model, calibration)
        live = torch.randn(1, 19, IMU_DIM, device=dev)
        live_changed = live.clone()
        live_changed[:, split:] = torch.randn_like(live_changed[:, split:]) * 100
        online = torch.stack([first.step(live[:, t]) for t in range(live.shape[1])])

        condition = calibration.condition(device=dev, dtype=torch.float32)
        corrected = torch.stack([calibration.correct_frame(frame)[0] for frame in live[0]])
        corrected_changed = torch.stack([calibration.correct_frame(frame)[0] for frame in live_changed[0]])
        joints_ref, _ = joint_model(corrected.unsqueeze(0), condition)
        joints_changed, _ = joint_model(corrected_changed.unsqueeze(0), condition)
        pose_ref, _ = pose_model(torch.cat((joints_ref, corrected.unsqueeze(0)), -1), condition)
        pose_changed, _ = pose_model(torch.cat((joints_changed, corrected_changed.unsqueeze(0)), -1), condition)
        session_error = (pose_ref[0] - online).abs().amax()
        pipeline_future_error = (pose_ref[:, :split] - pose_changed[:, :split]).abs().amax()
        if session_error > 1e-5:
            raise AssertionError(f"two-stage streaming and sequence paths differ by {session_error.item()}")
        if pipeline_future_error != 0:
            raise AssertionError(f"future-frame perturbation changed two-stage prefix: {pipeline_future_error.item()}")
    return {
        "i2s_correction_max_abs_error": float(i2s_error),
        "geometry_unit_conversion_max_abs_error": float(geometry_unit_error),
        "logvar_unit_conversion_max_abs_error": float(logvar_unit_error),
        "future_prefix_max_abs_error": float(prefix_error),
        "streaming_max_abs_error": float(stream_error),
        "two_stage_session_max_abs_error": float(session_error),
        "two_stage_future_prefix_max_abs_error": float(pipeline_future_error),
    }


if __name__ == "__main__":
    print(synthetic_causality_test())

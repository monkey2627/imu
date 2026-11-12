"""Calibration-motion encoder for the isolated line-9 experiments.

The encoder consumes one fixed calibration motion and produces an identity-level
latent, five sensor-to-segment (I2S) rotations, and subject geometry.  The latent
is deliberately independent of a target-motion window so it can be computed once
and persisted for all subsequent motions from the same calibrated session.

Rotation convention follows :mod:`mobileposer.line9.calib`::

    R_measured = R_bone @ R_i2s

Consequently, correcting an orientation means ``R_measured @ R_i2s.T``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


N_SENSORS = 5
N_BONES = 23
IMU_DIM = 60


def rotation_6d_to_matrix(rotation_6d: Tensor, eps: float = 1e-6) -> Tensor:
    """Convert the first two rotation columns to a proper rotation matrix.

    The last dimension follows the repository convention ``[col0, col1]``.
    Gram-Schmidt orthogonalization makes the network output continuous while the
    returned matrices remain in SO(3) up to floating-point precision.
    """

    if rotation_6d.shape[-1] != 6:
        raise ValueError("rotation_6d must have final dimension 6")
    first = F.normalize(rotation_6d[..., :3], dim=-1, eps=eps)
    second_raw = rotation_6d[..., 3:]
    second = second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first
    second = F.normalize(second, dim=-1, eps=eps)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def matrix_to_rotation_6d(rotation: Tensor) -> Tensor:
    """Return the first two columns of ``[..., 3, 3]`` matrices as ``[..., 6]``."""

    if rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must have final dimensions [3, 3]")
    return rotation[..., :, :2].transpose(-1, -2).reshape(*rotation.shape[:-2], 6)


def rotation_geodesic_loss(
    prediction: Tensor,
    target: Tensor,
    sensor_mask: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    """SO(3) geodesic error in radians for rotation matrices ``[..., 3, 3]``."""

    if prediction.shape != target.shape or prediction.shape[-2:] != (3, 3):
        raise ValueError("prediction and target must have equal [..., 3, 3] shapes")
    relative = prediction.transpose(-1, -2) @ target
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
    # atan2 is better behaved close to zero than acos.  The skew norm equals
    # 2*sin(theta), while the trace supplies cos(theta).
    skew = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    )
    sine = 0.5 * skew.norm(dim=-1)
    error = torch.atan2(sine, cosine.clamp(-1.0, 1.0))
    return _masked_reduction(error, sensor_mask, reduction)


def bone_length_relative_loss(
    prediction: Tensor,
    target: Tensor,
    bone_mask: Optional[Tensor] = None,
    reduction: str = "mean",
    eps: float = 1e-6,
) -> Tensor:
    """Absolute relative error for the 23 non-root SMPL bone lengths."""

    if prediction.shape != target.shape or prediction.shape[-1] != N_BONES:
        raise ValueError(f"bone lengths must have equal [..., {N_BONES}] shapes")
    if (target <= 0).any():
        raise ValueError("target bone lengths must be strictly positive")
    error = (prediction - target).abs() / target.clamp_min(eps)
    return _masked_reduction(error, bone_mask, reduction)


def lever_arm_loss(
    prediction: Tensor,
    target: Tensor,
    sensor_mask: Optional[Tensor] = None,
    reduction: str = "mean",
) -> Tensor:
    """Euclidean lever-arm error in metres for ``[..., 5, 3]`` tensors."""
    if prediction.shape != target.shape or prediction.shape[-2:] != (N_SENSORS, 3):
        raise ValueError("lever arms must have equal [..., 5, 3] shapes")
    return _masked_reduction((prediction - target).norm(dim=-1), sensor_mask, reduction)


def _masked_reduction(value: Tensor, mask: Optional[Tensor], reduction: str) -> Tensor:
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if mask is not None:
        mask = torch.as_tensor(mask, device=value.device, dtype=torch.bool)
        if mask.shape != value.shape:
            raise ValueError(f"mask shape {tuple(mask.shape)} != value shape {tuple(value.shape)}")
        if reduction == "none":
            return value.masked_fill(~mask, 0.0)
        selected = value.masked_select(mask)
        if selected.numel() == 0:
            raise ValueError("loss mask selects no values")
        return selected.mean() if reduction == "mean" else selected.sum()
    if reduction == "none":
        return value
    return value.mean() if reduction == "mean" else value.sum()


def correct_i2s_orientations(measured: Tensor, estimated_i2s: Tensor) -> Tensor:
    """Explicitly undo fixed I2S offsets: ``R_corrected = R_measured @ R_est.T``.

    Args:
        measured: ``[B,T,5,3,3]`` or any ``[...,5,3,3]`` tensor.
        estimated_i2s: one persistent estimate per item, ``[B,5,3,3]``.
    """

    if measured.shape[-3:] != (N_SENSORS, 3, 3):
        raise ValueError("measured must end in [5, 3, 3]")
    if estimated_i2s.ndim != 4 or estimated_i2s.shape[-3:] != (N_SENSORS, 3, 3):
        raise ValueError("estimated_i2s must have shape [B, 5, 3, 3]")
    if measured.shape[0] != estimated_i2s.shape[0]:
        raise ValueError("measured and estimated_i2s batch sizes differ")
    expand_dims = measured.ndim - estimated_i2s.ndim
    offset_t = estimated_i2s.transpose(-1, -2)
    for _ in range(expand_dims):
        offset_t = offset_t.unsqueeze(1)
    return measured @ offset_t


def correct_packed_imu(imu: Tensor, estimated_i2s: Tensor) -> Tensor:
    """Correct orientation channels of packed ``[...,60]`` IMU features.

    Acceleration channels are retained unchanged.  Lever-arm compensation is
    intentionally not attempted because it also requires angular velocity and
    acceleration (or a differentiable temporal estimator).
    """

    if imu.ndim != 3 or imu.shape[-1] != IMU_DIM:
        raise ValueError("imu must have shape [B, T, 60]")
    orientation = imu[..., 15:].reshape(*imu.shape[:2], N_SENSORS, 3, 3)
    corrected = correct_i2s_orientations(orientation, estimated_i2s)
    return torch.cat((imu[..., :15], corrected.reshape(*imu.shape[:2], 45)), dim=-1)


@dataclass
class CalibrationEncoding:
    """Reusable identity/session calibration state returned by the encoder."""

    latent: Tensor
    i2s_rotation_6d: Tensor
    i2s_rotation_matrix: Tensor
    lever_arm: Tensor
    bone_lengths: Tensor
    beta: Optional[Tensor]
    phase_attention: Tensor
    phase_mask: Tensor


class CalibrationEncoder(nn.Module):
    """Encode a fixed, phase-labelled calibration motion into persistent state.

    Frame features are processed by a packed bidirectional GRU.  Learned scalar
    attention pools frames separately inside each phase; a persistent learned
    query then attends over the resulting phase tokens.  This prevents a long
    phase from dominating solely because it has more frames.

    ``phase_ids`` are integers in ``[0, max_phases)`` at valid frames.  Padding
    may use ``-1`` and is controlled by ``mask``.  If phase IDs are unavailable,
    passing ``None`` treats the full calibration motion as one phase.
    """

    def __init__(
        self,
        *,
        input_dim: int = IMU_DIM,
        hidden_dim: int = 192,
        latent_dim: int = 256,
        max_phases: int = 8,
        gru_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.1,
        predict_beta: bool = True,
        bone_length_mean: Optional[Tensor] = None,
        lever_scale_m: float = 0.03,
    ) -> None:
        super().__init__()
        if input_dim != IMU_DIM:
            raise ValueError("line9 packed five-IMU input_dim must be 60")
        if hidden_dim <= 0 or latent_dim <= 0 or max_phases <= 0 or gru_layers <= 0:
            raise ValueError("model dimensions and phase/layer counts must be positive")
        if latent_dim % attention_heads:
            raise ValueError("latent_dim must be divisible by attention_heads")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.max_phases = max_phases
        self.predict_beta = bool(predict_beta)
        if lever_scale_m <= 0:
            raise ValueError("lever_scale_m must be positive")
        self.lever_scale_m = float(lever_scale_m)

        self.frame_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.phase_embedding = nn.Embedding(max_phases, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        encoded_dim = hidden_dim * 2
        self.frame_attention = nn.Linear(encoded_dim, 1)
        self.phase_projection = nn.Linear(encoded_dim, latent_dim)
        self.persistent_query = nn.Parameter(torch.empty(1, 1, latent_dim))
        self.phase_attention = nn.MultiheadAttention(
            latent_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.latent_norm = nn.LayerNorm(latent_dim)

        self.rotation_head = nn.Linear(latent_dim, N_SENSORS * 6)
        self.lever_head = nn.Linear(latent_dim, N_SENSORS * 3)
        self.bone_head = nn.Linear(latent_dim, N_BONES)
        self.beta_head = nn.Linear(latent_dim, 10) if self.predict_beta else None

        if bone_length_mean is None:
            mean = torch.full((N_BONES,), 0.25, dtype=torch.float32)
        else:
            mean = torch.as_tensor(bone_length_mean, dtype=torch.float32).detach().clone()
            if mean.shape != (N_BONES,) or not torch.isfinite(mean).all() or (mean <= 0).any():
                raise ValueError("bone_length_mean must contain 23 finite positive values")
        self.register_buffer("bone_length_mean", mean)
        self._reset_output_heads()

    def _reset_output_heads(self) -> None:
        nn.init.normal_(self.persistent_query, std=0.02)
        nn.init.zeros_(self.rotation_head.weight)
        identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        with torch.no_grad():
            self.rotation_head.bias.copy_(identity_6d.repeat(N_SENSORS))
        nn.init.zeros_(self.bone_head.weight)
        nn.init.zeros_(self.bone_head.bias)
        nn.init.zeros_(self.lever_head.weight)
        nn.init.zeros_(self.lever_head.bias)
        if self.beta_head is not None:
            nn.init.zeros_(self.beta_head.weight)
            nn.init.zeros_(self.beta_head.bias)

    def _validate_inputs(
        self, imu: Tensor, phase_ids: Optional[Tensor], mask: Optional[Tensor]
    ) -> tuple[Tensor, Tensor]:
        if imu.ndim != 3 or imu.shape[-1] != self.input_dim:
            raise ValueError(f"imu must have shape [B, T, {self.input_dim}]")
        batch, steps = imu.shape[:2]
        if mask is None:
            mask = torch.ones((batch, steps), dtype=torch.bool, device=imu.device)
        else:
            mask = torch.as_tensor(mask, dtype=torch.bool, device=imu.device)
            if mask.shape != (batch, steps):
                raise ValueError("mask must have shape [B, T]")
        if (~mask[:, 0]).any() or (mask.sum(dim=1) == 0).any():
            raise ValueError("each sequence must contain a non-empty valid prefix")
        # Packed RNNs require padding to be a suffix, as produced by line9's
        # collate_paired_windows.
        if ((~mask[:, :-1]) & mask[:, 1:]).any():
            raise ValueError("mask must describe contiguous valid prefixes")
        if phase_ids is None:
            phase_ids = torch.zeros((batch, steps), dtype=torch.long, device=imu.device)
        else:
            phase_ids = torch.as_tensor(phase_ids, dtype=torch.long, device=imu.device)
            if phase_ids.shape != (batch, steps):
                raise ValueError("phase_ids must have shape [B, T]")
        valid_ids = phase_ids[mask]
        if (valid_ids < 0).any() or (valid_ids >= self.max_phases).any():
            raise ValueError(f"valid phase_ids must lie in [0, {self.max_phases})")
        return phase_ids, mask

    def forward(
        self,
        calibration_imu: Tensor,
        phase_ids: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> CalibrationEncoding:
        phase_ids, mask = self._validate_inputs(calibration_imu, phase_ids, mask)
        safe_phase_ids = phase_ids.masked_fill(~mask, 0)
        frame = self.frame_projection(calibration_imu)
        frame = frame + self.phase_embedding(safe_phase_ids)
        lengths = mask.sum(dim=1).to(device="cpu", dtype=torch.long)
        packed = pack_padded_sequence(frame, lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.gru(packed)
        encoded, _ = pad_packed_sequence(
            encoded, batch_first=True, total_length=calibration_imu.shape[1]
        )

        membership = (
            safe_phase_ids[:, None, :]
            == torch.arange(self.max_phases, device=calibration_imu.device)[None, :, None]
        ) & mask[:, None, :]
        phase_mask = membership.any(dim=-1)
        score = self.frame_attention(encoded).squeeze(-1)[:, None, :]
        score = score.masked_fill(~membership, -torch.inf)
        # Missing phases would otherwise softmax an all-inf row into NaNs.
        score = score.masked_fill(~phase_mask[:, :, None], 0.0)
        weights = torch.softmax(score, dim=-1) * membership.to(score.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        phase_tokens = torch.einsum("bpt,btd->bpd", weights, encoded)
        phase_tokens = self.phase_projection(phase_tokens)

        query = self.persistent_query.expand(calibration_imu.shape[0], -1, -1)
        pooled, _ = self.phase_attention(
            query, phase_tokens, phase_tokens, key_padding_mask=~phase_mask, need_weights=False
        )
        latent = self.latent_norm(pooled[:, 0])

        rotation_6d = self.rotation_head(latent).reshape(-1, N_SENSORS, 6)
        rotation_matrix = rotation_6d_to_matrix(rotation_6d)
        lever_arm = self.lever_scale_m * torch.tanh(
            self.lever_head(latent).reshape(-1, N_SENSORS, 3)
        )
        # Log-residual parameterization keeps lengths positive and gives an
        # interpretable zero initialization at the training-set mean geometry.
        bone_lengths = self.bone_length_mean * torch.exp(
            self.bone_head(latent).clamp(min=-4.0, max=4.0)
        )
        beta = self.beta_head(latent) if self.beta_head is not None else None
        return CalibrationEncoding(
            latent=latent,
            i2s_rotation_6d=rotation_6d,
            i2s_rotation_matrix=rotation_matrix,
            lever_arm=lever_arm,
            bone_lengths=bone_lengths,
            beta=beta,
            phase_attention=weights,
            phase_mask=phase_mask,
        )

    def parameter_count(self, trainable_only: bool = True) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)


def synthetic_unit_test(device: str = "cpu") -> Dict[str, float]:
    """Run shape, SO(3), correction-convention, padding, and gradient checks."""

    torch.manual_seed(20261013)
    dev = torch.device(device)
    model = CalibrationEncoder(hidden_dim=32, latent_dim=64, max_phases=4).to(dev)
    batch, steps = 3, 17
    lengths = torch.tensor([17, 13, 9], device=dev)
    mask = torch.arange(steps, device=dev)[None] < lengths[:, None]
    phase_ids = (torch.arange(steps, device=dev)[None] // 5).expand(batch, -1).clone()
    phase_ids = phase_ids.masked_fill(~mask, -1)
    imu = torch.randn(batch, steps, IMU_DIM, device=dev)
    output = model(imu, phase_ids, mask)
    assert output.latent.shape == (batch, 64)
    assert output.i2s_rotation_matrix.shape == (batch, N_SENSORS, 3, 3)
    assert output.lever_arm.shape == (batch, N_SENSORS, 3)
    assert output.bone_lengths.shape == (batch, N_BONES)
    assert output.beta is not None and output.beta.shape == (batch, 10)

    eye = torch.eye(3, device=dev)
    orthogonality = (
        output.i2s_rotation_matrix.transpose(-1, -2) @ output.i2s_rotation_matrix - eye
    ).abs().max()
    determinant_error = (torch.linalg.det(output.i2s_rotation_matrix) - 1.0).abs().max()

    true_i2s = rotation_6d_to_matrix(torch.randn(batch, N_SENSORS, 6, device=dev))
    bone = rotation_6d_to_matrix(torch.randn(batch, steps, N_SENSORS, 6, device=dev))
    measured = bone @ true_i2s[:, None]
    correction_error = (correct_i2s_orientations(measured, true_i2s) - bone).abs().max()

    target_lengths = output.bone_lengths.detach() * 1.01
    loss = rotation_geodesic_loss(output.i2s_rotation_matrix, true_i2s)
    loss = loss + bone_length_relative_loss(output.bone_lengths, target_lengths)
    loss = loss + lever_arm_loss(output.lever_arm, torch.randn_like(output.lever_arm) * 0.01)
    if output.beta is not None:
        loss = loss + output.beta.square().mean()
    loss.backward()
    gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    if orthogonality > 1e-5 or determinant_error > 1e-5 or correction_error > 2e-5:
        raise AssertionError("rotation/correction invariant failed")
    if not gradients_finite:
        raise AssertionError("non-finite encoder gradients")
    return {
        "parameters": float(model.parameter_count()),
        "orthogonality_max": float(orthogonality.detach().cpu()),
        "determinant_error_max": float(determinant_error.detach().cpu()),
        "correction_error_max": float(correction_error.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }


if __name__ == "__main__":
    print(synthetic_unit_test())

"""Independent models for the line-9 Oracle-geometry experiment.

The public :mod:`mobileposer.models.joints` implementation is intentionally not
imported here.  ``B0`` reproduces its original joint regression network:

    Linear(60, 256) -> ReLU -> Dropout(0.4) ->
    2-layer bidirectional LSTM(256, 256) -> Linear(512, 72)

``B1`` and ``oracle`` have exactly the same trainable structure.  Their only
difference is the value sent to the geometry projection: B1 always uses a
fixed, identity-independent vector while oracle uses train-set-standardized
identity geometry.  The default 33 geometry channels are SMPL beta10 followed
by the 23 non-root bone lengths.  ``geom_dim=10`` supports beta-only ablations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


VALID_MODES = ("b0", "b1", "oracle")


@dataclass(frozen=True)
class GeometryStats:
    """Per-channel statistics fitted on training identities only."""

    mean: Tensor
    std: Tensor

    def __post_init__(self) -> None:
        mean = torch.as_tensor(self.mean, dtype=torch.float32).detach().clone()
        std = torch.as_tensor(self.std, dtype=torch.float32).detach().clone()
        if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
            raise ValueError("geometry mean/std must be one-dimensional and have equal shape")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("geometry mean/std must be finite")
        if (std <= 0).any():
            raise ValueError("geometry std must be strictly positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @property
    def dim(self) -> int:
        return int(self.mean.numel())


def fit_geometry_stats(geometry: Tensor, min_std: float = 1e-6) -> GeometryStats:
    """Fit normalization statistics from one row per *training* identity.

    The caller owns selection of the training fold.  Repeated sequences from an
    identity should not be passed here because that would over-weight identities
    with more motions.  Population standard deviation is used so this also works
    for a single-row smoke test; constant channels are clamped to ``min_std``.
    """

    geometry = torch.as_tensor(geometry, dtype=torch.float32)
    if geometry.ndim != 2 or geometry.shape[0] == 0 or geometry.shape[1] == 0:
        raise ValueError("geometry must have shape [n_train_identities, geom_dim]")
    if not torch.isfinite(geometry).all():
        raise ValueError("geometry used to fit normalization must be finite")
    if min_std <= 0:
        raise ValueError("min_std must be positive")
    return GeometryStats(
        mean=geometry.mean(dim=0),
        std=geometry.std(dim=0, unbiased=False).clamp_min(float(min_std)),
    )


class Line9Joints(nn.Module):
    """B0/B1/Oracle joint regressor with a common forward contract.

    Args:
        mode: ``"b0"``, ``"b1"``, or ``"oracle"``.
        geom_dim: Number of identity-geometry channels.  The formal experiment
            uses 33 (beta10 + 23 bone lengths); 10 selects the beta-only ablation.
        geometry_stats: Required for oracle and forbidden from being inferred by
            the model.  It must have been fitted using training identities only.
        fixed_geometry: Optional identity-independent B1 value in normalized
            coordinates.  It defaults to zero and is a non-trainable buffer.

    Forward args:
        imu: Tensor of shape ``[batch, time, 60]``.
        input_lengths: Optional valid lengths for padded sequences.
        geometry: Identity-level geometry of shape ``[batch, geom_dim]``.
            Required by oracle, ignored by B0, and deliberately ignored by B1.
    """

    input_dim = 60
    hidden_dim = 256
    output_dim = 72
    n_rnn_layer = 2
    dropout_probability = 0.4

    def __init__(
        self,
        mode: str = "b0",
        geom_dim: int = 33,
        geometry_stats: Optional[GeometryStats] = None,
        fixed_geometry: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        mode = mode.lower()
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        if geom_dim <= 0:
            raise ValueError("geom_dim must be positive")
        if geometry_stats is not None and geometry_stats.dim != geom_dim:
            raise ValueError(
                f"geometry_stats has {geometry_stats.dim} channels, expected {geom_dim}"
            )
        if mode == "oracle" and geometry_stats is None:
            raise ValueError(
                "oracle requires GeometryStats fitted on training identities; "
                "implicit full-data normalization is prohibited"
            )

        self.mode = mode
        self.geom_dim = int(geom_dim)

        # Keep public-RNN-compatible parameter names and layer definitions.  In
        # particular, batch_first remains False: PackedSequence is layout-free,
        # exactly as in mobileposer.models.rnn.RNN.
        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.rnn = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.n_rnn_layer,
            bidirectional=True,
        )
        self.linear2 = nn.Linear(self.hidden_dim * 2, self.output_dim)
        self.dropout = nn.Dropout(p=self.dropout_probability)

        if mode == "b0":
            self.geometry_projection = None
        else:
            # No bias: the B1 zero vector makes this branch exactly zero, so B1
            # has Oracle's parameter count without acquiring identity information
            # or an extra learnable offset relative to B0.
            self.geometry_projection = nn.Linear(
                self.geom_dim, self.hidden_dim, bias=False
            )

        if geometry_stats is None:
            geometry_mean = torch.zeros(self.geom_dim, dtype=torch.float32)
            geometry_std = torch.ones(self.geom_dim, dtype=torch.float32)
        else:
            geometry_mean = geometry_stats.mean
            geometry_std = geometry_stats.std
        self.register_buffer("geometry_mean", geometry_mean.clone())
        self.register_buffer("geometry_std", geometry_std.clone())

        if fixed_geometry is None:
            fixed_geometry = torch.zeros(self.geom_dim, dtype=torch.float32)
        fixed_geometry = torch.as_tensor(fixed_geometry, dtype=torch.float32)
        if fixed_geometry.shape != (self.geom_dim,):
            raise ValueError(f"fixed_geometry must have shape [{self.geom_dim}]")
        if not torch.isfinite(fixed_geometry).all():
            raise ValueError("fixed_geometry must be finite")
        self.register_buffer("fixed_geometry", fixed_geometry.detach().clone())

    def _conditioning(self, imu: Tensor, geometry: Optional[Tensor]) -> Optional[Tensor]:
        if self.mode == "b0":
            return None
        if self.mode == "b1":
            # Never inspect caller-provided geometry in the control condition.
            return self.fixed_geometry.to(dtype=imu.dtype).unsqueeze(0).expand(
                imu.shape[0], -1
            )
        if geometry is None:
            raise ValueError("oracle forward requires identity geometry")
        if geometry.ndim != 2 or geometry.shape != (imu.shape[0], self.geom_dim):
            raise ValueError(
                "geometry must have shape "
                f"[{imu.shape[0]}, {self.geom_dim}], got {tuple(geometry.shape)}"
            )
        if not torch.isfinite(geometry).all():
            raise ValueError("oracle geometry must be finite")
        geometry = geometry.to(device=imu.device, dtype=imu.dtype)
        return (geometry - self.geometry_mean.to(dtype=imu.dtype)) / self.geometry_std.to(
            dtype=imu.dtype
        )

    def forward(
        self,
        imu: Tensor,
        input_lengths: Optional[Sequence[int]] = None,
        geometry: Optional[Tensor] = None,
    ) -> Tensor:
        if imu.ndim != 3 or imu.shape[-1] != self.input_dim:
            raise ValueError(
                f"imu must have shape [batch, time, {self.input_dim}], "
                f"got {tuple(imu.shape)}"
            )

        data = self.linear1(imu)
        condition = self._conditioning(imu, geometry)
        if condition is not None:
            projected = self.geometry_projection(condition)
            data = data + projected[:, None, :]
        data = self.dropout(F.relu(data))

        if input_lengths is not None:
            lengths = torch.as_tensor(input_lengths, dtype=torch.long, device="cpu")
            if lengths.ndim != 1 or lengths.numel() != imu.shape[0]:
                raise ValueError("input_lengths must contain one length per batch row")
            if (lengths <= 0).any() or (lengths > imu.shape[1]).any():
                raise ValueError("input_lengths must lie in [1, sequence_length]")
            data = pack_padded_sequence(
                data, lengths, batch_first=True, enforce_sorted=False
            )
        data, _ = self.rnn(data)
        if input_lengths is not None:
            data, _ = pad_packed_sequence(data, batch_first=True)
        return self.linear2(data)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return the scalar parameter count (buffers are deliberately excluded)."""

    parameters: Iterable[nn.Parameter] = model.parameters()
    if trainable_only:
        parameters = (p for p in parameters if p.requires_grad)
    return sum(p.numel() for p in parameters)


def parameter_count_report(models: Mapping[str, nn.Module]) -> Dict[str, int]:
    """Produce a JSON-serializable parameter-count report."""

    return {name: count_parameters(model) for name, model in models.items()}


def assert_parameter_matching(
    b0: Line9Joints, b1: Line9Joints, oracle: Line9Joints
) -> Dict[str, int]:
    """Assert the capacity-control invariants and return their counts.

    B1 and Oracle must match exactly.  Removing their geometry projection must
    leave exactly B0's parameter names and shapes, proving that the recurrent
    backbone itself is unchanged.
    """

    if (b0.mode, b1.mode, oracle.mode) != ("b0", "b1", "oracle"):
        raise AssertionError("models must be supplied in B0, B1, Oracle order")
    counts = parameter_count_report({"b0": b0, "b1": b1, "oracle": oracle})
    if counts["b1"] != counts["oracle"]:
        raise AssertionError(f"B1/Oracle parameter mismatch: {counts}")

    def backbone_shapes(model: Line9Joints) -> Dict[str, torch.Size]:
        return {
            name: value.shape
            for name, value in model.named_parameters()
            if not name.startswith("geometry_projection.")
        }

    b0_shapes = backbone_shapes(b0)
    if backbone_shapes(b1) != b0_shapes or backbone_shapes(oracle) != b0_shapes:
        raise AssertionError("B0/B1/Oracle recurrent backbone structures differ")
    expected_extra = b1.geom_dim * b1.hidden_dim
    if counts["b1"] - counts["b0"] != expected_extra:
        raise AssertionError(
            f"unexpected conditioning parameter count: {counts}; "
            f"expected {expected_extra} extra"
        )
    return counts


def build_fair_models(
    geometry_stats: GeometryStats,
    fixed_geometry: Optional[Tensor] = None,
) -> Dict[str, Line9Joints]:
    """Construct and validate a formal B0/B1/Oracle model triplet."""

    geom_dim = geometry_stats.dim
    models = {
        "b0": Line9Joints(mode="b0", geom_dim=geom_dim),
        "b1": Line9Joints(
            mode="b1", geom_dim=geom_dim, fixed_geometry=fixed_geometry
        ),
        "oracle": Line9Joints(
            mode="oracle", geom_dim=geom_dim, geometry_stats=geometry_stats
        ),
    }
    assert_parameter_matching(models["b0"], models["b1"], models["oracle"])
    return models


__all__ = [
    "GeometryStats",
    "Line9Joints",
    "VALID_MODES",
    "assert_parameter_matching",
    "build_fair_models",
    "count_parameters",
    "fit_geometry_stats",
    "parameter_count_report",
]

"""Evaluation utilities for the line 9 geometry-oracle experiment.

Errors are Euclidean joint-position errors.  The default input unit is metres
and the reported unit is centimetres.  Confidence intervals resample people,
not windows or frames, so repeated windows from the same person never act as
independent bootstrap samples.

This module is deliberately independent of the public ``Joints`` model.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch


SENSED_JOINTS: Tuple[int, ...] = (18, 19, 4, 5, 0)
UNSENSED_JOINTS: Tuple[int, ...] = tuple(
    joint for joint in range(24) if joint not in SENSED_JOINTS
)
JOINT_GROUPS: Mapping[str, Tuple[int, ...]] = OrderedDict(
    [
        ("all", tuple(range(24))),
        ("unsensed", UNSENSED_JOINTS),
        ("shoulder", (16, 17)),
        ("elbow", (18, 19)),
        ("hip", (1, 2)),
        ("knee", (4, 5)),
    ]
)

ArrayLike = Union[np.ndarray, torch.Tensor]


def _as_joints(value: ArrayLike, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)
    if tensor.ndim == 3 and tensor.shape[-1] == 72:
        tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], 24, 3)
    if tensor.ndim != 4 or tuple(tensor.shape[-2:]) != (24, 3):
        raise ValueError(
            f"{name} must have shape [B,T,72] or [B,T,24,3], got "
            f"{tuple(tensor.shape)}"
        )
    return tensor


def _as_joint_mask(mask: ArrayLike, batch: int, frames: int) -> torch.Tensor:
    """Return a boolean [B,T,24] joint-validity mask.

    A coordinate-level mask marks a joint valid only when all three coordinates
    are valid.  This conservative rule avoids silently evaluating partial 3-D
    points.
    """
    value = torch.as_tensor(mask).detach().to(device="cpu")
    if value.shape == (batch, frames):
        value = value[..., None].expand(batch, frames, 24)
    elif value.shape == (batch, frames, 1):
        value = value.expand(batch, frames, 24)
    elif value.shape == (batch, frames, 24, 3):
        value = value.to(dtype=torch.bool).all(dim=-1)
    elif value.shape != (batch, frames, 24):
        raise ValueError(
            "mask must have shape [B,T], [B,T,1], [B,T,24], or "
            f"[B,T,24,3], got {tuple(value.shape)}"
        )
    return value.to(dtype=torch.bool)


def _normalise_identities(identities: Iterable[object], batch: int) -> Tuple[str, ...]:
    if isinstance(identities, (str, bytes)):
        values = (str(identities),)
    else:
        values = tuple(str(identity) for identity in identities)
    if len(values) != batch:
        raise ValueError(f"expected {batch} identities, got {len(values)}")
    if any(not identity for identity in values):
        raise ValueError("identity labels must be non-empty")
    return values


def _bootstrap_mean_ci(
    values: np.ndarray, n_bootstrap: int, seed: int, confidence: float
) -> Tuple[float, float]:
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("bootstrap needs at least one identity value")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if len(values) == 1:
        point = float(values[0])
        return point, point
    rng = np.random.default_rng(seed)
    # Identity is the sampling unit.  Each sampled identity retains all of its
    # frames/windows through the already-computed identity mean.
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_means, (tail, 1.0 - tail))
    return float(low), float(high)


class PositionEvaluationAccumulator:
    """Streaming, padding-safe joint-position evaluator.

    Call :meth:`update` once per batch and :meth:`compute` after the complete
    fold.  Multiple sequences/windows with the same identity are pooled before
    identity-level means and bootstrap confidence intervals are computed.
    """

    def __init__(self, output_scale: float = 100.0, unit: str = "cm") -> None:
        if output_scale <= 0:
            raise ValueError("output_scale must be positive")
        self.output_scale = float(output_scale)
        self.unit = unit
        self._sum: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._count: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._frames: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._sequences: Dict[str, int] = defaultdict(int)

    def update(
        self,
        prediction: ArrayLike,
        target: ArrayLike,
        mask: ArrayLike,
        identities: Iterable[object],
    ) -> None:
        pred = _as_joints(prediction, "prediction")
        truth = _as_joints(target, "target")
        if pred.shape != truth.shape:
            raise ValueError(
                f"prediction and target shapes differ: {tuple(pred.shape)} vs "
                f"{tuple(truth.shape)}"
            )
        batch, frames = pred.shape[:2]
        valid = _as_joint_mask(mask, batch, frames)
        # Broadcast frame masks can be shared-storage views, hence out-of-place.
        valid = valid & torch.isfinite(pred).all(dim=-1)
        valid = valid & torch.isfinite(truth).all(dim=-1)
        error = torch.linalg.vector_norm(pred - truth, dim=-1) * self.output_scale
        labels = _normalise_identities(identities, batch)

        for row, identity in enumerate(labels):
            self._sequences[identity] += 1
            for group, joints in JOINT_GROUPS.items():
                index = torch.as_tensor(joints, dtype=torch.long)
                group_valid = valid[row].index_select(-1, index)
                group_error = error[row].index_select(-1, index)
                count = int(group_valid.sum().item())
                if count == 0:
                    continue
                self._sum[identity][group] += float(group_error[group_valid].sum().item())
                self._count[identity][group] += count
                self._frames[identity][group] += int(group_valid.any(dim=-1).sum().item())

    def compute(
        self,
        n_bootstrap: int = 10_000,
        seed: int = 20261013,
        confidence: float = 0.95,
    ) -> dict:
        if not self._sequences:
            raise ValueError("cannot compute metrics before any batches are added")

        per_identity: Dict[str, dict] = OrderedDict()
        for identity in sorted(self._sequences):
            identity_metrics = OrderedDict()
            for group in JOINT_GROUPS:
                count = self._count[identity].get(group, 0)
                if count:
                    identity_metrics[group] = {
                        "mean": self._sum[identity][group] / count,
                        "n_joint_frames": count,
                        "n_frames": self._frames[identity][group],
                    }
            per_identity[identity] = {
                "n_sequences": self._sequences[identity],
                "metrics": identity_metrics,
            }

        aggregate = OrderedDict()
        for group_index, group in enumerate(JOINT_GROUPS):
            usable = [
                identity
                for identity in per_identity
                if group in per_identity[identity]["metrics"]
            ]
            if not usable:
                continue
            identity_values = np.asarray(
                [per_identity[i]["metrics"][group]["mean"] for i in usable],
                dtype=np.float64,
            )
            total_sum = sum(self._sum[i][group] for i in usable)
            total_count = sum(self._count[i][group] for i in usable)
            # A different deterministic stream per group avoids accidental CI
            # coupling while preserving exact reproducibility.
            ci = _bootstrap_mean_ci(
                identity_values, n_bootstrap, seed + group_index, confidence
            )
            aggregate[group] = {
                # Primary point estimate matches the identity-bootstrap estimand.
                "mean": float(identity_values.mean()),
                "ci95": [ci[0], ci[1]],
                "pooled_mean": float(total_sum / total_count),
                "n_identities": len(usable),
                "n_joint_frames": int(total_count),
                "n_frames": int(sum(self._frames[i][group] for i in usable)),
            }

        return {
            "unit": self.unit,
            "bootstrap": {
                "sampling_unit": "identity",
                "n_resamples": int(n_bootstrap),
                "confidence": float(confidence),
                "seed": int(seed),
            },
            "joint_groups": {name: list(joints) for name, joints in JOINT_GROUPS.items()},
            "aggregate": aggregate,
            "per_identity": per_identity,
        }


def evaluate_positions(
    prediction: ArrayLike,
    target: ArrayLike,
    mask: ArrayLike,
    identities: Iterable[object],
    n_bootstrap: int = 10_000,
    seed: int = 20261013,
    confidence: float = 0.95,
    output_scale: float = 100.0,
    unit: str = "cm",
) -> dict:
    """Evaluate one tensor collection; see :class:`PositionEvaluationAccumulator`."""
    evaluator = PositionEvaluationAccumulator(output_scale=output_scale, unit=unit)
    evaluator.update(prediction, target, mask, identities)
    return evaluator.compute(n_bootstrap, seed, confidence)


def paired_difference(
    minuend: Mapping[str, object],
    subtrahend: Mapping[str, object],
    minuend_name: str,
    subtrahend_name: str,
    n_bootstrap: int = 10_000,
    seed: int = 20261013,
    confidence: float = 0.95,
    strict_identities: bool = True,
) -> dict:
    """Compute ``minuend - subtrahend`` using matched identity means.

    Thus negative Oracle-B0 values mean that Oracle has lower error.  In strict
    mode (the default), mismatched identity sets raise instead of silently
    changing the comparison population.
    """
    if minuend.get("unit") != subtrahend.get("unit"):
        raise ValueError("paired reports use different units")
    left = minuend.get("per_identity")
    right = subtrahend.get("per_identity")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ValueError("reports must contain per_identity mappings")
    left_ids, right_ids = set(left), set(right)
    if strict_identities and left_ids != right_ids:
        raise ValueError(
            "paired reports have different identity sets: "
            f"only {minuend_name}={sorted(left_ids - right_ids)}, "
            f"only {subtrahend_name}={sorted(right_ids - left_ids)}"
        )

    common = sorted(left_ids & right_ids)
    metrics = OrderedDict()
    per_identity = OrderedDict((identity, OrderedDict()) for identity in common)
    for group_index, group in enumerate(JOINT_GROUPS):
        group_ids = [
            identity
            for identity in common
            if group in left[identity]["metrics"] and group in right[identity]["metrics"]
        ]
        if not group_ids:
            continue
        deltas = np.asarray(
            [
                left[i]["metrics"][group]["mean"]
                - right[i]["metrics"][group]["mean"]
                for i in group_ids
            ],
            dtype=np.float64,
        )
        ci = _bootstrap_mean_ci(deltas, n_bootstrap, seed + group_index, confidence)
        metrics[group] = {
            "mean_difference": float(deltas.mean()),
            "ci95": [ci[0], ci[1]],
            "n_identities": len(group_ids),
        }
        for identity, delta in zip(group_ids, deltas):
            per_identity[identity][group] = float(delta)

    return {
        "comparison": f"{minuend_name}-{subtrahend_name}",
        "interpretation": f"negative values favour {minuend_name}",
        "unit": minuend.get("unit"),
        "bootstrap": {
            "sampling_unit": "paired_identity_difference",
            "n_resamples": int(n_bootstrap),
            "confidence": float(confidence),
            "seed": int(seed),
        },
        "metrics": metrics,
        "per_identity": per_identity,
    }


def oracle_control_differences(
    reports: Mapping[str, Mapping[str, object]],
    n_bootstrap: int = 10_000,
    seed: int = 20261013,
    confidence: float = 0.95,
    strict_identities: bool = True,
) -> dict:
    """Return the protocol-required Oracle-B0 and B1-B0 paired comparisons."""
    missing = {"b0", "b1", "oracle"} - set(reports)
    if missing:
        raise ValueError(f"missing model reports: {sorted(missing)}")
    return OrderedDict(
        [
            (
                "oracle-b0",
                paired_difference(
                    reports["oracle"], reports["b0"], "oracle", "b0",
                    n_bootstrap, seed, confidence, strict_identities,
                ),
            ),
            (
                "b1-b0",
                paired_difference(
                    reports["b1"], reports["b0"], "b1", "b0",
                    n_bootstrap, seed + len(JOINT_GROUPS), confidence,
                    strict_identities,
                ),
            ),
        ]
    )


__all__ = [
    "JOINT_GROUPS",
    "SENSED_JOINTS",
    "UNSENSED_JOINTS",
    "PositionEvaluationAccumulator",
    "evaluate_positions",
    "paired_difference",
    "oracle_control_differences",
]

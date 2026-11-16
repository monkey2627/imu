"""DEPRECATED (2026-11-14): no longer called by run.py. Judging leg extension from the network knee
angle was wrong in both directions; see production/deadbug_legs.py and docs/13_死虫腿部重建排查_20261114.md.

Dead-bug-specific pose constraints for ambiguous five-IMU reconstruction."""
from __future__ import annotations

import math

import numpy as np
import torch

from production.arm_prior import PARENTS, _globals, _rotation_between, _scaled

LEGS = ((1, 4), (2, 5))  # hip/thigh joint, knee/shank joint
LEG_AXIS = np.array([0.0, -1.0, 0.0])


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


class DeadBugLegPrior:
    """Debounced, rate-limited dead-bug leg extension prior.

    A five-IMU layout has no thigh sensor, so a tabletop shank and a lowered
    straight-leg shank can look almost identical. The network's knee angle is
    used only as a weak state cue: one side must be clearly more extended than
    the other for several frames before correction starts. The correction then
    fades in instead of snapping the thigh onto the shank.
    """

    ENTER_FLEX_DEG = 55.0
    ENTER_SEPARATION_DEG = 10.0
    ENTER_SECONDS = 0.20
    EXIT_FLEX_DEG = 72.0
    EXIT_SECONDS = 0.15
    FILTER_SECONDS = 0.12
    ATTACK_PER_SECOND = 1.35  # about 0.75 s from zero to full strength
    RELEASE_PER_SECOND = 2.7
    MAX_STRENGTH = 0.88

    def __init__(self):
        self.reset()

    def reset(self):
        self.flex_ema = np.array([90.0, 90.0], dtype=np.float64)
        self.weights = np.zeros(2, dtype=np.float64)
        self.enter_times = np.zeros(2, dtype=np.float64)
        self.exit_time = 0.0
        self.active = -1
        self.initialized = False

    def apply(self, local, return_weights: bool = False, dt_seconds: float = 1.0 / 30.0):
        is_torch = isinstance(local, torch.Tensor)
        pose = (local.detach().cpu().numpy().astype(np.float64)
                if is_torch else np.asarray(local, np.float64))
        g = _globals(pose)
        flexes = np.empty(2, dtype=np.float64)
        for side, (hip, knee) in enumerate(LEGS):
            thigh_dir = g[hip] @ LEG_AXIS
            shank_dir = g[knee] @ LEG_AXIS
            flexes[side] = np.degrees(np.arccos(np.clip(thigh_dir @ shank_dir, -1.0, 1.0)))

        dt = float(np.clip(dt_seconds, 1.0 / 240.0, 0.2))
        if not self.initialized:
            self.flex_ema[:] = flexes
            self.initialized = True
        else:
            alpha = 1.0 - math.exp(-dt / self.FILTER_SECONDS)
            self.flex_ema += alpha * (flexes - self.flex_ema)

        if self.active < 0:
            candidate = int(np.argmin(self.flex_ema))
            other = 1 - candidate
            qualifies = (
                self.flex_ema[candidate] < self.ENTER_FLEX_DEG
                and self.flex_ema[other] - self.flex_ema[candidate] > self.ENTER_SEPARATION_DEG
            )
            if qualifies:
                self.enter_times[candidate] += dt
                self.enter_times[other] = 0.0
                if self.enter_times[candidate] >= self.ENTER_SECONDS:
                    self.active = candidate
                    self.enter_times[:] = 0.0
                    self.exit_time = 0.0
            else:
                self.enter_times[:] = 0.0
        else:
            if self.flex_ema[self.active] > self.EXIT_FLEX_DEG:
                self.exit_time += dt
                if self.exit_time >= self.EXIT_SECONDS:
                    self.active = -1
                    self.exit_time = 0.0
                    self.enter_times[:] = 0.0
            else:
                self.exit_time = 0.0

        targets = np.zeros(2, dtype=np.float64)
        if self.active >= 0:
            # Start gently near 65 degrees and reserve the strongest correction
            # for a clearly straight prediction (roughly 35 degrees or less).
            progress = (65.0 - self.flex_ema[self.active]) / 30.0
            targets[self.active] = self.MAX_STRENGTH * _smoothstep(progress)

        for side in range(2):
            rate = self.ATTACK_PER_SECOND if targets[side] > self.weights[side] else self.RELEASE_PER_SECOND
            max_step = rate * dt
            self.weights[side] += float(np.clip(targets[side] - self.weights[side], -max_step, max_step))

        out = pose.copy()
        for side, (hip, knee) in enumerate(LEGS):
            weight = float(self.weights[side])
            if weight <= 1e-6:
                continue
            thigh_dir = g[hip] @ LEG_AXIS
            shank_dir = g[knee] @ LEG_AXIS
            swing = _scaled(_rotation_between(thigh_dir, shank_dir), weight)
            new_thigh = swing @ g[hip]
            parent = g[PARENTS[hip]]
            out[hip] = parent.T @ new_thigh
            # Preserve the measured/model shank global rotation after moving thigh.
            out[knee] = new_thigh.T @ g[knee]

        if is_torch:
            out = torch.from_numpy(out).to(local)
        weights = self.weights.tolist()
        return (out, weights) if return_weights else out


def straighten_extending_legs(local, return_weights: bool = False):
    """Compatibility wrapper for one-off callers.

    Real-time and replay code retain a DeadBugLegPrior instance across frames.
    A single ambiguous frame intentionally receives no correction.
    """
    return DeadBugLegPrior().apply(local, return_weights=return_weights)

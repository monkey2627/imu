"""Curl-up contacts and temporal upper-arm stabilization.

The pelvis/buttocks remain on the mat while the shoulders are free to rise.  Planted feet use the
same geometry as the bridge solver, but bridge shoulder/arm floor contacts are deliberately disabled.
Forearm orientations are sensed; only the unobserved upper arms are rate-limited, with the elbow
counter-rotated so the sensed forearm global orientation is preserved.
"""
from __future__ import annotations

import math

import numpy as np

from production.arm_prior import PARENTS, _globals, _scaled
from production.bridge_contact import BridgeContact, GATE_SUPINE, _ramp

CURLUP_MODELS = frozenset({"waist_imudino_stream_curlup"})
ARM_TIME_CONSTANT_S = 0.10
ARM_MAX_SPEED_DEG_S = 240.0


class CurlUpContact:
    def __init__(self) -> None:
        self.contacts = BridgeContact(pelvis_only=True, floor_arms=False)
        self.reset("init")

    def reset(self, reason: str) -> None:
        self.contacts.reset(reason)
        self.previous_upper = {"left": None, "right": None}
        self.reason = reason

    def apply(self, local: np.ndarray, dt: float):
        local, contact = self.contacts.apply(local, dt)
        g = _globals(local)
        supine = _ramp(float(g[0][1, 2]), *GATE_SUPINE)
        arm_diag = {}
        for side, collar, shoulder, elbow in (("left", 13, 16, 18), ("right", 14, 17, 19)):
            current = g[shoulder]
            previous = self.previous_upper[side]
            raw_delta = 0.0
            applied_delta = 0.0
            corrected = current
            if previous is not None and supine > 0.0:
                delta = current @ previous.T
                raw_delta = _angle_deg(delta)
                alpha = 1.0 - math.exp(-dt / ARM_TIME_CONSTANT_S)
                if raw_delta > 1e-6:
                    max_weight = math.radians(ARM_MAX_SPEED_DEG_S) * dt / math.radians(raw_delta)
                    alpha = min(alpha, max_weight)
                smoothed = _scaled(delta, float(np.clip(alpha, 0.0, 1.0))) @ previous
                corrected = _scaled(smoothed @ current.T, supine) @ current
                applied_delta = _angle_deg(corrected @ previous.T)
                forearm = g[elbow]
                local[shoulder] = g[collar].T @ corrected
                local[elbow] = corrected.T @ forearm
            self.previous_upper[side] = corrected.copy()
            arm_diag[side] = {"rawDeltaDeg": round(raw_delta, 2),
                              "appliedDeltaDeg": round(applied_delta, 2)}
        contact["arms"] = arm_diag
        contact["upperArmSmoothing"] = round(supine, 3)
        return local, contact


def _angle_deg(rotation: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))

"""Experimental lying-cat rule: keep the waist/spine unconstrained.

Only the sensed support limbs are brought toward the floor.  This is an
offline comparison variant; it is intentionally not wired into production.
"""
from __future__ import annotations

import math
import numpy as np

from production.lyingcat_contact import LyingCatContact as Base
from production.arm_prior import _globals
from production.lyingcat_contact import ARMS, LEGS, UP, _elevation


class LyingCatContact(Base):
    def apply(self, local: np.ndarray, dt: float, measured=None, labels=None):
        local = self._anchor(np.asarray(local, dtype=np.float64).copy(), measured, labels)
        g = _globals(local)
        elevations = self._update_weights(g, dt)
        before = self._support_heights(local)

        # Keep the measured waist/pelvis and the network spine exactly as-is.
        # Only level sensed shanks and lay the palms/knees onto the support plane.
        self._level_legs(local)
        g = _globals(local)
        sensed = {joint: g[joint].copy() for joint in (18, 19)}
        _, floor = self._support_heights(local)
        if floor is not None:
            # The base residual solver only changes support segments, but its
            # Arm path also adjusts the upper arm.  That is acceptable here:
            # it places the palm without touching the waist/spine joints.
            self._close_residual(local, floor)
        self._flatten_hands(local)
        self._restore(local, sensed)

        g = _globals(local)
        p = self._positions(g)
        diag = {
            "enabled": True,
            "weights": {key: round(value, 3) for key, value in self.weights.items()},
            "spineDeg": 0.0,
            "spineTargetDeg": 0.0,
            "pelvisPitchDeg": {"network": round(_elevation(g[0] @ UP), 1),
                               "applied": round(_elevation(g[0] @ UP), 1)},
            "sensorElevationDeg": elevations,
            "noWaistConstraint": True,
        }
        if before[0] is not None and before[1] is not None:
            diag["handKneeGapBeforeCm"] = round(100.0 * (before[0] - before[1]), 1)
        if floor is not None:
            diag["supportPlaneM"] = round(floor, 3)
            diag["afterErrorCm"] = {
                name: round(100.0 * float(p[joint, 1] - floor), 1)
                for name, joint, key in (("leftWrist", 20, "leftArm"), ("rightWrist", 21, "rightArm"),
                                         ("leftKnee", 4, "leftLeg"), ("rightKnee", 5, "rightLeg"))
                if self.weights[key] > 0.5
            }
        return local, diag

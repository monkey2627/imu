"""Correct the sensor-frame error in SlimeVR's linear acceleration.

Evidence (docs/13_死虫腿部重建排查_20261114.md): integrating the shank and
forearm accelerations between still periods (zero-velocity update) gives
displacements that contradict the measured orientations -- a dead-bug leg
extension, whose ankle must move toward the feet and down by ~0.5 m, integrates
to ~0.6 m toward the head and up; a bird-dog leg lift integrates to the ankle
going *through the floor*.  Scoring all 48 axis permutations/sign flips of the
tracker-local acceleration against these physically known displacements picks
the same correction for every limb sensor: a 180 degree rotation about the
tracker's raw z axis (``diag(-1, -1, 1)`` in the firmware/sensor frame).  It
holds on the 2026-11-01 bird-dog recordings (different tracker ids) and the
2026-11-13 dead-bug recordings.

SolarXR gives both acceleration fields from the same tracker-local vector::

    accel_xyz                      = raw_rot * a_local          (legacy)
    reference_adjusted_accel_xyz   = L * raw_rot * a_local      (L: world yaw)

``L`` is a pure yaw rotation (exactly equal y components; constant per reset),
so it is recovered per frame from the two horizontal components.  The
corrected reference-adjusted acceleration is then
``L * raw_rot * Rz(180) * raw_rot^T * accel_xyz``.

This only changes linear acceleration.  Orientations, SlimeVR's own
consumers, the raw gravity-including acceleration used by the bird-dog
calibration and every recorded file are untouched.
"""
from __future__ import annotations

import numpy as np

SENSOR_FRAME_FIX = np.diag([-1.0, -1.0, 1.0])
_MIN_HORIZONTAL = 1e-5


def _quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    if not np.isfinite(n) or n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def _yaw_between(legacy: np.ndarray, adjusted: np.ndarray):
    """Yaw angle (about +y) mapping legacy -> adjusted, or None when undetermined."""
    lh = np.hypot(legacy[0], legacy[2])
    ah = np.hypot(adjusted[0], adjusted[2])
    if lh < _MIN_HORIZONTAL or ah < _MIN_HORIZONTAL:
        return None
    cross = legacy[2] * adjusted[0] - legacy[0] * adjusted[2]
    dot = legacy[0] * adjusted[0] + legacy[2] * adjusted[2]
    return float(np.arctan2(cross, dot))


class AccelerationFrameFix:
    """Per-tracker corrector; remembers the last determinable world yaw."""

    def __init__(self) -> None:
        self._yaw = {}

    def reset(self) -> None:
        self._yaw.clear()

    def correct(self, key: str, raw_quat_xyzw, legacy_xyz, adjusted_xyz):
        """Corrected reference-adjusted (server-frame) acceleration, or the input when not computable."""
        if adjusted_xyz is None:
            return None
        adjusted = np.asarray(adjusted_xyz, dtype=np.float64)
        if raw_quat_xyzw is None or legacy_xyz is None:
            return adjusted
        legacy = np.asarray(legacy_xyz, dtype=np.float64)
        yaw = _yaw_between(legacy, adjusted)
        if yaw is None:
            yaw = self._yaw.get(key)
            if yaw is None:
                return adjusted          # zero horizontal acceleration: nothing to rotate yet
        else:
            self._yaw[key] = yaw
        c, s = np.cos(yaw), np.sin(yaw)
        world_yaw = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        raw = _quat_xyzw_to_matrix(raw_quat_xyzw)
        return world_yaw @ raw @ SENSOR_FRAME_FIX @ raw.T @ legacy

    def correct_sample(self, sample) -> np.ndarray | None:
        """TrackerSample (dataclass) or recorded sample dict."""
        get = sample.get if isinstance(sample, dict) else lambda name: getattr(sample, name)
        key = get("tracker_key")
        return self.correct(key, get("raw_quat_xyzw"), get("accel_xyz"), get("reference_adjusted_accel_xyz"))

"""Forearm heading from the action itself (bird dog, dead bug): a level, still forearm is the reach,
and the reach points to the head (bird dog: forward from the four-point stance; dead bug: overhead
while lying on the back).

From the vertical start-pose forearm, world-yaw drift and the hand's twist are the same rotation, so
the start-pose fit (start_pose_fit.py) cannot separate them; the first held reach can.  Whenever a
forearm is held level and still for HOLD_FRAMES in the action's trunk posture, its world heading is
turned so that it points to the trunk's head direction.  The turn is spread over the following
frames (rate-limited) to avoid a visible jump.  Shared by run.py and production/live_replay.py.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

from production.pose_calibration import BONE_AXES, _yaw

# Action -> trunk posture in which a held, level forearm is the reach toward the head:
# bird dog (four-point, arm forward), dead bug (lying on the back, arm overhead).
ACTIONS = {"birddog": "four_point", "deadbug": "supine"}
HOLD_FRAMES = 30            # 0.5 s at 60 Hz
MAX_LEVEL_DEG = 20.0        # forearm elevation for a reach
STILL_DEG = 6.0
MIN_ERROR_DEG = 5.0         # ignore smaller heading errors
RATE_DEG_PER_FRAME = 3.0    # 180 deg/s while applying a correction
MAX_FOLLOWUP_DEG = 45.0     # after the first reach of a forearm: larger errors are not reaches (live dead
                            # bug 2026-11-19: arms laid along the body at the end read as 171-178 deg)


def _signed_yaw(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (rad) about +y turning the horizontal part of a onto that of b."""
    return float(np.arctan2(a[2] * b[0] - a[0] * b[2], a[0] * b[0] + a[2] * b[2]))


class ReachHeading:
    def __init__(self, labels):
        self.labels = list(labels)
        self.reset()

    def reset(self):
        self.action = None
        self.windows = {label: deque(maxlen=HOLD_FRAMES) for label in ("lw", "rw")}
        self.pending = {"lw": 0.0, "rw": 0.0}     # radians still to apply
        self.aligned = set()                      # forearms already aligned by a reach in this set
        self.events = []

    def activate(self, action):
        self.reset()
        self.action = action

    def update(self, calibrator, raw):
        """After each 60 Hz frame: `raw` is the frame's (n,3,3) uncorrected orientation."""
        if self.action not in ACTIONS or not calibrator.calibration_enabled or "waist" not in self.labels:
            return
        w = self.labels.index("waist")
        waist = calibrator.headings[w] @ raw[w] @ calibrator.corrections[w]
        chest_up = waist[1, 2]
        spine_level = abs(waist[1, 1]) < 0.5
        posture_ok = spine_level and (chest_up < -0.65 if ACTIONS[self.action] == "four_point" else chest_up > 0.65)
        head = waist @ np.array([0.0, 1.0, 0.0])
        for label, window in self.windows.items():
            if label not in self.labels:
                continue
            i = self.labels.index(label)
            self._apply_pending(calibrator, i, label)
            corrected = calibrator.headings[i] @ raw[i] @ calibrator.corrections[i]
            window.append(corrected)
            if not posture_ok or len(window) < HOLD_FRAMES or self.pending[label] != 0.0:
                continue
            rotations = Rotation.from_matrix(np.asarray(window))
            mean = rotations.mean()
            if np.degrees((mean.inv() * rotations).magnitude().max()) > STILL_DEG:
                continue
            direction = mean.as_matrix() @ BONE_AXES[label]
            if abs(np.degrees(np.arcsin(np.clip(direction[1], -1.0, 1.0)))) > MAX_LEVEL_DEG:
                continue
            error = _signed_yaw(direction, head)
            error_deg = abs(np.degrees(error))
            if label in self.aligned and error_deg > MAX_FOLLOWUP_DEG:
                # After the first reach only slow drift is corrected.  A level, still forearm far from the
                # head direction is the arm put down along the body (dead bug: hands to the feet), not a
                # reach -- toward the feet and overhead differ only by the heading this would "correct".
                self.events.append({"label": label, "errorDeg": round(float(np.degrees(error)), 1),
                                    "ignored": True})
            elif error_deg >= MIN_ERROR_DEG:
                self.pending[label] = error
                self.events.append({"label": label, "errorDeg": round(float(np.degrees(error)), 1)})
            if label not in self.aligned or error_deg <= MAX_FOLLOWUP_DEG:
                self.aligned.add(label)
            window.clear()

    def _apply_pending(self, calibrator, i, label):
        remaining = self.pending[label]
        if remaining == 0.0:
            return
        step = float(np.clip(remaining, -np.radians(RATE_DEG_PER_FRAME), np.radians(RATE_DEG_PER_FRAME)))
        calibrator.headings[i] = _yaw(step) @ calibrator.headings[i]
        self.pending[label] = 0.0 if abs(remaining - step) < 1e-9 else remaining - step
        self.windows[label].clear()

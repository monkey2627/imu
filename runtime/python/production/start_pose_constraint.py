"""Explicit priors for unobserved joints in a selected exercise start pose."""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from production.pose_calibration import SMPL_PARENTS


LEG_AXIS = np.array([0.0, -1.0, 0.0], dtype=np.float64)
TABLETOP_ACTIONS = frozenset({"deadbug"})
MOVEMENT_DEG = 12.0
RELEASE_PER_SECOND = 2.0


def _globals(local):
    result = []
    for joint, parent in enumerate(SMPL_PARENTS):
        result.append(local[joint] if parent < 0 else result[parent] @ local[joint])
    return result


def _rotation_between(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= max(np.linalg.norm(source), 1e-12)
    target /= max(np.linalg.norm(target), 1e-12)
    cross = np.cross(source, target)
    norm = np.linalg.norm(cross)
    dot = float(np.clip(source @ target, -1.0, 1.0))
    if norm < 1e-9:
        if dot > 0.0:
            return np.eye(3)
        reference = np.array([1.0, 0.0, 0.0]) if abs(source[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, reference)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(axis * np.pi).as_matrix()
    return Rotation.from_rotvec(cross / norm * math.atan2(norm, dot)).as_matrix()


def _scaled(rotation, weight):
    return Rotation.from_rotvec(Rotation.from_matrix(rotation).as_rotvec() * float(weight)).as_matrix()


def _angle_deg(left, right):
    return float(np.degrees(Rotation.from_matrix(np.asarray(left).T @ np.asarray(right)).magnitude()))


class StartPoseConstraint:
    """Hold ambiguous dead-bug thighs and upper arms in tabletop until movement.

    The sensed shank's measured global rotation is preserved.  Only the
    unmeasured thigh is rotated toward vertical, so the knee absorbs the known
    tabletop geometry.  The unobserved upper arms are aligned to the sensed
    forearms so shoulder, elbow and wrist form one upward-reaching chain.
    Each limb releases independently when it moves away from the start hold.
    """

    def __init__(self, labels):
        self.labels = list(labels)
        self.clear()

    def clear(self):
        self.action = None
        self.start_pose = None
        self.weights = np.zeros(2, dtype=np.float64)
        self.arm_weights = np.zeros(2, dtype=np.float64)
        self.reference = None

    def activate(self, action, start_pose):
        self.clear()
        self.action = action
        self.start_pose = start_pose
        if (action in TABLETOP_ACTIONS and start_pose == "tabletop"
                and {"ls", "rs"}.issubset(self.labels)):
            self.weights[:] = 1.0
            if {"lw", "rw"}.issubset(self.labels):
                self.arm_weights[:] = 1.0

    @property
    def enabled(self):
        return bool(np.any(self.weights > 1e-6) or np.any(self.arm_weights > 1e-6))

    def apply(self, local, measured, dt_seconds=1.0 / 30.0):
        is_torch = isinstance(local, torch.Tensor)
        pose = local.detach().cpu().numpy().astype(np.float64) if is_torch else np.asarray(local, dtype=np.float64)
        diag = {"action": self.action, "startPose": self.start_pose,
                "enabled": self.enabled, "weights": self.weights.round(3).tolist(),
                "armWeights": self.arm_weights.round(3).tolist(),
                "movementDeg": [0.0, 0.0], "armMovementDeg": [0.0, 0.0]}
        if not self.enabled or measured is None:
            return (local, diag)

        measured = np.asarray(measured, dtype=np.float64)
        if self.reference is None:
            self.reference = {label: measured[self.labels.index(label)].copy()
                              for label in ("lw", "rw", "ls", "rs") if label in self.labels}

        pairs = (("ls", "rw"), ("rs", "lw"))
        movement = []
        for side, labels in enumerate(pairs):
            angles = [_angle_deg(self.reference[label], measured[self.labels.index(label)])
                      for label in labels if label in self.reference]
            amount = max(angles, default=0.0)
            movement.append(amount)
            if amount >= MOVEMENT_DEG:
                self.weights[side] = max(0.0, self.weights[side] - RELEASE_PER_SECOND * dt_seconds)

        arm_movement = []
        for side, label in enumerate(("lw", "rw")):
            amount = (_angle_deg(self.reference[label], measured[self.labels.index(label)])
                      if label in self.reference else 0.0)
            arm_movement.append(amount)
            if amount >= MOVEMENT_DEG:
                self.arm_weights[side] = max(
                    0.0, self.arm_weights[side] - RELEASE_PER_SECOND * dt_seconds)

        g = _globals(pose)
        out = pose.copy()
        world_up = np.array([0.0, 1.0, 0.0])
        lateral_axis = g[3][:, 0]
        for side, (hip, knee, shank_label) in enumerate(((1, 4, "ls"), (2, 5, "rs"))):
            weight = float(self.weights[side])
            if weight <= 1e-6:
                continue
            thigh = g[hip] @ LEG_AXIS
            lateral = float(np.clip(thigh @ lateral_axis, -0.25, 0.25))
            target = world_up * math.sqrt(max(0.0, 1.0 - lateral * lateral)) + lateral_axis * lateral
            target /= max(np.linalg.norm(target), 1e-12)
            swing = _scaled(_rotation_between(thigh, target), weight)
            new_thigh = swing @ g[hip]
            # The thigh is unobserved, but the shank has an IMU.  Blend its
            # global rotation toward that measurement with the same release
            # weight; otherwise a standing-biased model shank would turn the
            # intended tabletop knee into a straight leg.
            measured_shank = measured[self.labels.index(shank_label)]
            shank_correction = _scaled(g[knee].T @ measured_shank, weight)
            new_shank = g[knee] @ shank_correction
            out[hip] = g[SMPL_PARENTS[hip]].T @ new_thigh
            out[knee] = new_thigh.T @ new_shank

        # Wrist sensors observe the forearms, not the upper arms.  In the
        # tabletop start pose both segments form one straight vertical chain,
        # so explicitly align the unobserved upper arm to the sensed forearm.
        for side, (shoulder, elbow, wrist_label, axis) in enumerate((
                (16, 18, "lw", np.array([1.0, 0.0, 0.0])),
                (17, 19, "rw", np.array([-1.0, 0.0, 0.0])))):
            weight = float(self.arm_weights[side])
            if weight <= 1e-6:
                continue
            measured_forearm = measured[self.labels.index(wrist_label)]
            target = measured_forearm @ axis
            swing = _scaled(_rotation_between(g[shoulder] @ axis, target), weight)
            new_upper = swing @ g[shoulder]
            forearm_correction = _scaled(g[elbow].T @ measured_forearm, weight)
            new_forearm = g[elbow] @ forearm_correction
            out[shoulder] = g[SMPL_PARENTS[shoulder]].T @ new_upper
            out[elbow] = new_upper.T @ new_forearm

        diag.update(enabled=self.enabled, weights=self.weights.round(3).tolist(),
                    armWeights=self.arm_weights.round(3).tolist(),
                    movementDeg=[round(value, 1) for value in movement],
                    armMovementDeg=[round(value, 1) for value in arm_movement])
        if is_torch:
            out = torch.from_numpy(out).to(local)
        return out, diag

    def snapshot(self):
        return {"action": self.action, "startPose": self.start_pose,
                "weights": self.weights.tolist(),
                "armWeights": self.arm_weights.tolist(),
                "reference": None if self.reference is None else
                    {key: value.tolist() for key, value in self.reference.items()}}

    def restore(self, state):
        if not state:
            return
        self.action = state.get("action")
        self.start_pose = state.get("startPose")
        self.weights = np.asarray(state.get("weights", (0.0, 0.0)), dtype=np.float64)
        self.arm_weights = np.asarray(state.get("armWeights", (0.0, 0.0)), dtype=np.float64)
        reference = state.get("reference")
        self.reference = None if reference is None else {
            key: np.asarray(value, dtype=np.float64) for key, value in reference.items()}

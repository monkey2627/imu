"""Straight-arm prior for reaching poses, applied to the model output.

With sensors on the forearms only, a forearm pointing forward is equally consistent with a straight
arm and with the upper arm swung aside (or hanging) and the elbow bent: the readings are identical.
The training data (AMASS) mostly shows the bent version (91% of level-forward forearm frames, see
server_training_20261030/experiments/audit_arm_forward_prior.py), so the models draw it even when the user
reaches forward.

When the measured forearm (calibrated input of the frame the output represents) is level and points
along the trunk's reach axis (arms forward while standing, the bird-dog reach, the dead-bug overhead
reach), the upper arm is swung onto the forearm direction (straight elbow) and the output forearm is
moved to the measurement.  Models differ in how faithfully they reproduce the forearm, so the sensed
part is taken from the sensor.  The weight fades from 1 (within FULL_DEG) to 0 (beyond NONE_DEG) so
nothing jumps.  Without a measurement the model's own forearm is used.  Not for poses where a level
forward forearm comes with a bent elbow by design (the side-bridge support arm): switch it off there.
"""
from __future__ import annotations

import numpy as np
import torch

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
ARMS = ((16, 18, np.array([1.0, 0.0, 0.0])), (17, 19, np.array([-1.0, 0.0, 0.0])))   # shoulder, elbow, bone axis
TRUNK = 9                       # spine3: chest frame (x left, y up the spine, z out of the chest)
LEVEL_DEG = 30.0                # forearm elevation limit (world)
FULL_DEG, NONE_DEG = 20.0, 40.0  # angle to the reach axis for full / no correction


def _globals(local: np.ndarray) -> list:
    g = []
    for j, p in enumerate(PARENTS):
        g.append(local[j] if p < 0 else g[p] @ local[j])
    return g


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.clip(a @ b, -1.0, 1.0))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1.0 / (1.0 + c))


def _scaled(rotation: np.ndarray, weight: float) -> np.ndarray:
    """Rotation `rotation` scaled to `weight` of its angle (same axis)."""
    angle = np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0))
    if angle < 1e-6 or weight >= 1.0:
        return rotation if weight >= 1.0 else np.eye(3)
    axis = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]]) / (2 * np.sin(angle))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = weight * angle
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


def reach_weight(forearm_dir: np.ndarray, trunk: np.ndarray) -> float:
    """1 when the forearm is level and points, in the horizontal plane, along the trunk's reach axis:
    the chest normal when the trunk is upright (arms forward), the spine when it lies flat (the
    bird-dog reach, the dead-bug overhead reach); whichever of the two is closer to horizontal.
    Compared by azimuth only, so a model trunk that is not exactly level still counts."""
    elevation = np.degrees(np.arcsin(np.clip(forearm_dir[1], -1.0, 1.0)))
    if abs(elevation) > LEVEL_DEG:
        return 0.0
    chest, spine = trunk @ np.array([0.0, 0.0, 1.0]), trunk @ np.array([0.0, 1.0, 0.0])
    reach = chest if np.hypot(chest[0], chest[2]) >= np.hypot(spine[0], spine[2]) else spine
    azimuth = lambda v: np.arctan2(v[0], v[2])
    angle = abs(np.degrees((azimuth(forearm_dir) - azimuth(reach) + np.pi) % (2 * np.pi) - np.pi))
    return float(np.clip((NONE_DEG - angle) / (NONE_DEG - FULL_DEG), 0.0, 1.0))


def straighten_reaching_arms(local, measured_forearms=None, return_weights: bool = False):
    """(24, 3, 3) local SMPL pose (torch or numpy) -> same type with reaching arms straightened.
    measured_forearms: optional (left, right) measured global forearm rotations (SMPL joints 18/19)."""
    is_torch = isinstance(local, torch.Tensor)
    pose = local.detach().cpu().numpy().astype(np.float64) if is_torch else np.asarray(local, np.float64)
    g = _globals(pose)
    out = pose.copy()
    weights = []
    for side, (shoulder, elbow, axis) in enumerate(ARMS):
        measured = None if measured_forearms is None else np.asarray(measured_forearms[side], np.float64)
        forearm_rot = g[elbow] if measured is None else measured
        forearm = forearm_rot @ axis
        w = reach_weight(forearm, g[TRUNK])
        weights.append(w)
        if w <= 0.0:
            continue
        upper = g[shoulder]
        swing = _scaled(_rotation_between(upper @ axis, forearm), w)
        new_upper = swing @ upper
        new_forearm = g[elbow] if measured is None else _scaled(measured @ g[elbow].T, w) @ g[elbow]
        out[shoulder] = g[PARENTS[shoulder]].T @ new_upper
        out[elbow] = new_upper.T @ new_forearm
    if is_torch:
        out = torch.from_numpy(out).to(local)
    return (out, weights) if return_weights else out

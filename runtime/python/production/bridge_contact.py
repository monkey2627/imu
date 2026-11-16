"""Glute bridge (臀桥) contact constraint and lie-flat calibration helpers.

The five sensors (forearms, shanks, waist) do not observe the thighs or the chest.  Lying on the back with the
feet planted, two contacts are known: the upper back (shoulder blades) or the buttocks rest on the floor, and the
feet stay flat.  Given the network's trunk (pelvis + spine, which follow the waist sensor) and the shank
directions, this fixes

* the pelvis height:  h = max(pelvis back surface on the floor, spine3 back surface on the floor), and
* each thigh's elevation: the ankle must be at foot-flat height,
      Lt * sin(theta) = ANKLE_HEIGHT - hip_height - Ls * shank_y,
  keeping the network's thigh heading (abduction) and the shank's global orientation.

Only the thigh swing is changed.  It fades in only while lying supine with the shank pointing down to a
planted foot, and the target is low-pass filtered.

Lie-flat calibration (exercise_calibration.py, 臀桥):
* ``waist_flat_fit``: waist mounting tilt so that, lying flat, the spine axis is horizontal and the chest
  normal points up (heading kept).  Applied to bridge models only (frame_chain.prepare_frame).
* ``trunk_bias``: mean spine1/2/3 local rotation the network outputs while lying flat with the knees bent;
  removed from every output of bridge models.
"""
from __future__ import annotations

import math

import numpy as np

from production.arm_prior import PARENTS, _rotation_between, _scaled

BRIDGE_MODELS = frozenset({"waist_imudino_stream_bridge_rehabv5"})
TRUNK_JOINTS = (3, 6, 9)                  # spine1-3.  Not neck/head: removing their lying bias put the back of
                                          # the head 3-9 cm below the shoulder blades (2026-11-16, 034331).
TRUNK_BIAS_MAX_SPREAD_DEG = 12.0          # outputs must be steady during the hold
WAIST_FLAT_MAX_DEG = 35.0

# SMPL neutral body (basicmodel), zero pose: see _body_constants() for how they were measured.
PELVIS_BACK_DEPTH = 0.153                 # pelvis joint -> back surface (buttocks), along -chest normal
SPINE3_BACK_DEPTH = 0.142                 # spine3 joint -> back surface (shoulder blades)
ANKLE_HEIGHT = 0.077                      # ankle joint above the sole, foot flat
ELBOW_DEPTH = 0.036                       # elbow joint -> back of the arm (arm lying by the side, palm down)
WRIST_DEPTH = 0.035
GATE_FOREARM_DEG = 30.0                   # |forearm elevation| for an arm lying on the floor
GATE_SUPINE = (0.45, 0.70)                # pelvis chest-normal up component: off .. full
GATE_SHANK_DEG = (-88.0, -15.0)           # shank elevation (knee -> ankle) for a planted foot
WEIGHT_RATE_PER_S = 3.0
TARGET_TIME_CONSTANT_S = 0.08


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _ramp(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def mean_rotation(rotations: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rotations, dtype=np.float64).mean(0))
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    return u @ vt


def angle_deg(rotation: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0))))


def waist_flat_fit(waist: np.ndarray):
    """Right-multiplied waist correction C so that (waist @ C) lies flat on the back: local y (toward the
    head) horizontal with the same heading, local z (chest normal) straight up.  Returns (C, tilt_deg)."""
    waist = np.asarray(waist, dtype=np.float64)
    head = waist[:, 1].copy()
    head[1] = 0.0
    if np.linalg.norm(head) < 0.3:
        return None, float("nan")
    head = _unit(head)
    up = np.array([0.0, 1.0, 0.0])
    target = np.stack([np.cross(head, up), head, up], 1)
    correction = waist.T @ target
    return correction, angle_deg(correction)


def trunk_bias(outputs: np.ndarray):
    """Per trunk joint mean local rotation of (N, 24, 3, 3) outputs and the worst spread (deg)."""
    outputs = np.asarray(outputs, dtype=np.float64)
    bias, spread = {}, 0.0
    for joint in TRUNK_JOINTS:
        mean = mean_rotation(outputs[:, joint])
        bias[joint] = mean
        deviations = [angle_deg(mean.T @ r) for r in outputs[:, joint]]
        spread = max(spread, float(np.percentile(deviations, 90)))
    return bias, spread


def apply_trunk_bias(local: np.ndarray, bias) -> np.ndarray:
    """Remove the lie-flat output bias.  spine1's global orientation is the waist measurement (root =
    waist @ spine1^T), so its bias is moved into the pelvis instead: spine1 global and the hip globals are
    kept, the pelvis is re-tilted.  Spine2/3 biases are removed from their local rotations."""
    if not bias:
        return local
    local = local.copy()
    bias = {int(k): np.asarray(v, dtype=np.float64) for k, v in bias.items()}
    if 3 in bias:
        root, spine1 = local[0], local[3]
        new_spine1 = spine1 @ bias[3].T
        new_root = root @ spine1 @ new_spine1.T
        for hip in (1, 2):
            local[hip] = new_root.T @ root @ local[hip]
        local[0], local[3] = new_root, new_spine1
    for joint in (6, 9):          # older calibrations may also hold neck/head entries: ignored
        if joint in bias:
            local[joint] = local[joint] @ bias[joint].T
    return local


class BridgeContact:
    """Per-output thigh solve from the bridge contacts (see module docstring)."""

    def __init__(self, fps: float = 60.0, pelvis_only: bool = False, floor_arms: bool = True) -> None:
        self.offsets = _rest_offsets()
        self.thigh_length = float(np.linalg.norm(self.offsets[4]))
        self.shank_length = float(np.linalg.norm(self.offsets[7]))
        self.upper_arm_length = float(np.linalg.norm(self.offsets[18]))
        self.forearm_length = float(np.linalg.norm(self.offsets[20]))
        self.fps = fps
        self.pelvis_only = pelvis_only
        self.floor_arms = floor_arms
        self.reset("init")

    def reset(self, reason: str) -> None:
        self.weight = {"left": 0.0, "right": 0.0, "leftArm": 0.0, "rightArm": 0.0}
        self.target = {"left": None, "right": None, "leftArm": None, "rightArm": None}
        self.reason = reason

    def _positions(self, g, pelvis_height):
        p = np.zeros((24, 3))
        p[0] = (0.0, pelvis_height, 0.0)
        for j in range(1, 24):
            p[j] = p[PARENTS[j]] + g[PARENTS[j]] @ self.offsets[j]
        return p

    def apply(self, local: np.ndarray, dt: float):
        """(24,3,3) local pose -> (corrected local pose, diagnostics)."""
        local = np.asarray(local, dtype=np.float64).copy()
        g = [None] * 24
        for j in range(24):
            g[j] = local[j] if PARENTS[j] < 0 else g[PARENTS[j]] @ local[j]
        g = np.stack(g)
        chest_up = float(g[0][1, 2])
        supine = _ramp(chest_up, *GATE_SUPINE)
        base = self._positions(g, 0.0)
        pelvis_contact = g[0][1, 2] * PELVIS_BACK_DEPTH
        shoulder_contact = g[9][1, 2] * SPINE3_BACK_DEPTH - base[9, 1]
        pelvis_height = pelvis_contact if self.pelvis_only else max(pelvis_contact, shoulder_contact)
        p = base + np.array([0.0, pelvis_height, 0.0])
        alpha = 1.0 - math.exp(-dt / TARGET_TIME_CONSTANT_S)
        diag = {"supine": round(supine, 3), "pelvisHeightM": round(float(pelvis_height), 3)}
        for side, hip, knee in (("left", 1, 4), ("right", 2, 5)):
            shank = _unit(g[knee] @ self.offsets[knee + 3])
            thigh = _unit(g[hip] @ self.offsets[knee])
            shank_elev = math.degrees(math.asin(float(np.clip(shank[1], -1, 1))))
            planted = 1.0 if GATE_SHANK_DEG[0] <= shank_elev <= GATE_SHANK_DEG[1] else 0.0
            sine = (ANKLE_HEIGHT - p[hip, 1] - self.shank_length * shank[1]) / self.thigh_length
            feasible = abs(sine) <= 0.98
            goal = supine * planted * (1.0 if feasible else 0.0)
            step = WEIGHT_RATE_PER_S * dt
            self.weight[side] += float(np.clip(goal - self.weight[side], -step, step))
            sine = float(np.clip(sine, -0.98, 0.98))
            self.target[side] = sine if self.target[side] is None else self.target[side] + alpha * (sine - self.target[side])
            heading = thigh.copy()
            heading[1] = 0.0
            if np.linalg.norm(heading) < 0.2:      # thigh near vertical: use the feet direction of the pelvis
                heading = -g[0][:, 1].copy()
                heading[1] = 0.0
            heading = _unit(heading)
            s = self.target[side]
            wanted = math.sqrt(1 - s * s) * heading + np.array([0.0, s, 0.0])
            before = math.degrees(math.asin(float(np.clip(thigh[1], -1, 1))))
            w = self.weight[side]
            if w > 1e-3:
                swing = _scaled(_rotation_between(thigh, wanted), w)
                new_hip = swing @ g[hip]
                local[hip] = g[0].T @ new_hip
                local[knee] = new_hip.T @ g[knee]          # keep the shank's global orientation
            diag[side] = {"weight": round(w, 3), "shankElevDeg": round(shank_elev, 1),
                          "thighElevBeforeDeg": round(before, 1),
                          "thighElevTargetDeg": round(math.degrees(math.asin(s)), 1), "feasible": bool(feasible)}
        # Arms lying on the floor by the side (the bridge keeps them there): the forearm direction is measured
        # (anchored to the sensor); slope the upper arm so the elbow rests on the floor instead of floating
        # at shoulder height.  Applies only while the forearm is near horizontal.
        if not self.floor_arms:
            return local, diag
        for side, collar, shoulder, elbow, wrist, hand in (("leftArm", 13, 16, 18, 20, 22),
                                                           ("rightArm", 14, 17, 19, 21, 23)):
            upper = _unit(g[shoulder] @ self.offsets[elbow])
            forearm = _unit(g[elbow] @ self.offsets[wrist])
            forearm_elev = math.degrees(math.asin(float(np.clip(forearm[1], -1, 1))))
            lying = 1.0 if abs(forearm_elev) <= GATE_FOREARM_DEG else 0.0
            elbow_goal = max(ELBOW_DEPTH, WRIST_DEPTH - self.forearm_length * forearm[1])
            sine = (elbow_goal - p[shoulder, 1]) / self.upper_arm_length
            feasible = abs(sine) <= 0.9
            goal = supine * lying * (1.0 if feasible else 0.0)
            step = WEIGHT_RATE_PER_S * dt
            self.weight[side] += float(np.clip(goal - self.weight[side], -step, step))
            sine = float(np.clip(sine, -0.9, 0.9))
            self.target[side] = sine if self.target[side] is None else self.target[side] + alpha * (sine - self.target[side])
            heading = forearm.copy()          # arm by the side: upper arm and forearm point the same way
            heading[1] = 0.0
            if np.linalg.norm(heading) < 0.3:
                heading = upper.copy()
                heading[1] = 0.0
            heading = _unit(heading)
            t = self.target[side]
            wanted = math.sqrt(1 - t * t) * heading + np.array([0.0, t, 0.0])
            w = self.weight[side]
            if w > 1e-3:
                swing = _scaled(_rotation_between(upper, wanted), w)
                new_upper = swing @ g[shoulder]
                local[shoulder] = g[collar].T @ new_upper
                local[elbow] = new_upper.T @ g[elbow]       # keep the measured forearm
                # Palm flat in line with the forearm: the network's flexed wrist put the fingers 1-8 cm
                # into the floor once the elbow rests on it (040120).
                for joint in (wrist, hand):
                    local[joint] = local[joint] @ _scaled(local[joint].T, w)
            diag[side] = {"weight": round(w, 3), "forearmElevDeg": round(forearm_elev, 1),
                          "upperArmElevBeforeDeg": round(math.degrees(math.asin(float(np.clip(upper[1], -1, 1)))), 1),
                          "upperArmElevTargetDeg": round(math.degrees(math.asin(t)), 1), "feasible": bool(feasible)}
        return local, diag


def _rest_offsets() -> np.ndarray:
    """(24,3) zero-pose joint offsets from the parent (same SMPL model as the network FK)."""
    from mobileposer.articulate.model import ParametricModel
    from production.paths import paths
    joints, _ = ParametricModel(str(paths.smpl_file)).get_zero_pose_joint_and_vertex()
    joints = joints.cpu().numpy().astype(np.float64)
    offsets = np.zeros_like(joints)
    for j in range(1, 24):
        offsets[j] = joints[j] - joints[PARENTS[j]]
    return offsets


def _body_constants():
    """How the depth/height constants above were measured (python -m production.bridge_contact)."""
    from mobileposer.articulate.model import ParametricModel
    from production.paths import paths
    joints, vertices = ParametricModel(str(paths.smpl_file)).get_zero_pose_joint_and_vertex()
    J, V = joints.numpy(), vertices.numpy()

    def depth(j, half_width):
        band = (abs(V[:, 1] - J[j, 1]) < 0.03) & (abs(V[:, 0] - J[j, 0]) < half_width)
        return float(J[j, 2] - V[band, 2].min())
    return {"pelvis_back": depth(0, 0.08), "spine3_back": depth(9, 0.10), "ankle_height": float(J[7, 1] - V[:, 1].min())}


if __name__ == "__main__":
    print(_body_constants())


def bridge_calibration_state(calibrator) -> dict:
    """Recording/replay form of the lie-flat calibration (calibration snapshot fields)."""
    waist, trunk = getattr(calibrator, "bridge_waist", None), getattr(calibrator, "bridge_trunk", None)
    return {"bridgeWaist": None if waist is None else np.asarray(waist).tolist(),
            "bridgeTrunk": None if trunk is None else {str(k): np.asarray(v).tolist() for k, v in trunk.items()}}


def load_bridge_calibration(calibrator, state: dict) -> None:
    """Inverse of bridge_calibration_state; recordings without the fields leave the calibrator unchanged."""
    if "bridgeWaist" in state:
        calibrator.bridge_waist = None if state["bridgeWaist"] is None else np.asarray(state["bridgeWaist"], dtype=np.float64)
    if "bridgeTrunk" in state:
        trunk = state["bridgeTrunk"]
        calibrator.bridge_trunk = None if trunk is None else {int(k): np.asarray(v, dtype=np.float64) for k, v in trunk.items()}

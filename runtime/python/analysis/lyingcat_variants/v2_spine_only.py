"""[Comparison snapshot, 2026-11-19: spine2/3 absorb the gap, pelvis follows the waist.]
Four-point kneeling support constraint for the lying-cat exercise.

Measured: forearms, shanks, waist.  Unobserved: upper arms, thighs, upper spine,
hands and feet.  Palms and knees are kept on one support plane, but the plane
is reached the way a person on all fours reaches it:

* A limb only counts as a support when its own sensor says so (forearm pointing
  down, shank about horizontal).  A lifted limb is released smoothly and keeps
  its network / measured pose.
* Forearms and shanks keep the calibrated measurement (passed in as
  ``measured``); they are never forced vertical.
* Supporting thighs are aimed straight down (knee under hip) and supporting
  upper arms continue the measured forearm (straight elbow).  Shanks are levelled and the
  palm / foot dorsum laid flat.
* The height difference between hands and knees is absorbed by pitching the
  upper spine (joints 6 and 9, above the waist sensor) -- the cat/cow degree of
  freedom -- instead of slanting the limbs.  Only a residual beyond the spine
  limit is closed by tilting the upper arms / thighs.

2026-11-19 rewrite: the previous version forced both forearms vertical and
reached the plane by slanting thighs ~45 deg and upper arms ~40 deg on the real
recording unity_20261116_195301 (see recordings/action_eval_20261119).
"""
from __future__ import annotations

import math

import numpy as np

from production.arm_prior import PARENTS, _globals, _rotation_between, _scaled
from production.bridge_contact import _ramp, _rest_offsets, _unit

WEIGHT_RATE_PER_S = 4.0
SPINE_TIME_CONSTANT_S = 0.10
MAX_SPINE_DEG = 40.0
# Forearm elevation (deg, negative = pointing down) -> support weight target.
ARM_SUPPORT_RAMP = (40.0, 55.0)      # -elevation 40 deg: lifted, 55 deg and steeper: on the mat
# |shank elevation| -> support weight target (knee and foot on the mat when the shank is level).
LEG_RELEASE_RAMP = (25.0, 40.0)
MIN_RESIDUAL_SINE = -1.0
MAX_RESIDUAL_SINE = -0.5             # residual closing never tilts a support segment above 30 deg from horizontal
SENSED = {"lw": 18, "rw": 19, "ls": 4, "rs": 5}
ARMS = (("left", 16, 18, 20, 22), ("right", 17, 19, 21, 23))       # side, upper arm, forearm, wrist, hand
LEGS = (("left", 1, 4, 7, 10), ("right", 2, 5, 8, 11))             # side, thigh, shank, ankle, foot
DOWN = np.array([0.0, -1.0, 0.0])
UP = np.array([0.0, 1.0, 0.0])


def _axis_rotation(axis, degrees):
    axis = _unit(np.asarray(axis, dtype=np.float64))
    a = math.radians(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k


def _elevation(direction):
    return math.degrees(math.asin(float(np.clip(_unit(direction)[1], -1.0, 1.0))))


class LyingCatContact:
    def __init__(self) -> None:
        self.offsets = _rest_offsets()
        self.reset("init")

    def reset(self, reason: str) -> None:
        self.weights = {"leftArm": 0.0, "rightArm": 0.0, "leftLeg": 0.0, "rightLeg": 0.0}
        self.spine_deg = 0.0
        self.reason = reason

    # ------------------------------------------------------------------ helpers
    def _positions(self, g) -> np.ndarray:
        p = np.zeros((24, 3), dtype=np.float64)
        for joint in range(1, 24):
            p[joint] = p[PARENTS[joint]] + g[PARENTS[joint]] @ self.offsets[joint]
        return p

    @staticmethod
    def _set_global(local, g, joint, new_global, keep_child):
        """Give ``joint`` a new global rotation; ``keep_child`` keeps its global rotation."""
        local[joint] = g[PARENTS[joint]].T @ new_global
        local[keep_child] = new_global.T @ g[keep_child]

    def _aim(self, local, g, joint, child, wanted, weight):
        """Swing ``joint`` so its bone (towards ``child``) points along ``wanted``, by ``weight``."""
        if weight <= 0.0:
            return
        direction = g[joint] @ self.offsets[child]
        swing = _scaled(_rotation_between(direction, wanted), weight)
        self._set_global(local, g, joint, swing @ g[joint], child)

    @staticmethod
    def _anchor(local, measured, labels):
        if measured is None or labels is None:
            return local
        g = _globals(local)
        overrides = {SENSED[label]: np.asarray(measured[index], dtype=np.float64)
                     for index, label in enumerate(labels) if label in SENSED}
        result = local.copy()
        for joint in sorted(overrides):
            g = _globals(result)
            result[joint] = g[PARENTS[joint]].T @ overrides[joint]
        return result

    def _headward(self, g, p):
        headward = p[12] - p[0]
        headward[1] = 0.0
        if np.linalg.norm(headward) < 0.2:
            headward = g[0] @ UP
            headward[1] = 0.0
        if np.linalg.norm(headward) < 1e-6:
            headward = np.array([0.0, 0.0, 1.0])
        return _unit(headward)

    @staticmethod
    def _restore(local, globals_by_joint):
        for joint in sorted(globals_by_joint):
            g = _globals(local)
            local[joint] = g[PARENTS[joint]].T @ globals_by_joint[joint]

    def _apply_spine(self, local, g, axis, degrees):
        """Pitch the upper spine by ``degrees`` (half at spine2, half at spine3)."""
        if abs(degrees) < 1e-6:
            return local
        out = local.copy()
        half = _axis_rotation(axis, degrees / 2.0)
        new6 = half @ g[6]
        new9 = half @ half @ g[9]
        out[6] = g[3].T @ new6
        out[9] = new6.T @ new9
        return out

    # ------------------------------------------------------------------ steps
    def _update_weights(self, g, dt):
        step = WEIGHT_RATE_PER_S * dt
        targets, elevations = {}, {}
        for side, _upper, forearm, wrist, _hand in ARMS:
            elevation = _elevation(g[forearm] @ self.offsets[wrist])
            elevations[f"{side}Forearm"] = round(elevation, 1)
            targets[f"{side}Arm"] = _ramp(-elevation, *ARM_SUPPORT_RAMP)
        for side, _thigh, shank, ankle, _foot in LEGS:
            elevation = _elevation(g[shank] @ self.offsets[ankle])
            elevations[f"{side}Shank"] = round(elevation, 1)
            targets[f"{side}Leg"] = 1.0 - _ramp(abs(elevation), *LEG_RELEASE_RAMP)
        for key, target in targets.items():
            self.weights[key] += float(np.clip(target - self.weights[key], -step, step))
        return elevations

    def _level_legs(self, local):
        g = _globals(local)
        for side, thigh, shank, ankle, foot in LEGS:
            weight = self.weights[f"{side}Leg"]
            if weight <= 0.0:
                continue
            direction = _unit(g[shank] @ self.offsets[ankle])
            level = direction.copy()
            level[1] = 0.0
            if np.linalg.norm(level) < 0.2:
                continue
            level = _unit(level)
            new_shank = _scaled(_rotation_between(direction, level), weight) @ g[shank]
            local[shank] = g[thigh].T @ new_shank
            # The ankle is unobserved: lay the foot back along the shank (dorsum on the mat).
            ankle_global = new_shank @ local[ankle]
            foot_vector = ankle_global @ self.offsets[foot]
            new_ankle = _scaled(_rotation_between(foot_vector, level), weight) @ ankle_global
            local[ankle] = new_shank.T @ new_ankle
            local[foot] = local[foot] @ _scaled(local[foot].T, weight)
            g = _globals(local)

    def _vertical_supports(self, local, kinds):
        """Thighs straight down (knee under hip); upper arms continue the measured
        forearm (straight elbow, shoulder over the hand)."""
        for kind in kinds:
            limbs = ARMS if kind == "Arm" else LEGS
            for side, segment, child, grandchild, _b in limbs:
                g = _globals(local)
                wanted = DOWN if kind == "Leg" else _unit(g[child] @ self.offsets[grandchild])
                self._aim(local, g, segment, child, wanted, self.weights[f"{side}{kind}"])

    def _support_heights(self, local):
        p = self._positions(_globals(local))
        arm_w = np.array([self.weights["leftArm"], self.weights["rightArm"]])
        leg_w = np.array([self.weights["leftLeg"], self.weights["rightLeg"]])
        hands = np.array([p[20, 1], p[21, 1]])
        knees = np.array([p[4, 1], p[5, 1]])
        hand = float(arm_w @ hands / arm_w.sum()) if arm_w.sum() > 1e-6 else None
        knee = float(leg_w @ knees / leg_w.sum()) if leg_w.sum() > 1e-6 else None
        return hand, knee

    def _spine_target(self, local, axis):
        """Spine pitch (deg) that puts the straight-arm hands on the knee plane.

        Closed form per candidate: pitching spine2 (joint 6) and spine3 (joint 9)
        by half the angle each moves a shoulder s to
        p6 + R(p9 - p6) + R R (s - p9).  With a straight elbow along the kept
        forearm, the wrist sits (upper + forearm length) * forearm sine below it.
        """
        g = _globals(local)
        p = self._positions(g)
        _, knee = self._support_heights(local)
        arm_w = np.array([self.weights["leftArm"], self.weights["rightArm"]])
        arm_w = arm_w / max(arm_w.sum(), 1e-9)
        drops = []
        for _side, upper, forearm, wrist, _hand in ARMS:
            along = g[forearm] @ self.offsets[wrist]
            drops.append(float(np.linalg.norm(self.offsets[forearm])) * float(_unit(along)[1]) + float(along[1]))
        drops = np.asarray(drops)

        def gap(degrees):
            half = _axis_rotation(axis, degrees / 2.0)
            full = half @ half
            base = p[6] + half @ (p[9] - p[6])
            shoulders = np.array([(base + full @ (p[joint] - p[9]))[1] for joint in (16, 17)])
            return float(arm_w @ (shoulders + drops)) - knee

        lo, hi = -MAX_SPINE_DEG, MAX_SPINE_DEG
        f_lo, f_hi = gap(lo), gap(hi)
        if f_lo * f_hi > 0.0:
            return lo if abs(f_lo) < abs(f_hi) else hi
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            f_mid = gap(mid)
            if f_mid * f_lo <= 0.0:
                hi = mid
            else:
                lo, f_lo = mid, f_mid
        return 0.5 * (lo + hi)

    def _close_residual(self, local, floor):
        """Tilt support segments only as much as needed to put palms/knees on ``floor``."""
        specs = [("Arm", side, upper, forearm, wrist) for side, upper, forearm, wrist, _ in ARMS]
        specs += [("Leg", side, thigh, shank, shank) for side, thigh, shank, _a, _f in LEGS]
        for kind, side, segment, child, contact in specs:
            weight = self.weights[f"{side}{kind}"]
            if weight <= 0.0:
                continue
            g = _globals(local)
            p = self._positions(g)
            vector = g[segment] @ self.offsets[child]
            length = float(np.linalg.norm(vector))
            below = p[contact, 1] - p[child, 1]          # forearm drop below the elbow (0 for the knee)
            sine = (floor - below - p[segment, 1]) / max(length, 1e-9)
            sine = float(np.clip(sine, MIN_RESIDUAL_SINE, MAX_RESIDUAL_SINE))
            heading = vector.copy()
            heading[1] = 0.0
            if np.linalg.norm(heading) < 1e-6:
                heading = self._headward(g, p) * (1.0 if kind == "Arm" else -1.0)
            heading = _unit(heading)
            wanted = math.sqrt(max(0.0, 1.0 - sine * sine)) * heading + sine * UP
            self._aim(local, g, segment, child, wanted, weight)

    def _flatten_hands(self, local):
        # In SMPL's zero pose the wrist->hand axis follows the fingers and local
        # -Y is the palm-facing direction.  On the mat the fingers point toward
        # the head while the palm faces down.
        g = _globals(local)
        p = self._positions(g)
        target = self._basis(self._headward(g, p), DOWN)
        for side, _upper, forearm, wrist, hand in ARMS:
            weight = self.weights[f"{side}Arm"]
            if weight <= 0.0:
                continue
            source = self._basis(_unit(self.offsets[hand]), DOWN)
            wanted_local = g[forearm].T @ (target @ source.T)
            local[wrist] = local[wrist] @ _scaled(local[wrist].T @ wanted_local, weight)
            local[hand] = local[hand] @ _scaled(local[hand].T, weight)

    @staticmethod
    def _basis(primary, secondary):
        primary = _unit(primary)
        secondary = secondary - float(np.dot(secondary, primary)) * primary
        secondary = _unit(secondary)
        return np.column_stack((primary, secondary, np.cross(primary, secondary)))

    # ------------------------------------------------------------------ entry
    def apply(self, local: np.ndarray, dt: float, measured=None, labels=None):
        local = self._anchor(np.asarray(local, dtype=np.float64).copy(), measured, labels)
        g = _globals(local)
        elevations = self._update_weights(g, dt)
        p = self._positions(g)
        before = self._support_heights(local)
        thigh_before = {side: round(_elevation(g[thigh] @ self.offsets[shank]), 1)
                        for side, thigh, shank, _a, _f in LEGS}
        upper_before = {side: round(_elevation(g[upper] @ self.offsets[forearm]), 1)
                        for side, upper, forearm, _w, _h in ARMS}

        self._level_legs(local)
        self._vertical_supports(local, ("Leg",))
        g = _globals(local)
        sensed = {joint: g[joint].copy() for joint in (18, 19)}   # forearms: measured, never moved by the spine

        arm_support = 0.5 * (self.weights["leftArm"] + self.weights["rightArm"])
        leg_support = 0.5 * (self.weights["leftLeg"] + self.weights["rightLeg"])
        support = min(arm_support, leg_support)
        # Under four-point support the upper-spine shape follows from the contacts:
        # drop the network's own spine2/3 bend before solving (it carried a ~26 deg bias live).
        if support > 0.0:
            for joint in (6, 9):
                local[joint] = local[joint] @ _scaled(local[joint].T, support)
            self._restore(local, sensed)
        g = _globals(local)
        p = self._positions(g)
        axis = np.cross(UP, self._headward(g, p))
        target = self._spine_target(local, axis) if support > 0.05 else 0.0
        alpha = 1.0 - math.exp(-dt / SPINE_TIME_CONSTANT_S)
        self.spine_deg += alpha * (target - self.spine_deg)
        local = self._apply_spine(local, g, axis, self.spine_deg * support)
        self._restore(local, sensed)
        self._vertical_supports(local, ("Arm",))

        _, floor = self._support_heights(local)
        if floor is not None:
            self._close_residual(local, floor)
        self._flatten_hands(local)

        g = _globals(local)
        p = self._positions(g)
        diag = {"enabled": True,
                "weights": {key: round(value, 3) for key, value in self.weights.items()},
                "spineDeg": round(self.spine_deg * support, 1), "spineTargetDeg": round(target, 1),
                "sensorElevationDeg": elevations,
                "thighElevationBeforeDeg": thigh_before, "upperArmElevationBeforeDeg": upper_before,
                "thighElevationAfterDeg": {side: round(_elevation(g[thigh] @ self.offsets[shank]), 1)
                                           for side, thigh, shank, _a, _f in LEGS},
                "upperArmElevationAfterDeg": {side: round(_elevation(g[upper] @ self.offsets[forearm]), 1)
                                              for side, upper, forearm, _w, _h in ARMS}}
        if before[0] is not None and before[1] is not None:
            diag["handKneeGapBeforeCm"] = round(100.0 * (before[0] - before[1]), 1)
        if floor is not None:
            diag["supportPlaneM"] = round(floor, 3)
            diag["afterErrorCm"] = {name: round(100.0 * float(p[joint, 1] - floor), 1)
                                    for name, joint, key in (("leftWrist", 20, "leftArm"), ("rightWrist", 21, "rightArm"),
                                                             ("leftKnee", 4, "leftLeg"), ("rightKnee", 5, "rightLeg"))
                                    if self.weights[key] > 0.5}
        return local, diag

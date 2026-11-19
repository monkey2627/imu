"""[Comparison copy of the ORIGINAL production/lyingcat_contact.py as it was before the
2026-11-19 rewrite; restored verbatim from the session read, only ``apply`` accepts and
ignores the new ``measured`` / ``labels`` keyword arguments.]

Four-point kneeling support constraint for the lying-cat exercise.

Forearm, shank and waist directions are measured.  Upper arms, thighs and feet
are not.  While the lying-cat action is active, solve the unobserved segments
and apply the exercise's known kneeling contact: wrists plus the complete
knee-to-ankle shanks and foot dorsums share one support plane.  Forearm global
rotations are preserved; small shank tilt is removed only for this action.
"""
from __future__ import annotations

import math

import numpy as np

from production.arm_prior import PARENTS, _globals, _rotation_between, _scaled
from production.bridge_contact import _rest_offsets, _unit

WEIGHT_RATE_PER_S = 4.0
TARGET_TIME_CONSTANT_S = 0.08


class LyingCatContact:
    def __init__(self) -> None:
        self.offsets = _rest_offsets()
        self.reset("init")

    def reset(self, reason: str) -> None:
        self.floor_height = None
        self.weight = 0.0
        self.target = {"leftArm": None, "rightArm": None, "leftLeg": None, "rightLeg": None}
        self.reason = reason

    def _positions(self, g) -> np.ndarray:
        p = np.zeros((24, 3), dtype=np.float64)
        for joint in range(1, 24):
            p[joint] = p[PARENTS[joint]] + g[PARENTS[joint]] @ self.offsets[joint]
        return p

    @staticmethod
    def _basis(primary, secondary):
        primary = _unit(primary)
        secondary = secondary - float(np.dot(secondary, primary)) * primary
        secondary = _unit(secondary)
        return np.column_stack((primary, secondary, np.cross(primary, secondary)))

    def _flatten_hands(self, local, g, p):
        headward = p[12] - p[0]
        headward[1] = 0.0
        if np.linalg.norm(headward) < 0.2:
            headward = g[0] @ np.array([0.0, 1.0, 0.0])
            headward[1] = 0.0
        headward = _unit(headward)
        target = self._basis(headward, np.array([0.0, -1.0, 0.0]))
        before = {}
        for side, forearm, wrist, hand in (("left", 18, 20, 22), ("right", 19, 21, 23)):
            hand_axis = _unit(self.offsets[hand])
            source = self._basis(hand_axis, np.array([0.0, -1.0, 0.0]))
            wanted_wrist = target @ source.T
            current = _unit(g[wrist] @ self.offsets[hand])
            before[side] = math.degrees(math.asin(float(np.clip(current[1], -1.0, 1.0))))
            local[wrist] = g[forearm].T @ wanted_wrist
            local[hand] = np.eye(3)
        return before

    def _vertical_forearms(self, local, g):
        before = {}
        down = np.array([0.0, -1.0, 0.0])
        for side, elbow, wrist in (("left", 18, 20), ("right", 19, 21)):
            direction = _unit(g[elbow] @ self.offsets[wrist])
            before[side] = math.degrees(math.asin(float(np.clip(direction[1], -1.0, 1.0))))
            wanted = _rotation_between(direction, down) @ g[elbow]
            local[elbow] = g[PARENTS[elbow]].T @ wanted
        return before

    def _solve_segment(self, local, g, p, key, parent, joint, child, target_y, dt, downstream_y=0.0):
        vector = g[joint] @ self.offsets[child]
        length = float(np.linalg.norm(vector))
        sine = (target_y - p[joint, 1] - downstream_y) / max(length, 1e-9)
        feasible = abs(sine) <= 1.0
        sine = float(np.clip(sine, -0.985, 0.985))
        alpha = 1.0 - math.exp(-dt / TARGET_TIME_CONSTANT_S)
        previous = self.target[key]
        self.target[key] = sine if previous is None else previous + alpha * (sine - previous)
        direction = _unit(vector)
        heading = direction.copy()
        heading[1] = 0.0
        if np.linalg.norm(heading) < 0.15:
            heading = np.array([0.0, 0.0, 1.0])
        heading = _unit(heading)
        s = self.target[key]
        wanted = math.sqrt(max(0.0, 1.0 - s * s)) * heading + np.array([0.0, s, 0.0])
        swing = _scaled(_rotation_between(direction, wanted), self.weight)
        new_global = swing @ g[joint]
        local[joint] = g[parent].T @ new_global
        local[child] = new_global.T @ g[child]
        return feasible, math.degrees(math.asin(float(np.clip(direction[1], -1.0, 1.0)))), math.degrees(math.asin(s))

    def _level_shank_and_foot(self, local, g, side):
        knee = 4 if side == "left" else 5
        ankle = 7 if side == "left" else 8
        foot = 10 if side == "left" else 11
        shank = _unit(g[knee] @ self.offsets[ankle])
        horizontal_shank = shank.copy()
        horizontal_shank[1] = 0.0
        if np.linalg.norm(horizontal_shank) < 0.2:
            return 0.0, 0.0
        horizontal_shank = _unit(horizontal_shank)
        swing = _rotation_between(shank, horizontal_shank)
        new_knee = swing @ g[knee]
        parent = 1 if side == "left" else 2
        local[knee] = g[parent].T @ new_knee
        foot_vector = _unit(g[ankle] @ self.offsets[foot])
        foot_swing = _rotation_between(foot_vector, horizontal_shank)
        new_ankle = foot_swing @ g[ankle]
        local[ankle] = new_knee.T @ new_ankle
        return (math.degrees(math.asin(float(np.clip(shank[1], -1.0, 1.0)))),
                math.degrees(math.asin(float(np.clip(foot_vector[1], -1.0, 1.0)))))

    def apply(self, local: np.ndarray, dt: float, measured=None, labels=None):
        local = np.asarray(local, dtype=np.float64).copy()
        g = _globals(local)
        p = self._positions(g)
        forearm_tilt = self._vertical_forearms(local, g)
        g = _globals(local)
        p = self._positions(g)
        hand_tilt = self._flatten_hands(local, g, p)
        g = _globals(local)
        p = self._positions(g)
        contacts = np.array([p[20, 1], p[21, 1], p[4, 1], p[5, 1],
                             p[7, 1], p[8, 1], p[10, 1], p[11, 1]])
        arm_bases = (p[16, 1] + (g[18] @ self.offsets[20])[1],
                     p[17, 1] + (g[19] @ self.offsets[21])[1])
        limb_ranges = (
            (arm_bases[0] - np.linalg.norm(self.offsets[18]), arm_bases[0] + np.linalg.norm(self.offsets[18])),
            (arm_bases[1] - np.linalg.norm(self.offsets[19]), arm_bases[1] + np.linalg.norm(self.offsets[19])),
            (p[1, 1] - np.linalg.norm(self.offsets[4]), p[1, 1] + np.linalg.norm(self.offsets[4])),
            (p[2, 1] - np.linalg.norm(self.offsets[5]), p[2, 1] + np.linalg.norm(self.offsets[5])),
        )
        low, high = max(value[0] for value in limb_ranges), min(value[1] for value in limb_ranges)
        plane_feasible = low <= high
        candidate = float(np.clip(np.median(contacts), low, high)) if plane_feasible else float(np.median(contacts))
        alpha = 1.0 - math.exp(-dt / TARGET_TIME_CONSTANT_S)
        if self.floor_height is None:
            self.floor_height = candidate
        else:
            self.floor_height += alpha * (candidate - self.floor_height)
            if plane_feasible:
                self.floor_height = float(np.clip(self.floor_height, low, high))
        max_error = float(np.max(np.abs(contacts - self.floor_height)))
        goal = 1.0 if plane_feasible else 0.0
        step = WEIGHT_RATE_PER_S * dt
        self.weight += float(np.clip(goal - self.weight, -step, step))

        diag = {"enabled": True, "weight": round(self.weight, 3),
                "supportPlaneM": round(self.floor_height, 3),
                "planeFeasible": bool(plane_feasible),
                "beforeErrorCm": round(100.0 * max_error, 1),
                "leftForearmTiltBeforeDeg": round(forearm_tilt["left"], 1),
                "rightForearmTiltBeforeDeg": round(forearm_tilt["right"], 1),
                "leftHandTiltBeforeDeg": round(hand_tilt["left"], 1),
                "rightHandTiltBeforeDeg": round(hand_tilt["right"], 1)}
        left_tilt, left_foot_tilt = self._level_shank_and_foot(local, g, "left")
        right_tilt, right_foot_tilt = self._level_shank_and_foot(local, g, "right")
        diag["leftShankTiltBeforeDeg"] = round(left_tilt, 1)
        diag["rightShankTiltBeforeDeg"] = round(right_tilt, 1)
        diag["leftFootTiltBeforeDeg"] = round(left_foot_tilt, 1)
        diag["rightFootTiltBeforeDeg"] = round(right_foot_tilt, 1)
        g = _globals(local)
        p = self._positions(g)
        specs = (
            ("leftArm", 13, 16, 18, float((g[18] @ self.offsets[20])[1])),
            ("rightArm", 14, 17, 19, float((g[19] @ self.offsets[21])[1])),
            ("leftLeg", 0, 1, 4, 0.0), ("rightLeg", 0, 2, 5, 0.0),
        )
        for key, parent, joint, child, downstream_y in specs:
            feasible, before, target = self._solve_segment(
                local, g, p, key, parent, joint, child, self.floor_height, dt, downstream_y)
            diag[key] = {"feasible": bool(feasible), "elevationBeforeDeg": round(before, 1),
                         "elevationTargetDeg": round(target, 1)}

        for joint in (10, 11):
            local[joint] = local[joint] @ _scaled(local[joint].T, self.weight)

        after = self._positions(_globals(local))
        errors = np.array([after[20, 1], after[21, 1], after[22, 1], after[23, 1],
                           after[4, 1], after[5, 1], after[7, 1], after[8, 1],
                           after[10, 1], after[11, 1]]) - self.floor_height
        diag["afterErrorCm"] = [round(float(100.0 * value), 1) for value in errors]
        return local, diag

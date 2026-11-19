"""Dead-bug supine floor contact: the back rests on the mat, nothing else sinks below it.

On the real recordings the trunk itself comes out flat, but the extended legs (37-49 % of
supine frames, up to 13-28 cm) and a head tipped backwards (up to 10 cm) dip below the back.
Rendered on a floor, the back then hovers; with a fixed floor the limbs sink into it.

While lying supine (spine about horizontal, chest facing up):
* the floor is the lowest point of the back surface (pelvis / spine1-3 mesh regions);
* a leg below the floor is lifted by flexing its hip (thigh swing about the hip, the
  measured shank keeps its global orientation) just until its lowest mesh point reaches
  the floor -- the extended leg then hovers at mat level as in the exercise;
* a head below the floor is tucked (neck flexion) until the back of the head rests on it.
Only lifting corrections are applied, never pushing anything down.  Mesh points use the
SMPL template rigidly attached to each vertex's dominant bone (subsampled).  Corrections
are low-pass filtered and rate limited, and fade with a supine gate.
"""
from __future__ import annotations

import math

import numpy as np

from production.arm_prior import PARENTS, _globals

GATE_SPINE_DEG = (20.0, 35.0)       # |pelvis->neck elevation| below 20 deg: lying, above 35: not
GATE_CHEST_UP = (0.45, 0.70)        # chest normal's up component
WEIGHT_RATE_PER_S = 4.0
TIME_CONSTANT_S = 0.08
MAX_RATE_DEG_PER_S = 240.0
MAX_LEG_LIFT_DEG = 70.0
MAX_HEAD_TUCK_DEG = 45.0
SUBSAMPLE = 3
REGIONS = {"back": (0, 3, 6, 9), "leftLeg": (1, 4, 7, 10), "rightLeg": (2, 5, 8, 11), "head": (12, 15)}
LIMBS = (("leftLeg", 1, 4), ("rightLeg", 2, 5), ("head", 12, None))   # region, rotated joint, kept child
UP = np.array([0.0, 1.0, 0.0])


def _ramp(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _axis_rotation(axis, degrees):
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    a = math.radians(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k


def _template():
    from mobileposer.articulate.model import ParametricModel
    from production.paths import paths
    body = ParametricModel(str(paths.smpl_file))
    joints, vertices = body.get_zero_pose_joint_and_vertex()
    joints = joints.cpu().numpy().astype(np.float64)
    vertices = vertices.cpu().numpy().astype(np.float64)
    dominant = body._skinning_weights.cpu().numpy().argmax(1)
    return joints, vertices, dominant


class DeadBugFloorContact:
    def __init__(self) -> None:
        joints, vertices, dominant = _template()
        self.offsets = np.zeros_like(joints)
        for j in range(1, 24):
            self.offsets[j] = joints[j] - joints[PARENTS[j]]
        self.regions = {}
        for name, bones in REGIONS.items():
            index = np.flatnonzero(np.isin(dominant, bones))[::SUBSAMPLE]
            self.regions[name] = (dominant[index], vertices[index] - joints[dominant[index]])
        self.reset("init")

    def reset(self, reason: str) -> None:
        self.weight = 0.0
        self.angles = {name: 0.0 for name, _j, _c in LIMBS}
        self.reason = reason

    def _positions(self, g):
        p = np.zeros((24, 3))
        for j in range(1, 24):
            p[j] = p[PARENTS[j]] + g[PARENTS[j]] @ self.offsets[j]
        return p

    def _points(self, name, g, p):
        bones, local = self.regions[name]
        return np.einsum("nab,nb->na", g[bones], local) + p[bones]

    @staticmethod
    def _lowest_after(joint, child, p, points, bones, axis, degrees):
        """Lowest region point after rotating ``joint`` about its own position.

        Without a kept child (head) everything below rotates with it.  With a kept child
        (thigh -> measured shank) only the thigh's points rotate; the shank and foot keep
        their orientation and translate with the moved knee.
        """
        rotation = _axis_rotation(axis, degrees)
        pivot = p[joint]
        if child is None:
            return float((pivot[1] + (rotation @ (points - pivot).T)[1]).min())
        own = bones == joint
        knee = p[child] - pivot
        shift = (rotation @ knee - knee)[1]
        lowest_own = (pivot[1] + (rotation @ (points[own] - pivot).T)[1]).min() if own.any() else np.inf
        lowest_rest = (points[~own, 1] + shift).min() if (~own).any() else np.inf
        return float(min(lowest_own, lowest_rest))

    def _required(self, name, joint, child, g, p, floor, axis, limit):
        points = self._points(name, g, p)
        bones = self.regions[name][0]
        if points[:, 1].min() >= floor:
            return 0.0, float(points[:, 1].min() - floor)
        lowest = lambda deg: self._lowest_after(joint, child, p, points, bones, axis, deg)
        # Lifting direction: whichever sign raises the region's lowest point.
        sign = 1.0 if lowest(5.0) >= lowest(-5.0) else -1.0
        lo, hi = 0.0, limit
        if lowest(sign * hi) < floor:
            return sign * hi, float(points[:, 1].min() - floor)
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            if lowest(sign * mid) >= floor:
                hi = mid
            else:
                lo = mid
        return sign * hi, float(points[:, 1].min() - floor)

    def apply(self, local: np.ndarray, dt: float):
        local = np.asarray(local, dtype=np.float64).copy()
        g = np.asarray(_globals(local))
        p = self._positions(g)
        spine = p[12] - p[0]
        spine_elev = math.degrees(math.asin(float(np.clip(spine[1] / max(np.linalg.norm(spine), 1e-9), -1, 1))))
        chest_up = float((g[9] @ np.array([0.0, 0.0, 1.0]))[1])
        goal = (1.0 - _ramp(abs(spine_elev), *GATE_SPINE_DEG)) * _ramp(chest_up, *GATE_CHEST_UP)
        step = WEIGHT_RATE_PER_S * dt
        self.weight += float(np.clip(goal - self.weight, -step, step))

        floor = float(self._points("back", g, p)[:, 1].min())
        axis = g[0] @ np.array([1.0, 0.0, 0.0])
        axis[1] = 0.0
        diag = {"enabled": True, "weight": round(self.weight, 3), "spineElevationDeg": round(spine_elev, 1),
                "chestUp": round(chest_up, 2)}
        alpha = 1.0 - math.exp(-dt / TIME_CONSTANT_S)
        max_step = MAX_RATE_DEG_PER_S * dt
        for name, joint, child in LIMBS:
            limit = MAX_HEAD_TUCK_DEG if name == "head" else MAX_LEG_LIFT_DEG
            target, clearance = (0.0, 0.0)
            if self.weight > 0.0 and np.linalg.norm(axis) > 1e-6:
                target, clearance = self._required(name, joint, child, g, p, floor, axis, limit)
            change = alpha * (target * self.weight - self.angles[name])
            self.angles[name] += float(np.clip(change, -max_step, max_step))
            diag[name] = {"belowFloorCm": round(100.0 * min(clearance, 0.0), 1),
                          "targetDeg": round(target, 1), "appliedDeg": round(self.angles[name], 1)}
            if abs(self.angles[name]) < 1e-4:
                continue
            rotation = _axis_rotation(axis, self.angles[name])
            new_global = rotation @ g[joint]
            local[joint] = g[PARENTS[joint]].T @ new_global
            if child is not None:                 # the measured shank keeps its global orientation
                local[child] = new_global.T @ g[child]
            g = np.asarray(_globals(local))
            p = self._positions(g)
        return local, diag

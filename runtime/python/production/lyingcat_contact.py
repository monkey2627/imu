"""Four-point kneeling support constraint for the lying-cat exercise.

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
* The pelvis is held near its slow mean pitch (it follows 30 % of the waist
  sensor's fast cat/cow pitch), so the hips stay over the knees and the waist
  reading shows up as lumbar bend between pelvis and spine1.
* The remaining height difference between hands and knees is absorbed by pitching the
  upper spine (joints 6 and 9, above the waist sensor) -- the cat/cow degree of
  freedom -- instead of slanting the limbs.  Only a residual beyond the spine
  limit is closed by tilting the upper arms / thighs.

2026-11-19 rewrite: the previous version forced both forearms vertical and
reached the plane by slanting thighs ~45 deg and upper arms ~40 deg on the real
recording unity_20261116_195301 (see recordings/action_eval_20261119).

2026-11-21 (baseline = git 420afe8, the rule behind the 2026-11-20 live
recordings; legs, pelvis and spine solving are unchanged from it):

* Support weights are seeded from their targets on the first frame after a
  reset.  run.py resets the rule on every start-pose rebase; ramping from 0
  then showed ~5 frames of the raw network legs (knees 20 cm off the mat live).
* Arm twist is distributed instead of piling up at the elbow: the upper arm is
  rolled about its own bone axis so the elbow's local twist stays within
  ELBOW_TWIST_MAX_DEG (no joint position moves; forearm global = sensor).
* Palm flattening keeps the wrist swing (dorsiflexion, needed for a flat palm)
  but caps the wrist's twist about the forearm axis at WRIST_TWIST_MAX_DEG and
  skips near-180 deg requests (ill-conditioned; they flipped the hand).
* No knee/shank latch and no positional contact anchors (the intermediate
  2026-11-20 version, analysis/lyingcat_variants/v6_knee_latch_20261121.py,
  lifted knees by up to 19 cm and bent elbows ~45 deg).

2026-11-21 (b): pelvis/hips move, palms held by the trunk (snapshot of the
step before: analysis/lyingcat_variants/v7_twist_only_20261121.py):

* The pelvis follows the full network/waist pitch (PELVIS_FOLLOW 0.3 -> 1.0).
  At 0.3 the pelvis moved 9-10 deg while the waist sensor moved 31 deg: the
  hips and thighs looked frozen and all of cat/cow sat in one waist kink.
* Knees: once four-point support has been stable for STABLE_S, each knee is
  latched directly under its hip.  Every frame the thigh is aimed so the knee
  keeps that spot horizontally (Unity/the renderer ground the body vertically,
  so the knee height follows the hip); the shank keeps its level global, so
  the knee angle follows the thigh.  No XYZ lock, no shank lock.
* Palms: latched at the same moment.  With the palm fixed, the measured
  forearm tilt says where the shoulder must be
  (shoulder = palm - arm_length * forearm_direction).  spine2/spine3 pitch and
  a small spine3 roll are solved for that shoulder target (fore-aft + each
  shoulder's height) instead of the old height-only 1-DOF pitch, which moved
  the shoulders fore-aft and slid the straight-arm hands 9-19 cm.  A bounded
  (<= ARM_BEND_MAX_DEG) upper-arm swing closes the rest; the elbow twist cap
  then removes any roll it introduced.
* Spine1 global stays the waist sensor; the pelvis-spine1 bend is unchanged.
"""
from __future__ import annotations

import math

import numpy as np

from production.arm_prior import PARENTS, _globals, _rotation_between, _scaled
from production.bridge_contact import _ramp, _rest_offsets, _unit

WEIGHT_RATE_PER_S = 4.0
SPINE_TIME_CONSTANT_S = 0.03         # solve is warm-started; 0.10 s lagged cat/cow by ~2 cm at the palms
MAX_SPINE_DEG = 40.0
# Pelvis pitch = slow mean + PELVIS_FOLLOW * (network pitch - slow mean).  1.0 lets
# the pelvis (and with it hips/thighs) take its full part in cat/cow.
PELVIS_SLOW_TIME_CONSTANT_S = 6.0
PELVIS_FOLLOW = 1.0
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
# Local twist limits along the arm (deg).  The forearm sensor's roll relative to
# the straight-aimed upper arm is ~25 deg on the real recordings; humeral rotation
# (shoulder) absorbs it so the elbow hinge carries almost no twist.
ELBOW_TWIST_MAX_DEG = 8.0
WRIST_TWIST_MAX_DEG = 12.0
MAX_FLATTEN_DEG = 150.0              # larger palm-flattening requests are ill-conditioned: skip them
# Contact latching (knees under hips, palms) once four-point support is stable.
STABLE_S = 0.5
KNEE_LEAK_S = 20.0
HAND_LEAK_S = 60.0
SPINE_SMOOTH_DEG = 0.15        # regulariser: penalise spine2 vs spine3 disagreement (per deg, in m)
ARM_BEND_MAX_DEG = 15.0
MAX_ROLL_DEG = 15.0
ROLL_REG = 0.05


def _twist_angle(rotation, axis):
    """Twist (deg) of ``rotation`` about unit ``axis`` (child frame), swing-twist split."""
    w = 0.5 * math.sqrt(max(0.0, 1.0 + float(np.trace(rotation))))
    v = np.array([rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0],
                  rotation[1, 0] - rotation[0, 1]])
    if w > 1e-6:
        v = v / (4.0 * w)
    else:                                    # 180 deg rotation: axis from the symmetric part
        sym = 0.5 * (rotation + np.eye(3))
        v = sym[int(np.argmax(np.diag(sym)))]
        v = v / max(np.linalg.norm(v), 1e-12)
    angle = 2.0 * math.degrees(math.atan2(float(v @ axis), w))
    return (angle + 180.0) % 360.0 - 180.0


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
        self.spine_ab = np.zeros(3)           # spine2 pitch, spine3 pitch, spine3 roll (deg)
        self.pelvis_slow_deg = None
        self.stable_s = 0.0
        self.knee_anchor = None
        self.hand_anchor = None
        self.seeded = False
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

    def _stabilise_pelvis(self, local, support, dt):
        """Hold the pelvis pitch near its slow mean; spine1 and the shanks keep their globals."""
        g = _globals(local)
        head = g[0] @ UP
        current = _elevation(head)
        alpha = 1.0 - math.exp(-dt / PELVIS_SLOW_TIME_CONSTANT_S)
        if self.pelvis_slow_deg is None:
            self.pelvis_slow_deg = current
        self.pelvis_slow_deg += alpha * (current - self.pelvis_slow_deg)
        desired = self.pelvis_slow_deg + PELVIS_FOLLOW * (current - self.pelvis_slow_deg)
        delta = support * (desired - current)
        horizontal = head.copy()
        horizontal[1] = 0.0
        if support <= 0.0 or abs(delta) < 1e-6 or np.linalg.norm(horizontal) < 1e-6:
            return current, current
        horizontal = _unit(horizontal)
        angle = math.radians(current + delta)
        wanted = math.cos(angle) * horizontal + math.sin(angle) * UP
        keep = {joint: g[joint].copy() for joint in (3, 4, 5)}
        local[0] = _rotation_between(head, wanted) @ local[0]
        self._restore(local, keep)
        return current, current + delta

    @staticmethod
    def _restore(local, globals_by_joint):
        for joint in sorted(globals_by_joint):
            g = _globals(local)
            local[joint] = g[PARENTS[joint]].T @ globals_by_joint[joint]

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
            if not self.seeded:
                # First frame after a reset: the previous output (same action) was
                # already constrained, so start at the sensed support state instead of 0.
                self.weights[key] = target
            else:
                self.weights[key] += float(np.clip(target - self.weights[key], -step, step))
        self.seeded = True
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
            # Keep the swing (wrist dorsiflexion lays the palm flat) but cap the
            # wrist's roll about the forearm axis: a large wrist twist is what
            # makes the SMPL forearm skin look wrung.
            axis = _unit(self.offsets[hand])
            twist = _twist_angle(wanted_local, axis)
            capped = float(np.clip(twist, -WRIST_TWIST_MAX_DEG, WRIST_TWIST_MAX_DEG))
            wanted_local = wanted_local @ _axis_rotation(axis, capped - twist)
            change = local[wrist].T @ wanted_local
            angle = math.degrees(math.acos(float(np.clip((np.trace(change) - 1.0) / 2.0, -1.0, 1.0))))
            if angle > MAX_FLATTEN_DEG:
                continue
            local[wrist] = local[wrist] @ _scaled(change, weight)
            local[hand] = local[hand] @ _scaled(local[hand].T, weight)

    def _limit_elbow_twist(self, local):
        """Move the elbow's excess local twist into a roll of the upper arm about its own bone.

        The upper arm is unobserved; rolling it about its own axis moves no joint
        position (elbow and, with the forearm global kept, the wrist stay put), so
        hand contact, forearm direction and spine are untouched.  Only the
        elbow's local roll and the shoulder's (humeral rotation) change.
        """
        for side, upper, forearm, wrist, _hand in ARMS:
            weight = self.weights[f"{side}Arm"]
            if weight <= 0.0:
                continue
            g = _globals(local)
            fore_global = g[forearm].copy()
            bone = _unit(self.offsets[forearm])          # upper-arm bone, upper-arm frame
            axis = _unit(self.offsets[wrist])            # forearm bone, forearm frame

            def elbow_twist(phi):
                return _twist_angle(_axis_rotation(bone, phi).T @ local[forearm], axis)

            twist = elbow_twist(0.0)
            wanted = float(np.clip(twist, -ELBOW_TWIST_MAX_DEG, ELBOW_TWIST_MAX_DEG))
            goal = twist + weight * (wanted - twist)
            if abs(goal - twist) < 1e-3:
                continue
            # twist(phi) ~= twist(0) - phi for a near-straight elbow; refine by secant.
            phi0, f0 = 0.0, twist - goal
            phi1 = f0
            f1 = elbow_twist(phi1) - goal
            for _ in range(6):
                if abs(f1) < 0.05 or abs(f1 - f0) < 1e-9:
                    break
                phi0, phi1, f0 = phi1, phi1 - f1 * (phi1 - phi0) / (f1 - f0), f1
                f1 = elbow_twist(phi1) - goal
            if abs(f1) > abs(f0) and abs(phi1) > 90.0:
                continue
            local[upper] = local[upper] @ _axis_rotation(bone, phi1)
            local[forearm] = (g[PARENTS[upper]] @ local[upper]).T @ fore_global

    # --------------------------------------------------------------- legs
    def _legs(self, local, dt):
        """Thigh aimed at the latched knee (or straight down before the latch); shank global kept."""
        leak = 1.0 - math.exp(-dt / KNEE_LEAK_S)
        for side, thigh, shank, _ankle, _foot in LEGS:
            weight = self.weights[f"{side}Leg"]
            g = _globals(local)
            p = self._positions(g)
            length = float(np.linalg.norm(self.offsets[shank]))
            under_hip = p[thigh] + length * DOWN
            if self.knee_anchor is None:
                wanted = DOWN
            else:
                anchor = self.knee_anchor[shank]
                anchor[[0, 2]] += leak * (under_hip[[0, 2]] - anchor[[0, 2]])
                # Two-segment leg: the knee keeps its latched spot on the mat
                # horizontally (Unity / the renderer ground the whole body, so its
                # absolute height follows the hip); the thigh stays as vertical as
                # that allows and the level shank keeps its global, so the knee
                # angle follows the thigh.
                horizontal = anchor - p[thigh]
                horizontal[1] = 0.0
                dist = float(np.linalg.norm(horizontal))
                limit = 0.5 * length                        # thigh at most 30 deg off vertical
                if dist > limit:
                    horizontal *= limit / dist
                wanted = horizontal + math.sqrt(length * length - float(horizontal @ horizontal)) * DOWN
            self._aim(local, g, thigh, shank, _unit(wanted), weight)

    # --------------------------------------------------------------- spine
    def _arm_lengths(self):
        return [float(np.linalg.norm(self.offsets[f]) + np.linalg.norm(self.offsets[w]))
                for _s, _u, f, w, _h in ARMS]

    def _rot9(self, axis, head, a, b, c):
        """Global rotation added at spine3: pitch a+b about ``axis`` after roll c about ``head``."""
        return _axis_rotation(axis, a + b) @ _axis_rotation(head, c)

    def _shoulders(self, p, axis, a, b, c=0.0, head=None):
        ra = _axis_rotation(axis, a)
        r9 = _axis_rotation(axis, a + b) if head is None else self._rot9(axis, head, a, b, c)
        p9 = p[6] + ra @ (p[9] - p[6])
        return np.array([p9 + r9 @ (p[j] - p[9]) for j in (16, 17)])

    def _solve_spine(self, local, axis, head):
        """(spine2, spine3) pitch putting the straight-arm shoulders over the latched palms."""
        g = _globals(local)
        p = self._positions(g)
        arm_w = np.array([self.weights["leftArm"], self.weights["rightArm"]])
        arm_w = arm_w / max(arm_w.sum(), 1e-9)
        lengths = self._arm_lengths()
        fdirs = [_unit(g[f] @ self.offsets[w]) for _s, _u, f, w, _h in ARMS]
        targets = np.array([self.hand_anchor[w] - L * fd
                            for (_s, _u, _f, w, _h), L, fd in zip(ARMS, lengths, fdirs)])

        def residual(x):
            s = self._shoulders(p, axis, x[0], x[1], x[2], head)
            diff = s - targets
            e = arm_w @ diff
            return np.array([e @ head, arm_w[0] * diff[0, 1] * 2, arm_w[1] * diff[1, 1] * 2,
                             SPINE_SMOOTH_DEG * 0.01 * (x[0] - x[1]), ROLL_REG * 0.01 * x[2]])

        x = self.spine_ab.copy()
        lim = np.array([MAX_SPINE_DEG, MAX_SPINE_DEG, MAX_ROLL_DEG])
        for _ in range(8):
            r = residual(x)
            jac = np.zeros((5, 3))
            for k in range(3):
                dx = np.zeros(3)
                dx[k] = 0.5
                jac[:, k] = (residual(x + dx) - r) / 0.5
            step = np.linalg.solve(jac.T @ jac + 1e-6 * np.eye(3), -jac.T @ r)
            x = np.clip(x + step, -lim, lim)
            if np.abs(step).max() < 0.02:
                break
        return x

    def _apply_spine(self, local, axis, head, a, b, c):
        g = _globals(local)
        new6 = _axis_rotation(axis, a) @ g[6]
        new9 = self._rot9(axis, head, a, b, c) @ g[9]
        local[6] = g[3].T @ new6
        local[9] = new6.T @ new9

    # --------------------------------------------------------------- arms
    def _hold_palms(self, local):
        """Bounded upper-arm swing toward the latched palm; the wrist never goes below the knees."""
        g = _globals(local)
        p = self._positions(g)
        floor = 0.5 * (p[4, 1] + p[5, 1])
        for side, upper, forearm, wrist, _h in ARMS:
            weight = self.weights[f"{side}Arm"]
            if weight <= 0.0:
                continue
            g = _globals(local)
            p = self._positions(g)
            fvec = g[forearm] @ self.offsets[wrist]
            fdir = _unit(fvec)
            target = self.hand_anchor[wrist].copy()
            target[1] = floor
            wanted = _unit(target - fvec - p[upper])
            bend = math.degrees(math.acos(float(np.clip(wanted @ fdir, -1, 1))))
            if bend > ARM_BEND_MAX_DEG:
                wanted = _unit(_scaled(_rotation_between(fdir, wanted), ARM_BEND_MAX_DEG / bend) @ fdir)
            current = g[upper] @ self.offsets[forearm]
            full = _rotation_between(current, wanted)
            low = min(p[wrist, 1], floor) - 0.01

            def wrist_y(s):
                return p[upper, 1] + float((_scaled(full, s) @ current)[1]) + fvec[1]

            lo, hi = 0.0, weight
            if wrist_y(hi) < low:
                for _ in range(12):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if wrist_y(mid) >= low else (lo, mid)
                hi = lo
            self._set_global(local, g, upper, _scaled(full, hi) @ g[upper], forearm)

    @staticmethod
    def _basis(primary, secondary):
        primary = _unit(primary)
        secondary = secondary - float(np.dot(secondary, primary)) * primary
        secondary = _unit(secondary)
        return np.column_stack((primary, secondary, np.cross(primary, secondary)))

    # ------------------------------------------------------------------ entry
    def apply(self, local, dt, measured=None, labels=None):
        local = self._anchor(np.asarray(local, dtype=np.float64).copy(), measured, labels)
        g = _globals(local)
        elevations = self._update_weights(g, dt)
        support = min(0.5 * (self.weights["leftArm"] + self.weights["rightArm"]),
                      0.5 * (self.weights["leftLeg"] + self.weights["rightLeg"]))
        if min(self.weights.values()) > 0.95:
            self.stable_s += dt
        else:
            self.stable_s, self.knee_anchor, self.hand_anchor = 0.0, None, None
        pelvis_before, pelvis_after = self._stabilise_pelvis(local, support, dt)

        self._level_legs(local)
        self._legs(local, dt)
        g = _globals(local)
        sensed = {joint: g[joint].copy() for joint in (18, 19)}
        if support > 0.0:
            for joint in (6, 9):
                local[joint] = local[joint] @ _scaled(local[joint].T, support)
            self._restore(local, sensed)
        g = _globals(local)
        p = self._positions(g)
        head = self._headward(g, p)
        axis = np.cross(UP, head)
        alpha = 1.0 - math.exp(-dt / SPINE_TIME_CONSTANT_S)
        if support <= 0.05:
            goal = np.zeros(3)
        elif self.hand_anchor is None:
            t = self._spine_target(local, axis)
            goal = np.array([t / 2.0, t / 2.0, 0.0])
        else:
            # palms latched: their height is the current knee plane (same support
            # surface); horizontally leak toward where the 1-DOF solve would put them
            knee_plane = 0.5 * (p[4, 1] + p[5, 1])
            for w in (20, 21):
                self.hand_anchor[w][1] = knee_plane
            t = self._spine_target(local, axis)
            free = self._shoulders(p, axis, t / 2.0, t / 2.0)
            leak = 1.0 - math.exp(-dt / HAND_LEAK_S)
            for (_s, _u, f, w, _h), L, s in zip(ARMS, self._arm_lengths(), free):
                free_wrist = s + L * _unit(g[f] @ self.offsets[w])
                self.hand_anchor[w][[0, 2]] += leak * (free_wrist[[0, 2]] - self.hand_anchor[w][[0, 2]])
            goal = self._solve_spine(local, axis, head)
        self.spine_ab += alpha * (goal - self.spine_ab)
        self._apply_spine(local, axis, head, *(self.spine_ab * support))
        self._restore(local, sensed)
        self._vertical_supports(local, ("Arm",))

        _, floor = self._support_heights(local)
        if self.hand_anchor is not None:
            self._hold_palms(local)
        elif floor is not None:
            self._close_residual(local, floor)
        self._limit_elbow_twist(local)
        self._flatten_hands(local)

        if self.stable_s >= STABLE_S and self.hand_anchor is None:
            p = self._positions(_globals(local))
            self.hand_anchor = {w: p[w].copy() for w in (20, 21)}
            # knees latch directly under the hips (the pre-latch residual closing may
            # have slanted a thigh; latching that would bake the slant in)
            self.knee_anchor = {k: p[h] + float(np.linalg.norm(self.offsets[k])) * DOWN
                                for h, k in ((1, 4), (2, 5))}

        g = _globals(local)
        p = self._positions(g)
        diag = {"enabled": True, "weights": {k: round(v, 3) for k, v in self.weights.items()},
                "spineDeg": [round(float(v), 1) for v in self.spine_ab * support],
                "pelvisPitchDeg": {"network": round(pelvis_before, 1), "applied": round(pelvis_after, 1)},
                "sensorElevationDeg": elevations, "anchorsActive": self.hand_anchor is not None,
                "thighElevationDeg": {side: round(_elevation(g[thigh] @ self.offsets[shank]), 1)
                                      for side, thigh, shank, _a, _f in LEGS},
                "elbowTwistDeg": {side: round(_twist_angle(local[forearm], _unit(self.offsets[wrist])), 1)
                                  for side, _u, forearm, wrist, _h in ARMS},
                "wristTwistDeg": {side: round(_twist_angle(local[wrist], _unit(self.offsets[hand])), 1)
                                  for side, _u, _f, wrist, hand in ARMS}}
        if floor is not None:
            diag["supportPlaneM"] = round(floor, 3)
            diag["afterErrorCm"] = {name: round(100.0 * float(p[joint, 1] - floor), 1)
                                    for name, joint, key in (("leftWrist", 20, "leftArm"), ("rightWrist", 21, "rightArm"),
                                                             ("leftKnee", 4, "leftLeg"), ("rightKnee", 5, "rightLeg"))
                                    if self.weights[key] > 0.5}
        if self.hand_anchor is not None:
            diag["anchorErrorCm"] = {
                name: round(100.0 * float(np.linalg.norm(p[j] - anchors[j])), 1)
                for name, j, anchors in (("leftWrist", 20, self.hand_anchor), ("rightWrist", 21, self.hand_anchor),
                                         ("leftKnee", 4, self.knee_anchor), ("rightKnee", 5, self.knee_anchor))}
        return local, diag

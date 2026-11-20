"""[Experiment 2026-11-21] Pelvis/hips take part in cat/cow; palms held by the trunk, not the arm.

On top of production/lyingcat_contact.py:

* Pelvis follows ``PELVIS_FOLLOW`` of the network/waist pitch (production: 0.3,
  which left the hips almost still and put a 20 deg kink at the waist).
* Knees: after four-point support settles, each knee is latched in the
  (root-fixed) frame and the thigh is aimed at it every frame -- the hip moves
  with the pelvis tilt, so the thigh angle and therefore the knee angle change,
  while the shank stays level on the mat.  The latch slowly leaks toward the
  knee-under-hip position so it cannot accumulate an offset.
* Palms: after support settles each wrist is latched.  A real palm stays put
  and the measured forearm tilt then says where the shoulder is:
  shoulder = palm - arm_length * forearm_direction.  The upper spine solves
  spine2 / spine3 pitch (2 DOF) for BOTH the height and the fore-aft position of
  that shoulder target, instead of only the height (the 1-DOF solve moved the
  shoulders fore-aft and dragged the straight-arm hands 9-19 cm).
  A small bounded upper-arm swing closes what is left.
"""
from __future__ import annotations

import math

import numpy as np

from production import lyingcat_contact as base
from production.arm_prior import PARENTS, _globals, _rotation_between, _scaled
from production.bridge_contact import _unit

PELVIS_FOLLOW = 1.0
STABLE_S = 0.5
KNEE_LEAK_S = 20.0
HAND_LEAK_S = 60.0
SPINE_SMOOTH_DEG = 0.15        # regulariser: penalise spine2 vs spine3 disagreement (per deg, in m)
ARM_BEND_MAX_DEG = 15.0
MAX_ROLL_DEG = 15.0
ROLL_REG = 0.05
SPINE_TIME_CONSTANT_S = 0.03     # solve already warm-started; 0.10 s lagged cat/cow by ~2 cm at the palms
UP, DOWN = base.UP, base.DOWN


class LyingCatContact(base.LyingCatContact):
    pelvis_follow = PELVIS_FOLLOW

    def reset(self, reason):
        super().reset(reason)
        self.stable_s = 0.0
        self.knee_anchor = None
        self.hand_anchor = None
        self.spine_ab = np.zeros(3)

    # --------------------------------------------------------------- pelvis
    def _stabilise_pelvis(self, local, support, dt):
        saved = base.PELVIS_FOLLOW
        base.PELVIS_FOLLOW = self.pelvis_follow
        try:
            return super()._stabilise_pelvis(local, support, dt)
        finally:
            base.PELVIS_FOLLOW = saved

    # --------------------------------------------------------------- legs
    def _legs(self, local, dt):
        """Thigh aimed at the latched knee (or straight down before the latch); shank global kept."""
        leak = 1.0 - math.exp(-dt / KNEE_LEAK_S)
        for side, thigh, shank, _ankle, _foot in base.LEGS:
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
                for _s, _u, f, w, _h in base.ARMS]

    def _rot9(self, axis, head, a, b, c):
        """Global rotation added at spine3: pitch a+b about ``axis`` after roll c about ``head``."""
        return base._axis_rotation(axis, a + b) @ base._axis_rotation(head, c)

    def _shoulders(self, p, axis, a, b, c=0.0, head=None):
        ra = base._axis_rotation(axis, a)
        r9 = base._axis_rotation(axis, a + b) if head is None else self._rot9(axis, head, a, b, c)
        p9 = p[6] + ra @ (p[9] - p[6])
        return np.array([p9 + r9 @ (p[j] - p[9]) for j in (16, 17)])

    def _solve_spine(self, local, axis, head):
        """(spine2, spine3) pitch putting the straight-arm shoulders over the latched palms."""
        g = _globals(local)
        p = self._positions(g)
        arm_w = np.array([self.weights["leftArm"], self.weights["rightArm"]])
        arm_w = arm_w / max(arm_w.sum(), 1e-9)
        lengths = self._arm_lengths()
        fdirs = [_unit(g[f] @ self.offsets[w]) for _s, _u, f, w, _h in base.ARMS]
        targets = np.array([self.hand_anchor[w] - L * fd
                            for (_s, _u, _f, w, _h), L, fd in zip(base.ARMS, lengths, fdirs)])

        def residual(x):
            s = self._shoulders(p, axis, x[0], x[1], x[2], head)
            diff = s - targets
            e = arm_w @ diff
            return np.array([e @ head, arm_w[0] * diff[0, 1] * 2, arm_w[1] * diff[1, 1] * 2,
                             SPINE_SMOOTH_DEG * 0.01 * (x[0] - x[1]), ROLL_REG * 0.01 * x[2]])

        x = self.spine_ab.copy()
        lim = np.array([base.MAX_SPINE_DEG, base.MAX_SPINE_DEG, MAX_ROLL_DEG])
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

    def _apply_spine2(self, local, axis, head, a, b, c):
        g = _globals(local)
        new6 = base._axis_rotation(axis, a) @ g[6]
        new9 = self._rot9(axis, head, a, b, c) @ g[9]
        local[6] = g[3].T @ new6
        local[9] = new6.T @ new9

    # --------------------------------------------------------------- arms
    def _hold_palms(self, local):
        """Bounded upper-arm swing toward the latched palm; the wrist never goes below the knees."""
        g = _globals(local)
        p = self._positions(g)
        floor = 0.5 * (p[4, 1] + p[5, 1])
        for side, upper, forearm, wrist, _h in base.ARMS:
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

    # --------------------------------------------------------------- entry
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
            for (_s, _u, f, w, _h), L, s in zip(base.ARMS, self._arm_lengths(), free):
                free_wrist = s + L * _unit(g[f] @ self.offsets[w])
                self.hand_anchor[w][[0, 2]] += leak * (free_wrist[[0, 2]] - self.hand_anchor[w][[0, 2]])
            goal = self._solve_spine(local, axis, head)
        self.spine_ab += alpha * (goal - self.spine_ab)
        self._apply_spine2(local, axis, head, *(self.spine_ab * support))
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
                "sensorElevationDeg": elevations, "anchorsActive": self.hand_anchor is not None}
        if self.hand_anchor is not None:
            diag["anchorErrorCm"] = {
                name: round(100.0 * float(np.linalg.norm(p[j] - anchors[j])), 1)
                for name, j, anchors in (("leftWrist", 20, self.hand_anchor), ("rightWrist", 21, self.hand_anchor),
                                         ("leftKnee", 4, self.knee_anchor), ("rightKnee", 5, self.knee_anchor))}
        return local, diag

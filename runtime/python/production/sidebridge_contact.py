"""Side-bridge (forearm side plank, legs straight) support rule.  Production: frame_chain.postprocess_output
applies it only for the actions sidebridge_left / sidebridge_right (user-approved 2026-11-22 after
analysis/sidebridge_compare_video.py, column 3).

Correct exercise (user, 2026-11-22): lying on the support side, the support elbow under the shoulder
with the forearm flat on the mat pointing forward, both legs straight and stacked, the top arm lying
along the body with the hand on the thigh.  The hips lift off the mat until head, trunk and legs form
one line, then lower again.  Supports: the support forearm and the support leg (foot; at rest also
hip and thigh).

What the network gets wrong on the real recording sidebridge_left_20261122_033550:
* the support forearm hovers 8-32 cm above the floor, elbow bent only 40-75 deg;
* knees bent ~30 deg, top hip flexed 30-40 deg, top leg lifted away (start pose is legs straight);
* the hip lift is invisible although both shank sensors rise 11-20 deg in every repetition.

Rule (measured = calibrated sensor orientations of the output's frame, as postprocess_output gets):
1. legs: each thigh takes its measured shank's orientation (knee straight, stacked by the hips);
   the shank rise then lifts the pelvis exactly as a straight leg does;
2. top arm: upper arm aligned with the measured forearm (straight arm resting along the thigh);
3. support arm: forearm = measurement; the upper arm is turned so the forearm's lowest surface
   point lies on the floor, the floor being the lowest surface point of the support leg and hip.
   The elbow bend follows from the geometry.
4. spine: the chest has no sensor.  When even a vertical upper arm cannot reach the floor (the network
   carries the shoulders up with the lifted pelvis), spine1-3 bend toward the support side until the
   shoulder is one near-vertical upper arm above the forearm -- the forearm holds the shoulders while
   the hips lift.
5. head: not of interest (user) -- neck and head continue the spine in one line.
Start template (as dead bug / bird dog use their start pose): in the start hold the hip, the support
leg and the forearm all lie on the mat.  A straight leg then slopes down from the hip to the foot
(the hip is ~10 cm thicker than the ankle), but the shank sensors read ~0 deg.  For the first
REST_CAPTURE_OUTPUTS gated outputs after a reset the rule solves the leg pitch that puts the support
hip and foot on the same floor and keeps that offset; afterwards the lift is the measured shank rise
relative to the start hold.
Landing snap (user, 2026-11-22: the hips were fully down between repetitions): after the first lift
the sensors keep a common ~4 deg offset (the support forearm, flat on the mat, reads +4 deg as well),
so the hips hovered 5-8 cm.  A small state machine on the support-shank rise: "up" once it exceeds
the floor baseline by UP_DEG (a held lift is never absorbed); "down" again when it has fallen back by
LANDED_FRACTION of that lift and is still -- the baseline then snaps to the reading (time constant
SNAP_TIME_CONSTANT_S).  While down it only follows drift within CREEP_DEG.  The lift is therefore
measured from the latest landing.
Everything fades in/out with a side-lying gate (pelvis lateral axis near vertical, rate limited).
"""
from __future__ import annotations

import math

import numpy as np

from production.arm_prior import PARENTS, _globals

REST_CAPTURE_OUTPUTS = 15      # start hold right after the rebase: ~0.5 s of outputs
MAX_LEG_PITCH_DEG = 25.0
UP_DEG = 7.0                    # support-shank rise above the floor baseline: hips lifted (no snapping)
LANDED_FRACTION = 0.6           # fallen back by this fraction of the lift, and still: landed -> snap
CREEP_DEG = 3.0                 # while down, follow slow drift within this band
SNAP_STILL_DEG = 1.5            # rise range within SNAP_WINDOW_S that counts as still
SNAP_WINDOW_S = 0.5
SNAP_TIME_CONSTANT_S = 0.4
MAX_DRIFT_DEG = 10.0
SUPPORT_UPPER_ARM_DEG = 80.0    # upper arm elevation (below horizontal) the spine bend aims for
MAX_SPINE_BEND_DEG = 35.0
SPINE = (3, 6, 9)
TOP_ARM_ABOVE_HIP_M = 0.08      # the top arm lies on the side of the hip, not through it
GATE_ROLL = (0.55, 0.75)        # |pelvis left axis . up|: side lying above 0.75
WEIGHT_RATE_PER_S = 4.0
UP = np.array([0.0, 1.0, 0.0])
SIDES = {   # support side -> joints / labels
    "left": dict(sign=+1.0, hip=1, knee=4, top_hip=2, top_foot=(5, 8, 11), support_leg_regions=(0, 1, 4, 7, 10),
                 sh=16, elbow=18, collar=13, arm_label="lw",
                 top_sh=17, top_elbow=19, top_collar=14, top_label="rw"),
    "right": dict(sign=-1.0, hip=2, knee=5, top_hip=1, top_foot=(4, 7, 10), support_leg_regions=(0, 2, 5, 8, 11),
                  sh=17, elbow=19, collar=14, arm_label="rw",
                  top_sh=16, top_elbow=18, top_collar=13, top_label="lw"),
}


def _ramp(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _rotation_between(a, b):
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    v = np.cross(a, b)
    s, c = np.linalg.norm(v), float(np.clip(a @ b, -1.0, 1.0))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    angle = math.atan2(s, c)
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def _scaled(rotation, weight):
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(Rotation.from_matrix(rotation).as_rotvec() * weight).as_matrix()


def _root(f, lo, hi, tol=1e-3, iterations=8):
    """Root of a monotone f on [lo, hi] (Illinois regula falsi); None if f does not change sign."""
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return None
    side = 0
    for _ in range(iterations):
        x = hi - fhi * (hi - lo) / (fhi - flo) if fhi != flo else 0.5 * (lo + hi)
        fx = f(x)
        if abs(fx) < tol:
            return x
        if fx * fhi > 0:
            hi, fhi = x, fx
            if side == -1:
                flo *= 0.5
            side = -1
        else:
            lo, flo = x, fx
            if side == 1:
                fhi *= 0.5
            side = 1
    return hi - fhi * (hi - lo) / (fhi - flo) if fhi != flo else 0.5 * (lo + hi)


def _template():
    from production.deadbug_floor import _template as floor_template
    return floor_template()


class SideBridgeContact:
    def __init__(self, side: str = "left", legs: bool = True, arm: bool = True, top_arm: bool = True,
                 head: bool = True) -> None:
        self.side, self.cfg = side, SIDES[side]
        self.legs, self.arm, self.top_arm, self.head = legs, arm, top_arm, head
        joints, vertices, dominant = _template()
        self.offsets = np.zeros_like(joints)
        for j in range(1, 24):
            self.offsets[j] = joints[j] - joints[PARENTS[j]]
        def region(bones, step=4):
            index = np.flatnonzero(np.isin(dominant, bones))[::step]
            return dominant[index], vertices[index] - joints[dominant[index]]
        self.floor_region = region(self.cfg["support_leg_regions"])
        self.hip_region = region(self.cfg["support_leg_regions"][:2])
        self.foot_region = region(self.cfg["support_leg_regions"][2:])
        self.top_foot_region = region(self.cfg["top_foot"])
        self.forearm_region = region((self.cfg["elbow"],), 1)
        self.reset("init")

    def reset(self, reason: str) -> None:
        self.weight = 0.0
        self.reason = reason
        self.rest_pitch = None         # (both legs, top leg) pitch offsets (deg) of the start hold
        self._pitch_samples = []
        self.rest_rise = None          # support-shank rise (deg) of the start hold
        self.drift = 0.0               # floor-baseline change since the start hold (landing snap)
        self._rises = []
        self.lifted = False
        self.peak = 0.0

    def _positions(self, g):
        p = np.zeros((24, 3))
        for j in range(1, 24):
            p[j] = p[PARENTS[j]] + g[PARENTS[j]] @ self.offsets[j]
        return p

    def _lowest(self, g, p, region):
        bones, local = region
        return float((p[bones] + np.einsum("nij,nj->ni", g[bones], local))[:, 1].min())

    def _leg_rotation(self, g, degrees):
        """Pitch both legs about the support hip's horizontal axis (+: feet up relative to the hip)."""
        leg = g[self.cfg["knee"]] @ np.array([0.0, -1.0, 0.0])
        axis = np.cross(leg, UP)
        axis /= max(np.linalg.norm(axis), 1e-9)
        a = math.radians(degrees)
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k

    def _with_leg_pitch(self, pose, degrees, hips=(1, 2)):
        g = np.asarray(_globals(pose))
        turn = self._leg_rotation(g, degrees)
        out = pose.copy()
        for hip in hips:
            out[hip] = g[0].T @ turn @ g[hip]
        return out

    def _solve_top_pitch(self, pose):
        """Top-leg pitch at which the top foot rests on the support foot (stacked legs)."""
        top = (self.cfg["top_hip"],)
        def gap(degrees):
            g = np.asarray(_globals(self._with_leg_pitch(pose, degrees, top)))
            p = self._positions(g)
            bones, local = self.foot_region
            support_top = float((p[bones] + np.einsum("nij,nj->ni", g[bones], local))[:, 1].max())
            return self._lowest(g, p, self.top_foot_region) - support_top
        root = _root(gap, -MAX_LEG_PITCH_DEG, MAX_LEG_PITCH_DEG)
        return 0.0 if root is None else root

    def _solve_rest_pitch(self, pose):
        """Leg pitch at which the support hip/thigh and the support foot touch the same floor."""
        def gap(degrees):
            g = np.asarray(_globals(self._with_leg_pitch(pose, degrees)))
            p = self._positions(g)
            return self._lowest(g, p, self.hip_region) - self._lowest(g, p, self.foot_region)
        root = _root(gap, -MAX_LEG_PITCH_DEG, MAX_LEG_PITCH_DEG)
        return 0.0 if root is None else root

    def _with_spine_bend(self, pose, degrees):
        """Bend spine1-3 (equal thirds) about the horizontal axis normal to the spine."""
        g = np.asarray(_globals(pose))
        spine = g[3] @ np.array([0.0, 1.0, 0.0])
        axis = np.cross(spine, UP)
        axis /= max(np.linalg.norm(axis), 1e-9)
        out = pose.copy()
        previous = g[0]
        for n, joint in enumerate(SPINE, 1):
            a = math.radians(degrees * n / len(SPINE))
            k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            turned = (np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k) @ g[joint]
            out[joint] = previous.T @ turned
            previous = turned
        return out

    def apply(self, local, dt, measured=None, labels=None):
        """local (24,3,3) SMPL local rotations -> (corrected local, diagnostics)."""
        cfg = self.cfg
        out = np.asarray(local, dtype=np.float64).copy()
        g = np.asarray(_globals(out))
        roll = float(g[0][1, 0])                      # pelvis left axis . up (<0: left side down)
        target = _ramp(-cfg["sign"] * roll, *GATE_ROLL)
        step = WEIGHT_RATE_PER_S * dt
        self.weight += float(np.clip(target - self.weight, -step, step))
        w = self.weight
        diag = {"side": self.side, "weight": round(w, 3), "roll": round(roll, 3)}
        if w <= 1e-6 or measured is None or labels is None:
            return out, diag
        meas = {label: np.asarray(measured[labels.index(label)], np.float64) for label in labels}
        new = out.copy()

        if self.legs:
            for hip, knee, label in ((1, 4, "ls"), (2, 5, "rs")):
                if label in meas:      # straight knee: the thigh takes the measured shank orientation
                    new[hip] = g[0].T @ meas[label]
                    new[knee] = np.eye(3)
            support = meas.get("ls" if self.side == "left" else "rs")
            rise = (None if support is None
                    else math.degrees(math.asin(float(np.clip(-(support @ np.array([0.0, -1.0, 0.0]))[1], -1, 1)))))
            if self.rest_pitch is None:
                pitch = self._solve_rest_pitch(new)
                top = self._solve_top_pitch(self._with_leg_pitch(new, pitch))
                if w >= 0.99:
                    self._pitch_samples.append((pitch, top, np.nan if rise is None else rise))
                    if len(self._pitch_samples) >= REST_CAPTURE_OUTPUTS:
                        values = np.asarray(self._pitch_samples)
                        self.rest_pitch = (float(np.median(values[:, 0])), float(np.median(values[:, 1])))
                        self.rest_rise = None if np.isnan(values[:, 2]).any() else float(np.median(values[:, 2]))
            else:
                if self.rest_rise is not None and rise is not None:
                    self._rises.append(rise)
                    del self._rises[:-max(2, int(round(SNAP_WINDOW_S / max(dt, 1e-3))))]
                    above = rise - self.rest_rise
                    still = max(self._rises) - min(self._rises) <= SNAP_STILL_DEG
                    if self.lifted:
                        self.peak = max(self.peak, above)
                        if still and self.peak - above >= LANDED_FRACTION * (self.peak - self.drift):
                            self.lifted = False
                    elif above - self.drift > UP_DEG:
                        self.lifted, self.peak = True, above
                    landing = not self.lifted and still and (
                        abs(above - self.drift) < CREEP_DEG or self.peak > self.drift + UP_DEG)
                    if landing:
                        self.drift += float(np.clip(above, -MAX_DRIFT_DEG, MAX_DRIFT_DEG) - self.drift) * min(
                            1.0, dt / SNAP_TIME_CONSTANT_S)
                        if abs(above - self.drift) < 0.5:
                            self.peak = self.drift      # this landing is done
                    diag["lifted"] = self.lifted
                pitch, top = self.rest_pitch
                pitch += self.drift          # cancels the drift: hips back on the mat when landed
                diag["floorDriftDeg"] = round(self.drift, 2)
            new = self._with_leg_pitch(new, pitch)
            new = self._with_leg_pitch(new, top, (cfg["top_hip"],))
            diag["legPitchDeg"] = round(pitch, 1)
            diag["topLegPitchDeg"] = round(top, 1)
        if self.arm and cfg["arm_label"] in meas:
            forearm = meas[cfg["arm_label"]]
            _b, verts = self.forearm_region
            drop = float((verts @ forearm.T)[:, 1].min())
            length = float(np.linalg.norm(self.offsets[cfg["elbow"]]))
            reach = length * math.sin(math.radians(SUPPORT_UPPER_ARM_DEG))
            g0 = np.asarray(_globals(new))
            target = self._lowest(g0, self._positions(g0), self.floor_region) - drop + reach  # spine-independent
            def excess(pose):
                return self._positions(np.asarray(_globals(pose)))[cfg["sh"], 1] - target
            bend = 0.0
            start = excess(new)
            if start > 0:
                sign = 1.0 if excess(self._with_spine_bend(new, 5.0)) < start else -1.0
                root = _root(lambda d: excess(self._with_spine_bend(new, sign * d)), 0.0, MAX_SPINE_BEND_DEG)
                bend = sign * (MAX_SPINE_BEND_DEG if root is None else root)
                new = self._with_spine_bend(new, bend)
            diag["spineBendDeg"] = round(abs(bend), 1)
        if self.head:
            new[12] = np.eye(3)
            new[15] = np.eye(3)
        if self.top_arm and cfg["top_label"] in meas:
            # Upper arm along the side of the trunk toward the top hip (the hand rests on the thigh);
            # the forearm keeps its measurement.
            forearm = meas[cfg["top_label"]]
            gn = np.asarray(_globals(new))
            p = self._positions(gn)
            bone = self.offsets[cfg["top_elbow"]] / np.linalg.norm(self.offsets[cfg["top_elbow"]])
            toward = p[cfg["top_hip"]] + UP * TOP_ARM_ABOVE_HIP_M - p[cfg["top_sh"]]
            upper = _rotation_between(gn[cfg["top_sh"]] @ bone, toward) @ gn[cfg["top_sh"]]
            new[cfg["top_sh"]] = gn[cfg["top_collar"]].T @ upper
            new[cfg["top_elbow"]] = upper.T @ forearm
        if self.arm and cfg["arm_label"] in meas:
            gn = np.asarray(_globals(new))
            p = self._positions(gn)
            floor = self._lowest(gn, p, self.floor_region)
            forearm = meas[cfg["arm_label"]]
            _b, verts = self.forearm_region
            drop = float((verts @ forearm.T)[:, 1].min())      # forearm's lowest point below the elbow joint
            shoulder = p[cfg["sh"]]
            length = float(np.linalg.norm(self.offsets[cfg["elbow"]]))
            u_y = float(np.clip((floor - drop - shoulder[1]) / length, -1.0, 1.0))
            head = p[12] - p[0]
            head[1] = 0.0
            head /= max(np.linalg.norm(head), 1e-9)
            direction = head * math.sqrt(max(0.0, 1.0 - u_y * u_y)) + UP * u_y
            bone = self.offsets[cfg["elbow"]] / length
            upper = _rotation_between(gn[cfg["sh"]] @ bone, direction) @ gn[cfg["sh"]]
            new[cfg["sh"]] = gn[cfg["collar"]].T @ upper
            new[cfg["elbow"]] = upper.T @ forearm
            diag.update(floorGapCm=round(100 * (shoulder[1] - length - (floor - drop)), 1) if u_y <= -1.0 else 0.0,
                        upperArmElevationDeg=round(math.degrees(math.asin(u_y)), 1))
        if w < 1.0 - 1e-9:             # fade by the side-lying gate
            for j in np.flatnonzero(np.abs(new - out).reshape(24, -1).max(1) > 1e-9):
                out[j] = out[j] @ _scaled(out[j].T @ new[j], w)
        else:
            out = new
        return out, diag

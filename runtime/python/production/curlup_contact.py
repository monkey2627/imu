"""Curl-up contacts and temporal upper-arm stabilization.

The pelvis/buttocks remain on the mat while the shoulders are free to rise.  Planted feet use the
same geometry as the bridge solver, but bridge shoulder/arm floor contacts are deliberately disabled.
Forearm orientations are sensed; only the unobserved upper arms are rate-limited, with the elbow
counter-rotated so the sensed forearm global orientation is preserved.

Hands behind the head (the crunch start pose, kept through the curls): the network places the
unobserved upper arm so that, with the sensed forearm, the hands end 25-40 cm from the head (live
crunch 2026-11-19).  When the sensed forearm can reach the back of the head from the shoulder (the
elbow it implies is an upper-arm length from the shoulder) and points toward the body centre, the
upper arm is turned so the wrist lands behind the head; the forearm stays as sensed.

Crunch trunk: the network barely curls the unobserved upper trunk (live 2026-11-21: spine3
elevation -12..+11 deg while the lumbar waist sensor rose 8-14 deg per curl and the head, judged from
forearms held rigidly behind it, turned 1.5-2.1x the waist), and its pelvis rocks -13..+12 deg, which
cancels the visible lift; its neck/head angles look arbitrary.  The elbow position varies per curl
(open: forearm turns with the head; pulled forward: it turns the other way), so the forearms cannot
measure the curl.  Lying on the back the trunk is therefore rebuilt from the waist sensor: pelvis
level on the mat, spine1 following the sensor's rise above its resting level, spine2/spine3 curling
to CURL_GAIN x that rise, neck and head tucking a little with it.  Thighs, forearms and upper arms
keep their global orientation.
"""
from __future__ import annotations

import math

import numpy as np

from production.arm_prior import PARENTS, _globals, _rotation_between, _scaled
from production.bridge_contact import BridgeContact, GATE_SUPINE, PELVIS_BACK_DEPTH, SPINE3_BACK_DEPTH, _ramp

CURLUP_MODELS = frozenset({"waist_imudino_stream_curlup"})
ARM_TIME_CONSTANT_S = 0.10
ARM_MAX_SPEED_DEG_S = 240.0
# Wrist behind the head, in the head joint frame (SMPL: +x body-left, +y up the head, +z face):
# fingers interlaced on the back of the head.
HAND_BEHIND_HEAD = {"left": np.array([0.10, 0.02, -0.09]), "right": np.array([-0.10, 0.02, -0.09])}
GATE_INWARD = (0.15, 0.45)        # sensed forearm toward the body centre (chest frame): off .. full
# ... or toward the head, or toward the back: elbows pulled forward during a curl point the forearm
# from the raised elbow down to the back of the head (live 190303: chest-frame direction
# (+-0.1, 0.3, -0.94), reach residual 0-3 cm), with little inward component.
GATE_HEADWARD = (0.35, 0.65)
GATE_BACKWARD = (0.55, 0.80)
ELBOW_SWIVEL_STEPS = 48
# The forearm's heading is the least reliable sensed quantity (live 205459: left and right forearms of a
# symmetric crunch 40 deg apart in heading), so the elbow is chosen by the forearm's gravity elevation
# (elbows raised forward <-> open) and a symmetric preference for the elbow out to the side.
ELBOW_ELEVATION_COST = 4.0        # per (sine of elevation error)^2
ELBOW_OUTWARD_COST = 0.3          # per unit of (1 - outward component of the upper arm)
# Sensed forearm elevation vs the solve's: full weight below 25 deg, none above 50.
FOREARM_DEVIATION_DEG = (25.0, 50.0)
# Upper-trunk curl.  Head/waist rotation ratio on open-elbow curls: 1.5-2.1 (232340, 234704, 190303).
CURL_GAIN = 1.8
CURL_MAX_DEG = 50.0               # extra flexion added at most
CURL_REST_RISE_S = 4.0            # resting levels follow a drop at once, a rise with this time constant
CURL_SPINE = (3, 6, 9)
NECK_TUCK, HEAD_TUCK = 0.20, 0.10     # neck / head flexion as a fraction of the upper-trunk curl
# The waist sensor is not a reliable curl signal: worn on the lumbar spine it tilts up 8-14 deg per curl
# (190303), worn low on the pelvis it tilts DOWN 2-7 deg (posterior tilt, live 213354), so its "rise"
# showed no curl at all.  The forearms' gravity elevation is: hands behind the head, it rose 35-50 deg
# above the calibrated start hold at the top of every curl (213354: hold -31, tops +12..+21, lying -50).
# Curl = max(waist-rise curl, forearm curl); the forearm baseline is the hold right after a reset/rebase.
FOREARM_CURL_DEG = (10.0, 45.0)   # forearm elevation above the start hold: curl starts .. full
FOREARM_CURL_MAX_DEG = 35.0       # upper-trunk curl at full
FOREARM_BASELINE_OUTPUTS = 15     # outputs after a reset averaged as the start-hold baseline
# Upper back on the mat: with the buttocks on the floor, the back surface behind spine2/spine3 and the
# back of the head must not go below it.  Lowering from a curl the network put the shoulder blades 6-10 cm
# under the floor (live 190303); a lowest-point grounding (Unity "人物贴地", the comparison videos) then
# lifted the whole body, so the buttocks and feet bounced up.  The trunk is curled just enough instead.
BACK_DEPTHS = ((6, 0.13), (9, SPINE3_BACK_DEPTH), (15, 0.10))     # joint, depth behind it (m)
# The hands lie between the head and the mat: with the head down, the behind-head target is raised to keep
# the wrist this far above the floor (live 190303: hand/wrist/elbow below the floor lifted the body).
WRIST_MIN_HEIGHT_M = 0.05
HAND_LENGTH_M = 0.10              # a forearm pointing down carries the hand this much further down
ELBOW_MIN_HEIGHT_M = 0.05


class CurlUpContact:
    def __init__(self) -> None:
        # Feet planted with the shank at -58..-65 deg (live 2026-11-21) needs a thigh up to ~78 deg on the SMPL
        # leg lengths, often just past the 0.98 sine limit: the solve then switched off and the network's
        # post-rebase thigh 37-57 deg BELOW level (knee in the floor) was shown.  A clamped, near-vertical
        # thigh is far closer; only a clearly impossible solve (|sine| > 1.25) keeps the network thigh.
        self.contacts = BridgeContact(pelvis_only=True, floor_arms=False, max_leg_sine=1.25)
        self.reset("init")

    def reset(self, reason: str) -> None:
        self.contacts.reset(reason)
        self.previous_upper = {"left": None, "right": None}
        self.rest = {"pelvis": None}
        self.forearm_baseline = []
        self.reason = reason

    def _track_rest(self, key, value, dt):
        rest = self.rest[key]
        if rest is None or value < rest:
            rest = value
        else:
            rest += (value - rest) * (1.0 - math.exp(-dt / CURL_REST_RISE_S))
        self.rest[key] = rest
        return rest

    def _forearm_curl(self, forearms):
        """Upper-trunk curl (deg) from the mean sensed forearm elevation above the start-hold baseline."""
        if forearms is None:
            return 0.0
        elevation = float(np.mean([math.degrees(math.asin(float(np.clip((np.asarray(f) @ axis)[1], -1.0, 1.0))))
                                   for f, axis in zip(forearms, (np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])))]))
        if len(self.forearm_baseline) < FOREARM_BASELINE_OUTPUTS:
            self.forearm_baseline.append(elevation)
            return 0.0
        above = elevation - float(np.mean(self.forearm_baseline))
        return FOREARM_CURL_MAX_DEG * _ramp(above, *FOREARM_CURL_DEG)

    def _crunch_trunk(self, local, dt, supine, waist=None, forearms=None):
        """Rebuild pelvis, spine, neck and head lying on the back (weight `supine`); see the module
        docstring.  Returns the upper-trunk curl (deg, spine3 elevation above level)."""
        g = _globals(local)
        lumbar = g[0] if waist is None else np.asarray(waist, dtype=np.float64)
        sensor = math.degrees(math.asin(float(np.clip(lumbar[1, 1], -1.0, 1.0))))
        rise = max(0.0, sensor - self._track_rest("pelvis", sensor, dt))
        forearm_curl = self._forearm_curl(forearms)
        if supine <= 0.0:
            return 0.0
        spine3 = float(np.clip(max(CURL_GAIN * rise, forearm_curl), 0.0, CURL_MAX_DEG))
        spine1 = min(rise, spine3)
        elevations = {3: spine1, 6: 0.5 * (spine1 + spine3), 9: spine3}
        # Pelvis flat on the mat: spine axis level and no roll, only the heading kept.  The waist sensor read
        # a steady 20-23 deg roll lying on the back (live 205459, a strap angle on the curved lower back),
        # which put one shoulder 12 cm below the other and made the two arms solve differently.
        head_dir = g[0][:, 1]
        level = np.array([head_dir[0], 0.0, head_dir[2]])
        if np.linalg.norm(level) > 1e-6:
            level /= np.linalg.norm(level)
            chest = np.array([0.0, 1.0, 0.0]) * (1.0 if g[0][1, 2] >= 0.0 else -1.0)
            root = np.stack([np.cross(level, chest), level, chest], axis=1)
        else:
            root = g[0]
        left = root[:, 0]
        keep = {j: g[j].copy() for j in (1, 2, 16, 17, 18, 19)}   # thighs, upper arms, forearms (global)
        target = {0: root}
        for j, degrees in elevations.items():
            target[j] = _axis_angle(left, math.radians(degrees)) @ root
        target[12] = target[9] @ _flex_x(NECK_TUCK * spine3)
        target[15] = target[12] @ _flex_x(HEAD_TUCK * spine3)
        new_g = {}
        for j in (0, 3, 6, 9, 12, 15):
            parent = new_g.get(PARENTS[j]) if PARENTS[j] >= 0 else None
            current = g[j] if parent is None else new_g_full(g, new_g, j)
            blended = _scaled(target[j] @ current.T, supine) @ current
            new_g[j] = blended
            local[j] = blended if PARENTS[j] < 0 else new_g[PARENTS[j]].T @ blended
        g2 = _globals(local)
        for j, wanted in keep.items():
            local[j] = g2[PARENTS[j]].T @ wanted
            g2 = _globals(local)
        # The back of the upper trunk and of the head must not go below the mat (lowest-point grounding
        # would lift the buttocks and feet): curl a little more if needed.
        if self._lowest_back(local, 0.0) < 0.0:
            lo, hi = 0.0, CURL_MAX_DEG
            if self._lowest_back(local, hi) < 0.0:
                lo = hi
            else:
                for _ in range(12):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if self._lowest_back(local, mid) < 0.0 else (lo, mid)
                lo = hi
            forearms = {j: g2[j].copy() for j in (18, 19)}
            self._flex(local, supine * lo)
            g2 = _globals(local)
            for j, forearm in forearms.items():
                local[j] = g2[PARENTS[j]].T @ forearm
            spine3 += supine * lo
        return spine3

    @staticmethod
    def _flex(local, degrees):
        """Add `degrees` of flexion spread over spine1-3 (about +x, body-left: head toward the chest)."""
        step = np.radians(degrees / len(CURL_SPINE))
        c, s = math.cos(step), math.sin(step)
        flex = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        for j in CURL_SPINE:
            local[j] = local[j] @ flex

    def _lowest_back(self, local, degrees):  # noqa: D401
        """Height (m) of the lowest upper-back / back-of-head point above the floor, the buttocks resting
        on it, after adding `degrees` of flexion."""
        trial = local.copy()
        if degrees > 0.0:
            self._flex(trial, degrees)
        g = _globals(trial)
        p = self._positions(g)
        floor_to_pelvis = float(g[0][1, 2]) * PELVIS_BACK_DEPTH
        return min(float(p[j][1] - g[j][1, 2] * depth) for j, depth in BACK_DEPTHS) + floor_to_pelvis

    def apply(self, local: np.ndarray, dt: float, waist: np.ndarray | None = None, forearms=None):
        """`waist`, `forearms` (lw, rw): the corrected sensor orientations of this output's frame (SMPL axes),
        if known."""
        local = np.asarray(local, dtype=np.float64).copy()
        supine = _ramp(float(local[0][1, 2]), *GATE_SUPINE)
        curl = self._crunch_trunk(local, dt, supine, waist, forearms)
        local, contact = self.contacts.apply(local, dt)
        contact["trunkCurlDeg"] = round(curl, 2)
        g = _globals(local)
        positions = self._positions(g)
        offsets = self.contacts.offsets
        arm_diag = {}
        for side, collar, shoulder, elbow in (("left", 13, 16, 18), ("right", 14, 17, 19)):
            current = g[shoulder]
            behind_head = 0.0
            wrist_target = None
            if supine > 0.0:
                current, wrist_target, behind_head = self._hand_behind_head(side, g, positions, shoulder, elbow,
                                                                            current, offsets, supine)
            previous = self.previous_upper[side]
            raw_delta = 0.0
            applied_delta = 0.0
            corrected = current
            if previous is not None and supine > 0.0:
                delta = current @ previous.T
                raw_delta = _angle_deg(delta)
                alpha = 1.0 - math.exp(-dt / ARM_TIME_CONSTANT_S)
                if raw_delta > 1e-6:
                    max_weight = math.radians(ARM_MAX_SPEED_DEG_S) * dt / math.radians(raw_delta)
                    alpha = min(alpha, max_weight)
                smoothed = _scaled(delta, float(np.clip(alpha, 0.0, 1.0))) @ previous
                corrected = _scaled(smoothed @ current.T, supine) @ current
                applied_delta = _angle_deg(corrected @ previous.T)
                local[shoulder] = g[collar].T @ corrected
                local[elbow] = corrected.T @ self._forearm_to(g, positions, collar, shoulder, elbow, corrected,
                                                              wrist_target, behind_head)
            elif behind_head > 0.0:
                local[shoulder] = g[collar].T @ corrected
                local[elbow] = corrected.T @ self._forearm_to(g, positions, collar, shoulder, elbow, corrected,
                                                              wrist_target, behind_head)
            if behind_head > 0.0:
                # Hands clasped behind the head keep the wrist about straight; the network's 50+ deg wrist
                # flexion put the fingers through the mat (live 190303).
                for joint in (elbow + 2, elbow + 4):
                    local[joint] = local[joint] @ _scaled(local[joint].T, behind_head)
            self.previous_upper[side] = corrected.copy()
            arm_diag[side] = {"rawDeltaDeg": round(raw_delta, 2),
                              "appliedDeltaDeg": round(applied_delta, 2),
                              "handBehindHead": round(behind_head, 3)}
        if supine > 0.0:
            self._arms_above_mat(local, supine, arm_diag)
        contact["arms"] = arm_diag
        contact["upperArmSmoothing"] = round(supine, 3)
        return local, contact


    def _positions(self, g):
        p = np.zeros((24, 3))
        for j in range(1, 24):
            p[j] = p[PARENTS[j]] + g[PARENTS[j]] @ self.contacts.offsets[j]
        return p

    def _forearm_to(self, g, p, collar, shoulder, elbow, upper, wrist_target, weight):
        """Forearm global: the sensed one, turned (minimal rotation, by `weight`) so that from the elbow
        the upper arm `upper` gives, it points at the behind-head wrist target."""
        if wrist_target is None or weight <= 0.0:
            return g[elbow]
        offsets = self.contacts.offsets
        shoulder_pos = p[collar] + g[collar] @ offsets[shoulder]
        elbow_pos = shoulder_pos + upper @ offsets[elbow]
        turn = _rotation_between(g[elbow] @ offsets[elbow + 2], wrist_target - elbow_pos)
        return _scaled(turn, weight) @ g[elbow]

    def _arms_above_mat(self, local, supine, arm_diag):
        """Last pass lying on the back: lift an upper arm whose elbow or wrist (plus the hand, for a forearm
        pointing down) is below the mat, the forearm keeping its global orientation.  The behind-head solve
        and the upper-arm smoothing can leave them there (live 232340: elbows/wrists below the floor in
        14% of the curl frames, lifting the whole body under lowest-point grounding)."""
        offsets = self.contacts.offsets
        length = {e: float(np.linalg.norm(offsets[e])) for e in (18, 19)}
        for side, collar, shoulder, elbow in (("left", 13, 16, 18), ("right", 14, 17, 19)):
            g = _globals(local)
            p = self._positions(g)
            floor = -float(g[0][1, 2]) * PELVIS_BACK_DEPTH
            forearm = g[elbow] @ offsets[elbow + 2]
            down = max(0.0, -float(forearm[1]) / float(np.linalg.norm(forearm)))
            upper_dir = g[shoulder] @ offsets[elbow] / length[elbow]
            needed = max(floor + ELBOW_MIN_HEIGHT_M - p[shoulder][1],
                         floor + WRIST_MIN_HEIGHT_M + HAND_LENGTH_M * down - forearm[1] - p[shoulder][1]) / length[elbow]
            if upper_dir[1] >= needed:
                continue
            vertical = min(float(needed), 1.0)
            horizontal = np.array([upper_dir[0], 0.0, upper_dir[2]])
            if np.linalg.norm(horizontal) < 1e-6:
                horizontal = g[0][:, 0] * (1.0 if side == "left" else -1.0)
                horizontal[1] = 0.0
            horizontal /= max(float(np.linalg.norm(horizontal)), 1e-9)
            wanted = horizontal * math.sqrt(max(0.0, 1.0 - vertical * vertical)) + np.array([0.0, vertical, 0.0])
            upper = _scaled(_rotation_between(upper_dir, wanted), supine) @ g[shoulder]
            local[shoulder] = g[collar].T @ upper
            local[elbow] = upper.T @ g[elbow]
            self.previous_upper[side] = upper.copy()
            arm_diag.setdefault(side, {})["liftedAboveMatDeg"] = round(_angle_deg(upper @ g[shoulder].T), 1)

    @staticmethod
    def _hand_behind_head(side, g, p, shoulder, elbow, upper, offsets, supine):
        """Two-bone solve with the wrist behind the head: the elbow lies on the circle the SMPL upper-arm and
        forearm lengths allow; of that circle the point whose forearm direction is closest to the sensed
        forearm is used (elbows open -> out to the side, pulled forward -> up), kept above the mat.
        Placing the sensed forearm unchanged left the elbow only 0.15-0.20 m from the shoulder, so the
        upper arm direction was arbitrary: it crossed the chest or pointed at the ceiling (live 205459).
        Returns (upper-arm global, wrist target, weight); weight 0 when the sensed forearm is far from
        any behind-head pose."""
        forearm = g[elbow] @ offsets[elbow + 2]                        # elbow -> wrist, as sensed
        sensed = forearm / float(np.linalg.norm(forearm))
        floor = -float(g[0][1, 2]) * PELVIS_BACK_DEPTH           # positions have the pelvis at the origin
        wrist_min = floor + WRIST_MIN_HEIGHT_M + HAND_LENGTH_M * max(0.0, -float(sensed[1]))
        wrist = p[15] + g[15] @ HAND_BEHIND_HEAD[side]
        wrist[1] = max(float(wrist[1]), wrist_min)
        local_dir = g[9].T @ sensed                                    # chest frame: +x left, +y head
        inward = float(local_dir[0]) * (-1.0 if side == "left" else 1.0)
        toward_head = max(_ramp(inward, *GATE_INWARD), _ramp(float(local_dir[1]), *GATE_HEADWARD),
                          _ramp(-float(local_dir[2]), *GATE_BACKWARD))
        upper_len = float(np.linalg.norm(offsets[elbow]))
        fore_len = float(np.linalg.norm(offsets[elbow + 2]))
        span = wrist - p[shoulder]
        d = float(np.clip(np.linalg.norm(span), abs(upper_len - fore_len) + 1e-3, upper_len + fore_len - 1e-3))
        axis = span / max(float(np.linalg.norm(span)), 1e-9)
        along = (upper_len ** 2 - fore_len ** 2 + d * d) / (2.0 * d)
        radius = math.sqrt(max(0.0, upper_len ** 2 - along ** 2))
        v1 = np.cross(axis, [0.0, 1.0, 0.0])
        if np.linalg.norm(v1) < 1e-6:
            v1 = np.cross(axis, [1.0, 0.0, 0.0])
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(axis, v1)
        centre = p[shoulder] + along * axis
        outward = g[9][:, 0] * (1.0 if side == "left" else -1.0)
        best = None
        for k in range(ELBOW_SWIVEL_STEPS):
            angle = 2.0 * math.pi * k / ELBOW_SWIVEL_STEPS
            e = centre + radius * (math.cos(angle) * v1 + math.sin(angle) * v2)
            fdir = wrist - e
            fdir /= max(float(np.linalg.norm(fdir)), 1e-9)
            cost = (ELBOW_ELEVATION_COST * (float(fdir[1]) - float(sensed[1])) ** 2
                    + ELBOW_OUTWARD_COST * (1.0 - float((e - p[shoulder]) @ outward) / upper_len)
                    + 10.0 * max(0.0, floor + ELBOW_MIN_HEIGHT_M - float(e[1])))
            if best is None or cost < best[0]:
                best = (cost, e, fdir)
        _cost, elbow_pos, fdir = best
        deviation = abs(math.degrees(math.asin(float(np.clip(fdir[1], -1.0, 1.0)))
                                     - math.asin(float(np.clip(sensed[1], -1.0, 1.0)))))
        weight = supine * toward_head * _ramp(-deviation, -FOREARM_DEVIATION_DEG[1], -FOREARM_DEVIATION_DEG[0])
        if weight <= 0.0:
            return upper, wrist, 0.0
        turn = _rotation_between(upper @ offsets[elbow], elbow_pos - p[shoulder])
        return _scaled(turn, weight) @ upper, wrist, weight

def new_g_full(g, new_g, j):
    """Global of joint j with its (already rebuilt) parent and its original local rotation."""
    parent = PARENTS[j]
    return new_g[parent] @ (g[parent].T @ g[j])


def _axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64) / max(float(np.linalg.norm(axis)), 1e-12)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * k @ k


def _flex_x(degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _angle_deg(rotation: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))

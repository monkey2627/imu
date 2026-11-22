"""Bird-dog support-knee rule.  Production: frame_chain.postprocess_output applies it only for the
action ``birddog`` (BIRDDOG_ACTIONS); every other action's output is untouched.  The per-frame
acceleration observer (``observe``, called by frame_chain.prepare_frame) only updates this rule's own
state.

Correct exercise (user, 2026-11-23): four-point kneeling start (hands and knees on the mat), the
contralateral arm and leg extend together to horizontal, then return to the four-point stance.  In
the start hold both knees and both hands are on the mat.

What goes wrong without it (real recording birddog_20261123_005257, user held the start pose for
15 s): the left thigh swings from -61 deg (vertical-ish) to -15 deg and the knee opens from ~90 to
~18 deg -- an extended leg -- from t=5.5 s on, survives the start-pose rebase at 8 s and stays until
the first repetition; the same drift recurs between repetitions and after returns.  The shank sensor
stays level all the time.  A level shank is ambiguous: lying on the mat (kneeling) and stretched back
horizontally give the same shank orientation; only the thigh differs and the thigh has no sensor.
The start-pose fit only corrects the sensors and StartPoseConstraint has no bird-dog weights.

Cues, checked on birddog_20261123_021217 (4 contralateral reps, 4 arm-only, 4 leg-only) and 005257:
* the opposite forearm is NOT usable (v1, analysis/scratch_birddog/v1_*): arm-only reaches released
  the leg and the network stretched it; leg-only lifts were pressed down;
* every lift starts with a shank-rise transient (16-31 deg, >= 12.9 deg even for a leg held exactly
  horizontal), arm-only reaches and holds stay within +-6 deg -- but a horizontal leg then reads level
  again, like a kneeling one (v2, shank angle only, pressed it down);
* the network's thigh is no landing cue (v3): after a return it often stays drifted for seconds;
* the shank's vertical displacement (SlimeVR linear acceleration with the accel_frame.py fix,
  integrated over one motion segment between two still periods, zero velocity at both ends) is
  clearly positive for a lift (+50..+80 cm; over-estimated ~2x but the sign and size separate) and
  negative for a return; arm-only reaches give no shank motion segment at all.
Per-leg state machine:
* down -> up: the shank rises UP_DEG above its floor baseline (a lift has started);
* up -> down: the shank is back near its baseline (rise < LAND_LEVEL_DEG) and either the lift had
  clearly raised the shank (peak rise >= HIGH_DEG: all raised-leg reps, 26-43 deg; lands at once) or,
  for a low lift, at the end of a motion segment the accumulated shank height is below LANDED_M (back
  on the mat); a level, still shank with the height kept is a horizontal leg and stays up.  Fallback: back for UP_TIMEOUT_S.  On
  landing the baseline snaps to the current reading (sensors keep ~+5 deg after a lift).
* up: released (the network's leg); down: kneeling, released continuously by the rise itself:
  support(leg) = gate * shank_not_hanging * (1 - ramp(shank rise above its floor baseline, LIFT_RAMP))
The network's own drift into an extended leg has no shank-rise event, so it stays down and is held.
* floor baseline: the shank elevation in the first CAPTURE_OUTPUTS gated outputs after a reset (run.py
  resets at the calibration start and at every start-pose rebase, i.e. in the four-point hold); while
  down and level it follows slow sensor drift (BASELINE_TIME_CONSTANT_S).  Before it is captured the
  absolute elevation is used (FALLBACK_RAMP).  Height: 0 at every still, level moment while down.
* gate: four-point posture -- the waist sensor's spine axis near horizontal (TRUNK_GATE_RAMP);
  shank_not_hanging: off when the shank points steeply down (standing / sitting).
The weight is rate limited (WEIGHT_RATE_PER_S) and seeded from its target on the first output
after a reset.  A supported thigh is swung toward straight down (knee under the hip, the quadruped
template) by its weight; the shank and everything below keep their global orientation (= sensor).
A released leg is the network's.

Contacts (user, 2026-11-23, after the compare video: "the whole body floats" -- with the legs fixed,
the knees were 10-28 cm above the floor; only toes and palms touched it):
* support foot: the network points the foot down and the toes prop the knee up; the ankle is laid
  back along the shank (dorsum on the mat, as lyingcat_contact) by the leg weight;
* support arms (measured forearm pointing down, ARM_SUPPORT_RAMP, rate limited): the upper arm
  continues the measured forearm (straight elbow, shoulder over the hand -- the quadruped template);
  the hand lies flat, fingers toward the head (the network's fingers pointed down and propped the
  trunk 7-19 cm up, as lyingcat_contact);
* an arm supports only while its forearm stays at its planted elevation (released 5-12 deg above
  it, ARM_RISE_RAMP; re-planted when it has come down again).  v7 used only the absolute -55 deg:
  the solve kept a lifting hand on the mat, the trunk pitched ~10 cm after it and snapped back ->
  the 2 cm jumps the user saw at every reach;
* knees level: pelvis roll about the head axis (spine1, thighs, shanks keep their globals); palms
  level: spine3 roll (MAX_ROLL_DEG).  With the pitch below all four supports share one plane, so the
  renderer's lowest-vertex grounding does not drop the body when one of them lifts;
* hands and knees on one plane: the chest has no sensor, so spine2/spine3 pitch (half each, about the
  horizontal axis across the body, at most MAX_SPINE_DEG, smoothed by SPINE_TIME_CONSTANT_S) is
  solved so the supporting palms' contact height equals the supporting knees'.  The measured forearms
  keep their globals.  A reaching arm / lifted leg takes no part (weight 0).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from production.arm_prior import PARENTS, _globals, _rotation_between, _scaled
from production.bridge_contact import _ramp, _rest_offsets, _unit
from production.lyingcat_contact import _twist_angle

LIFT_RAMP = (6.0, 14.0)              # shank rise above the floor baseline: >=14 deg -> leg released
FALLBACK_RAMP = (25.0, 40.0)         # absolute shank elevation, before the baseline is captured
CAPTURE_OUTPUTS = 15                 # ~0.5 s of gated outputs after a reset
MAX_BASELINE_DEG = 25.0              # only a near-level shank can be a floor baseline
CREEP_DEG = 4.0                      # below this rise the shank counts as level
BASELINE_TIME_CONSTANT_S = 5.0
SHANK_HANG_RAMP = (-50.0, -35.0)     # shank pointing down steeper than -50 deg: not kneeling
TRUNK_GATE_RAMP = (35.0, 50.0)       # |spine1 axis elevation|: four-point below 35 deg, off above 50
UP_DEG = 9.0                         # shank rise that starts a lift (state up)
LAND_LEVEL_DEG = 8.0                 # landing: shank back within this of the baseline (sensors keep
                                     # ~+5 deg after a lift, 021217); the baseline then snaps to it
HIGH_DEG = 18.0                      # a lift that peaked this high lands as soon as the shank is back
LANDED_M = 0.20                      # accumulated shank height below this when level and still: landed
UP_TIMEOUT_S = 8.0                   # up with a level shank for this long: assume landed
STILL_ACC = 0.35                     # m/s^2, mean |linear acceleration| over STILL_FRAMES
STILL_FRAMES = 12
MIN_SEGMENT_FRAMES = 15
MAX_SEGMENT_FRAMES = 10 * 60         # longer motion: displacement unreliable, height unknown
FRAME_S = 1.0 / 60.0
WEIGHT_RATE_PER_S = 4.0
ARM_SUPPORT_RAMP = (40.0, 55.0)      # forearm -elevation: <=40 deg reaching, >=55 deg on the mat
ARM_PLANT_DEG = 60.0                 # forearm at least this steep: planted, (re)capture its baseline
ARM_RISE_RAMP = (5.0, 12.0)          # forearm rise above its planted elevation: >=12 deg reaching
ARM_CREEP_DEG = 3.0
ARM_BASELINE_TIME_CONSTANT_S = 2.0
ARM_WEIGHT_RATE_PER_S = 8.0
ARM_REPLANT_S = 1.0                  # lifted arm steep again this long: planted (fallback)
MAX_ROLL_DEG = 15.0
MAX_SPINE_DEG = 40.0
SPINE_TIME_CONSTANT_S = 0.05
HAND_CONTACT_M = 0.03                # palm surface below the wrist joint (hand flat on the mat)
KNEE_CONTACT_M = 0.045               # front of the knee below the knee joint (mesh, 021217: 4.1-4.5 cm)
WRIST_TWIST_MAX_DEG = 12.0           # as lyingcat_contact
MAX_FLATTEN_DEG = 150.0
DOWN = np.array([0.0, -1.0, 0.0])
UP = np.array([0.0, 1.0, 0.0])
LEGS = (("left", 1, 4, 7), ("right", 2, 5, 8))     # side, thigh, shank, ankle
FEET = {"left": 10, "right": 11}
ARMS = (("left", 16, 18, 20), ("right", 17, 19, 21))   # side, upper arm, forearm, wrist
HANDS = {"left": 22, "right": 23}
SHANK_LABELS = {"left": "ls", "right": "rs"}


def _elevation(direction):
    return math.degrees(math.asin(float(np.clip(_unit(direction)[1], -1.0, 1.0))))


def _bisect(f, limit):
    """Root of f in [-limit, limit]; the end with the smaller |f| when there is no sign change."""
    lo, hi = -limit, limit
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0.0:
        return lo if abs(f_lo) < abs(f_hi) else hi
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if f_mid * f_lo <= 0.0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def _basis(primary, secondary):
    primary = _unit(primary)
    secondary = _unit(secondary - float(np.dot(secondary, primary)) * primary)
    return np.column_stack((primary, secondary, np.cross(primary, secondary)))


def _axis_rotation(axis, degrees):
    axis = _unit(np.asarray(axis, dtype=np.float64))
    a = math.radians(degrees)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k


def _segment_rise(vertical) -> float:
    """Vertical displacement over one motion segment, zero velocity at both ends."""
    a = np.asarray(vertical, dtype=np.float64)
    v = np.cumsum(a) * FRAME_S
    v -= np.linspace(0.0, 1.0, len(v)) * v[-1]
    return float(np.sum(v) * FRAME_S)


class BirdDogContact:
    def __init__(self) -> None:
        self.offsets = _rest_offsets()
        self.reset("init")

    def reset(self, reason: str) -> None:
        self.weights = {"left": 0.0, "right": 0.0}
        self.arm_weights = {"left": 0.0, "right": 0.0}
        self.spine_deg = 0.0
        self.roll_deg = 0.0
        self.arm_base = {"left": None, "right": None}
        self.arm_up = {"left": False, "right": False}
        self.arm_peak = {"left": 0.0, "right": 0.0}
        self.arm_steep_s = {"left": 0.0, "right": 0.0}
        self.baseline = {"left": None, "right": None}
        self.capture = {"left": [], "right": []}
        self.up = {"left": False, "right": False}
        self.level_s = {"left": 0.0, "right": 0.0}
        self.peak = {"left": 0.0, "right": 0.0}
        self.height = {"left": 0.0, "right": 0.0}
        self.events = {"left": [], "right": []}       # finished motion segments since the last output
        self.magnitudes = {side: deque(maxlen=STILL_FRAMES) for side in SHANK_LABELS}
        self.segment = {"left": [], "right": []}
        self.seeded = False
        self.reason = reason

    # ------------------------------------------------------------- 60 Hz acceleration
    def observe(self, labels, accelerations) -> None:
        """One input frame: (n,3) server-frame linear acceleration (y up) in label order."""
        for side, label in SHANK_LABELS.items():
            if label not in labels:
                continue
            a = np.asarray(accelerations[labels.index(label)], dtype=np.float64)
            if not np.all(np.isfinite(a)):
                continue
            self.magnitudes[side].append(float(np.linalg.norm(a)))
            still = (len(self.magnitudes[side]) == STILL_FRAMES
                     and float(np.mean(self.magnitudes[side])) < STILL_ACC)
            if not still:
                self.segment[side].append(float(a[1]))
            elif self.segment[side]:
                n = len(self.segment[side])
                if n > MAX_SEGMENT_FRAMES:
                    self.events[side].append(None)
                elif n >= MIN_SEGMENT_FRAMES:
                    self.events[side].append(_segment_rise(self.segment[side]))
                else:
                    self.events[side].append(0.0)
                self.segment[side] = []

    # ------------------------------------------------------------- per output
    def _target(self, side, shank, gate, dt):
        hang = _ramp(shank, *SHANK_HANG_RAMP)
        events, self.events[side] = self.events[side], []
        settled = bool(events) and not self.segment[side]     # a motion just ended, shank still now
        for rise_m in events:
            self.height[side] = None if rise_m is None or self.height[side] is None \
                else max(0.0, self.height[side] + rise_m)
        if self.baseline[side] is None:
            if gate > 0.9 and abs(shank) <= MAX_BASELINE_DEG:
                self.capture[side].append(shank)
                if len(self.capture[side]) >= CAPTURE_OUTPUTS:
                    self.baseline[side] = float(np.mean(self.capture[side]))
            return gate * hang * (1.0 - _ramp(shank, *FALLBACK_RAMP)), None
        rise = shank - self.baseline[side]
        level = rise < CREEP_DEG
        if self.up[side]:
            back = rise < LAND_LEVEL_DEG
            self.peak[side] = max(self.peak[side], rise)
            self.level_s[side] = self.level_s[side] + dt if back else 0.0
            landed = back and (self.peak[side] >= HIGH_DEG or (
                settled and self.height[side] is not None and self.height[side] < LANDED_M))
            if landed or self.level_s[side] >= UP_TIMEOUT_S:
                self.up[side] = False
                self.baseline[side] = shank                   # landing snap: the new floor reading
                rise, level = 0.0, True
        elif rise >= UP_DEG and gate > 0.9:
            self.up[side], self.level_s[side], self.peak[side] = True, 0.0, rise
        if self.up[side]:
            return 0.0, rise
        if level and not self.segment[side]:
            self.height[side] = 0.0                            # down, level and still: on the mat
        if level and gate > 0.9:
            self.baseline[side] += (1.0 - math.exp(-dt / BASELINE_TIME_CONSTANT_S)) * rise
        return gate * hang * (1.0 - _ramp(rise, *LIFT_RAMP)), rise

    # ------------------------------------------------------------- contacts
    def _positions(self, g):
        p = np.zeros((24, 3), dtype=np.float64)
        for joint in range(1, 24):
            p[joint] = p[PARENTS[joint]] + g[PARENTS[joint]] @ self.offsets[joint]
        return p

    def _lay_feet(self, local):
        """Support legs: foot laid back along the shank, dorsum on the mat (as lyingcat_contact)."""
        for side, _thigh, shank, ankle in LEGS:
            weight = self.weights[side]
            if weight <= 0.0:
                continue
            g = _globals(local)
            back = g[shank] @ self.offsets[ankle]
            back[1] = 0.0
            if np.linalg.norm(back) < 0.05:
                continue
            foot = FEET[side]
            vector = g[ankle] @ self.offsets[foot]
            new_ankle = _scaled(_rotation_between(vector, _unit(back)), weight) @ g[ankle]
            local[ankle] = g[shank].T @ new_ankle
            local[foot] = local[foot] @ _scaled(local[foot].T, weight)

    def _straighten_arms(self, local):
        """Support arms: upper arm along the measured forearm; the forearm keeps its global."""
        for side, upper, forearm, wrist in ARMS:
            weight = self.arm_weights[side]
            if weight <= 0.0:
                continue
            g = _globals(local)
            keep = g[forearm].copy()
            swing = _scaled(_rotation_between(g[upper] @ self.offsets[forearm], keep @ self.offsets[wrist]),
                            weight)
            new_upper = swing @ g[upper]
            local[upper] = g[PARENTS[upper]].T @ new_upper
            local[forearm] = new_upper.T @ keep

    def _flatten_hands(self, local):
        """Support hands flat on the mat, fingers toward the head (as lyingcat_contact: in SMPL's zero
        pose the wrist->hand axis follows the fingers and local -Y faces the palm); the wrist roll about
        the forearm is capped so the forearm skin is not wrung."""
        g = _globals(local)
        p = self._positions(g)
        head = p[12] - p[0]
        head[1] = 0.0
        if np.linalg.norm(head) < 0.1:
            return
        target = _basis(_unit(head), DOWN)
        for side, _upper, forearm, wrist in ARMS:
            weight = self.arm_weights[side]
            if weight <= 0.0:
                continue
            hand = HANDS[side]
            source = _basis(_unit(self.offsets[hand]), DOWN)
            wanted_local = g[forearm].T @ (target @ source.T)
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

    def _cross_axis(self, p):
        head = p[12] - p[0]
        head[1] = 0.0
        if np.linalg.norm(head) < 0.1:
            return None
        return _unit(np.cross(UP, _unit(head)))

    def _arm_planted(self, side, elevation, dt):
        """1 while the forearm stays at its planted elevation; released as soon as it starts to rise
        (an absolute threshold kept a lifting hand 'on the mat' until -55 deg: the trunk pitched ~10 cm
        after it and snapped back -- the jumps the user saw)."""
        base = self.arm_base[side]
        if self.arm_up[side]:
            # back on the mat only once the forearm has come down again from its highest point
            self.arm_peak[side] = max(self.arm_peak[side], elevation)
            self.arm_steep_s[side] = self.arm_steep_s[side] + dt if -elevation >= ARM_PLANT_DEG else 0.0
            if -elevation >= ARM_PLANT_DEG and (self.arm_peak[side] - elevation >= ARM_RISE_RAMP[1]
                                                or self.arm_steep_s[side] >= ARM_REPLANT_S):
                self.arm_base[side], self.arm_up[side] = elevation, False
                return 1.0
            return 0.0
        if base is None:
            if -elevation >= ARM_PLANT_DEG:                         # first planted reading
                self.arm_base[side] = elevation
                return 1.0
            return 0.0
        rise = elevation - base
        if rise >= ARM_RISE_RAMP[1]:
            self.arm_up[side], self.arm_peak[side], self.arm_steep_s[side] = True, elevation, 0.0
            return 0.0
        if rise < ARM_CREEP_DEG:
            self.arm_base[side] += (1.0 - math.exp(-dt / ARM_BASELINE_TIME_CONSTANT_S)) * rise
        return 1.0 - _ramp(rise, *ARM_RISE_RAMP)

    def _level_knees(self, local):
        """Both legs kneeling: roll the pelvis (spine1, thighs, shanks keep their globals) so both
        hips -- and the knees straight below them -- are at one height."""
        weight = min(self.weights["left"], self.weights["right"])
        if weight <= 0.0:
            return 0.0
        g = _globals(local)
        p = self._positions(g)
        head = self._head(p)
        if head is None:
            return 0.0
        legs = [(p[k, 1] - p[h, 1]) for h, k in ((1, 4), (2, 5))]          # knee below hip (kept)

        def diff(degrees):
            r = _axis_rotation(head, degrees)
            return ((p[0] + r @ (p[1] - p[0]))[1] + legs[0]) - ((p[0] + r @ (p[2] - p[0]))[1] + legs[1])

        degrees = weight * _bisect(diff, MAX_ROLL_DEG)
        if abs(degrees) < 1e-6:
            return 0.0
        keep = {joint: g[joint].copy() for joint in (1, 2, 3, 4, 5)}
        local[0] = _axis_rotation(head, degrees) @ local[0]
        for joint in sorted(keep):
            g = _globals(local)
            local[joint] = g[PARENTS[joint]].T @ keep[joint]
        return degrees

    def _head(self, p):
        head = p[12] - p[0]
        head[1] = 0.0
        return None if np.linalg.norm(head) < 0.1 else _unit(head)

    def _roll_target(self, local):
        """Spine3 roll (deg, about the head axis) that puts both supporting palms at one height."""
        weight = min(self.arm_weights["left"], self.arm_weights["right"])
        if weight <= 1e-3:
            return 0.0
        g = _globals(local)
        p = self._positions(g)
        head = self._head(p)
        if head is None:
            return 0.0
        drops = [(float(np.linalg.norm(self.offsets[forearm])) + float(np.linalg.norm(self.offsets[wrist])))
                 * float(_unit(g[forearm] @ self.offsets[wrist])[1]) for _s, _u, forearm, wrist in ARMS]

        def diff(degrees):
            r = _axis_rotation(head, degrees)
            s = [(p[9] + r @ (p[joint] - p[9]))[1] for joint in (16, 17)]
            return (s[0] + drops[0]) - (s[1] + drops[1])

        return weight * _bisect(diff, MAX_ROLL_DEG)

    def _apply_roll(self, local, degrees):
        if abs(degrees) < 1e-6:
            return
        g = _globals(local)
        head = self._head(self._positions(g))
        if head is None:
            return
        keep = {joint: g[joint].copy() for joint in (18, 19)}
        local[9] = g[6].T @ _axis_rotation(head, degrees) @ g[9]
        for joint in sorted(keep):
            g = _globals(local)
            local[joint] = g[PARENTS[joint]].T @ keep[joint]

    def _spine_target(self, local):
        """Spine2/3 pitch (deg) putting the supporting palms on the supporting knees' plane (straight
        arms along the measured forearms: wrist = shoulder + arm length * forearm direction)."""
        arm_w = np.array([self.arm_weights["left"], self.arm_weights["right"]])
        leg_w = np.array([self.weights["left"], self.weights["right"]])
        strength = min(1.0, arm_w.sum()) * min(1.0, leg_w.sum())
        if strength < 1e-3:
            return 0.0
        g = _globals(local)
        p = self._positions(g)
        axis = self._cross_axis(p)
        if axis is None:
            return 0.0
        knee = float(leg_w @ np.array([p[4, 1], p[5, 1]]) / leg_w.sum()) - KNEE_CONTACT_M
        arm_w = arm_w / arm_w.sum()
        drops = np.array([(float(np.linalg.norm(self.offsets[forearm])) + float(np.linalg.norm(self.offsets[wrist])))
                          * float(_unit(g[forearm] @ self.offsets[wrist])[1])
                          for _s, _u, forearm, wrist in ARMS]) - HAND_CONTACT_M

        def gap(degrees):
            half = _axis_rotation(axis, degrees / 2.0)
            full = half @ half
            base = p[6] + half @ (p[9] - p[6])
            shoulders = np.array([(base + full @ (p[joint] - p[9]))[1] for joint in (16, 17)])
            return float(arm_w @ (shoulders + drops)) - knee

        lo, hi = -MAX_SPINE_DEG, MAX_SPINE_DEG
        f_lo, f_hi = gap(lo), gap(hi)
        if f_lo * f_hi > 0.0:
            target = lo if abs(f_lo) < abs(f_hi) else hi
        else:
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                f_mid = gap(mid)
                if f_mid * f_lo <= 0.0:
                    hi = mid
                else:
                    lo, f_lo = mid, f_mid
            target = 0.5 * (lo + hi)
        return strength * target

    def _apply_spine(self, local, degrees):
        if abs(degrees) < 1e-6:
            return
        g = _globals(local)
        axis = self._cross_axis(self._positions(g))
        if axis is None:
            return
        keep = {joint: g[joint].copy() for joint in (18, 19)}         # measured forearms
        half = _axis_rotation(axis, degrees / 2.0)
        for joint in (6, 9):
            g = _globals(local)
            local[joint] = g[PARENTS[joint]].T @ half @ g[joint]
        for joint in sorted(keep):
            g = _globals(local)
            local[joint] = g[PARENTS[joint]].T @ keep[joint]

    def apply(self, local, dt):
        local = np.asarray(local, dtype=np.float64).copy()
        g = _globals(local)
        trunk = _elevation(g[3] @ self.offsets[6])
        gate = 1.0 - _ramp(abs(trunk), *TRUNK_GATE_RAMP)
        step = WEIGHT_RATE_PER_S * dt
        sensors = {"trunkDeg": round(trunk, 1)}
        for side, _thigh, shank, ankle in LEGS:
            elevation = _elevation(g[shank] @ self.offsets[ankle])
            target, rise = self._target(side, elevation, gate, dt)
            sensors[f"{side}ShankDeg"] = round(elevation, 1)
            sensors[f"{side}RiseDeg"] = None if rise is None else round(rise, 1)
            if self.seeded:
                self.weights[side] += float(np.clip(target - self.weights[side], -step, step))
            else:
                self.weights[side] = target
        arm_step = ARM_WEIGHT_RATE_PER_S * dt
        for side, _upper, forearm, wrist in ARMS:
            elevation = _elevation(g[forearm] @ self.offsets[wrist])
            sensors[f"{side}ForearmDeg"] = round(elevation, 1)
            target = gate * _ramp(-elevation, *ARM_SUPPORT_RAMP) * self._arm_planted(side, elevation, dt)
            if self.seeded:
                self.arm_weights[side] += float(np.clip(target - self.arm_weights[side], -arm_step, arm_step))
            else:
                self.arm_weights[side] = target
        first = not self.seeded
        self.seeded = True

        before = {}
        for side, thigh, shank, _ankle in LEGS:
            before[side] = _elevation(g[thigh] @ self.offsets[shank])
            weight = self.weights[side]
            if weight <= 0.0:
                continue
            swing = _scaled(_rotation_between(g[thigh] @ self.offsets[shank], DOWN), weight)
            new_thigh = swing @ g[thigh]
            local[thigh] = g[PARENTS[thigh]].T @ new_thigh
            local[shank] = new_thigh.T @ g[shank]          # shank keeps its (sensed) global
            g = _globals(local)
        self._lay_feet(local)
        pelvis_roll = self._level_knees(local)
        self._straighten_arms(local)
        self._flatten_hands(local)
        alpha = 1.0 if first else 1.0 - math.exp(-dt / SPINE_TIME_CONSTANT_S)
        self.spine_deg += alpha * (self._spine_target(local) - self.spine_deg)
        self._apply_spine(local, self.spine_deg)
        self._straighten_arms(local)
        self.roll_deg += alpha * (self._roll_target(local) - self.roll_deg)
        self._apply_roll(local, self.roll_deg)
        self._straighten_arms(local)
        g = _globals(local)
        diag = {"enabled": True, "gate": round(gate, 3),
                "armWeights": {k: round(v, 3) for k, v in self.arm_weights.items()},
                "spineDeg": round(self.spine_deg, 1), "spineRollDeg": round(self.roll_deg, 1),
                "pelvisRollDeg": round(pelvis_roll, 1),
                "weights": {k: round(v, 3) for k, v in self.weights.items()},
                "lifted": dict(self.up),
                "shankHeightM": {k: None if v is None else round(v, 3) for k, v in self.height.items()},
                "baselineDeg": {k: None if v is None else round(v, 1) for k, v in self.baseline.items()},
                "sensorElevationDeg": sensors,
                "thighElevationDeg": {side: {"network": round(before[side], 1),
                                             "applied": round(_elevation(g[thigh] @ self.offsets[shank]), 1)}
                                      for side, thigh, shank, _a in LEGS}}
        return local, diag

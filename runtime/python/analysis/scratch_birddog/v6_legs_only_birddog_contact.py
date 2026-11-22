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
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from production.arm_prior import PARENTS, _globals, _rotation_between, _scaled
from production.bridge_contact import _ramp, _rest_offsets, _unit

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
DOWN = np.array([0.0, -1.0, 0.0])
LEGS = (("left", 1, 4, 7), ("right", 2, 5, 8))     # side, thigh, shank, ankle
SHANK_LABELS = {"left": "ls", "right": "rs"}


def _elevation(direction):
    return math.degrees(math.asin(float(np.clip(_unit(direction)[1], -1.0, 1.0))))


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
        diag = {"enabled": True, "gate": round(gate, 3),
                "weights": {k: round(v, 3) for k, v in self.weights.items()},
                "lifted": dict(self.up),
                "shankHeightM": {k: None if v is None else round(v, 3) for k, v in self.height.items()},
                "baselineDeg": {k: None if v is None else round(v, 1) for k, v in self.baseline.items()},
                "sensorElevationDeg": sensors,
                "thighElevationDeg": {side: {"network": round(before[side], 1),
                                             "applied": round(_elevation(g[thigh] @ self.offsets[shank]), 1)}
                                      for side, thigh, shank, _a in LEGS}}
        return local, diag

"""Bird-dog support-knee rule.  Production: frame_chain.postprocess_output applies it only for the
action ``birddog`` (BIRDDOG_ACTIONS); every other action's output is untouched.

Correct exercise (user, 2026-11-23): four-point kneeling start (hands and knees on the mat), the
contralateral arm and leg extend together to horizontal, then return to the four-point stance.  In
the start hold both knees and both hands are on the mat.

What goes wrong without it (real recording birddog_20261123_005257, user held the start pose for
15 s): the left thigh swings from -61 deg (vertical-ish) to -15 deg and the knee opens from ~90 to
~18 deg -- an extended leg -- from t=5.5 s on, survives the start-pose rebase at 8 s and stays until
the first repetition; the same drift recurs between repetitions.  The shank sensor stays level all
the time.  A level shank is ambiguous: lying on the mat (kneeling) and stretched back give the same
shank orientation; only the thigh differs and the thigh has no sensor.  The start-pose fit only
corrects the sensors and StartPoseConstraint has no bird-dog weights.

Which leg is lifted: its own shank.  On birddog_20261123_021217 (4 contralateral reps, 4 arm-only,
4 leg-only) every real leg extension raised the shank 16-31 deg above the start hold, while arm-only
reaches and the holds stayed within +-6 deg.  The opposite forearm is NOT a usable cue (v1 of this
rule used it: arm-only reaches released the leg and the network stretched it; leg-only lifts were
pressed back down).  So:
  support(leg) = gate * shank_not_hanging * (1 - ramp(shank rise above its floor baseline, LIFT_RAMP))
* floor baseline: the shank elevation in the first CAPTURE_OUTPUTS gated outputs after a reset (run.py
  resets at the calibration start and at every start-pose rebase, i.e. in the four-point hold); while
  the leg is down (rise < CREEP_DEG) it follows slow sensor drift (BASELINE_TIME_CONSTANT_S).  Before
  it is captured the absolute elevation is used (FALLBACK_RAMP);
* shank_not_hanging: off when the shank points steeply down (standing / sitting);
* gate: four-point posture -- the waist sensor's spine axis near horizontal (TRUNK_GATE_RAMP).
The weight is rate limited (WEIGHT_RATE_PER_S) and seeded from its target on the first output
after a reset.  A supported thigh is swung toward straight down (knee under the hip, the quadruped
template) by its weight; the shank and everything below keep their global orientation (= sensor).
A released leg is the network's.
"""
from __future__ import annotations

import math

import numpy as np

from production.arm_prior import PARENTS, _globals, _rotation_between, _scaled
from production.bridge_contact import _ramp, _rest_offsets, _unit

LIFT_RAMP = (6.0, 14.0)              # shank rise above the floor baseline: >=14 deg -> leg released
FALLBACK_RAMP = (25.0, 40.0)         # absolute shank elevation, before the baseline is captured
CAPTURE_OUTPUTS = 15                 # ~0.5 s of gated outputs after a reset
MAX_BASELINE_DEG = 25.0              # only a near-level shank can be a floor baseline
CREEP_DEG = 4.0                      # below this rise the baseline follows sensor drift
BASELINE_TIME_CONSTANT_S = 5.0
SHANK_HANG_RAMP = (-50.0, -35.0)     # shank pointing down steeper than -50 deg: not kneeling
TRUNK_GATE_RAMP = (35.0, 50.0)       # |spine1 axis elevation|: four-point below 35 deg, off above 50
WEIGHT_RATE_PER_S = 4.0
DOWN = np.array([0.0, -1.0, 0.0])
LEGS = (("left", 1, 4, 7), ("right", 2, 5, 8))     # side, thigh, shank, ankle


def _elevation(direction):
    return math.degrees(math.asin(float(np.clip(_unit(direction)[1], -1.0, 1.0))))


class BirdDogContact:
    def __init__(self) -> None:
        self.offsets = _rest_offsets()
        self.reset("init")

    def reset(self, reason: str) -> None:
        self.weights = {"left": 0.0, "right": 0.0}
        self.baseline = {"left": None, "right": None}
        self.capture = {"left": [], "right": []}
        self.seeded = False
        self.reason = reason

    def _target(self, side, shank, gate, dt):
        hang = _ramp(shank, *SHANK_HANG_RAMP)
        if self.baseline[side] is None:
            if gate > 0.9 and abs(shank) <= MAX_BASELINE_DEG:
                self.capture[side].append(shank)
                if len(self.capture[side]) >= CAPTURE_OUTPUTS:
                    self.baseline[side] = float(np.mean(self.capture[side]))
            return gate * hang * (1.0 - _ramp(shank, *FALLBACK_RAMP)), None
        rise = shank - self.baseline[side]
        if rise < CREEP_DEG and gate > 0.9:
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
                "baselineDeg": {k: None if v is None else round(v, 1) for k, v in self.baseline.items()},
                "sensorElevationDeg": sensors,
                "thighElevationDeg": {side: {"network": round(before[side], 1),
                                             "applied": round(_elevation(g[thigh] @ self.offsets[shank]), 1)}
                                      for side, thigh, shank, _a in LEGS}}
        return local, diag

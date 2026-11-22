"""Bird-dog support-knee rule.  Production: frame_chain.postprocess_output applies it only for the
action ``birddog`` (BIRDDOG_ACTIONS); every other action's output is untouched.

Correct exercise (user, 2026-11-23): four-point kneeling start (hands and knees on the mat), the
contralateral arm and leg extend together to horizontal, then return to the four-point stance.  In
the start hold both knees and both hands are on the mat.

What goes wrong without it (real recording birddog_20261123_005257, user held the start pose for
15 s): the left thigh swings from -61 deg (vertical-ish) to -15 deg and the knee opens from ~90 to
~18 deg -- an extended leg -- from t=5.5 s on, survives the start-pose rebase at 8 s and stays until
the first repetition at 15 s; the same drift recurs between repetitions (t 21-22, 34-35, 40-41).  The
left shank sensor stays level (0 deg) all the time and both forearms stay vertical (-80..-89 deg).
A level shank is exactly ambiguous: lying on the mat (kneeling) and stretched back (extended leg)
give the same shank orientation; only the thigh differs and the thigh has no sensor.  The start-pose
fit only corrects the sensors and StartPoseConstraint has no bird-dog weights, so nothing on the
output side resolves it.  (The lying-cat rule judges leg support by the shank alone, which is the
same ambiguity, so it cannot simply be reused here.)

Rule: the contralateral forearm disambiguates.  A leg can only be extended while the opposite arm
is reaching; while the opposite forearm points down at the mat the leg is a support leg:
  support(leg) = gate * arm_down(opposite forearm) * shank_on_floor(own shank)
* arm_down: ramp on the measured forearm pointing down (ARM_SUPPORT_RAMP, as lyingcat);
* shank_on_floor: the shank is near level -- released when it rises above LEG_RELEASE_RAMP (a leg
  lifted without the arm still follows its sensor) or hangs steeply down (standing / sitting);
* gate: four-point posture -- the waist sensor's spine axis near horizontal (TRUNK_GATE_RAMP).
The weight is rate limited (WEIGHT_RATE_PER_S) and seeded from its target on the first output
after a reset (run.py resets on every start-pose rebase; starting at 0 would pop the knee).
A supported thigh is swung toward straight down (knee under the hip, the quadruped template) by its
weight; the shank and everything below keep their global orientation (shank = sensor).
"""
from __future__ import annotations

import math

import numpy as np

from production.arm_prior import PARENTS, _globals, _rotation_between, _scaled
from production.bridge_contact import _ramp, _rest_offsets, _unit

ARM_SUPPORT_RAMP = (40.0, 55.0)      # forearm -elevation: <=40 deg reaching, >=55 deg on the mat
LEG_RELEASE_RAMP = (25.0, 40.0)      # shank elevation: lifted above 40 deg -> released
SHANK_HANG_RAMP = (-50.0, -35.0)     # shank pointing down steeper than -50 deg: not kneeling
TRUNK_GATE_RAMP = (35.0, 50.0)       # |spine1 axis elevation|: four-point below 35 deg, off above 50
WEIGHT_RATE_PER_S = 4.0
DOWN = np.array([0.0, -1.0, 0.0])
# side, thigh, shank, ankle, opposite forearm, opposite wrist
LEGS = (("left", 1, 4, 7, 19, 21), ("right", 2, 5, 8, 18, 20))


def _elevation(direction):
    return math.degrees(math.asin(float(np.clip(_unit(direction)[1], -1.0, 1.0))))


class BirdDogContact:
    def __init__(self) -> None:
        self.offsets = _rest_offsets()
        self.reset("init")

    def reset(self, reason: str) -> None:
        self.weights = {"left": 0.0, "right": 0.0}
        self.seeded = False
        self.reason = reason

    def apply(self, local, dt):
        local = np.asarray(local, dtype=np.float64).copy()
        g = _globals(local)
        trunk = _elevation(g[3] @ self.offsets[6])
        gate = 1.0 - _ramp(abs(trunk), *TRUNK_GATE_RAMP)
        step = WEIGHT_RATE_PER_S * dt
        sensors, targets = {"trunkDeg": round(trunk, 1)}, {}
        for side, _thigh, shank, ankle, forearm, wrist in LEGS:
            arm = _elevation(g[forearm] @ self.offsets[wrist])
            leg = _elevation(g[shank] @ self.offsets[ankle])
            sensors[f"{side}ShankDeg"] = round(leg, 1)
            sensors[f"{side}OppositeForearmDeg"] = round(arm, 1)
            targets[side] = (gate * _ramp(-arm, *ARM_SUPPORT_RAMP)
                             * (1.0 - _ramp(leg, *LEG_RELEASE_RAMP)) * _ramp(leg, *SHANK_HANG_RAMP))
        for side, target in targets.items():
            if self.seeded:
                self.weights[side] += float(np.clip(target - self.weights[side], -step, step))
            else:
                self.weights[side] = target
        self.seeded = True

        before = {}
        for side, thigh, shank, _ankle, _forearm, _wrist in LEGS:
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
                "sensorElevationDeg": sensors,
                "thighElevationDeg": {side: {"network": round(before[side], 1),
                                             "applied": round(_elevation(g[thigh] @ self.offsets[shank]), 1)}
                                      for side, thigh, shank, _a, _f, _w in LEGS}}
        return local, diag

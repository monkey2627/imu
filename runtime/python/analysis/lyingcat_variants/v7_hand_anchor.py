"""[Experiment 2026-11-21] production rule + bounded palm anchor.

Upper arm swings toward a latched (leaky) wrist anchor while the forearm global
(sensor) is kept; the elbow bend this creates is capped at ``BEND_MAX_DEG``,
then production's elbow-twist limiter removes any roll it introduced.
"""
from __future__ import annotations

import math

import numpy as np

from production import lyingcat_contact as base
from production.arm_prior import _globals, _rotation_between, _scaled
from production.bridge_contact import _unit

BEND_MAX_DEG = 20.0
STRENGTH = 1.0
ANCHOR_LEAK_S = 10.0
STABLE_S = 0.5


class LyingCatContact(base.LyingCatContact):
    def reset(self, reason):
        super().reset(reason)
        self.hand_anchor = None
        self.stable_s = 0.0

    def _limit_elbow_twist(self, local):
        dt = getattr(self, "_dt", 1 / 30)
        p = self._positions(_globals(local))
        if min(self.weights.values()) > 0.95:
            self.stable_s += dt
        else:
            self.stable_s = 0.0
            self.hand_anchor = None
        if self.hand_anchor is None and self.stable_s >= STABLE_S:
            self.hand_anchor = {w: p[w].copy() for w in (20, 21)}
        if self.hand_anchor is not None:
            leak = 1.0 - math.exp(-dt / ANCHOR_LEAK_S)
            for side, upper, forearm, wrist, _h in base.ARMS:
                g = _globals(local)
                p = self._positions(g)
                anchor = self.hand_anchor[wrist]
                anchor += leak * (p[wrist] - anchor)          # slowly follow the free (straight-arm) hand
                target = anchor.copy()
                target[1] = p[wrist, 1]                        # height stays with the spine solve
                elbow = target - g[forearm] @ self.offsets[wrist]
                wanted = elbow - p[upper]
                fdir = _unit(g[forearm] @ self.offsets[wrist])
                # cap the elbow bend: upper-arm direction within BEND_MAX of the forearm direction
                wanted = _unit(wanted)
                bend = math.degrees(math.acos(float(np.clip(wanted @ fdir, -1, 1))))
                if bend > BEND_MAX_DEG:
                    wanted = _unit(_scaled(_rotation_between(fdir, wanted), BEND_MAX_DEG / bend) @ fdir)
                current = g[upper] @ self.offsets[forearm]
                swing = _scaled(_rotation_between(current, wanted), STRENGTH * self.weights[f"{side}Arm"])
                self._set_global(local, g, upper, swing @ g[upper], forearm)
        super()._limit_elbow_twist(local)

    def apply(self, local, dt, measured=None, labels=None):
        self._dt = dt
        return super().apply(local, dt, measured=measured, labels=labels)

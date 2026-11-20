"""[Experiment 2026-11-21] v7 hand anchor, but the swing is backed off so the wrist never sinks below the knee plane."""
from __future__ import annotations

import math

import numpy as np

from analysis.lyingcat_variants import v7_hand_anchor as v7
from production import lyingcat_contact as base
from production.arm_prior import _globals, _rotation_between, _scaled
from production.bridge_contact import _unit


class LyingCatContact(v7.LyingCatContact):
    def _limit_elbow_twist(self, local):
        dt = getattr(self, "_dt", 1 / 30)
        if min(self.weights.values()) > 0.95:
            self.stable_s += dt
        else:
            self.stable_s, self.hand_anchor = 0.0, None
        p = self._positions(_globals(local))
        if self.hand_anchor is None and self.stable_s >= v7.STABLE_S:
            self.hand_anchor = {w: p[w].copy() for w in (20, 21)}
        if self.hand_anchor is not None:
            leak = 1.0 - math.exp(-dt / v7.ANCHOR_LEAK_S)
            floor = 0.5 * (p[4, 1] + p[5, 1])
            for side, upper, forearm, wrist, _h in base.ARMS:
                g = _globals(local)
                p = self._positions(g)
                anchor = self.hand_anchor[wrist]
                anchor += leak * (p[wrist] - anchor)
                low = min(p[wrist, 1], floor)            # never push the hand lower than now / the knees
                target = anchor.copy(); target[1] = p[wrist, 1]
                fvec = g[forearm] @ self.offsets[wrist]
                fdir = _unit(fvec)
                wanted = _unit(target - fvec - p[upper])
                bend = math.degrees(math.acos(float(np.clip(wanted @ fdir, -1, 1))))
                if bend > v7.BEND_MAX_DEG:
                    wanted = _unit(_scaled(_rotation_between(fdir, wanted), v7.BEND_MAX_DEG / bend) @ fdir)
                current = g[upper] @ self.offsets[forearm]
                full = _rotation_between(current, wanted)
                length = float(np.linalg.norm(self.offsets[forearm]))
                lo, hi = 0.0, self.weights[f"{side}Arm"]
                def wrist_y(s):
                    return p[upper, 1] + float((_scaled(full, s) @ current)[1]) + fvec[1]
                if wrist_y(hi) < low - 0.003:
                    for _ in range(12):
                        mid = 0.5 * (lo + hi)
                        lo, hi = (mid, hi) if wrist_y(mid) >= low - 0.003 else (lo, mid)
                    hi = lo
                self._set_global(local, g, upper, _scaled(full, hi) @ g[upper], forearm)
        base.LyingCatContact._limit_elbow_twist(self, local)

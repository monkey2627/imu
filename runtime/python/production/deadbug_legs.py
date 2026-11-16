"""Dead-bug thigh estimation from shank translation, not from single-frame angles.

Why: the five-IMU layout has no thigh sensor.  A tabletop shank (thigh up, knee
~90 deg) and a lowered straight-leg shank are both close to horizontal, so the
shank orientation cannot tell them apart, and the network's thigh is wrong in
both directions on the 2026-11-13 recordings (docs/13).  What *does* differ is
where the ankle strap is: extending and lowering the leg moves it ~0.5-0.7 m
toward the feet and down.  That translation is in the (corrected, see
accel_frame.py) linear acceleration.

Model (supine, hip fixed relative to the waist sensor), per leg::

    sensor position  p = Lt * u(phi) + K * Ls * s
    u(phi)           = cos(phi) * A + sin(phi) * B

with A the feetward axis and B the chest normal of the waist sensor, s the
measured knee->ankle direction and phi the only unknown: the thigh angle in
the sagittal plane (0 = along the floor toward the feet, 90 = straight up).

Evidence, per leg, independently:

* still -> moving -> still segments are delimited by the shank's own gyro and
  acceleration.  At the end of a segment the acceleration is integrated with
  zero velocity at both ends (removes constant bias exactly), giving the
  segment displacement dp.  ``Lt*(u(phi_end) - u(phi_start)) = dp - K*Ls*ds``
  is solved in closed form for BOTH angles; the solution must be anatomically
  feasible and is checked against the angle carried into the segment.
* while moving, the acceleration is integrated causally and projected onto
  the one-dimensional set of reachable sensor positions each frame (removes
  drift normal to it), so a real extension is followed continuously.  This is
  less certain, so confidence decays with segment time and the next still
  period re-anchors the angle.
* feet on the floor (shank pointing down toward the feet) gives an initial
  angle from the floor height.

Nothing here reads the network's knee angle.  Without evidence (not supine,
never moved, inconsistent segment) the confidence is 0 and the network pose is
left unchanged.  The applied angle is rate limited.
"""
from __future__ import annotations

import collections
import dataclasses
import math

import numpy as np
import torch

from production.arm_prior import PARENTS, _globals, _rotation_between, _scaled

FPS = 60.0
THIGH_LENGTH = 0.389        # SMPL basicmodel_m rest pose, hip -> knee (m)
SHANK_LENGTH = 0.429        # knee -> ankle
SENSOR_FRACTION = 0.95      # ankle strap: 95% of knee -> ankle (wristankle attachment config)
LEGS = {"ls": (1, 4), "rs": (2, 5)}   # label -> (hip joint, knee joint)
SIDE_NAMES = {"ls": "left", "rs": "right"}

SUPINE_MIN_CHEST_UP = 0.6          # waist chest normal . world up
STILL_GYRO_DEG_S = 12.0
STILL_ACCEL = 0.6                  # m/s^2 after removing the still bias
MOVE_GYRO_DEG_S = 20.0
MOVE_ACCEL = 0.9
SETTLE_FRAMES = 10                 # consecutive still frames that close a segment
MARGIN_FRAMES = 6
NO_THIGH_MOTION = 0.45             # |chord| / Lt below this: knee moved < ~0.18 m (real moves: >= 1.1)
MAX_CHORD = 2.3                    # slightly over 2 (drift) is clamped; beyond is rejected
PHI_MIN, PHI_MAX = -15.0, 140.0
KNEE_MIN, KNEE_MAX = -10.0, 155.0  # knee flexion = phi - shank elevation
MAX_RATE_DEG_S = 90.0
# After a confident zero-velocity solve (the move has already happened) the shown thigh may catch up faster;
# at 90 deg/s a 100 deg extension finished showing ~1.1 s after the solve, often after the hold ended
# (2026-11-16 recording 045157).
CATCHUP_RATE_DEG_S = 270.0
CATCHUP_MIN_CONFIDENCE = 0.9
# Shank-pitch-only provisional decision, used when the integrated translation is unusable (drift in long,
# never-still segments).  Every real extension/return pitches the shank down >= 25 deg and back up; small
# adjustments of a held leg stay below ~15 deg.  The zero-velocity solve still confirms or reverts it.
DIP_ONLY_DEG = 25.0
DIP_ONLY_RECOVER_DEG = 8.0
EXTENSION_DOWN = 0.6               # extension displacement ~ (0.55 m feetward, 0.3 m down)
DECIDE_TRANSLATION = 0.15          # m along that direction ...
DECIDE_SPEED = 0.2                 # ... moving at m/s ...
DECIDE_SECONDS = 1.5               # ... within this time of the segment start (drift still small)
QUASI_STILL_GYRO_DEG_S = 20.0
QUASI_STILL_FRAMES = 6
LONG_SEGMENT_SECONDS = 3.0
DECIDE_SHANK_DIP = 10.0            # ... and the shank pitched down first (hip leads; seen in every real move)
PROVISIONAL_CONFIDENCE = 0.6
MAX_SEGMENT_SECONDS = 8.0
APPLY_MIN_CONFIDENCE = 0.35
FULL_WEIGHT_CONFIDENCE = 0.55
WEIGHT_RATE_PER_S = 1.5          # blend in/out over >= 0.67 s, never a snap
EXTENDED_ENTER, EXTENDED_EXIT = 40.0, 50.0     # phase labels only (hysteresis), never used as a trigger
FLEXED_ENTER, FLEXED_EXIT = 65.0, 55.0


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _angle_of(v, a_axis, b_axis) -> float:
    return math.degrees(math.atan2(float(v @ b_axis), float(v @ a_axis)))


def _u(phi_deg, a_axis, b_axis):
    r = math.radians(phi_deg)
    return math.cos(r) * a_axis + math.sin(r) * b_axis


def _wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


@dataclasses.dataclass
class LegState:
    phi: float | None = None          # estimated thigh angle (deg)
    confidence: float = 0.0
    shown: float | None = None        # rate-limited angle actually applied
    phase: str = "unknown"
    reason: str = "no evidence yet"
    moving: bool = False
    still_frames: int = 0
    bias: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    segment: dict | None = None
    recent: collections.deque = dataclasses.field(default_factory=lambda: collections.deque(maxlen=MARGIN_FRAMES))
    gyro: collections.deque = dataclasses.field(default_factory=lambda: collections.deque(maxlen=6))
    accel: collections.deque = dataclasses.field(default_factory=lambda: collections.deque(maxlen=6))
    previous_rot: np.ndarray | None = None
    events: list = dataclasses.field(default_factory=list)
    last_extended: float = 25.0       # most recent solved angles, used as provisional targets
    last_flexed: float = 110.0
    weight: float = 0.0


class DeadBugLegEstimator:
    """Per-leg thigh angle from shank translation; see the module docstring."""

    def __init__(self, labels, history: int = 32) -> None:
        self.labels = list(labels)
        self.history = history
        self.reset("init")

    # ------------------------------------------------------------------ state
    def reset(self, reason: str = "reset") -> None:
        self.legs = {label: LegState(reason=f"reset: {reason}") for label in LEGS}
        self.frame = -1
        self._per_frame = collections.deque(maxlen=self.history)
        self.reset_reason = reason

    # --------------------------------------------------------------- per frame
    def observe(self, orientations, accelerations) -> None:
        """One 60 Hz frame. orientations: (n_labels,3,3) calibrated SMPL-world bone rotations
        (the model's input frame); accelerations: (n_labels,3) corrected SMPL-world linear
        acceleration (m/s^2)."""
        self.frame += 1
        ori = np.asarray(orientations, dtype=np.float64)
        acc = np.asarray(accelerations, dtype=np.float64)
        waist = ori[self.labels.index("waist")]
        a_axis, b_axis = -waist[:, 1], waist[:, 2]
        supine = b_axis[1] >= SUPINE_MIN_CHEST_UP
        snapshot = {}
        for label, leg in self.legs.items():
            i = self.labels.index(label)
            self._step_leg(label, leg, ori[i], acc[i], a_axis, b_axis, waist[:, 0], supine)
            snapshot[label] = (leg.shown, self._applied_confidence(leg), leg.phase, leg.reason, leg.weight)
        self._per_frame.append((self.frame, snapshot))

    def _applied_confidence(self, leg: LegState) -> float:
        return 0.0 if leg.shown is None else float(leg.confidence)

    def _event(self, leg: LegState, kind: str, **details) -> None:
        item = {"frame": self.frame, "t": round(self.frame / FPS, 3), "event": kind, **details}
        leg.events.append(item)
        del leg.events[:-64]

    def _step_leg(self, label, leg, rot, acc, a_axis, b_axis, lateral, supine) -> None:
        shank = rot @ np.array([0.0, -1.0, 0.0])
        elevation = _angle_of(shank, a_axis, b_axis)
        gyro = 0.0 if leg.previous_rot is None else math.degrees(
            math.acos(float(np.clip((np.trace(leg.previous_rot.T @ rot) - 1.0) / 2.0, -1.0, 1.0)))) * FPS
        leg.previous_rot = rot
        leg.gyro.append(gyro)
        leg.accel.append(acc)
        g_mean = float(np.mean(leg.gyro))
        a_dev = float(np.linalg.norm(np.mean(leg.accel, axis=0) - leg.bias))
        leg.recent.append((acc.copy(), shank.copy()))

        if not supine:
            # Leaving the supine pose (sitting up, rolling): stop estimating and fade the correction
            # out at the normal weight rate; forget the angle only once nothing is applied any more.
            if leg.segment is not None or leg.confidence > 0.0:
                self._event(leg, "not_supine")
            leg.segment, leg.confidence, leg.still_frames = None, 0.0, 0
            leg.reason = "not supine: network pose kept"
            self._advance_shown(leg, elevation)
            if leg.weight <= 0.0:
                leg.phi = leg.shown = None
            leg.phase = "unknown"
            return

        still = g_mean < STILL_GYRO_DEG_S and a_dev < STILL_ACCEL
        moving = g_mean > MOVE_GYRO_DEG_S or a_dev > MOVE_ACCEL
        leg.still_frames = leg.still_frames + 1 if still else 0

        if leg.segment is None:
            if moving:
                self._start_segment(leg, a_axis, b_axis)
            elif leg.still_frames >= SETTLE_FRAMES:
                leg.bias += 0.05 * (acc - leg.bias)
        else:
            seg = leg.segment
            seg["acc"].append(acc.copy())
            seg["shank"].append(shank.copy())
            self._causal_step(leg, seg, acc, shank, a_axis, b_axis)
            if leg.still_frames >= SETTLE_FRAMES:
                self._finish_segment(leg, a_axis, b_axis, lateral, elevation)
            elif len(seg["acc"]) > MAX_SEGMENT_SECONDS * FPS:
                leg.segment = None
                leg.confidence *= 0.5
                leg.reason = "segment too long: confidence halved"
                self._event(leg, "segment_abandoned", seconds=MAX_SEGMENT_SECONDS)
        self._advance_shown(leg, elevation)
        self._label_phase(leg, elevation)

    # --------------------------------------------------------------- evidence
    def _start_segment(self, leg, a_axis, b_axis) -> None:
        leg.segment = {
            "start_frame": self.frame - len(leg.recent) + 1,
            "phi_start": leg.phi,
            "acc": [a.copy() for a, _ in leg.recent],
            "shank": [s.copy() for _, s in leg.recent],
            "announced": None,
            "displacement": np.zeros(3),
            "confidence_start": leg.confidence,
            "elev_start": _angle_of(leg.recent[0][1], a_axis, b_axis),
            "elev_min": min(_angle_of(sh, a_axis, b_axis) for _, sh in leg.recent),
            "ref_displacement": np.zeros(3), "ref_frames": 0, "quiet": 0,
        }
        leg.velocity = np.zeros(3)
        for a, _ in list(leg.recent)[1:]:
            self._integrate(leg, leg.segment, a)

    def _integrate(self, leg, seg, acc) -> None:
        leg.velocity = leg.velocity + (acc - leg.bias) / FPS
        seg["displacement"] = seg["displacement"] + leg.velocity / FPS

    def _causal_step(self, leg, seg, acc, shank, a_axis, b_axis) -> None:
        """Provisional direction while moving.  Double integration drifts (~0.3 m after 2 s on the
        2026-11-13 recordings), so only early, fast, sizeable translation along the extension
        direction starts a provisional move; the zero-velocity solve at the next still period decides."""
        self._integrate(leg, seg, acc)
        elevation = _angle_of(shank, a_axis, b_axis)
        seg["ref_frames"] += 1
        quiet = float(np.mean(leg.gyro)) < QUASI_STILL_GYRO_DEG_S
        seg["quiet"] = seg["quiet"] + 1 if quiet else 0
        if ((seg["announced"] is not None and seg["quiet"] == QUASI_STILL_FRAMES)
                or (seg["announced"] is None and quiet and elevation >= seg["elev_start"] - 2.0)):
            # Slow pause, or slow at the top of the shank's pitch: a possible new move starts here.
            # Re-reference so a long, never fully still segment does not carry seconds of drift.
            leg.velocity = np.zeros(3)
            seg["ref_displacement"] = seg["displacement"].copy()
            seg["ref_frames"] = 0
            seg["elev_start"] = elevation
            seg["elev_min"] = elevation
        seg["elev_min"] = min(seg["elev_min"], elevation)
        start = leg.phi                                       # settled or provisional angle so far
        if start is None:
            return
        axis = _unit(a_axis - EXTENSION_DOWN * b_axis)        # ankle goes toward the feet and down
        x = float((seg["displacement"] - seg["ref_displacement"]) @ axis)
        xv = float(leg.velocity @ axis)
        elapsed = seg["ref_frames"] / FPS
        dip = seg["elev_start"] - seg["elev_min"]
        evidence = {"translation": round(x, 3), "speed": round(xv, 3), "seconds": round(elapsed, 2),
                    "shankDip": round(dip, 1)}
        kind = None
        if elapsed <= DECIDE_SECONDS and dip >= DECIDE_SHANK_DIP:
            kind = self._decide(seg, start, x, xv)
        if kind is None and dip >= DIP_ONLY_DEG and elevation - seg["elev_min"] >= DIP_ONLY_RECOVER_DEG:
            if seg["announced"] is None and start >= FLEXED_EXIT:
                kind = "start_extend"
            elif seg["announced"] is None and start <= EXTENDED_EXIT:
                kind = "start_return"
            if kind is not None:
                evidence["rule"] = "shank_dip_only"
        if kind is None:
            return
        seg["announced"] = kind
        target = leg.last_extended if kind == "start_extend" else leg.last_flexed
        leg.phi, leg.confidence = target, PROVISIONAL_CONFIDENCE
        leg.reason = (f"{'extending' if kind == 'start_extend' else 'returning'}: ankle moved {x:+.2f} m along "
                      f"feet/down at {xv:+.2f} m/s within {elapsed:.2f} s, shank pitched {dip:.0f} deg "
                      f"-> provisional thigh {target:.0f} deg")
        self._event(leg, kind, phiStart=round(start, 1), provisionalTarget=round(target, 1), **evidence)

    @staticmethod
    def _decide(seg, start, x, xv):
        if (seg["announced"] != "start_extend" and start >= FLEXED_EXIT
                and x >= DECIDE_TRANSLATION and xv >= DECIDE_SPEED):
            return "start_extend"
        if (seg["announced"] != "start_return" and start <= EXTENDED_EXIT
                and x <= -DECIDE_TRANSLATION and xv <= -DECIDE_SPEED):
            return "start_return"
        return None

    def _finish_segment(self, leg, a_axis, b_axis, lateral, elevation) -> None:
        seg, leg.segment = leg.segment, None
        acc = np.asarray(seg["acc"], dtype=np.float64)
        shanks = np.asarray(seg["shank"], dtype=np.float64)
        velocity = np.cumsum(acc, axis=0) / FPS
        velocity -= np.outer(np.linspace(0.0, 1.0, len(velocity)), velocity[-1])  # zero velocity at both ends
        displacement = (np.cumsum(velocity, axis=0) / FPS)[-1]
        chord = (displacement - SENSOR_FRACTION * SHANK_LENGTH * (shanks[-1] - shanks[0])) / THIGH_LENGTH
        c = np.array([chord @ a_axis, chord @ b_axis])
        size = float(np.linalg.norm(c))
        lateral_m = abs(float(displacement @ lateral))
        e_start = _angle_of(shanks[0], a_axis, b_axis)
        details = {"seconds": round(len(acc) / FPS, 2), "displacementFeetward": round(float(displacement @ a_axis), 3),
                   "displacementUp": round(float(displacement @ b_axis), 3), "lateral": round(lateral_m, 3),
                   "chord": round(size, 3)}
        start = seg["phi_start"]
        leg.velocity = np.zeros(3)
        if size < NO_THIGH_MOTION:
            if start is not None:
                leg.phi = start
                leg.confidence = max(seg["confidence_start"], 0.7 if seg["announced"] else 0.0)
            leg.reason = f"segment {details['seconds']} s: ankle moved {np.linalg.norm(displacement):.2f} m, thigh unchanged"
            self._event(leg, "segment_no_thigh_motion", **details)
            return
        if size > MAX_CHORD:
            self._revert(leg, seg, start, details, "segment_rejected",
                         f"segment displacement too large ({size:.2f} Lt, integration drift)")
            return
        half = math.degrees(math.asin(min(size / 2.0, 1.0)))
        mean = math.degrees(math.atan2(-c[0], c[1]))           # c = 2 sin(half) (-sin(mean), cos(mean))
        candidates = []
        for phi_s, phi_e in ((mean - half, mean + half), (mean + 180.0 + half, mean + 180.0 - half)):
            phi_s, phi_e = _wrap(phi_s), _wrap(phi_e)
            feasible = (PHI_MIN - 10 <= phi_s <= PHI_MAX + 10 and PHI_MIN - 10 <= phi_e <= PHI_MAX + 10
                        and KNEE_MIN - 15 <= phi_s - e_start <= KNEE_MAX and KNEE_MIN - 15 <= phi_e - elevation <= KNEE_MAX)
            if feasible:
                candidates.append((phi_s, phi_e))
        if not candidates:
            self._revert(leg, seg, start, details, "segment_infeasible",
                         "segment displacement has no anatomically feasible thigh solution")
            return
        if start is not None:
            phi_s, phi_e = min(candidates, key=lambda sol: abs(sol[0] - start))
            mismatch = abs(phi_s - start)
            if mismatch >= 25.0 and details["seconds"] > LONG_SEGMENT_SECONDS:
                # Linear bias removal is weakest on long, never-still segments; a solution that
                # contradicts the settled start angle is not trusted over the in-segment decisions.
                details.update(phiStart=round(phi_s, 1), phiEnd=round(phi_e, 1), priorStart=round(start, 1))
                leg.confidence = PROVISIONAL_CONFIDENCE
                leg.reason = (f"long {details['seconds']} s segment contradicts the start angle by {mismatch:.0f} deg: "
                              f"kept in-segment result {leg.phi:.0f} deg (conf {leg.confidence:.2f})")
                self._event(leg, "segment_contradicts_prior", **details)
                return
            confidence = 0.95 if mismatch < 25 else 0.7 if mismatch < 45 else 0.4
            if seg["confidence_start"] < 0.6 and details["seconds"] <= 3.5 and lateral_m <= 0.25:
                confidence = max(confidence, 0.75)   # the prior itself was weak; a clean segment wins
        elif len(candidates) == 1:
            (phi_s, phi_e), mismatch, confidence = candidates[0], None, 0.75
        else:
            leg.reason = "two feasible thigh solutions and no prior angle: network pose kept"
            self._event(leg, "segment_ambiguous", **details)
            return
        if details["seconds"] > 5.0:
            confidence *= 0.7
        if lateral_m > 0.25:
            confidence *= 0.5
        phi_e = float(np.clip(phi_e, max(PHI_MIN, elevation + KNEE_MIN), min(PHI_MAX, elevation + KNEE_MAX)))
        leg.phi, leg.confidence = phi_e, confidence
        if confidence >= 0.7:
            if phi_e <= EXTENDED_EXIT:
                leg.last_extended = phi_e
            elif phi_e >= FLEXED_EXIT:
                leg.last_flexed = phi_e
        details.update(phiStart=round(phi_s, 1), phiEnd=round(phi_e, 1), confidence=round(confidence, 2),
                       priorStart=None if start is None else round(start, 1),
                       startMismatch=None if mismatch is None else round(mismatch, 1))
        leg.reason = (f"still after {details['seconds']} s segment: zero-velocity displacement "
                      f"{details['displacementFeetward']:+.2f} m feetward, {details['displacementUp']:+.2f} m up "
                      f"-> thigh {phi_s:.0f} -> {phi_e:.0f} deg (conf {confidence:.2f})")
        self._event(leg, "segment_solved", **details)

    def _revert(self, leg, seg, start, details, kind, why) -> None:
        """Unusable segment: undo any provisional move, keep the pre-segment angle at reduced confidence."""
        if start is not None:
            leg.phi = start
            leg.confidence = 0.7 * seg["confidence_start"]
        else:
            leg.confidence = 0.0
        leg.reason = f"{why}: kept the angle from before the segment (conf {leg.confidence:.2f})"
        self._event(leg, kind, **details)

    # ---------------------------------------------------------------- output
    def _advance_shown(self, leg, elevation) -> None:
        target_weight = 0.0 if leg.phi is None else float(np.clip(
            (leg.confidence - APPLY_MIN_CONFIDENCE) / (FULL_WEIGHT_CONFIDENCE - APPLY_MIN_CONFIDENCE), 0.0, 1.0))
        step = WEIGHT_RATE_PER_S / FPS
        leg.weight += float(np.clip(target_weight - leg.weight, -step, step))
        if leg.phi is None:
            return
        if leg.shown is None:
            leg.shown = leg.phi
            return
        rate = (CATCHUP_RATE_DEG_S if leg.confidence >= CATCHUP_MIN_CONFIDENCE and abs(leg.phi - leg.shown) > 30.0
                else MAX_RATE_DEG_S)
        step = rate / FPS
        leg.shown += float(np.clip(leg.phi - leg.shown, -step, step))
        leg.shown = float(np.clip(leg.shown, elevation + KNEE_MIN, elevation + KNEE_MAX))

    def _label_phase(self, leg, elevation) -> None:
        if leg.shown is None or leg.confidence < APPLY_MIN_CONFIDENCE:
            leg.phase = "unknown"
            return
        moving = leg.segment is not None
        previous = leg.phase
        if moving and leg.segment.get("announced") == "start_extend":
            leg.phase = "extending"
        elif moving and leg.segment.get("announced") == "start_return":
            leg.phase = "returning"
        elif previous in ("extended", "extending"):
            leg.phase = "extended" if leg.shown < EXTENDED_EXIT else ("flexed" if leg.shown > FLEXED_ENTER else "between")
        elif previous in ("flexed", "returning"):
            leg.phase = "flexed" if leg.shown > FLEXED_EXIT else ("extended" if leg.shown < EXTENDED_ENTER else "between")
        else:
            leg.phase = ("extended" if leg.shown < EXTENDED_ENTER else
                         "flexed" if leg.shown > FLEXED_ENTER else "between")

    def state_for_frame(self, lag_frames: int):
        """Per-leg (angle, confidence, phase, reason) for the frame an output lagging `lag_frames` represents."""
        target = self.frame - lag_frames
        for frame, snapshot in reversed(self._per_frame):
            if frame <= target:
                return frame, snapshot
        return None, None

    def apply(self, local, lag_frames: int, waist_rotation=None):
        """Rotate each thigh (in its sagittal plane) toward the estimated angle, weighted by confidence.

        local: (24,3,3) SMPL local pose (torch or numpy).  The shank's global rotation is kept,
        so the knee absorbs the change.  Returns (pose, diagnostics dict)."""
        frame, snapshot = self.state_for_frame(lag_frames)
        is_torch = isinstance(local, torch.Tensor)
        pose = local.detach().cpu().numpy().astype(np.float64) if is_torch else np.asarray(local, np.float64)
        g = _globals(pose)
        waist = g[3] if waist_rotation is None else np.asarray(waist_rotation, np.float64)
        a_axis, b_axis, lateral = -waist[:, 1], waist[:, 2], waist[:, 0]
        out = pose.copy()
        diag = {"frame": frame, "resetReason": self.reset_reason}
        for label, (hip, knee) in LEGS.items():
            item = {"applied": 0.0, "phase": "unknown", "confidence": 0.0, "phi": None, "reason": "no state"}
            if snapshot is not None:
                phi, conf, phase, reason, weight = snapshot[label]
                item.update(phi=None if phi is None else round(phi, 1), confidence=round(conf, 3), phase=phase, reason=reason)
                thigh = g[hip] @ np.array([0.0, -1.0, 0.0])
                item["networkPhi"] = round(_angle_of(thigh, a_axis, b_axis), 1)
                if phi is not None and weight > 1e-3:
                    side = float(np.clip(thigh @ lateral, -0.3, 0.3))
                    target = _unit(_u(phi, a_axis, b_axis) * math.sqrt(1.0 - side * side) + side * lateral)
                    swing = _scaled(_rotation_between(thigh, target), weight)
                    new_thigh = swing @ g[hip]
                    out[hip] = g[PARENTS[hip]].T @ new_thigh
                    out[knee] = new_thigh.T @ g[knee]
                    item["applied"] = round(weight, 3)
            diag[SIDE_NAMES[label]] = item
        if is_torch:
            out = torch.from_numpy(out).to(local)
        return out, diag

    def recent_events(self):
        return {SIDE_NAMES[label]: list(leg.events) for label, leg in self.legs.items()}

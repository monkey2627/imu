"""Per-frame and per-output processing shared by the realtime loop (run.py) and the
offline replay (deadbug_replay.py), so an offline reproduction runs the same chain.

Per 60 Hz frame (``prepare_frame``)::

    SolarXR sample -> [optional corrected acceleration] -> session.prepare_feature
      -> PoseCalibrator.correct_acceleration / correct -> model input offsets -> feature
      -> dead-bug leg estimator observes (calibrated orientation, corrected acceleration)

Per network output (``postprocess_output``)::

    raw output -> sensed-bone anchoring -> straight-arm prior -> dead-bug leg estimate
"""
from __future__ import annotations

import numpy as np
import torch

from production.accel_frame import AccelerationFrameFix
from production.arm_prior import straighten_reaching_arms
from production.bridge_contact import BRIDGE_MODELS, BridgeContact, apply_trunk_bias
from production.curlup_contact import CURLUP_MODELS, CurlUpContact
from production.deadbug_floor import DeadBugFloorContact
from production.deadbug_legs import DeadBugLegEstimator
from production.lyingcat_contact import LyingCatContact
from production.sidebridge_contact import SideBridgeContact
from production.start_pose_constraint import StartPoseConstraint
from production.model_backends import apply_input_offsets

DEADBUG_MODELS = frozenset({"waist_imudino_stream_deadbug_limbmix", "waist_imudino_stream_deadbug_rehabv5",
                            "waist_imudino_stream_deadbug_rehabv5_std_20261118",
                            "waist_imudino_stream_deadbug_simreal"})
# Actions whose output gets the side-bridge support rule (sidebridge_contact.py); no other action.
SIDEBRIDGE_ACTIONS = {"sidebridge_left": "left", "sidebridge_right": "right"}
# Models whose thighs are re-estimated from shank translation (deadbug_legs.py) in leg mode "auto".
# The RehabV5-std head predicts the dead-bug legs itself; on the real recordings 045157/063451 the
# estimator folded its tabletop knees/hips to ~140 deg, so it is excluded (2026-11-19, user choice).
DEADBUG_LEG_ESTIMATOR_MODELS = DEADBUG_MODELS - {"waist_imudino_stream_deadbug_rehabv5_std_20261118"}


class LegChain:
    """Acceleration-frame fix + dead-bug leg estimator, with explicit enable/reset."""

    def __init__(self, labels, mode: str = "auto") -> None:
        if mode not in ("auto", "on", "off"):
            raise ValueError(f"dead-bug leg mode must be auto/on/off, got {mode!r}")
        self.labels = list(labels)
        self.mode = mode
        self.fix = AccelerationFrameFix()
        self.estimator = DeadBugLegEstimator(self.labels)
        self.bridge = BridgeContact()
        self.curlup = CurlUpContact()
        self.lyingcat = LyingCatContact()
        self.deadbug_floor = DeadBugFloorContact()
        self.sidebridge = {side: SideBridgeContact(side) for side in ("left", "right")}
        self.start_pose = StartPoseConstraint(self.labels)
        self._reported = {}

    def enabled_for(self, model_id: str) -> bool:
        return self.mode == "on" or (self.mode == "auto" and model_id in DEADBUG_LEG_ESTIMATOR_MODELS)

    def reset(self, reason: str) -> None:
        self.fix.reset()
        self.estimator.reset(reason)
        self.bridge.reset(reason)
        self.curlup.reset(reason)
        self.lyingcat.reset(reason)
        self.deadbug_floor.reset(reason)
        for rule in self.sidebridge.values():
            rule.reset(reason)
        self.start_pose.clear()
        self._reported = {}

    def activate_start_pose(self, action: str, start_pose: str) -> None:
        self.start_pose.activate(action, start_pose)

    def corrected_server_acceleration(self, samples, keys, fallback) -> np.ndarray:
        """(n,3) server-frame linear acceleration with the sensor-frame fix (accel_frame.py)."""
        out = np.asarray(fallback, dtype=np.float32).copy()
        for i, key in enumerate(keys):
            value = self.fix.correct_sample(samples[key])
            if value is not None:
                out[i] = value
        return out

    def new_events(self):
        """Estimator events not yet reported (for diagnostics)."""
        fresh = {}
        for side, events in self.estimator.recent_events().items():
            seen = self._reported.get(side, -1)
            items = [e for e in events if e["frame"] > seen]
            if items:
                fresh[side] = items
                self._reported[side] = items[-1]["frame"]
        return fresh


def prepare_frame(session, calibrator, model_id, labels, quats, accel, samples, keys,
                  legs: LegChain, model_accel_fix: bool):
    """One 60 Hz frame -> (feature, model orientation, aligned acceleration, calibrated orientation).

    The leg estimator observes only while enabled for ``model_id``; the acceleration fix keeps its
    per-tracker state either way."""
    corrected = legs.corrected_server_acceleration(samples, keys, accel)
    model_accel = corrected if model_accel_fix else accel
    feature, calibrated_ori, aligned_accel = session.prepare_feature(quats, model_accel)
    gravity_quats = np.stack([samples[key].raw_quat_xyzw for key in keys])
    gravity_accels = np.stack([
        samples[key].raw_acceleration_xyz if samples[key].raw_acceleration_xyz is not None
        else (np.nan, np.nan, np.nan) for key in keys])
    calibrator.record_bird_dog_gravity(gravity_quats, gravity_accels)
    aligned_accel = calibrator.correct_acceleration(aligned_accel)
    base_ori = calibrator.correct(calibrated_ori)
    bridged = bridge_waist(model_id, labels, base_ori, calibrator)
    if bridged is not base_ori:
        # Anchoring / arm prior read the corrected history: give them the lie-flat waist too.
        calibrator._corrected[-1] = bridged.numpy().copy()
    model_ori = apply_input_offsets(model_id, labels, bridged)
    feature = feature.copy()
    feature[:aligned_accel.numel()] = aligned_accel.reshape(-1).numpy() / 30.0
    feature[-model_ori.numel():] = model_ori.reshape(-1).numpy()   # orientations follow the 15 accel values
    if legs.enabled_for(model_id):
        leg_accel = calibrator.correct_acceleration(session.heading.apply(corrected))
        legs.estimator.observe(base_ori.numpy(), leg_accel.numpy())
    return feature, model_ori, aligned_accel, base_ori


def bridge_waist(model_id, labels, base_ori, calibrator):
    """Bridge models only: apply the lie-flat waist correction (returns base_ori itself otherwise)."""
    waist = getattr(calibrator, "bridge_waist", None)
    if (model_id not in BRIDGE_MODELS or waist is None or not calibrator.calibration_enabled
            or "waist" not in labels):
        return base_ori
    out = base_ori.clone()
    i = labels.index("waist")
    out[i] = out[i] @ torch.from_numpy(np.asarray(waist)).to(out)
    return out


def rebuild_feature(feature, model_id, labels, base_ori, calibrator=None):
    """Same frame for another model: re-apply that model's input offsets (model switch)."""
    base = base_ori if isinstance(base_ori, torch.Tensor) else torch.from_numpy(np.asarray(base_ori))
    if calibrator is not None:
        base = bridge_waist(model_id, labels, base, calibrator)
    model_ori = apply_input_offsets(model_id, labels, base)
    feature = np.asarray(feature, dtype=np.float32).copy()
    feature[-model_ori.numel():] = model_ori.reshape(-1).numpy()
    return feature, model_ori


def push_frame(session, feature, model_ori, aligned_accel) -> None:
    if hasattr(session, "push_observation"):
        session.push_observation(feature, model_ori, aligned_accel)
    else:
        session.push_feature(feature)


def postprocess_output(output, calibrator, lag, labels, arm_prior_enabled, model_id, legs: LegChain | None,
                       dt: float = 2 / 60, action_id: str | None = None):
    """raw network output -> anchored -> arm prior -> dead-bug legs / bridge contacts.
    Returns (output, armWeights, legDiag); bridge diagnostics are in legDiag["bridge"]."""
    output = calibrator.anchor(output, lag)
    measured = calibrator.measured_frame(lag)
    arm_weights = [0.0, 0.0]
    if arm_prior_enabled and model_id not in CURLUP_MODELS and action_id != "lyingcat":
        forearms = None
        if measured is not None and "lw" in labels and "rw" in labels:
            forearms = (measured[labels.index("lw")], measured[labels.index("rw")])
        output, arm_weights = straighten_reaching_arms(
            output, forearms, return_weights=True,
            include_chest_reach=(model_id in DEADBUG_MODELS or action_id == "deadbug"))
    leg_diag = {"enabled": False}
    if legs is not None and legs.enabled_for(model_id):
        waist = None if measured is None or "waist" not in labels else measured[labels.index("waist")]
        output, leg_diag = legs.estimator.apply(output, lag, waist_rotation=waist)
        leg_diag["enabled"] = True
        events = legs.new_events()
        if events:
            leg_diag["events"] = events
    if legs is not None:
        output, start_pose_diag = legs.start_pose.apply(
            output, measured, dt_seconds=dt)
        leg_diag["startPoseConstraint"] = start_pose_diag
    if action_id == "deadbug" and legs is not None:
        local, floor_diag = legs.deadbug_floor.apply(output.detach().numpy().astype(np.float64), dt)
        output = torch.from_numpy(local).to(output)
        leg_diag["deadbugFloor"] = floor_diag
    if model_id in BRIDGE_MODELS:
        trunk = calibrator.bridge_trunk if calibrator.calibration_enabled else None
        local = apply_trunk_bias(output.detach().numpy().astype(np.float64), trunk)
        bridge_diag = {"trunkBias": trunk is not None,
                       "waistFlat": calibrator.bridge_waist is not None and calibrator.calibration_enabled}
        if legs is not None:
            local, contact = legs.bridge.apply(local, dt)
            bridge_diag.update(contact)
        output = torch.from_numpy(local).to(output)
        leg_diag["bridge"] = bridge_diag
    if model_id in CURLUP_MODELS and legs is not None:
        waist = None if measured is None or "waist" not in labels else measured[labels.index("waist")]
        forearms = (None if measured is None or "lw" not in labels or "rw" not in labels
                    else (measured[labels.index("lw")], measured[labels.index("rw")]))
        local, contact = legs.curlup.apply(output.detach().numpy().astype(np.float64), dt, waist=waist,
                                           forearms=forearms)
        output = torch.from_numpy(local).to(output)
        leg_diag["curlup"] = contact
    if action_id in SIDEBRIDGE_ACTIONS and legs is not None:
        # Side bridge only (sidebridge_contact.py): legs straight, support forearm on the floor.
        local, contact = legs.sidebridge[SIDEBRIDGE_ACTIONS[action_id]].apply(
            output.detach().numpy().astype(np.float64), dt, measured=measured, labels=labels)
        output = torch.from_numpy(local).to(output)
        leg_diag["sidebridge"] = contact
    if action_id == "lyingcat" and legs is not None:
        local, contact = legs.lyingcat.apply(output.detach().numpy().astype(np.float64), dt,
                                             measured=measured, labels=labels)
        output = torch.from_numpy(local).to(output)
        leg_diag["lyingcat"] = contact
    return output, arm_weights, leg_diag


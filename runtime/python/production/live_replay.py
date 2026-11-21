"""Reproduce the live run.py output of a Unity input recording, row by row, and check it.

The recording holds, in the order run.py wrote them: every 60 Hz input frame with the calibration
in effect (``tick``), the control/calibration events (``event``) and every live output
(``prediction``: network output and the final broadcast pose).  This replays them in that order
with the same shared functions as run.py (frame_chain.prepare_frame / postprocess_output,
pose_context.replay_start_context), infers exactly where live inferred, and compares.

The model's recurrent state before the recording started is unknown, so outputs only match from
the first ``modelContextRebase`` (the start-pose rebuild after an exercise calibration) on; earlier
predictions are reported separately.  Any later mismatch means offline != live: a bug to fix
before trusting an offline result.

    python -m production.live_replay <recording> [<recording> ...]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

from production.bridge_contact import BRIDGE_MODELS, TRUNK_BIAS_MAX_SPREAD_DEG, load_bridge_calibration
from production.calibration import bone_orientation, identity_heading
from production.frame_chain import LegChain, postprocess_output, prepare_frame, push_frame, rebuild_feature
from production.input_recording import _sample_for, resolve_recording_path
from production.layout import ResolvedLayout
from production.model_backends import (
    ACTION_RUNTIME_MODELS, make_waist_session, model_lag_frames, resolve_waist_model,
)
from production.pose_calibration import PoseCalibrator
from production.pose_context import (DEFAULT_CONTEXT_FRAMES, corrected_orientation, estimate_trunk_bias,
                                     replay_start_context)
from production.exercise_calibration import ExerciseCalibration
from production.reach_heading import ReachHeading
from production.start_pose_fit import fit_calibrator as fit_start_pose
from production.rotmat import UNITY_SMPL_BASIS, batch_smpl_matrix_to_unity_quat_xyzw

MAX_RECENT = 480                 # run.py keeps 480 observations for model switches
START_POSE_SEARCH_FRAMES = 8 * 60  # = run.py START_POSE_SEARCH_FRAMES
LEG_RESET_EVENTS = frozenset({"calibrationSessionStart", "exerciseCalibrationStart", "poseCalibrate",
                              "standCalibrate", "armsForwardCalibrate", "poseHoldCalibrate"})
MATCH_DEG = 0.5                  # float32 / thread-order noise is far below this
RULES_SETTLE_OUTPUTS = 30        # rule filters / sensor history from before the recording: ~1 s to forget


@dataclasses.dataclass
class LiveReplay:
    path: Path
    labels: list
    times: np.ndarray            # (P,) recording time of each live prediction
    models: list                 # (P,) model id
    synced: np.ndarray           # (P,) True once a modelContextRebase has been replayed
    raw_local: np.ndarray        # (P,24,3,3) replayed network output
    final_local: np.ndarray      # (P,24,3,3) replayed final output
    raw_error_deg: np.ndarray    # (P,) max joint angle vs the live network output
    final_error_deg: np.ndarray  # (P,) max joint angle vs the live broadcast pose
    rules_error_deg: np.ndarray  # (P,) postprocess of the LIVE network output vs the live broadcast pose
    rules_known: np.ndarray      # (P,) rule state known: after a reset/rebase or the first second
    rebases: list                # (time, action, start pose)
    reach_events: list = dataclasses.field(default_factory=list)   # simulate: (time, label, error deg)

    def summary(self) -> str:
        lines = [f"{self.path.name}: {len(self.times)} live outputs, rebases {self.rebases or 'none'}"]
        rules = self.rules_error_deg[self.rules_known]
        lines.append(f"  rules/calibration (live network output -> offline postprocess): max {rules.max():.3f} deg"
                     f"  p99 {np.percentile(rules, 99):.3f}  >{MATCH_DEG} deg: {(rules > MATCH_DEG).sum()}")
        if not self.synced.any():
            lines.append("  no live output after a modelContextRebase: model state unknown, model parity not checked")
        for name, mask in (("after rebase", self.synced), ("before rebase", ~self.synced)):
            mask = mask & np.isfinite(self.final_error_deg)
            if mask.any():
                raw, final = self.raw_error_deg[mask], self.final_error_deg[mask]
                lines.append(f"  {name}: n={mask.sum()}  network max {raw.max():.3f} deg  final max {final.max():.3f}"
                             f" deg  p99 {np.percentile(final, 99):.3f}  >{MATCH_DEG} deg: {(final > MATCH_DEG).sum()}")
        return "\n".join(lines)

    @property
    def parity_ok(self) -> bool:
        """Rules always; model too where the live model state is known (after a rebase)."""
        return bool((self.rules_error_deg[self.rules_known] <= MATCH_DEG).all()
                    and (self.final_error_deg[self.synced] <= MATCH_DEG).all()
                    and (self.raw_error_deg[self.synced] <= MATCH_DEG).all())


def _quat_angle_deg(a, b):
    """Max over joints of the rotation angle between two (24,4) quaternion sets."""
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    dot = np.clip(np.abs((a * b).sum(1)), 0.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(dot)).max())


def _unity_quats_to_smpl(quats):
    """(24,4) Unity quaternions (as broadcast/recorded) -> (24,3,3) SMPL local rotations."""
    from scipy.spatial.transform import Rotation
    matrices = Rotation.from_quat(np.asarray(quats, np.float64).reshape(-1, 4)).as_matrix()
    return (UNITY_SMPL_BASIS @ matrices @ UNITY_SMPL_BASIS).astype(np.float32)


def _set_calibration(calibrator, state):
    calibrator.headings = np.asarray(state["headings"], dtype=np.float64)
    calibrator.corrections = np.asarray(state["corrections"], dtype=np.float64)
    calibrator.calibration_enabled = state["calibrationEnabled"]
    calibrator.anchor_enabled = state["anchorEnabled"]
    load_bridge_calibration(calibrator, state)


def replay_live(path, leg_mode: str = "auto", simulate: bool = False) -> LiveReplay:
    """simulate=False: reproduce what live did (the recorded calibration and events drive everything).
    simulate=True: what the CURRENT live code would have done with this input -- the live start-pose
    fit (start_pose_fit.fit_calibrator) at Unity's bird-dog start-hold command, the rebase and the
    reach alignment run here exactly as in run.py, instead of reading their results from the file.
    Live processes such a command after the next input update; here it runs at the next non-input
    row, at most a few frames later."""
    path = resolve_recording_path(str(path)) if not Path(path).exists() else Path(path)
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    header = rows[0]
    labels = list(header["labels"])
    resolved = ResolvedLayout(header["layout"], labels, dict(header["labelToTrackerKey"]))
    heading = identity_heading(labels)
    acceleration_source = header.get("accelerationSource", "server")
    first = next((r for r in rows if r.get("type") == "prediction"), None)
    model_accel_fix = bool(first and first["diagnostics"].get("modelAccelFrameFix"))

    calibrator = PoseCalibrator(labels)
    initial = header.get("initialCalibration") or {}
    if "headings" in initial:
        _set_calibration(calibrator, {"calibrationEnabled": True, "anchorEnabled": False, **initial})
    arm_prior = initial.get("armPriorEnabled", True)
    def _set_tick_state(snapshot):
        """Apply the recorded calibration in effect (in simulate mode, once the simulated live logic owns
        the limb corrections, only the switches); returns the arm-prior switch."""
        if owned:
            kept = (calibrator.headings.copy(), calibrator.corrections.copy(),
                    calibrator.bridge_waist, calibrator.bridge_trunk)
            _set_calibration(calibrator, snapshot)
            (calibrator.headings, calibrator.corrections,
             calibrator.bridge_waist, calibrator.bridge_trunk) = kept
        else:
            _set_calibration(calibrator, snapshot)
        return snapshot["armPriorEnabled"]

    legs = LegChain(labels, leg_mode)
    # Second postprocess state machine, fed the LIVE network output: checks the rules alone.
    rules_legs = LegChain(labels, leg_mode)
    model = header.get("model")
    state = {"session": make_waist_session(model, resolved, heading, loaded=resolve_waist_model(model)),
             "model": model}
    recent = []
    action_id = (header.get("actionState") or {}).get("action")
    # run.py reset_action_state: a recording opened right after an action reset starts on a fresh model
    # session, empty sensor history and fresh rules -- everything is known from the first row on.
    synced = bool(header.get("modelSessionFresh"))
    for chain in (legs, rules_legs):
        chain.start_pose.restore(initial.get("startPoseConstraint"))
    out = {k: [] for k in ("times", "models", "synced", "raw", "final", "raw_err", "final_err", "rules_err",
                           "rules_known")}
    rebases = []
    reach = ReachHeading(labels)
    reach_log = []
    owned = False                # simulate: headings/corrections now come from the simulated live logic
    pending_fit = None           # simulate: (action, start pose, time) of Unity's start-hold command
    exercise_start = None        # simulate: (action, start pose) of the running exercise calibration
    fresh_frames = 0             # input frames since that command

    def switch(new_model):
        session = make_waist_session(new_model, resolved, heading, loaded=resolve_waist_model(new_model))
        for feature, raw_ori, accel in recent:
            base = corrected_orientation(raw_ori, calibrator)
            new_feature, new_ori = rebuild_feature(feature, new_model, labels, base, calibrator)
            push_frame(session, new_feature, new_ori, torch.from_numpy(accel))
        if hasattr(session, "finish_replay"):
            session.finish_replay()
        state.update(session=session, model=new_model)
        legs.reset(f"model switch -> {new_model}")
        rules_legs.reset(f"model switch -> {new_model}")

    def rebase(action, start_pose, frames, t):
        nonlocal action_id
        calibrator.refresh_corrected_history()
        context = recent[-int(frames):]
        if (simulate and action == "hipthrust" and state["model"] in BRIDGE_MODELS
                and calibrator.bridge_waist is not None):        # as run.py at the rebase
            scratch = make_waist_session(state["model"], resolved, heading, loaded=resolve_waist_model(state["model"]))
            bias, spread = estimate_trunk_bias(scratch, context, state["model"], labels, calibrator)
            print(f"  simulated trunk bias spread {spread:.1f} deg", flush=True)
            if bias is not None and spread <= TRUNK_BIAS_MAX_SPREAD_DEG:
                calibrator.bridge_trunk = bias
        session = make_waist_session(state["model"], resolved, heading, loaded=resolve_waist_model(state["model"]))
        replay_start_context(session, context, state["model"], labels, calibrator, frames=len(context))
        state["session"] = session
        recent[:] = context
        for chain in (legs, rules_legs):
            chain.reset("rebase")
            chain.activate_start_pose(action, start_pose)
        reach.activate(action)
        action_id = action
        rebases.append((round(t, 2), action, start_pose))

    for row in rows[1:]:
        kind = row.get("type")
        if pending_fit is not None and kind != "tick" and fresh_frames > 0:
            fit_action, fit_pose, fit_t = pending_fit
            pending_fit = None
            passed, message = fit_start_pose(calibrator, np.asarray(calibrator._history)[-START_POSE_SEARCH_FRAMES:],
                                             fit_action, fit_pose)
            owned = owned or passed
            print(f"  simulated {message}", flush=True)
            if passed:
                rebase(fit_action, fit_pose, DEFAULT_CONTEXT_FRAMES, fit_t)
                synced = True
        if kind == "tick":
            samples = {label: _sample_for(row["samples"][label], label) for label in labels}
            quats = np.asarray([samples[label].quat_xyzw for label in labels], dtype=np.float32)
            accel = np.asarray([samples[label].reference_adjusted_accel_xyz if acceleration_source == "server"
                                else samples[label].accel_xyz for label in labels], dtype=np.float32)
            for _ in range(max(1, int(row.get("ticks", 1)))):
                if row.get("calibration"):
                    arm_prior = _set_tick_state(row["calibration"])
                feature, model_ori, aligned_accel, _ = prepare_frame(
                    state["session"], calibrator, state["model"], labels, quats, accel, samples, labels,
                    legs, model_accel_fix)
                recent.append((feature.copy(), bone_orientation(quats, labels).numpy(), aligned_accel.numpy()))
                push_frame(state["session"], feature, model_ori, aligned_accel)
                if simulate:
                    before = len(reach.events)
                    reach.update(calibrator, calibrator._history[-1])
                    for event in reach.events[before:]:
                        reach_log.append((round(row["t"], 2), event["label"], event["errorDeg"]))
                fresh_frames += 1
            del recent[:-MAX_RECENT]
        elif kind == "event":
            name = row.get("name")
            if name in LEG_RESET_EVENTS:
                legs.reset(name)
                rules_legs.reset(name)
            if name == "exerciseCalibrationStart":
                action_id = row.get("action")
            elif name == "birdDogStartCalibrate":
                action_id = "birddog"
            if simulate and name in ("birdDogStartCalibrate", "exerciseCalibrationStart"):
                reach.activate(None)
            if simulate and name == "exerciseCalibrationStart":
                # as run.py: start() clears the hold history, the capture searches what follows
                exercise_start = (row.get("action"), row.get("startPose"))
                sequence = ExerciseCalibration(calibrator, simple_start_only=True)
                sequence.start(*exercise_start)
            if simulate and name == "birdDogBaseCalibrate":
                pending_fit, fresh_frames = ("birddog", "quadruped", row["t"]), 0
            elif simulate and name == "exercisePoseCalibrate" and exercise_start is not None:
                # run.py: ExerciseCalibration.advance -> capture at the end of the hold, same update.
                passed, message = sequence.capture(row["pose"], row["seconds"])
                if (passed and exercise_start[0] == "lyingcat"
                        and row.get("pose") == "cat_neutral"
                        and "start pose fit" not in message):
                    # Keep the replay identical to run.py: cat_neutral is the
                    # lying-cat mounting-calibration hold, and its fitted corrections
                    # must be committed before rebuilding the model context.
                    passed, message = fit_start_pose(
                        calibrator,
                        np.asarray(calibrator._history)[-START_POSE_SEARCH_FRAMES:],
                        exercise_start[0],
                        exercise_start[1],
                    )
                print(f"  simulated {message}", flush=True)
                if passed:
                    owned = True
                    rebase(exercise_start[0], exercise_start[1], DEFAULT_CONTEXT_FRAMES, row["t"])
                    synced = True
            elif name == "trackerBindingChanged" and row.get("resolved"):
                calibrator.reset()
                legs.reset(name)
                rules_legs.reset(name)
                recent.clear()
                switch(state["model"])
                synced = False
            elif (name == "actionLifecycle" and row.get("modelSessionReset")
                  and row.get("phase") in ("actionSwitch", "fullReset")
                  and row.get("resetReason") not in ("startReplay", "stopReplay", "trackerBindingChanged")):
                # run.py reset_action_state with the recording open: fresh session of the action's model.
                calibrator.clear_history()
                recent.clear()
                for chain in (legs, rules_legs):
                    chain.reset(row.get("resetReason") or "action reset")
                reach.reset()
                state.update(session=make_waist_session(row["model"], resolved, heading,
                                                        loaded=resolve_waist_model(row["model"])),
                             model=row["model"])
                action_id = row.get("action")
                synced = True
            elif name == "modelContextRebase" and not simulate:     # simulate decides its own rebases
                if "headings" in row and not owned:
                    calibrator.headings = np.asarray(row["headings"], dtype=np.float64)
                    calibrator.corrections = np.asarray(row["corrections"], dtype=np.float64)
                load_bridge_calibration(calibrator, row)
                rebase(row.get("action"), row.get("startPose"), row.get("frames", DEFAULT_CONTEXT_FRAMES), row["t"])
                synced = True
        elif kind == "prediction":
            diag = row["diagnostics"]
            if diag["model"] != state["model"]:
                switch(diag["model"])
            if row.get("calibration"):
                arm_prior = _set_tick_state(row["calibration"])
            current_action = diag.get("actionId", action_id) or (diag.get("exerciseCalibration") or {}).get("action")
            live_raw = torch.from_numpy(_unity_quats_to_smpl(row["rawLocalQuaternions"]))
            try:
                with torch.no_grad():
                    raw = state["session"].infer()
                modelled = True
            except RuntimeError:
                # Before the first rebase the replayed model may lack the live pre-recording history.
                if synced:
                    raise
                raw, modelled = live_raw.clone(), False
            raw_quats = batch_smpl_matrix_to_unity_quat_xyzw(raw.detach().numpy())
            final, _arm, _legs = postprocess_output(
                raw, calibrator, model_lag_frames(state["model"]), labels, arm_prior, state["model"], legs,
                dt=int(diag.get("inferEvery", 2)) / 60.0, action_id=current_action)
            final_quats = batch_smpl_matrix_to_unity_quat_xyzw(final.detach().numpy())
            rules_final, _arm, _legs = postprocess_output(
                live_raw, calibrator, model_lag_frames(state["model"]), labels, arm_prior, state["model"],
                rules_legs, dt=int(diag.get("inferEvery", 2)) / 60.0, action_id=current_action)
            out["rules_err"].append(_quat_angle_deg(batch_smpl_matrix_to_unity_quat_xyzw(
                rules_final.detach().numpy()), row["unityLocalQuaternions"]))
            # The first outputs also read sensor history from before the recording started (lagged
            # anchoring / arm prior), so they are never comparable, reset or not.
            out["rules_known"].append(synced or len(out["times"]) >= RULES_SETTLE_OUTPUTS)
            out["times"].append(row["t"])
            out["models"].append(state["model"])
            out["synced"].append(synced)
            out["raw"].append(raw.detach().numpy().astype(np.float32))
            out["final"].append(final.detach().numpy().astype(np.float32))
            out["raw_err"].append(_quat_angle_deg(raw_quats, row["rawLocalQuaternions"]) if modelled else np.nan)
            out["final_err"].append(_quat_angle_deg(final_quats, row["unityLocalQuaternions"]) if modelled
                                    else np.nan)
    return LiveReplay(
        path=Path(path), labels=labels, times=np.asarray(out["times"]), models=out["models"],
        synced=np.asarray(out["synced"], bool), raw_local=np.asarray(out["raw"]), final_local=np.asarray(out["final"]),
        raw_error_deg=np.asarray(out["raw_err"]), final_error_deg=np.asarray(out["final_err"]),
        rules_error_deg=np.asarray(out["rules_err"]), rules_known=np.asarray(out["rules_known"], bool),
        rebases=rebases, reach_events=reach_log)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", nargs="+")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    failed = False
    for recording in args.recordings:
        result = replay_live(recording)
        print(result.summary(), flush=True)
        failed |= not result.parity_ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

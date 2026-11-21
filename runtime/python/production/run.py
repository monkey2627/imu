"""End-to-end production bridge: SlimeVR-Server (SolarXR) -> selected model
backend -> local WebSocket broadcast for the Unity receiver script.

It subscribes to the SolarXR data feed as an ordinary client. The local
SlimeVR build can optionally expose reference-adjusted linear acceleration;
the legacy field and SlimeVR's own VMC/OSC output remain unchanged.

The realtime path uses SlimeVR's reference-adjusted rotation and the local
SlimeVR build's reference-adjusted linear acceleration. By default startup does
not capture a neutral standing window: Unity starts the action-specific pose
calibration and the completed start pose rebuilds the model history. The old
upright priming step remains available as an explicit diagnostic option.

Model/placement (since 2026-10-19): the SparseIMU baseline ``pnp5_60a`` -- 60 fps
data, five IMUs on PNP surface vertices (forearms near the wrist on the dorsal
side, shanks just below the knee on the inner front, pelvis on the sacrum),
online window 91 frames (80 past + 10 future, 167 ms lag). The window advances
at a fixed 60 Hz; the network runs every ``--infer-every`` frames (default 2,
i.e. 30 poses/s).

Usage (from repo root), with SlimeVR-Server running, trackers bound
(LEFT/RIGHT_LOWER_ARM, LEFT/RIGHT_LOWER_LEG, HIP) and a converted checkpoint
(scripts/convert_sparseimu_checkpoint.py) in checkpoints/<model>/::

    python -m production.run --model pnp5_60a_s42_ftr3

NOTE: this resolves the layout once at startup and does not hot-swap if
trackers connect/disconnect mid-session -- restart the script if your tracker
set changes.
"""
from __future__ import annotations

import argparse
import asyncio
import atexit
import os
import queue
import subprocess
import threading
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import websockets
import yaml

from production.calibration import bone_orientation, identity_heading
from production.model_backends import (
    apply_input_offsets,
    ACTION_RUNTIME_MODELS, DEADBUG_RUNTIME_MODEL, WAIST_CHECKPOINT_ROOT, WAIST_MODEL_FILES, WAIST_MODEL_LABELS,
    make_waist_session, resolve_waist_model, model_lag_frames,
)
from runtime_contract import FPS as MODEL_FPS, OUTPUT_INDEX, WINDOW
from production.inference import (
    DEFAULT_CHECKPOINT_ROOT, DEFAULT_MODEL, OUTPUT_LAG_FRAMES, InferenceSession, load_model,
)
from production.layout import LayoutMismatchError, ResolvedLayout, detect_layout
from production.rotmat import batch_smpl_matrix_to_unity_quat_xyzw, quat_xyzw_to_unity_quat_xyzw
from production.solarxr_client import TrackerSample, request_reset, stream_trackers
from production.pose_calibration import PoseCalibrator
from production.frame_chain import LegChain, postprocess_output, prepare_frame, push_frame, rebuild_feature
from production.bridge_contact import (BRIDGE_MODELS, TRUNK_BIAS_MAX_SPREAD_DEG, bridge_calibration_state,
                                       load_bridge_calibration)
from production.layout import BODY_PART_TO_LABEL
from production.stream_out import PoseBroadcaster
from production.web_visualizer import start_web_visualizer
from production.recording import RealtimeRecorder
from production.input_recording import (
    InputMux, InputRecorder, load_input_recording, recording_path_for, resolve_recording_path,
)
from production.pose_context import (DEFAULT_CONTEXT_FRAMES, corrected_orientation, estimate_trunk_bias,
                                     replay_start_context)

MAX_CATCHUP_TICKS = 4   # missed 60 Hz ticks that are re-filled (hold-last) before resyncing the clock
MAX_TRACKER_STALENESS_SECONDS = 0.15
DEFAULT_RUNTIME_MODEL = "waist_mobileposer_amass"
SOURCE_SWITCH_WARMUP_FRAMES = 12   # StreamPoser needs >=2 frames, Line14 reads frame -10
START_POSE_SEARCH_FRAMES = 8 * 60  # Unity's bird-dog start hold is 8 s
CODE_CHECK_SECONDS = 5.0


def _code_stamp() -> float:
    """Newest modification time of the realtime code (production/*.py)."""
    return max(p.stat().st_mtime for p in Path(__file__).resolve().parent.glob("*.py"))


def _slerp_xyzw(left, right, amount):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left /= max(np.linalg.norm(left), 1e-12)
    right /= max(np.linalg.norm(right), 1e-12)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = left + amount * (right - left)
        return (result / max(np.linalg.norm(result), 1e-12)).astype(np.float32)
    angle = np.arccos(dot)
    scale = np.sin(angle)
    return (np.sin((1.0 - amount) * angle) / scale * left
            + np.sin(amount * angle) / scale * right).astype(np.float32)


def _interpolate_snapshot(history, target_time, keys, acceleration_source):
    """Resample each tracker's own update timeline onto the fixed model clock."""
    samples = {}
    quats, accelerations = [], []
    vector_fields = ("raw_quat_xyzw", "quat_xyzw", "accel_xyz",
                     "reference_adjusted_accel_xyz", "raw_acceleration_xyz",
                     "raw_angular_velocity_xyz")
    for key in keys:
        timeline = history[key]
        if target_time <= timeline[0][0]:
            left = right = timeline[0]
        elif target_time >= timeline[-1][0]:
            left = right = timeline[-1]
        else:
            right_index = next(i for i, row in enumerate(timeline) if row[0] >= target_time)
            left, right = timeline[right_index - 1], timeline[right_index]
        amount = 0.0 if left is right else (target_time - left[0]) / (right[0] - left[0])
        a, b = left[1], right[1]
        values = {}
        for name in vector_fields:
            av, bv = getattr(a, name), getattr(b, name)
            if av is not None and bv is not None:
                if "quat" in name:
                    values[name] = tuple(float(v) for v in _slerp_xyzw(av, bv, amount))
                else:
                    values[name] = tuple((np.asarray(av) * (1.0 - amount)
                                          + np.asarray(bv) * amount).tolist())
            else:
                values[name] = bv if bv is not None else av
        samples[key] = dataclasses.replace(b, **values)
        quats.append(values["quat_xyzw"])
        accelerations.append(_sample_acceleration(samples[key], acceleration_source))
    return (np.asarray(quats, dtype=np.float32),
            np.asarray(accelerations, dtype=np.float32), samples)


class CompareJobs:
    """Build a standalone SMPL surface replay for each saved recording in the background."""

    def __init__(self) -> None:
        self._jobs: queue.Queue = queue.Queue()
        self._busy = False
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def idle(self) -> bool:
        return not self._busy and self._jobs.empty()

    def submit(self, recording: Path) -> None:
        self._jobs.put(Path(recording))

    def _run(self) -> None:
        from production.paths import paths
        flags = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
        while True:
            recording = self._jobs.get()
            self._busy = True
            log = recording.with_name(recording.stem + "_smpl.log")
            try:
                with log.open("w", encoding="utf-8") as output:
                    code = subprocess.call(
                        [sys.executable, "-m", "production.mesh_replay", str(recording)],
                        stdout=output, stderr=subprocess.STDOUT, cwd=str(paths.root_dir), creationflags=flags)
                viewer = recording.with_name(recording.stem + "_smpl.html")
                if code == 0 and viewer.exists():
                    print(f"SMPL surface replay ready: {viewer}", file=sys.stderr)
                else:
                    print(f"SMPL surface replay failed for {recording.name} (see {log})", file=sys.stderr)
                # Offline == live check (production/live_replay.py): any offline analysis of this recording
                # is only trustworthy if the replay reproduces the live output.
                parity = recording.with_name(recording.stem + "_parity.log")
                with parity.open("w", encoding="utf-8") as output:
                    code = subprocess.call(
                        [sys.executable, "-m", "production.live_replay", str(recording)],
                        stdout=output, stderr=subprocess.STDOUT, cwd=str(paths.root_dir), creationflags=flags)
                print(f"Offline/live parity {'OK' if code == 0 else 'MISMATCH'} for {recording.name} (see {parity})",
                      file=sys.stderr)
            except OSError as exc:
                print(f"SMPL surface replay could not start: {exc}", file=sys.stderr)
            finally:
                self._busy = False


class FrameClock:
    """Fixed-rate 60 Hz tick scheduler driven by SolarXR arrival times.

    Every call to ``due(now)`` returns how many model frames the window must
    advance (0 if the next tick is not due). Missed ticks are filled with the
    latest sample so the window's time axis stays uniform; a stall longer than
    ``MAX_CATCHUP_TICKS`` resynchronises the clock instead of bursting.
    """

    def __init__(self, fps: float, start: float) -> None:
        self.dt = 1.0 / fps
        self.next_tick = start

    def due(self, now: float) -> int:
        if now < self.next_tick:
            return 0
        ticks = int((now - self.next_tick) / self.dt) + 1
        if ticks > MAX_CATCHUP_TICKS:
            self.next_tick = now + self.dt
            return 1
        self.next_tick += ticks * self.dt
        return ticks


def normalize_checkpoint_path(value: str | Path) -> Path:
    """Normalize ASCII and typographic shell quotes without source encoding issues."""
    quote_chars = "\"'" + "".join(chr(code) for code in (0x2018, 0x2019, 0x201C, 0x201D))
    text = str(value).strip()
    while text and text[0] in quote_chars:
        text = text[1:]
    while text and text[-1] in quote_chars:
        text = text[:-1]
    return Path(text)


def _load_runtime_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as file:
        values = yaml.safe_load(file) or {}
    if not isinstance(values, dict):
        raise ValueError(f"realtime config must be a mapping: {path}")
    return values


async def _resilient_tracker_updates(recorder_holder=None, quiet=lambda: False, feed_interval_ms=16):
    while True:
        try:
            received = False
            async for update in stream_trackers(minimum_time_since_last_ms=feed_interval_ms):
                received = True
                yield update
            if recorder_holder and recorder_holder.get("latest") is not None:
                recorder_holder["latest"].clear()
            if received and not quiet():
                print("SolarXR stream ended; reconnecting in 1s...", file=sys.stderr)
                if recorder_holder and recorder_holder.get("recorder"):
                    recorder_holder["recorder"].event("solarxr_stream_ended")
        except (websockets.ConnectionClosed, OSError, ConnectionError) as exc:
            if not quiet():
                print(
                    f"SolarXR disconnected ({type(exc).__name__}: {exc}); reconnecting in 1s...",
                    file=sys.stderr,
                )
            if recorder_holder and recorder_holder.get("latest") is not None:
                recorder_holder["latest"].clear()
            if recorder_holder and recorder_holder.get("recorder"):
                recorder_holder["recorder"].event("solarxr_disconnected", error=repr(exc))
        await asyncio.sleep(1.0)


async def _wait_for_layout(tracker_stream, latest: Dict[str, TrackerSample]) -> ResolvedLayout:
    print("Waiting for a recognized tracker layout (bind trackers in SlimeVR-Server)...", file=sys.stderr)
    last_error = None
    async for update in tracker_stream:
        latest.update(update)
        try:
            resolved = detect_layout(latest)
        except LayoutMismatchError as exc:
            if str(exc) != last_error:
                print(f"  ... {exc}", file=sys.stderr)
                last_error = str(exc)
            continue
        print(f"Resolved layout: {resolved.name} (labels: {resolved.labels})", file=sys.stderr)
        return resolved
    raise RuntimeError("SolarXR stream ended before a layout was resolved")


async def _collect_heading_window(
    tracker_stream, latest: Dict[str, TrackerSample], resolved: ResolvedLayout, frames: int,
    recorder: RealtimeRecorder | None = None,
):
    """Collect `frames` samples per label while the user moves the trackers
    around (any rotation works -- twisting the torso, swinging limbs, etc.).
    A perfectly still hold will not work: see calibration.py."""
    raw: Dict[str, List] = {label: [] for label in resolved.labels}
    adjusted: Dict[str, List] = {label: [] for label in resolved.labels}
    print(
        f"Calibrating: rotate/move the trackers for the next {frames} frames "
        f"(twist your torso, swing your arms/legs -- holding still will NOT work)...",
        file=sys.stderr,
    )
    async for update in tracker_stream:
        latest.update(update)
        for label in resolved.labels:
            key = resolved.label_to_tracker_key[label]
            sample = latest.get(key)
            if sample is not None and sample.online:
                raw[label].append(sample.raw_quat_xyzw)
                adjusted[label].append(sample.quat_xyzw)
        if all(len(v) >= frames for v in raw.values()):
            break
    raw_arrays = {label: np.asarray(values[:frames], dtype=np.float32) for label, values in raw.items()}
    adjusted_arrays = {label: np.asarray(values[:frames], dtype=np.float32) for label, values in adjusted.items()}
    return raw_arrays, adjusted_arrays


async def _wait_for_sync_calibration(tracker_stream, latest, resolved, recorder=None) -> None:
    """Coordinate SlimeVR's manual upright/ski calibration with this bridge.

    SolarXR does not expose calibration-complete events, so the user confirms
    each SlimeVR step explicitly. The captured windows are diagnostic/reference
    samples; the existing heading fit still follows and remains unchanged.
    """
    print("\nSynchronized calibration mode", file=sys.stderr)
    print("1) In SlimeVR, perform the upright/standing calibration.", file=sys.stderr)
    await asyncio.to_thread(input, "   Press Enter here when it is complete... ")
    upright = await _collect_reference_window(tracker_stream, latest, resolved, 15, recorder)
    print("   Upright reference captured.", file=sys.stderr)
    print("2) In SlimeVR, perform the ski calibration.", file=sys.stderr)
    await asyncio.to_thread(input, "   Press Enter here when it is complete... ")
    ski = await _collect_reference_window(tracker_stream, latest, resolved, 15, recorder)
    # Report only relative motion; absolute quaternion signs are arbitrary.
    labels = resolved.labels
    deltas = np.linalg.norm(upright - ski, axis=2).mean(axis=0)
    print("   Ski reference captured. Mean quaternion delta by tracker: "
          + ", ".join(f"{label}={value:.3f}" for label, value in zip(labels, deltas)),
          file=sys.stderr)


async def _collect_reference_window(tracker_stream, latest, resolved, frames, recorder=None):
    values = []
    async for update in tracker_stream:
        latest.update(update)
        if recorder is not None:
            selected = [latest.get(resolved.label_to_tracker_key[label]) for label in resolved.labels]
            recorder.write("calibrationSample", {
                "hostUnixSeconds": time.time(),
                "hostMonotonicSeconds": time.perf_counter(),
                "trackers": [dataclasses.asdict(sample) for sample in selected if sample is not None],
            })
        row = []
        complete = True
        for label in resolved.labels:
            sample = latest.get(resolved.label_to_tracker_key[label])
            if sample is None or not sample.online:
                complete = False
                break
            row.append(sample.quat_xyzw)
        if complete:
            values.append(np.asarray(row, dtype=np.float32))
        if len(values) >= frames:
            return np.asarray(values[-frames:], dtype=np.float32)
    raise RuntimeError("SolarXR stream ended during synchronized calibration")


async def _collect_upright_calibration(tracker_stream, latest, resolved, frames, settle_seconds, acceleration_source):
    print("\nStand upright in the neutral pose (feet parallel, arms relaxed).", file=sys.stderr)
    settle_seconds = max(0.0, float(settle_seconds))
    if settle_seconds > 0:
        for remaining in range(int(np.ceil(settle_seconds)), 0, -1):
            print(f"  Capturing upright reference in {remaining}...", file=sys.stderr)
            await asyncio.sleep(1.0)
    print("  Capturing upright reference now...", file=sys.stderr)
    rows = []
    async for update in tracker_stream:
        latest.update(update)
        samples = [latest.get(resolved.label_to_tracker_key[label]) for label in resolved.labels]
        if any(sample is None or not sample.online for sample in samples):
            continue
        rows.append((
            np.asarray([sample.quat_xyzw for sample in samples], dtype=np.float32),
            np.asarray([_sample_acceleration(sample, acceleration_source) for sample in samples], dtype=np.float32),
        ))
        if len(rows) >= frames:
            print("Upright reference captured.", file=sys.stderr)
            return rows[-1]
    raise RuntimeError("SolarXR stream ended during upright calibration")


def _resolve_acceleration_source(requested: str, latest, resolved) -> str:
    available = all(
        latest.get(resolved.label_to_tracker_key[label]) is not None
        and latest[resolved.label_to_tracker_key[label]].reference_adjusted_accel_xyz is not None
        for label in resolved.labels
    )
    if requested == "server" and not available:
        raise RuntimeError(
            "--acceleration-source server requested, but the connected SlimeVR build "
            "does not provide reference_adjusted_linear_acceleration for every tracker"
        )
    return "server" if requested == "server" or (requested == "auto" and available) else "python"


def _binding_changed(latest, resolved: ResolvedLayout):
    """Description of the first model slot whose tracker is no longer bound to that body part, else None."""
    for label in resolved.labels:
        sample = latest.get(resolved.label_to_tracker_key[label])
        if sample is None or sample.body_part is None:
            continue
        bound = BODY_PART_TO_LABEL.get(sample.body_part)
        if bound != label:
            return f"{label} slot <- tracker {sample.tracker_key} now bound to {bound or sample.body_part}"
    return None


def _sample_acceleration(sample: TrackerSample, source: str):
    if source == "server":
        if sample.reference_adjusted_accel_xyz is None:
            raise RuntimeError(f"reference-adjusted acceleration disappeared for {sample.tracker_key}")
        return sample.reference_adjusted_accel_xyz
    return sample.accel_xyz


async def main_async(args: argparse.Namespace) -> None:
    # Fail fast on a missing/incompatible checkpoint before waiting for trackers.
    model_spec = normalize_checkpoint_path(args.checkpoint) if args.checkpoint is not None else args.model
    anchor = {"auto": None, "on": True, "off": False}[args.anchor]
    waist_root = args.waist_checkpoint_root
    loaded_waist = resolve_waist_model(args.model, waist_root) if args.checkpoint is None else None
    model = None if loaded_waist else load_model(model_spec, args.checkpoint_root, anchor=anchor)
    selected_model = args.model if loaded_waist else str(model_spec)
    selected_lag = model_lag_frames(selected_model)
    print(f"Model: {selected_model} ({MODEL_FPS} Hz, inference every {args.infer_every} frames "
          f"= {MODEL_FPS / args.infer_every:g} poses/s, output lag {selected_lag} frames "
          f"= {1000 * selected_lag / MODEL_FPS:.0f} ms)", file=sys.stderr)

    broadcaster = PoseBroadcaster()
    await broadcaster.serve(args.host, args.port)
    print(f"Broadcasting SMPL pose on ws://{args.host}:{args.port}", file=sys.stderr)
    if args.web:
        start_web_visualizer(
            args.web_host, args.web_port, args.port,
            model_options=tuple(WAIST_MODEL_LABELS.items()),
        )
        print(f"Realtime viewer: http://{args.web_host}:{args.web_port}/", file=sys.stderr)

    latest: Dict[str, TrackerSample] = {}
    recorder_holder = {"recorder": None, "latest": None}
    mux_holder = {}
    # SolarXR is drained continuously; a replay (Unity button or --replay) replaces it without the IMUs.
    tracker_stream = InputMux(
        _resilient_tracker_updates(recorder_holder, quiet=lambda: mux_holder["mux"].replaying,
                                   feed_interval_ms=args.feed_interval_ms)
    )
    mux_holder["mux"] = tracker_stream
    recorder_holder["latest"] = latest
    startup_replay = None
    if args.replay is not None:
        startup_replay = load_input_recording(resolve_recording_path(args.replay))
        tracker_stream.start_replay(startup_replay, startup_replay.label_to_key)
        print(f"Replaying recorded input (SlimeVR not needed): {startup_replay.path} "
              f"({startup_replay.duration:.1f}s, looping)", file=sys.stderr)

    resolved = await _wait_for_layout(tracker_stream, latest)
    if not loaded_waist and resolved.name != "pnp5_forearm_shank_pelvis":
        raise RuntimeError(
            f"{selected_model} was trained for HIP/pelvis placement, but SlimeVR resolved {resolved.name}. "
            "Start with --model waist_mobileposer_amass for the WAIST-mounted baselines."
        )
    acceleration_source = _resolve_acceleration_source(args.acceleration_source, latest, resolved)
    print(f"Acceleration source: {acceleration_source} (requested: {args.acceleration_source})", file=sys.stderr)

    recorder = None
    if args.record is not None:
        recorder = RealtimeRecorder(args.record, {
            "layout": resolved.name,
            "labels": resolved.labels,
            "labelToTrackerKey": resolved.label_to_tracker_key,
            "checkpoint": selected_model,
            "checkpointRoot": args.checkpoint_root,
            "anchored": model.anchor if model is not None else False,
            "modelFps": MODEL_FPS,
            "windowFrames": 480 if selected_model == "waist_imudino_fiveimu" else WINDOW,
            "outputIndex": 480 - selected_lag if selected_model == "waist_imudino_fiveimu" else OUTPUT_INDEX,
            "inferEvery": args.infer_every,
            "outputFps": MODEL_FPS / args.infer_every,
            "calibrationFrames": args.calibration_frames,
            "outputLagFrames": selected_lag,
            "accelerationSourceRequested": args.acceleration_source,
            "accelerationSourceResolved": acceleration_source,
        })
        recorder_holder["recorder"] = recorder
        atexit.register(recorder.close)
        recorder.event("layout_resolved")

    upright_reference = None
    if startup_replay is not None:
        upright_reference = startup_replay.upright
    elif args.upright_calibration and not args.skip_upright_calibration:
        upright_reference = await _collect_upright_calibration(
            tracker_stream, latest, resolved, args.upright_calibration_frames,
            args.upright_settle_seconds, acceleration_source,
        )
        if recorder is not None:
            recorder.write("upright", {
                "referenceAdjustedQuaternions": upright_reference[0],
                "linearAccelerationMps2": upright_reference[1],
            })

    if args.sync_calibration and startup_replay is None:
        await _wait_for_sync_calibration(tracker_stream, latest, resolved, recorder)

    python_heading = None
    heading = identity_heading(resolved.labels)
    print("Using SlimeVR reference-adjusted acceleration; Python heading fit skipped.", file=sys.stderr)
    if recorder is not None:
        recorder.write("heading", {
            "labels": heading.labels,
            "matrices": heading.heading,
            "diagnostics": heading.diagnostics,
        })

    if loaded_waist:
        if resolved.name != "wrists_shanks_waist":
            raise RuntimeError(f"{selected_model} requires wrists_shanks_waist, got {resolved.name}")
        session = make_waist_session(selected_model, resolved, heading, waist_root, loaded_waist)
    else:
        session = InferenceSession(layout=resolved, model=model, heading=heading)
    if upright_reference is not None:
        session.prime(*upright_reference)
    print("Calibrated. Streaming live pose...", file=sys.stderr)
    # Upright reference of the real wearer; a replay primes with its own recorded one instead.
    live_upright = None if startup_replay is not None else upright_reference
    input_recorder: InputRecorder | None = None
    control_message = ""
    # Per-set mounting correction from the coach's start pose, and sensed-bone anchoring of the output.
    calibrator = PoseCalibrator(resolved.labels)
    # Dead-bug thighs from shank translation (deadbug_legs.py) + the acceleration-frame fix (accel_frame.py).
    leg_chain = LegChain(resolved.labels, args.deadbug_legs)
    from production.exercise_calibration import EXERCISES, ExerciseCalibration
    exercise_calibration = ExerciseCalibration(calibrator, simple_start_only=True)
    from production.reach_heading import ReachHeading
    from production.start_pose_fit import fit_calibrator as fit_start_pose
    # Forearm heading from held reaches (bird dog), after the start-pose fit (start_pose_fit.py).
    reach_heading = ReachHeading(resolved.labels)
    pending_start_pose_fit = None    # (action, start pose): fit at the end of this update, then rebase
    runtime_action = None
    compare_jobs = None if args.no_auto_compare else CompareJobs()
    calibrator.calibration_enabled = not args.no_pose_calibration
    calibrator.anchor_enabled = args.anchor_sensed
    arm_prior_enabled = not args.no_arm_prior   # Unity turns it off for the side bridge (setPoseOptions)
    exercise_arm_prior_backup = None
    # Frames to collect before inferring after an input-source switch, or at startup without an upright prime.
    warmup_frames = 0 if upright_reference is not None else SOURCE_SWITCH_WARMUP_FRAMES

    def calibration_snapshot():
        return {"headings": calibrator.headings.tolist(),
                "corrections": calibrator.corrections.tolist(),
                "calibrationEnabled": calibrator.calibration_enabled,
                "anchorEnabled": calibrator.anchor_enabled, "armPriorEnabled": arm_prior_enabled,
                "startPoseConstraint": leg_chain.start_pose.snapshot(),
                **bridge_calibration_state(calibrator)}

    # ---- action lifecycle -------------------------------------------------------------------------
    # One action owns: the action route (model + postprocess rules), the committed start pose and its
    # constraint, every contact/leg rule state, the model's recurrent/window history and the sensor
    # history the lagged rules read.  All of it is rebuilt at an action switch or a calibration start
    # (reset_action_state).  Only the wearer's mounting calibration (headings/corrections) survives;
    # every start-pose fit overwrites it completely (start_pose_fit.fit_calibrator).
    calibration_session_id = 0
    active_start_pose = None        # start pose of the last successful calibration of runtime_action
    last_reset = None
    lifecycle_log: List[dict] = []
    session_fresh = False           # True while the model session has seen no frame since a reset

    def constraint_state():
        constraint = leg_chain.start_pose
        return {"enabled": constraint.enabled, "action": constraint.action, "startPose": constraint.start_pose,
                "weights": constraint.weights.round(3).tolist(), "armWeights": constraint.arm_weights.round(3).tolist()}

    def action_state():
        return {"action": runtime_action, "startPose": active_start_pose,
                "calibrationSessionId": calibration_session_id,
                "calibrationEnabled": calibrator.calibration_enabled, "anchorEnabled": calibrator.anchor_enabled,
                "startPoseConstraint": constraint_state(), "model": selected_model, "lastReset": last_reset}

    def lifecycle(phase, **extra):
        """Action/calibration/recording lifecycle event: stderr (one JSON line), the open recording and
        the next recording's header."""
        record = {"phase": phase, "action": runtime_action, "startPose": active_start_pose,
                  "calibrationSessionId": calibration_session_id,
                  "calibrationEnabled": calibrator.calibration_enabled, "anchorEnabled": calibrator.anchor_enabled,
                  "startPoseConstraint": constraint_state(), "model": selected_model, **extra}
        lifecycle_log.append({"hostUnixSeconds": time.time(), **record})
        del lifecycle_log[:-20]
        print("lifecycle " + json.dumps(record, ensure_ascii=False, default=str), file=sys.stderr)
        if input_recorder is not None:
            input_recorder.write_event("actionLifecycle", **record)

    def close_recording(reason):
        nonlocal input_recorder, control_message
        if input_recorder is None:
            return
        lifecycle("recordingStop", reason=reason, recordingFile=input_recorder.path.name)
        input_recorder.close()
        control_message = f"saved {input_recorder.path.name} ({input_recorder.seconds:.1f}s)"
        print(f"Saved input recording: {input_recorder.path}", file=sys.stderr)
        if compare_jobs is not None:
            compare_jobs.submit(input_recorder.path)
        input_recorder = None

    def recording_metadata(**extra):
        return {"model": selected_model, "feedIntervalMs": args.feed_interval_ms,
                "torchThreads": torch.get_num_threads(), "sampling": "uniform60_v2",
                "inferEvery": args.infer_every, "initialCalibration": calibration_snapshot(),
                "actionState": action_state(), "modelSessionFresh": session_fresh,
                "lifecycle": list(lifecycle_log), **extra}

    async def reset_action_state(reason, new_action):
        """Clear every action-level state and start `new_action` (None = no action) from a fresh model."""
        nonlocal runtime_action, active_start_pose, pending_start_pose_fit, arm_prior_enabled
        nonlocal exercise_arm_prior_backup, session, selected_model, selected_lag, warmup_frames
        nonlocal since_inference, last_reset, session_fresh
        previous_action, previous_pose = runtime_action, active_start_pose
        had_constraint = leg_chain.start_pose.action is not None
        changed = new_action != previous_action
        if changed:
            # One file per action: the previous action's recording never receives the next action.
            close_recording(f"action switch {previous_action} -> {new_action}")
        if exercise_calibration.active:
            lifecycle("calibrationEnd", outcome="cancelled", resetReason=reason)
        exercise_calibration.clear()        # restores the pre-calibration backup if one was running
        if exercise_arm_prior_backup is not None:
            arm_prior_enabled, exercise_arm_prior_backup = exercise_arm_prior_backup, None
        pending_start_pose_fit = None
        leg_chain.reset(f"action reset: {reason}")
        reach_heading.reset()
        calibrator.clear_history()
        if changed:
            calibrator.bridge_waist = calibrator.bridge_trunk = None    # hip-thrust lie-flat is action-level
        target = ACTION_RUNTIME_MODELS.get(new_action or "") or selected_model
        if target not in WAIST_MODEL_FILES or resolved.name != "wrists_shanks_waist":
            target = selected_model
        if target != selected_model:
            broadcaster.latest_model_command()      # the action route wins over a queued dropdown choice
        previous_model = selected_model
        loaded = (await asyncio.to_thread(resolve_waist_model, target, waist_root)
                  if target in WAIST_MODEL_FILES else None)
        if loaded is not None:
            session = await asyncio.to_thread(make_waist_session, target, resolved, heading, waist_root, loaded)
        else:
            session = await asyncio.to_thread(InferenceSession, layout=resolved, model=model, heading=heading)
        selected_model, selected_lag = target, model_lag_frames(target)
        recent_observations.clear()
        warmup_frames = SOURCE_SWITCH_WARMUP_FRAMES
        since_inference = 0
        session_fresh = True
        runtime_action, active_start_pose = new_action, None
        last_reset = {"reason": reason, "previousAction": previous_action, "previousStartPose": previous_pose,
                      "action": new_action, "previousModel": previous_model, "model": selected_model,
                      "modelSessionReset": True, "historyBufferCleared": True, "hostUnixSeconds": time.time()}
        if had_constraint:
            lifecycle("startPoseClear", resetReason=reason, previousAction=previous_action,
                      previousStartPose=previous_pose)
        if previous_model != selected_model:
            lifecycle("modelSwitch", resetReason=reason, previousModel=previous_model)
        lifecycle("actionSwitch" if changed else "fullReset", resetReason=reason, previousAction=previous_action,
                  previousStartPose=previous_pose, modelSessionReset=True, historyBufferCleared=True)

    def rebuild_session(prime):
        if selected_model in WAIST_MODEL_FILES:
            loaded = resolve_waist_model(selected_model, waist_root)
            new_session = make_waist_session(selected_model, resolved, heading, waist_root, loaded)
        else:
            new_session = InferenceSession(layout=resolved, model=model, heading=heading)
        if prime is not None:
            new_session.prime(*prime)
        return new_session

    def rebuild_context_session(observations):
        new_session = rebuild_session(None)
        replay_start_context(new_session, observations, selected_model, resolved.labels, calibrator)
        return new_session

    async def switch_runtime_model(requested_model: str, source: str) -> bool:
        """Switch a waist model and rebuild its state from the recent sensor history."""
        nonlocal session, selected_model, selected_lag, session_fresh
        if not requested_model or requested_model == selected_model:
            return False
        if requested_model not in WAIST_MODEL_FILES:
            raise ValueError(f"unsupported model {requested_model}")
        if resolved.name != "wrists_shanks_waist":
            raise RuntimeError(
                f"cannot switch to {requested_model}: current SlimeVR layout is {resolved.name}; bind WAIST"
            )
        loaded = await asyncio.to_thread(resolve_waist_model, requested_model, waist_root)
        new_session = await asyncio.to_thread(
            make_waist_session, requested_model, resolved, heading, waist_root, loaded
        )
        for old_feature, old_raw_ori, old_accel in recent_observations:
            old_base_ori = corrected_orientation(old_raw_ori, calibrator)
            new_feature, new_ori = rebuild_feature(
                old_feature, requested_model, resolved.labels, old_base_ori, calibrator)
            push_frame(new_session, new_feature, new_ori, torch.from_numpy(old_accel))
        if hasattr(new_session, "finish_replay"):
            new_session.finish_replay()
        previous_model = selected_model
        had_constraint = leg_chain.start_pose.action is not None
        session_fresh = not recent_observations
        session = new_session
        selected_model = requested_model
        selected_lag = model_lag_frames(selected_model)
        leg_chain.reset(f"model switch -> {selected_model}")
        if had_constraint:
            lifecycle("startPoseClear", resetReason=f"model switch ({source})")
        lifecycle("modelSwitch", resetReason=source, previousModel=previous_model,
                  replayedFrames=len(recent_observations))
        print(f"{source} selected model: {selected_model}", file=sys.stderr)
        return True

    async def apply_control(control):
        nonlocal session, input_recorder
        nonlocal control_message, warmup_frames, arm_prior_enabled
        nonlocal exercise_arm_prior_backup
        nonlocal runtime_action, pending_start_pose_fit, calibration_session_id, session_fresh
        name = control.get("command")
        if exercise_calibration.active and name in (
                "exerciseCalibrationStart",
                "poseCalibrate", "standCalibrate", "armsForwardCalibrate", "birdDogStartCalibrate",
                "birdDogBaseCalibrate", "birdDogLeftReachCalibrate", "birdDogRightReachCalibrate",
                "poseHoldCalibrate", "yawReset", "startReplay", "stopReplay", "calibrationSessionEnd"):
            raise RuntimeError("动作校准进行中，请先取消当前校准")
        if exercise_calibration.active and name == "setPoseOptions" and "calibration" in control:
            raise RuntimeError("请先完成或取消动作校准，再切换校准开关")
        if name == "coachPose":     # 30 Hz while recording; stored, never logged
            if input_recorder is not None and isinstance(control.get("q"), list):
                input_recorder.write_coach(control["q"])
            return False
        if name == "startRecording":
            if tracker_stream.replaying:
                raise RuntimeError("cannot record while replaying; return to live input first")
            if input_recorder is None:
                # A new set of the same action: fresh rule state, and the set begins in the action's
                # calibrated start pose again (the constraint used to be cleared here and never re-armed).
                leg_chain.reset("startRecording")
                if runtime_action and active_start_pose:
                    leg_chain.activate_start_pose(runtime_action, active_start_pose)
                    lifecycle("startPoseActivate", resetReason="startRecording")
                input_recorder = InputRecorder(
                    resolved.name, resolved.labels, resolved.label_to_tracker_key,
                    acceleration_source, live_upright, MODEL_FPS,
                    path=recording_path_for(control.get("name")),
                    metadata=recording_metadata(),
                )
                lifecycle("recordingStart", recordingFile=input_recorder.path.name)
                control_message = f"recording {input_recorder.path.name}"
        elif name == "actionSelect":
            # Unity's action selector (or a scene load) chose an action; no calibration yet.
            action = control.get("action") or None
            if action is not None and action not in EXERCISES and action not in ACTION_RUNTIME_MODELS:
                raise ValueError(f"unknown action {action!r}")
            if action != runtime_action:
                await reset_action_state("actionSelect", action)
                control_message = f"动作已切换为 {action}，请先完成该动作的起始姿势校准"
            else:
                lifecycle("actionSelect", unchanged=True)
        elif name == "calibrationSessionStart":
            if tracker_stream.replaying:
                raise RuntimeError("cannot record calibration while replaying; return to live input first")
            if not control.get("_internal"):
                # Legacy step flow (bird dog / scene default): a named action selects its route, an
                # unnamed session has no action.  Either way nothing of the previous action survives.
                action = control.get("name") if control.get("name") in ACTION_RUNTIME_MODELS else None
                await reset_action_state("calibrationSessionStart", action)
            calibration_session_id += 1
            if input_recorder is None:
                input_recorder = InputRecorder(
                    resolved.name, resolved.labels, resolved.label_to_tracker_key,
                    acceleration_source, live_upright, MODEL_FPS,
                    path=recording_path_for(control.get("name") or "calibration"),
                    metadata=recording_metadata(recordingReason="calibration session"),
                )
                lifecycle("recordingStart", recordingFile=input_recorder.path.name)
            input_recorder.write_event(name)
            lifecycle("calibrationStart", requestedStartPose=control.get("startPose"))
            control_message = f"recording calibration {input_recorder.path.name}"
        elif name == "calibrationSessionEnd":
            # The recording keeps running from the calibration into the action; only stopRecording closes it.
            if input_recorder is not None:
                input_recorder.write_event(name)
            lifecycle("calibrationEnd", outcome=exercise_calibration.outcome)
        elif name == "stopRecording":
            # Only the file ends: the action, its calibration and start pose stay for the next set of the
            # same action.  Action-level state is cleared by actionSelect / the next calibration start.
            close_recording("stopRecording")
        elif name == "exerciseCalibrationStart":
            if tracker_stream.replaying:
                raise RuntimeError("请先返回实时输入再校准")
            action = control.get("action")
            if action not in EXERCISES:
                raise ValueError(f"unknown exercise {action!r}")
            # Every calibration starts from a clean action state (as a freshly started Python would).
            await reset_action_state("exerciseCalibrationStart", action)
            prior_before_start = arm_prior_enabled
            reach_heading.activate(None)
            exercise_calibration.start(action, control.get("startPose"))
            if exercise_arm_prior_backup is None:
                exercise_arm_prior_backup = prior_before_start
            arm_prior_enabled = not args.no_arm_prior and not action.startswith("sidebridge")
            try:
                await apply_control({"command": "calibrationSessionStart", "name": action, "_internal": True,
                                     "startPose": exercise_calibration.start_pose})
            except Exception:
                exercise_calibration.cancel()
                arm_prior_enabled, exercise_arm_prior_backup = exercise_arm_prior_backup, None
                raise
            input_recorder.write_event(name, action=action, startPose=exercise_calibration.start_pose,
                                       calibrationSessionId=calibration_session_id,
                                       initialHeadings=exercise_calibration.backup[0].tolist(),
                                       initialCorrections=exercise_calibration.backup[1].tolist())
            input_recorder.write_event("setPoseOptions", armPrior=arm_prior_enabled)
            control_message = "按网页姿势提示完成校准；切换动作会取消本次校准"
        elif name == "exerciseCalibrationCancel":
            # The selected action stays (its route); its start pose is not established.
            exercise_calibration.cancel()
            if exercise_arm_prior_backup is not None:
                arm_prior_enabled, exercise_arm_prior_backup = exercise_arm_prior_backup, None
            if input_recorder is not None:
                input_recorder.write_event(name)
            await apply_control({"command": "calibrationSessionEnd"})
            control_message = "校准已取消，已恢复上一次校准结果"
        elif name == "poseCalibrate":
            leg_chain.reset(name)
            calibrator.request(control.get("targets") or {})
            if input_recorder is not None:
                input_recorder.write_event("poseCalibrate", targets=control.get("targets"))
            control_message = "pose calibration: hold the start pose"
        elif name == "standCalibrate":
            leg_chain.reset(name)
            calibrator.request_stand()
            if input_recorder is not None:
                input_recorder.write_event("standCalibrate")
        elif name == "armsForwardCalibrate":
            leg_chain.reset(name)
            calibrator.request_arms_forward()
            if input_recorder is not None:
                input_recorder.write_event("armsForwardCalibrate")
        elif name == "birdDogStartCalibrate":
            runtime_action = "birddog"
            await switch_runtime_model(ACTION_RUNTIME_MODELS["birddog"], "birddog action")
            broadcaster.latest_model_command()
            # Bird-dog now uses the selected four-point start pose only.  Keep the
            # legacy reach-fitting protocol available for old clients, but do not
            # enter it in the current simple-start runtime.
            if not exercise_calibration.simple_start_only:
                calibrator.start_bird_dog_sequence()
            reach_heading.activate(None)
            if input_recorder is not None:
                input_recorder.write_event(name)
        elif name == "birdDogBaseCalibrate":
            if not exercise_calibration.simple_start_only:
                calibrator.request_bird_dog_base()
            else:
                # Sent at the end of the four-point start hold: fit the sensors to the start pose
                # (start_pose_fit.py) in the main loop, where the exercise rebase also happens.
                pending_start_pose_fit = ("birddog", "quadruped")
            if input_recorder is not None:
                input_recorder.write_event(name)
        elif name == "birdDogLeftReachCalibrate":
            if not exercise_calibration.simple_start_only:
                calibrator.request_bird_dog_reach(left_side=True)
            if input_recorder is not None:
                input_recorder.write_event(name)
        elif name == "birdDogRightReachCalibrate":
            if not exercise_calibration.simple_start_only:
                calibrator.request_bird_dog_reach(left_side=False)
            if input_recorder is not None:
                input_recorder.write_event(name)
        elif name == "poseHoldCalibrate":
            leg_chain.reset(name)
            calibrator.request_pose_hold(control.get("pose"), control.get("seconds"))
            if input_recorder is not None:
                input_recorder.write_event(name, pose=control.get("pose"), seconds=control.get("seconds"))
        elif name == "setPoseOptions":
            if "armPrior" in control:
                arm_prior_enabled = bool(control["armPrior"])
            if "calibration" in control:
                calibrator.calibration_enabled = bool(control["calibration"])
            if "anchor" in control:
                calibrator.anchor_enabled = bool(control["anchor"])
            control_message = (f"pose calibration {'on' if calibrator.calibration_enabled else 'off'}, "
                               f"sensed-bone anchoring {'on' if calibrator.anchor_enabled else 'off'}, "
                               f"straight-arm prior {'on' if arm_prior_enabled else 'off'}")
        elif name == "yawReset":
            if tracker_stream.replaying:
                return False    # recorded input already carries its own reset state
            await asyncio.wait_for(request_reset(), timeout=3)
            leg_chain.reset(name)
            control_message = "SlimeVR yaw reset"
        elif name == "startReplay":
            if input_recorder is not None:
                input_recorder.close()
                input_recorder = None
            recording = load_input_recording(resolve_recording_path(control.get("path")))
            if recording.layout != resolved.name or set(recording.labels) != set(resolved.labels):
                raise RuntimeError(
                    f"{recording.path.name} was recorded with {recording.layout}, current layout is {resolved.name}")
            tracker_stream.start_replay(recording, resolved.label_to_tracker_key)
            calibrator.reset()
            reach_heading.reset()
            leg_chain.reset("startReplay")
            latest.clear()
            recent_observations.clear()
            session = await asyncio.to_thread(rebuild_session, recording.upright)
            warmup_frames = SOURCE_SWITCH_WARMUP_FRAMES
            session_fresh = False
            lifecycle("fullReset", resetReason="startReplay", modelSessionReset=True, historyBufferCleared=True)
            control_message = f"replaying {recording.path.name} ({recording.duration:.1f}s loop)"
        elif name == "stopReplay":
            if tracker_stream.replaying:
                tracker_stream.stop_replay()
                calibrator.reset()
                reach_heading.reset()
                leg_chain.reset("stopReplay")
                latest.clear()
                recent_observations.clear()
                session = await asyncio.to_thread(rebuild_session, live_upright)
                warmup_frames = SOURCE_SWITCH_WARMUP_FRAMES
                session_fresh = False
                lifecycle("fullReset", resetReason="stopReplay", modelSessionReset=True, historyBufferCleared=True)
                control_message = "live input" + ("" if live_upright is not None else " (no upright calibration)")
        else:
            return False
        print(f"Unity control: {name} -> {control_message}", file=sys.stderr)
        return name in ("startReplay", "stopReplay")

    async def wake_for_controls():
        # Controls are applied inside the input loop; keep it turning when no input arrives.
        while True:
            await asyncio.sleep(0.25)
            if not broadcaster.controls.empty():
                tracker_stream.wake()

    wake_task = asyncio.create_task(wake_for_controls())

    # A bridge left running keeps serving old code: Unity reuses whatever listens on the port (and loses
    # its process handle after a script recompile).  Once the production code changes, warn Unity; with
    # no Unity client, no recording and no background job, exit so the next Play starts the new code.
    code_started = _code_stamp()
    code_stale = False

    async def exit_when_code_changes():
        nonlocal code_stale
        while True:
            await asyncio.sleep(CODE_CHECK_SECONDS)
            code_stale = _code_stamp() > code_started
            if (code_stale and not broadcaster._clients and input_recorder is None
                    and (compare_jobs is None or compare_jobs.idle)):
                print("Production code changed and no Unity client is connected: exiting so the next "
                      "Unity Play starts the new code.", file=sys.stderr, flush=True)
                os._exit(0)

    stale_task = asyncio.create_task(exit_when_code_changes())

    frame_index = 0            # 60 Hz model frames pushed into the window
    pose_index = 0             # poses inferred/broadcast
    since_inference = 0
    source_update_index = -1
    clock = FrameClock(MODEL_FPS, time.perf_counter())
    recent_observations = []
    sensor_history = {}  # per-tracker timelines avoid stamping carried values as fresh
    tracker_last_seen = {}
    next_sample_time = None
    keys = [resolved.label_to_tracker_key[label] for label in resolved.labels]
    binding_problem = last_binding_message = None
    if startup_replay is not None:
        # Layout probing consumes a sample and loading may take seconds. Start
        # the inference replay at its first recorded model frame after setup.
        tracker_stream.start_replay(startup_replay, resolved.label_to_tracker_key)
    async for update in tracker_stream:
        switched = False
        for control in broadcaster.drain_controls():
            try:
                switched = await apply_control(control) or switched
            except Exception as exc:
                control_message = f"{control.get('command')} failed: {exc}"
                print(f"Unity control failed: {control_message}", file=sys.stderr)
        if switched:
            sensor_history.clear()
            tracker_last_seen.clear()
            next_sample_time = None
            since_inference = 0
        if switched or not update:
            continue    # wake-up only, or a sample from the input source just switched away from
        update_received_monotonic = getattr(update, "received_monotonic", time.perf_counter())
        replay_frames = []
        for queued in [update] + tracker_stream.drain_pending():
            if not queued:
                continue
            received_at = getattr(queued, "received_monotonic", time.perf_counter())
            latest.update(queued)
            for key in queued:
                tracker_last_seen[key] = received_at
                if key in keys:
                    timeline = sensor_history.setdefault(key, [])
                    if timeline and received_at - timeline[-1][0] > MAX_TRACKER_STALENESS_SECONDS:
                        timeline.clear()
                        next_sample_time = None
                    timeline.append((received_at, latest[key]))
                    del timeline[:-32]
            source_update_index += 1
            if not tracker_stream.replaying:
                rebound = _binding_changed(latest, resolved)
                if rebound is not None:
                    binding_problem = rebound
            if recorder is not None:
                recorder.write("solarxrUpdate", {
                    "sourceUpdate": source_update_index,
                    "receiveMonotonicSeconds": received_at,
                    "receiveUnixSeconds": time.time(),
                    "trackers": [dataclasses.asdict(sample) for sample in queued.values()],
                })
            try:
                if any(not latest[key].online
                       or received_at - tracker_last_seen.get(key, 0.0) > MAX_TRACKER_STALENESS_SECONDS
                       for key in keys):
                    continue
                if tracker_stream.replaying:
                    replay_frames.append((
                        np.asarray([latest[key].quat_xyzw for key in keys], dtype=np.float32),
                        np.asarray([_sample_acceleration(latest[key], acceleration_source)
                                    for key in keys], dtype=np.float32),
                        {key: latest[key] for key in keys},
                        getattr(queued, "calibration", None),
                    ))
            except KeyError:
                continue  # a required tracker dropped out; wait for it to come back
        if binding_problem is not None:
            # SlimeVR re-assigned trackers (server restart / reconnect renumbers devices): the startup
            # label -> tracker key map would feed the wrong body part into a model slot.  Re-resolve
            # from the live body parts and drop every piece of history built from the old routing.
            try:
                new_layout = detect_layout(latest)
            except LayoutMismatchError as exc:
                if binding_problem != last_binding_message:
                    print(f"Tracker binding changed; inference paused: {binding_problem} ({exc})", file=sys.stderr)
                    last_binding_message = binding_problem
                    if input_recorder is not None:
                        input_recorder.write_event("trackerBindingChanged", problem=binding_problem, resolved=False)
                binding_problem = None
                continue
            if new_layout.name != resolved.name:
                if binding_problem != last_binding_message:
                    print(f"Tracker binding changed to layout {new_layout.name}; restart Python", file=sys.stderr)
                    last_binding_message = binding_problem
                binding_problem = None
                continue
            print(f"Tracker binding changed ({binding_problem}); re-resolved {new_layout.label_to_tracker_key}",
                  file=sys.stderr)
            if input_recorder is not None:
                input_recorder.write_event("trackerBindingChanged", problem=binding_problem, resolved=True,
                                           labelToTrackerKey=new_layout.label_to_tracker_key)
                input_recorder.label_to_key = dict(new_layout.label_to_tracker_key)
            resolved = new_layout
            keys = [resolved.label_to_tracker_key[label] for label in resolved.labels]
            calibrator.reset()
            reach_heading.reset()
            leg_chain.reset("trackerBindingChanged")
            recent_observations.clear()
            sensor_history.clear()
            tracker_last_seen.clear()
            next_sample_time = None
            session = await asyncio.to_thread(rebuild_session, live_upright)
            warmup_frames = SOURCE_SWITCH_WARMUP_FRAMES
            binding_problem = last_binding_message = None
            control_message = "tracker binding changed: re-resolved, model history reset"
            session_fresh = False
            active_start_pose = None
            lifecycle("fullReset", resetReason="trackerBindingChanged", modelSessionReset=True,
                      historyBufferCleared=True)
            continue
        now = time.perf_counter()
        if not tracker_stream.replaying and any(key not in latest or key not in sensor_history or not sensor_history[key]
               or not latest[key].online
               or now - sensor_history[key][-1][0] > MAX_TRACKER_STALENESS_SECONDS
               for key in keys):
            next_sample_time = None
            continue
        if tracker_stream.replaying:
            # Recorded ticks have already been expanded by InputMux. Do not
            # re-clock or interpolate them according to this machine's speed.
            frames = replay_frames
        else:
            # Wait for all sensors to bracket the target. This one-source-frame
            # delay prevents extrapolation/hold-last zero-gyro artifacts.
            oldest = max(sensor_history[key][0][0] for key in keys)
            watermark = min(sensor_history[key][-1][0] for key in keys)
            if next_sample_time is None or next_sample_time < oldest:
                next_sample_time = oldest
            frames = []
            while next_sample_time <= watermark + 1e-9:
                frames.append((*_interpolate_snapshot(sensor_history, next_sample_time,
                                                      keys, acceleration_source), None))
                next_sample_time += clock.dt
        frame_steps = len(frames)
        ticks = frame_steps
        if not frame_steps:
            continue
        # Retain the final sample before the next clock tick for interpolation.
        for key in keys:
            timeline = sensor_history[key]
            while (len(timeline) > 2 and next_sample_time is not None
                   and timeline[1][0] <= next_sample_time):
                timeline.pop(0)

        host_unix = time.time()
        model_samples = None
        for quats, accel, samples, replay_calibration in frames:
            model_samples = samples
            if replay_calibration is not None:
                calibrator.headings = np.asarray(replay_calibration["headings"], dtype=np.float64)
                calibrator.corrections = np.asarray(replay_calibration["corrections"], dtype=np.float64)
                calibrator.calibration_enabled = replay_calibration["calibrationEnabled"]
                calibrator.anchor_enabled = replay_calibration["anchorEnabled"]
                arm_prior_enabled = replay_calibration["armPriorEnabled"]
                load_bridge_calibration(calibrator, replay_calibration)
            if input_recorder is not None:
                input_recorder.write_tick(samples, 1, calibration_snapshot())
            feature, calibrated_ori, aligned_accel, base_ori = prepare_frame(
                session, calibrator, selected_model, resolved.labels, quats, accel, samples, keys,
                leg_chain, args.model_accel_fix)
            # Keep the sensor orientation before pose calibration/model offsets, so a model switch or
            # action rebase applies the latest committed calibration to the entire rebuilt history.
            raw_ori = bone_orientation(quats, resolved.labels)
            recent_observations.append((feature.copy(), raw_ori.numpy(), aligned_accel.numpy()))
            push_frame(session, feature, calibrated_ori, aligned_accel)
            session_fresh = False
            if replay_calibration is None:     # a replayed recording already carries its headings
                reach_heading.update(calibrator, calibrator._history[-1])
        del recent_observations[:-480]
        calibration_message = calibrator.advance(frame_steps)
        exercise_event = exercise_calibration.advance(frame_steps)
        rebase = None     # (action, start pose): rebuild the model context from the start hold
        if exercise_event is not None:
            calibration_message = exercise_event["message"]
            # The lying-cat start hold is also the mounting-calibration pose.  Older
            # calibration recordings could report cat_neutral as successfully checked
            # while leaving all limb mounting corrections at identity.  That makes the
            # contact rule receive incorrectly oriented wrist/shank measurements.  Make
            # the fit explicit immediately before the context rebase; fitting from raw
            # history is idempotent when ExerciseCalibration already performed it.
            if (exercise_calibration.action == "lyingcat"
                    and exercise_event.get("pose") == "cat_neutral"
                    and "start pose fit" not in exercise_event.get("message", "")
                    and exercise_calibration.outcome == "complete"):
                fitted, fit_message = fit_start_pose(
                    calibrator,
                    np.asarray(calibrator._history)[-START_POSE_SEARCH_FRAMES:],
                    "lyingcat",
                    exercise_calibration.start_pose,
                )
                if fitted:
                    calibration_message += f"；{fit_message}"
                else:
                    # Do not rebase a lying-cat model on uncalibrated limb axes.  Keep
                    # the previous calibration and require the user to repeat the hold.
                    calibration_message += f"；{fit_message}"
                    exercise_calibration.outcome = "failed"
                    exercise_calibration.active = False
            if input_recorder is not None:
                input_recorder.write_event("exercisePoseCalibrate", **exercise_event)
            lifecycle("calibrationPoseStep", pose=exercise_event.get("pose"),
                      passed=exercise_calibration.outcome != "failed", message=exercise_event.get("message"))
            if not exercise_calibration.active:
                outcome = exercise_calibration.outcome
                if outcome != "complete" and exercise_arm_prior_backup is not None:
                    arm_prior_enabled = exercise_arm_prior_backup
                exercise_arm_prior_backup = None
                if input_recorder is not None:
                    input_recorder.write_event("exerciseCalibrationEnd", outcome=outcome,
                                               action=exercise_calibration.action,
                                               startPose=exercise_calibration.start_pose)
                if outcome == "complete":
                    rebase = (exercise_calibration.action, exercise_calibration.start_pose)
                await apply_control({"command": "calibrationSessionEnd"})
                calibration_message += ((f"；全部校准完成，模型已从 {exercise_calibration.start_pose} 起始姿态重建上下文")
                                        if outcome == "complete" else "；校准未通过，已恢复原校准，请重试")
        if pending_start_pose_fit is not None:
            fit_action, fit_pose = pending_start_pose_fit
            pending_start_pose_fit = None
            passed, calibration_message = fit_start_pose(
                calibrator, np.asarray(calibrator._history)[-START_POSE_SEARCH_FRAMES:], fit_action, fit_pose)
            if input_recorder is not None:
                input_recorder.write_event("startPoseFit", action=fit_action, startPose=fit_pose, passed=passed,
                                           message=calibration_message)
            calibration_message += (f"；模型已从 {fit_pose} 起始姿态重建上下文" if passed
                                    else "；未修正传感器，请摆好起始姿势后重新校准")
            if passed:
                rebase = (fit_action, fit_pose)
        if rebase is not None:
            rebase_action, rebase_pose = rebase
            calibrator.refresh_corrected_history()
            context = recent_observations[-DEFAULT_CONTEXT_FRAMES:]
            if (rebase_action == "hipthrust" and selected_model in BRIDGE_MODELS
                    and calibrator.bridge_waist is not None):
                bias, spread = await asyncio.to_thread(
                    estimate_trunk_bias, rebuild_session(None), context, selected_model, resolved.labels, calibrator)
                if bias is not None and spread <= TRUNK_BIAS_MAX_SPREAD_DEG:
                    calibrator.bridge_trunk = bias
                    calibration_message = (calibration_message or "") + f"；躯干平躺偏差已校准（波动{spread:.0f}°）"
                else:
                    calibration_message = ((calibration_message or "")
                                           + f"；躯干平躺偏差未校准（起始姿势不够稳定，波动{spread:.0f}°）")
            session = await asyncio.to_thread(rebuild_context_session, context)
            recent_observations[:] = context
            leg_chain.reset(f"start pose -> {rebase_action}/{rebase_pose}")
            leg_chain.activate_start_pose(rebase_action, rebase_pose)
            active_start_pose = rebase_pose if rebase_action == runtime_action else None
            reach_heading.activate(rebase_action)
            warmup_frames = 0
            since_inference = 0
            if input_recorder is not None:
                input_recorder.write_event(
                    "modelContextRebase", action=rebase_action, startPose=rebase_pose, frames=len(context),
                    headings=calibrator.headings.tolist(), corrections=calibrator.corrections.tolist(),
                    **bridge_calibration_state(calibrator))
            lifecycle("startPoseActivate", resetReason="modelContextRebase", rebaseAction=rebase_action,
                      rebaseStartPose=rebase_pose, contextFrames=len(context), modelSessionReset=True)
        if calibration_message:
            control_message = calibration_message
            print(calibration_message, file=sys.stderr)
            if input_recorder is not None:
                input_recorder.write_event("poseCalibrationResult", message=calibration_message,
                                           corrections=calibrator.corrections.tolist())
        frame_index += frame_steps
        since_inference += frame_steps
        model_frame = frame_index - 1      # index of the newest frame in the window
        if warmup_frames > 0:
            warmup_frames = max(0, warmup_frames - frame_steps)
            continue
        tracker_rows = None
        if recorder is not None:
            tracker_rows = []
            for label in resolved.labels:
                sample = model_samples[resolved.label_to_tracker_key[label]]
                tracker_rows.append({
                    "label": label,
                    "trackerKey": sample.tracker_key,
                    "deviceId": sample.device_id,
                    "trackerNum": sample.tracker_num,
                    "bodyPart": sample.body_part,
                    "status": sample.status,
                    "online": sample.online,
                    "rawQuaternion": sample.raw_quat_xyzw,
                    "referenceAdjustedQuaternion": sample.quat_xyzw,
                    "linearAccelerationMps2": sample.accel_xyz,
                    "referenceAdjustedLinearAccelerationMps2": sample.reference_adjusted_accel_xyz,
                })
        if since_inference < args.infer_every:
            if recorder is not None:
                # Every 60 Hz window advance is recorded so a replay can rebuild the exact window.
                recorder.write("windowFrame", {
                    "modelFrame": model_frame,
                    "ticksAdvanced": frame_steps,
                    "receiveMonotonicSeconds": update_received_monotonic,
                    "trackers": tracker_rows,
                    "feature": feature,
                })
            continue
        since_inference %= args.infer_every
        requested_model = broadcaster.latest_model_command()
        action_model = ACTION_RUNTIME_MODELS.get(runtime_action)
        if action_model:
            requested_model = action_model
        if requested_model and requested_model != selected_model:
            try:
                await switch_runtime_model(requested_model, "Unity")
            except Exception as exc:
                print(f"Model switch failed ({requested_model}): {exc}", file=sys.stderr)
        inference_start_monotonic = time.perf_counter()
        # MobilePose is CPU-bound and can take long enough to starve the
        # asyncio event loop.  Keep SolarXR's WebSocket reader responsive by
        # running the blocking model call in a worker thread.
        output = await asyncio.to_thread(session.infer)
        raw_output_quats = batch_smpl_matrix_to_unity_quat_xyzw(output.detach().numpy())
        exercise_calibration.observe_output(output.detach().numpy(), selected_model)
        output, arm_prior_weights, deadbug_legs = postprocess_output(
            output, calibrator, selected_lag, resolved.labels, arm_prior_enabled, selected_model, leg_chain,
            dt=args.infer_every / 60.0, action_id=runtime_action)
        inference_end_monotonic = time.perf_counter()
        quats_out = batch_smpl_matrix_to_unity_quat_xyzw(output.detach().numpy())
        tracker_rotations = np.stack([quat_xyzw_to_unity_quat_xyzw(q) for q in quats])
        contact = None     # the current baseline has no foot-contact head
        diagnostics = {
            "model": selected_model,
            "sourceUpdate": source_update_index,
            "modelInputFrame": model_frame,
            "modelOutputRepresentsFrame": (
                model_frame - selected_lag
                if model_frame >= selected_lag else -1
            ),
            "ticksAdvanced": frame_steps,
            "schedulerTicks": ticks,
            "receiveMonotonicSeconds": update_received_monotonic,
            "receiveUnixSeconds": host_unix,
            "inferenceStartMonotonicSeconds": inference_start_monotonic,
            "inferenceEndMonotonicSeconds": inference_end_monotonic,
            "modelLagFrames": selected_lag,
            "modelFps": MODEL_FPS,
            "inferEvery": args.infer_every,
            "armPriorWeights": arm_prior_weights,
            "deadbugLegs": deadbug_legs,
            "modelAccelFrameFix": bool(args.model_accel_fix),
            "labelToTrackerKey": dict(resolved.label_to_tracker_key),
            "footContactEnabled": False,
            "accelerationSource": acceleration_source,
            "inputSource": "replay" if tracker_stream.replaying else "live",
            "replayFile": tracker_stream.replay.path.name if tracker_stream.replaying else "",
            "recordingFile": input_recorder.path.name if input_recorder is not None else "",
            "recordingSeconds": input_recorder.seconds if input_recorder is not None else 0.0,
            "controlMessage": control_message,
            "exerciseCalibration": exercise_calibration.status(),
            # What the offline replay (production/live_replay.py) needs to reproduce this output.
            "actionId": runtime_action,
            "codeStale": code_stale,
            "reachHeadingEvents": reach_heading.events[-4:],
            # Action lifecycle (run.py reset_action_state): which action/start pose this output belongs to.
            "actionState": action_state(),
        }
        if input_recorder is not None:
            input_recorder.write_prediction(raw_output_quats, quats_out, diagnostics, calibration_snapshot())
        if recorder is not None:
            local = output.detach().numpy()
            global_rot = np.empty_like(local)
            parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
            for joint, parent in enumerate(parents):
                global_rot[joint] = local[joint] if parent < 0 else global_rot[parent] @ local[joint]
            frame_record = {
                **diagnostics,
                "frame": pose_index,
                "trackers": tracker_rows,
                "alignedAccelerationMps2": aligned_accel,
                "calibratedOrientationMatrices": calibrated_ori,
                "feature": feature,
                "smplLocalRotationMatrices": local,
                "smplGlobalRotationMatrices": global_rot,
                "unityLocalQuaternions": quats_out,
            }
        await broadcaster.broadcast(
            pose_index, resolved.name, quats_out, contact, tracker_rotations, diagnostics,
            model_name=selected_model,
        )
        if recorder is not None:
            frame_record["broadcastMonotonicSeconds"] = time.perf_counter()
            recorder.write("inferenceFrame", frame_record)
        pose_index += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="host to broadcast the SMPL pose stream on")
    parser.add_argument("--port", type=int, default=21200, help="port to broadcast the SMPL pose stream on")
    parser.add_argument("--web", action="store_true", help="serve a browser-based realtime stick-figure viewer")
    parser.add_argument("--web-host", default="127.0.0.1", help="web viewer bind address")
    parser.add_argument("--web-port", type=int, default=8765, help="web viewer HTTP port")
    parser.add_argument("--calibration-frames", type=int, default=0,
                         help="deprecated compatibility option; no Python heading fit is performed")
    parser.add_argument("--acceleration-source", choices=("server",), default="server",
                         help="use SlimeVR reference-adjusted acceleration")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT,
                         help="root directory containing <model>/base_model.pth + manifest.json")
    parser.add_argument("--model", default=DEFAULT_RUNTIME_MODEL,
                         help="converted checkpoint name under --checkpoint-root, or a directory/.pth path "
                              f"(default: {DEFAULT_MODEL}; identity-anchored variant: pnp5_60a_sc_ftr3)")
    parser.add_argument("--waist-checkpoint-root", type=Path, default=WAIST_CHECKPOINT_ROOT,
                        help="root directory containing the selectable waist-surface model checkpoints")
    parser.add_argument("--checkpoint", type=Path, default=None,
                         help="explicit combined checkpoint (.pth or its directory); overrides --model")
    parser.add_argument("--anchor", choices=("auto", "on", "off"), default="auto",
                         help="identity anchoring of sensed bones; auto follows the checkpoint manifest, "
                              "an explicit value that contradicts it is an error")
    parser.add_argument("--infer-every", type=int, default=None,
                         help=f"run the network every N frames of the {MODEL_FPS} Hz window "
                              "(default: 2 -> 30 poses/s; 1 for the stateful waist_imudino_stream_* models)")
    parser.add_argument("--output-fps", type=float, default=None,
                         help=f"alternative to --infer-every: poses per second; must divide {MODEL_FPS}")
    parser.add_argument("--include-contact", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--contact-interval", type=int, default=30, help=argparse.SUPPRESS)
    parser.add_argument("--sync-calibration", action="store_true",
                         help="pause for manual SlimeVR upright and ski calibration")
    parser.add_argument("--upright-calibration-frames", type=int, default=30,
                         help="neutral-pose samples used to prime the model context (default: 30)")
    parser.add_argument("--upright-settle-seconds", type=float, default=3.0,
                         help="automatic countdown before upright capture (default: 3 seconds)")
    parser.add_argument("--upright-calibration", action="store_true",
                         help="opt in to the old neutral-standing prompt before action calibration")
    parser.add_argument("--skip-upright-calibration", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--record", type=Path, default=None,
                         help="write live SolarXR inputs, exact model features and SMPL outputs as JSONL")
    parser.add_argument("--no-auto-compare", action="store_true",
                        help="do not build the SMPL surface replay after each saved recording")
    parser.add_argument("--no-pose-calibration", action="store_true",
                        help="ignore Unity's per-set coach start-pose mounting calibration")
    parser.add_argument("--anchor-sensed", action="store_true",
                        help="replace the sensed bones' output orientation with the measurement (off by default: "
                             "with unobserved thighs it can cross the ankles)")
    parser.add_argument("--no-arm-prior", action="store_true",
                        help="keep the model's upper arm when the forearm reaches forward/overhead (see arm_prior.py)")
    parser.add_argument("--deadbug-legs", choices=("auto", "on", "off"), default="auto",
                        help="dead-bug thigh estimate from shank translation (deadbug_legs.py); auto = only for "
                             "the dead-bug model, off = unconstrained network legs")
    parser.add_argument("--model-accel-fix", action="store_true",
                        help="also feed the models the sensor-frame-corrected acceleration (accel_frame.py); "
                             "off by default so existing model behaviour (bird dog etc.) is unchanged")
    parser.add_argument("--torch-threads", type=int, default=8,
                        help="PyTorch CPU threads; the default 48+ on this machine is slower and competes with Unity")
    parser.add_argument("--feed-interval-ms", type=int, default=16,
                        help="SolarXR minimum time between tracker updates; 16 gives ~60 Hz fresh samples for the "
                             "60 Hz model window (33 halves the rate and repeats every sample twice)")
    parser.add_argument("--replay", default=None,
                        help="drive the loop from a Unity input recording instead of SolarXR "
                             "(path, or a file name under realUse/recordings; empty string = newest)")
    parser.add_argument("--config", type=Path, default=None,
                         help="YAML defaults for the realtime bridge; explicit CLI values take precedence")
    args = parser.parse_args()
    config = _load_runtime_config(args.config)
    defaults = {
        "calibration_frames": 0,
        "checkpoint_root": DEFAULT_CHECKPOINT_ROOT,
        "model": DEFAULT_RUNTIME_MODEL,
        "anchor": "auto",
        "infer_every": None,
        "upright_calibration_frames": 30,
        "upright_settle_seconds": 3.0,
        "port": 21200,
        "waist_checkpoint_root": WAIST_CHECKPOINT_ROOT,
    }
    for name, default in defaults.items():
        if getattr(args, name) == default and name in config:
            value = config[name]
            if name in {"checkpoint_root", "waist_checkpoint_root"}:
                value = Path(value)
            setattr(args, name, value)
    if args.include_contact:
        print("Warning: --include-contact is ignored; the current model has no foot-contact head.", file=sys.stderr)
    if args.output_fps is not None:
        ratio = MODEL_FPS / args.output_fps
        if args.output_fps <= 0 or abs(ratio - round(ratio)) > 1e-6:
            parser.error(f"--output-fps must divide {MODEL_FPS} (e.g. 60, 30, 20), got {args.output_fps}")
        args.infer_every = int(round(ratio))
    if args.infer_every is None:
        # Stateful stream heads decode every frame anyway; emitting each one removes ~17 ms average wait.
        args.infer_every = 1 if str(args.model).startswith("waist_imudino_stream_") else 2
    if args.infer_every < 1:
        parser.error("--infer-every must be >= 1")
    torch.set_num_threads(max(1, args.torch_threads))
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

"""Offline reproduction of the realtime replay path for one Unity input recording.

Same order and functions as run.py in replay mode: the recording's upright sample primes the
session, 12 warm-up frames, the recorded calibration snapshot is restored on every frame, every
60 Hz frame goes through ``frame_chain.prepare_frame`` and every ``inferEvery``-th frame through
``session.infer`` + ``frame_chain.postprocess_output``.  run.py batches frames that arrive
together; here each frame is processed on its own (the ideal replay timing).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from production.calibration import identity_heading
from production.frame_chain import LegChain, postprocess_output, prepare_frame, push_frame
from production.input_recording import _sample_for, load_input_recording
from production.layout import ResolvedLayout
from production.model_backends import make_waist_session, model_lag_frames, resolve_waist_model
from production.pose_calibration import PoseCalibrator
from production.bridge_contact import load_bridge_calibration
from production.rotmat import batch_smpl_matrix_to_unity_quat_xyzw

WARMUP_FRAMES = 12   # run.py SOURCE_SWITCH_WARMUP_FRAMES


@dataclasses.dataclass
class ReplayResult:
    model: str
    lag: int
    infer_every: int
    labels: list
    n_frames: int
    input_frames: np.ndarray          # (P,) newest input frame of each prediction
    represented_frames: np.ndarray    # (P,) frame each prediction represents (input - lag)
    raw_local: np.ndarray             # (P,24,3,3) network output
    final_local: np.ndarray           # (P,24,3,3) after anchoring / arm prior / legs
    unity_quats: np.ndarray           # (P,24,4) exactly what run.py would broadcast
    diagnostics: list                 # per prediction
    calibrated_ori: np.ndarray        # (T,n,3,3) calibrated input orientations (before model offsets)
    model_ori: np.ndarray             # (T,n,3,3) orientations actually fed to the model
    leg_events: dict


def replay(path: Path, model_id: str | None = None, leg_mode: str = "auto", model_accel_fix: bool = False,
           infer_every: int | None = None, loaded=None, bridge_override: dict | None = None) -> ReplayResult:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    header = rows[0]
    recording = load_input_recording(path)
    model_id = model_id or header.get("model")
    infer_every = int(infer_every or header.get("inferEvery") or 2)
    labels = list(recording.labels)
    keys = [recording.label_to_key[label] for label in labels]
    resolved = ResolvedLayout(recording.layout, labels, dict(recording.label_to_key))
    heading = identity_heading(labels)
    session = make_waist_session(model_id, resolved, heading, loaded=loaded or resolve_waist_model(model_id))
    if recording.upright is not None:
        session.prime(*recording.upright)
    lag = model_lag_frames(model_id)
    calibrator = PoseCalibrator(labels)
    initial = header.get("initialCalibration") or {}
    calibrator.calibration_enabled = initial.get("calibrationEnabled", True)
    calibrator.anchor_enabled = initial.get("anchorEnabled", False)
    arm_prior_enabled = initial.get("armPriorEnabled", True)
    legs = LegChain(labels, leg_mode)
    legs.reset("replay start")

    warmup, since_inference, frame_index = WARMUP_FRAMES, 0, 0
    calibrated, model_oris = [], []
    input_frames, raw_out, final_out, quats_out, diagnostics = [], [], [], [], []
    for raw_samples, ticks, state in zip(recording.samples, recording.ticks,
                                         recording.calibration or [None] * len(recording.samples)):
        samples = {recording.label_to_key[label]: _sample_for(raw_samples[label], recording.label_to_key[label])
                   for label in labels}
        quats = np.asarray([samples[key].quat_xyzw for key in keys], dtype=np.float32)
        accel = np.asarray([samples[key].reference_adjusted_accel_xyz for key in keys], dtype=np.float32)
        for _ in range(max(1, ticks)):
            if state is not None:
                calibrator.headings = np.asarray(state["headings"], dtype=np.float64)
                calibrator.corrections = np.asarray(state["corrections"], dtype=np.float64)
                calibrator.calibration_enabled = state["calibrationEnabled"]
                calibrator.anchor_enabled = state["anchorEnabled"]
                arm_prior_enabled = state["armPriorEnabled"]
                load_bridge_calibration(calibrator, state)
            if bridge_override is not None:
                load_bridge_calibration(calibrator, bridge_override)
            feature, model_ori, aligned_accel, base_ori = prepare_frame(
                session, calibrator, model_id, labels, quats, accel, samples, keys, legs, model_accel_fix)
            calibrated.append(base_ori.numpy())
            model_oris.append(model_ori.numpy())
            push_frame(session, feature, model_ori, aligned_accel)
            calibrator.advance(1)
            frame_index += 1
            since_inference += 1
            if warmup > 0:
                warmup -= 1
                continue
            if since_inference < infer_every:
                continue
            since_inference %= infer_every
            with torch.no_grad():
                output = session.infer()
            raw = output.detach().numpy().astype(np.float32)
            output, arm_weights, leg_diag = postprocess_output(
                output, calibrator, lag, labels, arm_prior_enabled, model_id, legs, dt=infer_every / 60.0)
            final = output.detach().numpy().astype(np.float32)
            input_frames.append(frame_index - 1)
            raw_out.append(raw)
            final_out.append(final)
            quats_out.append(batch_smpl_matrix_to_unity_quat_xyzw(final))
            diagnostics.append({"armPriorWeights": arm_weights, "deadbugLegs": leg_diag,
                                "modelInputFrame": frame_index - 1, "modelOutputRepresentsFrame": frame_index - 1 - lag})
    input_frames = np.asarray(input_frames)
    return ReplayResult(
        model=model_id, lag=lag, infer_every=infer_every, labels=labels, n_frames=frame_index,
        input_frames=input_frames, represented_frames=input_frames - lag,
        raw_local=np.asarray(raw_out), final_local=np.asarray(final_out), unity_quats=np.asarray(quats_out),
        diagnostics=diagnostics, calibrated_ori=np.asarray(calibrated), model_ori=np.asarray(model_oris),
        leg_events=legs.estimator.recent_events(),
    )


def per_frame_unity_quaternions(result: ReplayResult) -> np.ndarray:
    """(T,24,4) Unity local quaternions on the input timeline (each output at the frame it represents)."""
    out = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (result.n_frames, 24, 1))
    filled = np.zeros(result.n_frames, dtype=bool)
    for frame, quats in zip(result.represented_frames, result.unity_quats):
        if 0 <= frame < result.n_frames:
            out[frame] = quats
            filled[frame] = True
    for i in range(1, result.n_frames):
        if not filled[i]:
            out[i] = out[i - 1]
    return out

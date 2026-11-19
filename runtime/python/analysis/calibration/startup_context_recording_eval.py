"""Compare upright, empty, and action-pose model startup on real recordings.

Each case uses the final eight continuous seconds of a successful action
calibration recording.  The action branch seeds a fresh model with the first
60 real IMU frames of that stable start-pose hold, then all branches render the
same following seven seconds.  No gap between two recording files is involved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from production.bridge_contact import load_bridge_calibration
from production.calibration import bone_orientation, identity_heading
from production.frame_chain import push_frame, rebuild_feature
from production.input_recording import load_input_recording
from production.layout import ResolvedLayout
from production.model_backends import make_waist_session, model_lag_frames, resolve_waist_model
from production.offline_compare import _expand, joint_positions, write_viewer
from production.paths import paths
from production.pose_calibration import PoseCalibrator
from production.pose_context import corrected_orientation, replay_start_context
from production.start_pose_constraint import StartPoseConstraint, _globals


CASES = {
    "deadbug": "calibration_20261116_044805.jsonl",
    "hipthrust": "calibration_20261116_212554.jsonl",
    "lyingcat": "calibration_20261116_195247.jsonl",
}
CONTEXT_FRAMES = 60
EVALUATION_FRAMES = 420
INFER_EVERY = 2
WARMUP_FRAMES = 12


def _header(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.loads(file.readline())


def _frames(recording):
    quats, accels, _ = _expand(recording, [])
    return quats, accels


def _calibrator(labels, state) -> PoseCalibrator:
    if not state:
        raise ValueError("recording has no calibration snapshot")
    calibrator = PoseCalibrator(labels)
    calibrator.headings = np.asarray(state["headings"], dtype=np.float64)
    calibrator.corrections = np.asarray(state["corrections"], dtype=np.float64)
    calibrator.calibration_enabled = bool(state.get("calibrationEnabled", True))
    calibrator.anchor_enabled = bool(state.get("anchorEnabled", False))
    load_bridge_calibration(calibrator, state)
    return calibrator


def _state(recording, newest=False):
    states = [state for state in recording.calibration if state]
    if not states:
        return None
    return states[-1] if newest else states[0]


def _observation(session, calibrator, model_id, labels, quats, accel):
    feature, _, _ = session.prepare_feature(quats, accel)
    raw_orientation = bone_orientation(quats, labels)
    headings = torch.from_numpy(calibrator.headings).to(raw_orientation)
    aligned_accel = (headings @ torch.from_numpy(accel).to(raw_orientation).unsqueeze(-1)).squeeze(-1)
    feature = np.asarray(feature, dtype=np.float32).copy()
    feature[:aligned_accel.numel()] = aligned_accel.reshape(-1).numpy() / 30.0
    return feature, raw_orientation.numpy(), aligned_accel.numpy()


def _run_variant(model_id, recording, context, mode, frames):
    resolved = ResolvedLayout(recording.layout, recording.labels, dict(recording.label_to_key))

    def new_session():
        loaded = resolve_waist_model(model_id)
        return make_waist_session(model_id, resolved, identity_heading(recording.labels), loaded=loaded)

    calibrator = _calibrator(recording.labels, _state(recording, newest=True))
    session = new_session()
    if mode == "upright":
        if recording.upright is None:
            raise ValueError("upright branch requested for recording without upright reference")
        session.prime(*recording.upright)
    elif mode == "action":
        replay_start_context(session, context, model_id, recording.labels, calibrator,
                             frames=CONTEXT_FRAMES)
    elif mode != "empty":
        raise ValueError(mode)

    quats, accels = frames
    identity = np.tile(np.eye(3, dtype=np.float32), (24, 1, 1))
    poses = np.repeat(identity[None], len(quats), axis=0)
    last = identity
    lag = model_lag_frames(model_id)
    for frame, (quat, accel) in enumerate(zip(quats, accels)):
        feature, raw_orientation, aligned_accel = _observation(
            session, calibrator, model_id, recording.labels, quat, accel)
        base = corrected_orientation(raw_orientation, calibrator)
        feature, model_orientation = rebuild_feature(
            feature, model_id, recording.labels, base, calibrator=calibrator)
        push_frame(session, feature, model_orientation, torch.from_numpy(aligned_accel))
        if frame >= (WARMUP_FRAMES if mode == "empty" else 0) and frame % INFER_EVERY == 1:
            last = session.infer().detach().numpy().astype(np.float32)
        represented = frame - lag
        if represented >= 0:
            poses[represented] = last
    # The final ``lag`` input frames do not have represented outputs yet.  Do
    # not leave their identity placeholders in stability metrics or viewers.
    return poses[:-lag] if lag > 0 else poses


def _rotation_angle(left, right):
    relative = np.swapaxes(left, -1, -2) @ right
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _apply_start_constraint(action, start_pose, poses, labels, calibrator, quats):
    constraint = StartPoseConstraint(labels)
    constraint.activate(action, start_pose)
    constrained = np.asarray(poses).copy()
    for frame in range(len(constrained)):
        raw = bone_orientation(quats[frame], labels).numpy()
        measured = corrected_orientation(raw, calibrator).numpy()
        constrained[frame], _ = constraint.apply(constrained[frame], measured, 1.0 / 60.0)
    return constrained


def _knee_flex(local, hip, knee):
    values = []
    axis = np.array([0.0, -1.0, 0.0])
    for pose in local:
        global_rot = _globals(pose)
        thigh, shank = global_rot[hip] @ axis, global_rot[knee] @ axis
        values.append(np.degrees(np.arccos(np.clip(thigh @ shank, -1.0, 1.0))))
    return np.asarray(values)


def _metrics(poses):
    start = WARMUP_FRAMES
    stop = min(len(next(iter(poses.values()))), start + 180)
    inference_frames = np.arange(start + (1 - start) % INFER_EVERY, stop, INFER_EVERY)
    result = {}
    for name, values in poses.items():
        selected = values[inference_frames]
        velocity = _rotation_angle(selected[:-1], selected[1:]) if len(selected) > 1 else np.zeros((0, 24))
        result[name] = {
            "first3sMeanStepDeg": round(float(np.mean(velocity)), 3),
            "first3sP95StepDeg": round(float(np.percentile(velocity, 95)), 3),
            "first3sMaxStepDeg": round(float(np.max(velocity)), 3),
            "firstSecondLeftKneeFlexDeg": round(float(np.mean(_knee_flex(values[:60], 1, 4))), 1),
            "firstSecondRightKneeFlexDeg": round(float(np.mean(_knee_flex(values[:60], 2, 5))), 1),
        }
    action = poses["action"]
    for other in ("upright", "empty"):
        difference = _rotation_angle(action[start:stop], poses[other][start:stop])
        result[f"actionVs{other.title()}"] = {
            "first3sMeanJointDeg": round(float(np.mean(difference)), 3),
            "first3sP95JointDeg": round(float(np.percentile(difference, 95)), 3),
            "at3sMeanJointDeg": round(float(np.mean(difference[-min(30, len(difference)):])), 3),
        }
    return result


def evaluate(case_name: str, output_dir: Path):
    calibration_name = CASES[case_name]
    calibration_path = paths.recordings / calibration_name
    calibration = load_input_recording(calibration_path)
    header = _header(calibration_path)
    model_id = header["model"]
    calibrator = _calibrator(calibration.labels, _state(calibration, newest=True))
    all_quats, all_accels = _frames(calibration)
    required = CONTEXT_FRAMES + EVALUATION_FRAMES
    if len(all_quats) < required:
        raise ValueError(f"{calibration_name} has {len(all_quats)} frames; need {required}")
    hold_quats, hold_accels = all_quats[-required:], all_accels[-required:]
    context_quats, context_accels = hold_quats[:CONTEXT_FRAMES], hold_accels[:CONTEXT_FRAMES]
    evaluation_frames = (hold_quats[CONTEXT_FRAMES:], hold_accels[CONTEXT_FRAMES:])
    resolved = ResolvedLayout(calibration.layout, calibration.labels, dict(calibration.label_to_key))
    feature_session = make_waist_session(
        model_id, resolved, identity_heading(calibration.labels), loaded=resolve_waist_model(model_id))
    context = [
        _observation(feature_session, calibrator, model_id, calibration.labels, quat, accel)
        for quat, accel in zip(context_quats[-CONTEXT_FRAMES:], context_accels[-CONTEXT_FRAMES:])
    ]

    poses = {mode: _run_variant(model_id, calibration, context, mode, evaluation_frames)
             for mode in ("upright", "empty", "action")}
    if case_name == "deadbug":
        poses["action+tabletop"] = _apply_start_constraint(
            "deadbug", "tabletop", poses["action"], calibration.labels,
            calibrator, evaluation_frames[0])
    metrics = _metrics(poses)
    metrics.update({
        "case": case_name,
        "model": model_id,
        "calibrationRecording": calibration_name,
        "contextFrames": len(context),
        "evaluationFrames": EVALUATION_FRAMES,
        "evaluationSeconds": EVALUATION_FRAMES / 60.0,
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{case_name}_startup_context.json"
    report_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    from mobileposer.articulate.model import ParametricModel
    body = ParametricModel(paths.smpl_file)
    step = 2
    tracks = {name: joint_positions(body, value, step) for name, value in poses.items()}
    viewer_path = output_dir / f"{case_name}_startup_context.html"
    write_viewer(viewer_path, f"{case_name}: upright vs empty vs action context",
                 60.0 / step, tracks, [], [], live_model=None)
    return report_path, viewer_path, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", metavar="CASE")
    parser.add_argument("--output-dir", type=Path,
                        default=paths.recordings / "startup_context_eval")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--summary-only", action="store_true",
                        help="rebuild summary.json from existing per-case reports")
    args = parser.parse_args()
    unknown = [name for name in args.cases if name not in CASES]
    if unknown:
        parser.error("unknown case(s): " + ", ".join(unknown))
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, args.torch_threads))
    summaries = []
    for name in (() if args.summary_only else (args.cases or CASES)):
        print(f"evaluating {name}...", flush=True)
        report, viewer, metrics = evaluate(name, args.output_dir)
        print(f"  report: {report}\n  viewer: {viewer}", flush=True)
        summaries.append(metrics)
    by_case = {item["case"]: item for item in summaries}
    for name in CASES:
        report = args.output_dir / f"{name}_startup_context.json"
        if name not in by_case and report.exists():
            by_case[name] = json.loads(report.read_text(encoding="utf-8"))
    summary = args.output_dir / "summary.json"
    summary.write_text(json.dumps([by_case[name] for name in CASES if name in by_case],
                                  ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()

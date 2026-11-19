"""Replay the real recordings of each exercise through the current production chain.

For every (action, recording) case three variants are produced with
``production.deadbug_replay.replay`` (same prepare_frame -> infer ->
postprocess_output -> rebaser order as run.py):

* ``new``      : routed action model (ACTION_RUNTIME_MODELS), new start flow
                 (first 60 frames used as the start-pose context, no upright prime),
                 recorded calibration and all action rules.
* ``old``      : previous model for that action, same new start flow and rules.

Metrics are computed on both the raw network output and the final output
(after anchoring / contact / start-pose rules).

There is no ground truth.  Metrics are observation-based:
* ``boneErrDeg``: angle between the predicted bone axis (forearm / shank) and the
  IMU-measured calibrated bone axis, both expressed in the waist frame, so a
  global heading offset does not count.  Large values = output contradicts the sensor.
* ``jitter``: per-prediction joint rotation step (deg), mean over body joints.
* ``kneeFlex`` / ``hipFlex`` / ``trunkPitch`` percentiles describe the range of
  motion actually produced (unobserved thighs are the usual failure mode).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from production.deadbug_replay import replay
from production.model_backends import ACTION_RUNTIME_MODELS
from production.offline_compare import joint_positions, write_viewer
from production.paths import paths
from production.pose_calibration import BONE_AXES, LABEL_JOINT
from production.start_pose_constraint import _globals

CASES = [
    # action, recording, previous model
    ("deadbug", "unity_20261116_045157.jsonl", "waist_imudino_stream_deadbug_rehabv5"),
    ("deadbug", "unity_20261116_063451.jsonl", "waist_imudino_stream_deadbug_rehabv5"),
    ("hipthrust", "unity_20261116_040120.jsonl", "waist_imudino_stream_bridge_rehabv5"),
    ("hipthrust", "unity_20261116_212712.jsonl", "waist_imudino_stream_bridge_rehabv5"),
    ("crunch", "unity_20261116_023352.jsonl", "waist_imudino_stream_bridge_rehabv5_std_20261118"),
    ("lyingcat", "unity_20261116_195301.jsonl", "waist_imudino_stream_birddog_rehabv5_std_20261118"),
    ("birddog", "BirdDog_RingSkinC_Recorded_20261113_005919.jsonl", "waist_imudino_stream_wristankle_limbmix"),
    ("birddog", "BirdDog_RingSkinC_Recorded_20261113_002435.jsonl", "waist_imudino_stream_wristankle_limbmix"),
]
START_POSE = {"deadbug": ("deadbug", "tabletop")}
CONTEXT_FRAMES = 60
SENSED = ("lw", "rw", "ls", "rs")
BODY_JOINTS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
DOWN = np.array([0.0, -1.0, 0.0])


def _angle(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def _rot_angle(a, b):
    rel = np.swapaxes(a, -1, -2) @ b
    return np.degrees(np.arccos(np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def _pct(values):
    values = np.asarray(values)
    return [round(float(np.percentile(values, q)), 1) for q in (10, 50, 90)] if len(values) else None


def metrics(result, local) -> dict:
    labels = result.labels
    glob = np.asarray([np.stack(_globals(p)) for p in local])   # (P,24,3,3)
    frames = np.clip(result.represented_frames, 0, len(result.calibrated_ori) - 1)
    measured = result.calibrated_ori[frames]                                    # (P,n,3,3)
    waist_index = labels.index("waist")
    waist_pred = glob[:, LABEL_JOINT["waist"]]
    waist_meas = measured[:, waist_index]
    out = {"predictions": int(len(glob))}
    bone = {}
    for label in SENSED:
        if label not in labels:
            continue
        axis = BONE_AXES[label]
        pred = np.einsum("pij,pjk,k->pi", np.swapaxes(waist_pred, -1, -2), glob[:, LABEL_JOINT[label]], axis)
        meas = np.einsum("pij,pjk,k->pi", np.swapaxes(waist_meas, -1, -2), measured[:, labels.index(label)], axis)
        err = _angle(pred, meas)
        bone[label] = {"mean": round(float(err.mean()), 1), "p90": round(float(np.percentile(err, 90)), 1)}
    out["boneErrDeg"] = bone
    out["waistGlobalErrDeg"] = round(float(_rot_angle(waist_pred, waist_meas).mean()), 1)
    step = _rot_angle(local[:-1, BODY_JOINTS], local[1:, BODY_JOINTS]).mean(axis=1)
    out["jitterStepDeg"] = {"mean": round(float(step.mean()), 2), "p95": round(float(np.percentile(step, 95)), 2)}
    # Range of motion: knee flexion, hip flexion (thigh vs trunk), trunk pitch from vertical.
    thigh_l, thigh_r = glob[:, 1] @ DOWN, glob[:, 2] @ DOWN
    shank_l, shank_r = glob[:, 4] @ DOWN, glob[:, 5] @ DOWN
    trunk_down = glob[:, 3] @ DOWN    # spine1 "down" axis points pelvis-ward
    out["kneeFlexDeg"] = {"L": _pct(_angle(thigh_l, shank_l)), "R": _pct(_angle(thigh_r, shank_r))}
    out["hipFlexDeg"] = {"L": _pct(_angle(trunk_down, thigh_l)), "R": _pct(_angle(trunk_down, thigh_r))}
    up = -trunk_down
    out["trunkElevationDeg"] = _pct(np.degrees(np.arcsin(np.clip(up[:, 1], -1, 1))))
    return out


def run_case(action, recording, old_model, out_dir, body):
    path = paths.recordings / recording
    new_model = ACTION_RUNTIME_MODELS[action]
    common = dict(start_context_frames=CONTEXT_FRAMES, start_offset_frames=CONTEXT_FRAMES,
                  start_pose_override=START_POSE.get(action))
    variants = {"new": dict(model_id=new_model, **common), "old": dict(model_id=old_model, **common)}
    report = {"action": action, "recording": recording, "newModel": new_model, "oldModel": old_model}
    tracks = {}
    for name, kwargs in variants.items():
        print(f"  {name}: {kwargs['model_id']}", flush=True)
        result = replay(path, **kwargs)
        report[name] = {"network": metrics(result, result.raw_local), "final": metrics(result, result.final_local)}
        # Viewer on the 30 Hz prediction timeline (infer_every=2).
        for stage, local in (("final", result.final_local), ("network", result.raw_local)):
            tracks[f"{name}-{stage}:{kwargs['model_id']}"] = joint_positions(body, local.astype(np.float32), 1)
    report["seconds"] = round(result.n_frames / 60.0, 1)
    stem = Path(recording).stem
    write_viewer(out_dir / f"{action}_{stem}.html", f"{action} {stem}: new vs old, network vs final",
                 60.0 / result.infer_every, tracks, [], [])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", default="", help="comma-separated subset of actions")
    parser.add_argument("--out-dir", type=Path, default=paths.recordings / "action_eval_20261119")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    from mobileposer.articulate.model import ParametricModel
    body = ParametricModel(paths.smpl_file)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(filter(None, args.actions.split(",")))
    reports = []
    for action, recording, old_model in CASES:
        if wanted and action not in wanted:
            continue
        print(f"{action}: {recording}", flush=True)
        reports.append(run_case(action, recording, old_model, args.out_dir, body))
        (args.out_dir / "summary.json").write_text(json.dumps(reports, ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
    print(f"summary: {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

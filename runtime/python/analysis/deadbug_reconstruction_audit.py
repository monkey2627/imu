"""Summarize dead-bug sensor activity and the exact Unity pose saved in recordings."""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from mobileposer.articulate.model import ParametricModel
from production.input_recording import load_input_recording, resolve_recording_path
from production.offline_compare import recorded_prediction_poses, joint_positions
from production.deadbug_prior import DeadBugLegPrior
from production.arm_prior import straighten_reaching_arms
from production.offline_compare import _expand, calibrated_orientations
from production.paths import paths


def angle(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1, 1)))


def audit(name):
    path = resolve_recording_path(name)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rec = load_input_recording(path)
    ticks = [r for r in rows if r.get("type") == "tick"]
    samples = [r["samples"] for r in ticks]
    print(f"\n{path.name}: {sum(rec.ticks)/60:.1f}s, {len(samples)} samples")
    print("events:", [(r.get("name"), r.get("pose"), r.get("outcome")) for r in rows if r.get("type") == "event"])
    print("layout:", rec.label_to_key, "calibration:", rows[0].get("initialCalibration", {}).get("corrections"))
    for label in ("lw", "rw", "ls", "rs", "waist"):
        q = np.asarray([s[label]["quat_xyzw"] for s in samples])
        rot = Rotation.from_quat(q)
        delta = (rot[:-1].inv() * rot[1:]).magnitude()
        print(f"  {label} sensor angular path={np.degrees(delta).sum():.0f} deg, p95 step={np.degrees(delta).max():.1f} deg")

    poses = recorded_prediction_poses(rows, sum(rec.ticks))
    if poses is None:
        print("  no saved Unity predictions")
        return
    body = ParametricModel(paths.smpl_file)
    pos = joint_positions(body, poses, 1)
    # SMPL joints: thighs 1/2 -> knees 4/5 -> ankles 7/8, upper arms 16/17 -> elbows 18/19 -> wrists 20/21.
    for start in range(0, len(pos), 240):
        end = min(start + 240, len(pos))
        p = pos[start:end]
        values = []
        for side, thigh, knee, ankle, shoulder, elbow, wrist in (
                ("L", 1, 4, 7, 16, 18, 20), ("R", 2, 5, 8, 17, 19, 21)):
            knee_flex = angle(p[:, thigh] - p[:, knee], p[:, ankle] - p[:, knee])
            elbow_flex = angle(p[:, shoulder] - p[:, elbow], p[:, wrist] - p[:, elbow])
            upper_arm = p[:, elbow] - p[:, shoulder]
            upper_arm_elev = np.degrees(np.arcsin(np.clip(upper_arm[:, 1] /
                                                np.maximum(np.linalg.norm(upper_arm, axis=1), 1e-8), -1, 1)))
            values.append(f"{side}: knee={np.median(knee_flex):.0f}deg armElev={np.median(upper_arm_elev):+.0f}deg elbow={np.median(elbow_flex):.0f}deg")
        print(f"  {start/60:4.0f}-{end/60:4.0f}s " + " | ".join(values))

    leg_prior = DeadBugLegPrior()
    constrained = np.stack([
        leg_prior.apply(p, dt_seconds=1.0 / 60.0) for p in poses
    ]).astype(np.float32)
    constrained_pos = joint_positions(body, constrained, 1)
    print("  deadbug leg prior knee-angle p10/p50/p90 before -> after:")
    for side, thigh, knee, ankle in (("L", 1, 4, 7), ("R", 2, 5, 8)):
        before = angle(pos[:, thigh] - pos[:, knee], pos[:, ankle] - pos[:, knee])
        after = angle(constrained_pos[:, thigh] - constrained_pos[:, knee],
                      constrained_pos[:, ankle] - constrained_pos[:, knee])
        fmt = lambda x: "/".join(f"{v:.0f}" for v in np.percentile(x, (10, 50, 90)))
        print(f"    {side}: {fmt(before)} -> {fmt(after)} deg")
        original_step = np.degrees((Rotation.from_matrix(poses[:-1, thigh]).inv() *
                                    Rotation.from_matrix(poses[1:, thigh])).magnitude())
        fixed_step = np.degrees((Rotation.from_matrix(constrained[:-1, thigh]).inv() *
                                 Rotation.from_matrix(constrained[1:, thigh])).magnitude())
        print(f"       hip local step p95/max: {np.percentile(original_step,95):.1f}/{original_step.max():.1f}"
              f" -> {np.percentile(fixed_step,95):.1f}/{fixed_step.max():.1f} deg")

    quats, _, events = _expand(rec, rows)
    ori, _, _ = calibrated_orientations(rec, quats, events)
    lw, rw = rec.labels.index("lw"), rec.labels.index("rw")
    arm_weights = np.asarray([
        straighten_reaching_arms(poses[t], (ori[t, lw], ori[t, rw]),
                                 return_weights=True, include_chest_reach=True)[1]
        for t in range(0, len(poses), 5)
    ])
    print("  deadbug arm prior active frames L/R:",
          *(f"{100*np.mean(arm_weights[:, side] > .25):.0f}% (p90 {np.percentile(arm_weights[:,side],90):.2f})"
            for side in range(2)))

    predictions = [r for r in rows if r.get("type") == "prediction"]
    raw = np.asarray([r["rawLocalQuaternions"] for r in predictions], dtype=float).reshape(-1, 24, 4)
    final = np.asarray([r["unityLocalQuaternions"] for r in predictions], dtype=float).reshape(-1, 24, 4)
    print("  postprocess raw->Unity local angle p50/p95:")
    for joint, label in ((1, "left hip"), (2, "right hip"), (4, "left knee"), (5, "right knee"),
                         (16, "left shoulder"), (17, "right shoulder"),
                         (18, "left elbow"), (19, "right elbow")):
        delta = np.degrees((Rotation.from_quat(raw[:, joint]).inv() *
                            Rotation.from_quat(final[:, joint])).magnitude())
        print(f"    {label}: {np.percentile(delta, 50):.1f}/{np.percentile(delta, 95):.1f} deg")


if __name__ == "__main__":
    for recording in sys.argv[1:]:
        audit(recording)

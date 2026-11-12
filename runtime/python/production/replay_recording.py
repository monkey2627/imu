"""Rebuild, compare, and optionally broadcast a realtime JSONL recording."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from runtime_contract import FPS as MODEL_FPS
from production.calibration import HeadingCalibration, calibrate_heading
from production.inference import (
    DEFAULT_CHECKPOINT_ROOT, DEFAULT_MODEL, OUTPUT_LAG_FRAMES, InferenceSession, load_model,
)
from production.layout import ResolvedLayout
from production.recording import load_recording
from production.rotmat import batch_smpl_matrix_to_unity_quat_xyzw
from production.stream_out import PoseBroadcaster


def _load_heading(metadata, rows, mode: str) -> HeadingCalibration:
    labels = list(metadata["labels"])
    if mode == "saved":
        row = next((row for row in rows if row.get("type") == "heading"), None)
        if row is None:
            raise ValueError("recording has no saved heading record")
        return HeadingCalibration(
            labels,
            torch.tensor(row["matrices"], dtype=torch.float32),
            row.get("diagnostics", {}),
        )
    samples = [row for row in rows if row.get("type") == "calibrationSample"]
    if not samples:
        raise ValueError("recording has no calibration samples for heading recomputation")
    by_key = metadata["labelToTrackerKey"]
    raw = {label: [] for label in labels}
    adjusted = {label: [] for label in labels}
    for row in samples:
        trackers = {sample["tracker_key"]: sample for sample in row["trackers"]}
        if any(by_key[label] not in trackers for label in labels):
            continue
        for label in labels:
            sample = trackers[by_key[label]]
            raw[label].append(sample["raw_quat_xyzw"])
            adjusted[label].append(sample["quat_xyzw"])
    missing = [label for label in labels if not raw[label]]
    if missing:
        raise ValueError(f"recording has no complete calibration samples for: {missing}")
    return calibrate_heading(
        labels,
        {label: np.asarray(raw[label], dtype=np.float32) for label in labels},
        {label: np.asarray(adjusted[label], dtype=np.float32) for label in labels},
    )


def replay_arrays(args):
    metadata, rows = load_recording(args.recording)
    if metadata.get("modelFps") != MODEL_FPS:
        raise ValueError(
            f"{args.recording} was recorded with modelFps={metadata.get('modelFps')!r} (legacy 30 Hz / "
            f"old-placement model); the current runtime window is {MODEL_FPS} Hz and cannot replay it faithfully."
        )
    # Every 60 Hz window advance is a windowFrame/inferenceFrame row; only inferenceFrame rows run the network.
    frames = [row for row in rows if row.get("type") in ("windowFrame", "inferenceFrame")]
    if not any(row["type"] == "inferenceFrame" for row in frames):
        raise ValueError("recording contains no inference frames")
    labels = list(metadata["labels"])
    resolved = ResolvedLayout(metadata["layout"], labels, dict(metadata["labelToTrackerKey"]))
    heading = _load_heading(metadata, rows, args.heading)
    checkpoint = args.checkpoint or (Path(metadata["checkpoint"]) if metadata.get("checkpoint") else None)
    model = load_model(checkpoint or DEFAULT_MODEL, args.checkpoint_root, anchor=metadata.get("anchored"))
    session = InferenceSession(resolved, model, heading)
    accel_field = ("referenceAdjustedLinearAccelerationMps2"
                   if metadata.get("accelerationSourceResolved") == "server" else "linearAccelerationMps2")
    upright = next((row for row in rows if row.get("type") == "upright"), None)
    if upright is not None:
        session.prime(
            np.asarray(upright["referenceAdjustedQuaternions"], dtype=np.float32),
            np.asarray(upright["linearAccelerationMps2"], dtype=np.float32),
        )

    quats = []
    report_rows = []
    for row in frames:
        trackers = {tracker["label"]: tracker for tracker in row["trackers"]}
        adjusted = np.asarray([trackers[label]["referenceAdjustedQuaternion"] for label in labels], dtype=np.float32)
        accel = np.asarray([trackers[label][accel_field] for label in labels], dtype=np.float32)
        if args.disable_acceleration:
            accel.fill(0)
        feature, _, _ = session.prepare_feature(adjusted, accel)
        if args.disable_orientation:
            feature[15:] = np.tile(np.eye(3, dtype=np.float32).reshape(-1), len(labels))
        for _ in range(int(row.get("ticksAdvanced", 1))):
            session.push_feature(feature)
        if row["type"] != "inferenceFrame":
            continue
        pose = session.infer().detach().numpy()
        unity = batch_smpl_matrix_to_unity_quat_xyzw(pose)
        quats.append(unity)
        saved_feature = np.asarray(row["feature"], dtype=np.float32)
        saved_pose = np.asarray(row["smplLocalRotationMatrices"], dtype=np.float32)
        relative_rotation = (
            Rotation.from_matrix(pose).inv()
            * Rotation.from_matrix(saved_pose)
        )
        relative_rotation_degrees = np.rad2deg(relative_rotation.magnitude())
        report_rows.append({
            "frame": row["frame"],
            "featureMaxAbs": float(np.max(np.abs(feature - saved_feature))),
            "featureRmse": float(np.sqrt(np.mean((feature - saved_feature) ** 2))),
            "poseMaxAbs": float(np.max(np.abs(pose - saved_pose))),
            "poseRmse": float(np.sqrt(np.mean((pose - saved_pose) ** 2))),
            "poseMaxRotationDegrees": float(np.max(relative_rotation_degrees)),
            "poseMeanRotationDegrees": float(np.mean(relative_rotation_degrees)),
        })
    summary = {
        "recording": str(args.recording),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "headingMode": args.heading,
        "frames": len(report_rows),
        "featureMaxAbs": max(row["featureMaxAbs"] for row in report_rows),
        "poseMaxAbs": max(row["poseMaxAbs"] for row in report_rows),
        "featureRmse": float(np.mean([row["featureRmse"] for row in report_rows])),
        "poseRmse": float(np.mean([row["poseRmse"] for row in report_rows])),
        "poseMaxRotationDegrees": max(row["poseMaxRotationDegrees"] for row in report_rows),
        "poseMeanRotationDegrees": float(np.mean([row["poseMeanRotationDegrees"] for row in report_rows])),
        "perFrame": report_rows,
    }
    return metadata, np.asarray(quats), summary


async def main_async(args):
    metadata, quats, summary = replay_arrays(args)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "perFrame"}, indent=2))
    if not args.broadcast:
        return
    broadcaster = PoseBroadcaster()
    server = await broadcaster.serve(args.host, args.port)
    interval = 1.0 / max(args.fps, 1.0)
    try:
        while True:
            for index, frame in enumerate(quats):
                started = time.perf_counter()
                await broadcaster.broadcast(
                    index, metadata["layout"] + "-jsonl-replay", frame,
                    diagnostics={
                        "sourceUpdate": index,
                        "modelInputFrame": index,
                        "modelOutputRepresentsFrame": (
                            index - OUTPUT_LAG_FRAMES
                            if index >= OUTPUT_LAG_FRAMES else -1
                        ),
                        "modelLagFrames": OUTPUT_LAG_FRAMES,
                    })
                await asyncio.sleep(max(0.0, interval - (time.perf_counter() - started)))
            if not args.loop:
                break
    finally:
        server.close()
        await server.wait_closed()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--heading", choices=("saved", "recompute"), default="saved")
    parser.add_argument("--disable-acceleration", action="store_true")
    parser.add_argument("--disable-orientation", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--broadcast", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=21200)
    parser.add_argument("--fps", type=float, default=30.0, help="broadcast rate of replayed poses (recording infers at modelFps/inferEvery)")
    parser.add_argument("--loop", action="store_true")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()

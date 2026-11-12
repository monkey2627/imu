"""Export the reconstructed pose stored in a Unity input recording to MP4.

The recording's ``unityLocalQuaternions`` are the final post-processed poses
that were sent to Unity, so this exporter does not rerun the inference model.
The default output is a skeleton; pass ``--mesh`` for a shaded SMPL surface.

Example:
    python scripts/export_reconstruction_video.py session.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mobileposer.articulate.model import ParametricModel
from production.input_recording import resolve_recording_path
from production.paths import paths
from production.pose_calibration import SMPL_PARENTS
from production.rotmat import UNITY_SMPL_BASIS


LEFT_JOINTS = {1, 4, 7, 10, 13, 16, 18, 20, 22}
BACKGROUND = (24, 19, 16)
LEFT_COLOR = (255, 183, 112)
RIGHT_COLOR = (101, 138, 255)
JOINT_COLOR = (243, 230, 216)
ROOT_COLOR = (78, 193, 242)
MESH_COLOR = np.array([214, 158, 92], dtype=np.float32)


def load_predictions(path: Path, field: str) -> tuple[np.ndarray, np.ndarray, dict]:
    times, quaternions = [], []
    header = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") == "header":
                header = row
            if row.get("type") != "prediction" or field not in row:
                continue
            values = np.asarray(row[field], dtype=np.float32)
            if values.size != 24 * 4:
                raise ValueError(f"expected 96 quaternion values, got {values.size}")
            times.append(float(row["t"]))
            quaternions.append(values.reshape(24, 4))
    if header is None:
        raise ValueError("recording has no header")
    if not times:
        raise ValueError(f"recording has no prediction rows containing {field}")
    return np.asarray(times), np.asarray(quaternions), header


def rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    # Both quaternion fields in the input recording are already in Unity's
    # left-handed convention.  run.py creates them by conjugating each SMPL
    # local rotation with UNITY_SMPL_BASIS before broadcasting.  Apply the
    # same self-inverse basis change here to recover the SMPL matrices used by
    # the model; feeding the Unity matrices directly into SMPL twists the body.
    unity = Rotation.from_quat(quaternions.reshape(-1, 4)).as_matrix()
    smpl = UNITY_SMPL_BASIS @ unity @ UNITY_SMPL_BASIS
    return smpl.reshape(len(quaternions), 24, 3, 3).astype(np.float32)


def joint_positions(quaternions: np.ndarray, body: ParametricModel | None = None) -> np.ndarray:
    rotations = rotation_matrices(quaternions)
    body = body or ParametricModel(paths.smpl_file)
    with torch.no_grad():
        _, joints = body.forward_kinematics(torch.from_numpy(rotations))[:2]
    joints = joints.numpy()
    return joints - joints[:, :1]


def resample(times: np.ndarray, joints: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    duration = float(times[-1])
    output_times = np.arange(0.0, duration + 0.5 / fps, 1.0 / fps)
    indices = np.searchsorted(times, output_times, side="right") - 1
    indices = np.clip(indices, 0, len(times) - 1)
    return output_times, joints[indices]


def camera_transform(points: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    x = cy * points[..., 0] + sy * points[..., 2]
    z = -sy * points[..., 0] + cy * points[..., 2]
    y = cp * points[..., 1] - sp * z
    depth = sp * points[..., 1] + cp * z
    return np.stack((x, y, depth), axis=-1)


def draw_frame(
    projected: np.ndarray,
    time_seconds: float,
    duration: float,
    title: str,
    width: int,
    height: int,
    scale: float,
) -> np.ndarray:
    image = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    center = np.array([width * 0.5, height * 0.52])
    pixels = np.empty((24, 2), dtype=np.float32)
    pixels[:, 0] = center[0] + projected[:, 0] * scale
    pixels[:, 1] = center[1] - projected[:, 1] * scale

    floor_y = int(height * 0.86)
    cv2.line(image, (int(width * 0.12), floor_y), (int(width * 0.88), floor_y), (54, 48, 43), 1, cv2.LINE_AA)
    order = sorted(
        range(1, 24),
        key=lambda joint: projected[joint, 2] + projected[SMPL_PARENTS[joint], 2],
    )
    for joint in order:
        parent = SMPL_PARENTS[joint]
        a = tuple(np.rint(pixels[parent]).astype(int))
        b = tuple(np.rint(pixels[joint]).astype(int))
        color = LEFT_COLOR if joint in LEFT_JOINTS else RIGHT_COLOR
        cv2.line(image, a, b, color, 8, cv2.LINE_AA)
    for joint, point in enumerate(pixels):
        color = ROOT_COLOR if joint == 0 else JOINT_COLOR
        radius = 7 if joint == 0 else 5
        cv2.circle(image, tuple(np.rint(point).astype(int)), radius, color, -1, cv2.LINE_AA)

    cv2.putText(image, title, (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (232, 237, 242), 2, cv2.LINE_AA)
    stamp = f"{time_seconds:05.1f} / {duration:05.1f} s"
    cv2.putText(image, stamp, (28, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (167, 177, 188), 1, cv2.LINE_AA)
    progress = 0.0 if duration <= 0 else time_seconds / duration
    cv2.line(image, (220, height - 35), (width - 28, height - 35), (56, 62, 70), 4, cv2.LINE_AA)
    cv2.line(image, (220, height - 35), (int(220 + (width - 248) * progress), height - 35), ROOT_COLOR, 4, cv2.LINE_AA)
    return image


def draw_mesh_frame(
    projected: np.ndarray,
    faces: np.ndarray,
    time_seconds: float,
    duration: float,
    title: str,
    width: int,
    height: int,
    scale: float,
) -> np.ndarray:
    image = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    center = np.array([width * 0.5, height * 0.52])
    pixels = np.empty((len(projected), 2), dtype=np.float32)
    pixels[:, 0] = center[0] + projected[:, 0] * scale
    pixels[:, 1] = center[1] - projected[:, 1] * scale
    pixels = np.rint(pixels).astype(np.int32)

    triangles = projected[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal_length = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(normal_length[:, None], 1e-8)
    light = np.array([-0.35, 0.75, 0.55], dtype=np.float32)
    light /= np.linalg.norm(light)
    intensity = 0.30 + 0.70 * np.abs(normals @ light)
    colors = np.clip(MESH_COLOR[None] * intensity[:, None], 0, 255).astype(np.uint8)
    order = np.argsort(triangles[..., 2].mean(axis=1))
    for face_index in order:
        color = tuple(int(value) for value in colors[face_index])
        cv2.fillConvexPoly(image, pixels[faces[face_index]], color, lineType=cv2.LINE_AA)

    floor_y = int(height * 0.86)
    cv2.line(image, (int(width * 0.12), floor_y), (int(width * 0.88), floor_y), (54, 48, 43), 1, cv2.LINE_AA)
    cv2.putText(image, title, (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (232, 237, 242), 2, cv2.LINE_AA)
    stamp = f"{time_seconds:05.1f} / {duration:05.1f} s"
    cv2.putText(image, stamp, (28, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (167, 177, 188), 1, cv2.LINE_AA)
    progress = 0.0 if duration <= 0 else time_seconds / duration
    cv2.line(image, (220, height - 35), (width - 28, height - 35), (56, 62, 70), 4, cv2.LINE_AA)
    cv2.line(image, (220, height - 35), (int(220 + (width - 248) * progress), height - 35), ROOT_COLOR, 4, cv2.LINE_AA)
    return image


def export_video(
    recording: Path,
    output: Path,
    fps: float,
    width: int,
    height: int,
    field: str,
    yaw: float,
    pitch: float,
    mesh: bool,
) -> None:
    times, quaternions, header = load_predictions(recording, field)
    output_times, quaternions = resample(times, quaternions, fps)
    body = ParametricModel(paths.smpl_file)
    joints = joint_positions(quaternions, body)
    viewed = camera_transform(joints, np.radians(yaw), np.radians(pitch))

    extent_x = np.percentile(np.abs(viewed[..., 0]), 99.5)
    extent_y = np.percentile(np.abs(viewed[..., 1]), 99.5)
    scale = min(width * 0.41 / max(extent_x, 1e-6), height * 0.40 / max(extent_y, 1e-6))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 video writer")
    title = ("SMPL | " if mesh else "") + (header.get("model") or "MotionRecover reconstruction")
    duration = float(output_times[-1])
    try:
        if mesh:
            faces = np.asarray(body.face, dtype=np.int32)
            rotations = rotation_matrices(quaternions)
            chunk_size = 24
            for start in range(0, len(rotations), chunk_size):
                stop = min(start + chunk_size, len(rotations))
                with torch.no_grad():
                    vertices = body.forward_kinematics(
                        torch.from_numpy(rotations[start:stop]), calc_mesh=True
                    )[2].numpy()
                vertices = camera_transform(vertices, np.radians(yaw), np.radians(pitch))
                for offset, frame in enumerate(vertices):
                    index = start + offset
                    writer.write(draw_mesh_frame(
                        frame, faces, float(output_times[index]), duration, title, width, height, scale
                    ))
                print(f"rendered {stop}/{len(rotations)} frames", flush=True)
        else:
            for time_seconds, frame in zip(output_times, viewed):
                writer.write(draw_frame(frame, float(time_seconds), duration, title, width, height, scale))
    finally:
        writer.release()
    print(f"video: {output}")
    print(f"frames: {len(output_times)}, fps: {fps:g}, duration: {duration:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", help="recording name or JSONL path")
    parser.add_argument("--out", type=Path, help="output MP4 path")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--raw", action="store_true",
        help="use the pre-anchor/pre-arm-prior pose instead of the final Unity output",
    )
    parser.add_argument("--mesh", action="store_true", help="render the full shaded SMPL body surface")
    parser.add_argument("--yaw", type=float, default=-28.0, help="camera yaw in degrees")
    parser.add_argument("--pitch", type=float, default=9.0, help="camera pitch in degrees")
    args = parser.parse_args()
    recording = resolve_recording_path(args.recording)
    suffix = "_smpl.mp4" if args.mesh else "_reconstruction.mp4"
    output = args.out or recording.with_name(recording.stem + suffix)
    field = "rawLocalQuaternions" if args.raw else "unityLocalQuaternions"
    export_video(recording, output, args.fps, args.width, args.height, field, args.yaw, args.pitch, args.mesh)


if __name__ == "__main__":
    main()

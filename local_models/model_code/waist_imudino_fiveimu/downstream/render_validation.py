"""Render BirdDog held-out predictions as 3-view skeleton overlays."""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw

from downstream.model import FiveIMUHead


def _window_starts(frames: int, length: int = 480, stride: int = 240) -> list[int]:
    """Return the same overlapping starts used by the feature-cache builder."""
    if frames < 16:
        return []
    starts = list(range(0, max(frames - length + 1, 1), stride))
    final_start = max(0, frames - length)
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _sixd_to_matrix(values: torch.Tensor) -> torch.Tensor:
    """Convert Zhou 6D column vectors to orthonormal rotation matrices."""
    # The training target stores the first two matrix columns consecutively.
    first = values[..., :3]
    second = values[..., 3:6]
    column1 = torch.nn.functional.normalize(first, dim=-1, eps=1e-8)
    second = second - (column1 * second).sum(-1, keepdim=True) * column1
    column2 = torch.nn.functional.normalize(second, dim=-1, eps=1e-8)
    column3 = torch.cross(column1, column2, dim=-1)
    return torch.stack((column1, column2, column3), dim=-1)


def _draw_view(canvas: Image.Image, joints: np.ndarray, parents: list[int], view: str,
               limits: tuple[float, float, float, float], color: tuple[int, int, int],
               label: str) -> None:
    """Draw one projected SMPL skeleton into a PIL RGB canvas."""
    if view == "front":
        axes, title, panel = (0, 1), "FRONT  (X / Y)", 0
    elif view == "side":
        axes, title, panel = (2, 1), "SIDE  (Z / Y)", 1
    else:
        axes, title, panel = (0, 2), "TOP  (X / Z)", 2
    x_min, x_max, y_min, y_max = limits
    panel_w, panel_h = canvas.width // 3, canvas.height
    margin = 42
    draw_w, draw_h = panel_w - 2 * margin, panel_h - 2 * margin
    origin = panel_w * panel
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((origin, 0, origin + panel_w - 1, panel_h - 1), outline=(48, 52, 60), width=1)
    draw.text((origin + 14, 12), title, fill=(220, 225, 235))

    def xy(point: np.ndarray) -> tuple[int, int]:
        px = origin + margin + int((point[axes[0]] - x_min) / max(x_max - x_min, 1e-5) * draw_w)
        py = panel_h - margin - int((point[axes[1]] - y_min) / max(y_max - y_min, 1e-5) * draw_h)
        return px, py

    points = [xy(point) for point in joints]
    for child in range(1, min(len(parents), len(points))):
        parent = parents[child]
        if parent is not None and 0 <= parent < len(points):
            draw.line((points[parent], points[child]), fill=color, width=3)
    for point in points:
        x, y = point
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    draw.text((origin + 14, panel_h - 24), label, fill=color)


def _group_limits(frames: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, tuple[float, float, float, float]]:
    """Choose stable, shared axes for all ground-truth and predicted frames."""
    joined = np.concatenate([np.concatenate(pair, axis=0).reshape(-1, 3) for pair in frames], axis=0)
    limits = {}
    for key, axes in (("front", (0, 1)), ("side", (2, 1)), ("top", (0, 2))):
        lower = np.percentile(joined[:, axes], 1.0, axis=0)
        upper = np.percentile(joined[:, axes], 99.0, axis=0)
        center = (lower + upper) * 0.5
        span = max(float((upper - lower).max()), 0.8) * 1.12
        limits[key] = (center[0] - span / 2, center[0] + span / 2,
                       center[1] - span / 2, center[1] + span / 2)
    return limits


def _render_group(group: str, clips: list[dict], parents: list[int], output: Path,
                  fps: int = 15, source_fps: int = 60) -> dict:
    """Write one validation source-video group as a three-view GT/pred overlay."""
    width, height = 1440, 480
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
        "-framerate", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    stride = max(1, round(source_fps / fps))
    frame_count = 0
    try:
        for clip in clips:
            gt, pred = clip["gt"], clip["pred"]
            n = min(len(gt), len(pred))
            for frame_index in range(0, n, stride):
                canvas = Image.new("RGB", (width, height), (18, 20, 25))
                draw = ImageDraw.Draw(canvas)
                title = (f"IMUDINO validation | group={group[7:15]} | clip={clip['clip_index']} "
                         f"| frame={frame_index}/{n} | t={frame_index / source_fps:.2f}s")
                draw.text((18, 48), title, fill=(245, 245, 245))
                for view in ("front", "side", "top"):
                    _draw_view(canvas, gt[frame_index], parents, view, clip["limits"][view],
                               (55, 205, 255), "Pseudo-label")
                    _draw_view(canvas, pred[frame_index], parents, view, clip["limits"][view],
                               (255, 130, 70), "IMUDINO prediction")
                draw.text((18, 78), "Root rotation and translation anchored to pseudo-label",
                          fill=(185, 190, 200))
                process.stdin.write(canvas.tobytes())
                frame_count += 1
    finally:
        if process.stdin:
            process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed ({return_code}): {stderr}")
    return {"group": group, "clips": len(clips), "rendered_frames": frame_count,
            "fps": fps, "path": str(output)}



def _project_joints(joints: np.ndarray, camera: np.ndarray,
                    intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project Y-up SMPL joints with DuoMo's camera-to-world transform."""
    rotation = camera[:, :3]
    translation = camera[:, 3]
    camera_xyz = (joints - translation) @ rotation
    depth = camera_xyz[:, 2]
    safe_depth = np.clip(depth, 1e-4, None)
    pixels = camera_xyz[:, :2] / safe_depth[:, None]
    pixels = pixels * np.array([intrinsic[0, 0], intrinsic[1, 1]]) + intrinsic[:2, 2]
    return pixels, depth


def _draw_projected(draw: ImageDraw.ImageDraw, points: np.ndarray, depth: np.ndarray,
                    parents: list[int], color: tuple[int, int, int], width: int,
                    canvas_size: tuple[int, int]) -> None:
    """Draw visible projected skeleton bones on a video frame."""
    w, h = canvas_size
    visible = (depth > 1e-3) & (points[:, 0] > -w) & (points[:, 0] < 2 * w)
    visible &= (points[:, 1] > -h) & (points[:, 1] < 2 * h)
    for child in range(1, min(len(parents), len(points))):
        parent = parents[child]
        if parent is not None and visible[child] and visible[parent]:
            draw.line((tuple(points[parent]), tuple(points[child])), fill=color, width=width)
    for index, point in enumerate(points):
        if visible[index]:
            x, y = map(int, point)
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)


def _render_rgb_job(job_id: str, clips: list[dict], parents: list[int],
                    output: Path, web_root: Path) -> dict:
    """Overlay validation and pseudo-label skeletons on the source RGB video."""
    import cv2

    job_dir = web_root / job_id
    video_path = job_dir / "input/duomo/track.mp4"
    camera_path = job_dir / "input/duomo/out/preprocess/camera.pt"
    if not video_path.is_file() or not camera_path.is_file():
        raise FileNotFoundError(f"Missing source video or camera calibration for {job_id}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not read source video: {video_path}")
    width0 = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height0 = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width, height = width0 // 2, height0 // 2
    width, height = width - width % 2, height - height % 2
    sx, sy = width / width0, height / height0
    camera_data = torch.load(camera_path, map_location="cpu", weights_only=False)
    intrinsics = camera_data["cam_int"].cpu().numpy()
    frame_overlays: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]] = defaultdict(list)

    # Map each 60 Hz pose clip onto its aligned 30 Hz camera/video frames.
    for clip in clips:
        clip_index = int(clip["clip_index"])
        duo_path = job_dir / f"input/duomo/out/clip{clip_index:03d}_duomo.pt"
        duo = torch.load(duo_path, map_location="cpu", weights_only=False)
        cameras = duo["cam_motion"].cpu().numpy()
        global_start = int(clip["source_start"])
        count = min(len(cameras), (len(clip["gt"]) + 1) // 2)
        for local_frame in range(count):
            pose_frame = min(local_frame * 2, len(clip["gt"]) - 1)
            video_frame = global_start + local_frame
            if video_frame >= len(intrinsics):
                continue
            K = intrinsics[video_frame].copy()
            K[0, :] *= sx
            K[1, :] *= sy
            gt_2d, gt_depth = _project_joints(clip["gt"][pose_frame], cameras[local_frame], K)
            pred_2d, pred_depth = _project_joints(clip["pred"][pose_frame], cameras[local_frame], K)
            frame_overlays[video_frame].append((gt_2d, gt_depth, pred_2d, pred_depth, clip_index))

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
        "-framerate", str(source_fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_index = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
            canvas = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(canvas)
            overlays = frame_overlays.get(frame_index, [])
            for gt_2d, gt_depth, pred_2d, pred_depth, clip_index in overlays:
                _draw_projected(draw, gt_2d, gt_depth, parents, (70, 230, 90), 3, (width, height))
                _draw_projected(draw, pred_2d, pred_depth, parents, (255, 90, 210), 2, (width, height))
            draw.rectangle((8, 8, min(width - 8, 575), 70), fill=(0, 0, 0))
            draw.text((18, 14), f"IMUDINO BirdDog validation | frame {frame_index}", fill=(255, 255, 255))
            draw.text((18, 38), "Green: pseudo-label   Magenta: IMUDINO prediction", fill=(245, 245, 245))
            draw.text((18, 56), "Root rotation/translation aligned to pseudo-label", fill=(200, 200, 200))
            encoder.stdin.write(canvas.tobytes())
            frame_index += 1
    finally:
        capture.release()
        if encoder.stdin:
            encoder.stdin.close()
    stderr = encoder.stderr.read().decode("utf-8", errors="replace")
    code = encoder.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed for {job_id} ({code}): {stderr}")
    return {
        "job_id": job_id,
        "source_video": json.loads((job_dir / "job.json").read_text())["source_path"],
        "output": str(output),
        "frames": frame_index,
        "fps": source_fps,
        "overlaid_frames": len(frame_overlays),
        "resolution": [width, height],
    }


def main() -> None:
    """Render every source-video group held out from the BirdDog training fold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    repo = args.repo
    cache_path = repo / "data/features/five_imu_v1/BirdDog.features.pt"
    processed_path = repo / "data/processed/wrists_shanks_waist/BirdDog.pt"
    checkpoint_path = repo / "runs/five_imu/best.pt"
    output_dir = args.output or repo / "runs/five_imu/validation_overlays"
    for path in (cache_path, processed_path, checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    # Load the exact cached validation features, source sequences, and best head.
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    source = torch.load(processed_path, map_location="cpu", weights_only=False)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state["epoch"] != 30 or state["sensor_order"] != ["lw", "rw", "ls", "rs", "waist"]:
        raise ValueError("Unexpected checkpoint contract")
    device = torch.device(args.device)
    head = FiveIMUHead().to(device).eval()
    head.load_state_dict(state["head"])

    # Map sorted source clips to the SHA groups used for the validation split.
    raw_root = Path("/home/duanyuhan/dyh/motion/flowhmr/output/training_data/BirdDog")
    raw_files = sorted(raw_root.glob("*/*_poses.npz"))
    if len(raw_files) != len(source["pose"]):
        raise ValueError("Raw and processed BirdDog sequence ordering differs")
    sequence_groups = []
    sequence_metadata = []
    for raw_file, pose in zip(raw_files, source["pose"]):
        sidecar = raw_file.with_name(raw_file.name.replace("_poses.npz", ".json"))
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if int(metadata["num_frames"]) != len(pose):
            raise ValueError(f"Frame mismatch in {raw_file}")
        sequence_groups.append(f"BirdDog:{metadata['source_video_sha256']}")
        sequence_metadata.append(metadata)

    # Check that the renderer applies the same deterministic held-out rule.
    expected_folds = torch.tensor([
        int.from_bytes(hashlib.sha256(f"42:{group}".encode()).digest()[:8], "big") % 10 == 0
        for group in cache["groups"]
    ], dtype=torch.bool)
    if not torch.equal(expected_folds, cache["folds"]):
        raise ValueError("Feature-cache validation folds do not match source-video groups")

    # Load the original SMPL implementation used to synthesize both datasets.
    sys.path.insert(0, "/home/duanyuhan/dyh/motion/lines/realuse_waist_surface_20261030/pnp_workspace")
    from mobileposer.articulate.model import ParametricModel
    from mobileposer.config import paths
    body = ParametricModel(paths.smpl_file)
    print(f"[render] cache_windows={len(cache['features'])} validation_windows={int(cache['folds'].sum())}", flush=True)

    # The cache rows follow sequence order and window order. Predict each held-out clip.
    row = 0
    group_clips: dict[str, list[dict]] = defaultdict(list)
    total_windows = 0
    for sequence_index, (pose, shape, tran, group, metadata) in enumerate(
        zip(source["pose"], source["shape"], source["tran"], sequence_groups, sequence_metadata)
    ):
        starts = _window_starts(len(pose))
        count = len(starts)
        if not count:
            continue
        rows = slice(row, row + count)
        current_groups = cache["groups"][rows]
        if any(value != group for value in current_groups):
            raise ValueError(f"Feature row order/group mismatch at sequence {sequence_index}")
        validation = bool(cache["folds"][row])
        if not torch.all(cache["folds"][rows] == validation):
            raise ValueError(f"One source clip crosses train/validation: sequence {sequence_index}")
        total_windows += count
        row += count
        if not validation:
            continue
        print(f"[render] sequence={sequence_index} frames={len(pose)} windows={count} group={group[8:16]}", flush=True)

        features = cache["features"][rows].to(device)
        with torch.inference_mode():
            prediction_6d = head(features, frames=480).float().cpu()
        # Stitch overlap by averaging the predicted 6D columns, then re-orthogonalize.
        prediction_sum = torch.zeros((len(pose), 23, 6), dtype=torch.float32)
        prediction_count = torch.zeros((len(pose), 1, 1), dtype=torch.float32)
        for local, start in enumerate(starts):
            valid = min(480, len(pose) - start)
            window_prediction = prediction_6d[local, :valid].reshape(valid, 23, 6)
            prediction_sum[start:start + valid] += window_prediction
            prediction_count[start:start + valid] += 1
        prediction_6d_full = prediction_sum / prediction_count.clamp_min(1)
        predicted_children = _sixd_to_matrix(prediction_6d_full)
        predicted_pose = pose.clone()
        predicted_pose[:, 1:] = predicted_children

        # Root orientation and translation are kept at pseudo-label values because
        # the trained target is articulated joints 1..23 only.
        gt_pose = pose.float()
        pred_pose = predicted_pose.float()
        clip_shape = shape.float().view(1, 10).expand(len(pose), -1)
        print(f"[render] forward_kinematics sequence={sequence_index}", flush=True)
        _, gt_joints = body.forward_kinematics(gt_pose, shape=clip_shape, tran=tran.float())
        _, pred_joints = body.forward_kinematics(pred_pose, shape=clip_shape, tran=tran.float())
        gt_array = gt_joints.numpy()
        pred_array = pred_joints.numpy()

        clip = {
            "clip_index": int(metadata["clip_index"]),
            "job_id": metadata["job_id"],
            "source_start": int(metadata["source_video_frames_30fps"][0]),
            "gt": gt_array,
            "pred": pred_array,
        }
        clip["limits"] = _group_limits([(gt_array, pred_array)])
        group_clips[group].append(clip)
        print(f"[render] sequence_done={sequence_index}", flush=True)

    if row != len(cache["features"]) or total_windows != len(cache["features"]):
        raise ValueError(f"Feature-cache window count mismatch: consumed {row}/{len(cache['features'])}")

    reports = []
    for group, clips in sorted(group_clips.items()):
        output = output_dir / f"BirdDog_val_{group[8:16]}_3view.mp4"
        print(f"[render] writing_group={group[8:16]} clips={len(clips)} path={output}", flush=True)
        reports.append(_render_group(group, clips, [None] + list(body.parent[1:24]), output))
        print(json.dumps(reports[-1], sort_keys=True), flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs: dict[str, list[dict]] = defaultdict(list)
    for clips in group_clips.values():
        for clip in clips:
            jobs[clip["job_id"]].append(clip)
    web_root = Path("/home/duanyuhan/dyh/motion/flowhmr/output/web_jobs")
    rgb_reports = []
    for job_id, clips in sorted(jobs.items()):
        output = output_dir / f"BirdDog_val_{job_id}_rgb_overlay.mp4"
        rgb_reports.append(_render_rgb_job(job_id, clips, [None] + list(body.parent[1:24]), output, web_root))
        print("[rgb overlay] " + json.dumps(rgb_reports[-1], sort_keys=True), flush=True)
    (output_dir / "index.json").write_text(json.dumps({
        "checkpoint": str(checkpoint_path),
        "validation_source_video_groups": len(group_clips),
        "validation_windows": int(cache["folds"].sum()),
        "root_alignment": "pseudo-label root rotation and translation",
        "joint_colors_rgb": {"pseudo_label": [55, 205, 255], "prediction": [255, 130, 70]},
        "videos": reports,
        "rgb_overlay_videos": rgb_reports,
    }, indent=2) + chr(10), encoding="utf-8")
    print(f"[render complete] {len(reports)} videos under {output_dir}", flush=True)


if __name__ == "__main__":
    main()

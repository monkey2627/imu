#!/usr/bin/env python3
"""Retarget one Unity HumanPose export and align it to a physical IMU timeline."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--humanpose", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("global", "calibrated"), default="calibrated")
    args = parser.parse_args()
    sys.path.insert(0, str(args.workspace.resolve()))
    from mobileposer.retarget_ours_humanpose import retarget_cache
    from mobileposer.articulate.model import ParametricModel
    from mobileposer.config import paths

    package = json.loads(args.humanpose.read_text(encoding="utf-8"))
    if package.get("frameCount") != len(package.get("frames", [])):
        raise ValueError("HumanPose frameCount mismatch")
    if len(package.get("frames", [])) < 2 or len(package.get("bones", [])) != 18:
        raise ValueError("Unexpected HumanPose export shape")
    if package.get("coordinateSystem") != "unity_world_xyz":
        raise ValueError("HumanPose export is not in the expected Unity world coordinate system")
    if any(len(frame.get("positions", [])) != 18 or not all(frame.get("present", []))
           for frame in package["frames"]):
        raise ValueError("HumanPose export has incomplete bone positions")
    expected_source = hashlib.sha256(args.humanpose.read_bytes()).hexdigest()
    physical = torch.load(args.input, map_location="cpu", weights_only=False)
    if physical["fps"] != 60 or physical["timestamps"].ndim != 1:
        raise ValueError("Physical input must have a 60 Hz timestamp vector")

    pose30, tran30, shape, _, calibration = retarget_cache(args.humanpose, mode=args.mode)
    source_times = np.asarray([f["time"] for f in package["frames"]], dtype=np.float64)
    target_times = physical["timestamps"].double().numpy()
    if len(pose30) != len(source_times) or np.any(np.diff(source_times) <= 0):
        raise ValueError("Retargeted pose and source timestamps do not match")
    if target_times[0] < source_times[0] - 1e-5 or target_times[-1] > source_times[-1] + 1 / 30:
        raise ValueError("HumanPose sequence does not cover physical IMU timeline")
    target_times = np.clip(target_times, source_times[0], source_times[-1])

    pose_np = pose30.detach().cpu().numpy()
    pose60 = np.stack([
        Slerp(source_times, Rotation.from_matrix(pose_np[:, joint]))(target_times).as_matrix()
        for joint in range(24)
    ], axis=1)
    tran_np = tran30.detach().cpu().numpy()
    tran60 = np.stack([
        np.interp(target_times, source_times, tran_np[:, axis]) for axis in range(3)
    ], axis=-1)
    pose60 = torch.from_numpy(pose60.astype(np.float32))
    tran60 = torch.from_numpy(tran60.astype(np.float32))
    if not torch.isfinite(pose60).all() or not torch.isfinite(tran60).all():
        raise ValueError("Retarget produced non-finite pose or translation")
    shape = shape.detach().cpu().float()
    body = ParametricModel(paths.smpl_file, device="cpu")
    with torch.no_grad():
        _, joints = body.forward_kinematics(pose60, shape, tran60)
    result = {
        "pose": pose60,
        "joint": joints.reshape(len(pose60), 24, 3),
        "tran": tran60,
        "shape": shape,
        "timestamps": physical["timestamps"].clone(),
        "fps": 60,
        "metadata": {
            "kind": "Unity HumanPose rotation retarget pseudo-labels",
            "independent_ground_truth": False,
            "retarget_mode": args.mode,
            "humanpose_export_sha256": expected_source,
            "source_motion_sha256": package.get("sourceMotionSha256"),
            "avatar_asset_path": package.get("avatarAssetPath"),
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "source_fps": package["fps"],
            "target_fps": 60,
            "resampling": "per-joint rotation SLERP; translation linear interpolation at physical input timestamps",
            "calibration": calibration,
            "shape": "SMPL shape calibrated from Unity rest-pose skeleton; not measured body shape",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()), "frames": len(pose60),
        "time": [float(target_times[0]), float(target_times[-1])],
        "joint_bounds_xyz": [joints.amin((0, 1)).tolist(), joints.amax((0, 1)).tolist()],
        "metadata": result["metadata"],
    }, indent=2))


if __name__ == "__main__":
    main()

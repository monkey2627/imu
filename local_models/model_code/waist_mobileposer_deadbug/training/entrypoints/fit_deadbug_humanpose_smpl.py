#!/usr/bin/env python3
"""Fit SMPL to Unity HumanPose joints with retargeted rotations as a prior."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--humanpose", type=Path, required=True)
    ap.add_argument("--retarget", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iterations", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.015)
    ap.add_argument("--position-weight", type=float, default=1.0)
    ap.add_argument("--rotation-prior-weight", type=float, default=0.02)
    ap.add_argument("--smooth-weight", type=float, default=0.01)
    ap.add_argument("--shape-bound", type=float, default=2.0)
    args = ap.parse_args()
    sys.path.insert(0, str(args.workspace.resolve()))
    from mobileposer.articulate import math
    from mobileposer.articulate.model import ParametricModel
    from mobileposer.config import paths

    package = json.loads(args.humanpose.read_text(encoding="utf-8"))
    bones = package["bones"]
    index = {name: i for i, name in enumerate(bones)}
    frames = package["frames"]
    source_t = np.asarray([f["time"] for f in frames], dtype=np.float64)
    source_p = np.asarray([[[v[k] for k in "xyz"] for v in f["positions"]] for f in frames], dtype=np.float32)
    source_p[..., 0] *= -1.0
    target = torch.load(args.retarget, map_location="cpu", weights_only=False)
    initial_pose_full = target["pose"].float()
    initial_tran_full = target["tran"].float()
    betas0 = target["shape"].float()
    target = torch.load(args.input, map_location="cpu", weights_only=False)
    target_t_full = target["timestamps"].double().numpy()
    if len(target_t_full) != len(initial_pose_full):
        raise ValueError("Initial retarget and physical input timeline differ")
    if source_t[-1] > target_t_full[-1] + 1 / 30:
        raise ValueError("Physical target timeline is too short for HumanPose observations")
    fit_t = np.clip(source_t, target_t_full[0], target_t_full[-1])
    full_pose_np = initial_pose_full.numpy()
    initial_pose_np = np.stack([
        Slerp(target_t_full, Rotation.from_matrix(full_pose_np[:, j]))(fit_t).as_matrix()
        for j in range(24)
    ], axis=1)
    initial_pose = torch.from_numpy(initial_pose_np.astype(np.float32))
    initial_tran_np = initial_tran_full.numpy()
    initial_tran = torch.from_numpy(np.stack([
        np.interp(fit_t, target_t_full, initial_tran_np[:, k]) for k in range(3)
    ], axis=-1).astype(np.float32))

    # SMPL index -> exported Unity Humanoid bone index.
    mapping = [(0,"pelvis",5.0),(1,"femur_l",2.0),(2,"femur_r",2.0),
               (3,"lumbar_body",1.5),(4,"tibia_l",3.0),(5,"tibia_r",3.0),
               (7,"talus_l",2.5),(8,"talus_r",2.5),(10,"toes_l",1.5),(11,"toes_r",1.5),
               (9,"thorax",1.5),(15,"head",1.2),(16,"humerus_l",2.0),(17,"humerus_r",2.0),
               (18,"ulna_l",2.0),(19,"ulna_r",2.0),(20,"hand_l",2.0),(21,"hand_r",2.0)]
    smpl_ids = torch.tensor([m[0] for m in mapping], dtype=torch.long)
    bone_ids = [index[m[1]] for m in mapping]
    weights = torch.tensor([m[2] for m in mapping], dtype=torch.float32)
    positions = torch.from_numpy(source_p[:, bone_ids])

    dev = torch.device(args.device)
    body = ParametricModel(paths.smpl_file, device=dev)
    initial_pose = initial_pose.to(dev)
    initial_tran = initial_tran.to(dev)
    target_pos = positions.to(dev)
    smpl_ids = smpl_ids.to(dev)
    weights = weights.to(dev)
    initial_r6d = math.rotation_matrix_to_r6d(initial_pose).reshape(len(source_t), 24, 6)
    pose_r6d = initial_r6d.detach().clone().requires_grad_(True)
    tran = initial_tran.detach().clone().requires_grad_(True)
    betas = betas0.to(dev).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([pose_r6d, tran, betas], lr=args.lr)
    initial_global = body.forward_kinematics_R(initial_pose)
    observed_global_ids = smpl_ids
    history = []

    for step in range(1, args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        pose = math.r6d_to_rotation_matrix(pose_r6d).reshape(-1, 24, 3, 3)
        global_rotation, joints = body.forward_kinematics(pose, betas, tran)
        residual = joints[:, smpl_ids] - target_pos
        abs_residual = residual.abs()
        huber = torch.where(abs_residual < 0.03, 0.5 * residual.square() / 0.03,
                            abs_residual - 0.015).sum(-1)
        pos_loss = (huber * weights).sum() / weights.sum()
        rot_now = global_rotation[:, observed_global_ids]
        rot_ref = initial_global[:, observed_global_ids]
        relative = rot_ref.transpose(-1, -2) @ rot_now
        rot_loss = (3.0 - relative.diagonal(dim1=-2, dim2=-1).sum(-1)).mean()
        prior_rel = initial_pose.transpose(-1, -2) @ pose
        pose_prior = (3.0 - prior_rel.diagonal(dim1=-2, dim2=-1).sum(-1)).mean()
        smooth_pose = (pose_r6d[1:] - pose_r6d[:-1]).square().mean()
        smooth_tran = (tran[1:] - tran[:-1]).square().mean()
        loss = (args.position_weight * pos_loss
                + args.rotation_prior_weight * (rot_loss + 0.25 * pose_prior)
                + args.smooth_weight * smooth_pose
                + 0.02 * smooth_tran
                )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([pose_r6d, tran, betas], 1.0)
        optimizer.step()
        with torch.no_grad():
            betas.clamp_(-args.shape_bound, args.shape_bound)
        if step == 1 or step % 50 == 0 or step == args.iterations:
            history.append({"step":step,"loss":float(loss.detach()),"position_huber":float(pos_loss.detach()),
                            "rotation_prior":float(rot_loss.detach()),"shape":betas.detach().cpu().tolist()})
            print(json.dumps(history[-1]), flush=True)

    with torch.no_grad():
        pose30 = math.r6d_to_rotation_matrix(pose_r6d).reshape(-1, 24, 3, 3)
        global30, joints30 = body.forward_kinematics(pose30, betas, tran)
        error = torch.linalg.vector_norm(joints30[:, smpl_ids] - target_pos, dim=-1)
        before_global, before_joint = body.forward_kinematics(initial_pose, betas0.to(dev), initial_tran)
        before_error = torch.linalg.vector_norm(before_joint[:, smpl_ids] - target_pos, dim=-1)
        final_shape = betas.detach().cpu()
        source_times = source_t
        target_times = target_t_full
        pose_np = pose30.cpu().numpy()
        pose60 = np.stack([Slerp(source_times, Rotation.from_matrix(pose_np[:, j]))(target_times).as_matrix()
                           for j in range(24)], axis=1)
        tran_np = tran.detach().cpu().numpy()
        tran60 = np.stack([np.interp(target_times, source_times, tran_np[:, a]) for a in range(3)], axis=-1)
        pose60 = torch.from_numpy(pose60.astype(np.float32))
        tran60 = torch.from_numpy(tran60.astype(np.float32))
        body_cpu = ParametricModel(paths.smpl_file, device="cpu")
        _, joint60 = body_cpu.forward_kinematics(pose60, final_shape, tran60)

    metadata = {
        "kind":"Unity HumanPose position-and-rotation constrained SMPL pseudo-labels",
        "independent_ground_truth":False,
        "humanpose_export_sha256":hashlib.sha256(args.humanpose.read_bytes()).hexdigest(),
        "source_motion_sha256":package.get("sourceMotionSha256"),
        "initial_retarget_sha256":hashlib.sha256(args.retarget.read_bytes()).hexdigest(),
        "input_sha256":hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "source_fps":package["fps"],"target_fps":60,"iterations":args.iterations,
        "weights":{"position":args.position_weight,"rotation_prior":args.rotation_prior_weight,"smooth":args.smooth_weight},
        "resampling":"per-joint rotation SLERP; translation linear interpolation at physical input timestamps",
        "metrics_before_m":{"mean":float(before_error.mean()),"p95":float(torch.quantile(before_error,.95)),"max":float(before_error.max())},
        "metrics_after_m":{"mean":float(error.mean()),"p95":float(torch.quantile(error,.95)),"max":float(error.max()),
                           "per_joint_mean":error.mean(0).cpu().tolist(),"smpl_joint_ids":smpl_ids.cpu().tolist()},
        "shape":final_shape.tolist(),"history":history,
    }
    out={"pose":pose60,"joint":joint60.reshape(len(pose60),24,3),"tran":tran60,"shape":final_shape,
         "timestamps":torch.from_numpy(target_times.copy()).float(),"fps":60,"metadata":metadata}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(out,args.output)
    print(json.dumps({"output":str(args.output.resolve()),"frames":len(pose60),"metrics_before_m":metadata["metrics_before_m"],
                      "metrics_after_m":metadata["metrics_after_m"],"shape":metadata["shape"]},indent=2))


if __name__ == "__main__": main()

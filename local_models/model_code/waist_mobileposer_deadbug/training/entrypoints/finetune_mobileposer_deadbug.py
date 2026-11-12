#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--crop", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from mobileposer.articulate import math
    from mobileposer.config import joint_set, model_config
    from mobileposer.utils.model_utils import load_model

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model_config.device = torch.device(args.device)
    model = load_model(str(args.model)).to(args.device)
    physical = torch.load(args.input, map_location="cpu", weights_only=False)
    target = torch.load(args.targets, map_location="cpu", weights_only=False)
    acc, ori = physical["acc"][0], physical["ori"][0]
    imu = torch.cat(((acc / 30).flatten(1), ori.flatten(1)), -1).to(args.device)
    target_pose = target["pose"].to(args.device)
    target_global, target_joint = model.bodymodel.forward_kinematics(target_pose)
    target_6d = math.rotation_matrix_to_r6d(target_global[:, joint_set.reduced]).reshape(len(imu), -1)
    target_joint = target_joint.reshape(len(imu), -1)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.joints, model.pose):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    def forward(segment):
        lengths = [len(segment)]
        joints = model.joints(segment[None], lengths)
        reduced = model.pose(torch.cat((joints, segment[None]), -1), lengths)
        pose = model._reduced_global_to_full(reduced)
        _, fk = model.bodymodel.forward_kinematics(pose)
        return joints[0], reduced[0], pose, fk

    def evaluate():
        model.eval()
        with torch.no_grad():
            joints, reduced, pose, fk = forward(imu)
            mpjpe = (fk - target_joint.view(-1, 24, 3)).norm(dim=-1).mean() * 100
            delta = pose[:, joint_set.reduced].transpose(-1, -2) @ target_pose[:, joint_set.reduced]
            angle = torch.acos(((delta.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)).mean()
            loss = (joints - target_joint).square().mean() + (reduced - target_6d).square().mean()
        return {"loss": float(loss), "mpjpe_cm": float(mpjpe), "local_rotation_deg": float(angle * 180 / np.pi)}

    before = evaluate()
    history = []
    model.eval()
    model.joints.joints.rnn.train()
    model.pose.pose.rnn.train()
    for step in range(1, args.steps + 1):
        start = int(torch.randint(0, max(1, len(imu) - args.crop + 1), (1,)))
        stop = min(start + args.crop, len(imu))
        joints, reduced, _, fk = forward(imu[start:stop])
        loss = (
            (joints - target_joint[start:stop]).square().mean()
            + (reduced - target_6d[start:stop]).square().mean()
            + (fk - target_joint[start:stop].view(-1, 24, 3)).square().mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if step % 50 == 0 or step == 1:
            row = {"step": step, "train_loss": float(loss)}
            history.append(row)
            print(json.dumps(row), flush=True)
    after = evaluate()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output / "model_finetuned.pth")
    result = {
        "kind": "same-sequence deadbug overfit",
        "before": before,
        "after": after,
        "steps": args.steps,
        "crop": args.crop,
        "learning_rate": args.lr,
        "trainable_modules": ["joints", "poser"],
        "frozen_modules": ["foot_contact", "velocity"],
        "train_and_evaluate_same_sequence": True,
        "pretrained": str(args.model.resolve()),
        "history": history,
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

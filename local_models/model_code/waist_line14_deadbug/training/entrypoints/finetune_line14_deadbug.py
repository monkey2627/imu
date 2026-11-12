#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--crop", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from mobileposer.articulate import math as M
    from mobileposer.articulate.model import ParametricModel
    from mobileposer.config import joint_set, paths
    from mobileposer.line9.models import GeometryStats, Line9Joints
    import line14_poser_chain as C

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    physical = torch.load(args.input, map_location="cpu", weights_only=False)
    target = torch.load(args.targets, map_location="cpu", weights_only=False)
    acc, ori = physical["acc"][0].float(), physical["ori"][0].float()
    imu = torch.cat(((acc / 30).flatten(1), ori.flatten(1)), -1).to(dev)
    local_pose = target["pose"].to(dev)
    shape = target["shape"].to(dev)
    body = ParametricModel(paths.smpl_file, device=dev)
    _, target_joints = body.forward_kinematics(local_pose, shape, tran=None)
    local6 = M.rotation_matrix_to_r6d(local_pose).reshape(1, len(imu), 24, 6)
    target_rot = C.global_targets(local6, body.parent)[0]
    target_joints = target_joints.reshape(len(imu), 72)

    jnet = Line9Joints("b0", geom_dim=1, geometry_stats=GeometryStats(torch.zeros(1), torch.ones(1))).to(dev)
    poser = C.ChainPoser(132).to(dev)
    state = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    jnet.load_state_dict(state["jnet"])
    poser.load_state_dict(state["poser"])
    optimizer = torch.optim.AdamW(list(jnet.parameters()) + list(poser.parameters()), lr=args.lr, weight_decay=0.01)

    def forward(segment):
        length = [len(segment)]
        joint = jnet(segment[None], length)
        rotation = poser(torch.cat((joint, segment[None]), -1), length)
        return joint[0], rotation[0]

    def evaluate():
        jnet.eval(); poser.eval()
        with torch.no_grad():
            joint, rotation = forward(imu)
            mpjpe = (joint.view(-1, 24, 3) - target_joints.view(-1, 24, 3)).norm(dim=-1).mean() * 100
            pred_r = M.r6d_to_rotation_matrix(rotation.reshape(-1, 6))
            true_r = M.r6d_to_rotation_matrix(target_rot.reshape(-1, 6))
            delta = pred_r.transpose(-1, -2) @ true_r
            angle = torch.acos(((delta.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)).mean()
            loss = (joint - target_joints).square().mean() + (rotation - target_rot).square().mean()
        return {"loss": float(loss), "mpjpe_cm": float(mpjpe), "global_rotation_deg": float(angle * 180 / np.pi)}

    before = evaluate()
    history = []
    for step in range(1, args.steps + 1):
        jnet.train(); poser.train()
        start = int(torch.randint(0, max(1, len(imu) - args.crop + 1), (1,)))
        stop = min(start + args.crop, len(imu))
        joint, rotation = forward(imu[start:stop])
        loss = (
            (joint - target_joints[start:stop]).square().mean()
            + (rotation - target_rot[start:stop]).square().mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(jnet.parameters()) + list(poser.parameters()), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == 1:
            row = {"step": step, "train_loss": float(loss)}
            history.append(row)
            print(json.dumps(row), flush=True)
    after = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "jnet": jnet.state_dict(),
        "poser": poser.state_dict(),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "personalization": "same-sequence deadbug overfit",
    }, args.output)
    result = {
        "kind": "same-sequence deadbug overfit",
        "before": before,
        "after": after,
        "steps": args.steps,
        "crop": args.crop,
        "learning_rate": args.lr,
        "train_and_evaluate_same_sequence": True,
        "input": "five physical IMUs: lw,rw,ls,rs,waist",
        "supervision": "same-capture FBX-to-SMPL pseudo-labels",
        "pretrained": str(args.checkpoint.resolve()),
        "history": history,
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

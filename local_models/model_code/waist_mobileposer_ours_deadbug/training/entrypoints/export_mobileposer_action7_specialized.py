#!/usr/bin/env python3
"""Run the specialized MobilePoser checkpoint and export action 7 poses."""
import argparse
import json
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--specialized', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.workspace))
    from mobileposer.config import model_config
    from mobileposer.utils.model_utils import load_model

    torch.set_num_threads(4)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_config.device = device
    model = load_model(str(args.baseline)).to(device)
    model.load_state_dict(torch.load(args.specialized, map_location=device, weights_only=False))
    model.eval()
    capture = args.data / 'captures' / '7'
    inp = torch.load(capture / 'inputs/wrists_shanks_waist/ours1.pt', map_location='cpu', weights_only=False)
    target = torch.load(capture / 'targets_humanpose_calibrated_60.pt', map_location='cpu', weights_only=False)
    if inp['fps'] != 60 or target['fps'] != 60 or not torch.equal(inp['timestamps'], target['timestamps']):
        raise ValueError('Action 7 input/target timestamps or frequency do not match')
    imu = torch.cat(((inp['acc'][0].float() / 30).flatten(1), inp['ori'][0].float().flatten(1)), -1)[None].to(device)
    gt_pose = target['pose'].float().to(device)
    with torch.no_grad():
        joints = model.joints(imu, [imu.shape[1]])
        reduced = model.pose(torch.cat((joints, imu), dim=-1), [imu.shape[1]])
        pred_pose = model._reduced_global_to_full(reduced)
        _, pred_joints = model.bodymodel.forward_kinematics(pred_pose)
        _, gt_joints = model.bodymodel.forward_kinematics(gt_pose)
    pred_joints = pred_joints.reshape(-1, 24, 3)
    gt_joints = gt_joints.reshape(-1, 24, 3)
    mpjpe = ((pred_joints - pred_joints[:, :1] - (gt_joints - gt_joints[:, :1])).norm(dim=-1).mean() * 100)
    delta = pred_pose.transpose(-1, -2) @ gt_pose
    angle = torch.acos(((delta.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)).mean() * (180.0 / 3.141592653589793)
    result = {'pose': pred_pose.detach().cpu(), 'target_pose': gt_pose.detach().cpu(),
              'shape': target['shape'].detach().cpu(), 'timestamps': target['timestamps'].detach().cpu(),
              'metadata': target.get('metadata', {}), 'mpjpe_cm': float(mpjpe),
              'local_rotation_error_deg': float(angle)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(json.dumps({'frames': len(gt_pose), 'mpjpe_cm': float(mpjpe),
                      'local_rotation_error_deg': float(angle), 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()

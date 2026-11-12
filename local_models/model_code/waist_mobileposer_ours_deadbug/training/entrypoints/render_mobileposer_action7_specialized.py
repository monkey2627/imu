#!/usr/bin/env python3
"""Render action 7 HumanPose SMPL pseudo-labels against the specialized model."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def renderer_setup(pyrender, width, height):
    renderer = pyrender.OffscreenRenderer(width, height)
    scene = pyrender.Scene(bg_color=(0.965, 0.969, 0.957, 1.0), ambient_light=(0.48, 0.48, 0.48))
    camera = pyrender.OrthographicCamera(xmag=0.92, ymag=1.72)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = (0.0, 0.82, 4.0)
    scene.add(camera, pose=camera_pose)
    for position, intensity in (((0.0, 2.5, 2.5), 2.3), ((-2.0, 1.0, 1.0), 1.1)):
        pose = np.eye(4)
        pose[:3, 3] = position
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=intensity), pose=pose)
    return renderer, scene


def draw_mesh(renderer, scene, pyrender, trimesh, vertices, faces, color):
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(*[c / 255.0 for c in color], 1.0), metallicFactor=0.0,
        roughnessFactor=0.72, doubleSided=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    node = scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True))
    image, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    scene.remove_node(node)
    return cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--specialized', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--fps', type=int, default=15)
    args = ap.parse_args()
    sys.path.insert(0, str(args.workspace))
    import pyrender
    import trimesh
    from mobileposer.config import model_config
    from mobileposer.utils.model_utils import load_model

    torch.set_num_threads(4)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_config.device = device
    model = load_model(str(args.baseline)).to(device)
    state = torch.load(args.specialized, map_location=device, weights_only=False)
    model.load_state_dict(state)
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
        pred_pose = model._reduced_global_to_full(reduced)[0]
        _, pred_joints = model.bodymodel.forward_kinematics(pred_pose)
        _, gt_joints = model.bodymodel.forward_kinematics(gt_pose)
    pred_joints = pred_joints.reshape(-1, 24, 3)
    gt_joints = gt_joints.reshape(-1, 24, 3)
    pred_rel = pred_joints - pred_joints[:, :1]
    gt_rel = gt_joints - gt_joints[:, :1]
    mpjpe = (pred_rel - gt_rel).norm(dim=-1).mean() * 100
    rotation_delta = pred_pose.transpose(-1, -2) @ gt_pose
    rotation_angle = torch.acos(((rotation_delta.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
    rotation_deg = rotation_angle.mean() * (180.0 / np.pi)

    timestamps = target['timestamps'].double().cpu().numpy()
    stride = max(1, round(60 / args.fps))
    indices = list(range(0, len(timestamps), stride))
    shape = target['shape'].float().reshape(1, -1).to(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    width, height, panel_w = 1400, 820, 700
    renderer, scene = renderer_setup(pyrender, panel_w, 720)
    tmp = args.output.with_name(args.output.stem + '.tmp.mp4')
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'Cannot open video writer: {tmp}')
    preview = None
    try:
        for k, frame in enumerate(indices):
            with torch.no_grad():
                _, _, gt_vertices = model.bodymodel.forward_kinematics(
                    gt_pose[frame:frame + 1], shape=shape, tran=None, calc_mesh=True)
                _, _, pred_vertices = model.bodymodel.forward_kinematics(
                    pred_pose[frame:frame + 1], shape=shape, tran=None, calc_mesh=True)
            left = draw_mesh(renderer, scene, pyrender, trimesh, gt_vertices[0].cpu().numpy(), model.bodymodel.face, (58, 131, 91))
            right = draw_mesh(renderer, scene, pyrender, trimesh, pred_vertices[0].cpu().numpy(), model.bodymodel.face, (194, 105, 53))
            canvas = np.full((height, width, 3), (246, 247, 244), dtype=np.uint8)
            canvas[:720, :panel_w] = left
            canvas[:720, panel_w:] = right
            cv2.putText(canvas, 'HumanPose SMPL pseudo-label', (28, 38), cv2.FONT_HERSHEY_SIMPLEX, .8, (35, 40, 45), 2, cv2.LINE_AA)
            cv2.putText(canvas, 'Action 7 specialized prediction', (panel_w + 28, 38), cv2.FONT_HERSHEY_SIMPLEX, .8, (35, 40, 45), 2, cv2.LINE_AA)
            cv2.line(canvas, (panel_w, 0), (panel_w, 720), (210, 212, 208), 1)
            cv2.putText(canvas, f'Dead bug | frame {frame}/{len(timestamps)-1} | t={timestamps[frame]:.2f}s',
                        (28, 758), cv2.FONT_HERSHEY_SIMPLEX, .65, (50, 54, 58), 1, cv2.LINE_AA)
            cv2.putText(canvas, f'Full-sequence MPJPE {float(mpjpe):.2f} cm | local rotation error {float(rotation_deg):.2f} deg',
                        (28, 795), cv2.FONT_HERSHEY_SIMPLEX, .62, (50, 54, 58), 1, cv2.LINE_AA)
            writer.write(canvas)
            if k == len(indices) // 3:
                preview = canvas.copy()
    finally:
        writer.release()
        renderer.delete()
    tmp.replace(args.output)
    preview_path = args.output.with_name(args.output.stem + '_preview.png')
    cv2.imwrite(str(preview_path), preview)
    report = {
        'action': 7, 'label': 'dead bug', 'frames': len(timestamps), 'rendered_frames': len(indices),
        'fps': args.fps, 'duration_seconds': float(timestamps[-1] - timestamps[0]),
        'checkpoint': str(args.specialized), 'reference': target.get('metadata', {}),
        'mpjpe_cm': float(mpjpe), 'local_rotation_error_deg': float(rotation_deg),
        'video': str(args.output), 'preview': str(preview_path),
        'metric_note': 'Root-relative SMPL joint MPJPE and mean local rotation geodesic error against HumanPose-retarget pseudo-labels; not independent mocap ground truth.'}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
